from __future__ import annotations

from contextlib import nullcontext
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from einx import mean, rearrange, softmax
from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image
from torchvision import utils

from gen.models.cross_attn.break_a_scene import aggregate_attention, save_cross_attention_vis
from gen.utils.decoupled_utils import load_tensor_dict
from gen.utils.logging_utils import log_info
from gen.utils.rotation_utils import compute_rotation_matrix_from_ortho6d, visualize_rotations
from gen.utils.tokenization_utils import get_tokens
from gen.utils.trainer_utils import TrainingState

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData
    from gen.models.cross_attn.base_model import AttentionMetadata

def repeat_batch(batch, bs: int):
    batch_ = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            assert v.shape[0] == 1
            batch_[k] = repeat(v[0], "... -> h ...", h=bs)

    return batch_

new_prompts = [
            "A photo of a {}",
            "A photo of {} in the jungle",
            "A photo of {} on a beach",
            "A photo of {} in Times Square",
            "A photo of {} in the moon",
            "A painting of {} in the style of Monet",
            "Oil painting of {}",
            "A Marc Chagall painting of {}",
            "A manga drawing of {}",
            "A watercolor painting of {}",
            "A statue of {}",
            "App icon of {}",
            "A sand sculpture of {}",
            "Colorful graffiti of {}",
            "A photograph of two {} on a table",
        ]

def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    cond: Optional[ConditioningData] = None,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if "input_ids" in batch:
        orig_input_ids = batch["input_ids"].clone()
        batch["input_ids"] = orig_input_ids.clone()

    if cond is None or len(cond.unet_kwargs) == 0:
        with torch.cuda.amp.autocast():
            assert self.cfg.inference.empty_string_cfg
            cond = self.get_standard_conditioning_for_inference(batch=batch, cond=cond)

    if "guidance_scale" not in cond.unet_kwargs and "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = self.cfg.inference.guidance_scale

    if "num_images_per_prompt" not in cond.unet_kwargs:
        kwargs["num_images_per_prompt"] = num_images_per_prompt

    if "return_attn_probs" in kwargs:
        cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["return_attn_probs"] = kwargs["return_attn_probs"]
        del kwargs["return_attn_probs"]

    # CFG must be enabled when masking as we make this assumption in attn_proc
    if self.cfg.model.attention_masking:
        assert kwargs["guidance_scale"] > 1

    kwargs["height"] = self.cfg.model.resolution
    kwargs["width"] = self.cfg.model.resolution

    desired_context = torch.cuda.amp.autocast() if self.cfg.model.freeze_unet is False or self.cfg.model.unfreeze_gated_cross_attn else nullcontext()
    with desired_context:
        images = self.pipeline(**cond.unet_kwargs, **kwargs).images

    if "formatted_input_ids" in batch:
        del batch["formatted_input_ids"]

    if "cross_attention_kwargs" in cond.unet_kwargs and "attn_meta" in cond.unet_kwargs["cross_attention_kwargs"] and "return_attn_probs" in cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]:
        del cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["return_attn_probs"]

    return images, cond


@torch.no_grad()
def run_quantitative_inference(self: BaseMapper, batch: dict, state: TrainingState):
    ret = {}
    if self.cfg.model.token_rot_pred_loss:
        with torch.cuda.amp.autocast():
            cond = self.get_standard_conditioning_for_inference(batch=batch)
            pred_data = self.denoise_rotation(batch=batch, cond=cond, scheduler=self.scheduler)
            pred_data.pred_6d_rot = pred_data.pred_6d_rot[pred_data.pred_mask]
            pred_data.gt_rot_6d = pred_data.gt_rot_6d[pred_data.pred_mask]
            pred_loss = F.mse_loss(pred_data.pred_6d_rot, pred_data.gt_rot_6d, reduction='none')
            ret['pred_loss'] = pred_loss.float().cpu()
    return ret


@torch.no_grad()
def run_qualitative_inference(self: BaseMapper, batch: dict, state: TrainingState):
    """
    TODO: Support batched inference at some point. Right now it is batched under the hood (for CFG and if num_images_per_prompt > 1) but we can likely do much better.
    """

    assert batch["input_ids"].shape[0] == 1

    if self.cfg.model.unet is False: return {}

    ret = {}

    orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
    gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(text="GT")
    ret["validation"] = gt_info

    added_kwargs = dict()
    if self.cfg.inference.visualize_attention_map:
        added_kwargs["return_attn_probs"] = True
        cond = ConditioningData()
        cond.unet_kwargs["cross_attention_kwargs"] = dict(attn_meta=AttentionMetadata())
        cond.attn_dict['return_attn_probs'] = True
        cond.attn_dict['attn_probs'] = []
        added_kwargs['cond'] = cond

    prompt_image, cond = self.infer_batch(
        batch=batch,
        num_images_per_prompt=self.cfg.inference.num_images_per_prompt,
        **added_kwargs
    )

    if self.cfg.inference.visualize_attention_map:
        attn = cond.unet_kwargs['cross_attention_kwargs']['attn_meta']['attn_probs']
        target_size = (self.cfg.model.latent_dim, self.cfg.model.latent_dim)
        all_attn = []
        bs = len(prompt_image)
        for k,v in attn.items():
            layer_attn = torch.stack(v, dim=0).cpu()[:, -bs:]
            layer_attn = rearrange("timesteps b (h w) tokens -> timesteps (b tokens) () h w", layer_attn, h=int(sqrt(layer_attn.shape[-2])))
            layer_attn = torch.mean(layer_attn, dim=0)
            layer_attn = F.interpolate(layer_attn.to(dtype=torch.float32), size=target_size, mode="bilinear", align_corners=False).squeeze(1)
            all_attn.append(layer_attn)

        all_attn = mean("[layers] (b tokens) h w -> b tokens h w", torch.stack(all_attn), b=bs)
        attn_imgs = []
        for b in range(bs):
            attn_ = all_attn[b]
            idx_mask = cond.learnable_idxs[1][cond.learnable_idxs[0] == b]
            tokens = attn_[idx_mask.to(device=attn_.device)]
            if tokens.shape[0] == 0:
                continue
            tokens = rearrange("t (h w) -> t 3 h w", softmax("t [hw]", rearrange("t h w -> t (h w)", tokens)), h=tokens.shape[-1])
            tokens = (tokens - torch.min(tokens)) / (torch.max(tokens) - torch.min(tokens))
            masks = Im(rearrange("h w masks -> masks 3 h w", batch['gen_segmentation'][b, ..., cond.mask_instance_idx]).float()).resize(32, 32)
            attn_imgs.append(Im.concat_vertical(Im.concat_horizontal(*tokens), Im.concat_horizontal(*masks.torch)))

        ret["attn_vis"] = Im.concat_vertical(*attn_imgs)

    if self.cfg.model.token_rot_pred_loss:
        with torch.cuda.amp.autocast():
            pred_data = self.denoise_rotation(batch=batch, cond=cond, scheduler=self.pipeline.scheduler)
            R_ref = compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d).cpu().numpy()
            R_pred = compute_rotation_matrix_from_ortho6d(pred_data.pred_6d_rot).cpu().numpy()
            assert R_ref.shape == R_pred.shape

            rot_imgs = []
            for i in range(R_ref.shape[0]):
                rot_imgs.append(Im(visualize_rotations(R_ref[i], R_pred[i])).torch)
            rot_imgs = torch.stack(rot_imgs)

            bs = batch['gen_pixel_values'].shape[0]
            all_imgs = []
            for i in range(bs):
                mask_ = ((cond.mask_batch_idx == i) & pred_data.pred_mask).to(rot_imgs.device)
                all_imgs.append(Im(rot_imgs[mask_]).grid())

            ret["rotations"] = Im.concat_vertical(*all_imgs, spacing=20, fill=(128, 128, 128))

    full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
    generated_images = Im.concat_horizontal(
        Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}")
        for i, prompt_image_ in enumerate(prompt_image)
    )
    ret["validation"] = Im.concat_horizontal(ret["validation"], generated_images)

    if self.cfg.inference.save_prompt_embeds:
        assert cond.mask_instance_idx.shape[0] == cond.mask_tokens.shape[0]

        orig_gen_segmentation = batch["gen_segmentation"].clone()
        all_masks = []
        for j in cond.mask_instance_idx:
            mask_image = Im(get_layered_image_from_binary_mask(orig_gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
            mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
            mask_alpha = (orig_gen_segmentation[..., [j]].squeeze() * (255 / 1.5)).cpu().numpy().astype(np.uint8)
            composited_image = orig_image.pil.copy().convert("RGBA")
            composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
            composited_image = composited_image.convert("RGB")
            all_masks.append(Im(composited_image).write_text(f"token_{j}").resize(256, 256).np.squeeze(0))

        ret["cond"] = {"mask_tokens": cond.mask_tokens, "mask_rgb": np.stack(all_masks), "orig_image": orig_image.np}

    if self.cfg.inference.vary_cfg_plot:
        scale_images = []
        for scale in [0.0, 3.0, 5.0, 7.5, 10.0]:
            if self.cfg.model.attention_masking and scale <= 1.0:
                continue
            prompt_images, cond = self.infer_batch(
                batch=batch, cond=cond, num_images_per_prompt=self.cfg.inference.num_images_per_prompt, guidance_scale=scale
            )
            scale_images.append(Im.concat_vertical(prompt_images).write_text(f"CFG Scale: {scale:.1f}"))

        ret["cfg_scale"] = Im.concat_horizontal(orig_image.to("cpu").copy.write_text("GT"), *scale_images)

    if self.cfg.model.break_a_scene_cross_attn_loss:
        batch_size = (2 if self.cfg.inference.guidance_scale > 1.0 else 1) * self.cfg.inference.num_images_per_prompt

        agg_attn_cond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size
        )
        attn_img_cond = Im(
            save_cross_attention_vis(tokenizer=self.tokenizer, tokens=batch["formatted_input_ids"][0], attention_maps=agg_attn_cond)
        ).write_text("Cond")

        agg_attn_uncond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=0, batch_size=batch_size
        )
        uncond_tokens = self.pipeline.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        attn_img_uncond = Im(
            save_cross_attention_vis(tokenizer=self.tokenizer, tokens=uncond_tokens["formatted_input_ids"][0], attention_maps=agg_attn_uncond)
        ).write_text("Uncond")

        ret["attn_vis"] = Im.concat_vertical(attn_img_cond, attn_img_uncond)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    if self.cfg.inference.num_masks_to_remove is not None:
        orig_gen_segmentation = batch["gen_segmentation"].clone()

        batched_gen_seg = []
        idxs = list(range(orig_gen_segmentation.shape[-1])[: self.cfg.inference.num_masks_to_remove])
        for j in idxs:
            batched_gen_seg.append(orig_gen_segmentation[..., torch.arange(orig_gen_segmentation.size(-1)) != j])

        batched_gen_seg = torch.cat(batched_gen_seg, dim=0)

        if batched_gen_seg.sum().item() != 0:
            batch_ = repeat_batch(batch, bs=len(idxs))
            batch_["gen_segmentation"] = batched_gen_seg

            prompt_images, _ = self.infer_batch(batch=batch_)
            if self.cfg.model.break_a_scene_cross_attn_loss:
                self.controller.reset()

            removed_mask_imgs = []
            for j in idxs:
                mask_image = Im(get_layered_image_from_binary_mask(orig_gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
                mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
                mask_alpha = (orig_gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
                composited_image = orig_image.pil.copy().convert("RGBA")
                composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
                composited_image = composited_image.convert("RGB")
                removed_mask_imgs.append(Im.concat_vertical(prompt_images[j], mask_image, composited_image, spacing=5, fill=(128, 128, 128)))

            ret["validation"] = Im.concat_horizontal(ret["validation"], Im.concat_horizontal(*removed_mask_imgs), spacing=15)
            batch["gen_segmentation"] = orig_gen_segmentation

    if self.cfg.inference.infer_new_prompts:
        prompts = [prompt.format(self.cfg.model.placeholder_token) for prompt in new_prompts]

        input_ids_ = []
        for prompt in prompts[: self.cfg.inference.max_batch_size]:
            input_ids_.append(get_tokens(tokenizer=self.tokenizer, prompt=prompt)[None])
        input_ids_ = torch.cat(input_ids_, dim=0).to(self.device)

        batch_ = repeat_batch(batch, bs=len(input_ids_))
        batch_["input_ids"] = input_ids_

        prompt_images, _ = self.infer_batch(batch=batch_)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

        ret["prompt_images"] = Im.concat_horizontal(
            Im(prompt_image).write_text(prompt, size=3.2) for prompt_image, prompt in zip(prompt_images, prompts)
        )
        ret["prompt_images"] = Im.concat_horizontal(orig_image, ret["prompt_images"], spacing=20)

    return ret


def take_from(slices: tuple[int, slice], data: tuple[dict]):
    output_mask_tokens = []
    output_mask_rgb = []
    for idx, sl in slices:
        output_mask_tokens.append(data[idx]["mask_tokens"][sl])
        output_mask_rgb.append(data[idx]["mask_rgb"][sl])

    output_mask_tokens = torch.cat(output_mask_tokens, dim=0)
    output_mask_rgb = np.concatenate(output_mask_rgb, axis=0)

    return output_mask_tokens, output_mask_rgb


@torch.no_grad()
def compose_two_images(self: BaseMapper, batch: dict, state: TrainingState, embed_path_1: Path, embed_path_2: Path):
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData

    image_0_dict = load_tensor_dict(embed_path_1)
    image_1_dict = load_tensor_dict(embed_path_2)

    image_0_tokens = image_0_dict["mask_tokens"]
    image_1_tokens = image_1_dict["mask_tokens"]

    prompt_image_0, cond_0 = self.infer_batch(
        batch=batch,
        cond=ConditioningData(mask_tokens=image_0_tokens, mask_batch_idx=torch.zeros((image_0_tokens.shape[0],), dtype=torch.int64)),
    )
    prompt_image_1, cond_1 = self.infer_batch(
        batch=batch,
        cond=ConditioningData(mask_tokens=image_1_tokens, mask_batch_idx=torch.zeros((image_1_tokens.shape[0],), dtype=torch.int64)),
    )

    final_tokens, final_rgb = take_from(
        slices=(
            (1, slice(0, 1)),
            (0, slice(1, 2)),
            (1, slice(1, 2)),
        ),
        data=(image_0_dict, image_1_dict),
    )

    mask_batch_idx = torch.zeros((final_tokens.shape[0],), dtype=torch.int64)
    prompt_image, cond = self.infer_batch(
        batch=batch, cond=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=4
    )

    Im.concat_vertical(
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Gen {i}") for i, prompt_image_ in enumerate(prompt_image)
        ),
        Im(utils.make_grid(Im(final_rgb).torch)).write_text("Combined Conditioned masks [GT]"),
        Im.concat_horizontal(
            Im(image_0_dict["orig_image"]).write_text("First Image GT", size=0.25),
            Im(prompt_image_0[0]).write_text("First Image Autoencoded", size=0.25),
        ),
        Im.concat_horizontal(
            Im(image_1_dict["orig_image"]).write_text("Second Image GT", size=0.25),
            Im(prompt_image_1[0]).write_text("Second Image Autoencoded", size=0.25),
        ),
    ).save("test_00.png")

    breakpoint()


def lerp(a, b, ts):
    return a + (b - a) * ts

@torch.no_grad()
def interpolate_latents(self: BaseMapper, batch: dict, state: TrainingState, embed_path_1: Path, embed_path_2: Path, remove_background_token: bool = False, batch_interp: bool = True, steps: int = 10):
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData

    image_0_dict = load_tensor_dict(embed_path_1)
    image_1_dict = load_tensor_dict(embed_path_2)

    image_0_tokens = image_0_dict["mask_tokens"]
    image_1_tokens = image_1_dict["mask_tokens"]

    # prompt_image_0, cond_0 = self.infer_batch(
    #     batch=batch,
    #     cond=ConditioningData(mask_tokens=image_0_tokens, mask_batch_idx=torch.zeros((image_0_tokens.shape[0],), dtype=torch.int64)),
    # )
    # prompt_image_1, cond_1 = self.infer_batch(
    #     batch=batch,
    #     cond=ConditioningData(mask_tokens=image_1_tokens, mask_batch_idx=torch.zeros((image_1_tokens.shape[0],), dtype=torch.int64)),
    # )

    interp_values = torch.linspace(0, 1, steps=steps)

    prompt_images = []
    if batch_interp:
        final_tokens = []
        mask_batch_idx = []
        for i, interp_value in enumerate(interp_values):
            interp_token = lerp(image_0_tokens, image_1_tokens, interp_value)
            final_tokens.append(interp_token)
            mask_batch_idx.append(i * torch.ones((interp_token.shape[0],), dtype=torch.int64))

        final_tokens = torch.cat(final_tokens, dim=0)
        mask_batch_idx = torch.cat(mask_batch_idx, dim=0)
        if "formatted_input_ids" in batch:
            del batch["formatted_input_ids"]

        batch_ = {}
        batch_["input_ids"] = batch["input_ids"].repeat(steps, 1)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and k not in batch_:
                assert v.shape[0] == 1
                batch_[k] = repeat(v[0], "... -> h ...", h=batch_["input_ids"].shape[0])

        prompt_images, cond = self.infer_batch(
            batch=batch_, cond=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=1
        )
    else:
        for i, interp_value in enumerate(interp_values):
            interp_token = lerp(image_0_tokens, image_1_tokens, interp_value)
            if remove_background_token:
                interp_token = interp_token[1:]
            prompt_image, cond = self.infer_batch(
                batch=batch, cond=ConditioningData(mask_tokens=interp_token, mask_batch_idx=torch.zeros((interp_token.shape[0],), dtype=torch.int64)), num_images_per_prompt=2
            )
            prompt_image = Im.concat_vertical(prompt_image)
            prompt_images.append(prompt_image)

    Im.concat_vertical(
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Weight {interp_values[i].item():.2f}") for i, prompt_image_ in enumerate(prompt_images)
        ),
        Im.concat_horizontal(
            Im(image_0_dict["orig_image"]).write_text("Left (0) GT", size=0.75),
            Im(image_1_dict["orig_image"]).write_text("Right (1) GT", size=0.75),
        ),
        spacing=50,
    ).save("test_01.png")

    breakpoint()
