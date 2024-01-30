from __future__ import annotations

from calendar import c
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from einops import repeat
from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image

from gen.models.cross_attn.break_a_scene import aggregate_attention, save_cross_attention_vis
from gen.utils.decoupled_utils import load_tensor_dict
from gen.utils.tokenization_utils import get_tokens
from gen.utils.trainer_utils import TrainingState

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData


def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    conditioning_data: Optional[ConditioningData] = None,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if "input_ids" in batch:
        orig_input_ids = batch["input_ids"].clone()
        batch["input_ids"] = orig_input_ids.clone()

    if conditioning_data is None or len(conditioning_data.unet_kwargs) == 0:
        with torch.cuda.amp.autocast():
            assert self.cfg.inference.empty_string_cfg
            conditioning_data = self.get_standard_conditioning_for_inference(batch=batch, conditioning_data=conditioning_data)

    if "guidance_scale" not in conditioning_data.unet_kwargs and "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = self.cfg.inference.guidance_scale

    if "num_images_per_prompt" not in conditioning_data.unet_kwargs:
        kwargs['num_images_per_prompt'] = num_images_per_prompt

    desired_context = nullcontext() if self.cfg.model.freeze_unet else torch.cuda.amp.autocast()
    with desired_context:
        images = self.pipeline(
            **conditioning_data.unet_kwargs,
            **kwargs
        ).images

    if "formatted_input_ids" in batch:
        del batch["formatted_input_ids"]

    return images, conditioning_data


@torch.no_grad()
def run_inference(self: BaseMapper, batch: dict, state: TrainingState):
    """
    TODO: Support batched inference at some point. Right now it is batched under the hood (for CFG and if num_images_per_prompt > 1) but we can likely do much better.
    """

    assert batch["input_ids"].shape[0] == 1

    ret = {}

    orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
    gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(
        text="GT", relative_font_scale=0.004
    )
    ret["validation"] = gt_info

    prompt_image, conditioning_data = self.infer_batch(
        batch=batch,
        num_images_per_prompt=self.cfg.inference.num_images_per_prompt,
    )

    full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
    generated_images = Im.concat_horizontal(
        Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}", relative_font_scale=0.004)
        for i, prompt_image_ in enumerate(prompt_image)
    )
    ret["validation"] = Im.concat_horizontal(ret["validation"], generated_images)

    if self.cfg.inference.vary_cfg_plot:
        scale_images = []
        for scale in [1.0, 3.0, 5.0, 7.5, 10.0]:
            prompt_images, conditioning_data = self.infer_batch(batch=batch, conditioning_data=conditioning_data, num_images_per_prompt=self.cfg.inference.num_images_per_prompt, guidance_scale=0)
            scale_images.append(Im.concat_vertical(prompt_images).write_text(f"CFG Scale: {scale:.1f}"))

        ret["cfg_scale"] = Im.concat_horizontal(scale_images)

    if self.cfg.inference.save_prompt_embeds:
        assert conditioning_data.mask_instance_idx.shape[0] == conditioning_data.mask_tokens.shape[0]

        orig_gen_segmentation = batch["gen_segmentation"].clone()
        all_masks = []
        for j in conditioning_data.mask_instance_idx:
            mask_image = Im(get_layered_image_from_binary_mask(orig_gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
            mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
            mask_alpha = (orig_gen_segmentation[..., [j]].squeeze() * (255 / 1.5)).cpu().numpy().astype(np.uint8)
            composited_image = orig_image.pil.copy().convert("RGBA")
            composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
            composited_image = composited_image.convert("RGB")
            all_masks.append(Im(composited_image).write_text(f"token_{j}").resize(256, 256).np.squeeze(0))

        ret["individual_masks"] = Im.concat_horizontal(*all_masks, spacing=10)
        ret["conditioning_data"] = {"mask_tokens": conditioning_data.mask_tokens, "mask_rgb": np.stack(all_masks), "orig_image": orig_image.np}

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

        batch_ = {}
        batch_["gen_segmentation"] = []
        idxs = list(range(orig_gen_segmentation.shape[-1])[: self.cfg.inference.num_masks_to_remove])
        for j in idxs:
            mask_ = orig_gen_segmentation[..., torch.arange(orig_gen_segmentation.size(-1)) != j]
            batch_["gen_segmentation"].append(mask_)

        batch_["gen_segmentation"] = torch.cat(batch_["gen_segmentation"], dim=0)
        batch_["input_ids"] = batch["input_ids"].repeat(batch_["gen_segmentation"].shape[0], 1)

        if batch_["gen_segmentation"].sum().item() != 0:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k not in batch_:
                    assert v.shape[0] == 1
                    batch_[k] = repeat(v[0], "... -> h ...", h=batch_["gen_segmentation"].shape[0])

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
        prompts = [
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
        prompts = [prompt.format(self.cfg.model.placeholder_token) for prompt in prompts]
        batch_ = {}

        batch_["input_ids"] = []
        for prompt in prompts[: self.cfg.inference.max_batch_size]:
            batch_["input_ids"].append(get_tokens(tokenizer=self.tokenizer, prompt=prompt)[None])
        batch_["input_ids"] = torch.cat(batch_["input_ids"], dim=0).to(self.device)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and k not in batch_:
                assert v.shape[0] == 1
                batch_[k] = repeat(v[0], "... -> h ...", h=batch_["input_ids"].shape[0])

        prompt_images, _ = self.infer_batch(batch=batch_)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

        ret["prompt_images"] = Im.concat_horizontal(
            Im(prompt_image).write_text(prompt, relative_font_scale=0.00125) for prompt_image, prompt in zip(prompt_images, prompts)
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
def run_custom_inference(self: BaseMapper, batch: dict, state: TrainingState, embed_path: Path):
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData

    image_0_dict = load_tensor_dict("/home/aswerdlow/research/gen/outputs/debug/debug_debug_2024-01-28_14_53_30/conditioning_data_0_1.npz")
    image_1_dict = load_tensor_dict("/home/aswerdlow/research/gen/outputs/debug/debug_debug_2024-01-28_14_53_30/conditioning_data_0_2.npz")

    image_0_tokens = image_0_dict["mask_tokens"]
    image_1_tokens = image_1_dict["mask_tokens"]

    prompt_image_0, conditioning_data_0 = self.infer_batch(
        batch=batch,
        conditioning_data=ConditioningData(mask_tokens=image_0_tokens, mask_batch_idx=torch.zeros((image_0_tokens.shape[0],), dtype=torch.int64)),
    )
    prompt_image_1, conditioning_data_1 = self.infer_batch(
        batch=batch,
        conditioning_data=ConditioningData(mask_tokens=image_1_tokens, mask_batch_idx=torch.zeros((image_1_tokens.shape[0],), dtype=torch.int64)),
    )

    final_tokens, final_rgb = take_from(
        slices=(
            # (1, slice(0, 1)),
            (0, slice(1, 2)),
            (1, slice(1, 2)),
        ),
        data=(image_0_dict, image_1_dict),
    )

    mask_batch_idx = torch.zeros((final_tokens.shape[0],), dtype=torch.int64)
    prompt_image, conditioning_data = self.infer_batch(
        batch=batch, conditioning_data=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=4
    )

    from torchvision import utils

    Im.concat_vertical(
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Gen {i}", relative_font_scale=0.004)
            for i, prompt_image_ in enumerate(prompt_image)
        ),
        Im(utils.make_grid(Im(final_rgb).torch)).write_text("Combined Conditioned masks [GT]"),
        Im.concat_horizontal(
            Im(image_0_dict["orig_image"]).write_text("First Image GT", relative_font_scale=0.001),
            Im(prompt_image_0[0]).write_text("First Image Autoencoded", relative_font_scale=0.001),
        ),
        Im.concat_horizontal(
            Im(image_1_dict["orig_image"]).write_text("Second Image GT", relative_font_scale=0.001),
            Im(prompt_image_1[0]).write_text("Second Image Autoencoded", relative_font_scale=0.001),
        ),
    ).save("test_00.png")

    breakpoint()
