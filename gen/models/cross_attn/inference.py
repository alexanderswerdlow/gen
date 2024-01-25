from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from einops import repeat
from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image

from gen.models.cross_attn.break_a_scene import aggregate_attention, save_cross_attention_vis
from gen.utils.logging_utils import log_info
from gen.utils.tokenization_utils import get_tokens
from gen.utils.trainer_utils import TrainingState

if TYPE_CHECKING:
    from gen.models.cross_attn.base_mapper import BaseMapper


def infer_batch(self: BaseMapper, batch: dict, num_images_per_prompt: int = 1, pipeline_kwargs: Optional[dict] = None) -> Image.Image:
    if pipeline_kwargs is None:
        with torch.cuda.amp.autocast():
            assert self.cfg.inference.empty_string_cfg
            pipeline_kwargs, conditioning_data = self.get_standard_conditioning_for_inference(batch=batch)

    images = self.pipeline(
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=self.cfg.inference.guidance_scale,
        **pipeline_kwargs,
    ).images

    return images, pipeline_kwargs, conditioning_data


@torch.no_grad()
def run_inference(self: BaseMapper, batch: dict, state: TrainingState):
    """
    TODO: Support batched inference at some point. Right now it is batched under the hood (for CFG and if num_images_per_prompt > 1) but we can likely do much better.
    """

    assert batch["input_ids"].shape[0] == 1
    orig_input_ids = batch["input_ids"].clone()

    ret = {}

    orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
    gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(
        text="GT", relative_font_scale=0.004
    )
    ret["validation"] = Im.concat_vertical(orig_image, gt_info)

    batch["input_ids"] = orig_input_ids.clone()
    prompt_image, pipeline_kwargs, conditioning_data = self.infer_batch(
        batch=batch,
        num_images_per_prompt=self.cfg.inference.num_images_per_prompt,
    )

    full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
    generated_images = Im.concat_horizontal(
        Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}", relative_font_scale=0.004)
        for i, prompt_image_ in enumerate(prompt_image)
    )
    ret["validation"] = Im.concat_vertical(ret["validation"], generated_images)

    if self.cfg.inference.save_prompt_embeds:
        assert conditioning_data["mask_instance_idx"].shape[0] == conditioning_data["mask_tokens"].shape[0]
        all_masks = [full_seg]
        for j in conditioning_data["mask_instance_idx"]:
            all_masks.append(
                Im(get_layered_image_from_binary_mask(batch["gen_segmentation"][..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8).write_text(
                    f"Mask: {j}"
                )
            )

        ret["individual_masks"] = Im.concat_horizontal(*all_masks, spacing=10)
        ret["mask_tokens"] = {"mask_tokens": conditioning_data["mask_tokens"][None]}

    if self.cfg.model.break_a_scene_cross_attn_loss:
        batch_size = (2 if self.cfg.inference.guidance_scale > 1.0 else 1) * self.cfg.inference.num_images_per_prompt

        agg_attn_cond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size
        )
        attn_img_cond = Im(save_cross_attention_vis(tokenizer=self.tokenizer, tokens=batch["input_ids"][0], attention_maps=agg_attn_cond)).write_text(
            "Cond"
        )

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
            save_cross_attention_vis(tokenizer=self.tokenizer, tokens=uncond_tokens["input_ids"][0], attention_maps=agg_attn_uncond)
        ).write_text("Uncond")

        ret["attn_vis"] = Im.concat_vertical(attn_img_cond, attn_img_uncond)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    if self.cfg.inference.num_masks_to_remove is not None:
        orig_gen_segmentation = batch["gen_segmentation"].clone()

        batch_ = {}
        batch_["gen_segmentation"] = []
        for j in range(orig_gen_segmentation.shape[-1])[: self.cfg.inference.num_masks_to_remove]:
            batch_["gen_segmentation"].append(orig_gen_segmentation[..., torch.arange(orig_gen_segmentation.size(-1)) != j])

        batch_["gen_segmentation"] = torch.cat(batch_["gen_segmentation"], dim=0)
        batch_["input_ids"] = orig_input_ids.clone()
        batch_["input_ids"] = batch_["input_ids"].repeat(batch_["gen_segmentation"].shape[0], 1)
    
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v not in batch_:
                    assert v.shape[0] == 1
                    batch_[k] = repeat(v[0], "... -> h ...", h=batch_["gen_segmentation"].shape[0])

        prompt_images, _, _ = self.infer_batch(batch=batch_)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

        removed_mask_imgs = []
        for j in range(orig_gen_segmentation.shape[-1])[: self.cfg.inference.num_masks_to_remove]:
            mask_image = Im(get_layered_image_from_binary_mask(orig_gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
            mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
            mask_alpha = (orig_gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
            composited_image = orig_image.pil.copy().convert("RGBA")
            composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
            composited_image = composited_image.convert("RGB")
            removed_mask_imgs.append(Im.concat_vertical(prompt_image[0], mask_image, composited_image, spacing=5, fill=(128, 128, 128)))

        ret["validation"] = Im.concat_horizontal(ret["validation"], *removed_mask_imgs)
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
        batch["input_ids"] = []
        for prompt in prompts:
            batch["input_ids"].append(get_tokens(tokenizer=self.tokenizer, prompt=prompt)[None])
        batch["input_ids"] = torch.cat(batch["input_ids"], dim=0)

        prompt_images, _, _ = self.infer_batch(batch=batch)

        ret["prompt_images"] = Im.concat_horizontal(
            prompt_image.write_text(prompt, relative_font_scale=0.003) for prompt_image, prompt in zip(prompt_images, prompts)
        )

    return ret


from pathlib import Path


@torch.no_grad()
def run_custom_inference(self: BaseMapper, batch: dict, state: TrainingState, filepath: Path, num_objects: int = 3):
    breakpoint()
