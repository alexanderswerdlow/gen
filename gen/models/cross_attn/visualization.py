import numpy as np
import torch
from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image

from gen.models.cross_attn.base_mapper import BaseMapper
from gen.models.cross_attn.break_a_scene import aggregate_attention, save_cross_attention_vis
from gen.utils.logging_utils import log_info
from gen.utils.trainer_utils import TrainingState


def infer_batch(
    self: BaseMapper,
    batch: dict,
    num_images_per_prompt: int = 1,
) -> Image.Image:
    with torch.cuda.amp.autocast():
        assert self.cfg.inference.empty_string_cfg
        pipeline_kwargs, input_prompt = self.get_standard_conditioning_for_inference(batch=batch)

    images = self.pipeline(
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=self.cfg.inference.guidance_scale,
        **pipeline_kwargs,
    ).images

    return images, input_prompt, pipeline_kwargs["prompt_embeds"]

@torch.no_grad()
def run_inference(self: BaseMapper, batch: dict, state: TrainingState):
    orig_input_ids = batch["input_ids"].clone()

    ret = {}

    orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
    gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(
        text="GT", relative_font_scale=0.004
    )

    batch["input_ids"] = orig_input_ids.clone()
    prompt_image, input_prompt, prompt_embeds = self.infer_batch(
        batch=batch,
        num_images_per_prompt=self.cfg.inference.num_images_per_prompt,
    )

    full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
    ret["image"] = Im.concat_horizontal(
        Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}", relative_font_scale=0.004)
        for i, prompt_image_ in enumerate(prompt_image)
    )

    if self.cfg.model.break_a_scene_cross_attn_loss:
        batch_size = (2 if self.cfg.inference.guidance_scale > 1.0 else 1) * self.cfg.inference.num_images_per_prompt

        agg_attn_cond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size
        )
        attn_img_cond = Im(save_cross_attention_vis(tokenizer=self.tokenizer, tokens=batch["input_ids"][0], attention_maps=agg_attn_cond)).write_text("Cond")

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

        ret['attn'] = Im.concat_vertical(attn_img_cond, attn_img_uncond)

    if self.cfg.inference.num_masks_to_remove is not None:
        gen_segmentation = batch["gen_segmentation"]
        removed_mask_imgs = []
        for j in range(gen_segmentation.shape[-1])[: self.cfg.inference.num_masks_to_remove]:
            log_info(f"Generating with removed mask {j}")
            batch["gen_segmentation"] = gen_segmentation[..., torch.arange(gen_segmentation.size(-1)) != j]
            batch["input_ids"] = orig_input_ids.clone()
            prompt_image, input_prompt, prompt_embeds = self.infer_batch(batch=batch)
            mask_image = Im(get_layered_image_from_binary_mask(gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
            mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
            mask_alpha = (gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
            composited_image = orig_image.pil.copy().convert("RGBA")
            composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
            composited_image = composited_image.convert("RGB")
            removed_mask_imgs.append(Im.concat_vertical(prompt_image, mask_image, composited_image, spacing=5, fill=(128, 128, 128)))

        ret["removed_masks"] = Im.concat_horizontal(removed_mask_imgs)

    return ret