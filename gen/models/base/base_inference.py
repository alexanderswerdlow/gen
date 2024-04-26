from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einx import mean, rearrange, softmax

from gen.datasets.abstract_dataset import Split
from gen.datasets.utils import get_stable_diffusion_transforms
from gen.models.base.losses import get_dustr_loss, transform_coordinate_space
from gen.models.dustr.depth_utils import encode_xyz, decode_xyz, xyz_to_depth, fill_invalid_regions
from gen.models.dustr.marigold import NearFarMetricNormalizer
from gen.utils.data_defs import undo_normalization_given_transforms
from gen.utils.trainer_utils import TrainingState
from image_utils import Im

if TYPE_CHECKING:
    from gen.models.base.base_model import BaseMapper, ConditioningData, InputData
    from gen.configs.base import BaseConfig



@torch.no_grad()
def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    cond: Optional[ConditioningData] = None,
    allow_get_cond: bool = True,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if cond is None or len(cond.unet_kwargs) == 0 and allow_get_cond:
        cond = self.get_hidden_state(batch, cond)
        cond.encoder_hidden_states = cond.encoder_hidden_states.repeat(2, 1, 1)
        cond.unet_kwargs["prompt_embeds"] = cond.encoder_hidden_states

    if "guidance_scale" not in cond.unet_kwargs and "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = self.cfg.inference.guidance_scale

    if "num_images_per_prompt" not in cond.unet_kwargs:
        kwargs["num_images_per_prompt"] = num_images_per_prompt

    kwargs["height"] = self.cfg.model.decoder_resolution
    kwargs["width"] = self.cfg.model.decoder_resolution
    if self.cfg.trainer.profile_memory or self.cfg.trainer.fast_eval:
        kwargs["num_inference_steps"] = 1 # Required otherwise the profiler will create traces that are too large

    needs_autocast = self.cfg.model.freeze_unet is False or self.cfg.model.unfreeze_single_unet_layer or self.cfg.model.dual_attention or self.cfg.model.duplicate_unet_input_channels
    with torch.cuda.amp.autocast(dtype=self.dtype) if needs_autocast else nullcontext():
        images = self.pipeline(**cond.unet_kwargs, **kwargs).images

    return images, cond


@torch.no_grad()
def run_qualitative_inference(self: BaseMapper, batch: InputData, state: TrainingState, accelerator: Optional[Any] = None) -> dict:
    ret = {}
    
    rgb_to_encode = torch.cat([batch.src_dec_rgb, batch.tgt_dec_rgb], dim=0).to(next(self.vae.parameters()).dtype)
    latents = self.vae.encode(rgb_to_encode).latent_dist.sample() # Convert images to latent space
    latents = latents * self.vae.config.scaling_factor

    pred_latents, cond = self.infer_batch(batch, concat_rgb=latents, num_images_per_prompt=1, output_type='latent')

    input_src, input_tgt = transform_coordinate_space(batch, batch.src_xyz, batch.tgt_xyz)
    input_xyz = rearrange('b h w xyz, b h w xyz -> (b + b) h w xyz', input_src, input_tgt)
    input_valid = rearrange('b h w, b h w -> (b + b) h w', batch.src_xyz_valid, batch.tgt_xyz_valid)
    _, xyz_valid, normalizer = encode_xyz(self.cfg, input_xyz, input_valid, self.vae, kwargs=dict(per_axis_quantile=self.cfg.model.predict_depth is False))
    pred_xyz, pred_xyz_mask = decode_xyz(self.cfg, pred_latents, xyz_valid, self.vae, normalizer)

    ret['wandb_metric_l2_scale_shift_inv'] = get_dustr_loss(batch, pred_xyz, pred_xyz_mask)
    
    _pred_mask = rearrange('b h w -> (b h w)', pred_xyz_mask)
    _pred_xyz = rearrange('b h w xyz -> (b h w) xyz', pred_xyz)
    _gt_xyz = rearrange('b h w xyz -> (b h w) xyz', input_xyz)

    gt_min, gt_max = _gt_xyz.min(dim=0)[0], _gt_xyz.max(dim=0)[0]
    _pred_xyz = (_pred_xyz - gt_min) / (gt_max - gt_min)
    _gt_xyz = (_gt_xyz - gt_min) / (gt_max - gt_min)

    ret['wandb_metric_valid_norm_xyz_mse'] = F.mse_loss(_pred_xyz[_pred_mask], _gt_xyz[_pred_mask], reduction="mean")

    pred_src_xyz, pred_tgt_xyz = torch.chunk(pred_xyz, 2, dim=0)

    src_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.src_dec_rgb)
    tgt_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.tgt_dec_rgb)
    
    imgs = []
    for b in range(batch.bs)[:1]:
        def get_depth(_gt, _pred, _intrinsics, _extrinsics):
            if self.cfg.model.predict_depth:
                _gt_depth = _gt[b].mean(dim=-1)
                _pred_depth = _pred[b].mean(dim=-1)
            else:
                _gt_depth = xyz_to_depth(_gt[b], _intrinsics[b], _extrinsics[b], simple=True)
                _pred_depth = xyz_to_depth(_pred[b], _intrinsics[b], _extrinsics[b], simple=True)

            _min, _max = _gt_depth.min(), _gt_depth.max()
            _pred_min, _pred_max = _pred_depth.min(), _pred_depth.max()

            _gt_depth = (_gt_depth - _min) / (_max - _min)
            _pred_depth_gt_norm = (_pred_depth - _min) / (_max - _min)
            _pred_depth_norm = (_pred_depth - _pred_min) / (_pred_max - _pred_min)

            return _gt_depth, _pred_depth_gt_norm, _pred_depth_norm
        
        left_gt_depth, left_pred_depth, left_pred_depth_norm = get_depth(fill_invalid_regions(batch.src_xyz, batch.src_xyz_valid), pred_src_xyz, batch.src_intrinsics, batch.src_extrinsics)
        right_gt_depth, right_pred_depth, right_pred_depth_norm = get_depth(fill_invalid_regions(batch.tgt_xyz, batch.tgt_xyz_valid), pred_tgt_xyz, batch.tgt_intrinsics, batch.tgt_extrinsics)
        left_gt_orig_depth, right_gt_orig_depth = batch.src_dec_depth[b], batch.tgt_dec_depth[b]

        # norm from 0 to 1
        _left_min, _left_max = left_gt_orig_depth.min(), left_gt_orig_depth.max()
        _right_min, _right_max = right_gt_orig_depth.min(), right_gt_orig_depth.max()
        left_gt_orig_depth = (left_gt_orig_depth - _left_min) / (_left_max - _left_min)
        right_gt_orig_depth = (right_gt_orig_depth - _right_min) / (_right_max - _right_min)

        _func = lambda x: rearrange('h w -> () h w 3', x)

        imgs.append(
            Im.concat_horizontal(
                Im.concat_vertical(
                    src_rgb[[b]],
                    Im(_func(left_gt_depth)).bool_to_rgb().write_text("GT PCD"),
                    Im(_func(left_gt_orig_depth)).bool_to_rgb().write_text("GT Stock Depth"),
                    Im(_func(left_pred_depth)).bool_to_rgb().write_text("Pred"),
                    Im(_func(left_pred_depth_norm)).bool_to_rgb().write_text("Pred Norm"),
                ),
                Im.concat_vertical(
                    tgt_rgb[[b]],
                    Im(_func(right_gt_depth)).bool_to_rgb().write_text("GT PCD"),
                    Im(_func(right_gt_orig_depth)).bool_to_rgb().write_text("GT Stock Depth"),
                    Im(_func(right_pred_depth)).bool_to_rgb().write_text("Pred"),
                    Im(_func(right_pred_depth_norm)).bool_to_rgb().write_text("Pred Norm")
                ),
            )
        )

    ret['wandb_metric_decoding_only_l2_scale_shift_inv'] = get_dustr_loss(batch, *autoencode_gt_xyz(self.cfg, batch, self.vae), transform_coordinates=True)
    ret['wandb_metric_decoding_only_holes_filled_l2_scale_shift_inv'] = get_dustr_loss(batch, *autoencode_gt_xyz(self.cfg, batch, self.vae, fill_holes=True), transform_coordinates=True)
    ret['imgs'] = Im.concat_horizontal(*imgs)

    return ret
                    
def autoencode_gt_xyz(cfg: BaseConfig, batch: InputData, vae, fill_holes=False):
    input_xyz = rearrange('b h w xyz, b h w xyz -> (b + b) h w xyz', batch.src_xyz, batch.tgt_xyz)
    input_valid = rearrange('b h w, b h w -> (b + b) h w', batch.src_xyz_valid, batch.tgt_xyz_valid)

    if fill_holes:
        input_xyz = fill_invalid_regions(input_xyz, input_valid)

    xyz_latents, xyz_valid, normalizer = encode_xyz(cfg, input_xyz, input_valid, vae)
    pred_xyz, pred_xyz_mask = decode_xyz(cfg, xyz_latents, xyz_valid, vae, normalizer)

    return pred_xyz, pred_xyz_mask