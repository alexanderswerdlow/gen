from __future__ import annotations

from contextlib import nullcontext
from json import encoder
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einx import mean, rearrange, softmax
from torchvision.transforms.functional import InterpolationMode, resize

from gen.datasets.abstract_dataset import Split
from gen.datasets.utils import get_stable_diffusion_transforms
from gen.models.dustr.depth_utils import decode_xyz, xyz_to_depth
from gen.models.dustr.marigold import NearFarMetricNormalizer
from gen.utils.data_defs import undo_normalization_given_transforms
from gen.utils.trainer_utils import TrainingState
from image_utils import Im

if TYPE_CHECKING:
    from gen.models.base.base_model import BaseMapper, ConditioningData, InputData


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
    
    rgb_to_encode = torch.cat([batch.src_dec_rgb, batch.tgt_dec_rgb], dim=0).to(dtype=self.dtype)
    latents = self.vae.encode(rgb_to_encode).latent_dist.sample() # Convert images to latent space
    latents = latents * self.vae.config.scaling_factor

    pred_latents, cond = self.infer_batch(batch, concat_rgb=latents, num_images_per_prompt=1, output_type='latent')

    cond.gt_xyz = rearrange('b h w xyz, b h w xyz -> (b + b) h w xyz', batch.src_xyz, batch.tgt_xyz)
    cond.xyz_valid = rearrange('b h w, b h w -> (b + b) h w', batch.src_xyz_valid, batch.tgt_xyz_valid)
    cond.xyz_normalizer = NearFarMetricNormalizer(min_max_quantile=0.1)
    cond.xyz_normalizer(cond.gt_xyz)

    pred_xyz, pred_mask = decode_xyz(pred_latents, cond.xyz_valid, self.vae, cond.xyz_normalizer)
    
    _pred_mask = rearrange('b h w -> (b h w)', pred_mask)
    _pred_xyz = rearrange('b h w xyz -> (b h w) xyz', pred_xyz)
    _gt_xyz = rearrange('b h w xyz -> (b h w) xyz', cond.gt_xyz)

    gt_min, gt_max = _gt_xyz.min(dim=0)[0], _gt_xyz.max(dim=0)[0]
    _pred_xyz = (_pred_xyz - gt_min) / (gt_max - gt_min)
    _gt_xyz = (_gt_xyz - gt_min) / (gt_max - gt_min)

    ret['wandb_metric_valid_norm_xyz_mse'] = F.mse_loss(_pred_xyz[_pred_mask], _gt_xyz[_pred_mask], reduction="mean")

    left_xyz, right_xyz = torch.chunk(pred_xyz, 2, dim=0)
    left_mask, right_mask = torch.chunk(pred_mask, 2, dim=0)

    src_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.src_dec_rgb)
    tgt_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.tgt_dec_rgb)
    
    imgs = []
    for b in range(batch.bs)[:1]:
        def get_depth(_gt, _pred, _intrinsics, _extrinsics):
            _gt_depth = xyz_to_depth(_gt[b], _intrinsics[b], _extrinsics[b])
            _pred_depth = xyz_to_depth(_pred[b], _intrinsics[b], _extrinsics[b])

            _min, _max = _gt_depth.min(), _gt_depth.max()
            _pred_min, _pred_max = _pred_depth.min(), _pred_depth.max()

            _gt_depth = (_gt_depth - _min) / (_max - _min)
            _pred_depth_gt_norm = (_pred_depth - _min) / (_max - _min)
            _pred_depth_norm = (_pred_depth - _pred_min) / (_pred_max - _pred_min)

            return _gt_depth, _pred_depth_gt_norm, _pred_depth_norm
        
        left_gt_depth, left_pred_depth, left_pred_depth_norm = get_depth(batch.src_xyz, left_xyz, batch.src_intrinsics, batch.src_extrinsics)
        right_gt_depth, right_pred_depth, right_pred_depth_norm = get_depth(batch.tgt_xyz, right_xyz, batch.tgt_intrinsics, batch.tgt_extrinsics)

        _func = lambda x: rearrange('h w -> () h w 3', x)

        imgs.append(
            Im.concat_horizontal(
                Im.concat_vertical(
                    src_rgb[[b]],
                    Im(_func(left_gt_depth)).bool_to_rgb().write_text("GT"),
                    Im(_func(left_pred_depth)).bool_to_rgb().write_text("Pred"),
                    Im(_func(right_pred_depth_norm)).bool_to_rgb().write_text("Pred"),
                ),
                Im.concat_vertical(
                    tgt_rgb[[b]],
                    Im(_func(right_gt_depth)).bool_to_rgb().write_text("GT"),
                    Im(_func(right_pred_depth)).bool_to_rgb().write_text("Pred"),
                    Im(_func(right_pred_depth_norm)).bool_to_rgb().write_text("Pred")
                ),
            )
        )
        
    from dust3r.losses import Regr3D_ScaleShiftInv, L21
    criterion = Regr3D_ScaleShiftInv(L21, gt_scale=True)
    
    gt1 = dict(pts3d=batch.src_xyz, camera_pose=batch.src_extrinsics, valid_mask=left_mask)
    gt2 = dict(pts3d=batch.tgt_xyz, camera_pose=batch.tgt_extrinsics, valid_mask=right_mask)
    pred1 = dict(pts3d=left_xyz)
    pred2 = dict(pts3d_in_other_view=right_xyz)

    loss, (left_loss, right_loss) = criterion(gt1, gt2, pred1, pred2)

    ret['wandb_metric_l2_scale_shift_inv'] = loss

    ret['imgs'] = Im.concat_horizontal(*imgs)
        
    return ret
