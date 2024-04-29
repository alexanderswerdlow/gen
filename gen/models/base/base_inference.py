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
from gen.models.dustr.depth_utils import encode_xyz, decode_xyz, get_input, xyz_to_depth
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
        _repeat_dim = 2 if self.cfg.model.duplicate_unet_input_channels else 1
        cond.encoder_hidden_states = cond.encoder_hidden_states.repeat(_repeat_dim, 1, 1)
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

def norm_two(arr1, arr2):
    _min, _max = arr1.min(), arr1.max()
    arr1 = (arr1 - _min) / (_max - _min)
    arr2 = (arr2 - _min) / (_max - _min)
    return arr1, arr2

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def get_valid_mse(arr1, arr2, mask, norm: bool = False):
    if norm:
        arr1, arr2 = norm_two(arr1, arr2)

    _gt_xyz = rearrange('b h w xyz -> (b h w) xyz', arr1)
    _pred_xyz = rearrange('b h w xyz -> (b h w) xyz', arr2)
    _pred_mask = rearrange('b h w -> (b h w)', mask)
    return F.mse_loss(_pred_xyz[_pred_mask], _gt_xyz[_pred_mask], reduction="mean")

def get_depth(cfg, _gt_depth, _gt, _pred, _intrinsics, _extrinsics, b):
    if cfg.model.predict_depth:
        _gt_depth = _gt_depth[b]
        _pred_depth = _pred[b].mean(dim=-1)
    else:
        _gt_depth = torch.from_numpy(xyz_to_depth(_gt[b], _intrinsics[b], _extrinsics[b], simple=True)).to(_gt)
        _pred_depth = torch.from_numpy(xyz_to_depth(_pred[b], _intrinsics[b], _extrinsics[b], simple=True)).to(_pred)

    return norm_two(_gt_depth, _pred_depth)

def get_valid(arr, mask):
    _arr = arr.clone()
    _arr[~mask] = 0
    return _arr

@torch.no_grad()
def run_qualitative_inference(self: BaseMapper, batch: InputData, state: TrainingState, accelerator: Optional[Any] = None) -> dict:
    ret = {}

    input_xyz, input_valid = get_input(self.cfg, batch)
    input_src_valid, input_tgt_valid = torch.chunk(input_valid, 2, dim=0)

    xyz_latents, xyz_valid, normalizer = encode_xyz(self.cfg, input_xyz, input_valid, self.vae, self.dtype, 2, kwargs=dict(per_axis_quantile=self.cfg.model.predict_depth is False))
    src_valid, tgt_valid = torch.chunk(xyz_valid, 2, dim=0)
    autoencoded_xyz = autoencode_gt_xyz(self.cfg, batch, self.vae, xyz_latents, normalizer)
    autoencoded_src_xyz, autoencoded_tgt_xyz = torch.chunk(autoencoded_xyz, 2, dim=0)
    if self.cfg.model.predict_depth:
        input_xyz = input_xyz[..., [0]]
    else:
        ret['wandb_metric_autoencode_l2_scale_shift_inv'] = get_dustr_loss(batch, autoencoded_xyz, xyz_valid)

    ret['wandb_metric_autoencode_valid_xyz_mse'] = get_valid_mse(input_xyz, autoencoded_xyz, xyz_valid, norm=self.cfg.model.predict_depth)
    if self.cfg.model.predict_depth is False:
        for i in range(3):
            ret[f'wandb_metric_autoencode_valid_xyz_mse_channel_{i}'] = get_valid_mse(input_xyz[..., [i]], autoencoded_xyz[..., [i]], xyz_valid)

    if self.cfg.model.unet is False:
        return ret

    rgb_latents = self.get_rgb_latents(batch)

    pipeline_kwargs = dict()
    if self.cfg.model.only_noise_tgt:
        pipeline_kwargs['concat_src_depth'] = xyz_latents[:xyz_latents.shape[0]//2]

    pred_latents, cond = self.infer_batch(batch, concat_rgb=rgb_latents, num_images_per_prompt=1, output_type='latent', **pipeline_kwargs)
    
    pred_xyz = decode_xyz(self.cfg, pred_latents, self.vae, normalizer)

    ret['wandb_metric_valid_xyz_mse'] = get_valid_mse(input_xyz, pred_xyz, xyz_valid)
    if self.cfg.model.predict_depth is False:
        ret['wandb_metric_l2_scale_shift_inv'] = get_dustr_loss(batch, pred_xyz, xyz_valid)
        for i in range(3):
            ret[f'wandb_metric_valid_xyz_mse_channel_{i}'] = get_valid_mse(input_xyz[..., [i]], pred_xyz[..., [i]], xyz_valid)

    pred_src_xyz, pred_tgt_xyz = torch.chunk(pred_xyz, 2, dim=0)

    src_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.src_dec_rgb)
    tgt_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.tgt_dec_rgb)

    imgs = []
    secondary_viz = []
    for b in range(batch.bs)[:1]:
        src_gt_depth, src_pred_depth = get_depth(self.cfg, batch.src_dec_depth, batch.src_xyz, pred_src_xyz, batch.src_intrinsics, batch.src_extrinsics, b)
        tgt_gt_depth, tgt_pred_depth = get_depth(self.cfg, batch.tgt_dec_depth, batch.tgt_xyz, pred_tgt_xyz, batch.tgt_intrinsics, batch.tgt_extrinsics, b)

        src_autoencoded_depth = torch.from_numpy(
            norm(xyz_to_depth(autoencoded_src_xyz[b], batch.src_intrinsics[b], batch.src_extrinsics[b], simple=True))
        ).to(self.device)
        tgt_autoencoded_depth = torch.from_numpy(
            norm(xyz_to_depth(autoencoded_tgt_xyz[b], batch.src_intrinsics[b], batch.src_extrinsics[b], simple=True))
        ).to(self.device)

        src_gt_orig_depth, tgt_gt_orig_depth = batch.src_dec_depth[b], batch.tgt_dec_depth[b]

        _func = lambda x: rearrange('h w -> () h w 3', x)

        imgs.append(
            Im.concat_horizontal(
                Im.concat_vertical(
                    src_rgb[[b]],
                    Im(_func(src_gt_depth)).bool_to_rgb().write_text("GT Projected PCD", size=0.6),
                    Im(_func(src_pred_depth)).bool_to_rgb().write_text("Pred PCD", size=0.6),
                    Im(_func(src_autoencoded_depth)).bool_to_rgb().write_text("Autoencoded PCD", size=0.6),
                    Im(_func(get_valid(src_pred_depth, src_valid[b]))).bool_to_rgb().write_text("Pred PCD Valid", size=0.6),
                    Im(src_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
                ),
                Im.concat_vertical(
                    tgt_rgb[[b]],
                    Im(_func(tgt_gt_depth)).bool_to_rgb().write_text("GT Projected PCD", size=0.6),
                    Im(_func(tgt_pred_depth)).bool_to_rgb().write_text("Pred PCD", size=0.6),
                    Im(_func(tgt_autoencoded_depth)).bool_to_rgb().write_text("Autoencoded PCD", size=0.6),
                    Im(_func(get_valid(tgt_pred_depth, tgt_valid[b]))).bool_to_rgb().write_text("Pred PCD Valid", size=0.6),
                    Im(tgt_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
                ),
            )
        )

        secondary_viz.append(
            Im.concat_horizontal(
                Im.concat_vertical(
                    Im(_func(src_gt_orig_depth)).bool_to_rgb().write_text("GT Depth", size=0.6),
                    Im(_func(get_valid(src_gt_orig_depth, input_src_valid[b]))).bool_to_rgb().write_text("GT Depth Dataset Valid", size=0.6),
                    Im(_func(get_valid(src_gt_orig_depth, src_valid[b]))).bool_to_rgb().write_text("GT Depth Truncated Valid", size=0.6),
                    Im(input_src_valid[b]).bool_to_rgb().write_text("Dataset Valid Mask", size=0.6),
                    Im(src_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
                ),
                Im.concat_vertical(
                    Im(_func(tgt_gt_orig_depth)).bool_to_rgb().write_text("GT Depth", size=0.6),
                    Im(_func(get_valid(tgt_gt_orig_depth, input_tgt_valid[b]))).bool_to_rgb().write_text("GT Depth Dataset Valid", size=0.6),
                    Im(_func(get_valid(tgt_gt_orig_depth, tgt_valid[b]))).bool_to_rgb().write_text("GT Depth Truncated Valid", size=0.6),
                    Im(input_tgt_valid[b]).bool_to_rgb().write_text("Dataset Valid Mask", size=0.6),
                    Im(tgt_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
                ),
            )
        ) 

    ret['imgs'] = Im.concat_horizontal(*imgs)
    ret['misc_data'] = Im.concat_horizontal(*imgs)

    return ret
                    
def autoencode_gt_xyz(cfg: BaseConfig, batch: InputData, vae, xyz_latents, normalizer):
    pred_xyz = decode_xyz(cfg, xyz_latents, vae, normalizer)
    return pred_xyz