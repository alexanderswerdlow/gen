
from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from einx import rearrange, mean

from gen.models.base.losses import transform_coordinate_space
from gen.models.dustr.marigold import NearFarMetricNormalizer

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.utils.data_defs import InputData

def xyz_to_depth(pcd, camera_intrinsics, camera_pose, simple: bool = False):
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.float().cpu().numpy()

    if isinstance(camera_intrinsics, torch.Tensor):
        camera_intrinsics = camera_intrinsics.float().cpu().numpy()

    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.float().cpu().numpy()

    if simple:
        depthmap = np.sqrt(np.sum(np.power(pcd - camera_pose[None, None, :3, 3], 2), axis=-1))
        return depthmap
    
    R_world2cam = camera_pose[:3, :3].T
    t_world2cam = -R_world2cam @ camera_pose[:3, 3]
    pcd_cam = np.einsum('ijk,kl->ijl', pcd, R_world2cam) + t_world2cam

    x_cam, y_cam, z_cam = pcd_cam[..., 0], pcd_cam[..., 1], pcd_cam[..., 2]
    fu, fv = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cu, cv = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    u = fu * x_cam / z_cam + cu
    v = fv * y_cam / z_cam + cv

    H, W, _ = pcd.shape
    depthmap = np.zeros((H, W), dtype=np.float32)
    valid = (z_cam > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid_z = z_cam[valid]
    valid_u = np.round(u[valid]).astype(int)
    valid_v = np.round(v[valid]).astype(int)

    valid_u = np.clip(valid_u, 0, W - 1)
    valid_v = np.clip(valid_v, 0, H - 1)

    depthmap[valid_v, valid_u] = valid_z

    

    return depthmap

def depreciated_fill_invalid_regions(input_xyz, input_valid):
    B, H, W, C = input_xyz.shape

    x = rearrange('b h w c -> b c h w', input_xyz.clone())
    x[rearrange('b h w -> b 3 h w', ~input_valid)] = 0
    
    n = 8
    scales = [x]
    for scale in range(1, n + 1):
        downsampled_size = (H // (2 ** scale), W // (2 ** scale))
        downsampled = F.interpolate(x, size=downsampled_size, mode='nearest')
        scales.append(F.interpolate(downsampled, size=(H, W), mode='nearest'))

    result = x.clone()
    
    zero_mask = rearrange('b h w -> b 3 h w', ~input_valid)
    for scale_img in scales:
        replace_mask = zero_mask & (scale_img != 0)
        result[replace_mask] = scale_img[replace_mask]
        zero_mask = zero_mask & (scale_img == 0)
    
    result = rearrange('b c h w -> b h w c', result)
    return result

@torch.no_grad()
def encode_xyz(cfg: BaseConfig, gt_points, init_valid_mask, vae, kwargs = None):
    _kwargs = dict(valid_mask=init_valid_mask, clip=True, per_axis_quantile=True)
    _kwargs.update(kwargs or {})

    normalizer = NearFarMetricNormalizer(min_max_quantile=cfg.model.xyz_min_max_quantile)
    pre_enc, post_enc_valid_mask = normalizer(gt_points, **_kwargs)
    post_enc_valid_mask = (~post_enc_valid_mask & init_valid_mask).to(torch.bool)

    if cfg.model.separate_xyz_encoding:
        pre_enc = rearrange('b xyz h w -> (b xyz) 3 h w', pre_enc)

    with torch.autocast(device_type="cuda", enabled=cfg.model.force_fp32_pcd_vae is False):
        latents = vae.encode(pre_enc).latent_dist.sample() * vae.config.scaling_factor

    if cfg.model.separate_xyz_encoding:
        latents = rearrange('(b xyz) c h w -> b (xyz c) h w', latents, xyz=3)

    return latents, post_enc_valid_mask, normalizer

@torch.no_grad()
def decode_xyz(cfg: BaseConfig, pred_latents, vae, normalizer):
    pred_latents = (1 / vae.config.scaling_factor) * pred_latents

    if cfg.model.separate_xyz_encoding:
        pred_latents = rearrange('b (xyz c) h w -> (b xyz) c h w', pred_latents, xyz=3)

    with torch.autocast(device_type="cuda", enabled=cfg.model.force_fp32_pcd_vae is False):
        decoded_points = vae.decode(pred_latents.to(torch.float32), return_dict=False)[0]
        if cfg.model.separate_xyz_encoding:
            decoded_points = mean('(b xyz) [c] h w -> b xyz h w', decoded_points, xyz=3)
        decoded_points = normalizer.denormalize(decoded_points)

    return decoded_points

def get_input(cfg: BaseConfig, batch: InputData):
    if cfg.model.predict_depth:
        input_src, input_tgt = batch.src_dec_depth, batch.tgt_dec_depth
        input_src, input_tgt = rearrange('b h w, b h w -> b h w 3, b h w 3', input_src, input_tgt)
    else:
        input_src, input_tgt = transform_coordinate_space(batch, batch.src_xyz, batch.tgt_xyz)

    input_xyz = rearrange('b h w xyz, b h w xyz -> (b + b) h w xyz', input_src, input_tgt)
    input_valid = rearrange('b h w, b h w -> (b + b) h w', batch.src_xyz_valid, batch.tgt_xyz_valid)
    return input_xyz, input_valid


if __name__ == "__main__":
    pass