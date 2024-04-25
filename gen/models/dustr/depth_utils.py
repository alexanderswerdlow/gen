
from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from einx import rearrange

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

    if simple:
        depthmap = np.sqrt(np.sum(np.power(pcd - camera_pose[None, None, :3, 3], 2), axis=-1))

    return depthmap

@torch.no_grad()
def encode_xyz(cfg: BaseConfig, gt_points, init_valid_mask, vae, min_max_quantile: float = 0.1, kwargs = None):
    _kwargs = dict(valid_mask=init_valid_mask, clip=True, per_axis_quantile=True)
    _kwargs.update(kwargs or {})

    normalizer = NearFarMetricNormalizer(min_max_quantile=min_max_quantile)
    pre_enc, post_enc_valid_mask = normalizer(gt_points, **_kwargs)
    post_enc_valid_mask = (~post_enc_valid_mask & init_valid_mask).to(torch.bool)
    with torch.autocast(device_type="cuda", enabled=cfg.model.force_fp32_pcd_vae is False):
        latents = vae.encode(pre_enc).latent_dist.sample() * vae.config.scaling_factor

    return latents, post_enc_valid_mask, normalizer

@torch.no_grad()
def decode_xyz(cfg: BaseConfig, pred_latents, post_enc_valid_mask, vae, normalizer):
    pred_latents = (1 / vae.config.scaling_factor) * pred_latents
    with torch.autocast(device_type="cuda", enabled=cfg.model.force_fp32_pcd_vae is False):
        decoded_points = vae.decode(pred_latents.to(torch.float32), return_dict=False)[0]
        decoded_points, outside_range_post = normalizer.denormalize(decoded_points)

    outside_range_post = outside_range_post.to(device=pred_latents.device)
    final_mask = post_enc_valid_mask & (~outside_range_post)

    return decoded_points, final_mask

if __name__ == "__main__":
    pass