from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einx import rearrange
from jaxtyping import Bool, Float, Integer
from tensordict import tensorclass
from torch import Tensor

from image_utils import Im, hist
from torchvision.transforms.functional import InterpolationMode, resize

from gen.models.dustr.depth_utils import xyz_to_depth
from gen.utils.visualization_utils import get_dino_pca

from image_utils import Im

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.utils.trainer_utils import TrainingState

@tensorclass
class InputData:
    id: Integer[Tensor, "b"] = None
    src_dec_rgb: Float[Tensor, "b c h w"] = None
    tgt_dec_rgb: Float[Tensor, "b c h w"] = None
    dec_rgb: Float[Tensor, "b n c h w"] = None

    src_enc_rgb: Optional[Float[Tensor, "b c h w"]] = None
    tgt_enc_rgb: Optional[Float[Tensor, "b c h w"]] = None

    src_dec_depth: Optional[Float[Tensor, "b h w"]] = None
    tgt_dec_depth: Optional[Float[Tensor, "b h w"]] = None
    dec_depth: Optional[Float[Tensor, "b n h w"]] = None

    src_xyz: Optional[Float[Tensor, "b h w 3"]] = None
    tgt_xyz: Optional[Float[Tensor, "b h w 3"]] = None
    xyz: Optional[Float[Tensor, "b n h w 3"]] = None

    src_xyz_valid: Optional[Bool[Tensor, "b h w"]] = None
    tgt_xyz_valid: Optional[Bool[Tensor, "b h w"]] = None
    xyz_valid: Optional[Bool[Tensor, "b n h w"]] = None

    src_intrinsics: Optional[Float[Tensor, "b 3 3"]] = None
    tgt_intrinsics: Optional[Float[Tensor, "b 3 3"]] = None
    intrinsics: Optional[Float[Tensor, "b n 3 3"]] = None

    src_extrinsics: Optional[Float[Tensor, "b 4 4"]] = None
    tgt_extrinsics: Optional[Float[Tensor, "b 4 4"]] = None
    extrinsics: Optional[Float[Tensor, "b n 4 4"]] = None
    
    metadata: Optional[dict] = None
    state: Optional[TrainingState] = None
    dtype: Optional[torch.dtype]  = None
    num_frames: Optional[int] = None
    attach_debug_info: bool = False

    @staticmethod
    def from_dict(batch: dict):
        batch = InputData(
            num_frames=1,
            batch_size=[batch["dec_rgb"].shape[0] if "dec_rgb" in batch else batch["src_dec_rgb"].shape[0]],
            **batch,
        )
        return batch
    
    @property
    def bs(self):
        return self.batch_size[0]
    
    @property
    def n(self):
        return 2 if self.dec_rgb is None else self.dec_rgb.shape[1]

def visualize_input_data(
        batch: InputData, 
        name: Optional[str] = None, 
        names: Optional[list[str]] = None, 
        cfg: Optional[BaseConfig] = None,
        return_img: bool = False,
        cond: Optional[Any] = None,
    ):
    if cfg is not None:
        rgb = rearrange('(b n) ... -> b n ...', undo_normalization_given_transforms(cfg.dataset.val.augmentation.src_transforms, rearrange('b n ... -> (b n) ...', batch.dec_rgb)), n=batch.n)
    else:
        rgb = (batch.dec_rgb + 1) / 2

    output_imgs = []

    for b in range(batch.bs):
        if names is not None:
            name = names[b]
        else:
            name = batch.metadata['name'][b]
        save_name = f'input_data_{name}_{b}.png'

        src_ = rgb[b][0]
        tgt_ = rgb[b][1]
        output_img = Im.concat_horizontal(src_, tgt_)

        def get_depth(_gt_depth, _intrinsics = None, _extrinsics = None, is_xyz: bool = True):
            if is_xyz:
                _gt_depth = xyz_to_depth(_gt_depth, _intrinsics, _extrinsics, simple=True)
            _min, _max = _gt_depth.min(), _gt_depth.max()
            _gt_depth = (_gt_depth - _min) / (_max - _min)
            return _gt_depth

        _func = lambda x: rearrange('h w -> () h w 3', x)

        _src = batch.src_dec_depth[b].clone()
        _src[~batch.src_xyz_valid[b]] = 0

        _tgt = batch.tgt_dec_depth[b].clone()
        _tgt[~batch.tgt_xyz_valid[b]] = 0
        
        output_img = Im.concat_vertical(
            output_img,
            *((Im.concat_horizontal(
                Im(_func(get_depth(batch.src_xyz[b], batch.src_intrinsics[b], batch.src_extrinsics[b]))).bool_to_rgb(),
                Im(_func(get_depth(batch.tgt_xyz[b], batch.tgt_intrinsics[b], batch.tgt_extrinsics[b]))).bool_to_rgb(),
            )) if batch.src_xyz is not None else ()),
            Im.concat_horizontal(
                Im(_func(get_depth(batch.src_dec_depth[b], is_xyz=False))).bool_to_rgb(),
                Im(_func(get_depth(batch.tgt_dec_depth[b], is_xyz=False))).bool_to_rgb(),
            ),
            Im.concat_horizontal(
                Im(_func(get_depth(_src, is_xyz=False))).bool_to_rgb(),
                Im(_func(get_depth(_tgt, is_xyz=False))).bool_to_rgb(),
            ),
            Im.concat_horizontal(
                Im(batch.src_xyz_valid[b]).bool_to_rgb(),
                Im(batch.tgt_xyz_valid[b]).bool_to_rgb(),
            ).write_text(f"{name}_idx_{batch.metadata['index'][b]}", size=0.5)
        )

        if cond is not None:
            def get_pca_img(_feat_map):
                try:
                    _feat_map = _feat_map.to(torch.float32).cpu()
                    return get_dino_pca(
                        _feat_map,
                        patch_h=cfg.model.encoder_latent_dim, 
                        patch_w=cfg.model.encoder_latent_dim, 
                        threshold=0.6, 
                        object_larger_than_bg=False, 
                        return_all=True
                    )
                except:
                    print("Error in get_pca_img")
                    return np.zeros((cfg.model.encoder_latent_dim, cfg.model.encoder_latent_dim, 3))

            if cond.src_feature_map is not None:
                for j in range(cond.src_feature_map.shape[1]):
                    _feats = cond.src_feature_map[b, j]
                    feature_img = get_pca_img(rearrange("d h w -> (h w) d", _feats.permute(2, 0, 1)))
                    output_img = Im.concat_horizontal(output_img, Im(feature_img).scale(10, resampling_mode=InterpolationMode.NEAREST))

            if cond.src_orig_feature_map is not None:
                _first_img = Im.concat_horizontal(Im(get_pca_img(cond.src_warped_feature_map[0, b])).scale(10, resampling_mode=InterpolationMode.NEAREST), Im(get_pca_img(cond.tgt_orig_feature_map[0, b])).scale(10, resampling_mode=InterpolationMode.NEAREST)).add_border(75, (128, 128, 128)).write_text("Left: Warped Src, Right: Tgt Orig", size=0.5)
                _second_img = Im.concat_horizontal(Im(get_pca_img(cond.src_orig_feature_map[0, b])).scale(10, resampling_mode=InterpolationMode.NEAREST), Im(get_pca_img(cond.tgt_warped_feature_map[0, b])).scale(10, resampling_mode=InterpolationMode.NEAREST)).add_border(75, (128, 128, 128)).write_text("Left: Orig Src, Right: Tgt Warped", size=0.5)
                output_img = Im.concat_horizontal(output_img, Im.concat_vertical(_first_img, _second_img, fill=(128, 128, 128)), spacing=40, fill=(128, 128, 128))

        if return_img:
            output_imgs.append(output_img)
        else:
            output_img.save(save_name)

    if return_img:
        return Im(torch.stack([img_.torch for img_ in output_imgs]))


def create_coordinate_array(H, W):
    # Create a meshgrid with normalized coordinates ranging from -1 to 1
    y = torch.linspace(-1, 1, steps=H)
    x = torch.linspace(-1, 1, steps=W)
    xv, yv = torch.meshgrid(x, y)
    # Stack to create the desired shape [2, H, W]
    coords = torch.stack((yv, xv), dim=0)
    return coords

def get_dropout_grid(latent_dim):
    return create_coordinate_array(latent_dim, latent_dim)

def get_tgt_grid(cfg, batch):
    downsampled_grid = F.interpolate(
        batch.tgt_grid,
        size=(cfg.model.decoder_latent_dim, cfg.model.decoder_latent_dim),
        mode='bilinear'
    )

    return downsampled_grid

def undo_normalization_given_transforms(normalization_transform, tensor):
    for transform_ in normalization_transform.transforms:
        if transform_.__class__.__name__ == "Normalize":
            mean = transform_.mean
            std = transform_.std
            break
    else:
        raise ValueError("Normalize transform not found in the given Compose object.")
    
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    
    return tensor * std + mean
