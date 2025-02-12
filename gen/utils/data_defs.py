from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Integer
from tensordict import tensorclass
from torch import Tensor

from image_utils import Im, hist
from image_utils.standalone_image_utils import integer_to_color
from torchvision.transforms.functional import InterpolationMode, resize

from gen.utils.visualization_utils import get_dino_pca

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.utils.trainer_utils import TrainingState

@tensorclass
class InputData:
    tgt_pixel_values: Float[Tensor, "b c h w"]
    tgt_segmentation: Integer[Tensor, "b h w c"]
    src_pixel_values: Float[Tensor, "b c h w"]
    src_segmentation: Integer[Tensor, "b h w c"]
    input_ids: Integer[Tensor, "b l"]
    tgt_grid: Optional[Float[Tensor, "b h w 2"]] = None
    src_grid: Optional[Float[Tensor, "b h w 2"]] = None
    state: Optional[TrainingState] = None
    dtype: Optional[torch.dtype]  = None
    num_frames: Optional[int] = None

    quaternions: Optional[Float] = None
    raw_object_quaternions: Optional[Float] = None
    camera_quaternions: Optional[Float] = None
    positions: Optional[Float] = None
    valid: Optional[Bool[Tensor, "b classes"]] = None
    src_valid: Optional[Bool[Tensor, "b classes"]] = None
    tgt_valid: Optional[Bool[Tensor, "b classes"]] = None
    categories: Optional[Integer] = None
    asset_id: Optional[Integer] = None
    metadata: Optional[dict[str, Any]] = None
    formatted_input_ids: Optional[Integer[Tensor, "b l"]] = None
    tgt_pad_mask: Optional[Bool[Tensor, "b h w"]] = None
    src_pad_mask: Optional[Bool[Tensor, "b h w"]] = None
    one_hot_tgt_segmentation: Optional[Integer[Tensor, "b h w c"]] = None
    tgt_pose: Optional[Float[Tensor, "b t 4 ..."]] = None
    src_pose: Optional[Float[Tensor, "b t 4 ..."]] = None
    raw_dataset_image: Optional[Integer[Tensor, "b h w c"]] = None
    id: Optional[Integer[Tensor, "b"]] = None
    tgt_enc_norm_pixel_values: Optional[Float[Tensor, "b c h w"]] = None
    tgt_enc_norm_segmentation: Optional[Integer[Tensor, "b h w c"]] = None
    tgt_enc_norm_valid: Optional[Bool[Tensor, "b classes"]] = None
    has_global_instance_ids: Optional[Bool[Tensor, "b"]] = None

    attach_debug_info: bool = False
    treat_as_train_batch: bool = False
    force_forward_encoder_normalized_tgt: bool = False
    force_encode_all_masks: Optional[Bool[Tensor, "b"]] = None
    force_use_tgt_pos_emb: bool = False
    force_use_orig_src_tokens: bool = False
    force_repeat_cond_data: bool = False
    shared_src_tgt_instance_idxs: Optional[Integer[Tensor, "b max_instances"]] = None

    @staticmethod
    def from_dict(batch: dict):
        batch = InputData(
            num_frames=1,
            batch_size=[batch["tgt_pixel_values"].shape[0]],
            **batch,
        )
        return batch
    
    def validate(self):
        assert -1 <= self.tgt_pixel_values.min() <= self.tgt_pixel_values.max() <= 1
        assert self.tgt_segmentation.dtype == torch.uint8
        assert self.src_segmentation.dtype == torch.uint8

    @property
    def bs(self):
        return self.batch_size[0]


def one_hot_to_integer(mask, num_overlapping_channels: int = 1, assert_safe: bool = True):
    """
    Compress mask: input_tensor is a boolean tensor of shape (B, H, W, C). M is the maximum number of overlapping channels.
    Returns a tensor of shape (B, H, W, M) with the indices of the M channels with the highest values.
    """
    return_hwc = False
    if mask.ndim == 3:
        return_hwc = True
        mask = mask.unsqueeze(0)

    B, H, W, C = mask.shape
    if assert_safe:
        assert (torch.sum(mask, dim=[-1]) <= num_overlapping_channels).all()

    channels = torch.arange(C, device=mask.device)[None, None, None, :]
    masked_indices = torch.where(mask, channels, 255)
    sorted_indices, _ = masked_indices.sort(dim=-1)
    compressed_mask = sorted_indices[..., :num_overlapping_channels].to(torch.uint8)

    # Pad output as the user expected num_overlapping_channels channels
    if compressed_mask.shape[-1] < num_overlapping_channels:
        num_extra_channels = num_overlapping_channels - compressed_mask.shape[-1]
        extra_channels = repeat(torch.ones_like(compressed_mask[..., 0]), "... -> ... d", d=num_extra_channels)
        compressed_mask = torch.cat((compressed_mask, 255 * extra_channels), dim=-1)
    
    return compressed_mask if not return_hwc else compressed_mask.squeeze(0)

def integer_to_one_hot(compressed_mask, num_channels: Optional[int] = None, add_background_channel: bool = False):
    """
    Decompress mask: (B, H, W, M) uint8 -> (B, H, W, C) bool
    """
    if num_channels is None:
        assert compressed_mask.dtype == torch.uint8
        num_channels = compressed_mask[compressed_mask < 255].max() + 1
    return get_one_hot_channels(compressed_mask, torch.arange(num_channels, device=compressed_mask.device), add_background_channel=add_background_channel)

def get_one_hot_channels(compressed_mask, indices, add_background_channel: bool = False):
    """
    Decompress mask (B, H, W, M) uint8 -> (B, H, W, C) bool, for specific channels.
    """
    return_hw = False
    return_hwc = False
    if compressed_mask.ndim == 2:
        return_hw = True
        compressed_mask = compressed_mask.unsqueeze(0).unsqueeze(-1)
    elif compressed_mask.ndim == 3:
        return_hwc = True
        compressed_mask = compressed_mask.unsqueeze(0)

    B, H, W, M = compressed_mask.shape
    C = indices.shape[0]
    indices_expanded = indices.expand(B, H, W, M, C)
    decompressed = (compressed_mask.unsqueeze(-1) == indices_expanded).any(dim=-2)

    if add_background_channel:
        decompressed = torch.cat(((compressed_mask == 255).all(dim=-1)[..., None], decompressed), dim=-1)
    
    if return_hw:
        decompressed = decompressed.squeeze(0).squeeze(-1)

    if return_hwc:
        decompressed = decompressed.squeeze(0)

    return decompressed

def replace_invalid(arr, mask):
    indices = torch.arange(mask.size(1), device=arr.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    inv_mask = ~rearrange(mask, "b c -> b () () c")
    new_arr = torch.where(inv_mask & (arr == indices), torch.tensor(255, device=arr.device, dtype=arr.dtype), arr)
    return new_arr

def visualize_input_data(
        batch: InputData, 
        name: Optional[str] = None, 
        names: Optional[list[str]] = None, 
        show_background_foreground_only: bool = False, 
        show_overlapping_masks: bool = False,
        remove_invalid: bool = True,
        cfg: Optional[BaseConfig] = None,
        image_only: bool = False,
        return_img: bool = False,
        cond: Optional[Any] = None,
        tokenizer=None,
    ):

    from image_utils import Im, onehot_to_color

    if show_background_foreground_only:
        batch.tgt_segmentation[batch.tgt_segmentation > 0] = 1
        batch.src_segmentation[batch.src_segmentation > 0] = 1
        batch.tgt_segmentation[batch.tgt_segmentation < 0] = 1
        batch.src_segmentation[batch.src_segmentation < 0] = 1
    
    if remove_invalid and batch.tgt_segmentation.shape[-1] == 1:
        batch.src_segmentation = replace_invalid(batch.src_segmentation, batch.src_valid if batch.src_valid is not None else batch.src_segmentation.nonzero())
        batch.tgt_segmentation = replace_invalid(batch.tgt_segmentation, batch.tgt_valid if batch.tgt_valid is not None else batch.tgt_segmentation.nonzero())

    if cfg is not None:
        tgt_rgb = undo_normalization_given_transforms(cfg.dataset.val.augmentation.src_transforms, batch.tgt_pixel_values)
        src_rgb = undo_normalization_given_transforms(cfg.dataset.val.augmentation.src_transforms, batch.src_pixel_values)
    else:
        tgt_rgb = (batch.tgt_pixel_values + 1) / 2
        src_rgb = Im(batch.src_pixel_values).denormalize().torch
    
    if batch.tgt_grid is not None:
        batch.tgt_grid = (batch.tgt_grid + 1) / 2

    if batch.src_grid is not None:
        batch.src_grid = (batch.src_grid + 1) / 2

    override_colors = {0: (128, 128, 128), 255: (0, 0, 0)}

    output_imgs = []

    for b in range(batch.bs):
        if names is not None:
            name = names[b]
        else:
            name = batch.metadata['name'][b]
        save_name = f'input_data_{name}_{b}.png'

        if image_only:
            Im(batch.raw_dataset_image[b]).save(save_name)
            continue
        
        if len(torch.unique(batch.tgt_segmentation[b])) == 1: continue

        tgt_one_hot = integer_to_one_hot(batch.tgt_segmentation[b], add_background_channel=False)
        src_one_hot = integer_to_one_hot(batch.src_segmentation[b], add_background_channel=False)

        src_ = Im.concat_vertical(
            Im(src_rgb[b]), 
            Im(onehot_to_color(src_one_hot.squeeze(0), override_colors=override_colors, ignore_empty=remove_invalid)),
        )

        if batch.src_grid is not None:
            src_ = Im.concat_vertical(
                src_, Im(torch.cat((batch.src_grid[b], batch.src_grid.new_zeros((1, *batch.src_grid.shape[2:]))), dim=0))
            )
            
        tgt_ = Im.concat_vertical(
            Im(tgt_rgb[b]), 
            Im(onehot_to_color(tgt_one_hot.squeeze(0), override_colors=override_colors, ignore_empty=remove_invalid)),
        )
        if batch.tgt_grid is not None:
            tgt_ = Im.concat_vertical(
                tgt_, Im(torch.cat((batch.tgt_grid[b], batch.tgt_grid.new_zeros((1, *batch.tgt_grid.shape[2:]))), dim=0))
            )
        
        output_img = Im.concat_horizontal(src_, tgt_)
        if show_overlapping_masks:
            masks = rearrange(tgt_one_hot, "h w c -> c h w")
            initial_num_classes = masks.sum(axis=0).max() + 1
            initial_image = integer_to_color(masks.sum(axis=0), colormap='hot', num_classes=initial_num_classes, ignore_empty=remove_invalid)
            first_hist = hist(np.sum(masks.cpu().numpy(), axis=0).reshape(-1), save=False)
            first_masks = Im(masks.unsqueeze(-1)).scale(0.5).grid(pad_value=0.5).write_text(f"{batch.tgt_segmentation[b].min().item()}-{batch.tgt_segmentation[b][batch.tgt_segmentation[b] != 255].max().item()}, uniq: {len(torch.unique(batch.tgt_segmentation[b]))}", size=0.2).torch.cpu()
            output_img = Im.concat_horizontal(output_img, Im.concat_vertical(first_masks, initial_image, fill=(128, 128, 128)), spacing=40, fill=(128, 128, 128))
            output_img = Im.concat_vertical(
                output_img,
                first_hist.scale(1),
                spacing=40,
                fill=(128, 128, 128)
            )

        if batch.raw_dataset_image is not None:
            output_img = Im.concat_horizontal(batch.raw_dataset_image[b], output_img, spacing=50, fill=(128, 128, 128))

        if batch.src_pose is not None and batch.src_pose.shape[1] == 4 and batch.src_pose.ndim == 3:
            from gen.datasets.scannetpp.scannetpp import get_distance_matrix_vectorized
            rot, dist = get_distance_matrix_vectorized(torch.stack((batch.src_pose[b], batch.tgt_pose[b]), dim=0))
            output_img = output_img.write_text(f"Relative Pose Rot: {rot[0, 1]:.2f}, Relative Pose Dist: {dist[0, 1]:.2f}", (10, 10), size=0.25)

        if batch.input_ids is not None and tokenizer is not None:
            output_img = output_img.write_text(tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True)[b], position=(0.9525, 0.01), size=0.35)
        
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
                    feature_img = get_pca_img(rearrange(_feats.permute(2, 0, 1), "d h w -> (h w) d"))
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
