from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Integer
from omegaconf import OmegaConf
from tensordict import tensorclass
from torch import Tensor

from gen.utils.trainer_utils import TrainingState
from image_utils import Im, hist
from image_utils.standalone_image_utils import integer_to_color

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig


@tensorclass
class InputData:
    gen_pixel_values: Float[Tensor, "b c h w"]
    gen_segmentation: Integer[Tensor, "b h w"]
    disc_pixel_values: Float[Tensor, "b c h w"]
    disc_segmentation: Integer[Tensor, "b h w"]
    input_ids: Integer[Tensor, "b l"]
    gen_grid: Optional[Float[Tensor, "b h w 2"]] = None
    disc_grid: Optional[Float[Tensor, "b h w 2"]] = None
    state: Optional[TrainingState] = None
    dtype: Optional[torch.dtype]  = None
    bs: Optional[int] = None
    num_frames: Optional[int] = None

    quaternions: Optional[Float] = None
    raw_object_quaternions: Optional[Float] = None
    camera_quaternions: Optional[Float] = None
    positions: Optional[Float] = None
    valid: Optional[Bool[Tensor, "b classes"]] = None
    categories: Optional[Integer] = None
    asset_id: Optional[Integer] = None
    metadata: Optional[dict[str, Any]] = None
    formatted_input_ids: Optional[Integer[Tensor, "b l"]] = None
    gen_pad_mask: Optional[Bool[Tensor, "b h w"]] = None
    disc_pad_mask: Optional[Bool[Tensor, "b h w"]] = None
    one_hot_gen_segmentation: Optional[Integer[Tensor, "b h w c"]] = None
    gen_pose_out: Optional[Float[Tensor, "b t 4"]] = None
    disc_pose_in: Optional[Float[Tensor, "b t 4"]] = None
    raw_dataset_image: Optional[Integer[Tensor, "b h w c"]] = None

    @staticmethod
    def from_dict(batch: dict):
        batch = InputData(
            num_frames=1,
            batch_size=[batch["gen_pixel_values"].shape[0]],
            bs=batch["gen_pixel_values"].shape[0],
            **batch,
        )
        return batch
    
    def validate(self):
        assert -1 <= self.gen_pixel_values.min() <= self.gen_pixel_values.max() <= 1
        assert self.gen_segmentation.dtype == torch.uint8
        assert self.disc_segmentation.dtype == torch.uint8


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
    return arr

def visualize_input_data(
        batch: InputData, 
        name: Optional[str] = None, 
        names: Optional[list[str]] = None, 
        show_background_foreground_only: bool = False, 
        show_overlapping_masks: bool = False,
        remove_invalid: bool = True,
        cfg: Optional[BaseConfig] = None,
    ):

    if show_background_foreground_only:
        batch.gen_segmentation[batch.gen_segmentation > 0] = 1
        batch.disc_segmentation[batch.disc_segmentation > 0] = 1
        batch.gen_segmentation[batch.gen_segmentation < 0] = 1
        batch.disc_segmentation[batch.disc_segmentation < 0] = 1
    
    if remove_invalid:
        batch.gen_segmentation = replace_invalid(batch.gen_segmentation, batch.valid)
        batch.disc_segmentation = replace_invalid(batch.disc_segmentation, batch.valid)

    if cfg is not None:
        gen_rgb = undo_normalization_given_transforms(cfg.dataset.validation_dataset.augmentation.source_transforms, batch.gen_pixel_values)
        disc_rgb = undo_normalization_given_transforms(cfg.dataset.validation_dataset.augmentation.source_transforms, batch.disc_pixel_values)
    else:
        gen_rgb = (batch.gen_pixel_values + 1) / 2
        disc_rgb = Im(batch.gen_pixel_values).denormalize().torch
    
    if batch.gen_grid is not None:
        batch.gen_grid = (batch.gen_grid + 1) / 2

    if batch.disc_grid is not None:
        batch.disc_grid = (batch.disc_grid + 1) / 2

    override_colors = {0: (128, 128, 128), 255: (0, 0, 0)}

    from image_utils import Im, onehot_to_color
    for b in range(batch.bs):
        gen_one_hot = integer_to_one_hot(batch.gen_segmentation[b], add_background_channel=False)
        disc_one_hot = integer_to_one_hot(batch.disc_segmentation[b], add_background_channel=False)

        if names is not None:
            name = names[b]
            
        gen_ = Im.concat_vertical(
            Im(gen_rgb[b]), 
            Im(onehot_to_color(gen_one_hot.squeeze(0), override_colors=override_colors)),
        )
        if batch.gen_grid is not None:
            gen_ = Im.concat_vertical(
                gen_, Im(torch.cat((batch.gen_grid[b], batch.gen_grid.new_zeros((1, *batch.gen_grid.shape[2:]))), dim=0))
            )

        disc_ = Im.concat_vertical(
            Im(disc_rgb[b]), 
            Im(onehot_to_color(disc_one_hot.squeeze(0), override_colors=override_colors)),
        )

        if batch.disc_grid is not None:
            disc_ = Im.concat_vertical(
                disc_, Im(torch.cat((batch.disc_grid[b], batch.disc_grid.new_zeros((1, *batch.disc_grid.shape[2:]))), dim=0))
            )
        
        output_img = Im.concat_horizontal(disc_, gen_)
        if show_overlapping_masks:
            masks = rearrange(gen_one_hot, "h w c -> c h w")
            initial_num_classes = masks.sum(axis=0).max() + 1
            initial_image = integer_to_color(masks.sum(axis=0), colormap='hot', num_classes=initial_num_classes, ignore_empty=False)
            first_hist = hist(np.sum(masks.cpu().numpy(), axis=0).reshape(-1), save=False)
            first_masks = Im(masks.unsqueeze(-1)).scale(0.5).grid(pad_value=0.5)
            output_img = Im.concat_horizontal(output_img, first_masks, spacing=50, fill=(128, 128, 128))
            output_img = Im.concat_vertical(
                output_img,
                initial_image,
                first_hist.scale(1),
                spacing=120,
                fill=(128, 128, 128)
            )

        output_img.save(f'input_data_{name}_{b}.png')



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

def get_gen_grid(cfg, batch):
    downsampled_grid = F.interpolate(
        batch.gen_grid,
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
