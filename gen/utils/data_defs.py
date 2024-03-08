import torch

from dataclasses import dataclass, field, fields
from typing import Any, Optional
import torch
import torch.utils.checkpoint
from jaxtyping import Bool, Float, Integer
from omegaconf import OmegaConf
from torch import Tensor
from gen.utils.trainer_utils import Trainable, TrainingState, unwrap
from tensordict import tensorclass
import torch.nn.functional as F

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

    @staticmethod
    def from_dict(batch: dict):
        batch = InputData(
            num_frames=1,
            batch_size=[batch["gen_pixel_values"].shape[0]],
            bs=batch["gen_pixel_values"].shape[0],
            **batch,
        )
        return batch

def get_one_hot_channels(seg, indices):
    """
    Parameters:
    - seg: [H, W] tensor with integers
    - indices: [M] tensor with selected indices for one-hot encoding.
    - N: Number of classes (int).
    
    Returns:
    - [H, W, M] tensor representing the one-hot encoded segmentation map for selected indices.
    """
    H, W = seg.shape
    M = len(indices)
    
    seg_expanded = seg.unsqueeze(-1).expand(H, W, M)
    indices_expanded = indices.expand(H, W, M)
    output = (seg_expanded == indices_expanded)
    return output

def one_hot_to_integer(one_hot_mask):
    values, indices = one_hot_mask.max(dim=-1)
    return torch.where(values > 0, indices, torch.tensor(-1))

def integer_to_one_hot(int_tensor, num_classes):
    mask = (int_tensor >= 0) & (int_tensor < num_classes)
    int_tensor = torch.where(mask, int_tensor, torch.tensor(0))
    one_hot = torch.nn.functional.one_hot(int_tensor, num_classes)
    one_hot = torch.where(mask.unsqueeze(-1), one_hot, False)
    return one_hot

def visualize_input_data(batch: InputData, name: Optional[str] = None, show_background_foreground_only: bool = False):
    if show_background_foreground_only:
        batch.gen_segmentation[batch.gen_segmentation > 0] = 1
        batch.disc_segmentation[batch.disc_segmentation > 0] = 1
        # batch.gen_segmentation[batch.gen_segmentation < 0] = 1
        # batch.disc_segmentation[batch.disc_segmentation < 0] = 1
    
    gen_one_hot = integer_to_one_hot(batch.gen_segmentation + 1, batch.gen_segmentation.max() + 1)
    disc_one_hot = integer_to_one_hot(batch.disc_segmentation + 1, batch.disc_segmentation.max() + 1)
    
    if batch.gen_grid is not None:
        batch.gen_grid = (batch.gen_grid + 1) / 2
    if batch.disc_grid is not None:
        batch.disc_grid = (batch.disc_grid + 1) / 2

    from image_utils import Im, get_layered_image_from_binary_mask
    for b in range(batch.bs):            
        gen_ = Im.concat_vertical(
            Im((batch.gen_pixel_values[b] + 1) / 2), 
            Im(get_layered_image_from_binary_mask(gen_one_hot[b].squeeze(0))),
        )
        if batch.gen_grid is not None:
            gen_ = Im.concat_vertical(
                gen_, Im(torch.cat((batch.gen_grid[b], batch.gen_grid.new_zeros((1, *batch.gen_grid.shape[2:]))), dim=0))
            )

        disc_ = Im.concat_vertical(
            Im((batch.disc_pixel_values[b] + 1) / 2), 
            Im(get_layered_image_from_binary_mask(disc_one_hot[b].squeeze(0))),
        )

        if batch.disc_grid is not None:
            disc_ = Im.concat_vertical(
                disc_, Im(torch.cat((batch.disc_grid[b], batch.disc_grid.new_zeros((1, *batch.disc_grid.shape[2:]))), dim=0))
            )
        Im.concat_horizontal(disc_, gen_).save(f'input_data_{name}_{b}.png')


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