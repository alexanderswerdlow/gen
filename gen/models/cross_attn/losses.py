from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from image_utils import Im
from PIL import Image

from gen.models.cross_attn.break_a_scene import AttentionStore

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import ConditioningData, InputData
    from gen.configs.base import BaseConfig
    
def break_a_scene_cross_attn_loss(cfg: BaseConfig, batch: InputData, controller: AttentionStore, conditioning_data: ConditioningData):
    attn_loss = 0
    batch_size: int = batch["disc_pixel_values"].shape[0]
    gen_seg_ = rearrange(batch["gen_segmentation"], "b h w c -> b c () h w").float()
    learnable_idxs = (batch["input_ids"] == conditioning_data.placeholder_token).nonzero(as_tuple=True)

    for batch_idx in range(batch_size):
        GT_masks = F.interpolate(input=gen_seg_[batch_idx], size=(16, 16))  # We interpolate per mask separately
        agg_attn = aggregate_attention(controller, res=16, from_where=("up", "down"), is_cross=True, select=batch_idx, batch_size=batch_size)

        cur_batch_mask = learnable_idxs[0] == batch_idx  # Find the masks for this batch
        token_indices = learnable_idxs[1][cur_batch_mask]

        # We may dropout masks so we need to map between dataset segmentation idx and the mask idx in the sentence
        segmentation_indices = conditioning_data.mask_instance_idx[cur_batch_mask] 
        attn_masks = agg_attn[..., token_indices]
        imgs = []
        for idx, mask_id in enumerate(segmentation_indices):
            asset_attn_mask = attn_masks[..., idx] / attn_masks[..., idx].max()

            # attn_loss += F.mse_loss(
            #     asset_attn_mask.float(),
            #     GT_masks[mask_id, 0].float(),
            #     reduction="mean",
            # )

            # This loss seems to be better
            attn_loss += F.mse_loss((asset_attn_mask.reshape(-1).softmax(dim=0).reshape(16, 16) * GT_masks[mask_id, 0].float()).sum(), torch.tensor(1.0).to(GT_masks.device))

        #     if batch["state"].true_step % 20 == 0:
        #         imgs.append(Im(Im.concat_horizontal(Im(GT_masks[mask_id, 0].float()[None, ..., None]), Im(asset_attn_mask[None, ..., None])).torch[0, ..., None]).pil)

        # if batch["state"].true_step % 20 == 0:
        #     Im.concat_vertical(*imgs).save(f'attn_{batch["state"].true_step}_{batch_idx}.png')
        #     Im(agg_attn.permute(2, 0, 1)[..., None]).normalize(normalize_min_max=True).save(f'all_attn_{batch["state"].true_step}_{batch_idx}.png')

    attn_loss = cfg.model.break_a_scene_cross_attn_loss_weight * (attn_loss / batch_size)
    controller.reset()
    return attn_loss


def remove_element(tensor, row_index):
    return torch.cat((tensor[..., :row_index], tensor[..., row_index + 1 :]), dim=-1)


def evenly_weighted_mask_loss(
    batch: InputData,
    conditioning_data: ConditioningData,
    pred: torch.Tensor,
    target: torch.Tensor,
):
    
    pred = rearrange(pred, 'b c h w -> b c (h w)')
    target = rearrange(target, 'b c h w -> b c (h w)')

    losses = []

    for b in range(batch['gen_pixel_values'].shape[0]):
        if conditioning_data.batch_cond_dropout is not None and conditioning_data.batch_cond_dropout[b].item():
            losses.append(F.mse_loss(pred[b], target[b], reduction="mean"))
            continue

        mask_idxs_for_batch = conditioning_data.mask_instance_idx[conditioning_data.mask_batch_idx == b]
        object_masks = batch["gen_segmentation"][b, ..., mask_idxs_for_batch]

        gt_masks = F.interpolate(rearrange(object_masks, 'h w c -> c () h w').float(), size=(64, 64)).squeeze(1)
        gt_masks = rearrange(gt_masks, 'c h w -> c (h w)') > 0.5
        
        batch_losses = []
        for i in range(object_masks.shape[-1]):
            pred_ = pred[b, :, gt_masks[i, :]]
            target_ = target[b, :, gt_masks[i, :]]
            loss = F.mse_loss(pred_, target_, reduction="mean")
            batch_losses.append(loss)

        if len(batch_losses) == 0:
            losses.append(torch.tensor(0.0))
        else:
            losses.append(torch.stack(batch_losses).mean())

    return torch.stack(losses).mean()

def break_a_scene_masked_loss(
    cfg: BaseConfig,
    batch: InputData,
    conditioning_data: ConditioningData
):
    max_masks = []
    for b in range(batch['gen_pixel_values'].shape[0]):
        mask_idxs_for_batch = conditioning_data.mask_instance_idx[conditioning_data.mask_batch_idx == b]
        object_masks = batch["gen_segmentation"][b, ..., mask_idxs_for_batch]
        if conditioning_data.batch_cond_dropout is not None and conditioning_data.batch_cond_dropout[b].item(): # We do not have conditioning for this entire sample so put loss on everything
            max_masks.append(object_masks.new_ones((512, 512)))
        elif object_masks.shape[-1] == 0:
            max_masks.append(object_masks.new_zeros((512, 512))) # Zero out loss if there are no masks
        else:
            max_masks.append(torch.max(object_masks, axis=-1).values)

    max_masks = torch.stack(max_masks, dim=0)[:, None]
    downsampled_mask = F.interpolate(input=max_masks.float(), size=(64, 64))

    return downsampled_mask
