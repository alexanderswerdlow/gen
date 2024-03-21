from __future__ import annotations

import abc
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einx import rearrange
from PIL import Image
from scipy.spatial.transform import Rotation as R

import wandb
from gen.models.cross_attn.break_a_scene import AttentionStore, aggregate_attention
from gen.utils.data_defs import get_one_hot_channels, integer_to_one_hot
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.pytorch3d_transforms import matrix_to_quaternion
from gen.utils.rotation_utils import (compute_rotation_matrix_from_ortho6d, get_discretized_zyx_from_quat, get_ortho6d_from_rotation_matrix,
                                      get_quat_from_discretized_zyx, quat_l1_loss)
from image_utils import Im
import einops

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import ConditioningData, InputData, TokenPredData


def break_a_scene_cross_attn_loss(cfg: BaseConfig, batch: InputData, controller: AttentionStore, cond: ConditioningData):
    attn_loss = 0
    batch_size: int = batch.src_pixel_values.shape[0]
    tgt_seg_ = rearrange(integer_to_one_hot(batch.tgt_segmentation, cfg.model.segmentation_map_size), "b h w c -> b c () h w").float()
    learnable_idxs = (batch.formatted_input_ids == cond.placeholder_token).nonzero(as_tuple=True)

    for batch_idx in range(batch_size):
        GT_masks = F.interpolate(input=tgt_seg_[batch_idx], size=(16, 16))  # We interpolate per mask separately
        agg_attn = aggregate_attention(controller, res=16, from_where=("up", "down"), is_cross=True, select=batch_idx, batch_size=batch_size)

        cur_batch_mask = learnable_idxs[0] == batch_idx  # Find the masks for this batch
        token_indices = learnable_idxs[1][cur_batch_mask]

        # We may dropout masks so we need to map between dataset segmentation idx and the mask idx in the sentence
        segmentation_indices = cond.mask_instance_idx[cur_batch_mask]
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
            attn_loss += F.mse_loss(
                (asset_attn_mask.reshape(-1).softmax(dim=0).reshape(16, 16) * GT_masks[mask_id, 0].float()).sum(),
                torch.tensor(1.0).to(GT_masks.device),
            )

        #     if batch.state.true_step % 20 == 0:
        #         imgs.append(Im(Im.concat_horizontal(Im(GT_masks[mask_id, 0].float()[None, ..., None]), Im(asset_attn_mask[None, ..., None])).torch[0, ..., None]).pil)

        # if batch.state.true_step % 20 == 0:
        #     Im.concat_vertical(*imgs).save(f'attn_{batch.state.true_step}_{batch_idx}.png')
        #     Im(agg_attn.permute(2, 0, 1)[..., None]).normalize(normalize_min_max=True).save(f'all_attn_{batch.state.true_step}_{batch_idx}.png')

    attn_loss = cfg.model.break_a_scene_cross_attn_loss_weight * (attn_loss / batch_size)
    controller.reset()
    return attn_loss


def remove_element(tensor, row_index):
    return torch.cat((tensor[..., :row_index], tensor[..., row_index + 1 :]), dim=-1)


def evenly_weighted_mask_loss(
    cfg: BaseConfig,
    batch: InputData,
    cond: ConditioningData,
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred = rearrange(pred, "b c h w -> b c (h w)")
    target = rearrange(target, "b c h w -> b c (h w)")

    losses = []

    for b in range(batch.tgt_pixel_values.shape[0]):
        if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
            losses.append(F.mse_loss(pred[b], target[b], reduction="mean"))
            continue

        mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
        object_masks = get_one_hot_channels(batch.tgt_segmentation[b], mask_idxs_for_batch)

        gt_masks = F.interpolate(rearrange(object_masks, "h w c -> c () h w").float(), size=(cfg.model.decoder_latent_dim, cfg.model.decoder_latent_dim)).squeeze(1)
        gt_masks = rearrange(gt_masks, "c h w -> c (h w)") > 0.5

        batch_losses = []
        for i in range(object_masks.shape[-1]):
            pred_ = pred[b, :, gt_masks[i, :]]
            tgt_ = target[b, :, gt_masks[i, :]]
            loss = F.mse_loss(pred_, tgt_, reduction="mean")
            batch_losses.append(loss)

        if len(batch_losses) == 0:
            losses.append(torch.tensor(0.0))
        else:
            losses.append(torch.stack(batch_losses).mean())

    return torch.stack(losses).mean()


def break_a_scene_masked_loss(cfg: BaseConfig, batch: InputData, cond: ConditioningData):
    max_masks = []
    decoder_resolution = cfg.model.decoder_resolution
    for b in range(batch.tgt_pixel_values.shape[0]):
        mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
        object_masks = get_one_hot_channels(batch.tgt_segmentation[b], mask_idxs_for_batch)
        if (
            cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item()
        ):  # We do not have conditioning for this entire sample so put loss on everything
            max_masks.append(object_masks.new_ones((decoder_resolution, decoder_resolution)))
        elif object_masks.shape[-1] == 0:
            max_masks.append(object_masks.new_zeros((decoder_resolution, decoder_resolution)))  # Zero out loss if there are no masks
        else:
            max_masks.append(torch.max(object_masks, axis=-1).values)

    max_masks = torch.stack(max_masks, dim=0)[:, None]
    loss_mask = F.interpolate(input=max_masks.float(), size=(cfg.model.decoder_latent_dim, cfg.model.decoder_latent_dim))

    if cfg.model.viz and batch.state.true_step % 1 == 0:
        from image_utils import Im

        rgb_ = Im((batch.tgt_pixel_values + 1) / 2)
        mask_ = Im(loss_mask).resize(rgb_.height, rgb_.width)
        Im.concat_horizontal(rgb_.grid(), mask_.grid()).save(f'rgb_{batch.state.true_step}.png')

    return loss_mask


def token_cls_loss(
    cfg: BaseConfig,
    batch: InputData,
    cond: ConditioningData,
    pred_data: TokenPredData,
):

    cls_pred = pred_data.cls_pred
    bs = batch.tgt_pixel_values.shape[0]
    device = batch.tgt_pixel_values.device

    assert cfg.model.background_mask_idx == 0

    losses = []
    correct_predictions = 0
    correct_top3_predictions = 0
    total_instances = 0

    for b in range(bs):
        if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
            continue

        # We only compute loss on non-dropped out masks
        instance_categories = batch.categories[b]

        # We align the dataset instance indices with the flattened pred indices
        mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
        pred_idxs_for_batch = torch.arange(cond.mask_instance_idx.shape[0], device=device)[cond.mask_batch_idx == b]

        # The background is always 0 so we must remove it if it exists and move everything else down
        remove_background_mask = mask_idxs_for_batch != 0

        mask_idxs_for_batch = mask_idxs_for_batch[remove_background_mask]
        pred_idxs_for_batch = pred_idxs_for_batch[remove_background_mask]

        pred_ = cls_pred[pred_idxs_for_batch]
        instance_categories = instance_categories[mask_idxs_for_batch - 1]

        if instance_categories.shape[0] == 0:
            continue  # This can happen if we previously dropout all masks except the background
        
        loss = F.cross_entropy(pred_, instance_categories.long())
        losses.append(loss)

        _, predicted_labels = pred_.max(dim=1)
        correct_predictions += (predicted_labels == instance_categories).sum().item()

        _, top3_preds = pred_.topk(k=3, dim=1)
        correct_top3_predictions += sum(instance_categories.view(-1, 1) == top3_preds).sum().item()

        total_instances += instance_categories.size(0)

    avg_loss = torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    accuracy = (correct_predictions / total_instances) if total_instances > 0 else 0
    top3_accuracy = (correct_top3_predictions / total_instances) if total_instances > 0 else 0

    return {
        "cls_pred_loss": avg_loss * cfg.model.token_cls_loss_weight,
        "metric_cls_pred_acc": torch.tensor(accuracy, device=device),
        "metric_cls_pred_top3_acc": torch.tensor(top3_accuracy, device=device),
    }

def get_relative_rot_data(cfg: BaseConfig, cond: ConditioningData, batch: InputData):
    """
    Returns a dict with keys as each batch index [referred to as the group index in this function].
    The values are themselves a dict containing the following:
    - Keys: The instance index for objects in the batch [group of frames]
    - Values: A list of tuples containing the batch index and the mask token index for the instance
    """
    group_size = cfg.model.predict_rotation_from_n_frames
    assert group_size == 2, "Only 2 frames are supported for now"
    bs = batch.tgt_pixel_values.shape[0]
    num_groups = bs // group_size
    all_rot_data = {}
    for group_idx in range(num_groups):
        # Defines the range of frames for a particular batch element
        group_batch_start = group_idx * group_size
        group_batch_end = group_batch_start + group_size

        batch_instance_data = defaultdict(list)
        for batch_idx in range(group_batch_start, group_batch_end):
            mask_tokens_for_frame = cond.mask_batch_idx == batch_idx
            instance_indices = cond.mask_instance_idx[mask_tokens_for_frame]
            mask_token_indices = torch.arange(cond.mask_instance_idx.shape[0], device=mask_tokens_for_frame.device)[mask_tokens_for_frame]
            for instance_idx, token_idx in zip(instance_indices, mask_token_indices):
                batch_instance_data[instance_idx.item()].append((batch_idx, token_idx.item()))

        all_rot_data[group_idx] = batch_instance_data

    return all_rot_data


def get_gt_rot(cfg: BaseConfig, cond: ConditioningData, batch: InputData, pred_data: TokenPredData):
    bs = batch.tgt_pixel_values.shape[0]
    device = batch.tgt_pixel_values.device
    gt_quat = batch.quaternions
    
    
    if cfg.model.predict_rotation_from_n_frames:
        gt_quat = gt_quat.clone()
        """
        We have input data with the first dim (num_groups group_size).
        num_groups is what we typically refer to as the batch_size and group_size is the number of frames we are using to predict rotation. [nominally 2]

        """
        group_size = cfg.model.predict_rotation_from_n_frames
        relative_rot_token_input_mask = []
        gt_rot_6d = []
        all_rot_data = get_relative_rot_data(cfg, cond, batch)
        for group_idx, group_instance_data in all_rot_data.items():
            for instance_idx, instance_rot_data in group_instance_data.items():
                batch_indices, token_indices = zip(*instance_rot_data)
                if len(instance_rot_data) != group_size or instance_idx == 0:
                    continue

                # Note that we take the difference between the rotations of the two objects. These rotations are relative to the canonical orientation relative to the camera [provided by the dataset]. 
                # Thus, with static objects, this is equivalent to simply taking the difference between the two camera rotations. However, this is more general and can handle moving objects.
                first_rot = gt_quat[batch_indices[0]][instance_idx - 1]
                second_rot = gt_quat[batch_indices[1]][instance_idx - 1]
                delta_rot = (R.from_quat(second_rot.cpu().numpy()) * R.from_quat(first_rot.cpu().numpy()).inv())

                gt_rot_6d.append(torch.from_numpy(delta_rot.as_matrix()[None]).to(device))
                relative_rot_token_input_mask.extend(token_indices)

        pred_data.gt_rot_6d = get_ortho6d_from_rotation_matrix(torch.cat(gt_rot_6d, dim=0))
        pred_data.mask_tokens = cond.mask_head_tokens[[*relative_rot_token_input_mask]]
        pred_data.token_output_mask = cond.mask_head_tokens.new_zeros((cond.mask_head_tokens.shape[0]), dtype=torch.bool)
        pred_data.token_output_mask[[*relative_rot_token_input_mask]] = True
    else:
        gt_rot_6d = []
        for group_idx in range(bs):
            gt_quat_ = gt_quat[group_idx][cond.mask_dropout[group_idx, 1:]].cpu()
            if gt_quat_.shape[0] == 0:
                gt_rot_6d_ = torch.zeros(0, 6, device=device)
            else:
                gt_rot_mat = R.from_quat(gt_quat_).as_matrix()
                gt_rot_6d_ = get_ortho6d_from_rotation_matrix(torch.from_numpy(gt_rot_mat).to(device))

            if (cond.mask_instance_idx[cond.mask_batch_idx == group_idx] == 0).any():
                # These are immediately removed by the mask below
                gt_rot_6d_ = torch.cat([gt_rot_6d_.new_zeros(1, 6), gt_rot_6d_], dim=0)
            
            gt_rot_6d.append(gt_rot_6d_)

        pred_data.token_output_mask = (cond.mask_instance_idx != 0)
        pred_data.mask_tokens = cond.mask_head_tokens[pred_data.token_output_mask]
        cond.mask_head_tokens = None
        pred_data.gt_rot_6d = torch.cat(gt_rot_6d, dim=0)[pred_data.token_output_mask]

    return pred_data


def token_rot_loss(cfg: BaseConfig, pred_data: TokenPredData, is_training: bool = False):
    assert cfg.model.background_mask_idx == 0

    ret = {}
    device = pred_data.pred_6d_rot.device
    pred_data.raw_pred_rot_logits = pred_data.pred_6d_rot.clone().detach() # For inference we save this for debugging
    
    if pred_data.pred_6d_rot.shape[0] == 0:
        ret['rot_pred_6d_l1_loss'] = torch.tensor(0.0, requires_grad=True, device=device)
    else:
        if cfg.model.discretize_rot_pred:
            num_bins = cfg.model.discretize_rot_bins_per_axis
            pred_zyx_quant, pred_zyx_residual = rearrange("all_masks axes (bin_logits + bin_residuals) -> all_masks axes bin_logits, all_masks axes bin_residuals", pred_data.pred_6d_rot, bin_logits=num_bins)
            _, quantized_rot_pred = pred_zyx_quant.max(dim=-1)
            pred_zyx_residual = torch.gather(pred_zyx_residual, 2, quantized_rot_pred.unsqueeze(-1)).squeeze(-1)

            # We need everything in (b, axes) for pred_data. Note that b in this context is actually (bs masks)
            pred_quat = get_quat_from_discretized_zyx(quantized_rot_pred + pred_zyx_residual, num_bins=num_bins)
            pred_data.pred_6d_rot = get_ortho6d_from_rotation_matrix(R.from_quat(pred_quat).as_matrix()).to(device)

            combine = lambda *y: [einops.rearrange(x, "masks axes ... -> (masks axes) ... ") for x in y]
            pred_zyx_residual, pred_zyx_quant, quantized_rot_pred = combine(pred_zyx_residual, pred_zyx_quant, quantized_rot_pred)

            gt_quat = torch.from_numpy(R.from_matrix(compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d).float().cpu().numpy()).as_quat()).to(device)
            gt_zyx_quant, gt_zyx_unquant = get_discretized_zyx_from_quat(gt_quat, num_bins=num_bins, return_unquantized=True)
            gt_zyx_quant, gt_zyx_unquant = einops.rearrange(gt_zyx_quant, "masks axes -> (masks axes)"), einops.rearrange(gt_zyx_unquant, "masks axes -> (masks axes)")
            zyx_residual = gt_zyx_unquant - gt_zyx_quant
            assert (-0.5 <= zyx_residual.min().item() <= zyx_residual.max().item() <= 0.5)

            ret['rot_pred_bin_ce_loss'] = F.cross_entropy(pred_zyx_quant, gt_zyx_quant)
            ret['rot_pred_residual_mse_loss'] = F.mse_loss(pred_zyx_residual, zyx_residual, reduction="mean")

            # We require that we correctly predict over *each* axis.
            correct_predictions = einops.rearrange(quantized_rot_pred == gt_zyx_quant, '(masks axes) -> masks axes', axes=3).all(dim=-1).sum().item()

            _, topk_quantized_rot_pred = pred_zyx_quant.topk(k=3, dim=1)
            correct_top3_predictions = einops.rearrange(gt_zyx_quant.view(-1, 1) == topk_quantized_rot_pred, '(masks axes) k -> masks axes k', axes=3).any(dim=-1).all(dim=-1).sum().item()

            total_instances = pred_data.pred_6d_rot.shape[0]

            accuracy = (correct_predictions / total_instances) if total_instances > 0 else 0
            top3_accuracy = (correct_top3_predictions / total_instances) if total_instances > 0 else 0
            ret.update({
                "metric_rot_pred_bin_acc": torch.tensor(accuracy, device=device),
                "metric_rot_pred_bin_top3_acc": torch.tensor(top3_accuracy, device=device),
                "metric_rot_pred_residual_distribution": wandb.Histogram(pred_zyx_residual.detach().float().cpu())
            })

        elif cfg.model.rotation_diffusion_parameterization == "sample":
            ret['rot_pred_6d_l1_loss'] = F.l1_loss(pred_data.pred_6d_rot, pred_data.gt_rot_6d, reduction="mean")
        elif cfg.model.rotation_diffusion_parameterization == "epsilon":
            ret['rot_pred_6d_l1_loss'] = F.l1_loss(pred_data.pred_6d_rot, pred_data.rot_6d_noise, reduction="mean")
        else:
            raise NotImplementedError

        # If training & denoising & using epsilon parameterization, we have a fake quat_l1_error
        gt_data = pred_data.rot_6d_noise if is_training and (not cfg.model.discretize_rot_pred) and cfg.model.rotation_diffusion_parameterization == "epsilon" else pred_data.gt_rot_6d
        pred_loss = quat_l1_loss(gt_data, pred_data.pred_6d_rot) # Not actually used as a loss, only an evaluation metric
        ret["metric_rot_pred_quat_l1_error"] = pred_loss.mean().float().cpu()
        ret["metric_rot_pred_quat_acc<0.025"] = (pred_loss < 0.025).float().cpu().sum() / pred_loss.shape[0]
        ret["metric_rot_pred_quat_acc<0.05"] = (pred_loss < 0.05).float().cpu().sum() / pred_loss.shape[0]

    return ret


def src_tgt_token_consistency_loss(
    cfg: BaseConfig,
    batch: InputData,
    cond: ConditioningData,
):
    losses = []
    for b in range(batch.bs):
        if batch.has_global_instance_ids[b].item() is False: continue
        src_valid = cond.mask_batch_idx == b
        tgt_valid = cond.tgt_mask_batch_idx == b

        src_loss_instance_idx = cond.mask_instance_idx[src_valid]
        tgt_loss_instance_idx = cond.tgt_mask_instance_idx[tgt_valid]
        shared_instance_ids, src_idx, tgt_idx = np.intersect1d(src_loss_instance_idx.cpu().numpy(), tgt_loss_instance_idx.cpu().numpy(), return_indices=True)
        src_mask_tokens = cond.src_mask_tokens[src_valid][src_idx]
        tgt_mask_tokens = cond.tgt_mask_tokens[tgt_valid][tgt_idx]
        loss = F.mse_loss(src_mask_tokens, tgt_mask_tokens, reduction="mean")
        losses.append(loss)
        
    avg_loss = torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=batch.device, requires_grad=True)
    return {
        "src_tgt_consistency_loss": avg_loss * cfg.model.src_tgt_consistency_loss_weight,
    }