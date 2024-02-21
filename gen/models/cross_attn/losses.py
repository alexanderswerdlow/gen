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
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.pytorch3d_transforms import matrix_to_quaternion
from gen.utils.rotation_utils import (compute_rotation_matrix_from_ortho6d, get_discretized_zyx_from_quat, get_ortho6d_from_rotation_matrix,
                                      get_quat_from_discretized_zyx)
from image_utils import Im
import einops

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import ConditioningData, InputData, TokenPredData


def break_a_scene_cross_attn_loss(cfg: BaseConfig, batch: InputData, controller: AttentionStore, cond: ConditioningData):
    attn_loss = 0
    batch_size: int = batch["disc_pixel_values"].shape[0]
    gen_seg_ = rearrange(batch["gen_segmentation"], "b h w c -> b c () h w").float()
    learnable_idxs = (batch["formatted_input_ids"] == cond.placeholder_token).nonzero(as_tuple=True)

    for batch_idx in range(batch_size):
        GT_masks = F.interpolate(input=gen_seg_[batch_idx], size=(16, 16))  # We interpolate per mask separately
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
    cfg: BaseConfig,
    batch: InputData,
    cond: ConditioningData,
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred = rearrange(pred, "b c h w -> b c (h w)")
    target = rearrange(target, "b c h w -> b c (h w)")

    losses = []

    for b in range(batch["gen_pixel_values"].shape[0]):
        if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
            losses.append(F.mse_loss(pred[b], target[b], reduction="mean"))
            continue

        mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
        object_masks = batch["gen_segmentation"][b, ..., mask_idxs_for_batch]

        gt_masks = F.interpolate(rearrange(object_masks, "h w c -> c () h w").float(), size=(cfg.model.latent_dim, cfg.model.latent_dim)).squeeze(1)
        gt_masks = rearrange(gt_masks, "c h w -> c (h w)") > 0.5

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


def break_a_scene_masked_loss(cfg: BaseConfig, batch: InputData, cond: ConditioningData):
    max_masks = []
    for b in range(batch["gen_pixel_values"].shape[0]):
        mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
        object_masks = batch["gen_segmentation"][b, ..., mask_idxs_for_batch]
        if (
            cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item()
        ):  # We do not have conditioning for this entire sample so put loss on everything
            max_masks.append(object_masks.new_ones((cfg.model.resolution, cfg.model.resolution)))
        elif object_masks.shape[-1] == 0:
            max_masks.append(object_masks.new_zeros((cfg.model.resolution, cfg.model.resolution)))  # Zero out loss if there are no masks
        else:
            max_masks.append(torch.max(object_masks, axis=-1).values)

    max_masks = torch.stack(max_masks, dim=0)[:, None]
    loss_mask = F.interpolate(input=max_masks.float(), size=(cfg.model.latent_dim, cfg.model.latent_dim))

    if cfg.model.viz and batch["state"].true_step % 1 == 0:
        from image_utils import Im

        rgb_ = Im((batch["gen_pixel_values"] + 1) / 2)
        mask_ = Im(loss_mask).resize(rgb_.height, rgb_.width)
        Im.concat_horizontal(rgb_.grid(), mask_.grid()).save(f'rgb_{batch["state"].true_step}.png')

    return loss_mask


def token_cls_loss(
    cfg: BaseConfig,
    batch: InputData,
    cond: ConditioningData,
    pred_data: TokenPredData,
):

    cls_pred = pred_data.cls_pred
    bs = batch["gen_pixel_values"].shape[0]
    device = batch["gen_pixel_values"].device

    assert cfg.model.background_mask_idx == 0

    losses = []
    correct_predictions = 0
    correct_top5_predictions = 0
    total_instances = 0

    for b in range(bs):
        if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
            continue

        # We only compute loss on non-dropped out masks
        instance_categories = batch["categories"][b][batch["valid"][b]]

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

        _, top5_preds = pred_.topk(k=5, dim=1)
        correct_top5_predictions += sum(instance_categories.view(-1, 1) == top5_preds).sum().item()

        total_instances += instance_categories.size(0)

    avg_loss = torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=device)
    accuracy = (correct_predictions / total_instances) if total_instances > 0 else 0
    top5_accuracy = (correct_top5_predictions / total_instances) if total_instances > 0 else 0

    return {
        "cls_pred_loss": avg_loss * 0.1,
        "metric_cls_pred_acc": torch.tensor(accuracy, device=device),
        "metric_cls_pred_top5_acc": torch.tensor(top5_accuracy, device=device),
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
    bs = batch["gen_pixel_values"].shape[0]
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
            for instance, idx in zip(instance_indices, mask_token_indices):
                batch_instance_data[instance.item()].append((batch_idx, idx.item()))

        all_rot_data[group_idx] = batch_instance_data

    return all_rot_data


def get_gt_rot(cfg: BaseConfig, cond: ConditioningData, batch: InputData, pred_data: TokenPredData):
    bs = batch["gen_pixel_values"].shape[0]
    device = batch["gen_pixel_values"].device
    gt_quat = batch["quaternions"]
    pred_data.relative_rot_pred_mask = torch.ones_like(cond.mask_instance_idx, dtype=torch.bool)
    if cfg.model.predict_rotation_from_n_frames:
        gt_quat = gt_quat.clone()
        """
        We have input data with the first dim (num_groups group_size).
        num_groups is what we typically refer to as the batch_size and group_size is the number of frames we are using to predict rotation. [nominally 2]

        """
        all_rot_data = get_relative_rot_data(cfg, cond, batch)
        for group_idx, group_instance_data in all_rot_data.items():
            for instance_idx, instance_rot_data in group_instance_data.items():
                if instance_idx == 0:
                    continue
                    
                batch_indices, instance_indices = zip(*instance_rot_data)
                group_size = cfg.model.predict_rotation_from_n_frames
                if len(instance_rot_data) != group_size:
                    pred_data.relative_rot_pred_mask[instance_indices] = False # TODO: Find a better way to handle this
                    continue

                # Note that we take the difference between the rotations of the two objects. These rotations are relative to the canonical orientation relative to the camera [provided by the dataset]. Thus, with static objects, this is equivalent to simply taking the difference between the two camera rotations. However, this is more general and can handle moving objects.
                first_rot = gt_quat[batch_indices[0]][batch["valid"][batch_indices[0]]][instance_idx - 1]
                second_rot = gt_quat[batch_indices[1]][batch["valid"][batch_indices[1]]][instance_idx - 1]
                delta_rot = (R.from_quat(second_rot.cpu().numpy()) * R.from_quat(first_rot.cpu().numpy()).inv()).as_quat()
                gt_quat[batch_indices[0]][batch["valid"][batch_indices[0]]][instance_idx - 1] = torch.from_numpy(delta_rot).to(gt_quat)
                gt_quat[batch_indices[1]][batch["valid"][batch_indices[1]]][instance_idx - 1] = torch.from_numpy(delta_rot).to(gt_quat)
        
    gt_rot_6d = []
    for group_idx in range(bs):
        # We only compute loss on non-dropped out masks
        gt_rot_mat = R.from_quat(gt_quat[group_idx][batch["valid"][group_idx]].cpu()).as_matrix()
        gt_rot_6d_ = get_ortho6d_from_rotation_matrix(torch.from_numpy(gt_rot_mat).to(device))
        batch_instance_idxs = cond.mask_instance_idx[cond.mask_batch_idx == group_idx]
        foreground_mask = batch_instance_idxs != 0
        batch_object_idxs = batch_instance_idxs - 1 # Object data does not include the background
        gt_rot_6d_ = gt_rot_6d_[batch_object_idxs[foreground_mask]]
        if torch.any(~foreground_mask):
            gt_rot_6d_ = torch.cat([gt_rot_6d_.new_zeros(1, 6), gt_rot_6d_], dim=0)
        gt_rot_6d.append(gt_rot_6d_)

    pred_data.gt_rot_6d = torch.cat(gt_rot_6d, dim=0)
    pred_data.pred_mask = cond.mask_instance_idx != 0

    return pred_data


def quat_l1_loss(rot1, rot2):
    mat1 = compute_rotation_matrix_from_ortho6d(rot1)
    quat1 = matrix_to_quaternion(mat1)

    mat2 = compute_rotation_matrix_from_ortho6d(rot2)
    quat2 = matrix_to_quaternion(mat2)

    quat_l1 = (quat1 - quat2).abs().sum(-1)
    quat_l1_ = (quat1 + quat2).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_
    return quat_l1


def token_rot_loss(cfg: BaseConfig, batch: InputData, cond: ConditioningData, pred_data: TokenPredData):
    assert cfg.model.background_mask_idx == 0

    ret = {}
    device = pred_data.pred_mask.device
    pred_data.raw_pred_rot_logits = pred_data.pred_6d_rot.clone().detach() # For inference we save this for debugging
    
    if pred_data.pred_mask.sum() == 0:
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

            # We need everything in (b axes) for loss calculations.
            num_valid_masks = pred_data.pred_mask.sum().item()
            if pred_data.pred_mask[pred_data.relative_rot_pred_mask].sum().item() == 0 or pred_zyx_residual.shape[0] == 0 or pred_zyx_quant.shape[0] == 0 or quantized_rot_pred.shape[0] == 0:
                ret['rot_pred_bin_ce_loss'] = torch.tensor(0.0, requires_grad=True, device=device)
                log_warn(f"We have no mask tokens to predict, skipping this batch...", main_process_only=False)
                return ret

            combine = lambda *y: [rearrange("masks axes ... -> (masks axes) ... ", x[pred_data.pred_mask[pred_data.relative_rot_pred_mask]]) for x in y]
            pred_zyx_residual, pred_zyx_quant, quantized_rot_pred = combine(pred_zyx_residual, pred_zyx_quant, quantized_rot_pred)

            gt_quat = torch.from_numpy(R.from_matrix(compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d[pred_data.pred_mask & pred_data.relative_rot_pred_mask]).float().cpu().numpy()).as_quat()).to(device)
            gt_zyx_quant, gt_zyx_unquant = get_discretized_zyx_from_quat(gt_quat, num_bins=num_bins, return_unquantized=True)
            gt_zyx_quant, gt_zyx_unquant = rearrange("masks axes -> (masks axes)", gt_zyx_quant), rearrange("masks axes -> (masks axes)", gt_zyx_unquant)
            zyx_residual = gt_zyx_unquant - gt_zyx_quant
            assert (-0.5 <= zyx_residual.min().item() <= zyx_residual.max().item() <= 0.5)

            ret['rot_pred_bin_ce_loss'] = F.cross_entropy(pred_zyx_quant, gt_zyx_quant)
            ret['rot_pred_residual_mse_loss'] = F.mse_loss(pred_zyx_residual, zyx_residual, reduction="mean")

            # We require that we correctly predict over *each* axis.
            correct_predictions = rearrange('(masks axes) -> masks axes', quantized_rot_pred == gt_zyx_quant, axes=3).all(dim=-1).sum().item()

            _, topk_quantized_rot_pred = pred_zyx_quant.topk(k=3, dim=1)
            correct_top3_predictions = rearrange('(masks axes) k -> masks axes k', gt_zyx_quant.view(-1, 1) == topk_quantized_rot_pred, axes=3).any(dim=-1).all(dim=-1).sum().item()

            total_instances = num_valid_masks

            accuracy = (correct_predictions / total_instances) if total_instances > 0 else 0
            top3_accuracy = (correct_top3_predictions / total_instances) if total_instances > 0 else 0
            ret.update({
                "metric_rot_pred_bin_acc": torch.tensor(accuracy, device=device),
                "metric_rot_pred_bin_top3_acc": torch.tensor(top3_accuracy, device=device),
                "metric_rot_pred_residual_distribution": wandb.Histogram(pred_zyx_residual.detach().float().cpu())
            })

        elif cfg.model.rotation_diffusion_parameterization == "sample":
            ret['rot_pred_6d_l1_loss'] = F.l1_loss(pred_data.pred_6d_rot[pred_data.pred_mask], pred_data.gt_rot_6d[pred_data.pred_mask], reduction="mean")
        elif cfg.model.rotation_diffusion_parameterization == "epsilon":
            ret['rot_pred_6d_l1_loss'] = F.l1_loss(pred_data.pred_6d_rot[pred_data.pred_mask], pred_data.rot_6d_noise[pred_data.pred_mask], reduction="mean")
        else:
            raise NotImplementedError

        pred_loss = quat_l1_loss(pred_data.gt_rot_6d[pred_data.pred_mask & pred_data.relative_rot_pred_mask], pred_data.pred_6d_rot[pred_data.pred_mask[pred_data.relative_rot_pred_mask]])
        ret["metric_rot_pred_quat_l1_error"] = pred_loss.mean().float().cpu()
        ret["metric_rot_pred_quat_acc<0.025"] = (pred_loss < 0.025).float().cpu().sum() / pred_loss.shape[0]
        ret["metric_rot_pred_quat_acc<0.05"] = (pred_loss < 0.05).float().cpu().sum() / pred_loss.shape[0]

    return ret
