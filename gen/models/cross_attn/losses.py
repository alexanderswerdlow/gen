from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einx import rearrange
from image_utils import Im
from PIL import Image
from scipy.spatial.transform import Rotation as R
import math

from gen.models.cross_attn.break_a_scene import AttentionStore, aggregate_attention
from gen.utils.rotation_utils import get_discretized_zyx_from_quat, get_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import ConditioningData, InputData
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import TokenPredData


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


def get_gt_rot(cfg: BaseConfig, cond: ConditioningData, batch: InputData, pred_data: TokenPredData):
    bs = batch["gen_pixel_values"].shape[0]

    gt_rot_6d = []
    for b in range(bs):
        # We only compute loss on non-dropped out masks
        gt_rot_mat = R.from_quat(batch["quaternions"][b][batch["valid"][b]].cpu()).as_matrix()
        gt_rot_6d_ = get_ortho6d_from_rotation_matrix(torch.from_numpy(gt_rot_mat).to(batch["disc_pixel_values"]))
        batch_instance_idxs = cond.mask_instance_idx[cond.mask_batch_idx == b]
        foreground_mask = batch_instance_idxs != 0
        batch_object_idxs = batch_instance_idxs - 1 # Object data does not include the background
        gt_rot_6d_ = gt_rot_6d_[batch_object_idxs[foreground_mask]]
        if torch.any(~foreground_mask):
            gt_rot_6d_ = torch.cat([gt_rot_6d_.new_zeros(1, 6), gt_rot_6d_], dim=0)
        gt_rot_6d.append(gt_rot_6d_)

    pred_data.gt_rot_6d = torch.cat(gt_rot_6d, dim=0)
    pred_data.pred_mask = cond.mask_instance_idx != 0

    return pred_data


def token_rot_loss(cfg: BaseConfig, batch: InputData, cond: ConditioningData, pred_data: TokenPredData):
    assert cfg.model.background_mask_idx == 0

    additional_metrics = {}
    
    if pred_data.pred_mask.sum() == 0:
        loss = torch.tensor(0.0, requires_grad=True, device=batch["device"])
    else:
        if cfg.model.discretize_rot_pred:
            pred_zyz = rearrange("b axes d -> (b axes) d", pred_data.pred_6d_rot[pred_data.pred_mask])
            device = pred_zyz.device

            num_bins = cfg.model.discretize_rot_bins_per_axis
            gt_quat = torch.from_numpy(R.from_matrix(compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d[pred_data.pred_mask]).float().cpu().numpy()).as_quat()).to(device)
            discretized_zyx = rearrange("b axes -> (b axes)", get_discretized_zyx_from_quat(gt_quat, num_bins=num_bins))

            loss = F.cross_entropy(pred_zyz, discretized_zyx)

            _, predicted_labels = pred_zyz.max(dim=1)
            correct_predictions = (predicted_labels == discretized_zyx).sum().item()

            _, top5_preds = pred_zyz.topk(k=2, dim=1)
            correct_top5_predictions = sum(discretized_zyx.view(-1, 1) == top5_preds).sum().item()

            total_instances = discretized_zyx.size(0)

            accuracy = (correct_predictions / total_instances) if total_instances > 0 else 0
            top2_accuracy = (correct_top5_predictions / total_instances) if total_instances > 0 else 0

            additional_metrics.update({
                "metric_rot_pred_acc": torch.tensor(accuracy, device=device),
                "metric_rot_pred_top2_acc": torch.tensor(top2_accuracy, device=device),
            })

        elif cfg.model.rotation_diffusion_parameterization == "sample":
            loss = F.l1_loss(pred_data.pred_6d_rot[pred_data.pred_mask], pred_data.gt_rot_6d[pred_data.pred_mask], reduction="mean")
        elif cfg.model.rotation_diffusion_parameterization == "epsilon":
            loss = F.l1_loss(pred_data.pred_6d_rot[pred_data.pred_mask], pred_data.rot_6d_noise[pred_data.pred_mask], reduction="mean")
        else:
            raise NotImplementedError

    return {"rot_pred_loss": loss, **additional_metrics}
