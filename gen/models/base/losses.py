from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch

from gen.models.dustr.geometry import geotrf, inv

if TYPE_CHECKING:
    from gen.models.base.base_model import BaseMapper, ConditioningData, InputData


def transform_coordinate_space(batch: InputData, src_xyz, tgt_xyz):
    in_camera1 = inv(batch.src_extrinsics)
    src_xyz = geotrf(in_camera1, src_xyz)
    tgt_xyz = geotrf(in_camera1, tgt_xyz)
    return src_xyz, tgt_xyz

def forward_dustr_criterion(batch: InputData, left_xyz, right_xyz, left_mask, right_mask):
    from dust3r.losses import L21, Regr3D_ScaleShiftInv
    criterion = Regr3D_ScaleShiftInv(L21, gt_scale=True)
    
    gt1 = dict(pts3d=batch.src_xyz, camera_pose=batch.src_extrinsics, valid_mask=left_mask)
    gt2 = dict(pts3d=batch.tgt_xyz, camera_pose=batch.tgt_extrinsics, valid_mask=right_mask)
    pred1 = dict(pts3d=left_xyz)
    pred2 = dict(pts3d_in_other_view=right_xyz)

    loss, (left_loss, right_loss) = criterion(gt1, gt2, pred1, pred2)
    return loss

def get_dustr_loss(batch, pred_xyz, pred_valid, transform_coordinates: bool = False):
    pred_src_xyz, pred_tgt_xyz = torch.chunk(pred_xyz, 2, dim=0)
    pred_src_valid, pred_tgt_valid = torch.chunk(pred_valid, 2, dim=0)

    if transform_coordinates:
        pred_src_xyz, pred_tgt_xyz = transform_coordinate_space(batch, pred_src_xyz, pred_tgt_xyz)

    return forward_dustr_criterion(batch, pred_src_xyz, pred_tgt_xyz, pred_src_valid, pred_tgt_valid)