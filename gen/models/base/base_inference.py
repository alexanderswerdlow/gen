from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einx import mean, rearrange, softmax

from gen.datasets.abstract_dataset import Split
from gen.datasets.utils import get_stable_diffusion_transforms
from gen.models.base.losses import get_dustr_loss, transform_coordinate_space
from gen.models.dustr.depth_utils import encode_xyz, decode_xyz, get_xyz_input, xyz_to_depth
from gen.models.dustr.marigold import NearFarMetricNormalizer
from gen.utils.data_defs import undo_normalization_given_transforms
from gen.utils.trainer_utils import TrainingState
from image_utils import Im

if TYPE_CHECKING:
    from gen.models.base.base_model import BaseMapper, ConditioningData, InputData
    from gen.configs.base import BaseConfig

@torch.no_grad()
def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    cond: Optional[ConditioningData] = None,
    allow_get_cond: bool = True,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if cond is None or len(cond.unet_kwargs) == 0 and allow_get_cond:
        cond = self.get_hidden_state(batch, cond)
        _repeat_dim = batch.n if self.cfg.model.duplicate_unet_input_channels else 1
        cond.encoder_hidden_states = cond.encoder_hidden_states.repeat(_repeat_dim, 1, 1)
        cond.unet_kwargs["prompt_embeds"] = cond.encoder_hidden_states

    if "guidance_scale" not in cond.unet_kwargs and "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = self.cfg.inference.guidance_scale

    if "num_images_per_prompt" not in cond.unet_kwargs:
        kwargs["num_images_per_prompt"] = num_images_per_prompt

    kwargs["height"] = self.cfg.model.decoder_resolution
    kwargs["width"] = self.cfg.model.decoder_resolution
    if self.cfg.trainer.profile_memory or self.cfg.trainer.fast_eval:
        kwargs["num_inference_steps"] = 1 # Required otherwise the profiler will create traces that are too large

    needs_autocast = self.cfg.model.freeze_unet is False or self.cfg.model.unfreeze_single_unet_layer or self.cfg.model.dual_attention or self.cfg.model.duplicate_unet_input_channels
    with torch.cuda.amp.autocast(dtype=self.dtype) if needs_autocast else nullcontext():
        images = self.pipeline(**cond.unet_kwargs, **kwargs).images

    return images, cond

def jointnorm(*args):
    _min, _max = min([x.min() for x in args]), max([x.max() for x in args])
    ret = tuple((x - _min) / (_max - _min) for x in args)
    if len(ret) == 1:
        return ret[0]
    return ret

def multinorm(*args):
    _min, _max = args[0].min(), args[0].max()
    return ((x - _min) / (_max - _min) for x in args)

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def norm_batch(x):
    orig_shape = x.shape
    x = rearrange('b ... -> b (...)', x)
    min_vals = x.min(dim=1, keepdim=True)[0]
    max_vals = x.max(dim=1, keepdim=True)[0]
    x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    return x.view(orig_shape)

def get_valid_mse(arr1, arr2, mask, norm_data: bool = False):
    if norm_data:
        arr1, arr2 = multinorm(arr1, arr2)

    _gt_xyz = rearrange('b h w ... -> (b h w) ...', arr1)
    _pred_xyz = rearrange('b h w ... -> (b h w) ...', arr2)
    _pred_mask = rearrange('b h w -> (b h w)', mask)
    return F.mse_loss(_pred_xyz[_pred_mask], _gt_xyz[_pred_mask], reduction="mean")

def get_depth(cfg, _gt_depth, _gt, _pred, _intrinsics, _extrinsics, b, joint_norm: bool = False):
    if cfg.model.predict_depth:
        _gt_depth = _gt_depth[b]
        _pred_depth = _pred[b].mean(dim=-1)
    else:
        _gt_depth = torch.from_numpy(xyz_to_depth(_gt[b], _intrinsics[b], _extrinsics[b], simple=True)).to(_gt)
        _pred_depth = torch.from_numpy(xyz_to_depth(_pred[b], _intrinsics[b], _extrinsics[b], simple=True)).to(_pred)

    if joint_norm:
        return multinorm(_gt_depth, _pred_depth)
    else:
        return norm(_gt_depth), norm(_pred_depth)

def get_valid(arr, mask):
    _arr = arr.clone()
    _arr[~mask] = 0
    return _arr

# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()

def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)

def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output + 1
    actual_target = target + 1
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    gt = gt.float().cpu().numpy()
    pred = pred.float().cpu().numpy()
    pred_arr = pred_arr.float().cpu().numpy()
    valid_mask = valid_mask.bool().cpu().numpy()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def get_metrics(batch, cfg, prefix, pred, gt, valid_mask):
    ret = get_metrics_(batch, cfg, prefix, pred, gt, valid_mask)
    ret.update(get_metrics_(batch, cfg, prefix, pred, gt, valid_mask, single_view=True))
    return ret

def get_metrics_(batch, cfg, prefix, pred, gt, valid_mask, single_view=False):
    pred = pred.squeeze(-1)
    gt = gt.squeeze(-1)
    valid_mask = valid_mask.squeeze(-1)

    if cfg.model.predict_depth:
        depth_pred = []
        if single_view is False:
            gt = rearrange('(n b) h w -> b (n h) w', gt, b=batch.n)
            pred = rearrange('(n b) h w -> b (n h) w', pred, b=batch.n)
            valid_mask = rearrange('(n b) h w -> b (n h) w', valid_mask, b=batch.n)
        
        for b in range(gt.shape[0]):
            _depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt[b],
                pred_arr=pred[b],
                valid_mask_arr=valid_mask[b],
                return_scale_shift=True,
            )
            _depth_pred = torch.from_numpy(_depth_pred).to(pred.device)
            depth_pred.append(_depth_pred)

        pred = torch.stack(depth_pred)

    if single_view:
        prefix = f"single_view/{prefix}"

    # ret = {
    #     f'{prefix}_mse': get_valid_mse(pred, gt, valid_mask),
    # }

    ret = dict()

    if cfg.model.predict_depth:
        ret.update({
            f'{prefix}_delta1_acc': delta1_acc(pred, gt, valid_mask),
            f'{prefix}_abs_rel_diff': abs_relative_difference(pred, gt, valid_mask),
        })

    return ret


@torch.no_grad()
def run_qualitative_inference(self: BaseMapper, batch: InputData, state: TrainingState, accelerator: Optional[Any] = None) -> dict:
    ret = {}

    input_xyz, input_valid = get_xyz_input(self.cfg, batch)
    xyz_latents, xyz_valid, normalizer = encode_xyz(self.cfg, input_xyz, input_valid, self.vae, self.dtype, batch.n, kwargs=dict(per_axis_quantile=self.cfg.model.predict_depth is False))

    autoencoded_xyz = autoencode_gt_xyz(self.cfg, batch, self.vae, xyz_latents, normalizer)
    if self.cfg.model.predict_depth:
        input_xyz = input_xyz[..., [0]]
    else:
        ret['wandb_metric_autoencode_l2_scale_shift_inv'] = get_dustr_loss(batch, autoencoded_xyz, xyz_valid)

    ret.update(get_metrics(batch, self.cfg, 'wandb_metric_autoencode', autoencoded_xyz, input_xyz, xyz_valid))

    if self.cfg.model.predict_depth is False:
        for i in range(3):
            ret[f'wandb_metric_autoencode_valid_xyz_mse_channel_{i}'] = get_valid_mse(input_xyz[..., [i]], autoencoded_xyz[..., [i]], xyz_valid)

    if self.cfg.model.unet is False:
        return ret

    rgb_latents = self.get_rgb_latents(batch)

    pipeline_kwargs = dict()
    if self.cfg.model.only_noise_tgt:
        assert self.cfg.model.n_view_pred is False
        pipeline_kwargs['concat_src_depth'] = xyz_latents[:xyz_latents.shape[0] // 2]

    if self.cfg.model.batched_denoise and self.cfg.model.num_training_views != batch.n:
        pipeline_kwargs['batched_denoise'] = batch.n

    pred_latents, cond = self.infer_batch(batch, concat_rgb=rgb_latents, num_images_per_prompt=1, output_type='latent', **pipeline_kwargs)
    pred_xyz = decode_xyz(self.cfg, pred_latents, self.vae, normalizer)

    ret.update(get_metrics(batch, self.cfg, 'wandb_metric_pred', pred_xyz, input_xyz, xyz_valid))

    if self.cfg.model.predict_depth is False:
        ret['wandb_metric_l2_scale_shift_inv'] = get_dustr_loss(batch, pred_xyz, xyz_valid)
        for i in range(3):
            ret[f'wandb_metric_valid_xyz_mse_channel_{i}'] = get_valid_mse(input_xyz[..., [i]], pred_xyz[..., [i]], xyz_valid)

    pred_marigold = True
    marigold_depth_pred = None
    if self.cfg.model.predict_depth and pred_marigold:
        dtype = torch.float16
        variant = "fp16"
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            "prs-eth/marigold-v1-0",
            custom_pipeline="marigold_depth_estimation",
            torch_dtype=dtype,
            variant=variant,
        )

        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(self.device)
        resample_method = 'bilinear'

        rgb_to_encode = self.get_rgb_input(batch)
        rgb_to_encode = (rgb_to_encode + 1) / 2
        rgb_to_encode = rearrange('b c h w -> b h w c', rgb_to_encode * 255).to(dtype=torch.uint8).cpu().numpy()
        from PIL import Image

        marigold_depth_pred = []
        for b in range(rgb_to_encode.shape[0]):
            input_image = Image.fromarray(rgb_to_encode[b])
            pipe_out = pipe(
                input_image,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
                ensemble_size=1,
            )
            marigold_depth_pred.append(pipe_out.depth_np)

        marigold_depth_pred = np.stack(marigold_depth_pred)
        marigold_depth_pred = torch.from_numpy(marigold_depth_pred).to(self.device)[..., None]
        ret.update(get_metrics(batch, self.cfg, 'wandb_metric_marigold', marigold_depth_pred, input_xyz, xyz_valid))

    orig_rgb = undo_normalization_given_transforms(get_stable_diffusion_transforms(resolution=self.cfg.model.decoder_resolution), batch.dec_rgb)

    imgs = []
    secondary_viz = []

    n = 1
    to_viz_indices = np.random.choice(list(range(batch.bs)), n, replace=False)
    input_xyz = rearrange('(n b) ... -> b n ...', input_xyz, n=batch.n)
    pred_xyz = rearrange('(n b) ... -> b n ...', pred_xyz, n=batch.n)
    autoencoded_xyz = rearrange('(n b) ... -> b n ...', autoencoded_xyz, n=batch.n)
    marigold_depth_pred = rearrange('(n b) ... -> b n ...', marigold_depth_pred, n=batch.n) if marigold_depth_pred is not None else None
    xyz_valid = rearrange('(n b) ... -> b n ...', xyz_valid, n=batch.n)

    for b in to_viz_indices:
        _gt_depth, _pred_depth = get_depth(self.cfg, batch.dec_depth, batch.xyz, pred_xyz, batch.intrinsics, batch.extrinsics, b)
        _autoencoded_depth = get_depth(self.cfg, batch.dec_depth, batch.xyz, autoencoded_xyz, batch.intrinsics, batch.extrinsics, b)[1]

        if marigold_depth_pred is not None:
            _marigold_depth_pred = get_depth(self.cfg, batch.dec_depth, batch.xyz, marigold_depth_pred, batch.intrinsics, batch.extrinsics, b)[1]
            
        _gt_orig_depth = batch.dec_depth[b]

        _func = lambda x: rearrange('h w -> () h w 3', x.to(dtype=torch.float32))
        _post = 'Depth' if self.cfg.model.predict_depth else 'PCD'

        norm_gt = jointnorm(_gt_orig_depth)
        
        imgs.append(
            Im.concat_horizontal(
                Im.concat_vertical(
                    orig_rgb[[b], view_idx],
                    Im(_func(norm_gt[view_idx])).bool_to_rgb().write_text(f"GT {_post}", size=0.6),
                    Im(_func(_pred_depth[view_idx])).bool_to_rgb().write_text(f"Pred {_post}", size=0.6),
                    *((Im(_func(_marigold_depth_pred[view_idx])).bool_to_rgb().write_text(f"Marigold {_post}", size=0.6),) if marigold_depth_pred is not None else ()),
                    Im(_func(_autoencoded_depth[view_idx])).bool_to_rgb().write_text(f"Autoencoded {_post}", size=0.6),
                    Im(_func(get_valid(_pred_depth[view_idx], xyz_valid[b, view_idx]))).bool_to_rgb().write_text(f"Pred {_post} Valid", size=0.6),
                    Im(xyz_valid[b, view_idx]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
                ) for view_idx in range(batch.n)
            )
        )
        # secondary_viz.append(
        #     Im.concat_horizontal(
        #         Im.concat_vertical(
        #             Im(_func(src_gt_orig_depth)).bool_to_rgb().write_text("GT Depth", size=0.6),
        #             Im(_func(get_valid(src_gt_orig_depth, input_src_valid[b]))).bool_to_rgb().write_text("GT Depth Dataset Valid", size=0.6),
        #             Im(_func(get_valid(src_gt_orig_depth, src_valid[b]))).bool_to_rgb().write_text("GT Depth Truncated Valid", size=0.6),
        #             Im(input_src_valid[b]).bool_to_rgb().write_text("Dataset Valid Mask", size=0.6),
        #             Im(src_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
        #         ),
        #         Im.concat_vertical(
        #             Im(_func(tgt_gt_orig_depth)).bool_to_rgb().write_text("GT Depth", size=0.6),
        #             Im(_func(get_valid(tgt_gt_orig_depth, input_tgt_valid[b]))).bool_to_rgb().write_text("GT Depth Dataset Valid", size=0.6),
        #             Im(_func(get_valid(tgt_gt_orig_depth, tgt_valid[b]))).bool_to_rgb().write_text("GT Depth Truncated Valid", size=0.6),
        #             Im(input_tgt_valid[b]).bool_to_rgb().write_text("Dataset Valid Mask", size=0.6),
        #             Im(tgt_valid[b]).bool_to_rgb().write_text("Truncated Valid Mask", size=0.6),
        #         ),
        #     )
        # ) 

    ret['imgs'] = Im.concat_horizontal(*imgs)
    return ret
                    
def autoencode_gt_xyz(cfg: BaseConfig, batch: InputData, vae, xyz_latents, normalizer):
    pred_xyz = decode_xyz(cfg, xyz_latents, vae, normalizer)
    return pred_xyz