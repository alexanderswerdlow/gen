from __future__ import annotations

from contextlib import nullcontext
from math import sqrt
from pathlib import Path
import traceback
from typing import TYPE_CHECKING, Any, Optional

from image_utils.standalone_image_utils import integer_to_color
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
import einops
from einx import mean, rearrange, softmax
from image_utils import ChannelRange, Im, onehot_to_color
from PIL import Image
from torchvision import utils
from gen import GSO_PCD_PATH

from gen.datasets.abstract_dataset import Split
from gen.models.cross_attn.break_a_scene import aggregate_attention, save_cross_attention_vis
from gen.models.cross_attn.losses import break_a_scene_masked_loss, get_relative_rot_data, token_cls_loss, token_rot_loss
from gen.utils.data_defs import get_one_hot_channels, get_tgt_grid, integer_to_one_hot, undo_normalization_given_transforms, visualize_input_data
from gen.utils.decoupled_utils import load_tensor_dict, to_numpy
from gen.utils.logging_utils import log_debug, log_info, log_warn
from gen.utils.rotation_utils import compute_rotation_matrix_from_ortho6d, visualize_rotations, visualize_rotations_pcds
from gen.utils.tokenization_utils import get_tokens
from gen.utils.trainer_utils import TrainingState
from gen.datasets.coco.coco_panoptic import CocoPanoptic
from dataclasses import asdict, fields
from torchvision.transforms.functional import InterpolationMode, resize

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData

gso_pcds = None

def get_valid_idxs(batch: InputData):
    assert batch.src_valid.shape[0] == 1

    _valids = torch.Tensor(batch.src_valid[0].nonzero()[:, 0])
    unique, counts = torch.unique(batch.src_segmentation.long(), return_counts=True)
    unique_counts = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
    sorted_unique_counts = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)
    idxs = [integer for integer, count in sorted_unique_counts]
    idxs = filter(lambda x: x != 255, idxs)
    idxs = torch.tensor(list(idxs)).to(batch.device)
    idxs = idxs[torch.isin(idxs, _valids)].long()

    return idxs

def label_to_color_image(label, colormap=None):
    if label.ndim != 2: raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))
    if colormap is None: raise ValueError('Expect a valid colormap.')
    return colormap[label]

def repeat_batch(batch: InputData, bs: int):
    return batch.clone().expand(bs)

def overlay_mask(initial_img: Im, mask: torch.Tensor | np.ndarray, color: tuple[int, int, int], alpha: float = 0.5) -> Im:
    mask = to_numpy(mask)[..., None]
    mask_rgb = np.full((mask.shape[0], mask.shape[1], 3), color, dtype=np.uint8)
    mask_alpha = (mask * (255 * alpha)).astype(np.uint8)
    composited_image = initial_img.pil.copy().convert("RGBA")
    composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
    return Im(composited_image.convert("RGB"))

def get_composited_mask(batch: dict, b: int, j: int, alpha=0.5, mask=None, color=(255, 0, 0)):
    orig_image = Im(((batch.tgt_pixel_values + 1) / 2)[b])
    mask = batch.one_hot_tgt_segmentation[b, ..., [j]].squeeze() if mask is None else mask
    return overlay_mask(orig_image, mask, color, alpha)

def get_metrics(self: BaseMapper, batch: InputData, cond: ConditioningData):
    loss_dict = self.compute_losses(batch, cond)
    loss_dict = {k: (float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items() }
    if self.cfg.model.encode_src_twice or self.cfg.model.encode_tgt_enc_norm:
        loss_dict['hist_src_mask_tokens'] = cond.src_mask_tokens.detach().float().cpu()
        loss_dict['hist_tgt_mask_tokens'] = cond.tgt_mask_tokens.detach().float().cpu()

    loss_dict = {f"wandb_{k}": v for k, v in loss_dict.items()}

    return loss_dict

new_prompts = [
    "A photo of a {}",
    "A photo of {} in the jungle",
    "A photo of {} on a beach",
    "A photo of {} in Times Square",
    "A photo of {} in the moon",
    "A painting of {} in the style of Monet",
    "Oil painting of {}",
    "A Marc Chagall painting of {}",
    "A manga drawing of {}",
    "A watercolor painting of {}",
    "A statue of {}",
    "App icon of {}",
    "A sand sculpture of {}",
    "Colorful graffiti of {}",
    "A photograph of two {} on a table",
]

@torch.no_grad()
def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    cond: Optional[ConditioningData] = None,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if batch.input_ids is not None:
        orig_input_ids = batch.input_ids.clone()
        batch.input_ids = orig_input_ids.clone()

    if cond is None or len(cond.unet_kwargs) == 0:
        with torch.cuda.amp.autocast():
            assert self.cfg.inference.empty_string_cfg
            cond = self.get_standard_conditioning_for_inference(batch=batch, cond=cond)

    if "guidance_scale" not in cond.unet_kwargs and "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = self.cfg.inference.guidance_scale

    if "num_images_per_prompt" not in cond.unet_kwargs:
        kwargs["num_images_per_prompt"] = num_images_per_prompt

    if "return_attn_probs" in kwargs:
        cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].return_attn_probs = kwargs["return_attn_probs"]
        del kwargs["return_attn_probs"]

    # CFG must be enabled when masking as we make this assumption in attn_proc
    if self.cfg.model.attention_masking:
        assert kwargs["guidance_scale"] > 1

    attn_meta = cond.unet_kwargs.get('cross_attention_kwargs', {}).get('attn_meta', None)
    if attn_meta is None or attn_meta.layer_idx is None:
        if 'cross_attention_kwargs' in cond.unet_kwargs and 'attn_meta' in cond.unet_kwargs['cross_attention_kwargs']:
            del cond.unet_kwargs['cross_attention_kwargs']['attn_meta']

    kwargs["height"] = self.cfg.model.decoder_resolution
    kwargs["width"] = self.cfg.model.decoder_resolution

    if self.cfg.trainer.profile_memory or self.cfg.trainer.fast_eval:
        kwargs["num_inference_steps"] = 1 # Required otherwise the profiler will create traces that are too large

    if self.cfg.model.add_grid_to_input_channels:
        kwargs["concat_latent_along_channel_dim"] = get_tgt_grid(self.cfg, batch)

    if self.cfg.model.add_text_tokens is False:
        negative_prompt_embeds = self.uncond_hidden_states
        prompt_embeds = cond.unet_kwargs['prompt_embeds']
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds.repeat(prompt_embeds.shape[0], 1, prompt_embeds.shape[-1] // negative_prompt_embeds.shape[-1])

    needs_autocast = self.cfg.model.freeze_unet is False or self.cfg.model.unfreeze_single_unet_layer or self.cfg.model.unfreeze_gated_cross_attn
    with torch.cuda.amp.autocast(dtype=self.dtype) if needs_autocast else nullcontext():
        images = self.pipeline(**cond.unet_kwargs, **kwargs).images

    batch.formatted_input_ids = None

    if (
        "cross_attention_kwargs" in cond.unet_kwargs
        and "attn_meta" in cond.unet_kwargs["cross_attention_kwargs"]
        and cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].return_attn_probs is not None
    ):
        cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].return_attn_probs = None

    return images, cond


@torch.no_grad()
def run_quantitative_inference(self: BaseMapper, batch: InputData, state: TrainingState):
    ret = {}

    if self.cfg.model.token_rot_pred_loss or self.cfg.model.token_cls_pred_loss or self.cfg.inference.compute_quantitative_token_metrics:
        with torch.cuda.amp.autocast():
            cond = self.get_standard_conditioning_for_inference(batch=batch)

        ret.update(get_metrics(self, batch, cond))

    pred_data = None
    if self.cfg.model.token_rot_pred_loss:
        with torch.cuda.amp.autocast():
            pred_data, rot_loss_data = self.denoise_rotation(batch=batch, cond=cond, scheduler=self.rotation_scheduler)

        ret.update({k:v.float().cpu() for k,v in rot_loss_data.items()})
        ret["rot_data"] = {
            "quaternions": batch.quaternions.float().cpu(),
            "raw_object_quaternions": batch.raw_object_quaternions.float().cpu(),
            "camera_quaternions": batch.camera_quaternions.float().cpu(),
            "metadata": batch.metadata,
            **{k:v.detach().float().cpu() for k,v in asdict(pred_data).items() if isinstance(v, torch.Tensor)}
        }

    if self.cfg.model.token_cls_pred_loss:
        if pred_data is None:
            with torch.cuda.amp.autocast():
                from gen.models.cross_attn.base_model import TokenPredData
                pred_data = TokenPredData()
                pred_data = self.token_mapper(batch=batch, cond=cond, pred_data=pred_data)
                
        loss_ret = token_cls_loss(self.cfg, batch, cond, pred_data)
        for k in list(loss_ret.keys()):
            if isinstance(loss_ret[k], torch.Tensor):
                loss_ret[k] = loss_ret[k].detach().float().cpu()
        ret.update(loss_ret)

    return ret

@torch.no_grad()
def run_qualitative_inference(self: BaseMapper, batch: InputData, state: TrainingState):
    """
    TODO: Support batched inference at some point. Right now it is batched under the hood (for CFG and if num_images_per_prompt > 1) but we can likely do much better.
    """
    if batch.bs == 0:
        log_info("No images in batch, skipping inference.")
        return {}

    if len(torch.unique(batch.src_segmentation)) == 1 or len(torch.unique(batch.tgt_segmentation)) == 1:
        log_info(f"Only one segmentation class, skipping inference. {batch.metadata}")
        return {}
    
    assert batch.input_ids.shape[0] == 1 or self.cfg.model.predict_rotation_from_n_frames is not None, f"Batched inference is not supported for qualitative inference, bs: {batch.input_ids.shape[0]}"
    orig_batch = batch.clone()
    # is_training_batch
    batch.one_hot_tgt_segmentation = integer_to_one_hot(batch.tgt_segmentation, num_channels=self.cfg.model.segmentation_map_size)

    ret = {}

    # We hope that the train/val src_transforms are the same.
    orig_src_image = Im(undo_normalization_given_transforms(self.cfg.dataset.val.augmentation.src_transforms, batch.src_pixel_values.clone()))
    orig_tgt_image = Im(undo_normalization_given_transforms(self.cfg.dataset.val.augmentation.tgt_transforms, batch.tgt_pixel_values.clone()))

    if self.cfg.model.token_rot_pred_loss:
        with torch.cuda.amp.autocast():
            cond = self.get_standard_conditioning_for_inference(batch=batch)
            try:
                pred_data, _ = self.denoise_rotation(batch=batch, cond=cond, scheduler=self.rotation_scheduler)
                R_ref = compute_rotation_matrix_from_ortho6d(pred_data.gt_rot_6d).cpu().numpy()
                R_pred = compute_rotation_matrix_from_ortho6d(pred_data.pred_6d_rot).cpu().numpy()
            except Exception as e:
                print(e)
                return ret

            assert R_ref.shape == R_pred.shape

            global gso_pcds
            if gso_pcds is None:
                # Load all PCDs once. This is wasteful but good enough.
                gso_pcds = np.load(GSO_PCD_PATH)

            bs = batch.tgt_pixel_values.shape[0]
            
            all_imgs = []
            all_videos = []
            rot_pred_idx = 0
            if self.cfg.model.predict_rotation_from_n_frames is None:
                for b in range(bs):
                    mask_ = ((cond.mask_batch_idx == b) & pred_data.token_output_mask).to(orig_tgt_image.device)
                    instance_idx = cond.mask_instance_idx[mask_]
                    pred_rot_viz = []
                    for j in instance_idx:
                        composited_image = get_composited_mask(batch, b, j)
                        asset_id = batch.asset_id[j - 1][b]
                        pcd = gso_pcds[asset_id]
                        if self.cfg.inference.visualize_rotation_denoising and pred_data.denoise_history_6d_rot.shape[1] > 1:
                            # When we perform rotation classification (binning), denoise_history_6d_rot is not valid so we skip this visualization
                            R_hist = compute_rotation_matrix_from_ortho6d(rearrange("m h ... -> (m h) ...", pred_data.denoise_history_6d_rot))
                            R_hist = rearrange("(m h) ... -> m h ...", R_hist, m=pred_data.gt_rot_6d.shape[0]).cpu().numpy()
                            def evenly_spaced_indices(n, percentage=0.1, minimum=3):
                                num_elements = max(minimum, int(n * percentage))
                                if num_elements <= minimum:
                                    return [0, n // 2, n - 1]
                                spacing = (n - 1) / (num_elements - 1)
                                return np.round(np.arange(0, n, spacing)).astype(int)

                            img = []
                            idx_ = evenly_spaced_indices(R_hist.shape[1])
                            for t in idx_:
                                img.append(Im.concat_horizontal(Im(visualize_rotations_pcds(R_ref[rot_pred_idx], R_hist[rot_pred_idx, t], pcd)).write_text(f"Timestep {pred_data.denoise_history_timesteps[t]}"), composited_image))

                            all_videos.append(Im(torch.stack([x.torch for x in img])).encode_video(fps=4))
                            img = Im.concat_vertical(*img)
                        else:
                            img = Im(visualize_rotations_pcds(R_ref[rot_pred_idx], R_pred[rot_pred_idx], pcd))

                        pred_rot_viz.append(Im.concat_vertical(img, composited_image))
                        rot_pred_idx += 1

                    try:
                        all_imgs.append(Im.concat_horizontal(*pred_rot_viz))
                    except Exception as e:
                        print(e)

            else:
                assert pred_data.token_output_mask.sum().item() // 2 == R_pred.shape[0]
                assert self.cfg.inference.num_images_per_prompt == 1
                assert self.cfg.inference.infer_new_prompts is False

                all_rot_data = get_relative_rot_data(self.cfg, cond, batch)
                group_size = self.cfg.model.predict_rotation_from_n_frames
                for group_idx, group_instance_data in all_rot_data.items():
                    assert group_idx == 0
                    pred_rot_viz = []
                    for instance_idx, instance_rot_data in group_instance_data.items():
                        if len(instance_rot_data) != group_size or instance_idx == 0:
                            continue
                        batch_indices, token_indices = zip(*instance_rot_data)
                        instance_indices = cond.mask_instance_idx[[token_indices]]
                        imgs_ = Im.concat_horizontal(*[get_composited_mask(batch, b, j) for b, j in zip(batch_indices, instance_indices)])
        
                        assert (cond.mask_instance_idx[[token_indices]] == cond.mask_instance_idx[token_indices[0]]).all().item()
                        assert (batch.asset_id[cond.mask_instance_idx[token_indices[0]] - 1][batch_indices[0]] == batch.asset_id[cond.mask_instance_idx[token_indices[1]] - 1][batch_indices[1]])

                        # Visualize rotations
                        asset_id = batch.asset_id[cond.mask_instance_idx[token_indices[0]] - 1][batch_indices[0]]
                        pcd = gso_pcds[asset_id]
                        img = Im(visualize_rotations_pcds(R_ref[rot_pred_idx], R_pred[rot_pred_idx], pcd))
                        pred_rot_viz.append(Im.concat_vertical(img, imgs_))
                        rot_pred_idx += 1
                    try:
                        all_imgs.append(Im.concat_vertical(*pred_rot_viz, spacing=10, fill=(128, 128, 128)))
                    except Exception as e:
                        print(e)

            ret["rotations"] = Im.concat_horizontal(*all_imgs, spacing=30, fill=(128, 128, 128))
            if len(all_videos) > 0: ret["rotation_videos"] = all_videos

    if self.cfg.model.unet is False:
        return ret

    bs_ = batch.input_ids.shape[0]
    b_ = 0
    gt_info = Im.concat_vertical(*[
        Im.concat_vertical(
            Im(orig_tgt_image.torch[b_]), 
            Im(onehot_to_color(batch.one_hot_tgt_segmentation[b_])).write_text(text="Target", color=(164, 164, 128))
        ) for b_ in range(bs_)
    ])

    if self.cfg.model.eschernet or self.cfg.model.add_grid_to_input_channels or self.cfg.model.modulate_src_tokens_with_tgt_pose or self.cfg.model.modulate_src_feature_map or self.cfg.model.encode_tgt_enc_norm or self.cfg.model.encode_src_twice:
        src_one_hot = integer_to_one_hot(batch.src_segmentation + 1, batch.src_segmentation.max() + 1)
        src_seg = Im(onehot_to_color(src_one_hot[b_].squeeze(0))).write_text("Source", color=(164, 164, 128))
        src_info = Im.concat_vertical(orig_src_image.resize(orig_tgt_image.height, orig_tgt_image.width), src_seg.resize(orig_tgt_image.height, orig_tgt_image.width))
        gt_info = Im.concat_horizontal(src_info, gt_info, spacing=20)

    ret["validation"] = gt_info

    added_kwargs = dict()
    if self.cfg.inference.visualize_attention_map:
        added_kwargs["return_attn_probs"] = True

    batch.formatted_input_ids = None
    batch.attach_debug_info = True
    batch.treat_as_train_batch = (state.split == Split.TRAIN) and torch.unique(batch.src_segmentation).shape[0] > 2
    prompt_image, cond = self.infer_batch(batch=batch, num_images_per_prompt=self.cfg.inference.num_images_per_prompt, **added_kwargs)
    batch.attach_debug_info = False
    batch.treat_as_train_batch = False

    if self.cfg.inference.compute_quantitative_token_metrics is False or self.cfg.trainer.custom_inference_every_n_steps is None:
        ret.update(get_metrics(self, batch, cond))
    try:
        ret["data_viz"] = visualize_input_data(orig_batch.clone(), cfg=self.cfg, show_overlapping_masks=True, remove_invalid=False, return_img=True, cond=cond, tokenizer=self.tokenizer)
    except Exception as e:
        log_warn(f"Failed to visualize input data", main_process_only=False)
        traceback.print_exc()
        ret["data_viz"] = Im.new(256, 256)
        
    cond.src_feature_map = None

    if (self.cfg.model.break_a_scene_masked_loss or self.cfg.model.src_tgt_consistency_loss_weight is not None or self.cfg.model.mask_dropped_tokens) and self.cfg.model.mask_token_conditioning:
        assert batch.bs == 1
        
        _orig_src_one_hot = integer_to_one_hot(batch.src_segmentation.clone())
        _diffusion_loss_mask = break_a_scene_masked_loss(self.cfg, batch, cond)

        _conditioned_src_seg = batch.src_segmentation.clone()
        _first_valid_seg = torch.isin(_conditioned_src_seg, cond.mask_instance_idx)
        _conditioned_src_seg[~_first_valid_seg] = 255
        _first_src_one_hot__ = integer_to_one_hot(_conditioned_src_seg)
        
        ret["loss_mask"] = Im.concat_horizontal(
            orig_src_image,
            *([Im(cond.encoder_input_pixel_values[0]).denormalize()] if self.cfg.model.mask_dropped_tokens else []),
            Im(onehot_to_color(_orig_src_one_hot[0])).write_text("Src Before Dropout", size=0.5),
            Im(onehot_to_color(_first_src_one_hot__[0])).write_text("Src After Dropout", size=0.5),
            Im(_first_valid_seg).bool_to_rgb().write_text(f"Unique: {torch.unique(_conditioned_src_seg).shape[0] - 1}", size=0.5),
            *([Im(_diffusion_loss_mask.bool()).bool_to_rgb().resize(orig_src_image.height, orig_src_image.width, resampling_mode=InterpolationMode.NEAREST)] if self.cfg.model.break_a_scene_masked_loss else []),
        )

        if self.cfg.model.mask_dropped_tokens and self.cfg.model.encode_src_twice:
            assert cond.encoder_input_pixel_values.shape[0] == 2
            _second_src_seg = batch.src_segmentation.clone()
            _second_valid_seg = torch.isin(_second_src_seg, cond.tgt_mask_instance_idx)
            _second_src_seg[~_second_valid_seg] = 255
            _second_src_one_hot__ = integer_to_one_hot(_second_src_seg)

            ret["loss_mask"] = Im.concat_vertical(
                ret["loss_mask"],
                Im.concat_horizontal(
                    orig_src_image,
                    *([Im(cond.encoder_input_pixel_values[1]).denormalize().write_text("Tgt img")] if self.cfg.model.mask_dropped_tokens else []),
                    Im(onehot_to_color(_orig_src_one_hot[0])).write_text("Tgt Before Dropout", size=0.5),
                    Im(onehot_to_color(_second_src_one_hot__[0])).write_text("Tgt After Dropout", size=0.5),
                    Im(_second_valid_seg).bool_to_rgb().write_text(f"Unique: {torch.unique(_second_src_seg).shape[0] - 1}", size=0.5),
                )
            )

        if self.cfg.model.mask_dropped_tokens: cond.encoder_input_pixel_values = None

    if self.cfg.inference.visualize_attention_map:
        attn = cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].attn_probs
        tgt_size = (self.cfg.model.decoder_latent_dim, self.cfg.model.decoder_latent_dim)
        all_attn = []
        bs = len(prompt_image)
        for k, v in attn.items():
            layer_attn = torch.stack(v, dim=0).cpu()[:, -bs:]
            layer_attn = rearrange("timesteps b (h w) tokens -> timesteps (b tokens) () h w", layer_attn, h=int(sqrt(layer_attn.shape[-2])))
            layer_attn = torch.mean(layer_attn, dim=0)
            layer_attn = F.interpolate(layer_attn.to(dtype=torch.float32), size=tgt_size, mode="bilinear", align_corners=False).squeeze(1)
            all_attn.append(layer_attn)

        try:
            all_attn = mean("[layers] (b tokens) h w -> b tokens h w", torch.stack(all_attn), b=bs)
            attn_imgs = []
            for b in range(bs):
                attn_ = all_attn[b]
                idx_mask = cond.learnable_idxs[1][cond.learnable_idxs[0] == b]
                tokens = attn_[idx_mask.to(device=attn_.device)]
                if tokens.shape[0] == 0:
                    continue
                tokens = rearrange("t (h w) -> t 3 h w", softmax("t [hw]", rearrange("t h w -> t (h w)", tokens)), h=tokens.shape[-1])
                tokens = (tokens - torch.min(tokens)) / (torch.max(tokens) - torch.min(tokens))
                masks = Im(rearrange("h w masks -> masks 3 h w", batch.one_hot_tgt_segmentation[b, ..., cond.mask_instance_idx]).float()).resize(*tgt_size)
                attn_imgs.append(Im.concat_vertical(Im.concat_horizontal(*tokens), Im.concat_horizontal(*masks.torch)))

            ret["attn_vis"] = Im.concat_vertical(*attn_imgs)
        except Exception as e:
            log_warn(f"Failed to visualize attention maps: {e.__traceback__}", main_process_only=False)
            traceback.print_exc()

    use_idx = self.cfg.model.predict_rotation_from_n_frames is not None

    _conditioned_src_seg = batch.src_segmentation.clone()
    _conditioned_src_seg[~torch.isin(_conditioned_src_seg, cond.mask_instance_idx)] = 255
    _conditioned_src_seg = integer_to_one_hot(_conditioned_src_seg)
    generated_images = Im.concat_vertical(
        *(Im(prompt_image_).write_text(text=f"Gen {i}", color=(164, 164, 128)) for i, prompt_image_ in enumerate(prompt_image)),
        Im(onehot_to_color(_conditioned_src_seg[i if use_idx else 0])).resize(Im(prompt_image[0]).height, Im(prompt_image[0]).width)
    )
    
    ret["validation"] = Im.concat_horizontal(ret["validation"], generated_images)

    if self.cfg.inference.save_prompt_embeds:
        assert cond.mask_instance_idx.shape[0] == cond.mask_tokens.shape[0]
        orig_tgt_segmentation = batch.one_hot_tgt_segmentation.clone()
        all_masks = []
        for j in cond.mask_instance_idx:
            composited_image = get_composited_mask(batch, 0, j)
            all_masks.append(composited_image.write_text(f"token_{j}").resize(256, 256).np)

        ret["cond"] = {"mask_tokens": cond.mask_tokens, "mask_rgb": np.stack(all_masks), "orig_image": orig_tgt_image.np}

    torch.cuda.empty_cache()

    if self.cfg.model.only_encode_shared_tokens and (self.cfg.model.encode_src_twice or self.cfg.model.encode_tgt_enc_norm):
        ret["view_pred"] = Im.new(64, 64)
        try:
            _batch = repeat_batch(batch, 5)
            assert _batch.bs == 4
            _batch.force_encode_all_masks = torch.full((_batch.bs,), False, dtype=torch.bool)
            _batch.force_encode_all_masks[[0, 3, 4]] = True

            with torch.cuda.amp.autocast():
                _cond = self.get_standard_conditioning_for_inference(batch=_batch)
            
            layers = _cond.encoder_hidden_states.shape[-1] // self.uncond_hidden_states.shape[-1]
            _cond.encoder_hidden_states[2:4, :, :] = rearrange('t d -> b t (l d)', self.uncond_hidden_states, l=layers, b=2)

            shared_tgt_tokens = _cond.tgt_mask_tokens[_cond.tgt_mask_batch_idx == 2]
            if shared_tgt_tokens.shape[0] > self.cfg.model.num_decoder_cross_attn_tokens:
                log_info(f"Got too many tokens...{batch.metadata}")
            _cond.encoder_hidden_states[2, :min(shared_tgt_tokens.shape[0], _cond.encoder_hidden_states.shape[1]), :] = shared_tgt_tokens[:_cond.encoder_hidden_states.shape[1]]

            all_tgt_tokens = _cond.tgt_mask_tokens[_cond.tgt_mask_batch_idx == 3]
            _cond.encoder_hidden_states[3, :all_tgt_tokens.shape[0], :] = all_tgt_tokens

            all_gt_src_tokens = _cond.gt_src_mask_token[_cond.mask_batch_idx == 4]
            _cond.encoder_hidden_states[4, :all_gt_src_tokens.shape[0], :] = all_gt_src_tokens
            
            _prompt_images, _ = self.infer_batch(batch=_batch, cond=_cond, num_images_per_prompt=1)
            _h, _w = _prompt_images[0].height, _prompt_images[0].width

            ret["view_pred"] = Im.concat_vertical(
                Im.concat_horizontal(
                    Im(_prompt_images[4]).write_text("Src Auto-Enc", size=0.5),
                    Im(_prompt_images[0]).write_text("Src Enc -> Tgt, All Tok", size=0.5),
                    Im(_prompt_images[1]).write_text("Src Enc -> Tgt, Shared Tok", size=0.5),
                    Im(_prompt_images[2]).write_text("GT Tgt Enc, Shared Tok", size=0.5),
                    Im(_prompt_images[3]).write_text("GT Tgt Enc, All Tok", size=0.5),
                ),
                Im.concat_horizontal(
                    Im(orig_src_image.torch.cpu()).resize(_h, _w).write_text("Src Image", size=0.5),
                    Im(orig_tgt_image.torch.cpu()).resize(_h, _w).write_text("GT Tgt Image", size=0.5),
                    spacing=(Im(_prompt_images[0]).width * 2)
                )
            )
        except:
            log_warn("Failed to visualize shared tokens", main_process_only=False)
            traceback.print_exc()

    if self.cfg.inference.visualize_positional_control:
        ret["positional_control"] = Im.new(64, 64)
        try:
            log_info(f"Visualizing positional control", main_process_only=False)
            num_from_top = 3
            max_num_viz = 4
            num_variations = 9

            _batch = repeat_batch(batch, num_variations)
            
            indices_ = batch.src_segmentation.view(batch.batch_size[0], -1)
            ones_ = torch.ones_like(indices_, dtype=torch.int32)
            all_non_empty_mask = ones_.new_zeros((batch.batch_size[0], 256))
            all_non_empty_mask.scatter_add_(1, indices_.long(), ones_)
            all_non_empty_mask = all_non_empty_mask[:, :-1]

            sorted, indices = torch.sort(all_non_empty_mask.squeeze(0), descending=True)
            indices = indices[num_from_top:sorted.nonzero().shape[0]][:max_num_viz]
            from gen.models.utils import positionalencoding2d
            for instance_idx in indices:
                with torch.cuda.amp.autocast():
                    _cond = self.get_standard_conditioning_for_inference(batch=_batch)

                    pos_emb = positionalencoding2d(self.cfg.model.pos_emb_dim, self.cfg.model.encoder_latent_dim, self.cfg.model.encoder_latent_dim, device=self.device, dtype=self.dtype)

                    instance_mask = _cond.mask_instance_idx == instance_idx
                    
                    if instance_mask.sum() == 0:
                        continue

                    _gt_centroids = _cond.mask_token_centroids[instance_mask]

                    variations = torch.tensor([(0, 0), (-4, 0), (-16, 0), (0, -8), (0, -16), (8, 0), (16, 0), (0, 8), (0, 16)])[:num_variations]

                    _gt_centroids = _gt_centroids + variations.to(_gt_centroids)
                    _gt_centroids = torch.clamp(_gt_centroids, 0, self.cfg.model.encoder_latent_dim - 1)
                    
                    _cond.gt_src_mask_token_pos_emb[(_cond.mask_instance_idx == instance_idx)] = einops.rearrange(pos_emb[:, _gt_centroids[:, 0], _gt_centroids[:, 1]], "d tokens -> tokens d")

                    new_src_tokens = self.mapper.inject_positional_information_film(_cond.src_mask_tokens_before_specialization, _cond.gt_src_mask_token_pos_emb)
                    _cond.mask_tokens[:] = new_src_tokens
                    _cond = self.update_hidden_state_with_mask_tokens(_batch, _cond)

                    _prompt_images, _ = self.infer_batch(batch=_batch, cond=_cond, num_images_per_prompt=1)

                    ret["positional_control"] = Im.concat_vertical(
                        ret["positional_control"],
                        Im.concat_vertical(
                            Im.concat_horizontal(
                                Im(_prompt_image).write_text(f"{(_variation.cpu().tolist())}", size=0.5) for _variation, _prompt_image in zip(variations, _prompt_images)
                            ),
                            Im.concat_horizontal(
                                Im(orig_src_image.torch).resize(orig_tgt_image.height, orig_tgt_image.width),
                                Im((batch.src_segmentation == instance_idx).squeeze(0)).resize(orig_tgt_image.height, orig_tgt_image.width).write_text(f"Instance {instance_idx}", size=0.5),
                            ),
                        )
                    )
        except:
            log_warn("Failed to visualize shared tokens", main_process_only=False)
            traceback.print_exc()


    if self.cfg.inference.vary_cfg_plot:
        scale_images = []
        for scale in [0.0, 3.0, 5.0, 7.5, 10.0]:
            if self.cfg.model.attention_masking and scale <= 1.0:
                continue
            prompt_images, cond = self.infer_batch(
                batch=batch, cond=cond, num_images_per_prompt=self.cfg.inference.num_images_per_prompt, guidance_scale=scale
            )
            scale_images.append(Im.concat_vertical(prompt_images).write_text(f"CFG Scale: {scale:.1f}"))

        assert batch.bs == 1
        ret["cfg_scale"] = Im.concat_horizontal(orig_tgt_image.to("cpu").copy.write_text("GT")[0], *scale_images)

    if self.cfg.model.break_a_scene_cross_attn_loss:
        batch_size = (2 if self.cfg.inference.guidance_scale > 1.0 else 1) * self.cfg.inference.num_images_per_prompt

        agg_attn_cond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size
        )
        attn_img_cond = Im(
            save_cross_attention_vis(tokenizer=self.tokenizer, tokens=batch.formatted_input_ids[0], attention_maps=agg_attn_cond)
        ).write_text("Cond")

        agg_attn_uncond = aggregate_attention(
            controller=self.controller, res=16, from_where=("up", "down"), is_cross=True, select=0, batch_size=batch_size
        )
        uncond_tokens = self.pipeline.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        attn_img_uncond = Im(
            save_cross_attention_vis(tokenizer=self.tokenizer, tokens=uncond_tokens["formatted_input_ids"][0], attention_maps=agg_attn_uncond)
        ).write_text("Uncond")

        ret["attn_vis"] = Im.concat_vertical(attn_img_cond, attn_img_uncond)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    log_debug("Before multi-mask", main_process_only=False)

    if self.cfg.inference.num_masks_to_remove is not None:
        modified_batches = []
        
        idxs = get_valid_idxs(batch)
        idxs = idxs[:self.cfg.inference.num_masks_to_remove]

        for _, j in enumerate(idxs):
            batch_ = batch.clone()
            # We only create tokens for segmentation channels with more than 0 valid pixels so 255 is always ignored.
            batch_.tgt_segmentation[batch_.tgt_segmentation == j] = 255
            batch_.src_segmentation[batch_.src_segmentation == j] = 255

            if (batch_.src_segmentation != 255).sum() == 0:
                log_warn("Skipping permutting masks...", main_process_only=False)
                continue
            modified_batches.append(batch_)

        modified_batches = torch.cat(modified_batches, dim=0) if len(modified_batches) > 0 else modified_batches

        if len(modified_batches) != 0:
            prompt_images, cond_ = self.infer_batch(batch=modified_batches, num_images_per_prompt=1)
            if self.cfg.model.break_a_scene_cross_attn_loss:
                self.controller.reset()
            
            removed_mask_imgs = []
            for (i, j) in enumerate(idxs):
                bs_ = batch.input_ids.shape[0]
                img_ = []
                for b_ in range(bs_):
                    assert len(set(cond_.mask_instance_idx[cond_.mask_batch_idx == i].tolist()).intersection(set([j]))) == 0

                    gen_img_ =  Im(prompt_images[(i * bs_) + b_])
                    mask_ = batch.one_hot_tgt_segmentation[b_, ..., j]

                    gen_img_ = overlay_mask(gen_img_, mask_, color=(0, 0, 0), alpha=0.0)
                    ref_img_ = overlay_mask(orig_tgt_image, mask_, color=(255, 0, 0), alpha=0.9)

                    img_.append(Im.concat_vertical(gen_img_, ref_img_, spacing=5, fill=(128, 128, 128)))
                removed_mask_imgs.append(Im.concat_vertical(*img_))
            
            removed_masks = Im.concat_horizontal(*removed_mask_imgs).write_text("Removed single mask token", size=0.25, color=(0, 255, 0))
            ret["validation"] = Im.concat_horizontal(ret["validation"], removed_masks, spacing=15)

    log_debug("Before single-mask", main_process_only=False)

    if self.cfg.inference.num_single_token_gen is not None:
        assert batch.bs == 1

        modified_batches = []
        idxs = get_valid_idxs(batch)
        idxs = idxs[:self.cfg.inference.num_masks_to_remove]
        
        for _, j in enumerate(idxs):
            batch_ = batch.clone()
            # We only create tokens for segmentation channels with more than 0 valid pixels so 255 is always ignored.
            batch_.tgt_segmentation[batch_.tgt_segmentation != j] = 255
            batch_.src_segmentation[batch_.src_segmentation != j] = 255

            if (batch_.src_segmentation != 255).sum() == 0:
                log_warn("Skipping permutting masks...", main_process_only=False)
                continue
            modified_batches.append(batch_)

        modified_batches = torch.cat(modified_batches, dim=0) if len(modified_batches) > 0 else modified_batches

        if len(modified_batches) != 0:
            prompt_images, cond_ = self.infer_batch(batch=modified_batches, num_images_per_prompt=1)
            if self.cfg.model.only_encode_shared_tokens is False: assert set(cond_.mask_instance_idx.tolist()) == set(idxs.tolist())
            if self.cfg.model.break_a_scene_cross_attn_loss:
                self.controller.reset()
            
            removed_mask_imgs = []
            for (i, j) in enumerate(idxs):
                bs_ = batch.input_ids.shape[0]
                img_ = []
                for b_ in range(bs_):
                    if self.cfg.model.only_encode_shared_tokens is False:  assert set(cond_.mask_instance_idx[cond_.mask_batch_idx == i].tolist()) == set([j.item()])
                    alpha_mask = ~orig_tgt_segmentation[b_, ..., j.long()]
                    gen_img_ =  Im(prompt_images[(i * bs_) + b_])
                    gen_img_ = overlay_mask(gen_img_, alpha_mask, color=(0, 0, 0), alpha=0.75)
                    ref_img_ = overlay_mask(orig_tgt_image, alpha_mask, color=(0, 0, 0), alpha=0.9)
                    img_.append(Im.concat_vertical(gen_img_, ref_img_, spacing=15, fill=(128, 128, 128)))
                removed_mask_imgs.append(Im.concat_vertical(*img_))
            
            removed_masks = Im.concat_horizontal(*removed_mask_imgs).write_text("Single token Conditioning.", size=0.25, color=(0, 255, 0))
            ret["validation"] = Im.concat_horizontal(ret["validation"], removed_masks, spacing=15)

    log_debug("After multi-mask", main_process_only=False)

    if self.cfg.inference.infer_new_prompts:
        prompts = [prompt.format(self.cfg.model.placeholder_token) for prompt in new_prompts]

        input_ids_ = []
        for prompt in prompts[: self.cfg.inference.max_batch_size]:
            input_ids_.append(get_tokens(tokenizer=self.tokenizer, prompt=prompt)[None])
        input_ids_ = torch.cat(input_ids_, dim=0).to(self.device)

        batch_ = repeat_batch(batch, bs=len(input_ids_))
        batch_.input_ids = input_ids_

        prompt_images, _ = self.infer_batch(batch=batch_)
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

        ret["prompt_images"] = Im.concat_horizontal(
            Im(prompt_image).write_text(prompt, size=0.5) for prompt_image, prompt in zip(prompt_images, prompts)
        )
        if ret["prompt_images"].torch.ndim == 3:
            ret["prompt_images"] = ret["prompt_images"][None]

        ret["prompt_images"] = Im.concat_horizontal(orig_tgt_image, ret["prompt_images"], spacing=20)

    img_ = []
    for b in range(len(batch.one_hot_tgt_segmentation)):
        seg = batch.one_hot_tgt_segmentation[b].long().argmax(dim=-1).data.cpu().numpy()
        rgb_seg = label_to_color_image(seg, colormap=CocoPanoptic.create_label_colormap())
        rgb_img = Image.fromarray((((batch.tgt_pixel_values[b] + 1) / 2).data.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
        img_.append(Im.concat_vertical(rgb_seg, rgb_img))
    ret["image_segmentation"] = Im.concat_horizontal(*img_)

    return ret


def take_from(slices: tuple[int, slice], data: tuple[dict]):
    output_mask_tokens = []
    output_mask_rgb = []
    for idx, sl in slices:
        output_mask_tokens.append(data[idx]["mask_tokens"][sl])
        output_mask_rgb.append(data[idx]["mask_rgb"][sl])

    output_mask_tokens = torch.cat(output_mask_tokens, dim=0)
    output_mask_rgb = np.concatenate(output_mask_rgb, axis=0)

    return output_mask_tokens, output_mask_rgb


@torch.no_grad()
def compose_two_images(self: BaseMapper, batch: InputData, state: TrainingState, embed_path_1: Optional[Path] = None, embed_path_2: Optional[Path] = None):
    from gen.models.cross_attn.base_model import ConditioningData

    if embed_path_1 is not None and embed_path_2 is not None:
        image_batch_tokens = {
            0: load_tensor_dict(embed_path_1),
            1: load_tensor_dict(embed_path_2)
        }
        prompt_image_0, cond_0 = self.infer_batch(
            batch=batch[0],
            cond=ConditioningData(
                mask_tokens=image_batch_tokens[0]["mask_tokens"], 
                mask_batch_idx=torch.zeros((image_batch_tokens[0]["mask_tokens"].shape[0],), dtype=torch.int64),
            ),
        )
        prompt_image_1, cond_1 = self.infer_batch(
            batch=batch[1],
            cond=ConditioningData(
                mask_tokens=image_batch_tokens[1]["mask_tokens"], 
                mask_batch_idx=torch.zeros((image_batch_tokens[1]["mask_tokens"].shape[0],), dtype=torch.int64),
            ),
        )
        orig_prompt_images = [*prompt_image_0, *prompt_image_1]
    else:
        orig_prompt_images, orig_cond = self.infer_batch(batch=batch)
        batch.one_hot_tgt_segmentation = integer_to_one_hot(batch.tgt_segmentation, num_classes=self.cfg.model.segmentation_map_size)
        image_batch_tokens = {}
        for b in range(batch.bs):
            orig_image = Im((batch.tgt_pixel_values[b] + 1) / 2)
            all_masks = []
            for j in orig_cond.mask_instance_idx[orig_cond.mask_batch_idx == b]:
                composited_image = get_composited_mask(batch, b, j)
                all_masks.append(composited_image.write_text(f"token_{j}").resize(self.cfg.model.decoder_resolution, self.cfg.model.decoder_resolution).np)

            image_batch_tokens[b] = {"mask_tokens": orig_cond.mask_tokens[orig_cond.mask_batch_idx == b], "mask_rgb": np.stack(all_masks), "orig_image": orig_image.np}
    
    num_tokens_first, num_tokens_second = image_batch_tokens[0]['mask_tokens'].shape[0], image_batch_tokens[1]['mask_tokens'].shape[0]
    num_elements_second = torch.randint(1, 3, (1,)).item()  # Take either 1 or 2 from second image
    start_index = torch.randint(0, max(1, num_tokens_second - num_elements_second + 1), (1,)).item()
    slices_to_take = (
        (0, slice(0, torch.randint(max(1, num_tokens_first // 2), max(2, num_tokens_first + 1), (1,)).item())),
        (1, slice(start_index, start_index + num_elements_second)),
    )

    final_tokens, final_rgb = take_from(
        slices=slices_to_take,
        data=(image_batch_tokens[0], image_batch_tokens[1]),
    )

    mask_batch_idx = torch.zeros((final_tokens.shape[0],), dtype=torch.int64)
    prompt_image, cond = self.infer_batch(
        batch=batch[[0]], cond=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=4
    )

    slice_description = ", ".join([f"Took {sl_.start}:{sl_.stop} from {batch_idx}" for i, (batch_idx, sl_) in enumerate(slices_to_take)])
    visualize_compose_img = Im.concat_vertical(
        Im.concat_horizontal(Im(prompt_image_).write_text(text=f"Gen {i}") for i, prompt_image_ in enumerate(prompt_image)),
        Im(utils.make_grid(Im(final_rgb).torch)).write_text("Conditioning Masks [GT] used", size=0.45, position=(0.05, 0.01)),
        Im.concat_horizontal(
            Im(image_batch_tokens[0]["orig_image"]).write_text("First Image GT", size=0.5),
            Im(orig_prompt_images[0]).write_text("First Image Autoencoded", size=0.5),
        ),
        Im.concat_horizontal(
            Im(image_batch_tokens[1]["orig_image"]).write_text("Second Image GT", size=0.5),
            Im(orig_prompt_images[1]).write_text("Second Image Autoencoded", size=0.5),
        ),
    ).write_text(slice_description, position=(0.9, 0.5), size=0.25)

    ret = {}
    ret["compose_two_images"] = visualize_compose_img
    return ret


@torch.no_grad()
def interpolate_latents(
    self: BaseMapper,
    batch: InputData,
    state: TrainingState,
    embed_path_1: Path,
    embed_path_2: Path,
    remove_background_token: bool = False,
    batch_interp: bool = True,
    steps: int = 10,
):
    def lerp(a, b, ts):
        return a + (b - a) * ts
    
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData

    image_0_dict = load_tensor_dict(embed_path_1)
    image_1_dict = load_tensor_dict(embed_path_2)

    image_0_tokens = image_0_dict["mask_tokens"]
    image_1_tokens = image_1_dict["mask_tokens"]

    # prompt_image_0, cond_0 = self.infer_batch(
    #     batch=batch,
    #     cond=ConditioningData(mask_tokens=image_0_tokens, mask_batch_idx=torch.zeros((image_0_tokens.shape[0],), dtype=torch.int64)),
    # )
    # prompt_image_1, cond_1 = self.infer_batch(
    #     batch=batch,
    #     cond=ConditioningData(mask_tokens=image_1_tokens, mask_batch_idx=torch.zeros((image_1_tokens.shape[0],), dtype=torch.int64)),
    # )

    interp_values = torch.linspace(0, 1, steps=steps)

    prompt_images = []
    if batch_interp:
        final_tokens = []
        mask_batch_idx = []
        for i, interp_value in enumerate(interp_values):
            interp_token = lerp(image_0_tokens, image_1_tokens, interp_value)
            final_tokens.append(interp_token)
            mask_batch_idx.append(i * torch.ones((interp_token.shape[0],), dtype=torch.int64))

        final_tokens = torch.cat(final_tokens, dim=0)
        mask_batch_idx = torch.cat(mask_batch_idx, dim=0)
        batch.formatted_input_ids = None

        batch_ = {}
        batch_["input_ids"] = batch.input_ids.repeat(steps, 1)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and k not in batch_:
                assert v.shape[0] == 1
                batch_[k] = repeat(v[0], "... -> h ...", h=batch_["input_ids"].shape[0])

        prompt_images, cond = self.infer_batch(
            batch=batch_, cond=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=1
        )
    else:
        for i, interp_value in enumerate(interp_values):
            interp_token = lerp(image_0_tokens, image_1_tokens, interp_value)
            if remove_background_token:
                interp_token = interp_token[1:]
            prompt_image, cond = self.infer_batch(
                batch=batch,
                cond=ConditioningData(mask_tokens=interp_token, mask_batch_idx=torch.zeros((interp_token.shape[0],), dtype=torch.int64)),
                num_images_per_prompt=2,
            )
            prompt_image = Im.concat_vertical(prompt_image)
            prompt_images.append(prompt_image)

    Im.concat_vertical(
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Weight {interp_values[i].item():.2f}") for i, prompt_image_ in enumerate(prompt_images)
        ),
        Im.concat_horizontal(
            Im(image_0_dict["orig_image"]).write_text("Left (0) GT", size=0.75),
            Im(image_1_dict["orig_image"]).write_text("Right (1) GT", size=0.75),
        ),
        spacing=50,
    ).save("test_01.png")

    breakpoint()


def hungarian_matching(tensor_a, tensor_b):
    from scipy.optimize import linear_sum_assignment
    dist_matrix = torch.cdist(tensor_a.float(), tensor_b.float())
    dist_matrix_np = dist_matrix.cpu().numpy()
    row_indices, col_indices = linear_sum_assignment(dist_matrix_np)
    return row_indices, col_indices

@torch.no_grad()
def interpolate_frames(
    self: BaseMapper, 
    batch: InputData, 
    state: TrainingState, 
    steps: int = 10,
    num_tokens_interpolate: int = 2
):
    def lerp(a, b, ts):
        return a + (b - a) * ts
    
    assert batch.bs == 1
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData
    batch.force_forward_encoder_normalized_tgt = True
    orig_prompt_images, orig_cond = self.infer_batch(batch=batch)
    batch.force_forward_encoder_normalized_tgt = False
    
    src_matches, tgt_matches = hungarian_matching(orig_cond.src_mask_tokens, orig_cond.tgt_mask_tokens)
    src_matches, tgt_matches = torch.from_numpy(src_matches[:num_tokens_interpolate]).to(self.device), torch.from_numpy(tgt_matches[:num_tokens_interpolate]).to(self.device)

    def get_indices(_matches, _instance_idxs, _seg):
        _valid_idxs = _instance_idxs[_matches]
        _one_hot_seg = get_one_hot_channels(_seg, _valid_idxs)
        return _one_hot_seg

    _src_one_hot = get_indices(src_matches, orig_cond.mask_instance_idx, batch.src_segmentation)
    _tgt_one_hot = get_indices(tgt_matches, orig_cond.tgt_mask_instance_idx, batch.tgt_enc_norm_segmentation)

    image_0_rgb = Im(batch.src_pixel_values).denormalize()
    image_1_rgb = Im(batch.tgt_enc_norm_pixel_values).denormalize()

    image_0_constant_tokens = orig_cond.mask_tokens[~torch.isin(torch.arange(orig_cond.mask_tokens.shape[0], device=batch.device), src_matches)]
    image_1_constant_tokens = orig_cond.tgt_mask_tokens[~torch.isin(torch.arange(orig_cond.tgt_mask_tokens.shape[0], device=batch.device), tgt_matches)]

    image_0_tokens = orig_cond.mask_tokens[src_matches]
    image_1_tokens = orig_cond.tgt_mask_tokens[tgt_matches]

    all_constant_tokens = (image_0_constant_tokens, image_1_constant_tokens, image_1_constant_tokens.new_zeros((0, image_0_tokens.shape[1])))

    interp_values = torch.linspace(0, 1, steps=steps)
    prompt_images = []
    final_tokens = []
    mask_batch_idx = []
    for i, _const_tokens in enumerate(all_constant_tokens):
        for j, interp_value in enumerate(interp_values):
            interp_token = lerp(image_0_tokens, image_1_tokens, interp_value)
            interp_token = torch.cat([_const_tokens, interp_token], dim=0)
            final_tokens.append(interp_token)
            mask_batch_idx.append((i * len(interp_values)) + j * torch.ones((interp_token.shape[0],), dtype=torch.int64))

    final_tokens = torch.cat(final_tokens, dim=0)
    mask_batch_idx = torch.cat(mask_batch_idx, dim=0)
    batch.formatted_input_ids = None

    batch_ = repeat_batch(batch, bs=steps * len(all_constant_tokens))
    prompt_images, cond = self.infer_batch(
        batch=batch_, cond=ConditioningData(mask_tokens=final_tokens, mask_batch_idx=mask_batch_idx), num_images_per_prompt=1
    )

    ret = {}

    ret['interp'] = Im.concat_vertical(
        Im.concat_horizontal(
            Im.concat_vertical(
                image_0_rgb.write_text("Left (0) GT", size=0.75).torch,
                Im(integer_to_color(batch.src_segmentation[0, ..., 0].long())).write_text("Before Matching"),
                Im(onehot_to_color(_src_one_hot[0], ignore_empty=False)).write_text("After Matching")), 
            Im.concat_vertical(
                image_1_rgb.write_text("Right (1) GT", size=0.75).torch,
                Im(integer_to_color(batch.tgt_enc_norm_segmentation[0, ..., 0].long())).write_text("Before Matching"),
                Im(onehot_to_color(_tgt_one_hot[0], ignore_empty=False)).write_text("After Matching")),
        ),
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Weight {interp_values[i].item():.2f}") for i, prompt_image_ in enumerate(prompt_images[:len(interp_values)])
        ).add_border(border=100, color=(128, 128, 128)).write_text("Background tokens from left img"),
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Weight {interp_values[i].item():.2f}") for i, prompt_image_ in enumerate(prompt_images[len(interp_values):2 * len(interp_values)])
        ).add_border(border=100, color=(128, 128, 128)).write_text("Background tokens from right img"),
        Im.concat_horizontal(
            Im(prompt_image_).write_text(text=f"Weight {interp_values[i].item():.2f}") for i, prompt_image_ in enumerate(prompt_images[2 * len(interp_values):])
        ).add_border(border=100, color=(128, 128, 128)).write_text("No background tokens"),
        spacing=50,
    )

    ret['interp'].save(f"{batch.metadata['name'][0]}_interp")

    return ret