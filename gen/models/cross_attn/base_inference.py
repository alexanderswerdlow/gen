from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import einops
from einx import mean, rearrange, softmax
from image_utils import ChannelRange, Im, onehot_to_color
from PIL import Image
from torchvision import utils
from gen import GSO_PCD_PATH

from gen.datasets.abstract_dataset import Split
from gen.utils.trainer_utils import TrainingState
from dataclasses import asdict, fields
from torchvision.transforms.functional import InterpolationMode, resize

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData


@torch.no_grad()
def infer_batch(
    self: BaseMapper,
    batch: InputData,
    num_images_per_prompt: int = 1,
    cond: Optional[ConditioningData] = None,
    allow_get_cond: bool = True,
    **kwargs,
) -> tuple[list[Any], dict, ConditioningData]:
    if batch.input_ids is not None:
        orig_input_ids = batch.input_ids.clone()
        batch.input_ids = orig_input_ids.clone()

    if cond is None or len(cond.unet_kwargs) == 0 and allow_get_cond:
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
def run_qualitative_inference(self: BaseMapper, batch: InputData, state: TrainingState):
    pass
    return ret
