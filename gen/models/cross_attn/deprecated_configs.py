from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from gen.models.utils import find_true_indices_batched

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import BaseMapper, ConditioningData, InputData


def init_shift_scale(self):
    def zero_module(module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    clip_proj_layers = []
    all_layer_data = [(k, v.shape[0]) for k, v in dict(self.unet.named_parameters()).items() if "resnets.0" in k and "conv1.weight" in k]
    dims = []
    dims.extend([v for k, v in all_layer_data if "down_blocks" in k])
    dims.extend([v for k, v in all_layer_data if "mid_block" in k])
    dims.extend([v for k, v in all_layer_data if "up_blocks" in k])

    for idx, dim in enumerate(dims):
        clip_proj_layers.append(nn.Conv2d(self.cfg.model.encoder_dim, dim * 2, kernel_size=1))
    self.clip_proj_layers = nn.ModuleList(clip_proj_layers)
    self.clip_proj_layers = zero_module(self.clip_proj_layers)


def forward_shift_scale(self, cond: Optional[ConditioningData], clip_feature_map, latent_dim):
    clip_feature_maps = []
    for _, layer in enumerate(self.clip_proj_layers):
        proj_feature_map = layer(rearrange(clip_feature_map, "b (h w) d -> b d h w", h=latent_dim, w=latent_dim))
        clip_feature_maps.append(proj_feature_map)

    cond.unet_kwargs["femb"] = clip_feature_maps


def shift_scale_uncond_hidden_states(self):
    from gen.utils.tokenization_utils import get_uncond_tokens

    uncond_input_ids = get_uncond_tokens(self.tokenizer, "A photo of").to(self.device)
    uncond_encoder_hidden_states = self.text_encoder(input_ids=uncond_input_ids[None]).last_hidden_state.to(dtype=self.dtype).squeeze(0)
    return uncond_encoder_hidden_states


def attention_masking(batch: InputData, cond: Optional[ConditioningData]):
    assert "cross_attention_kwargs" in cond.unet_kwargs
    assert "attn_meta" in cond.unet_kwargs["cross_attention_kwargs"]

    batch_size: int = batch.disc_pixel_values.shape[0]
    device: torch.device = batch.disc_pixel_values.device
    gen_seg_ = rearrange(batch.gen_segmentation, "b h w c -> b c () h w").float()
    learnable_idxs = (batch.formatted_input_ids == cond.placeholder_token).nonzero(as_tuple=True)
    h, w = 64, 64  # Max latent size in U-Net

    all_masks = []
    for batch_idx in range(batch_size):
        cur_batch_mask = learnable_idxs[0] == batch_idx  # Find the masks for this batch
        token_indices = learnable_idxs[1][cur_batch_mask]
        segmentation_indices = cond.mask_instance_idx[cur_batch_mask]

        GT_masks = F.interpolate(input=gen_seg_[batch_idx][segmentation_indices], size=(h, w)).squeeze(1) > 0.5
        tensor_nhw = torch.zeros(batch.formatted_input_ids.shape[-1], h, w, dtype=torch.bool)

        for i, d in enumerate(token_indices):
            tensor_nhw[d] = GT_masks[i]

        tensor_nhw[:, (~tensor_nhw.any(dim=0))] = True
        all_masks.append(tensor_nhw.to(device))

    cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].cross_attention_mask = torch.stack(all_masks, dim=0).to(dtype=torch.bool)


def handle_attention_masking_dropout(cond: Optional[ConditioningData], dropout_idx):
    attn_mask = cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].cross_attention_mask
    attn_mask[dropout_idx] = torch.ones((77, 64, 64)).to(device=cond.encoder_hidden_states.device, dtype=torch.bool)


def handle_attn_proc_cross_attn_masking(attn_meta, hidden_states, attn, query, batch_size):
    attention_mask = attn_meta.cross_attention_mask
    resized_attention_masks = []
    latent_dim = round(math.sqrt(hidden_states.shape[1]))
    for b in range(attention_mask.shape[0]):
        resized_attention_masks.append(find_true_indices_batched(original=attention_mask[b], dh=latent_dim, dw=latent_dim))
    resized_attention_masks = torch.stack(resized_attention_masks).to(hidden_states.device)
    resized_attention_masks = rearrange(resized_attention_masks, "b tokens h w -> b (h w) tokens")
    if resized_attention_masks.shape[0] != batch_size:
        # For CFG, we batch [uncond, cond]. To make it simpler, we correct the attention mask here [instead of in the pipeline code]
        assert batch_size % 2 == 0, "Batch size of the attention mask is incorrect"
        # TODO: This is very risky. We assume that the first half is always uncond and we may need to repeat the cond at the end
        # Essentially, CFG must always be enabled.
        if (cond_batch_size := (batch_size // resized_attention_masks.shape[0]) // 2) > 1:
            resized_attention_masks = resized_attention_masks.repeat_interleave(cond_batch_size, dim=0)
        resized_attention_masks = torch.cat([resized_attention_masks.new_full(resized_attention_masks.shape, True), resized_attention_masks], dim=0)

    attention_mask = resized_attention_masks.repeat_interleave(attn.heads, dim=0)
    attention_mask = (1 - attention_mask.to(query)) * -10000.0
    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((*attention_mask.shape[:2], 3))], dim=-1).contiguous()
    attention_mask = attention_mask[..., :77]  # See: https://github.com/facebookresearch/xformers/issues/683
    return attention_mask
