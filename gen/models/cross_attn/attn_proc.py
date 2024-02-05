import math
from typing import Callable, Optional, Union

import torch
import xformers
import xformers.ops
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from einx import rearrange

from gen.models.cross_attn.base_model import AttentionMetadata
from gen.models.cross_attn.deprecated_configs import handle_attn_proc_masking
from gen.models.utils import positionalencoding2d

def register_layerwise_attention(unet):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        if not (name.startswith("mid_block") or name.startswith("up_blocks") or name.startswith("down_blocks")):
            continue

        if ".attn2." in name or '.fuser.' in name:  # Cross-attn layers are named 'attn2' or 'fuser' for gated cross-attn
            cross_att_count += 1

        attn_procs[name] = XFormersAttnProcessor()

    unet.set_attn_processor(attn_procs)
    return cross_att_count


class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        attn_meta: Optional[AttentionMetadata] = None,
        can_override_encoder_hidden_states: bool = True, # With gated cross-attn, sometimes we want original conditioning from text-encoder
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # START MODIFICATION
        is_cross_attn = encoder_hidden_states is not None
        if encoder_hidden_states is not None and attn_meta is not None:
            is_frozen_after_gate = attn_meta.get("gate_scale", None) is not None
            if is_frozen_after_gate:
                frozen_dim = attn_meta.get("frozen_dim", None)
                encoder_hidden_states, frozen_encoder_hidden_states = rearrange('b t (d + f) -> b t d, b t f', encoder_hidden_states, f=frozen_dim)

            if can_override_encoder_hidden_states and is_frozen_after_gate:
                encoder_hidden_states = frozen_encoder_hidden_states
            else:
                # TODO: We assume that we *always* call all cross-attn layers in order, and that we never skip any.
                # This makes it easier for e.g., inference so we don't need to reset the counter, but is pretty hacky.
                cur_idx = attn_meta["layer_idx"] % attn_meta["num_layers"]
                cond_idx = min(cur_idx, (attn_meta["num_layers"] - 1) - cur_idx)
                encoder_hidden_states = encoder_hidden_states.chunk(attn_meta["num_cond_vectors"], dim=-1)[cond_idx]
                attn_meta["layer_idx"] = attn_meta["layer_idx"] + 1
                
                if attn_meta["add_pos_emb"]:
                    h, w = round(math.sqrt(hidden_states.shape[1])), round(math.sqrt(hidden_states.shape[1]))
                    pos_emb = positionalencoding2d(hidden_states.shape[-1], h, w, device=hidden_states.device, dtype=hidden_states.dtype)
                    hidden_states = hidden_states + rearrange("d h w -> () (h w) d", pos_emb)
        # END MODIFICATION

        batch_size, key_tokens, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # START MODIFICATION
        if is_cross_attn and attn_meta is not None and "attention_mask" in attn_meta:
            attention_mask = handle_attn_proc_masking(attn_meta, hidden_states, attn, query, batch_size)
        # END MODIFICATION

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
