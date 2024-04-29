from typing import Callable, Optional, Union

import torch
import xformers
import xformers.ops
from diffusers.models.attention_processor import Attention
from einx import rearrange, mean

from gen.models.base.base_defs import AttentionMetadata

xformers_available = True
def register_custom_attention(unet):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        if not (name.startswith("mid_block") or name.startswith("up_blocks") or name.startswith("down_blocks")):
            continue

        if ".attn2." in name or '.fuser.' in name:  # Cross-attn layers are named 'attn2' or 'fuser' for gated cross-attn
            cross_att_count += 1

        attn_procs[name] = XFormersAttnProcessor()

    unet.set_attn_processor(attn_procs)

    global xformers_available
    try:
        _ = xformers.ops.memory_efficient_attention(
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
            torch.randn((1, 2, 40), device="cuda"),
        )
    except Exception as e:
        xformers_available = False
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
        attn_meta: Optional[AttentionMetadata] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

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

        is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        is_dual_cross_attention = attn.attention_config is not None and (getattr(attn.attention_config, "dual_self_attention", False) and is_cross_attention)
        is_dual_self_attention = attn.attention_config is not None and (getattr(attn.attention_config, "dual_cross_attention", False) and is_cross_attention is False)
        if is_dual_cross_attention or is_dual_self_attention:
            left_state, right_state = hidden_states.chunk(2, dim=0)
            cond_states = ((left_state, right_state), (right_state, left_state)) if is_dual_cross_attention else ((left_state, left_state), (right_state, right_state))
            cond_modules = ((attn.to_q, attn.to_k, attn.to_v, attn.to_out, attn.v_2_to_v_1 if is_dual_cross_attention else None), (attn.v2_to_q, attn.v2_to_k, attn.v2_to_v, attn.v2_to_out, attn.v_1_to_v_2 if is_dual_cross_attention else None))
            combined_states = []

            for (self_state, cross_state), (_to_q, _to_k, _to_v, _to_out, _to_self) in zip(cond_states, cond_modules):
                query = _to_q(self_state)
                cross_state = _to_self(cross_state) if is_dual_cross_attention else cross_state
                key = _to_k(cross_state)
                value = _to_v(cross_state)

                query = attn.head_to_batch_dim(query).contiguous()
                key = attn.head_to_batch_dim(key).contiguous()
                value = attn.head_to_batch_dim(value).contiguous()

                if xformers_available is False:
                    attention_probs = attn.get_attention_scores(query, key, attention_mask)
                    hidden_states = torch.bmm(attention_probs, value)
                else:
                    hidden_states = xformers.ops.memory_efficient_attention(
                        query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                    )

                hidden_states = hidden_states.to(query.dtype)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = _to_out[0](hidden_states)
                # dropout
                hidden_states = _to_out[1](hidden_states)

                combined_states.append(hidden_states)

            hidden_states = torch.cat(combined_states, dim=0)
        else:
            if attn_meta.joint_attention is not None and is_cross_attention:
                encoder_hidden_states = rearrange(
                    '(views b) n c -> (repeat_views b) (views n) c',
                    hidden_states,
                    views=attn_meta.joint_attention,
                    repeat_views=attn_meta.joint_attention
                )
                encoder_hidden_states = attn.to_cross(encoder_hidden_states)

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            if xformers_available is False:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)
            else:
                hidden_states = xformers.ops.memory_efficient_attention(
                    query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                )

            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states