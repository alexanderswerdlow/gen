from collections import defaultdict
import math
from typing import Callable, Optional, Union

import torch
import xformers
import xformers.ops
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from einx import rearrange, mean

from gen.models.cross_attn.base_model import AttentionMetadata
from gen.models.cross_attn.deprecated_configs import handle_attn_proc_cross_attn_masking
from gen.models.cross_attn.eschernet import cape_embed_4dof, cape_embed_6dof
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

@torch.no_grad()
def handle_attn_proc_self_attn_masking(attn_meta, hidden_states, attn, query, batch_size):
    latent_dim = round(math.sqrt(hidden_states.shape[1]))
    if latent_dim == 64:
        return None
    attention_mask = attn_meta.self_attention_mask[latent_dim]
    attention_mask = torch.stack([(((attention_mask_ @ attention_mask_.T)) > 0).detach() for attention_mask_  in attention_mask])
    if attention_mask.shape[0] != batch_size:
        # For CFG, we batch [uncond, cond]. To make it simpler, we correct the attention mask here [instead of in the pipeline code]
        assert batch_size % 2 == 0, "Batch size of the attention mask is incorrect"
        # TODO: This is very risky. We assume that the first half is always uncond and we may need to repeat the cond at the end
        # Essentially, CFG must always be enabled.
        if (cond_batch_size := (batch_size // attention_mask.shape[0]) // 2) > 1:
            attention_mask = attention_mask.repeat_interleave(cond_batch_size, dim=0)
        attention_mask = torch.cat([attention_mask.new_full(attention_mask.shape, True), attention_mask], dim=0)

    attention_mask = ((~attention_mask).to(query) * -10000.0)
    attention_mask = attention_mask.view(attention_mask.shape[0], 1, *attention_mask.shape[1:]).expand(attention_mask.shape[0], attn.heads, *attention_mask.shape[1:]).contiguous().view(-1, *attention_mask.shape[1:])
    return attention_mask

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
        is_eschernet = attn_meta.posemb is not None
        is_cross_attn = encoder_hidden_states is not None

        if is_eschernet:
            # turn 2d attention into multiview attention
            is_6dof = len(attn_meta.posemb) == 4
            if is_6dof:
                [p_out, p_out_inv, p_in, p_in_inv] = attn_meta.posemb
                t_out, t_in = 1, 1  # t size
            else:
                p_out, p_in = attn_meta.posemb # BxTx4, # BxTx4
                t_out, t_in = p_out.shape[1], p_in.shape[1]
        
        is_trainable_cross_attn = False
        if encoder_hidden_states is not None and attn_meta is not None:
            training_gated_attn = attn_meta.gate_scale is not None
            if training_gated_attn:
                # We split the cond between the gated cross-attn and the frozen cross-attn
                encoder_hidden_states, frozen_encoder_hidden_states = rearrange('b t (d + f) -> b t d, b t f', encoder_hidden_states, f=attn_meta.frozen_dim)

            if can_override_encoder_hidden_states and training_gated_attn:
                encoder_hidden_states = frozen_encoder_hidden_states
            else:
                # TODO: We assume that we *always* call all cross-attn layers in order, and that we never skip any.
                # This makes it easier for e.g., inference so we don't need to reset the counter, but is pretty hacky.
                is_trainable_cross_attn = True
                cur_idx = attn_meta.layer_idx % attn_meta.num_layers
                if attn_meta.custom_map is not None:
                    cond_idx = attn_meta.custom_map[cur_idx]
                else:
                    cond_idx = min(cur_idx, (attn_meta.num_layers - 1) - cur_idx)
                encoder_hidden_states = encoder_hidden_states.chunk(attn_meta.num_cond_vectors, dim=-1)[cond_idx]
                attn_meta.layer_idx = attn_meta.layer_idx + 1
                
                if attn_meta.add_pos_emb:
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

        # apply 4DoF CaPE
        if is_eschernet and is_cross_attn: # We only have single-view prediction for now.
            if is_6dof:
                import einops
                p_in = einops.rearrange(p_in, "b f g -> b () f g")
                p_out = einops.rearrange(p_out, "b f g -> b () f g")
                p_out_inv = einops.rearrange(p_out_inv, "b f g -> b () f g")

                p_out_inv = einops.repeat(p_out_inv, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)  # query shape
                if is_cross_attn:
                    p_in = einops.repeat(p_in, 'b t_in f g -> b (t_in l) f g', l=key.shape[1] // t_in)  # key shape
                else:
                    p_in = einops.repeat(p_out, 'b t_out f g -> b (t_out l) f g', l=query.shape[1] // t_out)  # query shape
                
                # To debug gradients:
                # p_out_inv = torch.eye(4)[None][None].repeat(p_out_inv.shape[0], p_out_inv.shape[1], 1, 1)
                # p_in = torch.eye(4)[None][None].repeat(p_in.shape[0], p_in.shape[1], 1, 1)

                query = cape_embed_6dof(query, p_out_inv)  # query f_q @ (p_out)^(-T) .permute(0, 1, 3, 2)
                key = cape_embed_6dof(key, p_in)  # key f_k @ p_in
            else:
                p_out = rearrange('b t_out d -> b (t_out l) d', p_out, l=query.shape[1] // t_out)  # query shape
                if is_cross_attn:
                    p_in = rearrange('b t_in d -> b (t_in l) d', p_in, l=key.shape[1] // t_in)  # key shape
                else:
                    p_in = p_out
                query, key = cape_embed_4dof(p_out, p_in, query, key)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # START MODIFICATION
        if is_cross_attn and attn_meta is not None and attn_meta.cross_attention_mask is not None:
            attention_mask = handle_attn_proc_cross_attn_masking(attn_meta, hidden_states, attn, query, batch_size)
        elif is_cross_attn is False and attn_meta is not None and attn_meta.self_attention_mask is not None:
            assert attention_mask is None
            attention_mask = handle_attn_proc_self_attn_masking(attn_meta, hidden_states, attn, query, batch_size)

        if is_trainable_cross_attn and attn_meta.return_attn_probs is True:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

            if attn_meta.attn_probs is None:
                attn_meta.attn_probs = defaultdict(list)
                
            attn_meta.attn_probs[cur_idx].append(mean("(b [heads]) d tokens -> b d tokens", attention_probs, b=batch_size))
        else:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        # END MODIFICATION

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if is_eschernet:
            # reshape back
            hidden_states = rearrange('b (t_out l) d -> (b t_out) l d', hidden_states, t_out=t_out)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
