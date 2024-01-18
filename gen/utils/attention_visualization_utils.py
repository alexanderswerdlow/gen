from collections import defaultdict
import os
import math
from typing import Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0, LoRAAttnProcessor, LoRAAttnProcessor2_0

from gen.models.neti.xti_attention_processor import XTIAttenProc
import math

from gen.utils.logging_utils import log_info

attn_maps = defaultdict(list)
hooks = []

def attn_call(
    self,
    attn: Attention,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
    scale=1.0,
):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ####################################################################################################
    # (20,4096,77) or (40,1024,77)
    if hasattr(self, "store_attn_map"):
        self.attn_map = attention_probs
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
    self,
    attn: Attention,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
    scale: float = 1.0,
):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    # if self.store_attn_map:
    if hasattr(self, "store_attn_map"):
        hidden_states, attn_map = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        # (2,10,4096,77) or (2,20,1024,77)
        self.attn_map = attn_map
    else:
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def cross_attn_init():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call  # attn_call is faster
    # AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    # LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    LoRAAttnProcessor2_0.__call__ = lora_attn_call


def reshape_attn_map(attn_map):
    attn_map = torch.mean(attn_map, dim=0)  # mean by head dim: (20,4096,77) -> (4096,77)
    attn_map = attn_map.permute(1, 0)  # (4096,77) -> (77,4096)
    latent_size = int(math.sqrt(attn_map.shape[1]))
    latent_shape = (attn_map.shape[0], latent_size, -1)
    attn_map = attn_map.reshape(latent_shape)  # (77,4096) -> (77,64,64)

    return attn_map  # torch.sum(attn_map,dim=0) = [1,1,...,1]


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name].append(module.processor.attn_map)
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    global hooks
    for name, module in unet.named_modules():
        if not name.split(".")[-1].startswith("attn2"):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, XTIAttenProc):
            module.processor.store_attn_map = True

        # log_info(f'registering hook for {name}')

        hook = module.register_forward_hook(hook_fn(name))
        hooks.append(hook)

    return unet

def unregister_cross_attention_hook(unet):
    global hooks
    for name, module in unet.named_modules():
        if not name.split(".")[-1].startswith("attn2"):
            continue

        if isinstance(module.processor, AttnProcessor) or isinstance(module.processor, AttnProcessor2_0) or isinstance(module.processor, LoRAAttnProcessor) or isinstance(module.processor, LoRAAttnProcessor2_0) or isinstance(module.processor, XTIAttenProc):
            module.processor.store_attn_map = False

    for hook in hooks:
        hook.remove()

    hooks = []
    return unet


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


def retrieve_attn_maps_per_timestep(image_size, timesteps, detach=True, chunk=True) -> list[list[torch.Tensor]]:
    """
    Returns the attention maps at each timestep, for each input token
    """
    global attn_maps

    target_size = (image_size[0] // 8, image_size[1] // 8)
    attn_maps_per_timestep = []

    for t in range(timesteps):
        attn_maps_layers = []
        for _, attn_maps_layer_ in attn_maps.items():
            # Make sure we have saved maps at each timestep
            assert len(attn_maps_layer_) == timesteps

            attn_maps_layer_ = attn_maps_layer_[t]
            attn_maps_layer_ = attn_maps_layer_.detach().cpu() if detach else attn_maps_layer_

            # For CFG, we need to chop off the unconditional tokens [first half by convention]. First dim is (batch * heads).
            if chunk:
                attn_maps_layer_ = torch.chunk(attn_maps_layer_, 2)[1]  # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
            if len(attn_maps_layer_.shape) == 4:
                attn_maps_layer_ = attn_maps_layer_.squeeze()

            attn_maps_layer_ = mean_and_scale(attn_maps_layer_, target_size)  # (10,32*32,77) -> (77,64*64)
            attn_maps_layers.append(attn_maps_layer_)

        attn_maps_layers = torch.mean(torch.stack(attn_maps_layers, dim=0), dim=0) # (77,64*64)
        latent_size = int(math.sqrt(attn_maps_layers.shape[1]))
        attn_maps_per_timestep.append(attn_maps_layers.reshape(attn_maps_layers.shape[0], latent_size, latent_size)) # (77,64*64) -> (77,64,64)

    attn_maps = defaultdict(list)
    return attn_maps_per_timestep


def mean_and_scale(attn_map, target_size):
    """
    Average over the heads, rescale to resolution, and softmax over tokens. This contains the attention map for one layer and timestep.
    """
    attn_map = torch.mean(attn_map, dim=0)  # (10, 32*32, 77) -> (32*32, 77)
    attn_map = attn_map.permute(1, 0)  # (32*32, 77) -> (77, 32*32)

    if target_size[0] * target_size[1] != attn_map.shape[1]:
        temp_size = (int(math.sqrt(attn_map.shape[1])), int(math.sqrt(attn_map.shape[1])))
        attn_map = attn_map.view(attn_map.shape[0], *temp_size)  # (77, 32,32)
        attn_map = attn_map.unsqueeze(0)  # (77,32,32) -> (1,77,32,32)
        attn_map = F.interpolate(attn_map.to(dtype=torch.float32), size=target_size, mode="bilinear", align_corners=False).squeeze()  # (77,64,64)
    else:
        attn_map = attn_map.to(dtype=torch.float32)  # (77,64,64)

    attn_map = torch.softmax(attn_map, dim=0)
    attn_map = attn_map.reshape(attn_map.shape[0], -1)  # (77,64*64)
    return attn_map


def get_all_net_attn_maps(attn_maps_per_timestep, tokens):
    """
    Returns the attention maps at each timestep.
    Each pixel is normalized over all tokens [e.g., summing over then token dimension == 1]
    We then normalize the entire image by the max/min values.
    """
    attn_maps_img_by_timestep = []
    for attn_maps_single_timestep in attn_maps_per_timestep: # [(77,64,64), (77,64,64), ...)]
        total_attn_scores = 0
        attn_maps_img = []
        # attn_maps_single_timestep = attn_maps_single_timestep[:len(tokens)].softmax(dim=0)
        min_, max_ = torch.min(attn_maps_single_timestep).item(), torch.max(attn_maps_single_timestep).item()
        for i, (token, attn_map_single_token) in enumerate(zip(tokens, attn_maps_single_timestep)):
            attn_maps_img.append(get_attn_map_img(attn_map_single_token.cpu().numpy(), norm=None)) # (min_, max_)
            attn_map_score = torch.sum(attn_map_single_token)
            h, w = attn_map_single_token.shape
            attn_map_total = h * w
            attn_map_score = attn_map_score / attn_map_total
            total_attn_scores += attn_map_score
        attn_maps_img_by_timestep.append(attn_maps_img)

    log_info(f"total_attn_scores: {total_attn_scores}, tokens: {len(tokens)}, attn_maps_img: {len(attn_maps_single_timestep)}")

    return attn_maps_img_by_timestep


def save_net_attn_map(net_attn_maps, dir_name, tokenizer, tokens):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # tokens = prompt2tokens(tokenizer, prompt)
    total_attn_scores = 0
    attn_maps_img = []
    for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()
        h, w = attn_map.shape
        attn_map_total = h * w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score
        token = token.replace("</w>", "")
        image = save_attn_map_img(attn_map, f"{token}:{attn_map_score:.2f}", f"{dir_name}/{i}_{token}:{int(attn_map_score*100)}.png")
    log_info(f"total_attn_scores: {total_attn_scores}, tokens: {len(tokens)}, attn_maps_img: {len(net_attn_maps)}")


def resize_net_attn_map(attn_map_by_timestep, target_size):
    return [
        F.interpolate(net_attn_maps.to(dtype=torch.float32).unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze()
        for net_attn_maps in attn_map_by_timestep  # (77,64,64)
    ]


def save_attn_map_img(attn_map, title, save_path):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map)
    image.save(save_path, format="PNG", compression=0)


def get_attn_map_img(attn_map, norm: Optional[tuple[float]] = None):
    if norm is not None:
        normalized_attn_map = (attn_map - norm[0]) / (norm[1] - norm[0]) * 255
    else:
        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map).convert("RGB")
    return image
    # image.save(save_path, format='PNG', compression=0)
