import abc

import cv2
import numpy as np
import torch

from PIL import Image
from typing import Union, Tuple, List, Dict, Optional
import torch.nn.functional as nnf
from diffusers.models.cross_attention import CrossAttention
import torch.nn.functional as F

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def register_attention_control(unet, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id
            ]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = P2PCrossAttnProcessor(
            controller=controller, place_in_unet=place_in_unet
        )

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

def get_average_attention(controller: AttentionStore):
    average_attention = {
        key: [
            item / controller.cur_step
            for item in controller.attention_store[key]
        ]
        for key in controller.attention_store
    }
    return average_attention

def aggregate_attention(
    controller: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, batch_size: int
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    batch_size, -1, res, res, item.shape[-1]
                )[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out
    

def break_a_scene_loss(batch: dict, controller: AttentionStore):
    attn_loss = 0
    batch_size: int = batch["disc_pixel_values"].shape[0]
    for batch_idx in range(batch_size):
        GT_masks = F.interpolate(
            input=batch["instance_masks"][batch_idx], size=(16, 16)
        )
        agg_attn = aggregate_attention(
            controller,
            res=16,
            from_where=("up", "down"),
            is_cross=True,
            select=batch_idx,
            batch_size=batch_size
        )
        curr_cond_batch_idx = batch_size + batch_idx

        for mask_id in range(len(GT_masks)):
            curr_placeholder_token_id = self.placeholder_token_ids[
                batch["token_ids"][batch_idx][mask_id]
            ]

            asset_idx = (
                (
                    batch["input_ids"][curr_cond_batch_idx]
                    == curr_placeholder_token_id
                )
                .nonzero()
                .item()
            )
            asset_attn_mask = agg_attn[..., asset_idx]
            asset_attn_mask = (
                asset_attn_mask / asset_attn_mask.max()
            )
            attn_loss += F.mse_loss(
                GT_masks[mask_id, 0].float(),
                asset_attn_mask.float(),
                reduction="mean",
            )

    attn_loss = self.args.lambda_attention * (
        attn_loss / self.args.train_batch_size
    )

    self.controller.attention_store = {}
    self.controller.cur_step = 0