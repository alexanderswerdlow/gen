from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from image_utils import Im
from PIL import Image

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import ConditioningData
    from gen.configs.base import BaseConfig

def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
) -> Image.Image:
    """Displays a list of images in a grid."""
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    return pil_img


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 0.002 * min(h, w), 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 0.002 * min(h, w), 2, text_color, int(max(1, round(min(h, w) / 150))))
    return img


def save_cross_attention_vis(tokenizer, tokens, attention_maps, res=(64, 64)):
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().float().cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize(res))
        image = Im(image).write_text(tokenizer.decode(int(tokens[i])), relative_font_scale=0.004).np
        images.append(image)
    vis = view_images(np.stack(images, axis=0))
    return vis


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
            attn_ = attn.clone()
            h = attn.shape[0]
            attn_[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn_

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

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
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
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


def register_attention_control(controller, unet):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = P2PCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def get_average_attention(controller: AttentionStore):
    average_attention = {key: [item / controller.cur_step for item in controller.attention_store[key]] for key in controller.attention_store}
    return average_attention


def aggregate_attention(controller: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, batch_size: int):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(batch_size, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


def break_a_scene_cross_attn_loss(cfg: BaseConfig, batch: dict, controller: AttentionStore, conditioning_data: ConditioningData):
    attn_loss = 0
    batch_size: int = batch["disc_pixel_values"].shape[0]
    gen_seg_ = rearrange(batch["gen_segmentation"], "b h w c -> b c () h w").float()
    learnable_idxs = (batch["input_ids"] == conditioning_data.placeholder_token).nonzero(as_tuple=True)

    for batch_idx in range(batch_size):
        GT_masks = F.interpolate(input=gen_seg_[batch_idx], size=(16, 16))  # We interpolate per mask separately
        agg_attn = aggregate_attention(controller, res=16, from_where=("up", "down"), is_cross=True, select=batch_idx, batch_size=batch_size)

        cur_batch_mask = learnable_idxs[0] == batch_idx  # Find the masks for this batch
        token_indices = learnable_idxs[1][cur_batch_mask]

        segmentation_indices = conditioning_data.mask_instance_idx[
            cur_batch_mask
        ]  # We may dropout masks so we need to map between dataset segmentation idx and the mask idx in the sentence
        attn_masks = agg_attn[..., token_indices]

        for idx, mask_id in enumerate(segmentation_indices):
            asset_attn_mask = attn_masks[..., idx] / attn_masks[..., idx].max()
            attn_loss += F.mse_loss(
                GT_masks[mask_id, 0].float(),
                asset_attn_mask.float(),
                reduction="mean",
            )

    attn_loss = cfg.model.break_a_scene_cross_attn_loss_weight * (attn_loss / batch_size)
    controller.reset()
    return attn_loss


def remove_element(tensor, row_index):
    return torch.cat((tensor[..., :row_index], tensor[..., row_index + 1 :]), dim=-1)


def break_a_scene_masked_loss(
    cfg: BaseConfig,
    batch: dict,
    conditioning_data: ConditioningData
):
    max_masks = []
    for b in range(batch['gen_pixel_values'].shape[0]):
        mask_idxs_for_batch = conditioning_data.mask_instance_idx[conditioning_data.mask_batch_idx == b]
        object_masks = batch["gen_segmentation"][b, ..., mask_idxs_for_batch]
        if object_masks.shape[-1] == 0:
            max_masks.append(object_masks.new_zeros((512, 512))) # Zero out loss if there are no masks
        else:
            max_masks.append(torch.max(object_masks, axis=-1).values)

    max_masks = torch.stack(max_masks, dim=0)[:, None]
    downsampled_mask = F.interpolate(input=max_masks.float(), size=(64, 64))
    return downsampled_mask
