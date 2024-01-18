import random
from dataclasses import dataclass
from re import M
from typing import List, Optional

import torch
import torch.nn.functional as F
from flash_attn.modules.mha import MHA
from torch import nn
from gen.configs.base import BaseConfig
from gen.models.neti.decoder import DecoderTransformer
from accelerate.utils import PrecisionType
from gen.models.neti.positional_encoding import BasicEncoder, FixedPositionalEncoding, NeTIPositionalEncoding

import hydra

from gen.models.utils import FourierPositionalEncodingNDims, SinusoidalPosEmb


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float


UNET_LAYERS = ["IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "MID", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]


def _init_weights(m):
    initializer_range=0.02
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=initializer_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=initializer_range)

class CrossAttn(nn.Module):
    def __init__(self, cfg: BaseConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.neti_up_proj = nn.Sequential(nn.Linear(input_dim, self.cfg.model.cross_attn_dim), nn.LayerNorm(self.cfg.model.cross_attn_dim))
        self.decoder = hydra.utils.instantiate(
            self.cfg.model.decoder_transformer, _recursive_=False, embed_dim=self.cfg.model.cross_attn_dim, use_flash_attn=self.cfg.trainer.mixed_precision != "no"
        )
        self.cross_attn_proj = nn.Sequential(nn.Linear(self.cfg.model.cross_attn_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, **kwargs):
        attn_dict = kwargs.get("attn_dict")
        x = self.neti_up_proj(attn_dict["x"])

        del attn_dict["x"]

        output = self.decoder(x, mixer_kwargs=attn_dict)
        output = self.cross_attn_proj(output)

        return output


class NeTIMapper(nn.Module):
    """Main logic of our NeTI mapper."""

    def __init__(
        self,
        cfg: BaseConfig,
        output_dim: int,
        unet_layers: List[str] = UNET_LAYERS,
        use_nested_dropout: bool = True,
        nested_dropout_prob: float = 0.5,
        norm_scale: Optional[torch.Tensor] = None,
        use_positional_encoding: bool = True,
        num_pe_time_anchors: int = 10,
        pe_sigmas: PESigmas = PESigmas(sigma_t=0.03, sigma_l=2.0),
        output_bypass: bool = True,
    ):
        super().__init__()
        self.orig_output_dim = output_dim
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_prob = nested_dropout_prob
        self.norm_scale = norm_scale
        self.output_bypass = output_bypass
        self.cfg = cfg
        if self.output_bypass:
            output_dim *= 2  # Output two vectors

        self.use_positional_encoding = use_positional_encoding

        if self.cfg.model.use_single_token:
            self.mapper = nn.Sequential(nn.Linear(self.cfg.model.cross_attn_dim, self.orig_output_dim))
        else:
            if self.cfg.model.use_fixed_position_encoding:
                self.encoder = FourierPositionalEncodingNDims(dim=output_dim, sigmas=[pe_sigmas.sigma_t, pe_sigmas.sigma_l])
                self.learnable_token = nn.Parameter(torch.randn(output_dim))
            elif self.use_positional_encoding:
                self.encoder = NeTIPositionalEncoding(sigma_t=pe_sigmas.sigma_t, sigma_l=pe_sigmas.sigma_l).cuda()
                self.input_dim = num_pe_time_anchors * len(unet_layers)
            else:
                self.encoder = BasicEncoder().cuda()
                self.input_dim = 2

            if not self.cfg.model.use_fixed_position_encoding:
                self.set_net(num_unet_layers=len(unet_layers), num_time_anchors=num_pe_time_anchors, output_dim=output_dim)

            if self.cfg.model.mask_cross_attn:
                self.cross_attn = CrossAttn(cfg=cfg, input_dim=self.orig_output_dim, output_dim=output_dim)

        self.apply(_init_weights)
        
    def set_net(self, num_unet_layers: int, num_time_anchors: int, output_dim: int = 768):
        self.input_layer = self.set_input_layer(num_unet_layers, num_time_anchors)
        self.net = nn.Sequential(
            self.input_layer,
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        self.output_layer = nn.Sequential(nn.Linear(128, output_dim))

    def set_input_layer(self, num_unet_layers: int, num_time_anchors: int) -> nn.Module:
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w * 2, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(num_time_anchors, num_unet_layers)
        else:
            input_layer = nn.Identity()
        return input_layer

    def forward(self, timestep: torch.Tensor, unet_layer: torch.Tensor, truncation_idx: int = None) -> torch.Tensor:
        if self.cfg.model.use_fixed_position_encoding:
            embedding = self.learnable_token[None, :] + self.encoder(torch.stack((timestep, unet_layer), dim=-1))
        else:
            embedding = self.extract_hidden_representation(timestep, unet_layer)
            if self.use_nested_dropout:
                embedding = self.apply_nested_dropout(embedding, truncation_idx=truncation_idx)
            embedding = self.get_output(embedding)
        
        return embedding

    def get_encoded_input(self, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(timestep, unet_layer)

    def extract_hidden_representation(self, timestep: torch.Tensor, unet_layer: torch.Tensor) -> torch.Tensor:
        encoded_input = self.get_encoded_input(timestep, unet_layer)
        embedding = self.net(encoded_input)
        return embedding

    def apply_nested_dropout(self, embedding: torch.Tensor, truncation_idx: int = None) -> torch.Tensor:
        if self.training:
            if random.random() < self.nested_dropout_prob:
                dropout_idxs = torch.randint(low=0, high=embedding.shape[1], size=(embedding.shape[0],))
                for idx in torch.arange(embedding.shape[0]):
                    embedding[idx][dropout_idxs[idx] :] = 0
        if not self.training and truncation_idx is not None:
            for idx in torch.arange(embedding.shape[0]):
                embedding[idx][truncation_idx:] = 0
        return embedding

    def get_output(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.output_layer(embedding)
        if self.norm_scale is not None:
            embedding = F.normalize(embedding, dim=-1) * self.norm_scale
        return embedding
