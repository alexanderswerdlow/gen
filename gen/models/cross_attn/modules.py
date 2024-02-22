from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
from torch import Tensor, nn
from einx import rearrange, softmax
import einops

from gen.models.utils import _init_weights
from gen.utils.logging_utils import log_info
from gen.models.cross_attn.decoder import create_mlp_cls
from jaxtyping import Bool, Float, Integer
import math

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import InputData
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import ConditioningData
    from gen.models.cross_attn.base_model import TokenPredData

# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


try:
    from flash_attn.ops.activations import swiglu
except ImportError:
    swiglu = None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
except ImportError:
    ColumnParallelLinear, RowParallelLinear = None, None

try:
    from flash_attn.ops.fused_dense import FusedMLP, ParallelFusedMLP
except ImportError:
    FusedMLP, ParallelFusedMLP = None, None


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

# From DiT
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class FilmMlp(nn.Module):
    def __init__(
        self,
        in_features,
        cond_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4

        self.t_embedder = TimestepEmbedder(cond_features)

        self.activation = activation
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_features, **factory_kwargs)
        self.film1 = nn.Linear(cond_features, 2 * hidden_features, **factory_kwargs)

        self.fc2 = nn.Linear(hidden_features, hidden_features, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_features, **factory_kwargs)
        self.film2 = nn.Linear(cond_features, 2 * hidden_features, **factory_kwargs)

        self.fc3 = nn.Linear(hidden_features, out_features, **factory_kwargs)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.film1.weight, 0)
        nn.init.constant_(self.film1.bias, 0)
        nn.init.constant_(self.film2.weight, 0)
        nn.init.constant_(self.film2.bias, 0)

    def forward(self, x, t, cond):
        t_emb = self.t_embedder(t) # TODO: This is weird, out timestep embedding is 6 dims
        cond = cond + t_emb
        y = self.activation(self.norm1(self.fc1(x)))
        scale, shift = einops.rearrange(self.film1(cond), "b (n a) -> a b n", a=2)
        y = y * (1 - scale) + shift
        y = self.activation(self.norm2(self.fc2(y)))
        scale, shift = einops.rearrange(self.film2(cond), "b (n a) -> a b n", a=2)
        y = y * (1 - scale) + shift
        y = self.fc3(y)
        return y

class FilmMlpv2(nn.Module):
    def __init__(
        self,
        in_features,
        cond_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4

        self.t_embedder = TimestepEmbedder(256)

        self.activation = activation
        self.fc1 = nn.Linear(in_features + cond_features, hidden_features, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_features, **factory_kwargs)
        self.film1 = nn.Linear(256, 2 * hidden_features, **factory_kwargs)

        self.fc2 = nn.Linear(hidden_features, hidden_features, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_features, **factory_kwargs)
        self.film2 = nn.Linear(256, 2 * hidden_features, **factory_kwargs)

        self.fc3 = nn.Linear(hidden_features, out_features, **factory_kwargs)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.film1.weight, 0)
        nn.init.constant_(self.film1.bias, 0)
        nn.init.constant_(self.film2.weight, 0)
        nn.init.constant_(self.film2.bias, 0)

    def forward(self, x, t, cond):
        t_emb = self.t_embedder(t) # TODO: This is weird, out timestep embedding is 6 dims
        x = torch.cat((x, cond), dim=-1)
        y = self.activation(self.norm1(self.fc1(x)))
        scale, shift = einops.rearrange(self.film1(t_emb), "b (n a) -> a b n", a=2)
        y = y * (1 - scale) + shift
        y = self.activation(self.norm2(self.fc2(y)))
        scale, shift = einops.rearrange(self.film2(t_emb), "b (n a) -> a b n", a=2)
        y = y * (1 - scale) + shift
        y = self.fc3(y)
        return y
    
   
class TokenMapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # dim = self.cfg.model.token_embedding_dim * (self.cfg.model.num_conditioning_pairs if self.cfg.model.layer_specialization else 1)
        dim = self.cfg.model.token_embedding_dim
        if self.cfg.model.token_cls_pred_loss:
            self.cls_mlp = Mlp(in_features=dim, hidden_features=dim // 4, out_features=self.cfg.model.num_token_cls, activation=nn.GELU())

        if self.cfg.model.token_rot_pred_loss:
            if self.cfg.model.discretize_rot_pred:
                self.rot_mlp = Mlp(in_features=dim, hidden_features=dim // 2, out_features=(1 + self.cfg.model.discretize_rot_bins_per_axis) * 3, activation=nn.GELU())
            elif self.cfg.model.use_orig_film:
                self.rot_mlp = FilmMlp(in_features=6, cond_features=dim, out_features=6, activation=nn.GELU())
            elif self.cfg.model.use_larger_film:
                self.rot_mlp = FilmMlpv2(in_features=6, cond_features=dim, hidden_features=1024, out_features=6, activation=nn.GELU())
            else:
                self.rot_mlp = FilmMlpv2(in_features=6, cond_features=dim, hidden_features=256, out_features=6, activation=nn.GELU())

        self.apply(_init_weights)

    def forward(self, cond: ConditioningData, pred_data: TokenPredData):
        mask_tokens = cond.pred_mask_tokens
        if self.cfg.model.detach_mask_tokens_for_pred:
            mask_tokens = mask_tokens.detach()

        if self.cfg.model.token_cls_pred_loss:
            output = self.cls_mlp(mask_tokens)
            pred_data.cls_pred = output

        if self.cfg.model.token_rot_pred_loss:
            if self.cfg.model.discretize_rot_pred:
                pred = self.rot_mlp(mask_tokens)
                pred = rearrange("b (axes d) -> b axes d", pred, axes=3)
            else:
                pred = self.rot_mlp(pred_data.noised_rot_6d, pred_data.timesteps, mask_tokens)
            pred_data.pred_6d_rot = pred

        return pred_data


class FeatureMapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.cross_attn = CrossAttn(
            cfg=cfg, input_dim=self.cfg.model.encoder_dim, cross_attn_dim=self.cfg.model.cross_attn_dim, output_dim=cfg.model.token_embedding_dim
        )

        self.learnable_token = nn.Parameter(
            torch.randn(cfg.model.num_conditioning_pairs if self.cfg.model.per_layer_queries else 1, cfg.model.cross_attn_dim) * 0.02
        )

        # If we have per layer queries, we don't need to chop up the mask vector
        if self.cfg.model.layer_specialization and not self.cfg.model.per_layer_queries:
            self.layer_specialization = nn.Sequential(
                nn.Linear(cfg.model.token_embedding_dim // self.cfg.model.num_conditioning_pairs, cfg.model.token_embedding_dim),
                nn.LayerNorm(cfg.model.token_embedding_dim),
            )

        if self.cfg.model.feature_map_keys is not None:
            self.position_embedding = nn.Parameter(torch.randn(len(self.cfg.model.feature_map_keys), self.cfg.model.encoder_dim) * 0.02)  #

        # TODO: Double check this is working
        self.apply(_init_weights)


class CrossAttn(nn.Module):
    def __init__(self, cfg: BaseConfig, input_dim: int, cross_attn_dim: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.kv_up_proj = None
        if input_dim != cross_attn_dim:
            log_info("CrossAttn: input_dim != cross_attn_dim. Using a projection layer.")
            self.kv_up_proj = nn.Sequential(nn.Linear(input_dim, cross_attn_dim), nn.LayerNorm(cross_attn_dim))

        self.decoder = hydra.utils.instantiate(
            self.cfg.model.decoder_transformer,
            _recursive_=False,
            embed_dim=cross_attn_dim,
            use_flash_attn=self.cfg.trainer.mixed_precision != "no",
        )
        self.cross_attn_output_proj = nn.Sequential(nn.Linear(cross_attn_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, conditioning_data: ConditioningData):
        attn_dict = conditioning_data.attn_dict

        if self.kv_up_proj is not None:
            attn_dict["x_kv"] = self.kv_up_proj(attn_dict["x_kv"])

        x = attn_dict["x"]
        del attn_dict["x"]

        output = self.decoder(x, mixer_kwargs=attn_dict)
        output = self.cross_attn_output_proj(output)

        return output
