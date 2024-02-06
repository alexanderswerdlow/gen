from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
from torch import Tensor, nn


from gen.models.utils import _init_weights
from gen.utils.logging_utils import log_info
from gen.models.cross_attn.decoder import create_mlp_cls
from jaxtyping import Bool, Float, Integer

if TYPE_CHECKING:
    from gen.models.cross_attn.base_model import InputData
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import ConditioningData

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


class TokenMapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg

        dim = self.cfg.model.token_embedding_dim * self.cfg.model.num_conditioning_pairs
        if self.cfg.model.token_cls_pred_loss:
            self.cls_mlp = Mlp(in_features=dim, hidden_features=dim // 4, out_features=self.cfg.model.num_token_cls, activation=nn.GELU())

        if self.cfg.model.token_rot_pred_loss:
            self.rot_mlp = Mlp(in_features=dim, hidden_features=dim // 4, out_features=6, activation=nn.GELU())

        self.apply(_init_weights)

    def forward(self, cfg: BaseConfig, batch: InputData, cond: ConditioningData):
        ret = {}
        if self.cfg.model.token_cls_pred_loss:
            output = self.cls_mlp(cond.mask_tokens)
            pred = output.softmax(dim=-1)
            ret["cls_pred"] = pred

        if self.cfg.model.token_rot_pred_loss:
            output = self.rot_mlp(cond.mask_tokens)
            pred = output.softmax(dim=-1)
            ret["rot_pred"] = pred

        return ret


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
