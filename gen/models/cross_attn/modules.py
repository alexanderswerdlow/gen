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
        self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, out_features=self.cfg.model.num_token_cls, activation=nn.GELU())
        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(_init_weights)

    def forward(self, cfg: BaseConfig, batch: InputData, cond: ConditioningData):
        bs = batch['gen_pixel_values'].shape[0]
        device = batch['gen_pixel_values'].device

        assert cfg.model.background_mask_idx == 0
        output = self.mlp(cond.mask_tokens)
        pred = output.softmax(dim=-1)

        losses = []
        for b in range(bs):
            if cond.batch_cond_dropout is not None and cond.batch_cond_dropout[b].item():
                continue

            # We only compute loss on non-dropped out masks
            instance_categories = batch['categories'][b][batch['valid'][b]]

            # We align the dataset instance indices with the flattened pred indices
            mask_idxs_for_batch = cond.mask_instance_idx[cond.mask_batch_idx == b]
            pred_idxs_for_batch = torch.arange(cond.mask_instance_idx.shape[0], device=device)[cond.mask_batch_idx == b]

            # The background is always 0 so we must remove it if it exists and move everything else down
            remove_background_mask = mask_idxs_for_batch != 0

            mask_idxs_for_batch = mask_idxs_for_batch[remove_background_mask]
            pred_idxs_for_batch = pred_idxs_for_batch[remove_background_mask]

            pred_ = pred[pred_idxs_for_batch]
            instance_categories = instance_categories[mask_idxs_for_batch - 1]

            losses.append(self.loss_fn(pred_, instance_categories.long()))

        return torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=device)


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
