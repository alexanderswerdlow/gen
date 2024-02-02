from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
from torch import nn

from gen.models.utils import _init_weights
from gen.utils.logging_utils import log_info

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.models.cross_attn.base_model import ConditioningData

class Mapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.cross_attn = CrossAttn(cfg=cfg, input_dim=self.cfg.model.encoder_dim, cross_attn_dim=self.cfg.model.cross_attn_dim, output_dim=cfg.model.token_embedding_dim)

        self.learnable_token = nn.Parameter(torch.randn(cfg.model.num_conditioning_pairs if self.cfg.model.per_layer_queries else 1, cfg.model.cross_attn_dim) * .02)

        # If we have per layer queries, we don't need to chop up the mask vector
        if self.cfg.model.layer_specialization and not self.cfg.model.per_layer_queries:
            self.layer_specialization = nn.Sequential(
                nn.Linear(cfg.model.token_embedding_dim // self.cfg.model.num_conditioning_pairs, cfg.model.token_embedding_dim), 
                nn.LayerNorm(cfg.model.token_embedding_dim)
            )

        if self.cfg.model.feature_map_keys is not None:
            self.position_embedding = nn.Parameter(torch.randn(len(self.cfg.model.feature_map_keys), self.cfg.model.encoder_dim) * .02) # 

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
