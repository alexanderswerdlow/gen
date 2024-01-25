import hydra
import torch
from torch import nn

from gen.configs.base import BaseConfig
from gen.models.utils import _init_weights


class Mapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.learnable_token = nn.Parameter(torch.randn(cfg.model.cross_attn_dim))
        self.cross_attn = CrossAttn(cfg=cfg, input_dim=self.cfg.model.cross_attn_dim, output_dim=cfg.model.token_embedding_dim)
        self.apply(_init_weights)


class CrossAttn(nn.Module):
    def __init__(self, cfg: BaseConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.neti_up_proj = nn.Sequential(nn.Linear(input_dim, self.cfg.model.cross_attn_dim), nn.LayerNorm(self.cfg.model.cross_attn_dim))
        self.decoder = hydra.utils.instantiate(
            self.cfg.model.decoder_transformer,
            _recursive_=False,
            embed_dim=self.cfg.model.cross_attn_dim,
            use_flash_attn=self.cfg.trainer.mixed_precision != "no",
        )
        self.cross_attn_proj = nn.Sequential(nn.Linear(self.cfg.model.cross_attn_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, **kwargs):
        attn_dict = kwargs.get("attn_dict")
        x = self.neti_up_proj(attn_dict["x"])

        del attn_dict["x"]

        output = self.decoder(x, mixer_kwargs=attn_dict)
        output = self.cross_attn_proj(output)

        return output
