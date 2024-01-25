from functools import partial

import torch.nn as nn
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp


def create_mixer_cls(num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc, cross_attn=False):
    mixer_cls = partial(
        MHA,
        num_heads=num_heads,
        cross_attn=cross_attn,
        qkv_proj_bias=qkv_bias,
        dropout=attn_drop,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
    )
    return mixer_cls


def create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp):
    inner_dim = int(embed_dim * mlp_ratio)
    if not fused_mlp:
        mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=act_layer())
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
    return mlp_cls


def create_block(
    embed_dim,
    num_heads,
    mlp_ratio,
    qkv_bias,
    attn_drop_rate,
    norm_layer,
    act_layer,
    cross_attn,
    fused_bias_fc: bool = True,
    fused_mlp: bool = True,
    fused_dropout_add_ln: bool = False,
    use_flash_attn: bool = True,
    **kwargs,
):
    mixer_cls = create_mixer_cls(
        num_heads,
        qkv_bias,
        attn_drop_rate,
        use_flash_attn,
        fused_bias_fc,
        cross_attn=cross_attn,
    )
    mlp_cls = create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp)
    # TD [2022-10-15]: Force residual in fp32 in case of DeepSpeed
    block = Block(
        embed_dim, mixer_cls, mlp_cls, norm_cls=norm_layer, prenorm=True, fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=True, **kwargs
    )
    return block


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=None,
        act_layer=None,
        use_flash_attn: bool = True,
        fused_bias_fc: bool = True,
        fused_mlp: bool = True,
        add_self_attn: bool = True,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.blocks = nn.ModuleList(
            [
                create_decoder_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    attn_drop_rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=fused_bias_fc,
                    fused_mlp=fused_mlp,
                    add_self_attn=add_self_attn,
                )
                for _ in range(depth)
            ]
        )

        self.dropout = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(embed_dim)
        self.add_self_attn = add_self_attn

    def forward(self, x, **kwargs):
        hidden_states, residual = x, None
        cross_attn_dict = {
            "max_seqlen": kwargs["mixer_kwargs"]["max_seqlen"],
            "cu_seqlens": kwargs["mixer_kwargs"]["cu_seqlens"],
        }

        for _blocks in self.blocks:
            hidden_states, residual = _blocks[0](hidden_states=hidden_states, residual=residual, **kwargs)
            if self.add_self_attn:
                hidden_states, residual = _blocks[1](hidden_states=hidden_states, residual=residual, mixer_kwargs=cross_attn_dict, )

        residual = self.dropout(hidden_states) + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        return hidden_states


def create_decoder_block(embed_dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, norm_layer, act_layer, use_flash_attn, fused_bias_fc, fused_mlp, add_self_attn, **kwargs):
    return nn.ModuleList(
        [
            create_block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                attn_drop_rate,
                norm_layer,
                act_layer,
                use_flash_attn=use_flash_attn,
                fused_bias_fc=fused_bias_fc,
                fused_mlp=fused_mlp,
                cross_attn=True,
                **kwargs,
            )
        ]
        + (
            [
                create_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    attn_drop_rate,
                    norm_layer,
                    act_layer,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=fused_bias_fc,
                    fused_mlp=fused_mlp,
                    cross_attn=False,
                    **kwargs,
                )
            ]
            if add_self_attn
            else []
        )
    )
