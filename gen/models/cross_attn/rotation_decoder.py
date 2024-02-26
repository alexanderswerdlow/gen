# Copyright (c) 2022, Tri Dao.
# Inspired by / adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


def create_mixer_cls(
    num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc, cross_attn=False
):
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
    drop_rate,
    attn_drop_rate,
    drop_path1,
    drop_path2,
    norm_layer,
    act_layer,
    use_flash_attn,
    fused_bias_fc,
    fused_mlp,
    fused_dropout_add_ln,
    layer_idx=None,
    n_layer=None,
    last_layer_subset=False,
):
    mixer_cls = create_mixer_cls(
        num_heads,
        qkv_bias,
        attn_drop_rate,
        use_flash_attn,
        fused_bias_fc,
        cross_attn=(last_layer_subset and layer_idx == n_layer - 1),
    )
    mlp_cls = create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp)
    # TD [2022-10-15]: Force residual in fp32 in case of DeepSpeed
    block = Block(
        embed_dim,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_layer,
        prenorm=True,
        resid_dropout1=drop_rate,
        resid_dropout2=drop_rate,
        drop_path1=drop_path1,
        drop_path2=drop_path2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=True,
    )
    return block


class SelfAttentionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        seq_len=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        input_dim=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool == "token", "Only support pooling with CLS token"
        assert class_token
        assert init_values is None, "LayerScale is not supported yet"
        assert weight_init == ""
        assert fc_norm is None
        # pre_norm seems redundant, as there's a LayerNorm right at the start of each block, idk
        assert not pre_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = seq_len
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim is not None else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path1=dpr[i - 1] if i > 0 else 0.0,
                    drop_path2=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=fused_bias_fc,
                    fused_mlp=fused_mlp,
                    fused_dropout_add_ln=fused_dropout_add_ln,
                    layer_idx=i,
                    n_layer=depth,
                    last_layer_subset=(global_pool == "token"),
                )
                for i in range(depth)
            ]
        )

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(embed_dim)

        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")

        # Classifier Head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode == ""
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _pos_embed(self, x):
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if self.input_dim is not None:
            raise NotImplementedError("We may need to switch the order of x = x + self.pos_embed here.")
        
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return x

    def forward_features(self, x):
        """
        If all_tokens==False and self.global_pool == 'token', we only return the features for the
        cls token.
        """
        x = self.input_proj(x)

        hidden_states = self._pos_embed(x)
        residual = None

        for block in self.blocks[:-1]:
            hidden_states, residual = block(hidden_states, residual)
        # For the last layer, we only want the 1st token of the output. So we do cross-attention
        # where the query is the 1st token and the key/value is the whole sequence.
        hidden_states, residual = self.blocks[-1](
            hidden_states, residual, mixer_subset=slice(0, 1)
        )
        if not self.fused_dropout_add_ln:
            residual = self.drop_path(self.dropout(hidden_states)) + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            if self.drop_path.p == 0 or not self.training:
                rowscale = None
            else:
                rowscale = self.drop_path(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            # Set prenorm=False here since we don't need to the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
            )
        return hidden_states

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])
        return x
    
def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
