from jaxtyping import Bool, Float, Integer
from typing import Any, Optional
from dataclasses import dataclass, field
from torch import Tensor

@dataclass
class ConditioningData:
    xyz_normalizer: Optional[Any] = None
    xyz_valid: Optional[Bool[Tensor, "b h w"]] = None
    gt_xyz: Optional[Float[Tensor, "b h w xyz"]] = None
    timesteps: Optional[Integer[Tensor, "b"]] = None
    gt_decoder_xyz: Optional[Float[Tensor, "b h w xyz"]] = None
    gt_decoder_valid: Optional[Bool[Tensor, "b h w"]] = None

    # These are passed to the U-Net or pipeline
    encoder_hidden_states: Optional[Float[Tensor, "b d"]] = None
    unet_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)

@dataclass
class AttentionConfig:
    dual_self_attention: bool = False
    dual_cross_attention: bool = False
    joint_attention: bool = False
    add_cross_attn_pos_emb: Optional[int] = None

@dataclass
class AttentionMetadata:
    gate_scale: Optional[float] = None
    joint_attention: Optional[int] = None
    

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method