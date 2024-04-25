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
    # These are passed to the U-Net or pipeline
    encoder_hidden_states: Optional[Float[Tensor, "b d"]] = None
    unet_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)

@dataclass
class TokenPredData:
    gt_rot_6d: Optional[Float[Tensor, "n 6"]] = None
    noised_rot_6d: Optional[Float[Tensor, "n 6"]] = None
    rot_6d_noise: Optional[Float[Tensor, "n 6"]] = None
    timesteps: Optional[Integer[Tensor, "b"]] = None
    cls_pred: Optional[Float[Tensor, "n classes"]] = None
    pred_6d_rot: Optional[Float[Tensor, "n 6"]] = None
    token_output_mask: Optional[Bool[Tensor, "n"]] = None
    relative_rot_token_mask: Optional[Bool[Tensor, "n"]] = None
    denoise_history_6d_rot: Optional[Float[Tensor, "t n 6"]] = None
    denoise_history_timesteps: Optional[list[int]] = None
    raw_pred_rot_logits: Optional[Float[Tensor, "n ..."]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None

@dataclass
class AttentionConfig:
    dual_self_attention: bool = False
    dual_cross_attention: bool = False

@dataclass
class AttentionMetadata:
    gate_scale: Optional[float] = None
    

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method