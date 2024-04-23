from jaxtyping import Bool, Float, Integer
from typing import Any, Optional
from dataclasses import dataclass, field
from torch import Tensor

@dataclass
class ConditioningData:
    placeholder_token: Optional[int] = None
    attn_dict: Optional[dict[str, Tensor]] = None
    clip_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None
    mask_head_tokens: Optional[Float[Tensor, "n d"]] = None # We sometimes need this if we want to have some mask tokens with detached gradients
    mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    mask_dropout: Optional[Bool[Tensor, "n"]] = None
    batch_cond_dropout: Optional[Bool[Tensor, "b"]] = None
    input_prompt: Optional[list[str]] = None
    learnable_idxs: Optional[Any] = None
    batch_attn_masks: Optional[Float[Tensor, "b hw hw"]] = None
    
    src_mask_tokens: Optional[Float[Tensor, "n d"]] = None # We duplicate for loss calculation
    tgt_mask_tokens: Optional[Float[Tensor, "n d"]] = None
    tgt_mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    tgt_mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    tgt_mask_dropout: Optional[Bool[Tensor, "n"]] = None

    mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    src_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    tgt_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    gt_src_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    gt_src_mask_token: Optional[Float[Tensor, "n d"]] = None

    src_mask_tokens_before_specialization: Optional[Float[Tensor, "n d"]] = None # We duplicate for loss calculation
    tgt_mask_tokens_before_specialization: Optional[Float[Tensor, "n d"]] = None

    src_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    encoder_input_pixel_values: Optional[Float[Tensor, "b c h w"]] = None

    src_orig_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    tgt_orig_feature_map: Optional[Float[Tensor, "b d h w"]] = None

    src_warped_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    tgt_warped_feature_map: Optional[Float[Tensor, "b d h w"]] = None

    mask_token_centroids: Optional[Float[Tensor, "n 2"]] = None
    tgt_mask_token_centroids: Optional[Float[Tensor, "n 2"]] = None

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
class AttentionMetadata:
    layer_idx: Optional[int] = None
    num_layers: Optional[int] = None
    num_cond_vectors: Optional[int] = None
    add_pos_emb: Optional[bool] = None
    cross_attention_mask: Optional[Float[Tensor, "b d h w"]] = None
    self_attention_mask: Optional[Float[Tensor, "b hw hw"]] = None
    gate_scale: Optional[float] = None
    frozen_dim: Optional[int] = None
    return_attn_probs: Optional[bool] = None
    attn_probs: Optional[Float[Tensor, "..."]] = None
    custom_map: Optional[dict] = None
    posemb: Optional[tuple] = None

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method