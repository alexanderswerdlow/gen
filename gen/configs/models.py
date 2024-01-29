from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store, store_child_config
from gen.models.cross_attn.decoder import DecoderTransformer
from gen.models.encoders.encoder import BaseModel, ClipFeatureExtractor


class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1
    SODA = 2


@dataclass
class ModelConfig:
    name: ClassVar[str] = "model"

    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    token_embedding_dim: int = 768
    num_unet_cross_attn_layers: int = 16 # Number of cross-attentions between U-Net latents and text-tokens

    resolution: int = 512

    revision: Optional[str] = None
    variant: Optional[str] = None
    model_type: Optional[ModelType] = None

    model_type: ModelType = ModelType.BASE_MAPPER

    controlnet: bool = False # Add a controlnet on top of the main U-Net
    freeze_unet: bool = True
    lora_unet: bool = False # Perform low-rank adaptation on the main U-Net
    freeze_mapper: bool = False # Freezes the cross-attention mapper itself. Useful for debugging.
    freeze_text_encoder: bool = True # We freeze the CLIP Text encoder by default
    freeze_text_encoder_except_token_embeddings: bool = False # We can choose to only train the token embeddings like in break-a-scene
    freeze_clip: bool = True # We freeze the CLIP Vision encoder by default
    unfreeze_last_n_clip_layers: Optional[int] = None # Unfreeze specific clip layers
    unfreeze_unet_after_n_steps: Optional[int] = None

    dropout_masks: Optional[float] = None # We can randomly dropout object masks during training. The background is always preserved.
    per_timestep_conditioning: bool = True # Switches to the NeTI-style conditioning scheme where we get an embedding per T+L
    enable_neti: bool = False # Even in the NeTI-style conditioning scheme, we may want to disable the NeTI mapper itself and use a different T+L enc

    use_dataset_segmentation: bool = True # Determines if we use the dataset GT or SAM
    cross_attn_dim: int = 1024
    
    decoder_transformer: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)

    lora_rank: int = 4
    break_a_scene_cross_attn_loss: bool = False
    break_a_scene_cross_attn_loss_weight: float = 1e-2
    break_a_scene_masked_loss: bool = False
    background_mask_idx: int = 0 # Used for the break-a-scene mask loss to not count loss for the background mask
    placeholder_token: str = "masks"

    # Quick experiment configs
    break_a_scene_cross_attn_loss_second_stage: bool = False
    dropout_foreground_only: bool = False
    dropout_background_only: bool = False
    layer_specialization: bool = False # Whether to map token_embedding_dim -> num_unet_cross_attn_layers * token_embedding_dim so that each layer has its own embedding
    clip_shift_scale_conditioning: bool = False # Whether to use the CLIP shift and scale embeddings as conditioning
    placeholder_token_id: Optional[int] = None
    mask_cross_attn: bool = True

    # NeTI Specific Configs below
    encode_token_without_tl: bool = False # Maps single token to (2 * token_embedding_dim) instead of T+L mapping
    use_cls_token_projected: bool = False # These define where to get the single token
    use_cls_token_final_layer: bool = False
    use_cls_token_mean: bool = False


@dataclass
class ControlNetConfig(ModelConfig):
    model_type: ModelType = ModelType.CONTROLNET
    controlnet_model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    mask_cross_attn: bool = False


auto_store(ControlNetConfig, name="controlnet")
auto_store(
    ModelConfig,
    model_type=ModelType.BASE_MAPPER,
    name="basemapper",
    encoder=builds(ClipFeatureExtractor, return_only="ln_post", populate_full_signature=False),
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="cross_attn",
    unfreeze_last_n_clip_layers=None,
    dropout_masks=0.2,
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="neti",
    mask_cross_attn=False,
)
