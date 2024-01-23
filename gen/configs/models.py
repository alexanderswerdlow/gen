from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store, store_child_config
from gen.models.neti.decoder import DecoderTransformer
from gen.utils.encoder_utils import BaseModel, ClipFeatureExtractor


class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1
    SODA = 2


@dataclass
class ModelConfig:
    name: ClassVar[str] = "model"

    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    token_embedding_dim: int = 768
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
    break_a_scene_loss_weight: float = 1e-2

    # NeTI Specific Configs below
    placeholder_token: str = "android"
    placeholder_token_id: Optional[int] = None
    super_category_token: str = "object"
    mask_cross_attn: bool = True
    encode_token_without_tl: bool = False # Maps single token to (2 * token_embedding_dim) instead of T+L mapping
    use_cls_token_projected: bool = False # These define where to get the single token
    use_cls_token_final_layer: bool = False
    use_cls_token_mean: bool = False
    tmp_revert_to_neti_logic: bool = False
    use_custom_position_encoding: bool = False # Whether to use original the NeTI mapper or our custom T+L mapper
    use_timestep_layer_encoding: bool = True # Whether to use T+L in our custom mapper, otherwise just a learned emb
    enable_norm_scale: bool = True
    cross_attn_residual: bool = True
    use_nested_dropout: bool = True # Whether to use our Nested Dropout technique
    nested_dropout_prob: float = 0.5 # Probability to apply nested dropout during training
    normalize_mapper_output: bool = True # Whether to normalize the norm of the mapper's output vector
    target_norm: Optional[float] = None # Target norm for the mapper's output vector
    use_positional_encoding: bool = True # Whether to use positional encoding over the input to the mapper
    pe_sigmas: Dict[str, float] = field(default_factory=lambda: {"sigma_t": 0.03, "sigma_l": 2.0}) # Sigmas used for computing positional encoding
    num_pe_time_anchors: int = 10 # Number of time anchors for computing our positional encodings
    output_bypass: bool = True # Whether to output the textual bypass vector

    # pretrained_model_name_or_path: Optional[str] = "stabilityai/stable-diffusion-2-1"
    # token_embedding_dim: int = 1024
    # resolution: int = 768


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
    enable_norm_scale=False,
    use_timestep_layer_encoding=True,
    nested_dropout_prob=0.5,
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="neti",
    enable_norm_scale=False,
    use_timestep_layer_encoding=False,
    nested_dropout_prob=0.5,
    mask_cross_attn=False,
    output_bypass=False,
)
