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
    # pretrained_model_name_or_path: Optional[str] = "stabilityai/stable-diffusion-2-1"
    # token_embedding_dim: int = 1024
    # resolution: int = 768
    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    token_embedding_dim: int = 768
    resolution: int = 512

    revision: Optional[str] = None
    variant: Optional[str] = None
    model_type: Optional[ModelType] = None

    # Whether to use our Nested Dropout technique
    use_nested_dropout: bool = True
    # Probability to apply nested dropout during training
    nested_dropout_prob: float = 0.5
    # Whether to normalize the norm of the mapper's output vector
    normalize_mapper_output: bool = True
    # Target norm for the mapper's output vector
    target_norm: Optional[float] = None
    # Whether to use positional encoding over the input to the mapper
    use_positional_encoding: bool = True
    # Sigmas used for computing positional encoding
    pe_sigmas: Dict[str, float] = field(default_factory=lambda: {"sigma_t": 0.03, "sigma_l": 2.0})
    # Number of time anchors for computing our positional encodings
    num_pe_time_anchors: int = 10
    # Whether to output the textual bypass vector
    output_bypass: bool = True

    model_type: ModelType = ModelType.BASE_MAPPER
    placeholder_token: str = "android"
    placeholder_token_id: Optional[int] = None
    super_category_token: str = "object"

    mask_cross_attn: bool = True
    freeze_clip: bool = True
    unfreeze_last_n_clip_layers: Optional[int] = None
    dropout_masks: Optional[float] = None
    freeze_text_encoder: bool = True
    controlnet: bool = False
    
    enable_norm_scale: bool = True
    enable_neti: bool = False
    cross_attn_residual: bool = True
    use_dataset_segmentation: bool = True

    use_custom_position_encoding: bool = False # Whether to use original the NeTI mapper or our custom T+L mapper
    use_timestep_layer_encoding: bool = True # Whether to use T+L in our custom mapper, otherwise just a learned emb

    encode_token_without_tl: bool = False # Maps single token to (2 * token_embedding_dim) instead of T+L mapping
    use_cls_token_projected: bool = False # These define where to get the single token
    use_cls_token_final_layer: bool = False
    use_cls_token_mean: bool = False
    tmp_revert_to_neti_logic: bool = False

    cross_attn_dim: int = 1024

    decoder_transformer: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)


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
