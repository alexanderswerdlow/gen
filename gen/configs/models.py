from dataclasses import dataclass
from typing import ClassVar, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field

from hydra_zen import builds
from gen.configs.utils import auto_store, store_child_config
from gen.models.neti.decoder import DecoderTransformer
from hydra_zen.typing import Builds


class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1


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
    placeholder_token: str = "place"
    placeholder_token_id: Optional[int] = None
    super_category_token: str = "object"

    mask_cross_attn: bool = True
    freeze_clip: bool = True
    unfreeze_last_n_clip_layers: Optional[int] = None
    dropout_masks: Optional[float] = None
    freeze_text_encoder: bool = True
    controlnet: bool = False
    use_fixed_position_encoding: bool = False
    enable_norm_scale: bool = True
    enable_neti: bool = False
    cross_attn_residual: bool = True
    decoder_transformer: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    use_dataset_segmentation: bool = True
    use_cls_token_only: bool = False


@dataclass
class ControlNetConfig(ModelConfig):
    model_type: ModelType = ModelType.CONTROLNET
    controlnet_model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    mask_cross_attn: bool = False


auto_store(ControlNetConfig, name="controlnet")
auto_store(ModelConfig, model_type=ModelType.BASE_MAPPER, name="basemapper")
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="cross_attn",
    unfreeze_last_n_clip_layers=None,
    dropout_masks=0.2,
    enable_norm_scale=False,
    use_fixed_position_encoding=True,
    nested_dropout_prob=0,
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="neti",
    enable_norm_scale=False,
    use_fixed_position_encoding=False,
    nested_dropout_prob=0.5,
    mask_cross_attn=False,
    output_bypass=False,
)
