from dataclasses import dataclass
from typing import Dict, Optional
from hydra.core.config_store import ConfigStore
from enum import Enum
from dataclasses import dataclass, field

class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1

@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
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
    pe_sigmas: Dict[str, float] = field(default_factory=lambda: {'sigma_t': 0.03, 'sigma_l': 2.0})
    # Number of time anchors for computing our positional encodings
    num_pe_time_anchors: int = 10
    # Whether to output the textual bypass vector
    output_bypass: bool = True

@dataclass
class ControlNetConfig(ModelConfig):
    model_type: ModelType = ModelType.CONTROLNET
    controlnet_model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None

@dataclass
class BaseMapperConfig(ModelConfig):
    model_type: ModelType = ModelType.BASE_MAPPER
    placeholder_token: str = 'placeholder'
    super_category_token: str = 'object'

cs = ConfigStore.instance()
cs.store(group="model", name="controlnet", node=ControlNetConfig)
cs.store(group="model", name="basemapper", node=BaseMapperConfig)