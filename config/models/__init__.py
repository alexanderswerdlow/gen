from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    pass

@dataclass
class ControlNetConfig(ModelConfig):
    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    controlnet_model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None

cs = ConfigStore.instance()
cs.store(group="model", name="controlnet", node=ControlNetConfig)