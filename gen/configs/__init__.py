from hydra.core.config_store import ConfigStore
from gen.configs.base import BaseConfig
from gen.configs.datasets import DatasetConfig
from gen.configs.models import ModelConfig, ModelType
from gen.configs.trainer import TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)
