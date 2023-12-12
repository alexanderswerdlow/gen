from hydra.core.config_store import ConfigStore
from config.configs.configs import BaseConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)
