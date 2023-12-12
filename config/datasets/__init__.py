from hydra.core.config_store import ConfigStore

from dataclasses import field, dataclass
from glob import glob
from typing import List, Optional, Iterable, Union

@dataclass
class DatasetConfig:
    dataloader_num_workers: int = 0
    train_batch_size: int = 4

@dataclass
class HuggingFaceControlNetConfig(DatasetConfig):
    dataset_name: Optional[str] = "fusing/fill50k"
    dataset_config_name: Optional[str] = None
    dataset_split: Optional[tuple[str]] = None
    train_data_dir: Optional[str] = None
    image_column: Optional[str] = "image"
    conditioning_image_column: Optional[str] = "conditioning_image"
    caption_column: Optional[str] = "text"
    proportion_empty_prompts: float = 0
    validation_prompt: Optional[str] = None
    validation_image: Optional[tuple[str]] = None
    num_validation_images: int = 4
    cache_dir: Optional[str] = None
    resolution: Optional[int] = 512
    
cs = ConfigStore.instance()
cs.store(group="dataset", name="base", node=DatasetConfig)
cs.store(group="dataset", name="huggingface", node=HuggingFaceControlNetConfig)