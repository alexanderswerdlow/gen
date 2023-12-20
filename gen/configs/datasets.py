from hydra.core.config_store import ConfigStore

from dataclasses import field, dataclass
from glob import glob
from typing import List, Optional, Iterable, Union

@dataclass
class DatasetConfig:
    dataloader_num_workers: int = 1
    train_batch_size: int = 2
    resolution: Optional[int] = 512
    _target_: str = "gen.datasets.base_dataset"

@dataclass
class HuggingFaceControlNetConfig(DatasetConfig):
    _target_: str = "gen.datasets.controlnet_dataset"
    dataset_class: str = 'controlnet_dataset'
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

@dataclass
class CocoCaptions(DatasetConfig):
    _target_: str = "gen.datasets.coco_captions.CocoCaptions"
    override_text: bool = True
    
cs = ConfigStore.instance()
cs.store(group="dataset", name="base", node=DatasetConfig)
cs.store(group="dataset", name="huggingface", node=HuggingFaceControlNetConfig)
cs.store(group="dataset", name="coco_captions", node=CocoCaptions)