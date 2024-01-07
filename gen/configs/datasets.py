from dataclasses import dataclass
from typing import Optional
from typing import ClassVar

from hydra_zen import builds
from gen.configs.utils import auto_store
from gen.datasets.base_dataset import AbstractDataset
from gen.datasets.coco_captions import CocoCaptions

@dataclass
class DatasetConfig:
    name: ClassVar[str] = 'dataset'
    train_dataset: AbstractDataset
    validation_dataset: Optional[AbstractDataset]
    num_validation_images: int = 2
    overfit: bool = False

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
    cache_dir: Optional[str] = None

def get_train(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, shuffle=True, **kwargs)

def get_val(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, shuffle=False, **kwargs)

auto_store(
    DatasetConfig, 
    train_dataset=get_train(CocoCaptions, num_workers=2, batch_size=2), 
    validation_dataset=get_val(CocoCaptions, num_workers=2, batch_size=2), 
    name="coco_captions"
)