from dataclasses import dataclass
from typing import Optional
from typing import ClassVar

from hydra_zen import builds
from gen.configs.utils import auto_store
from gen.datasets.base_dataset import AbstractDataset
from gen.datasets.coco_captions import CocoCaptions
from gen.datasets.controlnet_dataset import ControlnetDataset

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


def get_train(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, shuffle=True, **kwargs)

def get_val(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, shuffle=False, **kwargs)

auto_store(
    DatasetConfig, 
    train_dataset=get_train(ControlnetDataset, num_workers=2, batch_size=2), 
    validation_dataset=get_val(ControlnetDataset, num_workers=2, batch_size=2), 
    name="controlnet"
)
auto_store(
    DatasetConfig, 
    train_dataset=get_train(CocoCaptions, num_workers=2, batch_size=2), 
    validation_dataset=get_val(CocoCaptions, num_workers=2, batch_size=2), 
    name="coco_captions"
)