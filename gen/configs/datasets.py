from dataclasses import dataclass
from typing import ClassVar, Optional

from hydra_zen import builds, make_config

from gen.configs.utils import auto_store, stored_child_config
from gen.datasets.augmentation.kornia_augmentation import Augmentation
from gen.datasets.base_dataset import AbstractDataset
from gen.datasets.coco_captions import CocoCaptions
from gen.datasets.controlnet_dataset import ControlnetDataset
from gen.datasets.kubrics.movi_dataset import MoviDataset


@dataclass
class DatasetConfig:
    name: ClassVar[str] = "dataset"
    train_dataset: AbstractDataset
    validation_dataset: Optional[AbstractDataset]
    num_validation_images: int = 2
    overfit: bool = False


@dataclass
class HuggingFaceControlNetConfig(DatasetConfig):
    _target_: str = "gen.datasets.controlnet_dataset"


def get_train(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, **kwargs)


def get_val(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, **kwargs)


auto_store(DatasetConfig, train_dataset=get_train(ControlnetDataset), validation_dataset=get_val(ControlnetDataset), name="controlnet")

auto_store(DatasetConfig, train_dataset=get_train(CocoCaptions), validation_dataset=get_val(CocoCaptions), name="coco_captions")

stored_child_config(DatasetConfig, "dataset", "coco_captions", "coco_captions_test")

auto_store(
    DatasetConfig,
    train_dataset=get_train(
        ControlnetDataset,
        num_workers=2,
        batch_size=2,
        conditioning_image_column="image",
        caption_column="prompt",
        dataset_config_name="2m_random_5k",
        dataset_name="poloclub/diffusiondb",
    ),
    validation_dataset=get_val(
        ControlnetDataset,
        num_workers=2,
        batch_size=2,
        conditioning_image_column="image",
        caption_column="prompt",
        dataset_config_name="2m_random_5k",
        dataset_name="poloclub/diffusiondb",
    ),
    name="diffusiondb",
)

augmentation = builds(Augmentation, populate_full_signature=True)
auto_store(DatasetConfig, train_dataset=get_train(MoviDataset, augmentation=augmentation), validation_dataset=get_val(MoviDataset, augmentation=augmentation), name="movi_e")