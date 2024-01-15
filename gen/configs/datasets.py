from dataclasses import dataclass
from typing import ClassVar, Optional

from hydra_zen import builds, make_config
from gen import MOVI_OVERFIT_DATASET_PATH

from gen.configs.utils import auto_store, store_child_config
from gen.datasets.augmentation.kornia_augmentation import Augmentation
from gen.datasets.base_dataset import AbstractDataset
from gen.datasets.coco_captions import CocoCaptions
from gen.datasets.controlnet_dataset import ControlnetDataset
from gen.datasets.imagenet_dataset import ImageNetCustomDataset
from gen.datasets.kubrics.movi_dataset import MoviDataset


@dataclass
class DatasetConfig:
    name: ClassVar[str] = "dataset"
    train_dataset: AbstractDataset
    validation_dataset: Optional[AbstractDataset]
    num_validation_images: int = 2
    overfit: bool = False
    reset_validation_dataset_every_epoch: bool = False


@dataclass
class HuggingFaceControlNetConfig(DatasetConfig):
    _target_: str = "gen.datasets.controlnet_dataset"


def get_dataset(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, **kwargs)


def get_override_dict(**kwargs):
    return dict(
        train_dataset=dict(**kwargs),
        validation_dataset=dict(**kwargs),
    )


auto_store(DatasetConfig, train_dataset=get_dataset(ControlnetDataset), validation_dataset=get_dataset(ControlnetDataset), name="controlnet")

auto_store(DatasetConfig, train_dataset=get_dataset(CocoCaptions), validation_dataset=get_dataset(CocoCaptions), name="coco_captions")

store_child_config(DatasetConfig, "dataset", "coco_captions", "coco_captions_test")

auto_store(
    DatasetConfig,
    train_dataset=get_dataset(
        ControlnetDataset,
        num_workers=2,
        batch_size=2,
        conditioning_image_column="image",
        caption_column="prompt",
        dataset_config_name="2m_random_5k",
        dataset_name="poloclub/diffusiondb",
    ),
    validation_dataset=get_dataset(
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

augmentation = builds(
    Augmentation,
    minimal_source_augmentation=True,
    source_only_augmentation=True,
    target_resolution="${model.resolution}",
    populate_full_signature=True,
)

auto_store(
    DatasetConfig,
    train_dataset=get_dataset(MoviDataset, augmentation=augmentation, resolution="${model.resolution}"),
    validation_dataset=get_dataset(MoviDataset, augmentation=augmentation, resolution="${model.resolution}"),
    name="movi_e",
)

auto_store(
    DatasetConfig,
    train_dataset=get_dataset(ImageNetCustomDataset, augmentation=augmentation, resolution="${model.resolution}"),
    validation_dataset=get_dataset(ImageNetCustomDataset, augmentation=augmentation, resolution="${model.resolution}"),
    name="imagenet",
)
