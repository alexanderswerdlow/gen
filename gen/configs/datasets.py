from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from hydra_zen import builds

from gen.configs.utils import auto_store, mode_store, store_child_config
from gen.datasets.abstract_dataset import AbstractDataset
from gen.datasets.augmentation.kornia_augmentation import Augmentation
from gen.datasets.dustr.co3d import Co3d
from gen.datasets.hypersim.hypersim import Hypersim
from gen.datasets.imagefolder.imagefolder import ImagefolderDataset


@dataclass
class DatasetConfig:
    name: ClassVar[str] = "dataset"
    train: AbstractDataset
    val: Optional[AbstractDataset]
    num_validation_images: int = 2
    overfit: bool = False
    reset_val_dataset_every_epoch: bool = False
    additional_train: Optional[dict[str, AbstractDataset]] = None
    additional_val: Optional[dict[str, AbstractDataset]] = None

@dataclass
class HuggingFaceControlNetConfig(DatasetConfig):
    _tgt_: str = "gen.datasets.controlnet_dataset"


def get_dataset(cls, **kwargs):
    return builds(cls, populate_full_signature=True, zen_partial=True, **kwargs)


def get_override_dict(**kwargs):
    return dict(
        train=dict(**kwargs),
        val=dict(**kwargs),
    )

augmentation = builds(
    Augmentation,
    enable_rand_augment=False,
    different_src_tgt_augmentation=False,
    src_random_scale_ratio=None,
    enable_random_resize_crop=False,
    enable_horizontal_flip=False,
    src_resolution=None,
    tgt_resolution=None,
    # A little hacky but very useful. We instantiate the model to get the transforms, making sure
    # that we always have the right transform
    src_transforms="${get_src_transform:model}",
    tgt_transforms="${get_tgt_transform:model}",
    populate_full_signature=True,
)

auto_store(DatasetConfig, 
    train=get_dataset(
        ImagefolderDataset,
        augmentation=augmentation,
    ), 
    val=get_dataset(
        ImagefolderDataset,
        augmentation=augmentation,
    ), 
    name="imagefolder"
)

auto_store(DatasetConfig, 
    train=get_dataset(
        Co3d,
        augmentation=None,
    ), 
    val=get_dataset(
        Co3d,
        augmentation=None,
    ), 
    name="co3d"
)

auto_store(DatasetConfig, 
    train=get_dataset(
        Hypersim,
        augmentation=augmentation,
        scratch_only=False,
    ), 
    val=get_dataset(
        Hypersim,
        augmentation=augmentation,
        scratch_only=False,
    ), 
    name="hypersim"
)

def get_datasets():  # TODO: These do not need to be global configs
    mode_store(
        name="example",
        dataset=dict(
            train=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False)),
            val=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False)),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagefolder"},
        ],
    )