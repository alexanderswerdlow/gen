from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from hydra_zen import builds

from gen import (IMAGENET_PATH, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH,
                 MOVI_OVERFIT_DATASET_PATH)
from gen.configs.utils import auto_store, mode_store, store_child_config
from gen.datasets.abstract_dataset import AbstractDataset
from gen.datasets.augmentation.kornia_augmentation import Augmentation
from gen.datasets.calvin.calvin import CalvinDataset
from gen.datasets.coco.coco_panoptic import CocoPanoptic
from gen.datasets.controlnet_dataset import ControlnetDataset
from gen.datasets.hypersim.hypersim import Hypersim
from gen.datasets.imagefolder.imagefolder import ImagefolderDataset
from gen.datasets.imagenet_dataset import ImageNetCustomDataset
from gen.datasets.kubrics.movi_dataset import MoviDataset
from gen.datasets.objaverse.objaverse import ObjaverseData
from gen.datasets.scannetpp.scannetpp import ScannetppIphoneDataset


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



auto_store(DatasetConfig, train=get_dataset(ControlnetDataset), val=get_dataset(ControlnetDataset), name="controlnet")


auto_store(
    DatasetConfig,
    train=get_dataset(
        ControlnetDataset,
        num_workers=2,
        batch_size=2,
        conditioning_image_column="image",
        caption_column="prompt",
        dataset_config_name="2m_random_5k",
        dataset_name="poloclub/diffusiondb",
    ),
    val=get_dataset(
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

auto_store(
    DatasetConfig,
    train=get_dataset(MoviDataset, augmentation=augmentation),
    val=get_dataset(MoviDataset, augmentation=augmentation),
    name="movi_e",
)

auto_store(
    DatasetConfig,
    train=get_dataset(ImageNetCustomDataset, augmentation=augmentation),
    val=get_dataset(ImageNetCustomDataset, augmentation=augmentation),
    name="imagenet",
)


auto_store(DatasetConfig, 
    train=get_dataset(
        CocoPanoptic,
        augmentation=augmentation,
        resolution="${model.decoder_resolution}",
        scratch_only=False,
    ), 
    val=get_dataset(
        CocoPanoptic,
        augmentation=augmentation,
        resolution="${model.decoder_resolution}",
        scratch_only=False,
    ), 
    name="coco_panoptic"
)

auto_store(DatasetConfig, 
    train=get_dataset(
        ObjaverseData,
        augmentation=augmentation,
        resolution="${model.decoder_resolution}",
    ), 
    val=get_dataset(
        ObjaverseData,
        augmentation=augmentation,
        resolution="${model.decoder_resolution}",
    ), 
    name="objaverse"
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

auto_store(DatasetConfig, 
    train=get_dataset(
        ScannetppIphoneDataset,
        augmentation=augmentation,
        scratch_only=False,
    ), 
    val=get_dataset(
        ScannetppIphoneDataset,
        augmentation=augmentation,
        scratch_only=False,
    ), 
    name="scannetpp"
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
        CalvinDataset,
        augmentation=augmentation,
    ), 
    val=get_dataset(
        CalvinDataset,
        augmentation=augmentation,
    ), 
    name="calvin"
)

store_child_config(DatasetConfig, "dataset", "coco_panoptic", "coco_panoptic_test")


shared_overfit_movi_args = dict(
    custom_split="train",
    path=MOVI_OVERFIT_DATASET_PATH,
    num_objects=1,
    augmentation=dict(enable_rand_augment=False, enable_random_resize_crop=False),
)

shared_movi_args = dict(
    path=MOVI_DATASET_PATH,
    num_objects=23,
    augmentation=dict(enable_rand_augment=False, enable_random_resize_crop=True, enable_horizontal_flip=False),
)

def get_datasets():  # TODO: These do not need to be global configs
    mode_store(
        name="movi",
        dataset=dict(
            train=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False), multi_camera_format=False),
            val=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False), multi_camera_format=False),
        ),
        model=dict(
            segmentation_map_size=24,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "movi_e"},
        ],
    )

    mode_store(
        name="movi_full",
        dataset=dict(
            train=dict(custom_split="train", **shared_movi_args),
            val=dict(custom_split="validation", **shared_movi_args),
        ),
        hydra_defaults=["movi"],
    )

    mode_store(
        name="movi_overfit",
        dataset=dict(
            train=dict(batch_size=20, subset_size=None, **shared_overfit_movi_args),
            val=dict(subset_size=4, **shared_overfit_movi_args),
            overfit=True,
        ),
        hydra_defaults=["movi"],
    )

    mode_store(
        name="movi_single_scene",
        dataset=dict(train=dict(subset=("video_0015",), fake_return_n=8), val=dict(subset=("video_0015",), fake_return_n=8)),
        hydra_defaults=["movi_overfit"],
    )

    mode_store(
        name="movi_single_frame",
        dataset=dict(train=dict(num_dataset_frames=1, fake_return_n=256), val=dict(num_dataset_frames=1, fake_return_n=256)),
        hydra_defaults=["movi_single_scene"],
    )

    mode_store(
        name="movi_augmentation",
        dataset=dict(
            train=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=True)),
            val=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=True)),
        ),
    )

    mode_store(
        name="no_movi_augmentation",
        dataset=dict(
            train=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False)),
            val=dict(augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False)),
        ),
    )

    mode_store(
        name="movi_validate_single_scene",
        dataset=dict(
            val=dict(
                subset_size=4,
                subset=("video_0018",),
                fake_return_n=8,
                random_subset=False,
                augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False, enable_rand_augment=False),
            ),
        ),
        hydra_defaults=["movi_single_scene"],
    )

    mode_store(
        name="movi_medium",
        dataset=dict(
            train=dict(
                custom_split="train",
                augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False, enable_rand_augment=False),
                path=MOVI_MEDIUM_PATH,
                num_objects=23,
                num_frames=8,
                num_cameras=2,
                multi_camera_format=True,
            ),
            val=dict(
                custom_split="validation",
                subset_size=8,
                random_subset=True,
                path=MOVI_MEDIUM_PATH,
                num_objects=23,
                num_frames=8,
                num_cameras=2,
                multi_camera_format=True,
                augmentation=dict(enable_horizontal_flip=False, enable_random_resize_crop=False, enable_rand_augment=False),
            ),
        ),
        model=dict(
                segmentation_map_size=24,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "movi_e"},
        ],
    )

    mode_store(
        name="movi_medium_two_objects",
        dataset=dict(
            train=dict(
                num_cameras=1,
                path=MOVI_MEDIUM_TWO_OBJECTS_PATH,
            ),
            val=dict(
                num_cameras=1,
                path=MOVI_MEDIUM_TWO_OBJECTS_PATH,
            ),
        ),
        hydra_defaults=["movi_medium"],
    )

    mode_store(
        name="single_scene",
        dataset=dict(
            train=dict(subset=("000001",), fake_return_n=8), val=dict(subset=("000001",), fake_return_n=8), overfit=True
        ),
    )

    mode_store(
        name="movi_medium_single_object",
        dataset=dict(
            train=dict(
                num_cameras=1,
                num_frames=24,
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
            ),
            val=dict(
                num_cameras=1,
                num_frames=24,
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
            ),
        ),
        hydra_defaults=["movi_medium"],
    )

    mode_store(
        name="imagenet",
        dataset=dict(
            train=dict(path=IMAGENET_PATH, augmentation=dict(enable_random_resize_crop=False, enable_horizontal_flip=False)),
            val=dict(path=IMAGENET_PATH, augmentation=dict(enable_random_resize_crop=False, enable_horizontal_flip=False)),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagenet"},
        ],
    )

    mode_store(
        name="coco_panoptic",
        dataset=dict(
            train=dict(
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                use_preprocessed_masks=True,
                preprocessed_mask_type="hipie",
                erode_dialate_preprocessed_masks=False,
                num_overlapping_masks=3,
                augmentation=dict(
                    initial_resolution=512,
                    center_crop=False,
                    reorder_segmentation=True,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=True,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((0.4, 1), (0.9, 1.1)),
                )
            ),
            val=dict(
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                use_preprocessed_masks=True,
                preprocessed_mask_type="hipie",
                erode_dialate_preprocessed_masks=False,
                num_overlapping_masks=3,
                augmentation=dict(
                    initial_resolution=512,
                    center_crop=True,
                    reorder_segmentation=True,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=False,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((0.6, 1.0), (1.0, 1.0)),
                )
            ),
        ),
        model=dict(
            segmentation_map_size=36,
            use_pad_mask_loss=False,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "coco_panoptic"},
        ],
    )


    mode_store(
        name="hypersim",
        dataset=dict(
            train=dict(
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                augmentation=dict(
                    initial_resolution=512,
                    center_crop=False,
                    reorder_segmentation=True,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=True,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((0.4, 1), (0.9, 1.1)),
                )
            ),
            val=dict(
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                augmentation=dict(
                    initial_resolution=512,
                    center_crop=True,
                    reorder_segmentation=True,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=False,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((0.6, 1.0), (1.0, 1.0)),
                )
            ),
        ),
        model=dict(
            segmentation_map_size=28,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "hypersim"},
        ],
    )

    mode_store(
        name="objaverse",
        dataset=dict(
            train=dict(
                augmentation=dict(
                    enable_random_resize_crop=False,
                    enable_horizontal_flip=False,
                )
            ),
            val=dict(
                augmentation=dict(
                    enable_random_resize_crop=False,
                    enable_horizontal_flip=False,
                )
            ),
        ),
        model=dict(
            segmentation_map_size=2,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "objaverse"},
        ],
    )

    mode_store(
        name="gt_coco_masks",
        dataset=dict(
            train=dict(
                preprocessed_mask_type=None,
                use_preprocessed_masks=False,
                object_ignore_threshold=0.0,
                top_n_masks_only=76,
                num_overlapping_masks=1,
                augmentation=dict(
                    reorder_segmentation=False,
                )
            ),
            val=dict(
                preprocessed_mask_type=None,
                use_preprocessed_masks=False,
                object_ignore_threshold=0.0,
                top_n_masks_only=76,
                augmentation=dict(
                    reorder_segmentation=False,
                )
            ),
        ),
    )

    mode_store(
        name="zoomed_coco_masks",
        dataset=dict(
            train=dict(
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    tgt_random_scale_ratio=((0.3, 0.8), (0.9, 1.1)),
                    enable_rand_augment=False,
                    enable_rotate=False,
                )
            ),
            val=dict(
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    tgt_random_scale_ratio=((0.3, 0.8), (0.9, 1.1)),
                    enable_rand_augment=False,
                    enable_rotate=False,
                )
            ),
        ),
    )

    mode_store(
        name="sam_coco_masks",
        dataset=dict(
            train=dict(
                object_ignore_threshold=0.1,
                top_n_masks_only=35,
                use_preprocessed_masks=True,
                preprocessed_mask_type="custom_postprocessed",
                erode_dialate_preprocessed_masks=False,
                num_overlapping_masks=2,
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    tgt_random_scale_ratio=((0.5, 0.9), (0.9, 1.1)),
                    enable_rand_augment=False,
                    enable_rotate=False,
                )
            ),
            val=dict(
                object_ignore_threshold=0.1,
                top_n_masks_only=35,
                use_preprocessed_masks=True,
                preprocessed_mask_type="custom_postprocessed",
                erode_dialate_preprocessed_masks=False,
                num_overlapping_masks=2,
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    tgt_random_scale_ratio=((0.5, 0.9), (1.0, 1.0)),
                    enable_rand_augment=False,
                    enable_rotate=False,
                )
            ),
        ),
    )

    mode_store(
        name="hypersim_multiview",
        dataset=dict(
            train=dict(
                camera_trajectory_window=32,
                return_different_views=True,
                bbox_overlap_threshold=0.65,
                bbox_area_threshold=0.75,
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                )
            ),
            val=dict(
                camera_trajectory_window=32,
                return_different_views=True,
                bbox_overlap_threshold=0.65,
                bbox_area_threshold=0.75,
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                )
            ),
        ),
        hydra_defaults=[
            "hypersim",
        ],
    )

    mode_store(
        name="scannetpp_multiview_dataset",
        model=dict(
            segmentation_map_size=36,
        ),
        dataset=dict(
            train=dict(
                image_pairs_per_scene=16384,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                )
            ),
            val=dict(
                image_pairs_per_scene=16384,
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                )
            ),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "scannetpp"},
        ],
    )

    mode_store(
        name="imagefolder",
        model=dict(
            encode_tgt_enc_norm=True,
        ),
        dataset=dict(
            train=builds(
                ImagefolderDataset,
                populate_full_signature=True,
                zen_partial=True,
                root=Path("/home/aswerdlo/data/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/custom/imagefolder/data_v0"),
                batch_size=1,
                augmentation=builds(
                    Augmentation,
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                    initial_resolution=512,
                    src_resolution=None,
                    tgt_resolution=None,
                    src_transforms="${get_src_transform:model}",
                    tgt_transforms="${get_tgt_transform:model}",
                    populate_full_signature=True,
                )
            ),
            val=builds(
                ImagefolderDataset,
                populate_full_signature=True,
                zen_partial=True,
                root=Path("/home/aswerdlo/data/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/custom/imagefolder/data_v0"),
                batch_size=1,
                augmentation=builds(
                    Augmentation,
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=True,
                    enable_random_resize_crop=False, 
                    enable_horizontal_flip=False,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                    initial_resolution=512,
                    src_resolution=None,
                    tgt_resolution=None,
                    src_transforms="${get_src_transform:model}",
                    tgt_transforms="${get_tgt_transform:model}",
                    populate_full_signature=True,
                )
            ),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagefolder"},
        ],
    )

