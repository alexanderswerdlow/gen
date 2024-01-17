from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from hydra_zen import MISSING, builds, store
from omegaconf import OmegaConf

from gen import IMAGENET_PATH, MOVI_OVERFIT_DATASET_PATH
from gen.configs.datasets import DatasetConfig
from gen.configs.hydra import get_hydra_config
from gen.configs.inference import InferenceConfig
from gen.configs.models import ModelConfig
from gen.configs.slurm import SlurmConfig
from gen.configs.trainer import TrainerConfig
from gen.configs.utils import destructure_store, exp_store, mode_store
from gen.utils.encoder_utils import ResNet50

defaults = [
    "_self_",
    {"trainer": "base"},
    {"dataset": "coco_captions"},
    {"model": "basemapper"},
    {"inference": "basemapper"},
    {"slurm": "default"},
]


@dataclass
class BaseConfig:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    inference: InferenceConfig = MISSING
    slurm: SlurmConfig = MISSING


    exp: Optional[str] = None
    debug: bool = False
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False
    overfit: bool = False
    run_inference: bool = False

    output_dir: Optional[Path] = None # Auto generated but can be specified
    first_level_output_path: Path = Path("outputs")
    second_level_output_path: Optional[Path] = None # Auto generated if not specified
    
    logging_dir: Path = Path("logs")  # Folder inside the experiment folder

    sweep_id: Optional[str] = None # ID of the entire sweep
    sweep_run_id: Optional[str] = None # ID of the specific run in a sweep
    reference_dir: Optional[Path] = None # Used to symlink slurm logs

    run_name: Optional[str] = None # Run name used for wandb
    cwd: Optional[Path] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)


def get_run_dir(_, *, _root_: BaseConfig):
    if _root_.output_dir is not None:
        return _root_.output_dir

    exp_name = f"{_root_.exp}_" if _root_.exp else ""
    overfit_str = "overfit_" if _root_.overfit else ""
    debug_str = "debug_" if _root_.debug else ""
    _root_.run_name = f"{overfit_str}{debug_str}{exp_name}"

    if _root_.second_level_output_path is None:
        _root_.second_level_output_path = "debug" if _root_.debug else ("inference" if _root_.run_inference else "train")

    if _root_.sweep_run_id is not None:
        _root_.sweep_id = str(_root_.sweep_id)
        _root_.sweep_run_id = str(_root_.sweep_run_id)

        _root_.run_name = _root_.run_name + "_" + _root_.sweep_id + "_" + _root_.sweep_run_id
        return Path(_root_.first_level_output_path) / _root_.second_level_output_path / _root_.sweep_id / _root_.sweep_run_id
    else:
        _root_.run_name = _root_.run_name + f'{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}'
        return Path(_root_.first_level_output_path) / _root_.second_level_output_path / _root_.run_name


OmegaConf.register_new_resolver("get_run_dir", get_run_dir)

store(get_hydra_config())

exp_store(
    name="gen",
    inference=dict(visualize_attention_map=True),
    trainer=dict(
        num_train_epochs=1000,
        checkpointing_steps=1000,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        eval_every_n_epochs=None,
        eval_every_n_steps=1000,
        tracker_project_name="gen",
        enable_dynamic_grad_accum=True,
    ),
    dataset=dict(
        num_validation_images=1,
        train_dataset=dict(batch_size=8),
        validation_dataset=dict(batch_size=1, random_subset=4),
        reset_validation_dataset_every_epoch=True,
    ),
    model=dict(),
    hydra_defaults=[
        "_self_",
        {"override /dataset": "movi_e"},
        {"override /model": "cross_attn"},
    ],
)

mode_store(
    name="fast",
    debug=True,
    trainer=dict(num_train_epochs=1, eval_every_n_epochs=1, eval_every_n_steps=None),
    dataset=dict(
        train_dataset=dict(batch_size=8, random_subset=16, num_workers=0), validation_dataset=dict(batch_size=1, random_subset=2, num_workers=0)
    ),
)

mode_store(
    name="overfit",
    debug=True,
    inference=dict(num_masks_to_remove=2, num_denoising_steps=50),
    trainer=dict(
        num_train_epochs=1000,
        eval_every_n_epochs=10,
        eval_every_n_steps=1000,
        checkpointing_steps=1000,
    ),
    dataset=dict(
        train_dataset=dict(batch_size=4, random_subset=8, num_workers=0),
        overfit=True,
    ),
)


def get_override_dict(**kwargs):
    return dict(
        train_dataset=dict(**kwargs),
        validation_dataset=dict(**kwargs),
    )


shared_overfit_movi_args = dict(
    custom_split="train",
    path=MOVI_OVERFIT_DATASET_PATH,
    num_objects=1,
    legacy_transforms=False,
    augmentation=dict(minimal_source_augmentation=True, enable_crop=False),
)

# Tmp debugging params
mode_store(
    name="overfit_movi",
    debug=True,
    inference=dict(num_masks_to_remove=None, visualize_attention_map=True, empty_string_cfg=True, guidance_scale=7.5),
    trainer=dict(
        gradient_accumulation_steps=4, num_train_epochs=10000, eval_every_n_steps=100, learning_rate=1e-3, eval_on_start=False, log_gradients=100
    ),
    dataset=dict(
        train_dataset=dict(batch_size=8, random_subset=None, **shared_overfit_movi_args),
        validation_dataset=dict(random_subset=4, evenly_spaced_subset=False, **shared_overfit_movi_args),
        overfit=False,
    ),
)


mode_store(
    name="overfit_movi_single_frame",
    dataset=dict(
        train_dataset=dict(num_dataset_frames=1, subset=("video_0015",)),
        validation_dataset=dict(num_dataset_frames=1, subset=("video_0015",)),
    ),
)

mode_store(
    name="overfit_imagenet",
    model=dict(use_dataset_segmentation=False),
    dataset=dict(
        train_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=True)),
        validation_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=False, enable_horizontal_flip=False), random_subset=4),
        overfit=False,
    ),
    hydra_defaults=[
        "_self_",
        {"override /dataset": "imagenet"},
    ],
)

mode_store(
    name="controlnet",
    model=dict(controlnet=True),
    trainer=dict(learning_rate=5e-6, scale_lr_batch_size=True),
    dataset=dict(
        train_dataset=dict(batch_size=4),
    ),
)

mode_store(
    name="cls_token_only",
    model=dict(use_cls_token_only=True, mask_cross_attn=False, cross_attn_dim=768),
)

mode_store(name="resnet", model=dict(encoder=builds(ResNet50, populate_full_signature=False), cross_attn_dim=256))

mode_store(
    name="should_work",
    inference=dict(guidance_scale=0.0),
    model=dict(use_dataset_segmentation=True, mask_cross_attn=False, use_cls_token_only=True),
    trainer=dict(learning_rate=1e-5),
)

destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()
