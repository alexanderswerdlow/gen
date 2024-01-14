from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from hydra_zen import MISSING, builds, hydrated_dataclass, store
from gen import MOVI_OVERFIT_DATASET_PATH

from gen.configs.datasets import DatasetConfig
from gen.configs.hydra import get_hydra_config
from gen.configs.inference import InferenceConfig
from gen.configs.models import ModelConfig
from gen.configs.trainer import TrainerConfig
from gen.configs.utils import destructure_store, exp_store, mode_store

defaults = [
    "_self_",
    {"trainer": "base"},
    {"dataset": "coco_captions"},
    {"model": "basemapper"},
    {"inference": "basemapper"},
]


@dataclass
class BaseConfig:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    inference: InferenceConfig = MISSING

    top_level_output_path: Optional[Path] = Path("outputs")
    logging_dir: Optional[Path] = Path("logs")  # Folder inside the experiment folder
    exp: Optional[str] = None
    debug: bool = False
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False
    overfit: bool = False
    run_inference: bool = False

    # These are set in code
    output_dir: Optional[Path] = None
    run_name: Optional[str] = None
    cwd: Optional[Path] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)




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
    dataset=dict(num_validation_images=1, train_dataset=dict(batch_size=8), validation_dataset=dict(batch_size=1, random_subset=4)),
    model=dict(unfreeze_last_n_clip_layers=None, dropout_masks=0.2, enable_norm_scale=False, use_fixed_position_encoding=True, nested_dropout_prob=0),
    hydra_defaults=[
        "_self_",
        {"override /dataset": "movi_e"},
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
    # subset=("video_0015",),
    custom_split="train",
    path=MOVI_OVERFIT_DATASET_PATH,
    num_objects=1,
    augmentation=dict(minimal_source_augmentation=True, enable_crop=False),
)

mode_store(
    name="overfit_movi",
    debug=True,
    inference=dict(num_masks_to_remove=None),
    trainer=dict(gradient_accumulation_steps=4, num_train_epochs=10000, eval_every_n_steps=100, learning_rate=1e-3, eval_at_start=False, log_gradients=10),
    dataset=dict(
        train_dataset=dict(batch_size=8, random_subset=None, **shared_overfit_movi_args),
        validation_dataset=dict(random_subset=4, evenly_spaced_subset=False, **shared_overfit_movi_args),
        overfit=False,
    ),
)

destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()
