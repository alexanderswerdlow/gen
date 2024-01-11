from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from hydra_zen import MISSING, store

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


store(get_hydra_config(), group="hydra", name="default")

exp_store(
    name="gen",
    trainer=dict(num_train_epochs=1000, checkpointing_steps=1000, gradient_accumulation_steps=4, learning_rate=5e-5, eval_every_n_epochs=None, eval_every_n_steps=1000, tracker_project_name='gen'),
    dataset=dict(num_validation_images=1, train_dataset=dict(batch_size=8), validation_dataset=dict(batch_size=1, random_subset=4)),
    model=dict(unfreeze_last_n_clip_layers=6, dropout_masks=0.2),
    hydra_defaults=[
        "_self_",
        {"override /dataset": "movi_e"},
    ],
)

mode_store(
    name="fast",
    debug=True,
    trainer=dict(num_train_epochs=1, eval_every_n_epochs=1, eval_every_n_steps=None),
    dataset=dict(train_dataset=dict(batch_size=8, random_subset=16, num_workers=0), validation_dataset=dict(batch_size=1, random_subset=2, num_workers=0)),
)

mode_store(
    name="overfit",
    debug=True,
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

mode_store(
    name="overfit_movi",
    debug=True,
    trainer=dict(gradient_accumulation_steps=1),
    dataset=dict(
        train_dataset=dict(batch_size=4, random_subset=8, num_workers=0, subset=('video_0000',)),
        overfit=True,
    ),
)

destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()
