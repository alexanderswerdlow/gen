from collections import namedtuple
from dataclasses import dataclass, field

from pathlib import Path
from typing import Any, List, Optional

from hydra_zen import MISSING, store

from gen.configs.datasets import DatasetConfig
from gen.configs.hydra import get_hydra_config
from gen.configs.models import ModelConfig
from gen.configs.trainer import TrainerConfig
from gen.configs.utils import destructure_store, mode_store, exp_store

defaults = [
    "_self_",
    {"trainer": "base"},
    {"dataset": "coco_captions"},
    {"model": "basemapper"},
]


@dataclass
class BaseConfig:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING

    top_level_output_path: Optional[Path] = Path("outputs")
    logging_dir: Optional[Path] = Path("logs")  # Folder inside the experiment folder
    exp: Optional[str] = None
    debug: bool = False
    tracker_project_name: str = "controlnet"  # wandb project name
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False
    overfit: bool = False

    # These are set in code
    output_dir: Optional[Path] = None
    run_name: Optional[str] = None
    cwd: Optional[Path] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)


store(get_hydra_config(), group="hydra", name="default")

exp_store(
    name="demo_exp",
    trainer=dict(num_train_epochs=1000, num_val_steps=500, checkpointing_steps=10000),
    dataset=dict(
        num_validation_images=1,
        train_dataset=dict(batch_size=8),
        validation_dataset=dict(batch_size=1, random_subset=4)
    ),
    hydra_defaults=[
        "_self_",
        {"override /dataset": "coco_captions"},
    ],
)

mode_store(
    name="fast", 
    debug=True, 
    trainer=dict(
        num_train_epochs=1, 
        num_val_steps=2
    ),
    dataset=dict(
        train_dataset=dict(
            batch_size=2, 
            random_subset=8, 
            num_workers=0
        ), 
        validation_dataset=dict(
            batch_size=1, 
            random_subset=2, 
            num_workers=0
        )
    ),
)

mode_store(
    name="overfit", 
    debug=True, 
    trainer=dict(
        num_train_epochs=100, 
        eval_every_n_epochs=10,
        num_val_steps=1000,
        checkpointing_steps=1000,
    ),
    dataset=dict(
        train_dataset=dict(
            batch_size=4, 
            random_subset=8, 
            num_workers=0
        ),
        overfit=True,
    ),
)

destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()
