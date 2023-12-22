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
    {"dataset": "huggingface"},
    {"model": "basemapper"},
]

@dataclass
class BaseConfig:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING

    top_level_output_path: Optional[Path] = Path("outputs")
    logging_dir: Optional[Path] = Path("logs") # Folder inside the experiment folder
    exp: Optional[str] = None
    debug: bool = False
    tracker_project_name: str = "controlnet" # wandb project name
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False

    # These are set in code
    output_dir: Optional[Path] = None
    run_name: Optional[str] = None
    cwd: Optional[Path] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)

store(get_hydra_config(), group="hydra", name="default")

exp_store(name="demo_exp", trainer=dict(seed=0))
mode_store(name="overfit", debug=True, trainer=dict(max_epochs=1))

destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()