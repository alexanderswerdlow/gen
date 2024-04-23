from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

from hydra_zen import MISSING, store
from omegaconf import OmegaConf
from gen import CHECKPOINT_DIR

from gen.configs.datasets import DatasetConfig
from gen.configs.experiments import get_experiments
from gen.configs.hydra import get_hydra_config
from gen.configs.inference import InferenceConfig
from gen.configs.models import ModelConfig
from gen.configs.trainer import TrainerConfig
from gen.configs.utils import destructure_store, exp_store, mode_store
import hydra

from gen.datasets.run_dataloader import iterate_dataloader
from gen.models.encoders.encoder import TimmModel
from gen.utils.logging_utils import log_info

defaults = [
    "_self_",
    {"trainer": "base"},
    {"dataset": "imagefolder"},
    {"model": "basemapper"},
    {"inference": "basemapper"},
]


@dataclass
class BaseConfig:
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    inference: InferenceConfig = MISSING

    exp: Optional[str] = None
    debug: bool = False
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False
    overfit: bool = False
    run_inference: bool = False
    run_dataloader_only: bool = False

    output_dir: Optional[Path] = None  # Auto generated but can be specified
    first_level_output_path: Path = Path("outputs")
    second_level_output_path: Optional[Path] = None  # Auto generated if not specified

    logging_dir: Path = Path("logs")  # Folder inside the experiment folder
    checkpoint_dir: Path = CHECKPOINT_DIR

    sweep_id: Optional[str] = None  # ID of the entire sweep
    sweep_run_id: Optional[str] = None  # ID of the specific run in a sweep
    reference_dir: Optional[Path] = None  # Used to symlink slurm logs

    run_name: Optional[str] = None  # Run name used for wandb
    wandb_url: Optional[str] = None  # Set in code
    wandb_run_id: Optional[str] = None  # Set in code
    cwd: Optional[Path] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)


def get_run_dir(_, *, _root_: BaseConfig):
    if _root_.output_dir is not None:
        return _root_.output_dir

    exp_name = f"{_root_.exp}" if _root_.exp else ""
    overfit_str = "overfit_" if _root_.overfit else ""
    debug_str = "debug_" if _root_.debug else ""
    _root_.run_name = f"{overfit_str}{debug_str}{exp_name}"

    if _root_.second_level_output_path is None:
        _root_.second_level_output_path = "debug" if _root_.debug else ("inference" if _root_.run_inference else "train")

    if _root_.sweep_run_id is not None:
        _root_.sweep_id = str(_root_.sweep_id)
        _root_.sweep_run_id = str(_root_.sweep_run_id)

        _root_.run_name = f"{_root_.run_name}_{_root_.sweep_id}_{_root_.sweep_run_id}"
        return Path(_root_.first_level_output_path) / _root_.second_level_output_path / _root_.sweep_id / _root_.sweep_run_id
    else:
        _root_.run_name = _root_.run_name + f'{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}'
        return Path(_root_.first_level_output_path) / _root_.second_level_output_path / _root_.run_name

def get_src_transform(_, *, _root_: BaseConfig):
    """
    We instantiate the model container (without actually loading the params) to get the transforms.
    """
    model: TimmModel = hydra.utils.instantiate(OmegaConf.to_container(_root_.model.encoder), deferred_init=True)
    return model.transform

def get_tgt_transform(_, *, _root_: BaseConfig):
    from gen.datasets.utils import get_stable_diffusion_transforms
    return get_stable_diffusion_transforms(resolution=_root_.model.decoder_resolution)
    
def get_dtype(_, *, _root_: BaseConfig):
    from accelerate.utils import PrecisionType
    import torch
    if _root_.trainer.mixed_precision == PrecisionType.BF16:
        return torch.bfloat16
    elif _root_.trainer.mixed_precision == PrecisionType.FP16:
        return torch.float16
    else:
        return torch.float32

OmegaConf.register_new_resolver("get_run_dir", get_run_dir)
OmegaConf.register_new_resolver("get_src_transform", get_src_transform)
OmegaConf.register_new_resolver("get_tgt_transform", get_tgt_transform)
OmegaConf.register_new_resolver("get_dtype", get_dtype)
OmegaConf.register_new_resolver("eval", eval)

store(get_hydra_config())
exp_store(
    name="gen",
    trainer=dict(
        tracker_project_name="gen",
        num_train_epochs=10000,
        max_train_steps=100000,
        eval_every_n_epochs=None,
        eval_every_n_steps=500,
        checkpointing_steps=1000,
        checkpoints_total_limit=1,
        save_accelerator_format=True,
        learning_rate=4e-7,
        lr_scheduler="constant_with_warmup",
        gradient_accumulation_steps=4,
        enable_dynamic_grad_accum=True,
        scale_lr_batch_size=True,
        log_gradients=100,
        gradient_checkpointing=True,
        compile=False,
        validate_training_dataset=True,
        inference_train_switch=False,
    ),
    dataset=dict(
        num_validation_images=1,
        train=dict(num_workers=4, batch_size=8),
        val=dict(batch_size=1, subset_size=4),
        reset_val_dataset_every_epoch=True,
    ),
    inference=dict(
        guidance_scale=7.5,
        use_ddim=True,
        max_batch_size=4,
    ),
    hydra_defaults=[
        "_self_",
    ],
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
        train=dict(batch_size=4, subset_size=8, num_workers=0),
        overfit=True,
    ),
)

get_experiments()
destructure_store(BaseConfig, name="config")
store.add_to_hydra_store()
