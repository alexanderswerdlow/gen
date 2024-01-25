from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import extract_model_from_parallel
from image_utils import Im

from gen.utils.decoupled_utils import is_main_process
from gen.utils.logging_utils import log_info

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

def handle_checkpointing_dirs(cfg: BaseConfig, prefix: str):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if cfg.trainer.checkpoints_total_limit is not None:
        checkpoints = os.listdir(cfg.checkpoint_dir)
        checkpoints = [d for d in checkpoints if d.startswith(f"{prefix}_")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= cfg.trainer.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - cfg.trainer.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            log_info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            log_info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(cfg.checkpoint_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)


@dataclass
class TrainingState:
    epoch_step: int  # Step in the current epoch. Resets every epoch.
    total_epoch_steps: int  # Total number of steps in the current epoch. [E.g., dataloader size on a single GPU]
    global_step: int  # Current number of steps which does not reset.
    epoch: int

class Trainable(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict):
        ...

    @abstractmethod
    def set_training_mode(self):
        ...

    @abstractmethod
    def set_inference_mode(self):
        ...

    @abstractmethod
    def checkpoint(self, accelerator: Accelerator, state: TrainingState, path: Path):
        ...

    def run_inference(self, batch: dict, state: TrainingState) -> dict[str, Im]:
        ...
    

def check_every_n_steps(state: TrainingState, n: int, run_first: bool = False, all_processes: bool = False):
    return (state.global_step % n == 0 and (run_first or state.global_step > 0)) and (is_main_process() or all_processes)


def check_every_n_epochs(state: TrainingState, n: Optional[int], run_first: bool = False, all_processes: bool = False):
    # Check if the current step is the last one in the epoch. We always want to run on the last step of the epoch. If we have n=5, then we run at the end of epochs 0 [if except_first == False], 5, 10, 15, etc.
    return (
        n is not None
        and (state.epoch_step == state.total_epoch_steps - 1)
        and ((state.epoch + 1) % n == 0 or (state.epoch == 0 and run_first))
        and (is_main_process() or all_processes)
    )


def every_n_steps(func, *wrapper_args, **wrapper_kwargs):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        if check_every_n_steps(state, *wrapper_args, **wrapper_kwargs):
            return func(*args, **kwargs)

    return wrapper


def every_n_epochs(func, *wrapper_args, **wrapper_kwargs):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        if check_every_n_epochs(state, *wrapper_args, **wrapper_kwargs):
            return func(*args, **kwargs)

    return wrapper


def unwrap(model):
    """
    In DDP/torch.compile and some other situations, our nn.Module is wrapped so to access class attributes we often need to unwrap it.
    """
    # equiv to. unwrap
    if PartialState._shared_state == {}:
        # Accelerate is initialized
        return extract_model_from_parallel(model)
    else:
        from torch.nn.parallel import DistributedDataParallel
        if isinstance(model, DistributedDataParallel):
            return model.module
        else:
            return model


if __name__ == "__main__":
    assert check_every_n_steps(TrainingState(epoch_step=0, total_epoch_steps=0, global_step=0, epoch=0), 10)
    assert not check_every_n_steps(TrainingState(epoch_step=0, total_epoch_steps=0, global_step=0, epoch=0), 10, run_first=True)
    assert check_every_n_steps(TrainingState(epoch_step=0, total_epoch_steps=0, global_step=10, epoch=0), 10)

    assert check_every_n_epochs(TrainingState(epoch_step=9, total_epoch_steps=10, global_step=0, epoch=0), 1)
    assert not check_every_n_epochs(TrainingState(epoch_step=9, total_epoch_steps=10, global_step=0, epoch=0), 1, run_first=True)
    assert check_every_n_epochs(TrainingState(epoch_step=9, total_epoch_steps=10, global_step=0, epoch=5), 5)
