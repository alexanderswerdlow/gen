from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import extract_model_from_parallel
from image_utils import Im

from gen.utils.decoupled_utils import is_main_process
from gen.utils.logging_utils import log_info, log_error


if TYPE_CHECKING:
    from gen.configs.base import BaseConfig


def load_from_ckpt(cfg: BaseConfig, accelerator: Accelerator, model: nn.Module, load_model: bool) -> int:
    """
    Loads the model [or just returns the checkpoint global step]
    """
    if cfg.trainer.ckpt == "latest":
        # Get the most recent checkpoint
        dirs = os.listdir(cfg.checkpoint_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = Path(cfg.trainer.ckpt)

    if path.is_dir() and any(child.is_dir() and child.name == "state" for child in path.iterdir()):
        path = path / "state"

    if path is None:
        log_error(f"Checkpoint '{cfg.trainer.ckpt}' does not exist. Exiting.")
        raise FileNotFoundError
    else:
        log_info(f"Resuming from checkpoint {path}")

        # TODO: @Tsung-Wei Ke tested this and found that it doesn't work, at least in some of the cases we used.
        # We should see if we can still load the optimizer states.

        # from accelerate.utils.modeling import load_checkpoint_in_model
        # if path.is_file() or cfg.trainer.load_weights_only_no_state:
        #     load_checkpoint_in_model(model, str(path))
        # else:
        #     accelerator.load_state(path)

        if load_model:
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict, strict=cfg.trainer.strict_load)
        try:
            if path.is_file():
                global_step = int(path.parent.parent.name.split("_")[-1])
            else:
                global_step = int(path.name.split("_")[-1] if "_" in path.name else path.parent.name.split("_")[-1])
        except:
            log_error(f"Could not parse global step from checkpoint path {path}. Setting to 0.")
            global_step = 0

        # first_epoch = global_step // num_update_steps_per_epoch
        first_epoch = 0
        log_info(f"Continuing from epoch {first_epoch} and global step {global_step}")
        return global_step


def handle_checkpointing_dirs(cfg: BaseConfig, prefix: str):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if cfg.trainer.checkpoints_total_limit is not None:
        if not os.path.exists(cfg.checkpoint_dir):
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
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
    num_epoch_steps: int  # Total number of steps in the current epoch. [E.g., dataloader size on a single GPU]
    global_step: int  # Current number of steps which does not reset.
    true_step: int
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

    def on_sync_gradients(self):
        pass

    def get_param_groups(self) -> Optional[dict[str, Any]]:
        return None
    
    def process_input(self, batch: dict) -> Any:
        return batch


def check_every_n_steps(
    state: TrainingState,
    n: Optional[int],
    run_first: bool = False,
    all_processes: bool = False,
    decay_steps: bool = False,
    max_eval_interval: Optional[int] = None,
    decrease_n_runs: Optional[int] = None,
):
    if n is None: return False
    if decay_steps:
        max_eval_interval = max_eval_interval or n * 5
        decrease_n_runs = decrease_n_runs or 5
        n = min(n * ((state.global_step // (decrease_n_runs * n)) + 1), max_eval_interval)
    return (state.global_step % n == 0 and (run_first or state.global_step > 0)) and (is_main_process() or all_processes)


def check_every_n_epochs(state: TrainingState, n: Optional[int], run_first: bool = False, all_processes: bool = False):
    # Check if the current step is the last one in the epoch. We always want to run on the last step of the epoch. If we have n=5, then we run at the end of epochs 0 [if except_first == False], 5, 10, 15, etc.
    return (
        n is not None
        and (state.epoch_step == state.num_epoch_steps - 1)
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

def print_memory(verbose: bool = False):
    max_cur, max_peak = -1, -1
    max_cur_device, max_peak_device = -1, -1
    for device in range(torch.cuda.device_count()):
        current_reserved_memory_MB = torch.cuda.memory_reserved(device=torch.device(f'cuda:{device}')) / (2**20)
        peak_reserved_memory_MB = torch.cuda.max_memory_reserved(device=torch.device(f'cuda:{device}')) / (2**20)

        if current_reserved_memory_MB > max_cur:
            max_cur = current_reserved_memory_MB
            max_cur_device = device
        
        if peak_reserved_memory_MB > max_peak:
            max_peak = peak_reserved_memory_MB
            max_peak_device = device

    if verbose:
        log_info(torch.cuda.memory_summary(abbreviated=False))
    log_info(f"GPU Memory Current: {max_cur:.2f} MB on rank {max_cur_device}, Peak Reserved: {max_peak:.2f} MB on rank {max_peak_device}")
    

if __name__ == "__main__":
    # assert check_every_n_steps(TrainingState(epoch_step=0, num_epoch_steps=0, global_step=0, epoch=0), 10)
    # assert not check_every_n_steps(TrainingState(epoch_step=0, num_epoch_steps=0, global_step=0, epoch=0), 10, run_first=True)
    # assert check_every_n_steps(TrainingState(epoch_step=0, num_epoch_steps=0, global_step=10, epoch=0), 10)

    # assert check_every_n_epochs(TrainingState(epoch_step=9, num_epoch_steps=10, global_step=0, epoch=0), 1)
    # assert not check_every_n_epochs(TrainingState(epoch_step=9, num_epoch_steps=10, global_step=0, epoch=0), 1, run_first=True)
    # assert check_every_n_epochs(TrainingState(epoch_step=9, num_epoch_steps=10, global_step=0, epoch=5), 5)

    for i in range(50000):
        if check_every_n_steps(TrainingState(epoch_step=i, num_epoch_steps=10, global_step=i, epoch=0, true_step=i), 500, decay_steps=True):
            print(i)
            