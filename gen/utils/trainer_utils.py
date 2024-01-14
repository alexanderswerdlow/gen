import os
import shutil
from dataclasses import dataclass
from functools import wraps

from accelerate import Accelerator
from gen.utils.logging_utils import log_info

from gen.configs.base import BaseConfig




def handle_checkpointing(cfg: BaseConfig, accelerator: Accelerator, global_step: int):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if cfg.trainer.checkpoints_total_limit is not None:
        checkpoints = os.listdir(cfg.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= cfg.trainer.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - cfg.trainer.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            log_info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            log_info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    log_info(f"Saved state to {save_path}")

@dataclass
class TrainingState:
    epoch_step: int
    total_epoch_steps: int
    global_step: int
    epoch: int
    accelerator: Accelerator


def every_n_steps(func, n: int, except_first: bool = False, all_processes: bool = False):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        if (state.global_step % n == 0 and (not except_first or state.global_step > 0)) and (state.accelerator.is_local_main_process or not state.accelerator.use_distributed or all_processes):
            return func(*args, **kwargs)

    return wrapper

def every_n_epochs(func, n: int, except_first: bool = False, all_processes: bool = False):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        # Check if the current step is the last one in the epoch
        if (state.epoch_step == state.total_epoch_steps - 1) and ((state.epoch + 1) % n == 0 or (not except_first)) and (state.accelerator.is_local_main_process or not state.accelerator.use_distributed or all_processes):
            return func(*args, **kwargs)

    return wrapper


def custom_ddp_unwrap(model):
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(model, DistributedDataParallel):
        return model.module
    else:
       return model
