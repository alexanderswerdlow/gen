import builtins
import logging
import os
import random
import shutil
import warnings
from pathlib import Path

import diffusers
import hydra
import numpy as np
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils import check_min_version
from hydra.utils import get_original_cwd
from hydra_zen import MISSING, ZenField, make_config, store
from image_utils import library_ops  # This overrides repr() for tensors
from ipdb import set_trace
from omegaconf import OmegaConf, open_dict
from tqdm.auto import tqdm

from gen.configs.base import BaseConfig
from gen.utils.decoupled_utils import check_gpu_memory_usage
from train import run

check_min_version("0.24.0")

builtins.st = set_trace # We import st everywhere
os.environ["HYDRA_FULL_ERROR"] = "1"

logger = get_logger(__name__)

@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: BaseConfig):
    with open_dict(cfg):
        cfg.cwd = str(get_original_cwd())

    # Hydra automatically changes the working directory, but we stay at the project directory.
    os.chdir(cfg.cwd)

    if cfg.attach:
        import subprocess

        import debugpy
        from image_utils import library_ops
        subprocess.run("kill -9 $(lsof -i :5678 | grep $(whoami) | awk '{print $2}')", shell=True)
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f'{cfg.exp}_' if cfg.exp else ''
    overfit_str = 'overfit_' if cfg.overfit else ''
    debug_str = 'debug_' if cfg.debug else ''
    cfg.run_name = f'{overfit_str}{debug_str}{exp_name}{datetime_str}'
    cfg.output_dir = cfg.top_level_output_path / ('debug' if cfg.debug else 'train') / cfg.run_name
    cfg.output_dir.mkdir(exist_ok=True, parents=True)

    original_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / '.hydra'
    if original_output_dir.exists():
        shutil.move(original_output_dir, cfg.output_dir)

    logging_dir = Path(cfg.output_dir, cfg.logging_dir)
    log_file_path = logging_dir / "output.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    logger.logger.addHandler(file_handler)

    print(OmegaConf.to_yaml(cfg))

    if cfg.trainer.seed is not None:
        np.random.seed(cfg.trainer.seed)
        random.seed(cfg.trainer.seed)
        torch.manual_seed(cfg.trainer.seed)
        cudnn.deterministic = False
        warnings.warn('We are seeding training but disabling the CUDNN deterministic setting for performance reasons.')
        
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.allow_tf32 = True # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        mixed_precision=cfg.trainer.mixed_precision,
        log_with=cfg.trainer.log_with,
        project_config=accelerator_project_config,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.tracker_project_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs=dict(
                wandb=dict(
                    name=cfg.run_name,
                    tags=cfg.tags,
                    dir=cfg.top_level_output_path,
                    sync_tensorboard=cfg.profile
                )
            )
        )
        wandb.run.log_code(include_fn=lambda path: any(path.endswith(f) for f in (
            '.py', '.yaml', '.yml', '.txt', '.md'
        )))

    check_gpu_memory_usage()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.trainer.seed is not None:
        set_seed(cfg.trainer.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # TODO: Verify this is what we want to do
    if cfg.trainer.scale_lr:
        cfg.trainer.learning_rate = (
            cfg.trainer.learning_rate * cfg.trainer.gradient_accumulation_steps * cfg.dataset.train_dataset.batch_size * accelerator.num_processes
        )

    if cfg.profile:
        torch.cuda.memory._record_memory_history()

    try:
        run(cfg, accelerator)
    except Exception as e:
        logger.error(e)
        if accelerator.is_main_process:
            print('Exception...')
            import sys
            import traceback

            import ipdb
            import lovely_tensors
            traceback.print_exc()
            ipdb.post_mortem(e.__traceback__)
            sys.exit(1)
        raise
    finally:
        pass
if __name__ == '__main__':
    main()
