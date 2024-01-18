import autoroot

import os
import random
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
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration
from diffusers.utils import check_min_version
from hydra.utils import get_original_cwd
from image_utils import library_ops  # This overrides repr() for tensors
from omegaconf import OmegaConf, open_dict

from gen.configs.base import BaseConfig
from gen.utils.decoupled_utils import check_gpu_memory_usage, get_num_gpus, is_main_process, set_global_breakpoint
from gen.utils.logging_utils import log_error, log_info, log_warn, set_logger
from inference import inference
from train import train

check_min_version("0.25.0")

os.environ["HYDRA_FULL_ERROR"] = "1"

set_global_breakpoint()  # Overrides breakpoint() to use ipdb.set_trace() instead and handle distributed training
set_logger(__name__)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: BaseConfig):
    with open_dict(cfg):
        cfg.cwd = str(get_original_cwd())

    # Hydra automatically changes the working directory, but we stay at the project directory.
    os.chdir(cfg.cwd)

    if cfg.attach:
        import subprocess

        import debugpy

        subprocess.run("kill -9 $(lsof -i :5678 | grep $(whoami) | awk '{print $2}')", shell=True)
        debugpy.listen(5678)
        log_info("Waiting for debugger attach")
        debugpy.wait_for_client()

    cfg.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    # log_file_path = logging_dir / "output.log"
    # log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # set_log_file(log_file_path)

    if is_main_process() and cfg.reference_dir is not None:
        try:
            reference_dir = Path(cfg.reference_dir)
            assert reference_dir.exists()
            symlink_dir = cfg.output_dir / "slurm"
            symlink_dir.symlink_to(reference_dir)
        except:
            log_warn(f"Could not symlink {reference_dir} to {symlink_dir}")

    if cfg.trainer.seed is not None:
        np.random.seed(cfg.trainer.seed)
        random.seed(cfg.trainer.seed)
        torch.manual_seed(cfg.trainer.seed)
        torch.cuda.manual_seed_all(cfg.trainer.seed)
        cudnn.deterministic = False
        log_warn("We are seeding training but disabling the CUDNN deterministic setting for performance reasons.")

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.allow_tf32 = True  # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    num_gpus = get_num_gpus()
    if cfg.trainer.enable_dynamic_grad_accum:
        if num_gpus > cfg.trainer.dynamic_grad_accum_default_gpus:
            log_warn(f"You are using more GPUs than {cfg.trainer.dynamic_grad_accum_default_gpus} GPUs. Disabling dynamic gradient accumulation")
        else:
            assert cfg.trainer.dynamic_grad_accum_default_gpus % num_gpus == 0
            grad_accum_factor = int(cfg.trainer.dynamic_grad_accum_default_gpus / num_gpus)
            cfg.trainer.gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps * grad_accum_factor
            log_info(
                f"Using dynamic gradient accumulation with {num_gpus} GPUs so scaling by {grad_accum_factor} to {cfg.trainer.gradient_accumulation_steps} gradient accumulation steps."
            )

    if cfg.trainer.scale_lr_gpus_grad_accum:
        # For n GPUs, we have an effective xN batch size so we need to scale the learning rate.
        # Similarly, if we accumulate gradients (e.g., training on 1 GPU), we need to scale the learning rate.
        scale_factor = num_gpus * cfg.trainer.gradient_accumulation_steps
        cfg.trainer.learning_rate = cfg.trainer.learning_rate * scale_factor
        log_info(
            f"Scaling learning rate by {scale_factor} for {num_gpus} GPUs and {cfg.trainer.gradient_accumulation_steps} gradient accumulation steps. Final LR: {cfg.trainer.learning_rate}."
        )

    if cfg.trainer.scale_lr_batch_size:
        cfg.trainer.learning_rate = cfg.trainer.learning_rate * cfg.dataset.train_dataset.batch_size
        log_info(f"Scaling learning rate by {cfg.dataset.train_dataset.batch_size} to {cfg.trainer.learning_rate}.")

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=cfg.trainer.gradient_accumulation_steps, adjust_scheduler=False)
    accelerator = Accelerator(
        mixed_precision=cfg.trainer.mixed_precision,
        log_with=cfg.trainer.log_with,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    assert accelerator.num_processes == num_gpus
    cfg.trainer.num_gpus = accelerator.num_processes

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_main_process():
        accelerator.init_trackers(
            cfg.trainer.tracker_project_name + ("_inference" if cfg.run_inference else ""),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs=dict(wandb=dict(name=cfg.run_name, tags=cfg.tags, dir=cfg.output_dir, sync_tensorboard=cfg.profile)),
        )
        wandb.run.log_code(include_fn=lambda path: any(path.endswith(f) for f in (".py", ".yaml", ".yml", ".txt", ".md")))
        log_info(OmegaConf.to_yaml(cfg))

    check_gpu_memory_usage()

    log_info(accelerator.state, main_process_only=False)

    if is_main_process():
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.profile:
        torch.cuda.memory._record_memory_history()

    try:
        if cfg.run_inference:
            inference(cfg, accelerator)
        else:
            train(cfg, accelerator)

    except Exception as e:
        log_error(e)
        if is_main_process():
            log_info("Exception...")
            import sys
            import traceback

            import ipdb

            traceback.print_exc()
            ipdb.post_mortem(e.__traceback__)
            sys.exit(1)
        raise
    finally:
        pass


if __name__ == "__main__":
    main()
