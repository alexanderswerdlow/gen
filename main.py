import autoroot
import os
import pickle
import random
import string
from pathlib import Path

import cloudpickle
import diffusers
import hydra
import numpy as np
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration
from diffusers.utils import check_min_version
from hydra.utils import get_original_cwd
from image_utils import library_ops  # This overrides repr() for tensors
from omegaconf import OmegaConf, open_dict

from gen.configs.base import BaseConfig
from gen.utils.decoupled_utils import check_gpu_memory_usage, get_num_gpus, get_rank, is_main_process, set_global_breakpoint
from gen.utils.logging_utils import log_error, log_info, log_warn, set_logger
from inference import inference
from train import Trainer

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

    assert not cfg.trainer.resume or cfg.trainer.ckpt is not None, "You must specify a checkpoint to resume from."

    if cfg.trainer.resume:
        if Path(cfg.trainer.ckpt).is_file():
            top_level_dir = Path(cfg.trainer.ckpt).parent.parent.parent
        else:
            top_level_dir = Path(cfg.trainer.ckpt).parent.parent
        with open(top_level_dir / '.hydra' / 'final_config.pkl', "rb") as f:
            loaded_cfg = pickle.load(f)

        # We take these old params and overwrite everything else
        cfg.wandb_run_id = loaded_cfg.wandb_run_id
        cfg.run_name = loaded_cfg.run_name
        cfg.output_dir = loaded_cfg.output_dir
        cfg.sweep_id = loaded_cfg.sweep_id
        cfg.sweep_run_id = loaded_cfg.sweep_run_id
    
    cfg.logging_dir = Path(cfg.output_dir, cfg.logging_dir)
    if is_main_process():
        if cfg.checkpoint_dir.is_absolute():
            cfg.checkpoint_dir = cfg.checkpoint_dir / cfg.output_dir.name
            if cfg.checkpoint_dir.exists() and not cfg.trainer.resume:
                cfg.checkpoint_dir = cfg.checkpoint_dir / "".join(random.choices(string.ascii_letters, k=10))
            
            cfg.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            symlink_dir = cfg.output_dir / "checkpoints"
            symlink_dir.symlink_to(cfg.checkpoint_dir)
        else:
            cfg.checkpoint_dir = Path(cfg.output_dir, cfg.checkpoint_dir)
    else:
        cfg.checkpoint_dir = None

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

    cfg.trainer.seed = cfg.trainer.seed + int(get_rank())
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
    torch.set_float32_matmul_precision("high")
    log_warn("Setting matmul precision to high. Setting to medium may be faster.")

    if cfg.trainer.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        log_warn("Setting anomaly detection to True. This will slow down training.")

    if cfg.profile:
        cfg.trainer.gradient_accumulation_steps = 1
        cfg.trainer.enable_dynamic_grad_accum = False
        log_warn("Disabling all gradient accumulation for profiling!")

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

    cfg.trainer.initial_learning_rate = cfg.trainer.learning_rate
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

    if cfg.trainer.finetune_learning_rate is not None:
        scale_factor = cfg.trainer.learning_rate / cfg.trainer.initial_learning_rate
        cfg.trainer.finetune_learning_rate = scale_factor * cfg.trainer.finetune_learning_rate
        log_info(f"Scaling finetuning learning rate by {scale_factor} to {cfg.trainer.finetune_learning_rate}.")

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=cfg.logging_dir)
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=cfg.trainer.gradient_accumulation_steps, adjust_scheduler=False)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.trainer.find_unused_parameters)
    accelerator = Accelerator(
        mixed_precision=cfg.trainer.mixed_precision,
        log_with=cfg.trainer.log_with,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        kwargs_handlers=[kwargs]
    )
    assert accelerator.num_processes == num_gpus
    cfg.trainer.num_gpus = accelerator.num_processes

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    cfg.trainer.dtype = str(weight_dtype)
    cfg.trainer.device = accelerator.device

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_main_process():
        wandb_kwargs = dict(name=cfg.run_name, tags=cfg.tags, dir=cfg.first_level_output_path, sync_tensorboard=cfg.profile)
        if cfg.wandb_run_id is None:
            cfg.wandb_run_id = wandb.util.generate_id()
        
        wandb_kwargs['id'] = cfg.wandb_run_id

        if cfg.trainer.resume:
            wandb_kwargs['resume'] = 'must'
        
        project_name_suffix = "_inference" if cfg.run_inference else ""
        project_name_suffix = "_debug" if cfg.debug else project_name_suffix
        cfg.trainer.tracker_project_name = f"{cfg.trainer.tracker_project_name}{project_name_suffix}"
        accelerator.init_trackers(
            cfg.trainer.tracker_project_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            init_kwargs=dict(wandb=wandb_kwargs),
        )
        wandb.run.log_code(include_fn=lambda path: any(path.endswith(f) for f in (".py", ".yaml", ".yml", ".txt", ".md")), exclude_fn=lambda path: "outputs/" in path)
        cfg.wandb_url = wandb.run.get_url()

        log_info(OmegaConf.to_yaml(cfg))
        
        with open(cfg.output_dir / '.hydra' / 'final_config.pkl', "wb") as f:
            cloudpickle.dump(cfg, f)
        
        # If the code changes, we may be unable to load the pickled config so we also save the yaml.
        OmegaConf.save(config=cfg, f=cfg.output_dir / '.hydra' / 'final_config.yaml', resolve=False)

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
            train = Trainer(cfg=cfg, accelerator=accelerator)
            train.train()

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
