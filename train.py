load_time = __import__("time").time()

from functools import partial
from importlib.util import find_spec
import itertools
import math
import os
import types
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Iterable, Union
import importlib
import hydra
import torch
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import wandb
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from gen.configs import BaseConfig, ModelType
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.models.base.base_model import BaseMapper
from gen.models.utils import get_model_from_cfg, set_default_inference_func, set_inference_func
from gen.utils.decoupled_utils import Profiler, get_num_gpus, get_rank, is_main_process, save_memory_profile, show_memory_usage, try_except, write_to_file, print_memory
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.trainer_utils import (
    Trainable,
    TrainingState,
    check_every_n_epochs,
    check_every_n_steps,
    handle_checkpointing_dirs,
    load_from_ckpt,
    unwrap,
)
from inference import run_inference_dataloader
from hydra.utils import instantiate

def trainable_parameters(module, requires_grad: bool):
    for name, param in module.named_parameters():
        if param.requires_grad or requires_grad is False:
            yield name, param


def get_named_params(models: tuple[Union[nn.Module, dict]], requires_grad=True):
    return dict(
        itertools.chain(
            *(trainable_parameters(model, requires_grad=requires_grad) for model in models if isinstance(model, nn.Module)),
            *(np.items() for np in models if isinstance(np, dict)),
        )
    )


def validate_params(cfg: BaseConfig, models: Iterable[nn.Module], dtype: torch.dtype):
    # In general, we want all trainable params in FP32 and all non-trainable params possibly in BF16
    num_requires_grad, num_no_grad = 0, 0
    for k, p in get_named_params(models, requires_grad=False).items():
        if p.requires_grad:
            assert p.dtype == torch.float32, f"Param {k} is trainable but not in {torch.float32}"
            num_requires_grad += 1
        elif not p.requires_grad:
            num_no_grad += 1
            if any(k.startswith(prefix) for prefix in cfg.trainer.param_dtype_exception_prefixes):
                continue
            assert p.dtype == dtype, f"Param {k} is non-trainable but not in {dtype}"
            
    log_info(f"Found {num_requires_grad} trainable {torch.float32} and {num_no_grad} non-trainable {dtype} params.")

class Trainer:
    def __init__(self, cfg: BaseConfig, accelerator: Accelerator):
        self.cfg = cfg
        self.accelerator = accelerator

        self.dtype = getattr(torch, self.cfg.trainer.dtype.split(".")[-1])

        self.model: Trainable = None  # Generally, we try to have a single top-level nn.Module that contains all the other models
        self.models: list[Union[nn.Module, dict]] = None  # We support multiple models or dicts of named parameters if necessary
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.tokenizer: AutoTokenizer = None

        self.train_dataloader: DataLoader = None
        self.validation_dataloader: DataLoader = None

        if self.cfg.trainer.set_even_batches_false:
            accelerator.even_batches = False

        self.init_models()
        self.init_dataloader()
        self.init_optimizer()
        self.init_lr_scheduler()

    def init_models(self):
        # TODO: Define a better interface for different models once we get a better idea of the requires inputs/outputs
        # Right now we just conditionally call methods in the respective files based on the model_type enum
        assert is_xformers_available()

        self.models = []
        match self.cfg.model.model_type:
            case ModelType.BASE_MAPPER:
                model = get_model_from_cfg(self.cfg)
                self.models.append(model)

                # TODO: We also call load_from_ckpt in train() [but load_model is False]. This may be due to an issue calling it at that location.
                if self.cfg.trainer.ckpt:
                    _ = load_from_ckpt(cfg=self.cfg, accelerator=self.accelerator, model=model, load_model=True, load_accelerator_state=self.cfg.trainer.load_accelerator_state)

                self.model = self.accelerator.prepare(model)
                self.tokenizer = unwrap(model).tokenizer

                if is_main_process():
                    summary(model, col_names=("trainable", "num_params"), depth=3)

                if (param_groups_ := unwrap(self.model).get_param_groups()) is not None:
                    assert len([p for d_ in param_groups_ for p in list(d_["params"])]) == len(get_named_params(self.models).values())

        validate_params(self.cfg, self.models, self.dtype)

    def init_dataloader(self):
        log_info("Creating train_dataset + self.train_dataloader")
        self.train_dataloader_holder: AbstractDataset = instantiate(self.cfg.dataset.train)(cfg=self.cfg, split=Split.TRAIN, tokenizer=self.tokenizer)

        additional_train_datasets = None
        if exists(self.cfg.dataset.additional_train):
            additional_train_datasets = {dataset_name: instantiate(dataset_cfg)(cfg=self.cfg, split=Split.TRAIN, tokenizer=self.tokenizer) for dataset_name, dataset_cfg in self.cfg.dataset.additional_train.items()}

        self.train_dataloader: DataLoader = self.train_dataloader_holder.get_dataloader(additional_datasets=additional_train_datasets, pin_memory=True)
        assert len(self.train_dataloader) > 0

        log_info("Creating val_dataset + self.validation_dataloader")
        self.val_dataset_holder: AbstractDataset = instantiate(self.cfg.dataset.val)(cfg=self.cfg, split=Split.VALIDATION, tokenizer=self.tokenizer)

        additional_val_datasets = None
        if exists(self.cfg.dataset.additional_val):
            additional_val_datasets = {dataset_name: instantiate(dataset_cfg)(cfg=self.cfg, split=Split.VALIDATION, tokenizer=self.tokenizer) for dataset_name, dataset_cfg in self.cfg.dataset.additional_val.items()}

        if self.cfg.dataset.overfit: self.val_dataset_holder.get_dataset = lambda: self.train_dataloader.dataset

        g = torch.Generator()
        g.manual_seed(0 + get_rank())
        self.val_dataset_holder.batch_size = self.cfg.dataset.val.batch_size
        self.val_dataset_holder.subset_size = max(self.cfg.trainer.num_gpus * self.val_dataset_holder.batch_size, self.cfg.dataset.val.batch_size)
        self.validation_dataloader = self.val_dataset_holder.get_dataloader(generator=g, pin_memory=False, additional_datasets=additional_val_datasets)
        self.validation_dataloader = self.accelerator.prepare_data_loader(self.validation_dataloader, device_placement=False)

        assert len(self.validation_dataloader) > 0

    def init_optimizer(self):        
        kwargs = dict()
        module_name, class_name = self.cfg.trainer.optimizer_cls.path.rsplit(".", 1)
        optimizer_class = getattr(importlib.import_module(module_name), class_name)
            
        if optimizer_class == torch.optim.AdamW:
            kwargs['eps'] = self.cfg.trainer.adam_epsilon
            kwargs['betas'] = (self.cfg.trainer.adam_beta1, self.cfg.trainer.adam_beta2)

        # Check that both fused and 8bit are not both true
        assert not (self.cfg.trainer.use_fused_adam and self.cfg.trainer.use_8bit_adam)

        if self.cfg.trainer.use_fused_adam and find_spec("apex"):
            from apex.optimizers import FusedAdam
            optimizer_class = FusedAdam
            log_info("Using FusedAdam...")

        if self.cfg.trainer.use_8bit_adam and find_spec("bitsandbytes"):
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            log_info("Using 8bit AdamW...")
        
        if self.cfg.trainer.momentum is not None:
            kwargs['momentum'] = self.cfg.trainer.momentum

        log_info(f"Using optimizer_class: {optimizer_class}")
            
        self.optimizer = optimizer_class(
            get_named_params(self.models).values() if (params_ := unwrap(self.model).get_param_groups()) is None else params_,
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay,
            **kwargs
        )

    def init_lr_scheduler(self):
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.cfg.trainer.gradient_accumulation_steps)
        if self.cfg.trainer.max_train_steps is None:
            self.cfg.trainer.max_train_steps = self.cfg.trainer.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.cfg.trainer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.trainer.lr_warmup_steps
            * self.cfg.trainer.num_gpus,  # TODO: We might not need to scale here. See src/accelerate/scheduler.py
            num_training_steps=self.cfg.trainer.max_train_steps * self.cfg.trainer.num_gpus,
            num_cycles=self.cfg.trainer.lr_num_cycles,
            power=self.cfg.trainer.lr_power,
        )

        # Prepare everything with our `self.accelerator`.
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )
        self.train_dataloader = self.accelerator.prepare_data_loader(self.train_dataloader, device_placement=True)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.cfg.trainer.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.cfg.trainer.max_train_steps = self.cfg.trainer.num_train_epochs * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        self.cfg.trainer.num_train_epochs = math.ceil(self.cfg.trainer.max_train_steps / num_update_steps_per_epoch)

    @try_except
    def checkpoint(self, state: TrainingState):
        prefix = "checkpoint"
        self.cfg.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        handle_checkpointing_dirs(self.cfg, prefix="checkpoint")
        save_path = self.cfg.checkpoint_dir / f"{prefix}_{state.global_step}"
        save_path.mkdir(exist_ok=True, parents=True)
        unwrap(self.model).checkpoint(self.accelerator, state, save_path)

        param_keys = get_named_params(self.models).keys()
        write_to_file(path=Path(self.cfg.output_dir, self.cfg.logging_dir) / "params.log", text="global_step:\n" + str(param_keys))

    @torch.no_grad()
    def validate(self, state: TrainingState):
        # TODO: Cleanup all the validation code once we figure out the possible memory leak.
        log_debug("Starting Validate", main_process_only=False)
        validation_start_time = time()

        self.model.eval()
        unwrap(self.model).set_inference_mode(init_pipeline=self.cfg.trainer.init_pipeline_inference)

        log_debug(f"Running validation on val dataloder", main_process_only=False)
        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=self.validation_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="val/",
            init_pipeline=False,
        )
        
        self.model.train()
        BaseMapper.set_training_mode(cfg=self.cfg, _other=self.model, device=self.accelerator.device, dtype=self.dtype, set_grad=False)

        log_info(
            f"Finished validation at global step {state.global_step}, epoch {state.epoch}. Wandb URL: {self.cfg.get('wandb_url', None)}. Took: {__import__('time').time() - validation_start_time:.2f} seconds"
        )

    def after_backward(self, state: TrainingState):
        tr = self.cfg.trainer
        if check_every_n_steps(
            state, tr.eval_steps, run_first=tr.eval_on_start, all_processes=True, decay_steps=tr.eval_decay_steps
        ) or check_every_n_epochs(state, tr.eval_epochs, all_processes=True):
            self.validate(state)

    def train(self):
        tr = self.cfg.trainer

        total_batch_size = self.cfg.dataset.train.batch_size * tr.num_gpus * tr.gradient_accumulation_steps

        log_info("***** Running training *****")
        log_info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        log_info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        log_info(f"  Num Epochs = {tr.num_train_epochs}")
        log_info(f"  Instantaneous batch size per device = {self.cfg.dataset.train.batch_size}")
        log_info(f"  Gradient Accumulation steps = {tr.gradient_accumulation_steps}")
        log_info(f"  Num GPUs = {tr.num_gpus}")
        log_info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        log_info(f"  Total optimization steps = {tr.max_train_steps}")

        if len(self.train_dataloader.dataset) < total_batch_size:
            log_warn("The training dataloader is smaller than the total batch size. This may lead to unexpected behaviour.")

        # Potentially load in the weights and states from a previous save. Due to an accelerate bug, we actually load the model in init_models and only fetch the step here.
        if tr.ckpt:
            initial_global_step = load_from_ckpt(cfg=self.cfg, accelerator=self.accelerator, model=self.model, load_model=False)
        else:
            initial_global_step = 0

        true_step = 0
        global_step = initial_global_step
        first_epoch = 0

        if self.cfg.profile:
            profiler = Profiler(output_dir=self.cfg.output_dir, warmup_steps=tr.profiler_warmup_steps, active_steps=tr.profiler_active_steps, record_memory=True)

        progress_bar = tqdm(range(0, tr.max_train_steps), initial=initial_global_step, desc="Steps", disable=not is_main_process(), leave=False)

        if is_main_process() and tr.log_gradients is not None:
            wandb.watch(self.model, log="all" if tr.log_parameters else "gradients", log_freq=tr.log_gradients)

        log_info(f"load_time: {time() - load_time} seconds")
        log_info(f"Train Dataloader Size on single GPU: {len(self.train_dataloader)}")

        global_step_metrics = defaultdict(float)
        global_extra_wandb_metrics = dict()
        accumulate_steps = 0  # TODO: Figure out what happens if we end the dataloader between gradient update steps
        last_end_step_time = time()

        start_timing("Dataloading")
        for epoch in range(first_epoch, tr.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                end_timing()
                step_start_time = time()
                global_step_metrics["dataloading_time"] += step_start_time - last_end_step_time
                if is_main_process() and global_step == 0 and accumulate_steps == 1:
                    log_info(f"time to complete 1st step: {step_start_time - load_time} seconds")

                with self.accelerator.accumulate(*filter(lambda x: isinstance(x, nn.Module), self.models)):
                    start_timing("Forward Pass")
                    global_step_metrics["examples_seen_per_gpu"] += len(next(iter(batch.values())))
                    state: TrainingState = TrainingState(
                        epoch_step=step, num_epoch_steps=len(self.train_dataloader), global_step=global_step, epoch=epoch, true_step=true_step
                    )

                    start_forward_time = time()
                    batch = unwrap(self.model).process_input(batch, state)
                    batch = batch.to(self.accelerator.device)
                    match self.cfg.model.model_type:
                        case ModelType.BASE_MAPPER:
                            losses = self.model(batch, state)
                    global_step_metrics["forward_pass_time"] += time() - start_forward_time

                    true_step += 1
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            global_step_metrics[k.removeprefix("metric_")] += v.detach().cpu().item()
                        else:
                            global_extra_wandb_metrics[k.removeprefix("metric_")] = v

                    losses = dict(filter(lambda item: not item[0].startswith("metric_"), losses.items())) # Allow for custom metrics that are not losses
                    loss = sum(losses.values())
                    global_step_metrics["loss"] += loss.detach().cpu().item()  # Only on the main process to avoid syncing
                    end_timing()

                    if tr.backward_pass:
                        start_timing("Backward Pass")
                        start_backward_time = time()
                        # The below lines may be silently skipped for gradient accumulation
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(get_named_params(self.models).values(), tr.max_grad_norm)

                        self.optimizer.step()
                        self.lr_scheduler.step()

                        zero_grad_kwargs = dict()
                        if 'apex' not in self.cfg.trainer.optimizer_cls.path:
                            zero_grad_kwargs['set_to_none'] = tr.set_grads_to_none
                        else:
                            log_warn("Not setting to none.")

                        self.optimizer.zero_grad(**zero_grad_kwargs)
                        end_timing()
                        global_step_metrics["backward_pass_time"] += time() - start_backward_time
                    else:
                        log_warn("Skipping backward pass!")

                    accumulate_steps += 1

                # Important: A single "global_step" is a single optimizer step. The accumulate decorator silently skips backward + optimizer to allow for gradient accumulation. A "true_step" counts the number of forward passes (on a per-GPU basis). The condition below should only happen immediately after a backward + optimizer step.
                if self.accelerator.sync_gradients:
                    start_timing("On Sync Gradients")
                    del batch, loss, losses
                    unwrap(self.model).on_sync_gradients(state)

                    if self.cfg.trainer.profile_memory and global_step + 1 >= tr.max_train_steps:
                        break

                    if self.cfg.profile and profiler.step(global_step):
                        log_info(f"Profiling finished at step: {global_step}")
                        break
                    
                    self.after_backward(state)

                    if check_every_n_steps(state, tr.ckpt_steps, run_first=False, all_processes=False):
                        self.checkpoint(state)

                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "max_gpu_memory_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
                        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                        "examples_seen": global_step * total_batch_size,
                        **{k: (v / accumulate_steps if ('backward' not in k) else v) for k, v in global_step_metrics.items()},
                        **{f"lr_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())},
                        **global_extra_wandb_metrics
                    }
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)
                    global_step_metrics = defaultdict(float)
                    accumulate_steps = 0
                    end_timing()

                if global_step >= tr.max_train_steps:
                    break

                last_end_step_time = time()
                start_timing("Dataloading")

        # Create the pipeline using using the trained modules and save it.
        self.accelerator.wait_for_everyone()

        if self.cfg.trainer.profile_memory:
            print_memory(verbose=True)
            save_memory_profile(self.cfg.output_dir / "profile")

        if self.cfg.profile:
            profiler.finish()
            exit()

        if is_main_process():
            if global_step > 100:
                self.checkpoint(state)

        self.accelerator.end_training()
