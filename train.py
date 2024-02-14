load_time = __import__("time").time()

import itertools
import math
import os
import types
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Iterable, Union

import hydra
import torch
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

import wandb
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from gen.configs import BaseConfig, ModelType
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.models.utils import get_model_from_cfg
from gen.utils.decoupled_utils import Profiler, get_rank, is_main_process, write_to_file
from gen.utils.logging_utils import log_error, log_info, log_warn
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


def validate_params(models: Iterable[nn.Module], dtype: torch.dtype):
    # In general, we want all trainable params in FP32 and all non-trainable params possibly in BF16
    for p in get_named_params(models).values():
        if p.requires_grad:
            assert p.dtype == torch.float32
        elif not p.requires_grad:
            assert p.dtype == dtype


class Trainer:
    def __init__(self, cfg: BaseConfig, accelerator: Accelerator):
        self.cfg = cfg
        self.accelerator = accelerator

        self.dtype = getattr(torch, self.cfg.trainer.dtype.split(".")[-1])

        self.model: Trainable = None  # Generally, we try to have a single top-level nn.Module that contains all the other models
        self.models: list[Union[nn.Module, dict]] = None  # We support multiple models or dicts of named parameters if necessary
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.tokenizer: CLIPTokenizer = None

        self.train_dataloader: DataLoader = None
        self.validation_dataloader: DataLoader = None

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
                self.model = self.accelerator.prepare(model)
                self.tokenizer = unwrap(model).tokenizer

                if is_main_process():
                    summary(model, col_names=("trainable", "num_params"), depth=3)

                if (param_groups_ := unwrap(self.model).get_param_groups()) is not None:
                    assert len([p for d_ in param_groups_ for p in list(d_["params"])]) == len(get_named_params(self.models).values())

        validate_params(self.models, self.dtype)

    def init_dataloader(self):
        log_info("Creating train_dataset + self.train_dataloader")
        self.train_dataloader_holder: AbstractDataset = hydra.utils.instantiate(self.cfg.dataset.train_dataset, _recursive_=True)(
            cfg=self.cfg, split=Split.TRAIN, tokenizer=self.tokenizer, accelerator=self.accelerator
        )
        self.train_dataloader: DataLoader = self.train_dataloader_holder.get_dataloader()
        assert len(self.train_dataloader) > 0

        log_info("Creating validation_dataset + self.validation_dataloader")
        self.validation_dataset_holder: AbstractDataset = hydra.utils.instantiate(self.cfg.dataset.validation_dataset, _recursive_=True)(
            cfg=self.cfg, split=Split.VALIDATION, tokenizer=self.tokenizer, accelerator=self.accelerator
        )

        if self.cfg.dataset.overfit:
            self.validation_dataset_holder.get_dataset = lambda: self.train_dataloader.dataset

        self.validation_dataloader: DataLoader = self.validation_dataset_holder.get_dataloader()
        assert len(self.validation_dataloader) > 0

    def init_optimizer(self):
        optimizer_class = torch.optim.AdamW
        self.optimizer = optimizer_class(
            get_named_params(self.models).values() if (params_ := unwrap(self.model).get_param_groups()) is None else params_,
            lr=self.cfg.trainer.learning_rate,
            betas=(self.cfg.trainer.adam_beta1, self.cfg.trainer.adam_beta2),
            weight_decay=self.cfg.trainer.adam_weight_decay,
            eps=self.cfg.trainer.adam_epsilon,
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
        self.optimizer, self.lr_scheduler, self.train_dataloader, self.validation_dataloader = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler, self.train_dataloader, self.validation_dataloader
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.cfg.trainer.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.cfg.trainer.max_train_steps = self.cfg.trainer.num_train_epochs * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        self.cfg.trainer.num_train_epochs = math.ceil(self.cfg.trainer.max_train_steps / num_update_steps_per_epoch)

    def checkpoint(self, state: TrainingState):
        prefix = "checkpoint"
        handle_checkpointing_dirs(self.cfg, prefix="checkpoint")
        save_path = self.cfg.checkpoint_dir / f"{prefix}_{state.global_step}"
        save_path.mkdir(exist_ok=True, parents=True)
        unwrap(self.model).checkpoint(self.accelerator, state, save_path)

    def validate(self, state: TrainingState):
        validation_start_time = time()

        if self.cfg.dataset.reset_validation_dataset_every_epoch or self.cfg.trainer.base_model_custom_validation_steps:
            if state.epoch == 0:
                self.validation_dataset_holder.subset_size = self.cfg.trainer.num_gpus
            else:
                self.validation_dataset_holder.subset_size = max(self.cfg.dataset.validation_dataset.subset_size, self.cfg.trainer.num_gpus)

            g = torch.Generator()
            g.manual_seed(state.global_step + get_rank())
            self.validation_dataset_holder.batch_size = self.cfg.dataset.validation_dataset.batch_size
            self.validation_dataloader = self.validation_dataset_holder.get_dataloader(generator=g)
            self.validation_dataloader = self.accelerator.prepare(self.validation_dataloader)

        param_keys = get_named_params(self.models).keys()
        write_to_file(path=Path(self.cfg.output_dir, self.cfg.logging_dir) / "params.log", text="global_step:\n" + str(param_keys))

        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=self.validation_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="val/",
        )

        if self.cfg.trainer.validate_training_dataset:
            self.validate_train_dataloader(state)

        unwrap(self.model).set_training_mode()
        validate_params(self.models, self.dtype)

        log_info(
            f"Finished validation at global step {state.global_step}, epoch {state.epoch}. Wandb URL: {self.cfg.get('wandb_url', None)}. Took: {__import__('time').time() - validation_start_time:.2f} seconds"
        )

    def validate_train_dataloader(self, state: TrainingState):
        self.train_dataloader_holder.subset_size = self.validation_dataset_holder.subset_size
        self.train_dataloader_holder.random_subset = self.validation_dataset_holder.random_subset
        self.train_dataloader_holder.batch_size = self.validation_dataset_holder.batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader(pin_memory=False)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=self.train_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="train/",
        )

        self.train_dataloader_holder.subset_size = self.cfg.dataset.train_dataset.subset_size
        self.train_dataloader_holder.random_subset = self.cfg.dataset.train_dataset.random_subset
        self.train_dataloader_holder.batch_size = self.cfg.dataset.train_dataset.batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader()
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

    # TODO: This is model-specific and should be achieved by inheritance or some other mechanism
    def base_model_validate(self, state: TrainingState):
        from gen.models.cross_attn.base_inference import run_qualitative_inference, run_quantitative_inference

        unwrap(self.model).run_inference = types.MethodType(run_quantitative_inference, unwrap(self.model))

        subset_size, batch_size = len(self.validation_dataset_holder), self.train_dataloader_holder.batch_size
        self.validation_dataset_holder.subset_size = subset_size
        self.train_dataloader_holder.random_subset = False
        self.validation_dataset_holder.batch_size = batch_size
        self.validation_dataloader = self.validation_dataset_holder.get_dataloader(pin_memory=False)
        self.validation_dataloader = self.accelerator.prepare(self.validation_dataloader)

        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=self.validation_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="val/",
            init_pipeline=False
        )

        self.train_dataloader_holder.subset_size = subset_size
        self.train_dataloader_holder.random_subset = False
        self.train_dataloader_holder.batch_size = batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader(pin_memory=False)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=self.train_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="train/",
            init_pipeline=False
        )

        self.train_dataloader_holder.subset_size = self.cfg.dataset.train_dataset.subset_size
        self.train_dataloader_holder.random_subset = self.cfg.dataset.train_dataset.random_subset
        self.train_dataloader_holder.batch_size = self.cfg.dataset.train_dataset.batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader()
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        unwrap(self.model).run_inference = types.MethodType(run_qualitative_inference, unwrap(self.model))

        import gc; gc.collect()
        torch.cuda.empty_cache()

        unwrap(self.model).set_training_mode()
        validate_params(self.models, self.dtype)

    def unfreeze_unet(self, state: TrainingState):
        log_warn(f"Unfreezing UNet at {state.global_step} steps")
        model_: Trainable = unwrap(self.model)
        model_.unfreeze_unet()
        self.models.append(model_)
        del self.optimizer
        optimizer_class = torch.optim.AdamW
        self.optimizer = optimizer_class(
            get_named_params(self.models).values(),
            lr=self.cfg.trainer.finetune_learning_rate,
            betas=(self.cfg.trainer.adam_beta1, self.cfg.trainer.adam_beta2),
            weight_decay=self.cfg.trainer.adam_weight_decay,
            eps=self.cfg.trainer.adam_epsilon,
        )
        del self.lr_scheduler
        self.lr_scheduler = get_scheduler(
            self.cfg.trainer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.trainer.lr_warmup_steps * self.cfg.trainer.num_gpus,
            num_training_steps=self.cfg.trainer.max_train_steps * self.cfg.trainer.num_gpus,
            num_cycles=self.cfg.trainer.lr_num_cycles,
            power=self.cfg.trainer.lr_power,
        )
        self.optimizer, self.lr_scheduler, self.model = self.accelerator.prepare(self.optimizer, self.lr_scheduler, model_)
        validate_params(self.models, self.dtype)
        if is_main_process():
            summary(unwrap(self.model), col_names=("trainable", "num_params"), depth=4)

    def train(self):
        tr = self.cfg.trainer

        total_batch_size = self.cfg.dataset.train_dataset.batch_size * tr.num_gpus * tr.gradient_accumulation_steps

        log_info("***** Running training *****")
        log_info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        log_info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        log_info(f"  Num Epochs = {tr.num_train_epochs}")
        log_info(f"  Instantaneous batch size per device = {self.cfg.dataset.train_dataset.batch_size}")
        log_info(f"  Gradient Accumulation steps = {tr.gradient_accumulation_steps}")
        log_info(f"  Num GPUs = {tr.num_gpus}")
        log_info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        log_info(f"  Total optimization steps = {tr.max_train_steps}")

        if len(self.train_dataloader.dataset) < total_batch_size:
            log_warn("The training dataloader is smaller than the total batch size. This may lead to unexpected behaviour.")

        true_step = 0
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        initial_global_step = load_from_ckpt(cfg=self.cfg, accelerator=self.accelerator, model=self.model) if tr.ckpt else 0

        if self.cfg.profile:
            profiler = Profiler(output_dir=self.cfg.output_dir, active_steps=tr.profiler_active_steps)

        progress_bar = tqdm(range(0, tr.max_train_steps), initial=initial_global_step, desc="Steps", disable=not is_main_process(), leave=False)

        if is_main_process() and tr.log_gradients is not None:
            wandb.watch(self.model, log="all" if tr.log_parameters else "gradients", log_freq=tr.log_gradients)

        log_info(f"load_time: {time() - load_time} seconds")
        log_info(f"Train Dataloader Size on single GPU: {len(self.train_dataloader)}")

        global_step_metrics = defaultdict(float)
        accumulate_steps = 0  # TODO: Figure out what happens if we end the dataloader between gradient update steps
        last_end_step_time = time()
        for epoch in range(first_epoch, tr.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                step_start_time = time()
                global_step_metrics["dataloading_time"] += step_start_time - last_end_step_time
                if is_main_process() and global_step == 0 and accumulate_steps == 1:
                    log_info(f"time to complete 1st step: {step_start_time - load_time} seconds")

                with self.accelerator.accumulate(*filter(lambda x: isinstance(x, nn.Module), self.models)):
                    global_step_metrics["examples_seen_per_gpu"] += batch["gen_pixel_values"].shape[0]
                    state: TrainingState = TrainingState(
                        epoch_step=step, num_epoch_steps=len(self.train_dataloader), global_step=global_step, epoch=epoch, true_step=true_step
                    )

                    batch["state"] = state
                    start_forward_time = time()
                    match self.cfg.model.model_type:
                        case ModelType.BASE_MAPPER:
                            losses = self.model(batch)
                    global_step_metrics["forward_pass_time"] += time() - start_forward_time

                    true_step += 1
                    for k, v in losses.items():
                        global_step_metrics[k.removeprefix("metric_")] += v.detach().item()

                    losses = dict(filter(lambda item: not item[0].startswith("metric_"), losses.items())) # Allow for custom metrics that are not losses
                    loss = sum(losses.values())
                    global_step_metrics["loss"] += loss.detach().item()  # Only on the main process to avoid syncing
                    accumulate_steps += 1

                    # The below lines may be silently skipped for gradient accumulation
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(get_named_params(self.models).values(), tr.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=tr.set_grads_to_none)

                # Important: A single "global_step" is a single optimizer step. The accumulate decorator silently skips backward + optimizer to allow for gradient accumulation. A "true_step" counts the number of forward passes (on a per-GPU basis). The condition below should only happen immediately after a backward + optimizer step.
                if self.accelerator.sync_gradients:
                    if check_every_n_steps(state, tr.checkpointing_steps, run_first=False, all_processes=False):
                        self.checkpoint(state)

                    if check_every_n_steps(
                        state, tr.eval_every_n_steps, run_first=tr.eval_on_start, all_processes=True, decay_steps=True
                    ) or check_every_n_epochs(state, tr.eval_every_n_epochs, all_processes=True):
                        self.validate(state)

                    if check_every_n_steps(state, tr.base_model_custom_validation_steps, run_first=tr.eval_on_start, all_processes=True):
                        self.base_model_validate(state)

                    if self.cfg.model.unfreeze_unet_after_n_steps and global_step == self.cfg.model.unfreeze_unet_after_n_steps:
                        self.unfreeze_unet(state)

                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "gpu_memory_usage_gb": max(torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved()) / (1024**3),
                        "examples_seen": global_step * total_batch_size,
                        **{k: v / accumulate_steps for k, v in global_step_metrics.items()},
                        **{f"lr_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())},
                    }
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)
                    global_step_metrics = defaultdict(float)
                    accumulate_steps = 0

                if global_step >= tr.max_train_steps:
                    break

                elif self.cfg.profile and profiler.step(global_step):
                    assert tr.gradient_accumulation_steps == 1
                    log_info(f"Profiling finished at step: {global_step}")
                    break

                last_end_step_time = time()

        # Create the pipeline using using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if is_main_process():
            if self.cfg.profile:
                profiler.finish()
                exit()

            self.checkpoint(state)

        self.accelerator.end_training()
