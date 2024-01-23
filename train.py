load_time = __import__("time").time()

import itertools
import math
import os
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm.auto import tqdm

from gen.configs import BaseConfig, ModelType
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.checkpoint_handler import CheckpointHandler
from gen.models.neti.validator import ValidationHandler
from gen.models.neti_base_model import BaseMapper as OriginalBaseMapper
from gen.utils.decoupled_utils import Profiler, is_main_process, write_to_file
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.trainer_utils import TrainingState, check_every_n_epochs, check_every_n_steps, handle_checkpointing, unwrap
from transformers import CLIPTokenizer

def trainable_parameters(module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def get_named_params_to_optimize(models: tuple[Union[nn.Module, dict]]):
    return dict(
        itertools.chain(
            *(trainable_parameters(model) for model in models if isinstance(model, nn.Module)), *(np.items() for np in models if isinstance(np, dict))
        )
    )


class Trainer:
    def __init__(self, cfg: BaseConfig, accelerator: Accelerator):
        self.cfg = cfg
        self.accelerator = accelerator

        self.model: nn.Module = None  # Generally, we try to have a single top-level nn.Module that contains all the other models
        self.models: list[Union[nn.Module, dict]] = None  # We support multiple models or dicts of named parameters if necessary
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.tokenizer: CLIPTokenizer = None
        self.checkpoint_handler: CheckpointHandler = None
        self.validator: ValidationHandler = None

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
        weight_dtype = getattr(torch, self.cfg.trainer.dtype.split(".")[-1])
        match self.cfg.model.model_type:
            case ModelType.SODA:
                pass
            case ModelType.BASE_MAPPER:
                assert self.cfg.model.model_type == ModelType.BASE_MAPPER
                if self.cfg.model.per_timestep_conditioning:
                    model = OriginalBaseMapper(self.cfg)
                    model.prepare_for_training(self.cfg.trainer.dtype, self.accelerator)

                    if self.cfg.model.freeze_text_encoder:
                        self.models.append(unwrap(model.text_encoder).text_model.embeddings.mapper)
                    else:
                        self.models.append(unwrap(model.text_encoder))

                    if self.cfg.model.freeze_unet is False or self.cfg.model.lora_unet:
                        self.models.append(model.unet)

                    if self.cfg.model.controlnet:
                        self.models.append(model.controlnet)

                    summary(unwrap(model.text_encoder).text_model.embeddings, col_names=("trainable", "num_params"), verbose=2)
                else:
                    model = BaseMapper(self.cfg)
                    model.add_adapters()
                    self.models.append(model)
                    self.model = self.accelerator.prepare(model)

                self.tokenizer = unwrap(model).tokenizer
                self.checkpoint_handler: CheckpointHandler = CheckpointHandler(cfg=self.cfg, save_root=self.cfg.checkpoint_dir)
                self.validator: ValidationHandler = ValidationHandler(cfg=self.cfg, weights_dtype=weight_dtype)

                if is_main_process():
                    summary(model, col_names=("trainable", "num_params"), depth=3)

    def init_dataloader(self):
        log_info("Creating train_dataset + self.train_dataloader")
        self.train_dataloader: DataLoader = hydra.utils.instantiate(self.cfg.dataset.train_dataset, _recursive_=True)(
            cfg=self.cfg, split=Split.TRAIN, tokenizer=self.tokenizer, accelerator=self.accelerator
        ).get_dataloader()

        log_info("Creating validation_dataset + self.validation_dataloader")
        self.validation_dataset_holder: AbstractDataset = hydra.utils.instantiate(self.cfg.dataset.validation_dataset, _recursive_=True)(
            cfg=self.cfg, split=Split.VALIDATION, tokenizer=self.tokenizer, accelerator=self.accelerator
        )

        if self.cfg.dataset.overfit:
            self.validation_dataset_holder.get_dataset = lambda: self.train_dataloader.dataset

        self.validation_dataloader = self.validation_dataset_holder.get_dataloader()

    def init_optimizer(self):
        optimizer_class = torch.optim.AdamW
        log_warn("TOOD: Scale LR based on accum")
        self.optimizer = optimizer_class(
            get_named_params_to_optimize(self.models).values(),
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

    def load_from_ckpt(self):
        if self.cfg.trainer.ckpt == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.checkpoint_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = Path(self.cfg.trainer.ckpt)

        if path is None:
            log_error(f"Checkpoint '{self.cfg.trainer.ckpt}' does not exist. Exiting.")
            raise FileNotFoundError
        else:
            log_info(f"Resuming from checkpoint {path}")
            if path.is_file() or self.cfg.trainer.load_weights_only_no_state:
                from accelerate.utils.modeling import load_checkpoint_in_model

                load_checkpoint_in_model(self.model, str(path))
            else:
                self.accelerator.load_state(path)

            if path.is_file():
                global_step = int(path.parent.name.split("-")[-1])
            else:
                global_step = int(path.name.split("-")[1])

            # first_epoch = global_step // num_update_steps_per_epoch
            first_epoch = 0
            log_info(f"Continuing training from epoch {first_epoch} and global step {global_step}")
            return global_step

    def checkpoint(self, global_step: int, model: nn.Module, checkpoint_handler: CheckpointHandler):
        if self.cfg.model.lora_unet:
            log_error("LoRA UNet checkpointing not implemented")

        if self.cfg.model.model_type == ModelType.BASE_MAPPER:
            checkpoint_handler.save_model(model=model, accelerator=self.accelerator, save_name=f"{global_step}")
            if self.cfg.trainer.save_self.accelerator_format:
                handle_checkpointing(self.cfg, self.accelerator, global_step)
                save_path = self.cfg.checkpoint_dir / f"checkpoint-model-{global_step}"
                self.accelerator.save_model(model, save_path, safe_serialization=False)
        else:
            handle_checkpointing(self.cfg, self.accelerator, global_step)

    def validate(self, state: TrainingState):
        self.accelerator.free_memory()
        validation_start_time = time()
        if self.cfg.dataset.reset_validation_dataset_every_epoch:
            if state.epoch == 0:
                self.validation_dataset_holder.subset_size = self.cfg.trainer.num_gpus
            else:
                self.validation_dataset_holder.subset_size = self.cfg.dataset.validation_dataset.subset_size
            self.validation_dataloader = self.validation_dataset_holder.get_dataloader()
            self.validation_dataloader = self.accelerator.prepare(self.validation_dataloader)

        param_keys = get_named_params_to_optimize(self.models).keys()
        write_to_file(path=Path(self.cfg.output_dir, self.cfg.logging_dir) / "params.log", text="global_step:\n" + str(param_keys))
        match self.cfg.model.model_type:
            case ModelType.BASE_MAPPER:
                self.validator.infer(
                    accelerator=self.accelerator,
                    validation_dataloader=self.validation_dataloader,
                    model=unwrap(self.model),
                    global_step=state.global_step,
                )
        self.accelerator.free_memory()
        log_info(
            f"Finished validation at global step {state.global_step}, epoch {state.epoch}. Wandb URL: {self.cfg.wandb_url}. Took: {__import__('time').time() - validation_start_time:.2f} seconds"
        )

    def unfreeze_unet(self, state: TrainingState):
        log_warn(f"Unfreezing UNet at {state.global_step} steps")
        self.cfg.model.freeze_unet = False
        self.model.unet.requires_grad_(True)
        self.models.append(self.model.unet)
        del optimizer
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            get_named_params_to_optimize(self.models).values(),
            lr=self.cfg.trainer.finetune_learning_rate,
            betas=(self.cfg.trainer.adam_beta1, self.cfg.trainer.adam_beta2),
            weight_decay=self.cfg.trainer.adam_weight_decay,
            eps=self.cfg.trainer.adam_epsilon,
        )
        del lr_scheduler
        lr_scheduler = get_scheduler(
            self.cfg.trainer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.trainer.lr_warmup_steps * self.cfg.trainer.num_gpus,
            num_training_steps=self.cfg.trainer.max_train_steps * self.cfg.trainer.num_gpus,
            num_cycles=self.cfg.trainer.lr_num_cycles,
            power=self.cfg.trainer.lr_power,
        )
        optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
        summary(self.model)

    def train(self):
        total_batch_size = self.cfg.dataset.train_dataset.batch_size * self.cfg.trainer.num_gpus * self.cfg.trainer.gradient_accumulation_steps

        log_info("***** Running training *****")
        log_info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        log_info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        log_info(f"  Num Epochs = {self.cfg.trainer.num_train_epochs}")
        log_info(f"  Instantaneous batch size per device = {self.cfg.dataset.train_dataset.batch_size}")
        log_info(f"  Gradient Accumulation steps = {self.cfg.trainer.gradient_accumulation_steps}")
        log_info(f"  Num GPUs = {self.cfg.trainer.num_gpus}")
        log_info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        log_info(f"  Total optimization steps = {self.cfg.trainer.max_train_steps}")

        if len(self.train_dataloader.dataset) < total_batch_size:
            log_warn("The training dataloader is smaller than the total batch size. This may lead to unexpected behaviour.")

        true_step = 0
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        initial_global_step = self.load_from_ckpt() if self.cfg.trainer.ckpt else 0

        if self.cfg.profile:
            profiler = Profiler(output_dir=self.cfg.output_dir, active_steps=self.cfg.trainer.profiler_active_steps)

        progress_bar = tqdm(
            range(0, self.cfg.trainer.max_train_steps), initial=initial_global_step, desc="Steps", disable=not is_main_process(), leave=False
        )

        if is_main_process() and self.cfg.trainer.log_gradients is not None:
            wandb.watch(self.model, log="all" if self.cfg.trainer.log_parameters else "gradients", log_freq=self.cfg.trainer.log_gradients)

        log_info(f"load_time: {time() - load_time} seconds")
        log_info(f"Train Dataloader Size on single GPU: {len(self.train_dataloader)}")

        examples_seen_one_gpu = 0
        loss_per_global_step = 0
        dataloading_time_per_global_step = 0
        last_end_step_time = time()
        for epoch in range(first_epoch, self.cfg.trainer.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                step_start_time = time()
                dataloading_time_per_global_step += step_start_time - last_end_step_time
                if is_main_process() and global_step == 1:
                    log_info(f"time to complete 1st step: {step_start_time - load_time} seconds")

                with self.accelerator.accumulate(*filter(lambda x: isinstance(x, nn.Module), self.models)):
                    examples_seen_one_gpu += batch["gen_pixel_values"].shape[0]
                    state: TrainingState = TrainingState(
                        epoch_step=step,
                        total_epoch_steps=len(self.train_dataloader),
                        global_step=global_step,
                        epoch=epoch,
                    )

                    match self.cfg.model.model_type:
                        case ModelType.BASE_MAPPER:
                            loss = self.model(batch)

                    true_step += 1
                    loss_per_global_step += loss.detach().item()  # Only on the main process to avoid syncing

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(get_named_params_to_optimize(self.models).values(), self.cfg.trainer.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=self.cfg.trainer.set_grads_to_none)

                # Important Note: Right now a single "global_step" is a single gradient update step (same if we don't have grad accum)
                # This is different from "step" which only counts the number of forward passes
                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if check_every_n_steps(state, self.cfg.trainer.checkpointing_steps, run_first=False, all_processes=False):
                        self.checkpoint(self.cfg, self.accelerator, global_step, self.model, self.checkpoint_handler)

                    if (
                        check_every_n_steps(state, self.cfg.trainer.eval_every_n_steps, run_first=self.cfg.trainer.eval_on_start, all_processes=True)
                        or (self.cfg.trainer.eval_on_start and global_step == initial_global_step)
                        or check_every_n_epochs(state, self.cfg.trainer.eval_every_n_epochs, all_processes=True)
                    ):
                        self.validate(state)

                    if self.cfg.model.unfreeze_unet_after_n_steps and global_step == self.cfg.model.unfreeze_unet_after_n_steps:
                        self.unfreeze_unet(state)

                    loss_per_global_step /= self.cfg.trainer.gradient_accumulation_steps
                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "loss": loss_per_global_step,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "gpu_memory_usage_gb": max(torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved()) / (1024**3),
                        "examples_seen": global_step * total_batch_size,
                        "examples_seen_one_gpu": examples_seen_one_gpu,
                        "dataloading_time_per_global_step": dataloading_time_per_global_step / self.cfg.trainer.gradient_accumulation_steps,
                    }
                    dataloading_time_per_global_step = 0
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)
                    loss_per_global_step = 0

                if global_step >= self.cfg.trainer.max_train_steps:
                    break

                elif self.cfg.profile and profiler.step(global_step):
                    log_info(f"Profiling finished at step: {global_step}")
                    break

                last_end_step_time = time()

            # TODO: Something weird happens with webdataset:
            # UserWarning: Length of IterableDataset <abc.WebDataset_Length object at 0x7f0748da4640> was reported to be 2 (when accessing len(dataloader)), but 3 samples have been fetched.
            # if step >= len(self.train_dataloader) - 1:
            #     log_info(f"Exited early at step {global_step}")
            #     break

        # Create the pipeline using using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if is_main_process():
            if self.cfg.profile:
                profiler.finish()
                exit()

            self.checkpoint(global_step, self.model, self.checkpoint_handler)

        self.accelerator.end_training()
