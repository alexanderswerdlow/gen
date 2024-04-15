
import traceback
import types
from functools import partial
from time import time

import torch
import torch.utils.checkpoint
from torchinfo import summary

import wandb
from diffusers.optimization import get_scheduler
from gen.models.utils import set_default_inference_func, set_inference_func
from gen.utils.decoupled_utils import get_num_gpus, get_rank, is_main_process, show_memory_usage
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.trainer_utils import Trainable, TrainingState, check_every_n_epochs, check_every_n_steps, unwrap
from inference import run_inference_dataloader
from train import Trainer, get_named_params, validate_params


class BaseTrainer(Trainer):
    @torch.no_grad()
    def validate_compose(self, state: TrainingState):
        assert self.cfg.dataset.reset_val_dataset_every_epoch
        from gen.models.cross_attn.base_inference import compose_two_images
        set_inference_func(unwrap(self.model), partial(compose_two_images))
        self.val_dataset_holder.batch_size = 2
        self.validate(state=state)
        self.val_dataset_holder.batch_size = self.cfg.dataset.val.batch_size
        set_default_inference_func(unwrap(self.model), self.cfg)

    def validate_validation_dataloader(self, state: TrainingState):
        if self.cfg.dataset.reset_val_dataset_every_epoch:
            g = torch.Generator()
            g.manual_seed(state.global_step + get_rank())
            self.val_dataset_holder.batch_size = self.cfg.dataset.val.batch_size
            self.val_dataset_holder.subset_size = max(self.cfg.trainer.num_gpus * self.val_dataset_holder.batch_size, self.cfg.dataset.val.batch_size)
            self.val_dataset_holder.num_workers = self.cfg.dataset.val.num_workers
            log_debug(f"Resetting Validation Dataloader with {self.val_dataset_holder.num_workers} workers", main_process_only=False)
            self.validation_dataloader = self.val_dataset_holder.get_dataloader(generator=g, pin_memory=False)
            self.validation_dataloader = self.accelerator.prepare_data_loader(self.validation_dataloader, device_placement=False)

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

        if self.cfg.dataset.reset_val_dataset_every_epoch:
            del self.validation_dataloader

    def validate_train_dataloader(self, state: TrainingState):
        log_debug("Starting Validate Train Dataloader", main_process_only=False)
        self.train_dataloader_holder.num_workers = self.val_dataset_holder.num_workers
        self.train_dataloader_holder.subset_size = self.val_dataset_holder.subset_size
        self.train_dataloader_holder.random_subset = self.val_dataset_holder.random_subset
        self.train_dataloader_holder.batch_size = self.val_dataset_holder.batch_size
        self.train_dataloader_holder.repeat_dataset_n_times = None

        validate_train_dataloader = self.train_dataloader_holder.get_dataloader(pin_memory=False)
        validate_train_dataloader = self.accelerator.prepare(validate_train_dataloader)

        run_inference_dataloader(
            accelerator=self.accelerator,
            dataloader=validate_train_dataloader,
            model=self.model,
            state=state,
            output_path=self.cfg.output_dir / "images",
            prefix="train/",
            init_pipeline=False,
        )

        self.train_dataloader_holder.num_workers = self.cfg.dataset.train.num_workers
        self.train_dataloader_holder.subset_size = self.cfg.dataset.train.subset_size
        self.train_dataloader_holder.random_subset = self.cfg.dataset.train.random_subset
        self.train_dataloader_holder.batch_size = self.cfg.dataset.train.batch_size
        self.train_dataloader_holder.repeat_dataset_n_times = self.cfg.dataset.train.repeat_dataset_n_times
        log_debug("Finished Validate Train Dataloader", main_process_only=False)

    # TODO: This is model-specific and should be achieved by inheritance or some other mechanism
    def base_model_validate(self, state: TrainingState):
        start_time = time()

        log_info(f"Starting base model validation at global step {state.global_step}, epoch {state.epoch}")
        from gen.models.cross_attn.base_inference import run_qualitative_inference, run_quantitative_inference

        unwrap(self.model).run_inference = types.MethodType(run_quantitative_inference, unwrap(self.model))

        g = torch.Generator()
        g.manual_seed(0)
        
        subset_size = len(self.val_dataset_holder)
        subset_size = min(subset_size, self.cfg.dataset.val.subset_size) if self.cfg.dataset.val.subset_size is not None else subset_size
        subset_size = min(subset_size, self.cfg.trainer.custom_inference_dataset_size) if self.cfg.trainer.custom_inference_dataset_size is not None else subset_size
        batch_size = self.cfg.trainer.custom_inference_batch_size if self.cfg.trainer.custom_inference_batch_size is not None else self.train_dataloader_holder.batch_size

        self.val_dataset_holder.subset_size = subset_size
        self.val_dataset_holder.random_subset = self.cfg.trainer.custom_inference_fixed_shuffle
        self.val_dataset_holder.batch_size = batch_size
        self.validation_dataloader = self.val_dataset_holder.get_dataloader(pin_memory=False, generator=g)
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
        self.train_dataloader_holder.random_subset = self.cfg.trainer.custom_inference_fixed_shuffle
        self.train_dataloader_holder.batch_size = batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader(pin_memory=False, generator=g)
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

        self.train_dataloader_holder.subset_size = self.cfg.dataset.train.subset_size
        self.train_dataloader_holder.random_subset = self.cfg.dataset.train.random_subset
        self.train_dataloader_holder.batch_size = self.cfg.dataset.train.batch_size
        self.train_dataloader = self.train_dataloader_holder.get_dataloader()
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        unwrap(self.model).run_inference = types.MethodType(run_qualitative_inference, unwrap(self.model))

        import gc; gc.collect()
        torch.cuda.empty_cache()

        from gen.models.cross_attn.base_model import BaseMapper
        BaseMapper.set_training_mode(cfg=self.cfg, _other=self.model, device=self.accelerator.device, dtype=self.dtype, set_grad=False)
        validate_params(self.models, self.dtype)
        
        log_info(f"Finished base model validation at global step {state.global_step}, epoch {state.epoch}. Took: {time() - start_time:.2f} seconds")

    def unfreeze_unet(self, state: TrainingState):
        log_warn(f"Unfreezing UNet at {state.global_step} steps")
        model_: Trainable = unwrap(self.model)
        model_.unfreeze_unet()
        self.models.append(model_)
        del self.optimizer
        optimizer_class = self.cfg.trainer.optimizer_cls
        self.optimizer = optimizer_class(
            get_named_params(self.models).values(),
            lr=self.cfg.trainer.finetune_learning_rate,
            betas=(self.cfg.trainer.adam_beta1, self.cfg.trainer.adam_beta2),
            weight_decay=self.cfg.trainer.weight_decay,
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

    @torch.no_grad()
    def validate(self, state: TrainingState):
        # TODO: Cleanup all the validation code once we figure out the possible memory leak.
        log_info("Starting Validation", main_process_only=False)
        validation_start_time = time()

        self.model.eval()
        unwrap(self.model).set_inference_mode(init_pipeline=self.cfg.trainer.init_pipeline_inference)

        log_info(f"Running validation on val dataloder", main_process_only=False)
        if self.cfg.trainer.validate_validation_dataset:
            self.validate_validation_dataloader(state)

        log_info(f"Running validation on train dataloder", main_process_only=False)
        if self.cfg.trainer.validate_training_dataset:
            self.validate_train_dataloader(state)
        
        self.model.train()
        from gen.models.cross_attn.base_model import BaseMapper
        BaseMapper.set_training_mode(cfg=self.cfg, _other=self.model, device=self.accelerator.device, dtype=self.dtype, set_grad=False)

        log_info(
            f"Finished validation at global step {state.global_step}, epoch {state.epoch}. Wandb URL: {self.cfg.get('wandb_url', None)}. Took: {__import__('time').time() - validation_start_time:.2f} seconds"
        )

    def after_backward(self, state: TrainingState):
        tr = self.cfg.trainer
        if check_every_n_steps(
            state, tr.eval_every_n_steps, run_first=tr.eval_on_start, all_processes=True, decay_steps=tr.eval_decay_steps,
        ) or check_every_n_epochs(state, tr.eval_every_n_epochs, all_processes=True):
            with show_memory_usage():
                try:
                    self.validate(state)
                    if self.cfg.trainer.compose_inference:
                        self.validate_compose(state)
                except Exception as e:
                    if get_num_gpus() > 1 and state.global_step > 100:
                        traceback.print_exc()
                        log_error(f"Error during validation: {e}. Continuing...", main_process_only=False)
                    else:
                        raise

        if check_every_n_steps(state, tr.custom_inference_every_n_steps, run_first=tr.eval_on_start, all_processes=True):
            self.base_model_validate(state)

        if self.cfg.model.unfreeze_unet_after_n_steps and state.global_step == self.cfg.model.unfreeze_unet_after_n_steps:
            self.unfreeze_unet(state)