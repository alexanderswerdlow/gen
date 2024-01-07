load_time = __import__('time').time()

import math
import os

import hydra
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from ipdb import set_trace
from torchinfo import summary
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from gen.configs import BaseConfig, ModelType
from gen.datasets.base_dataset import AbstractDataset
from gen.models.base_mapper_model import BaseMapper
from gen.models.controlnet_model import (controlnet_forward,
                                         get_controlnet_model, log_validation,
                                         pre_train_setup_controlnet)
from gen.models.neti.checkpoint_handler import CheckpointHandler
from gen.models.neti.validator import ValidationHandler
from gen.utils.decoupled_utils import Profiler
from gen.utils.trainer_utils import handle_checkpointing

logger = get_logger(__name__)

def run(cfg: BaseConfig, accelerator: Accelerator):
    # TODO: Define a better interface for different models once we get a better idea of the requires inputs/outputs
    # Right now we just conditionally call methods in the respective files based on the model_type enum
    assert is_xformers_available()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    match cfg.model.model_type:
        case ModelType.CONTROLNET:
            tokenizer, noise_scheduler, text_encoder, vae, unet, controlnet = get_controlnet_model(cfg, accelerator)
            params_to_optimize = controlnet.parameters
        case ModelType.BASE_MAPPER:
            assert cfg.model.model_type == ModelType.BASE_MAPPER
            model = BaseMapper(cfg)
            params_to_optimize = model.text_encoder.text_model.embeddings.mapper.parameters
            tokenizer = model.tokenizer
            summary(model, col_names=("trainable",  "num_params"))
            checkpoint_handler: CheckpointHandler = CheckpointHandler(cfg=cfg, save_root=cfg.output_dir)
            validator: ValidationHandler = ValidationHandler(cfg=cfg, weights_dtype=weight_dtype)

    if accelerator.is_local_main_process:
        if cfg.model.model_type == ModelType.CONTROLNET: summary(controlnet)

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize(),
        lr=cfg.trainer.learning_rate,
        betas=(cfg.trainer.adam_beta1, cfg.trainer.adam_beta2),
        weight_decay=cfg.trainer.adam_weight_decay,
        eps=cfg.trainer.adam_epsilon,
    )

    train_dataloader: DataLoader = hydra.utils.instantiate(cfg.dataset.train_dataset, _recursive_=False)(cfg=cfg, tokenizer=tokenizer, accelerator=accelerator).get_dataloader()

    validation_dataset_holder: AbstractDataset = hydra.utils.instantiate(cfg.dataset.validation_dataset, _recursive_=False)(cfg=cfg, tokenizer=tokenizer, accelerator=accelerator)

    if cfg.dataset.overfit:
        validation_dataset_holder.get_dataset = lambda: train_dataloader.dataset
    
    validation_dataloader = validation_dataset_holder.get_dataloader()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.trainer.gradient_accumulation_steps)
    if cfg.trainer.max_train_steps is None:
        cfg.trainer.max_train_steps = cfg.trainer.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.trainer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.trainer.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.trainer.lr_num_cycles,
        power=cfg.trainer.lr_power,
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    match cfg.model.model_type:
        case ModelType.CONTROLNET:
            vae, unet, text_encoder, controlnet = pre_train_setup_controlnet(weight_dtype, cfg, accelerator, vae, unet, text_encoder, controlnet)
        case ModelType.BASE_MAPPER:
            model.pre_train_setup_base_mapper(weight_dtype, accelerator)
            noise_scheduler, vae, unet, text_encoder = model.noise_scheduler, model.vae, model.unet, model.text_encoder

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.trainer.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.trainer.max_train_steps = cfg.trainer.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.trainer.num_train_epochs = math.ceil(cfg.trainer.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = cfg.dataset.train_dataset.batch_size * accelerator.num_processes * cfg.trainer.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.trainer.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataset.train_dataset.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.trainer.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.trainer.resume_from_checkpoint:
        if cfg.trainer.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.trainer.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.trainer.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.trainer.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if cfg.profile:
        profiler = Profiler(output_dir=cfg.output_dir, active_steps=cfg.trainer.profiler_active_steps)

    progress_bar = tqdm(range(0, cfg.trainer.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    if cfg.trainer.log_gradients is not None:
        wandb.watch(model, log_freq=cfg.trainer.log_gradients)
    print(f'load_time: {__import__("time").time() - load_time} seconds')

    image_logs = None
    for epoch in range(first_epoch, cfg.trainer.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet if cfg.model.model_type == ModelType.CONTROLNET else text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                match cfg.model.model_type:
                    case ModelType.CONTROLNET:
                        model_pred = controlnet_forward(batch, noisy_latents, timesteps, weight_dtype, unet, text_encoder, controlnet)
                    case ModelType.BASE_MAPPER:
                        model_pred = model(batch, noisy_latents, timesteps, weight_dtype)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # TODO: WARNING: THIS MAY NOT WORK! IF THERE ARE ISSUES WITH OPTIMIZATION, THIS MAY BE THE CAUSE
                    # params_to_optimize should be a function pointer
                    params_to_clip = params_to_optimize()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.trainer.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.trainer.set_grads_to_none)

            # Important Note: Right now a single "global_step" is a single gradient update step (same if we don't have grad accum)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    if global_step % cfg.trainer.checkpointing_steps == 0:
                        handle_checkpointing(cfg, accelerator, global_step)
                    
                    if global_step % cfg.trainer.num_val_steps == 0 or (step == 0 and cfg.trainer.eval_every_n_epochs and epoch % cfg.trainer.eval_every_n_epochs == 0):
                        logger.info(f'Starting validation at step {global_step}, epoch {epoch}')
                        match cfg.model.model_type:
                            case ModelType.CONTROLNET:
                                if cfg.dataset.validation_prompt:
                                    image_logs = log_validation(vae, text_encoder, tokenizer, unet, controlnet, cfg, accelerator, weight_dtype, global_step)
                            case ModelType.BASE_MAPPER:
                                validator.infer(accelerator, validation_dataloader, tokenizer, text_encoder, unet, vae, cfg.dataset.num_validation_images, global_step)
                        logger.info(f'Finished validation at step {global_step}, epoch {epoch}')

                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item() / cfg.trainer.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= cfg.trainer.max_train_steps:
                break
            elif cfg.profile and profiler.step(global_step):
                print(f"Profiling finished at step: {global_step}")
                break

            # TODO: Something weird happens with webdataset:
            # UserWarning: Length of IterableDataset <abc.WebDataset_Length object at 0x7f0748da4640> was reported to be 2 (when accessing len(dataloader)), but 3 samples have been fetched.
            if step >= len(train_dataloader) - 1:
                break   

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.profile:
                profiler.finish()
                exit()

        match cfg.model.model_type:
            case ModelType.CONTROLNET:
                controlnet = accelerator.unwrap_model(controlnet)
                controlnet.save_pretrained(cfg.output_dir)
            case ModelType.BASE_MAPPER:
                checkpoint_handler.save_model(text_encoder=text_encoder, accelerator=accelerator, save_name='last')

    accelerator.end_training()