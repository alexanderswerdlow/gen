import os
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from packaging import version
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig
from accelerate import Accelerator

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from gen.configs import BaseConfig
import wandb

logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def get_controlnet_model(cfg: BaseConfig, accelerator: Accelerator) -> tuple[AutoTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel]:
    # Load the tokenizer
    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name, revision=cfg.model.revision, use_fast=False)
    elif cfg.model.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.model.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(cfg.model.pretrained_model_name_or_path, cfg.model.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.model.revision, variant=cfg.model.variant
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=cfg.model.revision, variant=cfg.model.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=cfg.model.revision, variant=cfg.model.variant
    )

    if cfg.model.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(cfg.model.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if cfg.trainer.enable_xformers_memory_efficient_attention:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    return tokenizer, noise_scheduler, text_encoder, vae, unet, controlnet

def pre_train_setup_controlnet(weight_dtype: torch.dtype, cfg: BaseConfig, accelerator: Accelerator, vae, unet, text_encoder, controlnet):
    controlnet = accelerator.prepare(controlnet)

    if cfg.trainer.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}.")

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if cfg.trainer.compile:
        unet.to(memory_format=torch.channels_last)
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)

        controlnet.to(memory_format=torch.channels_last)
        controlnet = torch.compile(controlnet, mode="reduce-overhead", fullgraph=True)

    return vae, unet, text_encoder, controlnet

def controlnet_forward(batch, noisy_latents, timesteps, weight_dtype, unet, text_encoder, controlnet):
    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_image,
        return_dict=False,
    )

    # Predict the noise residual
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=[
            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
    ).sample

    return model_pred

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, cfg: BaseConfig, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=cfg.model.revision,
        variant=cfg.model.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if cfg.trainer.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if cfg.trainer.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.trainer.seed)

    if len(cfg.dataset.validation_image) == len(cfg.dataset.validation_prompt) and type(cfg.dataset.validation_prompt) != str:
        validation_images = cfg.dataset.validation_image
        validation_prompts = cfg.dataset.validation_prompt
    else: # Assume we have a list of images and 1 prompt
        validation_images = cfg.dataset.validation_image
        validation_prompts = cfg.dataset.validation_prompt * len(cfg.dataset.validation_image)

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(cfg.dataset.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs