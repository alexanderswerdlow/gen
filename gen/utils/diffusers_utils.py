from typing import Optional, Union

import torch
from gen.models.base.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.logging import disable_progress_bar
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel

from gen.configs.base import BaseConfig
from gen.utils.logging_utils import log_debug, log_info
from gen.utils.trainer_utils import Trainable, unwrap


def load_stable_diffusion_model(
    cfg: BaseConfig,
    device: torch.device,
    model: Trainable,
    torch_dtype: torch.dtype,
    tokenizer: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[CLIPTextModel] = None,
    unet: Optional[UNet2DConditionModel] = None,
    vae: Optional[AutoencoderKL] = None,
) -> Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline]:
    
    log_debug("Loading Diffusion Pipeline...", main_process_only=False)
    """Loads SD model given the current text encoder and our mapper."""
    assert not cfg.model.controlnet or hasattr(model, "controlnet"), "You must pass a controlnet model to use controlnet."

    disable_progress_bar()
    cls = StableDiffusionControlNetPipeline if cfg.model.controlnet else StableDiffusionPipeline
    pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path

    kwargs = dict(pretrained_model_name_or_path=pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unwrap(unet), vae=unwrap(vae), tokenizer=tokenizer, text_encoder=unwrap(text_encoder))

    if cfg.model.controlnet:
        kwargs["controlnet"] = model.controlnet

    if tokenizer is None:
        kwargs["tokenizer"] = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    if text_encoder is None:
        text_cls = CLIPTextModel
        encoder_name = "runwayml/stable-diffusion-v1-5" if pretrained_model_name_or_path == "lambdalabs/sd-image-variations-diffusers" else pretrained_model_name_or_path
        kwargs["text_encoder"] = text_cls.from_pretrained(
            encoder_name,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
    
    pipeline = cls.from_pretrained(**kwargs)
    pipeline = pipeline.to(device)
    # We do not set enable_xformers_memory_efficient_attention because we generally use a custom attention processor

    if cfg.inference.use_ddim:
        scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    scheduler.set_timesteps(cfg.inference.num_denoising_steps, device=pipeline.device)
    pipeline.scheduler = scheduler
    pipeline.set_progress_bar_config(disable=True)

    # if cfg.model.unet_lora:
    #     pipeline.load_lora_weights(args.output_dir)

    return pipeline
