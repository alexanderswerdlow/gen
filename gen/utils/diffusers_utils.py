from typing import Optional, Union

import torch
from gen.models.cross_attn.pipeline_stable_diffusion import StableDiffusionPipeline
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
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.utils.trainer_utils import Trainable, unwrap


def load_stable_diffusion_model(
    cfg: BaseConfig,
    device: torch.device,
    model: Trainable,
    torch_dtype: torch.dtype,
    tokenizer: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[NeTICLIPTextModel] = None,
    unet: Optional[UNet2DConditionModel] = None,
    vae: Optional[AutoencoderKL] = None,
) -> Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline]:
    """Loads SD model given the current text encoder and our mapper."""
    assert not cfg.model.controlnet or hasattr(model, "controlnet"), "You must pass a controlnet model to use controlnet."

    disable_progress_bar()
    cls = StableDiffusionControlNetPipeline if cfg.model.controlnet else StableDiffusionPipeline
    pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path

    kwargs = dict(pretrained_model_name_or_path=pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unwrap(unet), vae=vae)

    if cfg.model.controlnet:
        kwargs["controlnet"] = model.controlnet

    if tokenizer is None:
        kwargs["tokenizer"] = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    if text_encoder is None:
        text_cls = NeTICLIPTextModel if cfg.model.per_timestep_conditioning else CLIPTextModel
        kwargs["text_encoder"] = text_cls.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
        if cfg.model.per_timestep_conditioning:
            text_encoder.text_model.set_mapper(mapper=model, cfg=cfg)

    if cfg.model.per_timestep_conditioning:
        unwrap(text_encoder).eval()
        if not cfg.model.freeze_unet:
            unwrap(pipeline.unet).eval()

    pipeline = cls.from_pretrained(**kwargs)
    pipeline = pipeline.to(device)

    if cfg.inference.use_ddim:
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    scheduler.set_timesteps(cfg.inference.num_denoising_steps, device=pipeline.device)
    pipeline.scheduler = scheduler
    pipeline.set_progress_bar_config(disable=True)

    # if cfg.model.lora_unet:
    #     pipeline.load_lora_weights(args.output_dir)

    if cfg.model.per_timestep_conditioning:
        pipeline.unet.set_attn_processor(XTIAttenProc())

    return pipeline
