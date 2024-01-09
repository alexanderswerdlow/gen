from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from tqdm import tqdm
from transformers import CLIPTokenizer
from gen.configs.base import BaseConfig
from gen.models.base_mapper_model import BaseMapper
import torch.distributed as dist

from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.prompt_manager import PromptManager
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.models.neti.sd_pipeline import sd_pipeline_call
from image_utils import Im
from torch.nn.parallel import DistributedDataParallel

from gen.utils.trainer_utils import every_n_steps
from inference import run_inference_batch, run_inference_dataloader

if is_wandb_available():
    import wandb


class ValidationHandler:
    def __init__(self, cfg: BaseConfig, weights_dtype: torch.dtype):
        self.cfg = cfg
        self.weight_dtype = weights_dtype

    # @every_n_steps(n=self.cfg)
    def infer(
        self,
        accelerator: Accelerator,
        validation_dataloader: torch.utils.data.DataLoader,
        model: BaseMapper,
        tokenizer: CLIPTokenizer,
        text_encoder: NeTICLIPTextModel,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        num_images_per_prompt: int,
        global_step: int,
        seeds: Optional[List[int]] = None,
    ):
        if seeds is None:
            seeds = list(range(num_images_per_prompt))

        """ Runs inference during our training scheme. """
        pipeline = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, unet, vae)
        prompt_manager = PromptManager(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            timesteps=pipeline.scheduler.timesteps,
            placeholder_token=self.cfg.model.placeholder_token,
            placeholder_token_id=self.cfg.model.placeholder_token_id,
            model=model,
            torch_dtype=self.weight_dtype,
        )

        run_inference_dataloader(
            accelerator=accelerator,
            pipeline=pipeline,
            prompt_manager=prompt_manager,
            dataloader=validation_dataloader,
            output_path=self.cfg.output_dir / 'images',
            remove_masks_for_editing=True,
            num_masks_to_remove=3,
            global_step=global_step,
        )

        del pipeline
        torch.cuda.empty_cache()
        accelerator.unwrap_model(text_encoder).text_model.embeddings.mapper.train()

    def load_stable_diffusion_model(
        self, accelerator: Accelerator, tokenizer: CLIPTokenizer, text_encoder: NeTICLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL
    ) -> StableDiffusionPipeline:
        """Loads SD model given the current text encoder and our mapper."""
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        num_denoising_steps = 50
        pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        accelerator.unwrap_model(text_encoder).text_model.embeddings.mapper.eval()

        return pipeline