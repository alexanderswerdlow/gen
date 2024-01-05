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

from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.prompt_manager import PromptManager
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.models.neti.sd_pipeline import sd_pipeline_call

if is_wandb_available():
    import wandb


class ValidationHandler:
    def __init__(self, cfg: BaseConfig, weights_dtype: torch.dtype):
        self.cfg = cfg
        self.weight_dtype = weights_dtype

    def infer(
            self,
            accelerator: Accelerator,
            validation_dataloader: torch.utils.data.DataLoader,
            tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel,
            unet: UNet2DConditionModel, 
            vae: AutoencoderKL,
            num_images_per_prompt: int,
            step: int,
            seeds: Optional[List[int]] = None,
        ):
    
        if seeds is None:
            seeds = list(range(num_images_per_prompt))

        """ Runs inference during our training scheme. """
        pipeline, model = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, unet, vae)
        model, validation_dataloader = accelerator.prepare(model, validation_dataloader)
        prompt_manager = PromptManager(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, timesteps=pipeline.scheduler.timesteps, placeholder_token=self.cfg.model.placeholder_token, placeholder_token_id=self.cfg.model.placeholder_token_id, model=model, torch_dtype=self.weight_dtype)

        joined_images = []
        for batch in validation_dataloader:
            images = self.infer_on_prompt(pipeline=pipeline, prompt_manager=prompt_manager, num_images_per_prompt=num_images_per_prompt, seeds=seeds, batch=batch)
            prompt_image = Image.fromarray(np.concatenate(images, axis=1))
            joined_images.append(prompt_image)

        final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")
        self.log_with_accelerator(accelerator, joined_images, step=step)
        del pipeline
        torch.cuda.empty_cache()
        text_encoder.text_model.embeddings.mapper.train()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self, pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        seeds: List[int],
                        batch: dict,
                        num_images_per_prompt: int = 1) -> List[Image.Image]:
        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager, batch=batch)
        all_images = []
        for idx in tqdm(range(num_images_per_prompt)):
            generator = torch.Generator(device='cuda').manual_seed(seeds[idx])
            images = sd_pipeline_call(pipeline, prompt_embeds=prompt_embeds, generator=generator, num_images_per_prompt=1).images
            all_images.extend(images)
        return all_images

    @staticmethod
    def compute_embeddings(prompt_manager: PromptManager, batch: dict) -> torch.Tensor:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(batch)
        return prompt_embeds

    def load_stable_diffusion_model(
            self, 
            accelerator: Accelerator,
            tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel,
            unet: UNet2DConditionModel,
            vae: AutoencoderKL
        ) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, 
            text_encoder=accelerator.unwrap_model(text_encoder), 
            tokenizer=tokenizer, 
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        num_denoising_steps = 50
        pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        text_encoder.text_model.embeddings.mapper.eval()

        model = BaseMapper(self.cfg, init_modules=False)
        model.tokenizer = tokenizer
        model.text_encoder = text_encoder
        model.unet = unet
        model.vae = vae
        if self.cfg.trainer.enable_xformers_memory_efficient_attention:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        unet.set_attn_processor(XTIAttenProc())

        return pipeline, model

    def log_with_accelerator(self, accelerator: Accelerator, images: List[Image.Image], step: int):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(
                    "validation", np_images, step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log({"validation": [wandb.Image(image, caption=f"{i}: {self.cfg.eval.validation_prompts[i]}") for i, image in enumerate(images)]})
