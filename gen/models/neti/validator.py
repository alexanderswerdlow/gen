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
            model: BaseMapper,
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
        pipeline = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, unet, vae)
        prompt_manager = PromptManager(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, timesteps=pipeline.scheduler.timesteps, placeholder_token=self.cfg.model.placeholder_token, placeholder_token_id=self.cfg.model.placeholder_token_id, model=model, torch_dtype=self.weight_dtype)

        joined_images = []
        for idx, batch in tqdm(enumerate(validation_dataloader), leave=False, disable=not accelerator.is_local_main_process):
            images = self.infer_on_prompt(accelerator=accelerator, pipeline=pipeline, prompt_manager=prompt_manager, num_images_per_prompt=num_images_per_prompt, seeds=seeds, batch=batch)
            images = torch.from_numpy(np.concatenate([Im(batch["gen_pixel_values"]).denormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).pil, *images], axis=1)).to(batch["gen_pixel_values"].device)
            joined_images.append(images)
            # TODO: Show mask information

            # TODO: Something weird happens with webdataset:
            # UserWarning: Length of IterableDataset <abc.WebDataset_Length object at 0x7f0748da4640> was reported to be 2 (when accessing len(dataloader)), but 3 samples have been fetched.
            if idx >= len(validation_dataloader) - 1:
                break
        
        joined_images = accelerator.gather(joined_images)
        if accelerator.is_local_main_process:
            joined_images = [Image.fromarray(img.cpu().numpy()) for img in joined_images]
            final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
            img_path = (self.cfg.output_dir / 'images')
            img_path.mkdir(exist_ok=True)
            final_image.save(img_path / f"val-image-{step}.png")
            self.log_with_accelerator(accelerator, joined_images, step=step)
        del pipeline
        torch.cuda.empty_cache()

        accelerator.unwrap_model(text_encoder).text_model.embeddings.mapper.train()

        # if self.cfg.trainer.seed is not None:
        #     set_seed(self.cfg.trainer.seed)

    def infer_on_prompt(self,
                        accelerator: Accelerator,
                        pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        seeds: List[int],
                        batch: dict,
                        num_images_per_prompt: int = 1
                    ) -> List[Image.Image]:
        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager, batch=batch)
        all_images = []
        for idx in tqdm(range(num_images_per_prompt), leave=False, disable=not accelerator.is_local_main_process):
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
        accelerator.unwrap_model(text_encoder).text_model.embeddings.mapper.eval()

        return pipeline

    def log_with_accelerator(self, accelerator: Accelerator, images: List[Image.Image], step: int):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log({"validation": [wandb.Image(image) for i, image in enumerate(images)]})
