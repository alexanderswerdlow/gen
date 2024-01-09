import autoroot

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import hydra

import numpy as np
import torch
from tqdm import tqdm
import typer
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from image_utils import Im, get_layered_image_from_binary_mask
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from transformers import CLIPTokenizer

from gen.configs.base import BaseConfig
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.checkpoint_handler import CheckpointHandler
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.neti_mapper import UNET_LAYERS, NeTIMapper
from gen.models.neti.prompt_manager import PromptManager
from gen.models.neti.sd_pipeline import sd_pipeline_call
from gen.models.neti.xti_attention_processor import XTIAttenProc
from accelerate import Accelerator
from accelerate.utils import PrecisionType
import torch.distributed as dist

def remove_row(tensor, row_index):
    return torch.cat((tensor[:row_index], tensor[row_index+1:]))

def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new("RGB", (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def inference(inference_cfg: BaseConfig, accelerator: Accelerator):
    if inference_cfg.inference.inference_dir is None:
        assert inference_cfg.inference.input_dir is not None, "You must pass an input_dir if you do not specify inference_dir"
        inference_cfg.inference.inference_dir = inference_cfg.inference.input_dir / f"inference_{inference_cfg.inference.iteration}"
    if inference_cfg.inference.mapper_checkpoint_path is None:
        assert inference_cfg.inference.input_dir is not None, "You must pass an input_dir if you do not specify mapper_checkpoint_path"
        inference_cfg.inference.mapper_checkpoint_path = (
            inference_cfg.inference.input_dir / "checkpoints" / f"mapper-steps-{inference_cfg.inference.iteration}.pt"
        )
    if inference_cfg.inference.learned_embeds_path is None:
        assert inference_cfg.inference.input_dir is not None, "You must pass an input_dir if you do not specify learned_embeds_path"
        inference_cfg.inference.learned_embeds_path = (
            inference_cfg.inference.input_dir / "checkpoints" / f"learned_embeds-steps-{inference_cfg.inference.iteration}.bin"
        )

    inference_cfg.inference.inference_dir.mkdir(exist_ok=True, parents=True)
    if type(inference_cfg.inference.truncation_idxs) == int:
        inference_cfg.inference.truncation_idxs = [inference_cfg.inference.truncation_idxs]
    torch_dtype = torch.bfloat16 if inference_cfg.trainer.mixed_precision == PrecisionType.BF16 else torch.float32

    train_cfg, mapper = CheckpointHandler.load_mapper(inference_cfg.inference.mapper_checkpoint_path)

    pipeline, placeholder_token, placeholder_token_id = load_stable_diffusion_model(
        pretrained_model_name_or_path=train_cfg.model.pretrained_model_name_or_path,
        mapper=mapper,
        learned_embeds_path=inference_cfg.inference.learned_embeds_path,
        torch_dtype=torch_dtype,
    )

    pipeline.text_encoder.text_model.embeddings.mapper.eval()
    model = BaseMapper(train_cfg, init_modules=False)
    model.tokenizer = pipeline.tokenizer
    model.text_encoder = pipeline.text_encoder
    model.unet = pipeline.unet
    model.vae = pipeline.vae
    if train_cfg.trainer.enable_xformers_memory_efficient_attention:
        import xformers

        pipeline.unet.enable_xformers_memory_efficient_attention()
    model.unet.set_attn_processor(XTIAttenProc())

    model.pre_train_setup_base_mapper(torch_dtype, accelerator, bypass_dtype_check=True)

    prompt_manager = PromptManager(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        timesteps=pipeline.scheduler.timesteps,
        unet_layers=UNET_LAYERS,
        placeholder_token=placeholder_token,
        placeholder_token_id=placeholder_token_id,
        torch_dtype=torch_dtype,
        model=model,
    )

    validation_dataloader = hydra.utils.instantiate(train_cfg.dataset.validation_dataset, _recursive_=False)(
        cfg=inference_cfg, split=Split.VALIDATION, tokenizer=pipeline.tokenizer, accelerator=accelerator
    ).get_dataloader()

    validation_dataloader, model = accelerator.prepare(validation_dataloader, model)

    for i, batch in tqdm(enumerate(validation_dataloader), leave=False, disable=not accelerator.is_local_main_process):
        output_path = inference_cfg.output_dir
        output_path.mkdir(exist_ok=True, parents=True)

        # for truncation_idx in inference_cfg.inference.truncation_idxs:
        #     print(f"Running with truncation index: {truncation_idx}")
        #     prompt_image = run_inference_batch(
        #         batch=batch,
        #         pipeline=pipeline,
        #         prompt_manager=prompt_manager,
        #         seeds=inference_cfg.inference.seeds,
        #         num_images_per_prompt=1,
        #         truncation_idx=truncation_idx,
        #     )
        #     save_name = f'{i}.png'
        #     prompt_image.save(inference_cfg.inference.inference_dir / save_name)

        for truncation_idx in inference_cfg.inference.truncation_idxs:
            print(f"Running with truncation index: {truncation_idx}")
            gen_segmentation = batch["gen_segmentation"]
            images = []
            masks = []
            input_ids = batch['input_ids']
            for j in range(gen_segmentation.shape[-1])[:10]:
                batch["gen_segmentation"] = gen_segmentation[..., torch.arange(gen_segmentation.size(-1)) != j]
                batch["input_ids"] = input_ids.clone() # We modify input_ids inside the mapper, so we need to clone it
                prompt_image = run_inference_batch(
                    batch=batch,
                    pipeline=pipeline,
                    prompt_manager=prompt_manager,
                    seeds=inference_cfg.inference.seeds,
                    num_images_per_prompt=1,
                    truncation_idx=truncation_idx,
                )
                images.append(prompt_image)
                masks.append(get_layered_image_from_binary_mask(gen_segmentation[..., [j]].squeeze(0)))
                        
            gen_images = Im.concat_horizontal(((batch['gen_pixel_values'].squeeze(0) + 1) / 2).cpu(), *images)
            gen_masks = Im.concat_horizontal(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)),*masks)
            output = Im.concat_vertical(gen_images, gen_masks)
            output_filename = output_path / f'{dist.get_rank() if accelerator.use_distributed else 0}{i}_{j}.png'
            output.save(output_filename)
            print(f"Saved to {output_filename}")


def run_inference_batch(
    pipeline: StableDiffusionPipeline,
    prompt_manager: PromptManager,
    seeds: List[int],
    batch: dict,
    num_images_per_prompt: int = 1,
    truncation_idx: Optional[int] = None,
) -> Image.Image:
    assert num_images_per_prompt == 1, "Only num_images_per_prompt=1 is supported for now"
    assert len(seeds) == 1, "Only one seed is supported for now"
    with torch.autocast("cuda"):
        with torch.no_grad():
            prompt_embeds = prompt_manager.embed_prompt(batch=batch, num_images_per_prompt=num_images_per_prompt, truncation_idx=truncation_idx)
    generator = torch.Generator(device="cuda").manual_seed(seeds[0])
    images = sd_pipeline_call(pipeline, prompt_embeds=prompt_embeds, generator=generator, num_images_per_prompt=num_images_per_prompt).images[0]
    return images


def load_stable_diffusion_model(
    pretrained_model_name_or_path: str,
    learned_embeds_path: Path,
    mapper: Optional[NeTIMapper] = None,
    num_denoising_steps: int = 50,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[StableDiffusionPipeline, str, int]:
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = NeTICLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    )
    if mapper is not None:
        text_encoder.text_model.embeddings.set_mapper(mapper)
    placeholder_token, placeholder_token_id = CheckpointHandler.load_learned_embed_in_clip(
        learned_embeds_path=learned_embeds_path, text_encoder=text_encoder, tokenizer=tokenizer
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch_dtype, text_encoder=text_encoder, tokenizer=tokenizer
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
    pipeline.unet.set_attn_processor(XTIAttenProc())
    return pipeline, placeholder_token, placeholder_token_id
