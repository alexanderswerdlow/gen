import autoroot

import math
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import PrecisionType
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline, UNet2DConditionModel)
from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from gen.configs.base import BaseConfig
from gen.configs.inference import InferenceConfig
from gen.datasets.base_dataset import Split
from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.checkpoint_handler import CheckpointHandler
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.neti_mapper import UNET_LAYERS, NeTIMapper
from gen.models.neti.sd_pipeline import sd_pipeline_call
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.utils.attention_visualization_utils import (
    cross_attn_init, get_all_net_attn_maps, get_net_attn_map,
    register_cross_attention_hook, resize_net_attn_map)
from gen.utils.decoupled_utils import get_rank, is_main_process
from gen.utils.logging_utils import log_info
from accelerate.utils import gather_object as accelerate_gather_object, gather as accelerate_gather

def gather(device: torch.device, img: Union[Im, Iterable[Im]], gather_different: bool = False):
    if gather_different:
        ret = accelerate_gather_object(img)
    else:
        if isinstance(img, Iterable):
            tensor = torch.cat([img_.torch.to(device).unsqueeze(0) for img_ in img], dim=0)
        else:
            tensor = img.torch.to(device).unsqueeze(0)

        concat_tensor = accelerate_gather(tensor)

        try:
            ret = [Im(concat_tensor[i]) for i in range(concat_tensor.shape[0])]
        except:
            print(concat_tensor.shape, concat_tensor.dtype, concat_tensor.device)
            raise

    return ret


def remove_row(tensor, row_index):
    return torch.cat((tensor[:row_index], tensor[row_index + 1 :]))


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

    train_cfg, mapper, clip_state_dict = CheckpointHandler.load_mapper(inference_cfg.inference.mapper_checkpoint_path)
    train_cfg.inference = inference_cfg.inference

    pipeline = load_stable_diffusion_model(
        accelerator=accelerator,
        model=mapper,
        torch_dtype=torch_dtype,
        cfg=train_cfg,
    )

    model = BaseMapper(train_cfg, init_modules=False)

    if clip_state_dict is not None:
        model.clip.load_state_dict(clip_state_dict)
    else:
        warnings.warn("No clip state dict found in mapper checkpoint, using pretrained clip state dict")

    model.tokenizer = pipeline.tokenizer
    model.text_encoder = pipeline.text_encoder
    model.unet = pipeline.unet
    model.vae = pipeline.vae
    if train_cfg.trainer.enable_xformers_memory_efficient_attention:
        import xformers

        pipeline.unet.enable_xformers_memory_efficient_attention()
    model.unet.set_attn_processor(XTIAttenProc())

    model.prepare_for_training(torch_dtype, accelerator, bypass_dtype_check=True)

    validation_dataloader = hydra.utils.instantiate(train_cfg.dataset.validation_dataset, _recursive_=False)(
        cfg=inference_cfg, split=Split.VALIDATION, tokenizer=pipeline.tokenizer, accelerator=accelerator
    ).get_dataloader()

    validation_dataloader, model = accelerator.prepare(validation_dataloader, model)

    run_inference_dataloader(
        accelerator=accelerator,
        model=model,
        pipeline=pipeline,
        output_path=inference_cfg.output_dir,
        dataloader=validation_dataloader,
        inference_cfg=inference_cfg.inference,
    )


def run_inference_dataloader(
    accelerator: Accelerator,
    model: BaseMapper,
    pipeline: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline],
    output_path: Path,
    dataloader: torch.utils.data.DataLoader,
    inference_cfg: InferenceConfig,
    global_step: Optional[int] = None,
):
    output_path.mkdir(exist_ok=True, parents=True)
    all_output_images = []
    all_output_attn_viz = []

    log_info(f"Running inference. Dataloder size: {len(dataloader)}")
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        log_info(f"Generating with batch {i}")
        orig_input_ids = batch["input_ids"].clone()

        images = []
        orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
        gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))

        batch["input_ids"] = orig_input_ids.clone()
        prompt_image, input_prompt, prompt_embeds = run_inference_batch(
            batch=batch,
            model=model,
            pipeline=pipeline,
            visualize_attention_map=inference_cfg.visualize_attention_map,
            inference_cfg=inference_cfg,
        )

        images.append(Im(prompt_image))
        if inference_cfg.visualize_attention_map:
            net_attn_maps = get_net_attn_map(prompt_image.size, chunk=False)
            net_attn_maps = resize_net_attn_map(net_attn_maps, prompt_image.size)
            tokens = [x.replace("</w>", "") for x in input_prompt[0]]
            attn_maps = get_all_net_attn_maps(net_attn_maps, tokens)
            mask_idx = 0
            output_cols = []
            for idx, (attn_map, token) in enumerate(zip(attn_maps, tokens)):
                if token == "placeholder":
                    mask_bool = batch["gen_segmentation"][..., mask_idx].squeeze(0).cpu().bool().numpy()
                    orig_image_ = orig_image.np.copy()
                    orig_image_[~mask_bool] = 0
                    orig_image_ = Im(orig_image_, channel_range=ChannelRange.UINT8)

                    output_col = Im.concat_vertical(Im(attn_map.convert("RGB")), orig_image_, spacing=5).write_text(f"mask: {mask_idx}")
                    mask_idx += 1
                else:
                    orig_image_ = Im(255 * np.ones((inference_cfg.resolution, inference_cfg.resolution, 3), dtype=np.uint8))
                    output_col = Im.concat_vertical(Im(attn_map.convert("RGB")), orig_image_, spacing=5).write_text(f"{token}")

                output_cols.append(output_col)

            output_attn_viz = Im.concat_horizontal(output_cols, spacing=5)
            all_output_attn_viz.append(output_attn_viz)

            if is_main_process():
                embeds_ = torch.stack([v[1] for k,v in prompt_embeds[0].items() if 'CONTEXT_TENSOR' in k and 'BYPASS' not in k], dim=0)
                bypass_embeds_ = torch.stack([v[1] for k,v in prompt_embeds[0].items() if 'CONTEXT_TENSOR' in k and 'BYPASS' in k], dim=0)
                
                inputs_ = pipeline.tokenizer(" ".join([x.replace("</w>", "") for x in input_prompt[0]]), max_length=pipeline.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                inputs_['input_ids'] = inputs_['input_ids'].to(pipeline.text_encoder.device)
                inputs_['attention_mask'] = inputs_['attention_mask'].to(pipeline.text_encoder.device)
                regular_embeds_ = pipeline.text_encoder(**inputs_).last_hidden_state
                embeds_, bypass_embeds_, regular_embeds_ = embeds_[:, :len(input_prompt[0])], bypass_embeds_[:, :len(input_prompt[0])], regular_embeds_[:, :len(input_prompt[0])]

                output_path_ = output_path / 'tmp.png'
                embeds_.plt.fig.savefig(output_path_)
                log_with_accelerator(accelerator, [Im(output_path_)], global_step=i, name="embeds", save_folder=output_path)
                bypass_embeds_.plt.fig.savefig(output_path_)
                log_with_accelerator(accelerator, [Im(output_path_)], global_step=i, name="bypass_embeds", save_folder=output_path)
                regular_embeds_.plt.fig.savefig(output_path_)
                log_with_accelerator(accelerator, [Im(output_path_)], global_step=i, name="regular_embeds", save_folder=output_path)

        if inference_cfg.num_masks_to_remove is not None:
            gen_segmentation = batch["gen_segmentation"]

            for j in range(gen_segmentation.shape[-1])[: inference_cfg.num_masks_to_remove]:
                log_info(f"Generating with removed mask {j}")
                batch["gen_segmentation"] = gen_segmentation[..., torch.arange(gen_segmentation.size(-1)) != j]
                batch["input_ids"] = orig_input_ids.clone()
                prompt_image, input_prompt, prompt_embeds = run_inference_batch(
                    batch=batch,
                    model=model,
                    pipeline=pipeline,
                    inference_cfg=inference_cfg,
                )
                mask_image = Im(get_layered_image_from_binary_mask(gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
                mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
                mask_alpha = (gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
                composited_image = orig_image.pil.copy().convert("RGBA")
                composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
                composited_image = composited_image.convert("RGB")
                images.append(Im.concat_vertical(prompt_image, mask_image, composited_image, spacing=5, fill=(128, 128, 128)))

        gen_results = Im.concat_horizontal(images, spacing=5, fill=(128, 128, 128))
        output_images = Im.concat_horizontal(gt_info, gen_results, spacing=50, fill=(255, 255, 255))
        all_output_images.append(output_images)

    device = batch["gen_pixel_values"].device
    output_images = gather(device, all_output_images)
    log_with_accelerator(
        accelerator=accelerator,
        images=output_images,
        save_folder=output_path,
        name="validation",
        global_step=(global_step if global_step is not None else i),
    )

    if inference_cfg.visualize_attention_map:
        output_attn_viz = gather(device, all_output_attn_viz, gather_different=True)
        log_with_accelerator(
            accelerator=accelerator,
            images=output_attn_viz,
            save_folder=output_path,
            name="attention",
            global_step=(global_step if global_step is not None else i),
        )

    log_info(f"Saved to {output_path}")


def run_inference_batch(
    batch: dict,
    model: BaseMapper,
    pipeline: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline],
    inference_cfg: InferenceConfig,
    seed: int = 42,
    num_images_per_prompt: int = 1,
    truncation_idx: Optional[int] = None,
    visualize_attention_map: bool = False,
) -> Image.Image:
    assert num_images_per_prompt == 1, "Only num_images_per_prompt=1 is supported for now"
    if visualize_attention_map:
        cross_attn_init()
        pipeline.unet = register_cross_attention_hook(pipeline.unet)

    negative_prompt_embeds = None
    with torch.no_grad():
        if not inference_cfg.empty_string_cfg:
            negative_prompt_embeds, _ = model.embed_prompt(
                batch=batch,
                num_images_per_prompt=num_images_per_prompt,
                truncation_idx=truncation_idx,
                disable_conditioning=True,
                timesteps=pipeline.scheduler.timesteps,
            )
        prompt_embeds, input_prompt = model.embed_prompt(
            batch=batch,
            num_images_per_prompt=num_images_per_prompt,
            truncation_idx=truncation_idx,
            timesteps=pipeline.scheduler.timesteps,
        )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = sd_pipeline_call(
        pipeline,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=inference_cfg.guidance_scale,
    ).images[0]
    return images, input_prompt, prompt_embeds


def load_stable_diffusion_model(
    cfg: BaseConfig,
    accelerator: Accelerator,
    model: BaseMapper,
    torch_dtype: torch.dtype,
    tokenizer: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[NeTICLIPTextModel] = None,
    unet: Optional[UNet2DConditionModel] = None,
    vae: Optional[AutoencoderKL] = None,
) -> Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline]:
    """Loads SD model given the current text encoder and our mapper."""
    assert not cfg.model.controlnet or hasattr(model, "controlnet"), "You must pass a controlnet model to use controlnet."
    cls = StableDiffusionControlNetPipeline if cfg.model.controlnet else StableDiffusionPipeline
    pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path

    kwargs = dict(pretrained_model_name_or_path=pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, vae=vae)

    if cfg.model.controlnet:
        kwargs['controlnet'] = model.controlnet

    if tokenizer is None:
        kwargs['tokenizer'] = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    if text_encoder is None:
        kwargs['text_encoder'] = NeTICLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
        text_encoder.text_model.set_mapper(mapper=model, cfg=cfg)

    pipeline = cls.from_pretrained(**kwargs)
    pipeline = pipeline.to(accelerator.device)

    # placeholder_token, placeholder_token_id = CheckpointHandler.load_learned_embed_in_clip(
    #     learned_embeds_path=learned_embeds_path, text_encoder=text_encoder, tokenizer=tokenizer
    # )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.scheduler.set_timesteps(cfg.inference.num_denoising_steps, device=pipeline.device)
    pipeline.unet.set_attn_processor(XTIAttenProc())

    accelerator.unwrap_model(text_encoder).eval()

    if cfg.model.controlnet:
        accelerator.unwrap_model(pipeline.controlnet).set_attn_processor(XTIAttenProc())
        accelerator.unwrap_model(pipeline.controlnet).eval()

    return pipeline

def log_with_accelerator(accelerator: Accelerator, images: List[Image.Image], global_step: int, name: str, save_folder: Optional[Path] = None):
    save_folder.parent.mkdir(exist_ok=True)

    if len(images) == 1:
        save_path = save_folder / f"{name}_{global_step}.png"
        Im(images[0]).save(save_path)
    else:
        for i in range(len(images)):
            save_path = save_folder / f"{name}_{global_step}_{i}.png"
            Im(images[i]).save(save_path)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(name, np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({name: [wandb.Image(image.pil) for i, image in enumerate(images)]}, step=global_step)
