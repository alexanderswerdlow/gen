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
from accelerate.utils import gather as accelerate_gather
from accelerate.utils import gather_object as accelerate_gather_object
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.logging import disable_progress_bar
from gen.utils.trainer_utils import TrainingState, Trainable

from image_utils import ChannelRange, Im, get_layered_image_from_binary_mask
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel
from gen.utils.trainer_utils import Trainable

from gen.configs.base import BaseConfig
from gen.configs.inference import InferenceConfig
from gen.datasets.base_dataset import Split
from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.checkpoint_handler import CheckpointHandler
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.sd_pipeline import sd_pipeline_call
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.models.utils import get_model_from_cfg
from gen.utils.attention_visualization_utils import (
    cross_attn_init,
    get_all_net_attn_maps,
    register_cross_attention_hook,
    resize_net_attn_map,
    retrieve_attn_maps_per_timestep,
    unregister_cross_attention_hook,
)
from gen.utils.decoupled_utils import get_rank, is_main_process
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.trainer_utils import unwrap

from gen.models.break_a_scene import aggregate_attention, save_cross_attention_vis


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


def inference(cfg: BaseConfig, accelerator: Accelerator):
    model = get_model_from_cfg(cfg)

    validation_dataloader = hydra.utils.instantiate(cfg.dataset.validation_dataset, _recursive_=False)(
        cfg=cfg, split=Split.VALIDATION, tokenizer=model.tokenizer, accelerator=accelerator
    ).get_dataloader()

    validation_dataloader, model = accelerator.prepare(validation_dataloader, model)

    model.run_inference(accelerator=accelerator, state=TrainingState(0, 0, 0, 0), dataloader=validation_dataloader)


def run_inference_dataloader(
    accelerator: Accelerator,
    model: BaseMapper,
    pipeline: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline],
    output_path: Path,
    dataloader: torch.utils.data.DataLoader,
    cfg: BaseConfig,
    global_step: Optional[int] = None,
):
    output_path.mkdir(exist_ok=True, parents=True)
    all_output_images = []
    all_output_attn_viz = []

    log_info(f"Running inference. Dataloder size: {len(dataloader)}")
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        from image_utils import Im

        log_info(f"Generating with batch {i}")
        orig_input_ids = batch["input_ids"].clone()

        images = []
        orig_image = Im((batch["gen_pixel_values"].squeeze(0) + 1) / 2)
        gt_info = Im.concat_vertical(orig_image, get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0))).write_text(
            text="GT", relative_font_scale=0.004
        )

        batch["input_ids"] = orig_input_ids.clone()
        prompt_image, input_prompt, prompt_embeds = run_inference_batch(
            batch=batch,
            model=model,
            pipeline=pipeline,
            visualize_attention_map=cfg.inference.visualize_attention_map,
            inference_cfg=cfg.inference,
            num_images_per_prompt=cfg.inference.num_images_per_prompt,
        )

        full_seg = Im(get_layered_image_from_binary_mask(batch["gen_segmentation"].squeeze(0)))
        images.append(
            Im.concat_horizontal(
                Im.concat_vertical(prompt_image_, full_seg).write_text(text=f"Gen {i}", relative_font_scale=0.004)
                for i, prompt_image_ in enumerate(prompt_image)
            )
        )

        if cfg.model.break_a_scene_cross_attn_loss:
            batch_size = (2 if cfg.inference.guidance_scale > 1.0 else 1) * cfg.inference.num_images_per_prompt

            agg_attn_cond = aggregate_attention(controller=model.controller, res=16, from_where=("up", "down"), is_cross=True, select=-1, batch_size=batch_size)
            attn_img_cond = Im(save_cross_attention_vis(tokenizer=model.tokenizer, tokens=batch["input_ids"][0], attention_maps=agg_attn_cond)).write_text("Cond")

            agg_attn_uncond = aggregate_attention(controller=model.controller, res=16, from_where=("up", "down"), is_cross=True, select=0, batch_size=batch_size)
            uncond_tokens = pipeline.tokenizer(
                [""],
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            attn_img_uncond = Im(save_cross_attention_vis(tokenizer=model.tokenizer, tokens=uncond_tokens['input_ids'][0], attention_maps=agg_attn_uncond)).write_text("Uncond")

            all_output_attn_viz.append(Im.concat_vertical(attn_img_cond, attn_img_uncond))

        elif cfg.inference.visualize_attention_map:
            desired_res = (64, 64)
            # inference_cfg.inference.guidance_scale > 1.0
            attn_maps_per_timestep = retrieve_attn_maps_per_timestep(
                image_size=prompt_image.size, timesteps=pipeline.scheduler.timesteps.shape[0], chunk=False
            )
            if attn_maps_per_timestep[0].shape[-2] != desired_res[0] or attn_maps_per_timestep[0].shape[-1] != desired_res[1]:
                attn_maps_per_timestep = resize_net_attn_map(attn_maps_per_timestep, desired_res)
            tokens = [x.replace("</w>", "") for x in input_prompt[0]]
            attn_maps_img_by_timestep = get_all_net_attn_maps(attn_maps_per_timestep, tokens)
            mask_idx = 0
            output_cols = []

            # fmt: off
            attn_viz_ = Im.concat_vertical(list(
                    Im.concat_horizontal(attn_maps, spacing=5).write_text(f"Timestep: {pipeline.scheduler.timesteps[idx].item()}", relative_font_scale=0.004)
                    for idx, attn_maps in enumerate(attn_maps_img_by_timestep)
            )[::5],spacing=5)
            # fmt: on

            for _, (attn_map, token) in enumerate(zip(attn_maps_img_by_timestep[0], tokens)):
                if token == model.cfg.inference.model.placeholder_token:
                    mask_bool = batch["gen_segmentation"][..., mask_idx].squeeze(0).cpu().bool().numpy()
                    orig_image_ = orig_image.np.copy()
                    orig_image_[~mask_bool] = 0
                    orig_image_ = Im(orig_image_, channel_range=ChannelRange.UINT8)
                    text_to_write = f"mask: {mask_idx}"
                    mask_idx += 1
                else:
                    orig_image_ = Im(255 * np.ones((desired_res[0], desired_res[1], 3), dtype=np.uint8))
                    text_to_write = f"{token}"

                output_cols.append(orig_image_.resize(*desired_res).write_text(text_to_write, relative_font_scale=0.004))

            all_output_attn_viz.append(
                Im.concat_vertical(attn_viz_, Im.concat_horizontal(output_cols, spacing=5), Im(prompt_image).resize(*desired_res))
            )

            if is_main_process() and cfg.inference.visualize_embeds:
                embeds_ = torch.stack([v[-1] for k, v in prompt_embeds[0].items() if "CONTEXT_TENSOR" in k and "BYPASS" not in k], dim=0)
                bypass_embeds_ = None
                if any("BYPASS" in k for k in prompt_embeds[0].keys()):
                    bypass_embeds_ = torch.stack([v[-1] for k, v in prompt_embeds[0].items() if "CONTEXT_TENSOR" in k and "BYPASS" in k], dim=0)

                inputs_ = pipeline.tokenizer(
                    " ".join([x.replace("</w>", "") for x in input_prompt[0]]),
                    max_length=pipeline.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs_["input_ids"] = inputs_["input_ids"].to(pipeline.text_encoder.device)
                inputs_["attention_mask"] = inputs_["attention_mask"].to(pipeline.text_encoder.device)
                regular_embeds_ = pipeline.text_encoder(**inputs_).last_hidden_state
                embeds_, regular_embeds_ = embeds_[:, : len(input_prompt[0])], regular_embeds_[:, : len(input_prompt[0])]
                if bypass_embeds_ is not None:
                    bypass_embeds_ = bypass_embeds_[:, : len(input_prompt[0])]

                try:
                    output_path_ = output_path / "tmp.png"
                    embeds_.plt.fig.savefig(output_path_)
                    log_with_accelerator(
                        accelerator,
                        [Im(output_path_)],
                        global_step=(global_step if global_step is not None else i),
                        name="embeds",
                        save_folder=output_path,
                    )
                    if bypass_embeds_ is not None:
                        bypass_embeds_.plt.fig.savefig(output_path_)
                        log_with_accelerator(
                            accelerator,
                            [Im(output_path_)],
                            global_step=(global_step if global_step is not None else i),
                            name="bypass_embeds",
                            save_folder=output_path,
                        )
                    regular_embeds_.plt.fig.savefig(output_path_)
                    log_with_accelerator(
                        accelerator,
                        [Im(output_path_)],
                        global_step=(global_step if global_step is not None else i),
                        name="regular_embeds",
                        save_folder=output_path,
                    )
                except:
                    log_warn("Nan likely found in embeds")

        if cfg.inference.num_masks_to_remove is not None:
            gen_segmentation = batch["gen_segmentation"]

            for j in range(gen_segmentation.shape[-1])[: cfg.inference.num_masks_to_remove]:
                log_info(f"Generating with removed mask {j}")
                batch["gen_segmentation"] = gen_segmentation[..., torch.arange(gen_segmentation.size(-1)) != j]
                batch["input_ids"] = orig_input_ids.clone()
                prompt_image, input_prompt, prompt_embeds = run_inference_batch(
                    batch=batch,
                    model=model,
                    pipeline=pipeline,
                    inference_cfg=cfg.inference,
                )
                mask_image = Im(get_layered_image_from_binary_mask(gen_segmentation[..., [j]].squeeze(0)), channel_range=ChannelRange.UINT8)
                mask_rgb = np.full((mask_image.shape[0], mask_image.shape[1], 3), (255, 0, 0), dtype=np.uint8)
                mask_alpha = (gen_segmentation[..., [j]].squeeze() * (255 / 2)).cpu().numpy().astype(np.uint8)
                composited_image = orig_image.pil.copy().convert("RGBA")
                composited_image.alpha_composite(Image.fromarray(np.dstack((mask_rgb, mask_alpha))))
                composited_image = composited_image.convert("RGB")
                images.append(Im.concat_vertical(prompt_image, mask_image, composited_image, spacing=5, fill=(128, 128, 128)))

        gen_results = Im.concat_horizontal(images, spacing=5, fill=(128, 128, 128))
        output_images = Im.concat_horizontal(gt_info, gen_results, spacing=10, fill=(255, 255, 255))
        all_output_images.append(output_images)

    device = batch["gen_pixel_values"].device
    output_images = gather(device, all_output_images)
    log_with_accelerator(
        accelerator=accelerator,
        images=output_images,
        save_folder=output_path,
        name="validation",
        global_step=(global_step if global_step is not None else i),
        spacing=25,
    )

    if cfg.model.break_a_scene_cross_attn_loss or cfg.inference.visualize_attention_map:
        output_attn_viz = gather(device, all_output_attn_viz, gather_different=True)
        log_with_accelerator(
            accelerator=accelerator,
            images=output_attn_viz,
            save_folder=output_path,
            name="attention",
            global_step=(global_step if global_step is not None else i),
        )

    log_info(f"Saved to {output_path}")


@torch.no_grad()
def run_inference_batch(
    batch: dict,
    model: BaseMapper,
    pipeline: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline],
    inference_cfg: InferenceConfig,
    num_images_per_prompt: int = 1,
    truncation_idx: Optional[int] = None,
    visualize_attention_map: bool = False,
) -> Image.Image:
    if visualize_attention_map:
        # cross_attn_init()
        pipeline.unet = register_cross_attention_hook(pipeline.unet)

    with torch.cuda.amp.autocast():
        negative_prompt_embeds = None
        if inference_cfg.use_custom_pipeline:
            if not inference_cfg.empty_string_cfg:
                pipeline_kwargs, _ = model.get_neti_conditioning_for_inference(
                    batch=batch,
                    num_images_per_prompt=num_images_per_prompt,
                    truncation_idx=truncation_idx,
                    disable_conditioning=True,
                    timesteps=pipeline.scheduler.timesteps,
                )
            pipeline_kwargs, input_prompt = model.get_neti_conditioning_for_inference(
                batch=batch,
                num_images_per_prompt=num_images_per_prompt,
                truncation_idx=truncation_idx,
                timesteps=pipeline.scheduler.timesteps,
            )
        else:
            assert inference_cfg.empty_string_cfg
            pipeline_kwargs, input_prompt = model.get_standard_conditioning_for_inference(batch=batch)

    if inference_cfg.use_custom_pipeline:
        images = sd_pipeline_call(
            pipeline=pipeline,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=inference_cfg.guidance_scale,
            batched_cfg=inference_cfg.batched_cfg,
            **pipeline_kwargs,
        ).images
    else:
        images = pipeline(
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=inference_cfg.guidance_scale,
            **pipeline_kwargs,
        ).images

    if visualize_attention_map:
        pipeline.unet = unregister_cross_attention_hook(pipeline.unet)

    return images, input_prompt, pipeline_kwargs["prompt_embeds"]


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
    else:
        model.set_inference_mode()

    pipeline = cls.from_pretrained(**kwargs)
    pipeline = pipeline.to(accelerator.device)

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

    # if cfg.model.lora_unet:
    #     pipeline.load_lora_weights(args.output_dir)

    if cfg.model.per_timestep_conditioning:
        pipeline.unet.set_attn_processor(XTIAttenProc())

    return pipeline


def log_with_accelerator(
    accelerator: Accelerator, images: List[Image.Image], global_step: int, name: str, save_folder: Optional[Path] = None, spacing: int = 15
):
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
            tracker.log({name: wandb.Image(Im.concat_horizontal(images, spacing=spacing, fill=(255, 255, 255)).pil)}, step=global_step)
