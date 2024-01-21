import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer

from gen.configs.base import BaseConfig
from gen.models.base_mapper_model import BaseMapper
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.utils.trainer_utils import unwrap
from inference import load_stable_diffusion_model, run_inference_dataloader


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
        global_step: int,
    ):
        """Runs inference during our training scheme."""
        pipeline = load_stable_diffusion_model(
            cfg=self.cfg,
            accelerator=accelerator,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            vae=vae,
            model=model,
            torch_dtype=self.weight_dtype,
        )

        run_inference_dataloader(
            accelerator=accelerator,
            model=model,
            pipeline=pipeline,
            dataloader=validation_dataloader,
            output_path=self.cfg.output_dir / "images",
            global_step=global_step,
            inference_cfg=self.cfg.inference,
        )

        del pipeline
        torch.cuda.empty_cache()
        unwrap(text_encoder).text_model.embeddings.mapper.train()
        if self.cfg.model.controlnet: unwrap(model).controlnet.train()
        if not self.cfg.model.freeze_text_encoder: unwrap(text_encoder).train()
        if not self.cfg.model.freeze_unet: unwrap(model).unet.train()
