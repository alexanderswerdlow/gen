import torch
from accelerate import Accelerator
from gen.configs.base import BaseConfig
from gen.models.cross_attn.base import BaseMapper
from gen.utils.diffusers_utils import load_stable_diffusion_model
from inference import run_inference_dataloader


class ValidationHandler:
    def __init__(self, cfg: BaseConfig, weights_dtype: torch.dtype):
        self.cfg = cfg
        self.weight_dtype = weights_dtype

    def infer(
        self,
        accelerator: Accelerator,
        validation_dataloader: torch.utils.data.DataLoader,
        model: BaseMapper,
        global_step: int,
    ):
        """Runs inference during our training scheme."""
        pipeline = load_stable_diffusion_model(
            cfg=self.cfg,
            accelerator=accelerator,
            tokenizer=model.tokenizer,
            text_encoder=model.text_encoder,
            unet=model.unet,
            vae=model.vae,
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
            cfg=self.cfg.inference,
        )

        del pipeline
        torch.cuda.empty_cache()
        model.set_training_mode()