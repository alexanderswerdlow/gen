
from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPTokenizer

from diffusers.models.attention import BasicTransformerBlock
from diffusers.training_utils import EMAModel, cast_training_params
from gen.models.base.attn_proc import register_custom_attention
from gen.models.base.base_defs import AttentionConfig, Dummy
from gen.models.base.pipeline_stable_diffusion import StableDiffusionPipeline
from gen.models.encoders.encoder import BaseModel
from gen.utils.decoupled_utils import get_modules
from gen.utils.diffusers_utils import load_stable_diffusion_model
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.trainer_utils import Trainable, TrainingState, unwrap
from gen.utils.visualization_utils import get_dino_pca, viz_feats

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig
    from gen.utils.trainer_utils import TrainingState
    from gen.models.base.base_model import BaseMapper


def initialize_custom_models(self: BaseMapper):
    if self.cfg.model.stock_dino_v2:
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.encoder = torch.compile(self.encoder)
    elif self.cfg.model.enable_encoder:
        self.encoder = hydra.utils.instantiate(self.cfg.model.encoder, compile=False)
    
    if self.cfg.model.stock_dino_v2 or self.cfg.model.enable_encoder:
        self.encoder.to(self.dtype)

    if self.cfg.model.enc_lora:
        self.encoder.requires_grad_(True)
        from peft import LoraConfig, inject_adapter_in_model
        module_regex = r".*blocks\.(20|21|22|23)\.mlp\.fc\d" if 'large' in self.cfg.model.encoder.model_name else r".*blocks\.(10|11)\.mlp\.fc\d"
        lora_config = LoraConfig(r=self.cfg.model.enc_lora_rank, lora_alpha=self.cfg.model.enc_lora_alpha, lora_dropout=self.cfg.model.enc_lora_dropout, target_modules=module_regex)
        self.encoder = inject_adapter_in_model(lora_config, self.encoder, adapter_name='lora')
        # self.encoder = torch.compile(self.encoder, mode="max-autotune-no-cudagraphs", fullgraph=True)

def initialize_diffusers_models(self: BaseModel) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
    # Load the tokenizer
    tokenizer_encoder_name = "runwayml/stable-diffusion-v1-5" if self.cfg.model.use_sd_15_tokenizer_encoder else self.cfg.model.pretrained_model_name_or_path
    revision = None if self.cfg.model.use_sd_15_tokenizer_encoder else self.cfg.model.revision
    self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_encoder_name, subfolder="tokenizer", revision=revision, use_fast=False
    )

    # Load scheduler and models
    self.noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")

    if self.cfg.model.vae:
        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant
        )

    unet_kwargs = dict()
    if self.cfg.model.unet:
        if self.cfg.model.autoencoder_slicing:
            self.vae.enable_slicing()

        if self.cfg.model.dual_attention:
            unet_kwargs["attention_config"] = AttentionConfig(dual_self_attention=True, dual_cross_attention=True)
            unet_kwargs["strict_load"] = False
        elif self.cfg.model.joint_attention:
            unet_kwargs["attention_config"] = AttentionConfig(joint_attention=True)
            unet_kwargs["strict_load"] = False

        if self.cfg.model.add_cross_attn_pos_emb is not None:
            unet_kwargs["attention_config"].add_cross_attn_pos_emb = self.cfg.model.add_cross_attn_pos_emb

        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant, **unet_kwargs
        )

        if self.cfg.model.duplicate_unet_input_channels and self.cfg.model.pretrained_model_name_or_path != "prs-eth/marigold-v1-0":
            new_dim = self.unet.conv_in.in_channels * 4 if self.cfg.model.separate_xyz_encoding else self.unet.conv_in.in_channels * 2
            conv_in_updated = torch.nn.Conv2d(new_dim, self.unet.conv_in.out_channels, kernel_size=self.unet.conv_in.kernel_size, padding=self.unet.conv_in.padding)
            conv_in_updated.requires_grad_(False)
            self.unet.conv_in.requires_grad_(False)
            if self.cfg.model.separate_xyz_encoding:
                conv_in_updated.weight[:,:4,:,:].copy_(self.unet.conv_in.weight / 4)
                conv_in_updated.weight[:,4:8,:,:].copy_(self.unet.conv_in.weight / 4)
                conv_in_updated.weight[:,8:12,:,:].copy_(self.unet.conv_in.weight / 4)
                conv_in_updated.weight[:,12:16,:,:].copy_(self.unet.conv_in.weight / 4)
            else:
                conv_in_updated.weight[:,:4,:,:].copy_(self.unet.conv_in.weight / 2)
                conv_in_updated.weight[:,4:8,:,:].copy_(self.unet.conv_in.weight / 2)
            conv_in_updated.bias.copy_(self.unet.conv_in.bias)
            self.unet.conv_in = conv_in_updated
            self.unet.conv_in.requires_grad_(True)

            if self.cfg.model.separate_xyz_encoding:
                conv_out_updated = torch.nn.Conv2d(self.unet.conv_out.in_channels, self.unet.conv_out.out_channels * 3, kernel_size=self.unet.conv_out.kernel_size, padding=self.unet.conv_out.padding)
                conv_out_updated.requires_grad_(False)
                self.unet.conv_out.requires_grad_(False)
                conv_out_updated.weight[:4, ...].copy_(self.unet.conv_out.weight / 3)
                conv_out_updated.weight[4:8, ...].copy_(self.unet.conv_out.weight / 3)
                conv_out_updated.weight[8:12, ...].copy_(self.unet.conv_out.weight / 3)
                conv_out_updated.bias[:4].copy_(self.unet.conv_out.bias / 3)
                conv_out_updated.bias[4:8].copy_(self.unet.conv_out.bias / 3)
                conv_out_updated.bias[8:12].copy_(self.unet.conv_out.bias / 3)
                self.unet.conv_out = conv_out_updated
                self.unet.conv_out.requires_grad_(True)

        if self.cfg.model.dual_attention:
            from accelerate.utils import set_module_tensor_to_device
            param_names = ["to_k", "to_q", "to_v", "to_out.0"]
            for k, v in self.unet.named_parameters():
                for name in param_names:
                    if k.rsplit('.', 1)[0].endswith(name):
                        if 'v2_' in k:
                            orig_weight_name = k.replace("v2_", "")
                            set_module_tensor_to_device(self.unet, k, self.device, value=self.unet.state_dict()[orig_weight_name])
                            log_info(f"Setting {k} to {orig_weight_name}")

                if 'v_1_to_v_2' in k or 'v_2_to_v_1' in k:
                    param = 0.02 * torch.randn(v.shape, device=self.device) if 'weight' in k else torch.zeros(v.shape, device=self.device)
                    set_module_tensor_to_device(self.unet, k, self.device, value=param)
                    log_info(f"Initializing {k}")

        if self.cfg.model.joint_attention:
            from accelerate.utils import set_module_tensor_to_device
            for k, v in self.unet.named_parameters():
                if 'to_cross' in k:
                    param = 0.02 * torch.randn(v.shape, device=self.device) if 'weight' in k else torch.zeros(v.shape, device=self.device)
                    set_module_tensor_to_device(self.unet, k, self.device, value=param)
                    log_info(f"Initializing {k}")

                if 'cross_attn_pos_emb' in k:
                    param = 0.02 * torch.randn(v.shape, device=self.device)
                    set_module_tensor_to_device(self.unet, k, self.device, value=param)
                    log_info(f"Initializing {k}")

        if self.cfg.model.ema and not self.cfg.model.freeze_unet:
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config)
            log_warn("Using EMA for U-Net. Inference has not het been handled properly.")
    else:
        self.unet = Dummy() # For rotation denoising only

def add_unet_adapters(self: BaseModel):
    assert not (self.cfg.model.freeze_unet is False and self.cfg.model.unet_lora)
    if self.cfg.model.unet_lora:
        from peft import LoraConfig
        unet_lora_config = LoraConfig(
            r=self.cfg.model.unet_lora_rank,
            lora_alpha=self.cfg.model.unet_lora_rank,
            use_dora=self.cfg.model.use_dora,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config)
        cast_training_params(self.unet, dtype=torch.float32)

    if self.cfg.trainer.enable_xformers_memory_efficient_attention:
        self.unet.enable_xformers_memory_efficient_attention()
        if self.cfg.model.controlnet:
            self.controlnet.enable_xformers_memory_efficient_attention()

    if self.cfg.model.dual_attention or self.cfg.model.joint_attention:
        register_custom_attention(self.unet)

    if self.cfg.trainer.gradient_checkpointing:
        if self.cfg.model.enable_encoder:
            if hasattr(self.encoder, "base_model"):
                self.encoder.base_model.set_grad_checkpointing()
                log_info("Setting CLIP Gradient checkpointing")
            elif hasattr(self.encoder, "model"):
                self.encoder.model.set_grad_checkpointing()
                log_info("Setting CLIP Gradient checkpointing")

        self.unet.enable_gradient_checkpointing()

@staticmethod
def set_training_mode(cfg, _other, device, dtype, set_grad: bool = False):
    """
    Set training mode for the proper models and freeze/unfreeze them.

    We set the weights to weight_dtype only if they are frozen. Otherwise, they are left in FP32.

    We have the set_grad param as changing requires_grad after can cause issues. 
    
    For example, we want to first freeze the U-Net then add the LoRA adapter. We also see this error:
    `element 0 of tensors does not require grad and does not have a grad_fn`
    """
    md = cfg.model
    _dtype = dtype
    _device = device
    other = unwrap(_other)

    if hasattr(other, "controller"):
        other.controller.reset()

    if hasattr(other, "pipeline"):  # After validation, we need to clear this
        del other.pipeline
        torch.cuda.empty_cache()
        gc.collect()
        log_debug("Cleared pipeline", main_process_only=False)
    
    if hasattr(other, "vae"):
        if set_grad:
            other.vae.to(device=_device, dtype=torch.float32 if (md.force_fp32_pcd_vae or md.unfreeze_vae_decoder) else _dtype)
            other.vae.requires_grad_(False)
            if md.unfreeze_vae_decoder:
                other.vae.decoder.requires_grad_(True)

        if md.unfreeze_vae_decoder:
            other.vae.eval()
            other.vae.decoder.train()
        else:
            other.vae.eval()

    if md.enable_encoder:
        if md.freeze_enc:
            if set_grad:
                other.encoder.to(device=_device, dtype=_dtype)
                other.encoder.requires_grad_(False)
            other.encoder.eval()
            log_warn("CLIP is frozen for debugging")
        else:
            if set_grad:
                other.encoder.requires_grad_(True)
            other.encoder.train()
            log_warn("CLIP is unfrozen")

    if md.enc_lora:
        if set_grad:
            for k, p in other.clip.named_parameters():
                if "lora" in k:
                    log_warn(f"Unfreezing {k}, converting to {torch.float32}")
                    p.requires_grad = True

    if md.unfreeze_last_n_enc_layers is not None and md.enc_lora is False:
        log_warn(f"Unfreezing last {md.unfreeze_last_n_enc_layers} CLIP layers")
        if hasattr(other.clip, "base_model"):
            model_ = other.clip.base_model
        elif hasattr(other.clip, "model"):
            model_ = other.clip.model

        for block in model_.blocks[-md.unfreeze_last_n_enc_layers :]:
            if set_grad:
                block.requires_grad_(True)
            block.train()

    if md.unet and md.freeze_unet:
        if set_grad:
            other.unet.to(device=_device, dtype=_dtype)
            other.unet.requires_grad_(False)
        other.unet.eval()

        if md.unfreeze_single_unet_layer:
            for m in get_modules(other.unet, BasicTransformerBlock)[:1]:
                if set_grad:
                    m.requires_grad_(True)
                    m.to(dtype=torch.float32)
                m.train()

        if md.duplicate_unet_input_channels and md.freeze_self_attn is False:
            modules_to_unfreeze = [other.unet.conv_in, other.unet.conv_out, other.unet.conv_norm_out]
            for m in modules_to_unfreeze:
                if set_grad:
                    m.requires_grad_(True)
                    m.to(dtype=torch.float32)
                m.train()

        if md.dual_attention or md.joint_attention:
            for m in get_modules(other.unet, BasicTransformerBlock):
                if set_grad:
                    m.requires_grad_(True)
                    m.to(dtype=torch.float32)

                    if md.freeze_self_attn:
                        m.attn1.requires_grad_(False)
                        m.attn1.to(dtype=_dtype)

                m.train()
                if md.freeze_self_attn:
                    m.attn1.eval()
    else:
        if set_grad:
            other.unet.requires_grad_(True)
            if cfg.model.ema:
                other.ema_unet.requires_grad_(True)

        other.unet.train()
        if cfg.model.ema:
            other.ema_unet.train()

def set_inference_mode(self: BaseModel, init_pipeline: bool = True):
    if init_pipeline and self.cfg.model.unet and getattr(self, "pipeline", None) is None:
        self.pipeline: Union[StableDiffusionControlNetPipeline, StableDiffusionPipeline] = load_stable_diffusion_model(
            cfg=self.cfg,
            device=self.device,
            tokenizer=self.tokenizer,
            text_encoder=None,
            unet=self.unet,
            vae=self.vae,
            model=self,
            torch_dtype=self.dtype,
        )


def get_custom_params(self: BaseMapper):
    """
    Returns all params directly managed by the top-level model.
    Other params may be nested [e.g., in diffusers]
    """
    params = {}
    
    if self.cfg.model.enable_encoder:
        params.update({k:p for k,p in self.encoder.named_parameters() if p.requires_grad})

    return params


def get_unet_params(self: BaseMapper):
    is_unet_trainable = self.cfg.model.unet and (not self.cfg.model.freeze_unet or self.cfg.model.unet_lora)
    return {k:v for k,v in self.unet.named_parameters() if v.requires_grad} if is_unet_trainable else dict()


def get_param_groups(self: BaseMapper):
    if self.cfg.model.finetune_unet_with_different_lrs:
        unet_params = self.get_unet_params()
        custom_params = self.get_custom_params()

        def get_params(params, keys):
            group = {k: v for k, v in params.items() if all([key in k for key in keys])}
            for k, p in group.items():
                del params[k]
            return group

        if self.cfg.model.lr_finetune_version == 0:
            return [  # Order matters here
                {"params": get_params(custom_params, ("token_predictor",)).values(), "lr": self.cfg.trainer.learning_rate},
                {"params": get_params(unet_params, ("attn2",)).values(), "lr": self.cfg.trainer.learning_rate / 10},
                {"params": custom_params.values(), "lr": self.cfg.trainer.learning_rate / 10},
                {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate / 100},
            ]
    else:
        return None


def checkpoint(self, accelerator: Accelerator, state: TrainingState, path: Path):
    # TODO: save_state/load_state saves everything from prepare() regardless of whether it's frozen
    # This is very inefficient but simpler for now.
    accelerator.save_state(path / "state", safe_serialization=False)
    if hasattr(self, 'mapper'):
        accelerator.save_model(self.mapper, save_directory=path / "model", safe_serialization=False)
    if self.cfg.model.unet_lora:
        from peft.utils import get_peft_model_state_dict

        unet_lora_state_dict = get_peft_model_state_dict(self.unet)
        cls = StableDiffusionControlNetPipeline if self.cfg.model.controlnet else StableDiffusionPipeline
        cls.save_lora_weights(
            save_directory=path / "pipeline",
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    extra_pkl = {"cfg": OmegaConf.to_container(self.cfg, resolve=True)}

    torch.save(extra_pkl, path / "data.pkl")
    log_info(f"Saved state to {path}")