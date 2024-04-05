from ast import Pass
import dataclasses
import gc
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from einx import add, rearrange, where
from jaxtyping import Bool, Float, Integer
from omegaconf import OmegaConf
from peft import LoraConfig
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel

from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers.utils.torch_utils import randn_tensor
from gen.configs import BaseConfig
from gen.models.cross_attn.break_a_scene import register_attention_control
from gen.models.cross_attn.deprecated_configs import (attention_masking, forward_shift_scale, handle_attention_masking_dropout, init_shift_scale, shift_scale_uncond_hidden_states)
from gen.models.cross_attn.eschernet import get_relative_pose
from gen.models.cross_attn.losses import (break_a_scene_cross_attn_loss, break_a_scene_masked_loss, evenly_weighted_mask_loss, get_gt_rot, src_tgt_token_consistency_loss,
                                          token_cls_loss, token_rot_loss)
from gen.models.cross_attn.modules import FeatureMapper, TokenMapper
from gen.models.encoders.encoder import BaseModel
from gen.models.utils import find_true_indices_batched, positionalencoding2d
from gen.utils.data_defs import InputData, get_dropout_grid, get_one_hot_channels, get_tgt_grid, undo_normalization_given_transforms
from gen.utils.decoupled_utils import get_modules, to_numpy
from gen.utils.diffusers_utils import load_stable_diffusion_model
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.tokenization_utils import _get_tokens, get_uncond_tokens
from gen.utils.trainer_utils import Trainable, TrainingState, unwrap
from gen.utils.visualization_utils import get_dino_pca, viz_feats

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

@dataclass
class ConditioningData:
    placeholder_token: Optional[int] = None
    attn_dict: Optional[dict[str, Tensor]] = None
    clip_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None
    mask_head_tokens: Optional[Float[Tensor, "n d"]] = None # We sometimes need this if we want to have some mask tokens with detached gradients
    mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    mask_dropout: Optional[Bool[Tensor, "n"]] = None
    batch_cond_dropout: Optional[Bool[Tensor, "b"]] = None
    input_prompt: Optional[list[str]] = None
    learnable_idxs: Optional[Any] = None
    batch_attn_masks: Optional[Float[Tensor, "b hw hw"]] = None
    
    src_mask_tokens: Optional[Float[Tensor, "n d"]] = None # We duplicate for loss calculation
    tgt_mask_tokens: Optional[Float[Tensor, "n d"]] = None
    tgt_mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    tgt_mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    tgt_mask_dropout: Optional[Bool[Tensor, "n"]] = None

    src_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    encoder_input_pixel_values: Optional[Float[Tensor, "b c h w"]] = None

    # These are passed to the U-Net or pipeline
    encoder_hidden_states: Optional[Float[Tensor, "b d"]] = None
    unet_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)

@dataclass
class TokenPredData:
    gt_rot_6d: Optional[Float[Tensor, "n 6"]] = None
    noised_rot_6d: Optional[Float[Tensor, "n 6"]] = None
    rot_6d_noise: Optional[Float[Tensor, "n 6"]] = None
    timesteps: Optional[Integer[Tensor, "b"]] = None
    cls_pred: Optional[Float[Tensor, "n classes"]] = None
    pred_6d_rot: Optional[Float[Tensor, "n 6"]] = None
    token_output_mask: Optional[Bool[Tensor, "n"]] = None
    relative_rot_token_mask: Optional[Bool[Tensor, "n"]] = None
    denoise_history_6d_rot: Optional[Float[Tensor, "t n 6"]] = None
    denoise_history_timesteps: Optional[list[int]] = None
    raw_pred_rot_logits: Optional[Float[Tensor, "n ..."]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None

@dataclass
class AttentionMetadata:
    layer_idx: Optional[int] = None
    num_layers: Optional[int] = None
    num_cond_vectors: Optional[int] = None
    add_pos_emb: Optional[bool] = None
    cross_attention_mask: Optional[Float[Tensor, "b d h w"]] = None
    self_attention_mask: Optional[Float[Tensor, "b hw hw"]] = None
    gate_scale: Optional[float] = None
    frozen_dim: Optional[int] = None
    return_attn_probs: Optional[bool] = None
    attn_probs: Optional[Float[Tensor, "..."]] = None
    custom_map: Optional[dict] = None
    posemb: Optional[tuple] = None

class Dummy:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method
    
class BaseMapper(Trainable):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg: BaseConfig = cfg

        # dtype of most intermediate tensors and frozen weights. Notably, we always use FP32 for trainable params.
        self.dtype = getattr(torch, cfg.trainer.dtype.split(".")[-1])

        self.initialize_diffusers_models()
        self.initialize_custom_models()
        BaseMapper.set_training_mode(cfg=self.cfg, _other=self, dtype=self.dtype, device=self.device, set_grad=True) # This must be called before we set LoRA

        if self.cfg.model.unet: self.add_unet_adapters()

        if self.cfg.trainer.compile:
            log_info("Using torch.compile()...")
            self.mapper = torch.compile(self.mapper, mode="reduce-overhead", fullgraph=True)

            if hasattr(self, "clip"):
                self.clip = torch.compile(self.clip, mode="reduce-overhead", fullgraph=True)
            
            if hasattr(self, "token_mapper"): 
                self.token_mapper = torch.compile(self.token_mapper, mode="reduce-overhead")

            # TODO: Compile currently doesn't work with flash-attn apparently
            # self.unet.to(memory_format=torch.channels_last)
            # self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)
            # if self.cfg.model.controlnet:
            #     self.controlnet = self.controlnet.to(memory_format=torch.channels_last)
            #     self.controlnet: ControlNetModel = torch.compile(self.controlnet, mode="reduce-overhead", fullgraph=True)

        from gen.models.cross_attn.base_inference import infer_batch

        BaseMapper.infer_batch = infer_batch

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_custom_models(self):
        if self.cfg.model.mask_token_conditioning:
            self.mapper = FeatureMapper(cfg=self.cfg).to(self.cfg.trainer.device)
            self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False) #, compile=not self.cfg.model.clip_lora)
            self.clip.to(self.dtype)

            if self.cfg.model.clip_lora:
                self.clip.requires_grad_(True)
                from peft import inject_adapter_in_model, LoraConfig
                module_regex = r".*blocks\.(20|21|22|23)\.mlp\.fc\d" if 'large' in self.cfg.model.encoder.model_name else r".*blocks\.(10|11)\.mlp\.fc\d"
                lora_config = LoraConfig(r=self.cfg.model.clip_lora_rank, lora_alpha=self.cfg.model.clip_lora_alpha, lora_dropout=self.cfg.model.clip_lora_dropout, target_modules=module_regex)
                self.clip = inject_adapter_in_model(lora_config, self.clip, adapter_name='lora')
                # self.clip = torch.compile(self.clip, mode="max-autotune-no-cudagraphs", fullgraph=True)

        if self.cfg.model.use_dataset_segmentation is False:
            from gen.models.encoders.sam import HQSam
            self.hqsam = HQSam(model_type="vit_b")

        if self.cfg.model.clip_shift_scale_conditioning:
            init_shift_scale(self)

        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            self.token_mapper = TokenMapper(cfg=self.cfg).to(self.cfg.trainer.device)

    def initialize_diffusers_models(self) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        tokenizer_encoder_name = "runwayml/stable-diffusion-v1-5" if self.cfg.model.use_sd_15_tokenizer_encoder else self.cfg.model.pretrained_model_name_or_path
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_encoder_name, subfolder="tokenizer", revision=self.cfg.model.revision, use_fast=False
        )

        # Load scheduler and models
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.rotation_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.model.rotation_diffusion_timestep,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=self.cfg.model.rotation_diffusion_parameterization,
        )

        if self.cfg.model.add_text_tokens:
            self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
                tokenizer_encoder_name, subfolder="text_encoder", revision=self.cfg.model.revision
            )
        

        unet_kwargs = dict()
        if self.cfg.model.gated_cross_attn:
            unet_kwargs["attention_type"] = "gated-cross-attn"
            unet_kwargs["low_cpu_mem_usage"] = False

        if self.cfg.model.unet:
            self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant
            )
            if self.cfg.model.autoencoder_slicing:
                self.vae.enable_slicing()
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant, **unet_kwargs
            )

            if self.cfg.model.add_grid_to_input_channels:
                new_dim = self.unet.conv_in.in_channels + 2
                conv_in_updated = torch.nn.Conv2d(
                    new_dim, self.unet.conv_in.out_channels, kernel_size=self.unet.conv_in.kernel_size, padding=self.unet.conv_in.padding
                )
                conv_in_updated.requires_grad_(False)
                self.unet.conv_in.requires_grad_(False)
                torch.nn.init.zeros_(conv_in_updated.weight)
                conv_in_updated.weight[:,:4,:,:].copy_(self.unet.conv_in.weight)
                conv_in_updated.bias.copy_(self.unet.conv_in.bias)
                self.unet.conv_in = conv_in_updated

            if self.cfg.model.ema and not self.cfg.model.freeze_unet:
                self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config)
                log_warn("Using EMA for U-Net. Inference has not het been handled properly.")
        else:
            self.unet = Dummy() # For rotation denoising only

        if self.cfg.model.controlnet:
            self.controlnet: ControlNetModel = ControlNetModel.from_unet(self.unet, conditioning_channels=2)

    def add_unet_adapters(self):
        assert not (self.cfg.model.freeze_unet is False and (self.cfg.model.unfreeze_unet_after_n_steps is not None or self.cfg.model.unet_lora))
        if self.cfg.model.unet_lora:
            unet_lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=self.cfg.model.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)
            cast_training_params(self.unet, dtype=torch.float32)

        if self.cfg.trainer.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
            if self.cfg.model.controlnet:
                self.controlnet.enable_xformers_memory_efficient_attention()

        if self.cfg.trainer.gradient_checkpointing:
            if hasattr(self.clip, "base_model"):
                self.clip.base_model.set_grad_checkpointing()
                log_info("Setting CLIP Gradient checkpointing")
            elif hasattr(self.clip, "model"):
                self.clip.model.set_grad_checkpointing()
                log_info("Setting CLIP Gradient checkpointing")

            self.unet.enable_gradient_checkpointing()
            if self.cfg.model.controlnet:
                self.controlnet.enable_gradient_checkpointing()
            if self.cfg.model.freeze_text_encoder is False and self.cfg.model.add_text_tokens:
                self.text_encoder.gradient_checkpointing_enable()
            
        if self.cfg.model.break_a_scene_cross_attn_loss:
            from gen.models.cross_attn.break_a_scene import AttentionStore

            self.controller = AttentionStore()
            register_attention_control(self.controller, self.unet)

        elif self.cfg.model.layer_specialization or self.cfg.model.eschernet:
            from gen.models.cross_attn.attn_proc import register_layerwise_attention

            num_cross_attn_layers = register_layerwise_attention(self.unet)
            if self.cfg.model.layer_specialization and (not self.cfg.model.custom_conditioning_map):
                assert num_cross_attn_layers == (2 * self.cfg.model.num_conditioning_pairs * (2 if self.cfg.model.gated_cross_attn else 1))
   
        if self.cfg.model.per_layer_queries:
            assert self.cfg.model.layer_specialization

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

        if set_grad is False and cfg.trainer.inference_train_switch is False:
            return

        if set_grad and md.unet:
            other.vae.to(device=_device, dtype=_dtype)
            other.vae.requires_grad_(False)

        if md.use_dataset_segmentation is False:
            other.hqsam.eval()
            if set_grad:
                other.hqsam.requires_grad_(False)

        if md.mask_token_conditioning:
            if md.freeze_clip:
                if set_grad:
                    other.clip.to(device=_device, dtype=_dtype)
                    other.clip.requires_grad_(False)
                other.clip.eval()
                log_warn("CLIP is frozen for debugging")
            else:
                if set_grad:
                    other.clip.requires_grad_(True)
                other.clip.train()
                log_warn("CLIP is unfrozen")

            if md.clip_lora:
                if set_grad:
                    for k, p in other.clip.named_parameters():
                        if "lora" in k:
                            log_warn(f"Unfreezing {k}, converting to {torch.float32}")
                            p.requires_grad = True

        if md.unfreeze_last_n_clip_layers is not None and md.clip_lora is False:
            log_warn(f"Unfreezing last {md.unfreeze_last_n_clip_layers} CLIP layers")
            if hasattr(other.clip, "base_model"):
                model_ = other.clip.base_model
            elif hasattr(other.clip, "model"):
                model_ = other.clip.model

            for block in model_.blocks[-md.unfreeze_last_n_clip_layers :]:
                if set_grad:
                    block.requires_grad_(True)
                block.train()

            # if set_grad:
            #     model_.norm.requires_grad_(True)
            # model_.norm.train()

        if hasattr(other, "clip") and set_grad:
            cast_training_params(other.clip, dtype=torch.float32)
            print_trainable_parameters(other.clip)

        if md.unfreeze_resnet:
            for module in list(other.clip.base_model.children())[:6]:
                if set_grad:
                    module.requires_grad_(True)
                module.train()
        if md.freeze_unet:
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
        else:
            if set_grad:
                other.unet.requires_grad_(True)
                if cfg.model.ema:
                    other.ema_unet.requires_grad_(True)
            other.unet.train()
            if cfg.model.ema:
                other.ema_unet.train()

        if md.unfreeze_gated_cross_attn:
            for m in get_modules(other.unet, BasicTransformerBlock):
                assert hasattr(m, "fuser")
                assert hasattr(m, "attn2")
                if set_grad: # TODO: Initialize weights somewhere else
                    # We copy the cross-attn weights from the frozen module to this
                    m.fuser.attn.load_state_dict(m.attn2.state_dict())
                    m.fuser.requires_grad_(True)
                    m.fuser.to(dtype=torch.float32)
                m.fuser.train()

        if md.add_text_tokens:
            if md.freeze_text_encoder:
                if set_grad:
                    other.text_encoder.to(device=_device, dtype=_dtype)
                    other.text_encoder.requires_grad_(False)
                other.text_encoder.eval()
            else:
                if set_grad:
                    other.text_encoder.requires_grad_(True)
                if set_grad and md.freeze_text_encoder_except_token_embeddings:
                    other.text_encoder.text_model.encoder.requires_grad_(False)
                    other.text_encoder.text_model.final_layer_norm.requires_grad_(False)
                    other.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
                other.text_encoder.train()

        if md.mask_token_conditioning:
            if md.freeze_mapper:
                if set_grad:
                    other.mapper.requires_grad_(False)
                other.mapper.eval()
            else:
                if set_grad:
                    other.mapper.requires_grad_(True)
                other.mapper.train()

        if md.controlnet:
            if set_grad:
                other.controlnet.requires_grad_(True)
            other.controlnet.train()

        if md.clip_shift_scale_conditioning:
            if set_grad:
                other.clip_proj_layers.requires_grad_(True)
            other.clip_proj_layers.train()

        if md.token_cls_pred_loss or md.token_rot_pred_loss:
            if set_grad:
                other.token_mapper.requires_grad_(True)
            other.token_mapper.train()

        if md.freeze_token_encoder:
            if set_grad:
                other.mapper.to(device=_device, dtype=_dtype)
                other.mapper.requires_grad_(False)
            other.mapper.eval()

            if set_grad:
                other.mapper.token_modulator.requires_grad_(True)
                other.mapper.token_modulator.to(device=_device, dtype=torch.float32)
            other.mapper.token_modulator.train()

    def unfreeze_unet(self):
        self.cfg.model.freeze_unet = False
        self.unet.to(device=self.device, dtype=torch.float32)
        self.unet.requires_grad_(True)
        self.unet.train()
        if self.cfg.model.break_a_scene_cross_attn_loss_second_stage:
            self.cfg.model.break_a_scene_masked_loss = True

    def set_inference_mode(self, init_pipeline: bool = True):
        if init_pipeline and self.cfg.model.unet and getattr(self, "pipeline", None) is None:
            self.pipeline: Union[StableDiffusionControlNetPipeline, StableDiffusionPipeline] = load_stable_diffusion_model(
                cfg=self.cfg,
                device=self.device,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder if self.cfg.model.add_text_tokens else None,
                unet=self.unet,
                vae=self.vae,
                model=self,
                torch_dtype=self.dtype,
            )
        else:
            # For rotation denoising only
            self.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )

        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    def get_custom_params(self):
        """
        Returns all params directly managed by the top-level model.
        Other params may be nested [e.g., in diffusers]
        """
        params = []
        if self.cfg.model.clip_shift_scale_conditioning:
            params.extend(self.clip_proj_layers.parameters())
        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            params.extend(self.token_mapper.parameters())
                        
        params.extend(self.mapper.parameters())

        # Add clip parameters
        params.extend([p for p in self.clip.parameters() if p.requires_grad])

        return params

    def get_unet_params(self):
        is_unet_trainable = self.cfg.model.unet and (not self.cfg.model.freeze_unet or self.cfg.model.unet_lora)
        return {k:v for k,v in dict(self.unet.named_parameters()) if v.requires_grad} if is_unet_trainable else dict()
    
    def get_param_groups(self):
        if self.cfg.model.finetune_unet_with_different_lrs:
            unet_params = self.get_unet_params()

            def get_params(params, keys):
                group = {k: v for k, v in params.items() if all([key in k for key in keys])}
                for k, p in group.items():
                    del params[k]
                return group

            if self.cfg.model.lr_finetune_version == 0:
                return [  # Order matters here
                    {"params": self.get_custom_params(), "lr": self.cfg.trainer.learning_rate * 2},
                    {"params": get_params(unet_params, ("attn2",)).values(), "lr": self.cfg.trainer.learning_rate},
                    {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate / 10},
                ]
            elif self.cfg.model.lr_finetune_version == 1:
                return [  # Order matters here
                    {"params": self.get_custom_params(), "lr": self.cfg.trainer.learning_rate},
                    {"params": get_params(unet_params, ("attn2",)).values(), "lr": self.cfg.trainer.learning_rate / 4},
                    {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate / 8},
                ]
            elif self.cfg.model.lr_finetune_version == 2:
                return [  # Order matters here
                    {"params": self.get_custom_params(), "lr": self.cfg.trainer.learning_rate * 2},
                    {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate},
                ]
        elif self.cfg.model.unfreeze_gated_cross_attn:
            unet_params = self.get_unet_params()

            def get_params(params, keys):
                group = {k: v for k, v in params.items() if all([key in k for key in keys]) and v.requires_grad}
                for k, p in group.items():
                    del params[k]
                return group

            return [  # Order matters here
                {"params": self.get_custom_params(), "lr": 2 * self.cfg.trainer.learning_rate},
                {"params": get_params(unet_params, ("fuser",)).values(), "lr": self.cfg.trainer.learning_rate},
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

    def on_sync_gradients(self, state: TrainingState):
        if self.cfg.model.unet and not self.cfg.model.freeze_unet and self.cfg.model.ema:
            self.ema_unet.step(self.unet.parameters())

    def process_input(self, batch: dict, state: TrainingState) -> InputData:
        if not isinstance(batch, InputData):
            batch: InputData = InputData.from_dict(batch)

        batch.state = state
        batch.dtype = self.dtype
        return batch

    def get_standard_conditioning_for_inference(
        self,
        batch: InputData,
        disable_conditioning: bool = False,
        cond: Optional[ConditioningData] = None,
    ):
        # Validate input
        bs: int = batch.src_pixel_values.shape[0]
        for field in dataclasses.fields(batch):
            if isinstance(getattr(batch, field.name), torch.Tensor):
                assert getattr(batch, field.name).shape[0] == bs

        assert batch.formatted_input_ids is None

        cond = self.get_hidden_state(batch, add_conditioning=disable_conditioning is False, cond=cond)

        if self.cfg.model.unet is False or self.cfg.model.return_mean_pooled_mask_tokens: return cond

        bs = batch.src_pixel_values.shape[0]
        cond.input_prompt = [[x for x in self.tokenizer.convert_ids_to_tokens(batch.formatted_input_ids[y]) if "<|" not in x] for y in range(bs)]

        # passed to StableDiffusionPipeline/StableDiffusionControlNetPipeline
        cond.unet_kwargs["prompt_embeds"] = cond.encoder_hidden_states
        if self.cfg.model.controlnet:
            cond.unet_kwargs["image"] = self.get_controlnet_conditioning(batch)

        return cond

    def denoise_rotation(self, batch: InputData, cond: ConditioningData, scheduler: DDPMScheduler):
        pred_data = TokenPredData()
        pred_data = get_gt_rot(self.cfg, cond, batch, pred_data)
        
        scheduler.set_timesteps(num_inference_steps=self.cfg.model.rotation_diffusion_timestep, device=self.device)
        timesteps = scheduler.timesteps
       
        latents = randn_tensor(pred_data.gt_rot_6d.shape, device=self.device, dtype=self.dtype)
        if self.cfg.model.rotation_diffusion_start_timestep is not None:
            latents = scheduler.add_noise(pred_data.gt_rot_6d, latents, timesteps[None, -self.cfg.model.rotation_diffusion_start_timestep].repeat(latents.shape[0]))
        
        sl = slice(-self.cfg.model.rotation_diffusion_start_timestep, None) if self.cfg.model.rotation_diffusion_start_timestep else slice(None)
        latent_history = []
        pred_data.denoise_history_timesteps = []
        for t in timesteps[sl]:
            pred_data.noised_rot_6d = latents
            pred_data.timesteps = t[None].repeat(latents.shape[0])

            # predict the noise residual
            pred_data = self.token_mapper(batch=batch, cond=cond, pred_data=pred_data)

            if self.cfg.model.discretize_rot_pred:                
                latents = pred_data.pred_6d_rot
            else:
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(pred_data.pred_6d_rot, t, latents).prev_sample

            latent_history.append(latents)
            pred_data.denoise_history_timesteps.append(t)

            if self.cfg.model.discretize_rot_pred:
                break

        pred_data.pred_6d_rot = latents
        pred_data.denoise_history_6d_rot = torch.stack(latent_history, dim=1)
        pred_data.rot_6d_noise = pred_data.gt_rot_6d # During inference we just set our epsilon loss [if enabled] to to be the final sample loss
        token_rot_loss_data = token_rot_loss(self.cfg, pred_data, is_training=self.training)
        token_rot_loss_data = {k:v for k, v in token_rot_loss_data.items() if isinstance(v, torch.Tensor)}

        return pred_data, token_rot_loss_data

    def check_add_segmentation(self, batch: InputData):
        """
        This function checks if we have segmentation for the current batch. If we do not, we add dummy segmentation or use HQSam to get segmentation.
        """

        if self.cfg.model.use_dummy_mask:
            # Very hacky. We ignore masks with <= 32 pixels later on.
            object_is_visible = (rearrange('b h w c -> b c (h w)', batch.src_segmentation) > 0).sum(dim=-1) > 0
            rearrange('b h w c -> b c (h w)', batch.src_segmentation)[object_is_visible] = 1
            return batch
        elif self.cfg.model.use_dataset_segmentation:
            return batch  # We already have segmentation

        # SAM requires NumPy [0, 255]
        sam_input = rearrange("b c h w -> b h w c", (((batch.tgt_pixel_values + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy())
        sam_seg = []
        bs: int = batch.src_pixel_values.shape[0]
        max_masks = 4

        for i in range(bs):
            masks = self.hqsam.forward(sam_input[i])
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            masks = masks[:max_masks]  # We only have 77 tokens
            original = torch.from_numpy(np.array([masks[i]["segmentation"] for i in range(len(masks))]))

            if original.shape[0] == 0:  # Make dummy mask with all ones
                original = (
                    batch.src_segmentation[i].new_ones((1, batch.src_segmentation[i].shape[0], batch.src_segmentation[i].shape[1])).cpu()
                )
            else:  # Add additional mask to capture any pixels that are not part of any mask
                original = torch.cat(((~original.any(dim=0))[None], original), dim=0)

            if original.shape[0] != 0 and not self.training:  # During inference, we update batch with the SAM masks for later visualization
                seg_ = original.permute(1, 2, 0).long().clone()
                sam_seg.append(torch.nn.functional.pad(seg_, (0, (max_masks + 1) - seg_.shape[-1]), "constant", 0))

        if len(sam_seg) > 0:
            batch.src_segmentation = torch.stack(sam_seg, dim=0)

        return batch

    @cached_property
    def placeholder_token_id(self):
        placeholder_token_id = self.tokenizer(self.cfg.model.placeholder_token, add_special_tokens=False).input_ids
        assert len(placeholder_token_id) == 1 and placeholder_token_id[0] != self.tokenizer.eos_token_id
        return placeholder_token_id[0]

    @cached_property
    def mask_tokens_ids(self):
        return self.tokenizer(f"{self.cfg.model.placeholder_token} and", add_special_tokens=False).input_ids

    @cached_property
    def eos_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

    @cached_property
    def pad_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    @cached_property
    def uncond_hidden_states(self):
        if self.cfg.model.add_text_tokens:
            uncond_input_ids = get_uncond_tokens(self.tokenizer).to(self.device)
            uncond_encoder_hidden_states = self.text_encoder(input_ids=uncond_input_ids[None]).last_hidden_state.to(dtype=self.dtype).squeeze(0)
        else:
            uncond_encoder_hidden_states = torch.zeros((self.cfg.model.num_decoder_cross_attn_tokens, self.cfg.model.token_embedding_dim), device=self.device, dtype=self.dtype)
        return uncond_encoder_hidden_states

    def get_feature_map(self, batch: InputData, cond: ConditioningData):
        bs: int = batch.src_pixel_values.shape[0]
        device = batch.tgt_pixel_values.device
        dtype = self.dtype

        clip_input = batch.src_pixel_values.to(device=device, dtype=dtype)

        if self.cfg.model.add_grid_to_input_channels:
            clip_input = torch.cat((clip_input, batch.src_grid), dim=1)

        if self.cfg.model.mask_dropped_tokens:
            for b in range(bs):
                dropped_mask = ~torch.isin(batch.src_segmentation[b], cond.mask_instance_idx[cond.mask_batch_idx == b]).any(dim=-1)
                clip_input[b, :, dropped_mask] = 0

        if batch.attach_debug_info:
            cond.encoder_input_pixel_values = clip_input.clone()

        clip_feature_map = self.clip.forward_model(clip_input)  # b (h w) d

        if self.cfg.model.debug_feature_maps and batch.attach_debug_info:
            from image_utils import Im
            from torchvision.transforms.functional import InterpolationMode, resize
            import copy
            orig_trained = copy.deepcopy(self.clip.state_dict())
            trained_viz = viz_feats(clip_feature_map, "trained_feature_map")
            self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False).to(torch.bfloat16).to(self.device)
            clip_feature_map = self.clip.forward_model(clip_input)
            stock_viz = viz_feats(clip_feature_map, "stock_feature_map")
            Im.concat_vertical(stock_viz, trained_viz).save(batch.metadata['name'][0])
            self.clip.load_state_dict(orig_trained)

        if isinstance(clip_feature_map, dict):
            for k in clip_feature_map.keys():
                if k != 'ln_post' and clip_feature_map[k].ndim == 3 and clip_feature_map[k].shape[1] == bs:
                    clip_feature_map[k] = rearrange("l b d -> b l d", clip_feature_map[k])

        if self.cfg.model.feature_map_keys is not None:
            clip_feature_map = torch.stack([clip_feature_map[k] for k in self.cfg.model.feature_map_keys], dim=0)
            clip_feature_map = rearrange("n b (h w) d -> b n (h w) d", clip_feature_map)  # (b, n, (h w), d)
        else:
            clip_feature_map = rearrange("b (h w) d -> b () (h w) d", clip_feature_map)  # (b, 1, (h w), d)

        clip_feature_map = clip_feature_map.to(self.dtype)
        latent_dim = self.cfg.model.encoder_latent_dim

        if clip_feature_map.shape[-2] != latent_dim**2 and "resnet" not in self.clip.model_name:
            clip_feature_map = clip_feature_map[:, :, 1:, :]
            if "dino" in self.clip.model_name and "reg" in self.clip.model_name:
                clip_feature_map = clip_feature_map[:, :, 4:, :]

        if batch.attach_debug_info:
            cond.src_feature_map = rearrange("b n (h w) d -> b n h w d", clip_feature_map.clone(), h=latent_dim, w=latent_dim)
        
        if self.cfg.model.add_pos_emb:
            pos_emb = positionalencoding2d(clip_feature_map.shape[-1], latent_dim, latent_dim, device=clip_feature_map.device, dtype=clip_feature_map.dtype).to(clip_feature_map)
            clip_feature_map = add("... (h w) d, d h w -> ... (h w) d", clip_feature_map, pos_emb)

        if self.cfg.model.feature_map_keys is not None:
            clip_feature_map = add("b n (h w) d, n d -> b n (h w) d", clip_feature_map, self.mapper.position_embedding)

        return clip_feature_map

    def add_cross_attn_params(self, batch: InputData, cond: ConditioningData):
        """
        This function sets up the attention parameters for the mask conditioning:
            - Gets CLIP features for K/V and flattens them
            - Updates our input tokens to be "A photo of mask_0 and mask_1" and so on
        """
        bs: int = batch.src_pixel_values.shape[0]
        device = batch.tgt_pixel_values.device
        latent_dim = self.cfg.model.encoder_latent_dim

        batch_one_hot_masks = []
        feature_map_masks = []
        mask_batch_idx = []
        mask_instance_idx = []
        mask_dropout = []
        segmentation_map_size = self.cfg.model.segmentation_map_size
        training_mask_dropout = self.cfg.model.training_mask_dropout
        dropout_foreground_only = self.cfg.model.dropout_foreground_only
        dropout_background_only = self.cfg.model.dropout_background_only
        background_mask_idx = self.cfg.model.background_mask_idx

        indices_ = batch.src_segmentation.view(batch.batch_size[0], -1)
        ones_ = torch.ones_like(indices_, dtype=torch.bool)
        all_non_empty_mask = ones_.new_zeros((batch.batch_size[0], 256))
        all_non_empty_mask.scatter_add_(1, indices_.long(), ones_)  # Perform batched bincount
        all_non_empty_mask = all_non_empty_mask[:, :-1]

        assert batch.src_segmentation is not None
        for i in range(bs):
            one_hot_idx = torch.arange(segmentation_map_size, device=device)
            non_empty_mask = all_non_empty_mask[i] > 0
            non_empty_mask[1:] &= batch.valid[i]

            if training_mask_dropout is not None and (self.training or batch.treat_as_train_batch):
                dropout_mask = non_empty_mask.new_full((non_empty_mask.shape[0],), False, dtype=torch.bool)
                non_empty_idx = non_empty_mask.nonzero().squeeze(1)

                if self.cfg.model.only_encode_shared_tokens:
                    allowed_idxs = batch.shared_src_tgt_instance_idxs[i][batch.shared_src_tgt_instance_idxs[i] != 255]
                    non_empty_idx = non_empty_idx[torch.isin(non_empty_idx, allowed_idxs)]

                if self.cfg.model.encode_src_twice and i >= batch.bs // 2:
                    first_img_non_empty_idx = mask_instance_idx[i - batch.bs // 2]
                    non_empty_idx = first_img_non_empty_idx

                if non_empty_idx.shape[0] > 0:
                    if self.cfg.model.less_token_dropout:
                        num_sel_masks = torch.randint(max(1, int(non_empty_idx.shape[0] * 3/4)), non_empty_idx.shape[0] + 1, (1,)) # We randomly take 1 to n available masks
                    else:
                        num_sel_masks = torch.randint(1, non_empty_idx.shape[0] + 1, (1,)) # We randomly take 1 to n available masks
                    selected_masks = non_empty_idx[torch.randperm(len(non_empty_idx))[:num_sel_masks]] # Randomly select the subset of masks
                    dropout_mask[selected_masks] = True

                if dropout_foreground_only:
                    dropout_mask[background_mask_idx] = True  # We always keep the background mask
                elif dropout_background_only:
                    dropout_mask[torch.arange(dropout_mask.shape[0]) != background_mask_idx] = True

                if dropout_mask.sum().item() == 0:
                    if non_empty_idx.shape[0] > 0:
                        dropout_mask[non_empty_idx[0]] = True
                    else:
                        log_warn("We would have dropped all masks but instead we preserved the background", main_process_only=False)
                        dropout_mask[background_mask_idx] = True
            else:
                dropout_mask = non_empty_mask.new_full((non_empty_mask.shape[0],), True, dtype=torch.bool)

            combined_mask = dropout_mask & non_empty_mask

            if self.cfg.model.max_num_training_masks is not None and (self.training or batch.treat_as_train_batch):
                keep_idxs = combined_mask.nonzero().squeeze(1)
                keep_idxs = keep_idxs[torch.randperm(keep_idxs.shape[0])]
                keep_idxs = keep_idxs[:self.cfg.model.max_num_training_masks]
                combined_mask = combined_mask.new_full((combined_mask.shape[0],), False, dtype=torch.bool)
                combined_mask[keep_idxs] = True
        
            one_hot_idx = one_hot_idx[combined_mask]

            if one_hot_idx.shape[0] == 0:
                if len(torch.unique(batch.src_segmentation[i])) <= 1:
                    log_warn(f"We have no masks for image {i}. Valid Mask Sum: {non_empty_mask.sum().item()}, Dropout Mask Sum: {dropout_mask.sum().item()}, Pixel Count Mask Sum: {(all_non_empty_mask[i] > 0).sum().item()}, Unique Src Seg: {len(torch.unique(batch.src_segmentation[i]))}. Element: {batch.metadata['name'][i]}", main_process_only=False)
                continue

            assert batch.src_pixel_values.shape[-1] == batch.src_pixel_values.shape[-2]
            one_hot_mask = get_one_hot_channels(batch.src_segmentation[i], one_hot_idx).permute(2, 0, 1)
            feature_map_mask_ = find_true_indices_batched(original=one_hot_mask, dh=latent_dim, dw=latent_dim)
            feature_map_masks.append(feature_map_mask_)
            mask_batch_idx.append(i * feature_map_mask_.new_ones((feature_map_mask_.shape[0]), dtype=torch.long))
            mask_instance_idx.append(one_hot_idx)
            mask_dropout.append(combined_mask)

            if self.cfg.model.masked_self_attention:
                def get_attn_mask(dim_):
                    flattened_mask = einops.rearrange(F.interpolate(
                        einops.rearrange(one_hot_mask.float(), "c h w -> 1 c h w"),
                        size=(dim_, dim_),
                        mode='bilinear'
                    ), "() c h w -> (h w) c")
                    return flattened_mask
                batch_one_hot_masks.append({(self.cfg.model.decoder_latent_dim // 2**(dim)): get_attn_mask(self.cfg.model.decoder_latent_dim // 2**(dim)) for dim in range(5)})

        # If the 1st image has 5 masks and the 2nd has 3 masks, we will have an integer tensor of shape (total == 8,) for 8 different cross-attns. The sequence length for each is thus the number of valid "pixels" (KVs)
        feature_map_masks = torch.cat(feature_map_masks, dim=0)  # feature_map_mask is a boolean mask of (total, h, w)
        feature_map_masks = einops.rearrange(feature_map_masks, "total h w -> total (h w)").to(device)
        mask_batch_idx = torch.cat(mask_batch_idx, dim=0).to(device)
        mask_instance_idx = torch.cat(mask_instance_idx, dim=0).to(device)
        mask_dropout = torch.stack(mask_dropout, dim=0).to(device)

        cond.mask_batch_idx = mask_batch_idx
        cond.mask_instance_idx = mask_instance_idx
        cond.mask_dropout = mask_dropout

        clip_feature_map = self.get_feature_map(batch, cond)

        if self.cfg.model.return_mean_pooled_mask_tokens:
            if self.cfg.model.mean_pooled_mask_tokens_in_bbox:
                feature_map_masks = einops.rearrange(feature_map_masks, "total (h w) -> total h w", h=latent_dim, w=latent_dim).to(device)
                indices = torch.where(feature_map_masks)
                result = torch.zeros_like(feature_map_masks, dtype=torch.bool, device=device)
                
                for i in range(feature_map_masks.size(0)):
                    min_y, max_y = indices[1][indices[0] == i].min().item(), indices[1][indices[0] == i].max().item()
                    min_x, max_x = indices[2][indices[0] == i].min().item(), indices[2][indices[0] == i].max().item()
                    result[i, min_x:max_x+1, min_y:max_y+1] = True

                feature_map_masks = result
                feature_map_masks = einops.rearrange(feature_map_masks, "total h w -> total (h w)").to(device)
                
            clip_feature_map = einops.rearrange(clip_feature_map, "b layers tokens d -> b tokens (layers d)")
            feature_maps__ = torch.stack([clip_feature_map[batch_idx] for batch_idx, mask in zip(mask_batch_idx, feature_map_masks)])
            masked_feature_maps = feature_maps__ * feature_map_masks.unsqueeze(-1)
            cond.mask_tokens = masked_feature_maps.sum(dim=1) / feature_map_masks.sum(dim=1).unsqueeze(-1)
            return cond

        # In the simplest case, we have one query per mask, and a single feature map. However, we support multiple queries per mask 
        # [to generate tokens for different layers] and also support each query to have K/Vs  from different feature maps, as long as all feature maps 
        # have the same dimension allowing us to assume that each mask has the same number of pixels (KVs) for each feature map.
        num_queries_per_mask = self.cfg.model.num_layer_queries if self.cfg.model.per_layer_queries else 1
        num_feature_maps = clip_feature_map.shape[1]
        seqlens_k = feature_map_masks.sum(dim=-1) * num_feature_maps  # We sum the number of valid "pixels" in each mask
        seqlens_k = einops.repeat(seqlens_k, 'masks -> (masks layers)', layers=num_queries_per_mask)

        assert round(math.sqrt(clip_feature_map.shape[-2])) == latent_dim

        max_seqlen_k = seqlens_k.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0, dtype=torch.torch.int32), (1, 0))

        flat_features = einops.repeat(clip_feature_map[mask_batch_idx], "masks n hw d -> (masks layers n hw) d", layers=num_queries_per_mask)
        flat_mask = einops.repeat(feature_map_masks, "masks hw -> (masks layers n hw)", layers=num_queries_per_mask, n=num_feature_maps) # Repeat mask over layers
        k_features = flat_features[flat_mask]

        # The actual query is obtained from the mapper class (a learnable tokens)        
        cu_seqlens_q = F.pad(torch.arange(seqlens_k.shape[0]).to(torch.int32) + 1, (1, 0)).to(device)
        max_seqlen_q = 1  # We are doing attention pooling so we have one query per mask

        cond.attn_dict = dict(x_kv=k_features, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)

        if len(batch_one_hot_masks) > 0:
            cond.batch_attn_masks = {k: [m[k] for m in batch_one_hot_masks] for k in batch_one_hot_masks[0].keys()}
            
        if self.cfg.model.clip_shift_scale_conditioning:
            forward_shift_scale(self, cond, clip_feature_map, latent_dim)

        return cond
    
    def get_input_for_batched_src_tgt(self, batch: InputData):
        src_batch = batch.clone()
        tgt_batch = batch.clone()

        tgt_batch.src_pixel_values = tgt_batch.tgt_enc_norm_pixel_values
        tgt_batch.src_segmentation = tgt_batch.tgt_enc_norm_segmentation

        src_batch.tgt_enc_norm_pixel_values = None
        src_batch.tgt_enc_norm_segmentation = None
        tgt_batch.tgt_enc_norm_pixel_values = None
        tgt_batch.tgt_enc_norm_segmentation = None

        input_batch = torch.cat((src_batch, tgt_batch), dim=0)
        return input_batch

    def compute_mask_tokens(self, batch: InputData, cond: ConditioningData):
        """
        Generates mask tokens by adding segmentation if necessary, then setting up and calling the cross attention module
        """
        batch = self.check_add_segmentation(batch)

        if self.cfg.model.only_encode_shared_tokens:
            unique_idxs = []
            for b in range(batch.bs):
                valid_src_idxs = torch.unique(batch.src_segmentation[b][batch.src_segmentation[b] != 255])
                valid_tgt_idxs = torch.unique(batch.tgt_segmentation[b][batch.tgt_segmentation[b] != 255])
                allowed_idxs = torch.from_numpy(np.intersect1d(valid_src_idxs.cpu().numpy(), valid_tgt_idxs.cpu().numpy())).to(batch.device)
                allowed_idxs = torch.cat((allowed_idxs, allowed_idxs.new_full((255 - allowed_idxs.shape[0],), 255)))
                unique_idxs.append(allowed_idxs)
            batch.shared_src_tgt_instance_idxs = torch.stack(unique_idxs, dim=0)

        if self.cfg.model.encode_tgt or batch.force_forward_encoder_normalized_tgt:
            assert self.cfg.model.encode_src_twice is False or batch.force_forward_encoder_normalized_tgt
            input_batch = self.get_input_for_batched_src_tgt(batch)
        elif self.cfg.model.encode_src_twice:
            input_batch = torch.cat((batch, batch), dim=0)
        else:
            input_batch = batch
        
        cond = self.add_cross_attn_params(input_batch, cond)
        if self.cfg.model.return_mean_pooled_mask_tokens: return cond

        orig_queries = einops.repeat(self.mapper.learnable_token, "layers d -> (masks layers) d", masks=cond.mask_batch_idx.shape[0])
        x_kv_orig = cond.attn_dict["x_kv"].clone()

        queries = orig_queries.clone()
        cond.attn_dict["x"] = queries.to(self.dtype)
        cond.attn_dict["x_kv"] = x_kv_orig
        cond.mask_tokens = self.mapper.cross_attn(cond).to(self.dtype)
        
        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            if self.cfg.model.detach_features_before_cross_attn:
                cond.attn_dict["x"] = orig_queries.to(self.dtype)
                cond.attn_dict["x_kv"] = x_kv_orig.detach() # mapper.position_embedding is frozen
                cond.mask_head_tokens = self.mapper.cross_attn(cond).to(self.dtype)
            else:
                cond.mask_head_tokens = cond.mask_tokens

        if self.cfg.model.per_layer_queries: # Break e.g., 1024 -> 16 x 64
            if cond.mask_tokens is not None:
                cond.mask_tokens = einops.rearrange(cond.mask_tokens, "(tokens layers) d -> tokens (layers d)", tokens=cond.mask_batch_idx.shape[0])

            if cond.mask_head_tokens is not None:
                cond.mask_head_tokens = einops.rearrange(cond.mask_head_tokens, "(tokens layers) d -> tokens (layers d)", tokens=cond.mask_batch_idx.shape[0])

        if self.cfg.model.encode_src_twice or self.cfg.model.encode_tgt:
           cond.attn_dict = None
           token_mask = cond.mask_batch_idx < batch.bs
           tgt_token_mask = cond.mask_batch_idx >= batch.bs

           cond.tgt_mask_tokens = cond.mask_tokens[tgt_token_mask]
           cond.tgt_mask_batch_idx = cond.mask_batch_idx[tgt_token_mask] - batch.bs
           cond.tgt_mask_instance_idx = cond.mask_instance_idx[tgt_token_mask]
           cond.tgt_mask_dropout = cond.mask_dropout[torch.arange(cond.mask_dropout.shape[0], device=self.device) >= batch.bs]

           cond.mask_tokens = cond.mask_tokens[token_mask]
           cond.mask_batch_idx = cond.mask_batch_idx[token_mask]
           cond.mask_instance_idx = cond.mask_instance_idx[token_mask]
           cond.mask_dropout = cond.mask_dropout[torch.arange(cond.mask_dropout.shape[0], device=self.device) < batch.bs]
               
        if self.cfg.model.modulate_src_tokens_with_tgt_pose:
            hidden_dim = cond.mask_tokens.shape[1]

            relative_pose = get_relative_pose(batch.src_pose, batch.tgt_pose).to(batch.src_pose)
            register_tokens = self.mapper.camera_embed(relative_pose.view(batch.bs, -1).to(self.dtype)).unsqueeze(1)
            register_tokens = torch.cat((register_tokens, register_tokens.new_zeros((*register_tokens.shape[:-1], hidden_dim - register_tokens.shape[-1]))), dim=-1)

            if self.cfg.model.modulate_src_tokens_with_mlp:
                token_modulator_input = torch.cat((cond.mask_tokens, register_tokens.squeeze(1)[cond.mask_batch_idx]), dim=-1)
                cond.mask_tokens = self.mapper.token_modulator(token_modulator_input)
            elif self.cfg.model.modulate_src_tokens_with_film:
                _output = self.mapper.token_modulator(register_tokens.squeeze(1))[cond.mask_batch_idx]
                scale, shift = einops.rearrange(_output, "b (n a) -> a b n", a=2)
                cond.mask_tokens = cond.mask_tokens * (1 - scale) + shift
            else:
                batch_size = cond.mask_batch_idx.max().item() + 1  

                seq_lengths = torch.bincount(cond.mask_batch_idx)
                cu_seqlens = torch.cumsum(seq_lengths, dim=0)  
                cu_seqlens = torch.cat([torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device), cu_seqlens])

                max_seqlen = seq_lengths.max().item()  
                split_tokens = torch.split(cond.mask_tokens, seq_lengths.tolist())  
                new_tokens = [torch.cat([tokens, register_tokens[i]], dim=0) for i, tokens in enumerate(split_tokens)]

                new_cond_mask_tokens = torch.cat(new_tokens, dim=0)  
                new_cu_seqlens = cu_seqlens + torch.arange(batch_size + 1, device=cu_seqlens.device)  
                new_max_seqlen = max_seqlen + 1

                output = self.mapper.token_modulator(x=new_cond_mask_tokens, mixer_kwargs={'cu_seqlens': new_cu_seqlens.to(torch.int32), 'max_seqlen': new_max_seqlen})

                split_tokens = torch.split(output, (seq_lengths + 1).tolist(), dim=0)
                new_tokens = [tokens[:-1] for i, tokens in enumerate(split_tokens)]
                cond.mask_tokens = torch.cat(new_tokens, dim=0)

            cond.src_mask_tokens = cond.mask_tokens.clone()
        
        if self.cfg.model.layer_specialization and self.cfg.model.num_conditioning_pairs != self.cfg.model.num_layer_queries: # Break e.g., 1024 -> 16 x 64
            def layer_specialization(_mask_tokens, layers):
                layerwise_mask_tokens = einops.rearrange(_mask_tokens, "tokens (layers d) -> tokens layers d", layers=layers)
                layerwise_mask_tokens = self.mapper.layer_specialization(layerwise_mask_tokens)  # Batched 64 -> 1024
                return einops.rearrange(layerwise_mask_tokens, "tokens layers d -> tokens (layers d)").to(self.dtype)
            
            if cond.mask_tokens is not None: cond.mask_tokens = layer_specialization(cond.mask_tokens, self.cfg.model.num_conditioning_pairs)
            if cond.mask_head_tokens is not None: cond.mask_head_tokens = layer_specialization(cond.mask_head_tokens, self.cfg.model.num_conditioning_pairs // self.cfg.model.num_layer_queries)

        if self.cfg.model.encode_src_twice:
            cond.tgt_mask_tokens = layer_specialization(cond.tgt_mask_tokens, self.cfg.model.num_conditioning_pairs)
            cond.src_mask_tokens = cond.mask_tokens.clone()

        return cond

    def update_hidden_state_with_mask_tokens(
        self,
        batch: InputData,
        cond: ConditioningData,
    ):
        bs = batch.input_ids.shape[0]

        if self.cfg.model.layer_specialization:
            cond.encoder_hidden_states = rearrange("b tokens d -> b tokens (n d)", cond.encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)

        if self.cfg.model.add_text_tokens:
            batch.formatted_input_ids = batch.input_ids.clone()
            for b in range(bs):
                cur_ids = batch.input_ids[b]
                token_is_padding = (cur_ids == self.pad_token_id).nonzero()  # Everything after EOS token should also be a pad token
                assert (token_is_padding.shape[0] == (batch.input_ids.shape[1] - token_is_padding[0])).item()

                mask_part_of_batch = (cond.mask_batch_idx == b).nonzero().squeeze(1)  # Figure out which masks we need to add
                masks_prompt = torch.tensor(self.mask_tokens_ids * mask_part_of_batch.shape[0])[:-1].to(self.device)
                assert token_is_padding.shape[0] >= masks_prompt.shape[0]  # We need at least as many pad tokens as we have masks

                # We take everything before the placeholder token and combine it with "placeholder_token and placeholder_token and ..."
                # We then add the rest of the sentence on (including the EOS token and padding tokens)
                placeholder_locs = (cur_ids == self.placeholder_token_id).nonzero()
                assert placeholder_locs.shape[0] == 1  # We should only have one placeholder token
                start_of_prompt = cur_ids[: placeholder_locs[0]]
                end_of_prompt = cur_ids[placeholder_locs[0] + 1 :]
                eos_token = torch.tensor([self.eos_token_id]).to(self.device)

                batch.formatted_input_ids[b] = torch.cat((start_of_prompt, masks_prompt, end_of_prompt, eos_token), dim=0)[:cur_ids.shape[0]]

            # Overwrite mask locations
            cond.learnable_idxs = (batch.formatted_input_ids == self.placeholder_token_id).nonzero(as_tuple=True)
            cond.encoder_hidden_states[cond.learnable_idxs[0], cond.learnable_idxs[1]] = cond.mask_tokens.to(cond.encoder_hidden_states)
        else:
            batch.formatted_input_ids = torch.zeros_like(batch.input_ids)
            batch_learnable_idx, batch_learnable_pos = torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
            for batch_idx in range(batch.bs):
                mask = (cond.mask_batch_idx == batch_idx)
                selected_tokens = cond.mask_tokens[mask]
                num_tokens = selected_tokens.size(0)
                if self.training is False:
                    selected_tokens = selected_tokens[:min(num_tokens, self.cfg.model.num_decoder_cross_attn_tokens)]
                cond.encoder_hidden_states[batch_idx, :num_tokens, :] = selected_tokens

                batch_learnable_idx = torch.cat((batch_learnable_idx, torch.full((num_tokens,), batch_idx, dtype=torch.long)))
                batch_learnable_pos = torch.cat((batch_learnable_pos, torch.arange(min(num_tokens, cond.encoder_hidden_states.shape[1]), dtype=torch.long)))

            cond.learnable_idxs = (batch_learnable_idx, batch_learnable_pos)
        
        attn_meta: AttentionMetadata = cond.unet_kwargs["cross_attention_kwargs"]['attn_meta']
        if self.cfg.model.layer_specialization:
            attn_meta.layer_idx = 0
            attn_meta.num_layers = (8 * 2) if self.cfg.model.custom_conditioning_map else (self.cfg.model.num_conditioning_pairs * 2)
            attn_meta.num_cond_vectors = self.cfg.model.num_conditioning_pairs
            attn_meta.add_pos_emb = self.cfg.model.add_pos_emb

            if self.cfg.model.custom_conditioning_map:
                attn_meta.custom_map = {
                    0: 0, 1: 0, 2: 0, 3: 0,
                    4: 1, 5: 1, 6: 1, 7: 1,
                    8: 1, 9: 1, 10: 1, 11: 1,
                    12: 0, 13: 0, 14: 0, 15: 0,
                }

            if self.cfg.model.gated_cross_attn:
                attn_meta.gate_scale = 1
                with torch.no_grad():
                    frozen_enc = self.text_encoder(input_ids=_get_tokens(self.tokenizer, "A photo")[None].to(cond.encoder_hidden_states.device))[0].to(dtype=self.dtype)
                cond.encoder_hidden_states = rearrange("b t (layers d), () t d -> b t ((layers + 1) d)", cond.encoder_hidden_states, frozen_enc)
                attn_meta.frozen_dim = frozen_enc.shape[-1]

        if self.cfg.model.layer_specialization is False and self.cfg.model.eschernet:
            attn_meta.layer_idx = 0
            attn_meta.num_layers = 1
            attn_meta.num_cond_vectors = self.cfg.model.num_conditioning_pairs
            attn_meta.add_pos_emb = self.cfg.model.add_pos_emb

        if self.cfg.model.clip_shift_scale_conditioning:
            assert not self.cfg.model.layer_specialization
            cond.encoder_hidden_states[:] = shift_scale_uncond_hidden_states(self)

        if self.cfg.model.eschernet:
            if self.cfg.model.eschernet_6dof:
                attn_meta.posemb = [
                    batch.tgt_pose.to(self.dtype).detach(),
                    torch.linalg.inv(batch.tgt_pose).to(self.dtype).detach(),
                    batch.src_pose.to(self.dtype).detach(), 
                    torch.linalg.inv(batch.src_pose).to(self.dtype).detach()
                ]
            else:
                attn_meta.posemb = [batch.tgt_pose, batch.src_pose]

        if self.cfg.model.masked_self_attention:
            attn_meta.self_attention_mask = cond.batch_attn_masks

        return cond

    def get_hidden_state(
        self,
        batch: InputData,
        add_conditioning: bool = True,
        cond: Optional[ConditioningData] = None,  # We can optionally specify mask tokens to use [e.g., for composing during inference]
    ) -> ConditioningData:
        if cond is None:
            cond = ConditioningData()
        
        if len(cond.unet_kwargs) == 0:
            cond.unet_kwargs["cross_attention_kwargs"] = dict(attn_meta=AttentionMetadata())

        cond.placeholder_token = self.placeholder_token_id
        
        if self.cfg.model.add_text_tokens:
            with torch.no_grad(): cond.encoder_hidden_states = self.text_encoder(input_ids=batch.input_ids)[0].to(dtype=self.dtype)
        else:
            cond.encoder_hidden_states = torch.zeros((batch.bs, self.cfg.model.num_decoder_cross_attn_tokens, self.cfg.model.token_embedding_dim), dtype=self.dtype, device=self.device)

        if add_conditioning and self.cfg.model.mask_token_conditioning:
            if cond.mask_tokens is None or cond.mask_batch_idx is None:
                cond = self.compute_mask_tokens(batch, cond)
                if self.cfg.model.return_mean_pooled_mask_tokens: return cond
            
            if cond.mask_tokens is not None:
                cond = self.update_hidden_state_with_mask_tokens(batch, cond)

            if self.cfg.model.attention_masking:
                attention_masking(batch, cond)

        return cond

    def get_controlnet_conditioning(self, batch):
        return batch.src_segmentation.permute(0, 3, 1, 2).to(dtype=self.dtype)

    def dropout_cfg(self, cond: ConditioningData):
        device: torch.device = cond.encoder_hidden_states.device
        frozen_dim: int = cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].frozen_dim
        frozen_dim = -frozen_dim if frozen_dim is not None else None

        # We dropout the entire conditioning [all-layers] for a subset of batches.
        if self.cfg.model.training_cfg_dropout is not None:
            uncond_encoder_hidden_states = self.uncond_hidden_states

            if self.cfg.model.layer_specialization:  # If we have different embeddings per-layer, we need to repeat the uncond embeddings
                uncond_encoder_hidden_states = rearrange("tokens d -> tokens (n d)", uncond_encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)

            dropout_mask = torch.rand(cond.encoder_hidden_states.shape[0], device=device) < self.cfg.model.training_cfg_dropout
            cond.encoder_hidden_states[dropout_mask, ..., :frozen_dim] = uncond_encoder_hidden_states
            cond.batch_cond_dropout = dropout_mask

            if self.cfg.model.attention_masking:
                handle_attention_masking_dropout(cond, dropout_mask)

        # We also might dropout only specific pairs of layers. In this case, we use the same uncond embeddings.
        # Note that there is a very rare chance that we could dropout all layers but still compute loss.
        if self.cfg.model.layer_specialization and self.cfg.model.training_layer_dropout is not None:
            dropout_mask = torch.rand(cond.encoder_hidden_states.shape[0], self.cfg.model.num_conditioning_pairs, device=device) < self.cfg.model.training_layer_dropout
            cond.encoder_hidden_states[..., :frozen_dim] = where("b n, tokens d, b tokens (n d) -> b tokens (n d)", dropout_mask, self.uncond_hidden_states, cond.encoder_hidden_states[..., :frozen_dim])

    def inverted_cosine(self, timesteps):
        return torch.arccos(torch.sqrt(timesteps))

    def forward(self, batch: InputData):
        assert batch.formatted_input_ids is None

        if self.cfg.model.predict_rotation_from_n_frames:
            # We keep the tensor format as (bs, ...) even though it's really ((bs frames), ...) but specify num_frames
            assert batch.bs % self.cfg.model.predict_rotation_from_n_frames == 0
            batch.num_frames = self.cfg.model.predict_rotation_from_n_frames

        batch.tgt_pixel_values = torch.clamp(batch.tgt_pixel_values, -1, 1)

        # Sample a random timestep for each element in batch
        if self.cfg.model.use_inverted_noise_schedule:
            timesteps = ((torch.arccos(timesteps / self.noise_scheduler.config.num_train_timesteps) / (torch.pi / 2)) * self.noise_scheduler.config.num_train_timesteps).long()
        elif self.cfg.model.diffusion_timestep_range is not None:
            timesteps = torch.randint(self.cfg.model.diffusion_timestep_range[0], self.cfg.model.diffusion_timestep_range[1], (batch.bs,), device=self.device).long()
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch.bs,), device=self.device).long()

        cond = self.get_hidden_state(batch, add_conditioning=True)

        if self.training: self.dropout_cfg(cond)

        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            pred_data = TokenPredData()

            # We must call this before rotation/cls prediction to properly setup the GT and choose which mask tokens to predict
            # Make sure to also call token_rot_loss afterwards.
            if self.cfg.model.token_rot_pred_loss:
                pred_data = get_gt_rot(self.cfg, cond, batch, pred_data)
                if self.cfg.model.token_rot_pred_loss and not self.cfg.model.predict_rotation_from_n_frames:
                    max_timesteps = self.cfg.model.rotation_diffusion_start_timestep if self.cfg.model.rotation_diffusion_start_timestep else self.cfg.model.rotation_diffusion_timestep
                    pred_data.timesteps = torch.randint(0, max_timesteps, (pred_data.mask_tokens.shape[0],), device=self.device).long()
                    pred_data.rot_6d_noise = torch.randn_like(pred_data.gt_rot_6d)
                    pred_data.noised_rot_6d = self.rotation_scheduler.add_noise(pred_data.gt_rot_6d, pred_data.rot_6d_noise, pred_data.timesteps)
            
            pred_data = self.token_mapper(batch=batch, cond=cond, pred_data=pred_data)

        encoder_hidden_states = cond.encoder_hidden_states

        if self.cfg.model.unet:
            latents = self.vae.encode(batch.tgt_pixel_values.to(dtype=self.dtype)).latent_dist.sample() # Convert images to latent space
            latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents) # Sample noise that we'll add to the latents

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            attn_meta = cond.unet_kwargs.get('cross_attention_kwargs', {}).get('attn_meta', None)
            if attn_meta is None or attn_meta.layer_idx is None:
                if 'cross_attention_kwargs' in cond.unet_kwargs and 'attn_meta' in cond.unet_kwargs['cross_attention_kwargs']:
                    del cond.unet_kwargs['cross_attention_kwargs']['attn_meta']

            if self.cfg.model.add_grid_to_input_channels:
                downsampled_grid = get_tgt_grid(self.cfg, batch)
                if self.cfg.model.dropout_grid_conditioning is not None:
                    dropout = torch.rand(batch.bs, device=batch.device) < self.cfg.model.dropout_grid_conditioning
                    dropout_grid = get_dropout_grid(self.cfg.model.decoder_latent_dim).to(downsampled_grid)
                    downsampled_grid[dropout] = dropout_grid
                noisy_latents = torch.cat([noisy_latents, downsampled_grid], dim=1)

            if self.cfg.model.controlnet:
                controlnet_image = self.get_controlnet_conditioning(batch)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(torch.float32),
                    down_block_additional_residuals=[sample.to(dtype=self.dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
                ).sample
            else:
                # TODO: We shouldn't need to cast to FP32 here
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(torch.float32), **cond.unet_kwargs).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        losses = dict()
        if self.cfg.model.unet:
            if batch.tgt_pad_mask is not None and self.cfg.model.use_pad_mask_loss:
                loss_mask = rearrange("b h w -> b () h w ", ~batch.tgt_pad_mask)
                loss_mask = F.interpolate(
                    loss_mask.float(),
                    size=(self.cfg.model.decoder_latent_dim, self.cfg.model.decoder_latent_dim),
                    mode='nearest'
                )
                model_pred, target = model_pred * loss_mask, target * loss_mask

            if self.cfg.model.break_a_scene_masked_loss and self.cfg.model.mask_token_conditioning:
                loss_mask = break_a_scene_masked_loss(cfg=self.cfg, batch=batch, cond=cond)
                model_pred, target = model_pred * loss_mask, target * loss_mask

            if self.cfg.model.weighted_object_loss:
                losses["diffusion_loss"] = evenly_weighted_mask_loss(cfg=self.cfg, batch=batch, cond=cond, pred=model_pred.float(), target=target.float())
            else:
                losses["diffusion_loss"] = F.mse_loss(model_pred.float(), target.float(), reduction="mean") * self.cfg.model.diffusion_loss_weight

            if self.cfg.model.break_a_scene_cross_attn_loss:
                losses["cross_attn_loss"] = break_a_scene_cross_attn_loss(cfg=self.cfg, batch=batch, controller=self.controller, cond=cond)

        if self.cfg.model.token_cls_pred_loss:
            losses.update(token_cls_loss(cfg=self.cfg, batch=batch, cond=cond, pred_data=pred_data))

        if self.cfg.model.token_rot_pred_loss:
            losses.update(token_rot_loss(cfg=self.cfg, pred_data=pred_data))

        if self.cfg.model.src_tgt_consistency_loss_weight is not None:
            losses.update(src_tgt_token_consistency_loss(cfg=self.cfg, batch=batch, cond=cond))

        losses["metric_num_mask_tokens"] = cond.mask_tokens.shape[0]
            
        return losses
