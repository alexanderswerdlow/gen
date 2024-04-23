import dataclasses
import gc
import math
from contextlib import nullcontext
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
from peft import LoraConfig, get_peft_model
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from torch import FloatTensor, Tensor
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel

from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers.utils.torch_utils import randn_tensor
from gen.configs import BaseConfig
from gen.models.cross_attn.break_a_scene import register_attention_control
from gen.models.cross_attn.deprecated_configs import (attention_masking, forward_shift_scale, handle_attention_masking_dropout, init_shift_scale,
                                                      shift_scale_uncond_hidden_states)
from gen.models.cross_attn.eschernet import get_relative_pose
from gen.models.cross_attn.losses import (break_a_scene_cross_attn_loss, break_a_scene_masked_loss, cosine_similarity_loss, evenly_weighted_mask_loss,
                                          get_gt_rot, src_tgt_feature_map_consistency_loss, src_tgt_token_consistency_loss,
                                          tgt_positional_information_loss, token_cls_loss, token_rot_loss)
from gen.models.cross_attn.modules import FeatureMapper, TokenMapper
from gen.models.encoders.encoder import BaseModel
from gen.models.utils import find_true_indices_batched, positionalencoding2d
from gen.utils.data_defs import InputData, get_dropout_grid, get_one_hot_channels, get_tgt_grid, undo_normalization_given_transforms
from gen.utils.decoupled_utils import get_modules, to_numpy
from gen.utils.diffusers_utils import load_stable_diffusion_model
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.misc_utils import compute_centroids
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

    mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    src_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    tgt_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    gt_src_mask_token_pos_emb: Optional[Float[Tensor, "n d"]] = None
    gt_src_mask_token: Optional[Float[Tensor, "n d"]] = None

    src_mask_tokens_before_specialization: Optional[Float[Tensor, "n d"]] = None # We duplicate for loss calculation
    tgt_mask_tokens_before_specialization: Optional[Float[Tensor, "n d"]] = None

    src_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    encoder_input_pixel_values: Optional[Float[Tensor, "b c h w"]] = None

    src_orig_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    tgt_orig_feature_map: Optional[Float[Tensor, "b d h w"]] = None

    src_warped_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    tgt_warped_feature_map: Optional[Float[Tensor, "b d h w"]] = None

    mask_token_centroids: Optional[Float[Tensor, "n 2"]] = None
    tgt_mask_token_centroids: Optional[Float[Tensor, "n 2"]] = None

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
        self.dtype = getattr(torch, cfg.trainer.dtype.split(".")[-1]) if isinstance(cfg.trainer.dtype, str) else cfg.trainer.dtype

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
            if self.cfg.model.modulate_src_feature_map:
                from gen.models.encoders.vision_transformer import vit_base_patch14_dinov2
            
            if self.cfg.model.custom_dino_v2:
                self.clip = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
                self.clip = torch.compile(self.clip)
            else:
                self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False)
            self.clip.to(self.dtype)

            if self.cfg.model.clip_lora:
                self.clip.requires_grad_(True)
                from peft import LoraConfig, inject_adapter_in_model
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
        tokenizer_encoder_name = "CompVis/stable-diffusion-v1-4" if self.cfg.model.use_sd_15_tokenizer_encoder else self.cfg.model.pretrained_model_name_or_path
        revision = None if self.cfg.model.use_sd_15_tokenizer_encoder else self.cfg.model.revision
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_encoder_name, subfolder="tokenizer", revision=revision, use_fast=False
        )

        # Load scheduler and models
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.rotation_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.model.rotation_diffusion_timestep,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=self.cfg.model.rotation_diffusion_parameterization,
        )
        
        unet_kwargs = dict()
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
            if (self.cfg.model.freeze_text_encoder is False and self.cfg.model.add_text_tokens) or self.cfg.model.text_encoder_lora:
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

        if md.modulate_src_feature_map:
            if hasattr(other.clip, "base_model"):
                model_ = other.clip.base_model
            elif hasattr(other.clip, "model"):
                model_ = other.clip.model

            if set_grad:
                model_.mid_blocks.requires_grad_(True)
                model_.final_blocks.requires_grad_(True)
                model_.final_norm.requires_grad_(True)

            model_.mid_blocks.train()
            model_.final_blocks.train()
            model_.final_norm.train()

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

        if md.add_text_tokens or md.tgt_positional_information_from_lang:
            if md.text_encoder_lora:
                other.text_encoder.train()
            elif md.freeze_text_encoder:
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

            if md.modulate_src_tokens_with_tgt_pose or md.modulate_src_feature_map:
                if set_grad:
                    other.mapper.token_predictor.requires_grad_(True)
                    other.mapper.token_predictor.to(device=_device, dtype=torch.float32)

                    other.mapper.layer_specialization.requires_grad_(True)
                    other.mapper.layer_specialization.to(device=_device, dtype=torch.float32)
                other.mapper.token_predictor.train()
                other.mapper.layer_specialization.train()

            if md.tgt_positional_information_from_lang:
                if set_grad:
                    other.mapper.predict_positional_information.requires_grad_(True)
                    other.mapper.predict_positional_information.to(device=_device, dtype=torch.float32)
                
                other.mapper.predict_positional_information.train()

                if set_grad:
                    other.mapper.positional_information_mlp.requires_grad_(True)
                    other.mapper.positional_information_mlp.to(device=_device, dtype=torch.float32)
                other.mapper.positional_information_mlp.train()

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

        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    def get_custom_params(self):
        """
        Returns all params directly managed by the top-level model.
        Other params may be nested [e.g., in diffusers]
        """
        params = {}
        if self.cfg.model.clip_shift_scale_conditioning:
            params.update(dict(self.clip_proj_layers.named_parameters()))
        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            params.update(dict(self.token_mapper.named_parameters()))
                        
        params.update(dict(self.mapper.named_parameters()))

        # Add clip parameters
        params.update({k:p for k,p in self.clip.named_parameters() if p.requires_grad})

        return params

    def get_unet_params(self):
        is_unet_trainable = self.cfg.model.unet and (not self.cfg.model.freeze_unet or self.cfg.model.unet_lora)
        return {k:v for k,v in self.unet.named_parameters() if v.requires_grad} if is_unet_trainable else dict()
    
    def get_param_groups(self):
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

    def on_sync_gradients(self, state: TrainingState):
        if self.cfg.model.unet and not self.cfg.model.freeze_unet and self.cfg.model.ema:
            self.ema_unet.step(self.unet.parameters())

    def process_input(self, batch: dict, state: TrainingState) -> InputData:
        if "input_ids" in batch and isinstance(batch["input_ids"], list):
            batch["input_ids"] = _get_tokens(self.tokenizer, batch["input_ids"])
        if not isinstance(batch, InputData):
            batch: InputData = InputData.from_dict(batch)

        batch.state = state
        batch.dtype = self.dtype
        return batch

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
        if self.cfg.model.tgt_positional_information_from_lang is False:
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

    def get_pose_embedding(self, batch: InputData, src_pose: FloatTensor, tgt_pose: FloatTensor, hidden_dim: int):
        if self.cfg.model.use_euler_camera_emb:
            relative_rot = torch.bmm(tgt_pose[:, :3, :3].mT, src_pose[:, :3, :3])
            rot = Rotation.from_matrix(relative_rot.float().cpu().numpy())
            relative_rot = torch.from_numpy(rot.as_euler("xyz", degrees=False)).to(src_pose)
            relative_trans = torch.bmm(tgt_pose[:, :3, :3].mT, (src_pose[:, :3, 3] - tgt_pose[:, :3, 3])[:, :, None]).squeeze(-1)
            relative_pose = torch.cat((relative_rot, relative_trans), dim=1)
        else:
            relative_pose = get_relative_pose(src_pose, tgt_pose).to(src_pose)
        register_tokens = self.mapper.token_predictor.camera_embed(relative_pose.view(batch.bs, -1).to(self.dtype)).unsqueeze(1)
        register_tokens = torch.cat((register_tokens, register_tokens.new_zeros((*register_tokens.shape[:-1], hidden_dim - register_tokens.shape[-1]))), dim=-1)
        return register_tokens

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

        if self.cfg.model.modulate_src_feature_map:
            pose_emb = self.get_pose_embedding(batch, batch.src_pose, batch.tgt_pose, self.cfg.model.encoder_dim) + self.mapper.token_predictor.camera_position_embedding
            clip_feature_map = self.clip.forward_model(clip_input, y=pose_emb)  # b (h w) d

            clip_feature_map['mid_blocks'] = clip_feature_map['mid_blocks'][:, 1:, :]
            clip_feature_map['final_norm'] = clip_feature_map['final_norm'][:, 1:, :]

            orig_bs = batch.bs // 2
            assert self.cfg.model.encode_tgt_enc_norm

            _orig_feature_maps = torch.stack((clip_feature_map['blocks.5'], clip_feature_map['norm']), dim=0)[:, :, 5:, :]
            _warped_feature_maps = torch.stack((clip_feature_map['mid_blocks'], clip_feature_map['final_norm']), dim=0)[:, :, 5:, :]

            cond.src_orig_feature_map = _orig_feature_maps[:, :orig_bs].clone()
            cond.tgt_orig_feature_map = _orig_feature_maps[:, orig_bs:].clone()

            cond.src_warped_feature_map = _warped_feature_maps[:, :orig_bs].clone()
            cond.tgt_warped_feature_map = _warped_feature_maps[:, orig_bs:].clone()

            clip_feature_map['blocks.5'] = torch.cat((clip_feature_map['mid_blocks'][:orig_bs], clip_feature_map['blocks.5'][orig_bs:]), dim=0)
            clip_feature_map['norm'] = torch.cat((clip_feature_map['final_norm'][:orig_bs], clip_feature_map['norm'][orig_bs:]), dim=0)
        elif self.cfg.model.custom_dino_v2:
            with torch.no_grad():
                # [:, 4:]
                clip_feature_map = {f'blocks.{i}':v for i,v in enumerate(self.clip.get_intermediate_layers(x=clip_input, n=24 if 'large' in self.cfg.model.encoder.model_name else 12, norm=True))}
        else:
            with torch.no_grad() if self.cfg.model.freeze_clip and self.cfg.model.unfreeze_last_n_clip_layers is None else nullcontext():
                clip_feature_map = self.clip.forward_model(clip_input)  # b (h w) d

                if self.cfg.model.norm_vit_features:
                    for k in clip_feature_map.keys():
                        if "blocks" in k:
                            clip_feature_map[k] = self.clip.base_model.norm(clip_feature_map[k])

        if self.cfg.model.debug_feature_maps and batch.attach_debug_info:
            import copy
            from torchvision.transforms.functional import InterpolationMode, resize
            from image_utils import Im
            orig_trained = copy.deepcopy(self.clip.state_dict())
            trained_viz = viz_feats(clip_feature_map, "trained_feature_map")
            self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False).to(self.dtype).to(self.device)
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

        if clip_feature_map.shape[-2] != latent_dim**2 and "resnet" not in self.cfg.model.encoder.model_name:
            clip_feature_map = clip_feature_map[:, :, 1:, :]
            if "dino" in self.cfg.model.encoder.model_name and "reg" in self.cfg.model.encoder.model_name:
                clip_feature_map = clip_feature_map[:, :, 4:, :]

        if batch.attach_debug_info:
            cond.src_feature_map = rearrange("b n (h w) d -> b n h w d", clip_feature_map.clone(), h=latent_dim, w=latent_dim)

        if self.cfg.model.merge_feature_maps:
            clip_feature_map = rearrange("b n (h w) d -> b () (h w) (n d)", clip_feature_map)
        
        if self.cfg.model.add_pos_emb:
            pos_emb = positionalencoding2d(clip_feature_map.shape[-1], latent_dim, latent_dim, device=clip_feature_map.device, dtype=clip_feature_map.dtype).to(clip_feature_map)
            clip_feature_map = add("... (h w) d, d h w -> ... (h w) d", clip_feature_map, pos_emb)
        elif self.cfg.model.add_learned_pos_emb_to_feature_map:
            clip_feature_map = add("... (h w) d, (h w) d -> ... (h w) d", clip_feature_map, self.mapper.feature_map_pos_emb)

        if self.cfg.model.feature_map_keys is not None and self.cfg.model.merge_feature_maps is False:
            clip_feature_map = add("b n (h w) d, n d -> b n (h w) d", clip_feature_map, self.mapper.position_embedding)

        return clip_feature_map

    def update_hidden_state_with_mask_tokens(
        self,
        batch: InputData,
        cond: ConditioningData,
    ):
        bs = batch.input_ids.shape[0]

        cond.encoder_hidden_states[cond.learnable_idxs[0], cond.learnable_idxs[1]] = cond.mask_tokens.to(cond.encoder_hidden_states)

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
        
        cond.encoder_hidden_states = torch.zeros((batch.bs, self.cfg.model.num_decoder_cross_attn_tokens, self.cfg.model.token_embedding_dim), dtype=self.dtype, device=self.device)

        if add_conditioning and self.cfg.model.mask_token_conditioning:
            if cond.mask_tokens is None or cond.mask_batch_idx is None:
                cond = self.compute_mask_tokens(batch, cond)
            
            if cond.mask_tokens is not None:
                cond = self.update_hidden_state_with_mask_tokens(batch, cond)

        return cond

    def dropout_cfg(self, cond: ConditioningData):
        pass

    def inverted_cosine(self, timesteps):
        return torch.arccos(torch.sqrt(timesteps))

    def forward(self, batch: InputData, state: TrainingState, cond: Optional[ConditioningData] = None):
        batch.tgt_pixel_values = torch.clamp(batch.tgt_pixel_values, -1, 1)

        if self.cfg.model.diffusion_timestep_range is not None:
            timesteps = torch.randint(self.cfg.model.diffusion_timestep_range[0], self.cfg.model.diffusion_timestep_range[1], (batch.bs,), device=self.device).long()
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch.bs,), device=self.device).long()

        if cond is None:
            cond = self.get_hidden_state(batch, add_conditioning=True)
            if self.training: self.dropout_cfg(cond)

        pred_data = None

        encoder_hidden_states = cond.encoder_hidden_states
        
        model_pred, target = None, None
        if self.cfg.model.unet and self.cfg.model.disable_unet_during_training is False:
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

            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(torch.float32), **cond.unet_kwargs).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        return self.compute_losses(batch, cond, model_pred, target, pred_data, state)

    def compute_losses(
            self,
            batch: InputData,
            cond: ConditioningData,
            model_pred: Optional[torch.FloatTensor] = None,
            target: Optional[torch.FloatTensor] = None,
            pred_data: Optional[TokenPredData] = None,
            state: Optional[TrainingState] = None,
        ):
        
        losses = dict()
        if self.cfg.model.unet and model_pred is not None and target is not None:
            mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            losses.update({"metric_diffusion_mse": mse_loss})
            losses["diffusion_loss"] = mse_loss * self.cfg.model.diffusion_loss_weight

        return losses