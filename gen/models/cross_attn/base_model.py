import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from beartype import beartype
from einx import add, rearrange, set_at, where
from jaxtyping import Bool, Float, Integer
from omegaconf import OmegaConf
from peft import LoraConfig
from torch import Tensor
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel
import itertools
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.training_utils import cast_training_params
from gen.configs import BaseConfig
from gen.models.cross_attn.break_a_scene import register_attention_control
from gen.models.cross_attn.deprecated_configs import (attention_masking, forward_shift_scale, handle_attention_masking_dropout, init_shift_scale,
                                                      shift_scale_uncond_hidden_states)
from gen.models.cross_attn.losses import break_a_scene_cross_attn_loss, break_a_scene_masked_loss, evenly_weighted_mask_loss, token_cls_loss, token_rot_loss, get_gt_rot
from gen.models.cross_attn.modules import FeatureMapper, TokenMapper
from gen.models.encoders.encoder import BaseModel, ClipFeatureExtractor
from gen.models.utils import find_true_indices_batched, positionalencoding2d
from gen.utils.decoupled_utils import get_modules
from gen.utils.diffusers_utils import load_stable_diffusion_model
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.rotation_utils import compute_rotation_matrix_from_ortho6d, get_discretized_zyx_from_quat, get_ortho6d_from_rotation_matrix, get_quat_from_discretized_zyx
from gen.utils.tokenization_utils import _get_tokens, get_uncond_tokens
from gen.utils.trainer_utils import Trainable, TrainingState
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler
from scipy.spatial.transform import Rotation as R
import einops

class InputData(TypedDict):
    gen_pixel_values: Float[Tensor, "b c h w"]
    gen_segmentation: Integer[Tensor, "b h w c"]
    disc_pixel_values: Float[Tensor, "b c h w"]
    input_ids: Integer[Tensor, "b l"]
    state: Optional[TrainingState]
    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    bs: Optional[int]
    num_frames: Optional[int]


@dataclass
class ConditioningData:
    placeholder_token: Optional[int] = None
    attn_dict: Optional[dict[str, Tensor]] = None
    clip_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None
    mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    mask_dropout: Optional[Bool[Tensor, "n"]] = None
    batch_cond_dropout: Optional[Bool[Tensor, "b"]] = None
    input_prompt: Optional[list[str]] = None
    learnable_idxs: Optional[Any] = None

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

class AttentionMetadata(TypedDict):
    layer_idx: int
    num_layers: int
    num_cond_vectors: int
    add_pos_emb: bool
    attention_mask: Optional[Float[Tensor, "b d h w"]]
    gate_scale: Optional[float]
    frozen_dim: Optional[int]
    return_attn_probs: bool
    attn_probs: Optional[Float[Tensor, "..."]]

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
        self.set_training_mode(set_grad=True) # This must be called before we set LoRA

        if self.cfg.model.unet: self.add_unet_adapters()

        from gen.models.cross_attn.base_inference import infer_batch

        BaseMapper.infer_batch = infer_batch

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_custom_models(self):
        self.mapper = FeatureMapper(cfg=self.cfg).to(self.cfg.trainer.device)
        self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder)

        if self.cfg.model.use_dataset_segmentation is False:
            from gen.models.encoders.sam import HQSam

            self.hqsam = HQSam(model_type="vit_b")

        if self.cfg.model.clip_shift_scale_conditioning:
            init_shift_scale(self)

        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            self.token_mapper = TokenMapper(cfg=self.cfg).to(self.cfg.trainer.device)

    def initialize_diffusers_models(self) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.cfg.model.revision, use_fast=False
        )

        # Load scheduler and models
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.rotation_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.model.rotation_diffusion_timestep,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=self.cfg.model.rotation_diffusion_parameterization,
        )
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfg.model.revision
        )
        

        unet_kwargs = dict()
        if self.cfg.model.gated_cross_attn:
            unet_kwargs["attention_type"] = "gated-cross-attn"
            unet_kwargs["low_cpu_mem_usage"] = False

        if self.cfg.model.unet:
            self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant
            )
            self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant, **unet_kwargs
            )
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
            self.unet.enable_gradient_checkpointing()
            if self.cfg.model.controlnet:
                self.controlnet.enable_gradient_checkpointing()
            if self.cfg.model.freeze_text_encoder is False:
                self.text_encoder.gradient_checkpointing_enable()

        if self.cfg.trainer.compile:
            self.clip: ClipFeatureExtractor = torch.compile(self.clip, mode="reduce-overhead", fullgraph=True)

            # TODO: Compile currently doesn't work with flash-attn apparently
            # self.unet.to(memory_format=torch.channels_last)
            # self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)
            # if self.cfg.model.controlnet:
            #     self.controlnet = self.controlnet.to(memory_format=torch.channels_last)
            #     self.controlnet: ControlNetModel = torch.compile(self.controlnet, mode="reduce-overhead", fullgraph=True)

        if self.cfg.model.break_a_scene_cross_attn_loss:
            from gen.models.cross_attn.break_a_scene import AttentionStore

            self.controller = AttentionStore()
            register_attention_control(self.controller, self.unet)

        elif self.cfg.model.layer_specialization:
            from gen.models.cross_attn.attn_proc import register_layerwise_attention

            num_cross_attn_layers = register_layerwise_attention(self.unet)
            assert num_cross_attn_layers == (2 * self.cfg.model.num_conditioning_pairs * (2 if self.cfg.model.gated_cross_attn else 1))
   
        if self.cfg.model.per_layer_queries:
            assert self.cfg.model.layer_specialization

    def set_training_mode(self, set_grad: bool = False):
        """
        Set training mode for the proper models and freeze/unfreeze them.

        We set the weights to weight_dtype only if they are frozen. Otherwise, they are left in FP32.

        We have the set_grad param as changing requires_grad after can cause issues. 
        
        For example, we want to first freeze the U-Net then add the LoRA adapter. We also see this error:
        `element 0 of tensors does not require grad and does not have a grad_fn`
        """
        md = self.cfg.model

        if set_grad and md.unet:
            self.vae.to(device=self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        if md.use_dataset_segmentation is False:
            self.hqsam.eval()
            if set_grad:
                self.hqsam.requires_grad_(False)

        if md.freeze_clip:
            if set_grad:
                self.clip.requires_grad_(False)
            self.clip.eval()
            log_warn("CLIP is frozen for debugging")
        else:
            if set_grad:
                self.clip.requires_grad_(True)
            self.clip.train()
            log_warn("CLIP is unfrozen")

        if md.unfreeze_last_n_clip_layers is not None:
            log_warn(f"Unfreezing last {md.unfreeze_last_n_clip_layers} CLIP layers")
            for block in self.clip.base_model.transformer.resblocks[-md.unfreeze_last_n_clip_layers :]:
                if set_grad:
                    block.requires_grad_(True)
                block.train()

        if md.unfreeze_resnet:
            for module in list(self.clip.base_model.children())[:6]:
                if set_grad:
                    module.requires_grad_(True)
                module.train()

        if md.freeze_unet:
            if set_grad:
                self.unet.to(device=self.device, dtype=self.dtype)
                self.unet.requires_grad_(False)
            self.unet.eval()
        else:
            if set_grad:
                self.unet.requires_grad_(True)
            self.unet.train()

        if md.unfreeze_gated_cross_attn:
            for m in get_modules(self.unet, BasicTransformerBlock):
                assert hasattr(m, "fuser")
                assert hasattr(m, "attn2")
                if set_grad: # TODO: Initialize weights somewhere else
                    # We copy the cross-attn weights from the frozen module to this
                    m.fuser.attn.load_state_dict(m.attn2.state_dict())
                    m.fuser.requires_grad_(True)
                    m.fuser.to(dtype=torch.float32)
                m.fuser.train()

        if md.freeze_text_encoder:
            if set_grad:
                self.text_encoder.to(device=self.device, dtype=self.dtype)
                self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            if set_grad:
                self.text_encoder.requires_grad_(True)
            if set_grad and md.freeze_text_encoder_except_token_embeddings:
                self.text_encoder.text_model.encoder.requires_grad_(False)
                self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
                self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            self.text_encoder.train()

        if md.freeze_mapper:
            if set_grad:
                self.mapper.requires_grad_(False)
            self.mapper.eval()
        else:
            if set_grad:
                self.mapper.requires_grad_(True)
            self.mapper.train()

        if md.controlnet:
            if set_grad:
                self.controlnet.requires_grad_(True)
            self.controlnet.train()

        if md.clip_shift_scale_conditioning:
            if set_grad:
                self.clip_proj_layers.requires_grad_(True)
            self.clip_proj_layers.train()

        if md.token_cls_pred_loss or md.token_rot_pred_loss:
            if set_grad:
                self.token_mapper.requires_grad_(True)
            self.token_mapper.train()

        if hasattr(self, "controller"):
            self.controller.reset()

        if hasattr(self, "pipeline"):  # After validation, we need to clear this
            del self.pipeline
            torch.cuda.empty_cache()

    def unfreeze_unet(self):
        self.cfg.model.freeze_unet = False
        self.unet.to(device=self.device, dtype=torch.float32)
        self.unet.requires_grad_(True)
        self.unet.train()
        if self.cfg.model.break_a_scene_cross_attn_loss_second_stage:
            self.cfg.model.break_a_scene_masked_loss = True

    def set_inference_mode(self, init_pipeline: bool = True):
        if init_pipeline and self.cfg.model.unet:
            self.pipeline: Union[StableDiffusionControlNetPipeline, StableDiffusionPipeline] = load_stable_diffusion_model(
                cfg=self.cfg,
                device=self.device,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
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

        self.eval()
            
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
        is_unet_trainable = self.cfg.model.unet and not self.cfg.model.freeze_unet
        return dict(self.unet.named_parameters()) if is_unet_trainable else dict()
    
    def get_param_groups(self):
        if self.cfg.model.finetune_unet_with_different_lrs:
            unet_params = self.get_unet_params()

            def get_params(params, keys):
                group = {k: v for k, v in params.items() if all([key in k for key in keys])}
                for k, p in group.items():
                    del params[k]
                return group

            return [  # Order matters here
                {"params": self.get_custom_params(), "lr": self.cfg.trainer.learning_rate * 2},
                {"params": get_params(unet_params, ("attn2",)).values(), "lr": self.cfg.trainer.learning_rate},
                {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate / 10},
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

    def get_standard_conditioning_for_inference(
        self,
        batch: InputData,
        disable_conditioning: bool = False,
        cond: Optional[ConditioningData] = None,
    ):
        # Validate input
        bs: int = batch["disc_pixel_values"].shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == bs

        assert "formatted_input_ids" not in batch

        cond = self.get_hidden_state(batch, add_conditioning=disable_conditioning is False, cond=cond)

        bs = batch["disc_pixel_values"].shape[0]
        cond.input_prompt = [[x for x in self.tokenizer.convert_ids_to_tokens(batch["formatted_input_ids"][y]) if "<|" not in x] for y in range(bs)]

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
            object_is_visible = (rearrange('b h w c -> b c (h w)', batch["gen_segmentation"]) > 0).sum(dim=-1) > 0
            rearrange('b h w c -> b c (h w)', batch["gen_segmentation"])[object_is_visible] = 1
            return batch
        elif self.cfg.model.use_dataset_segmentation:
            return batch  # We already have segmentation

        # SAM requires NumPy [0, 255]
        sam_input = rearrange("b c h w -> b h w c", (((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy())
        sam_seg = []
        bs: int = batch["disc_pixel_values"].shape[0]
        max_masks = 4

        for i in range(bs):
            masks = self.hqsam.forward(sam_input[i])
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            masks = masks[:max_masks]  # We only have 77 tokens
            original = torch.from_numpy(np.array([masks[i]["segmentation"] for i in range(len(masks))]))

            if original.shape[0] == 0:  # Make dummy mask with all ones
                original = (
                    batch["gen_segmentation"][i].new_ones((1, batch["gen_segmentation"][i].shape[0], batch["gen_segmentation"][i].shape[1])).cpu()
                )
            else:  # Add additional mask to capture any pixels that are not part of any mask
                original = torch.cat(((~original.any(dim=0))[None], original), dim=0)

            if original.shape[0] != 0 and not self.training:  # During inference, we update batch with the SAM masks for later visualization
                seg_ = original.permute(1, 2, 0).long().clone()
                sam_seg.append(torch.nn.functional.pad(seg_, (0, (max_masks + 1) - seg_.shape[-1]), "constant", 0))

        if len(sam_seg) > 0:
            batch["gen_segmentation"] = torch.stack(sam_seg, dim=0)

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
        uncond_input_ids = get_uncond_tokens(self.tokenizer).to(self.device)
        uncond_encoder_hidden_states = self.text_encoder(input_ids=uncond_input_ids[None]).last_hidden_state.to(dtype=self.dtype).squeeze(0)
        return uncond_encoder_hidden_states

    def add_cross_attn_params(self, batch, cond: ConditioningData):
        """
        This function sets up the attention parameters for the mask conditioning:
            - Gets CLIP features for K/V and flattens them
            - Updates our input tokens to be "A photo of mask_0 and mask_1" and so on
        """
        bs: int = batch["disc_pixel_values"].shape[0]
        device = batch["gen_pixel_values"].device
        dtype = self.dtype

        clip_feature_map = self.clip(batch["disc_pixel_values"].to(device=device, dtype=dtype))  # b (h w) d

        if isinstance(clip_feature_map, dict):
            for k in clip_feature_map.keys():
                if k != 'ln_post' and clip_feature_map[k].ndim == 3 and clip_feature_map[k].shape[1] == bs:
                    clip_feature_map[k] = rearrange("l b d -> b l d", clip_feature_map[k])

        if self.cfg.model.feature_map_keys is not None:
            clip_feature_map = torch.stack([clip_feature_map[k] for k in self.cfg.model.feature_map_keys], dim=0)
            clip_feature_map = rearrange("n b (h w) d -> b n (h w) d", clip_feature_map)  # (b, n, (h w), d)
        else:
            clip_feature_map = rearrange("b (h w) d -> b () (h w) d", clip_feature_map)  # (b, 1, (h w), d)

        if "resnet" not in self.clip.model_name:
            clip_feature_map = clip_feature_map[:, :, 1:, :]
            if "dino" in self.clip.model_name and "reg" in self.clip.model_name:
                clip_feature_map = clip_feature_map[:, :, 4:, :]

        latent_dim = round(math.sqrt(clip_feature_map.shape[-2]))

        if self.cfg.model.add_pos_emb:
            pos_emb = positionalencoding2d(clip_feature_map.shape[-1], latent_dim, latent_dim, device=clip_feature_map.device, dtype=clip_feature_map.dtype).to(clip_feature_map)
            clip_feature_map = add("... (h w) d, d h w -> ... (h w) d", clip_feature_map, pos_emb)

        if self.cfg.model.feature_map_keys is not None:
            clip_feature_map = add("b n (h w) d, n d -> b n (h w) d", clip_feature_map, self.mapper.position_embedding)

        feature_map_masks = []
        mask_batch_idx = []
        mask_instance_idx = []
        mask_dropout = []

        assert "gen_segmentation" in batch
        for i in range(bs):
            one_hot_mask: Bool[Tensor, "d h w"] = batch["gen_segmentation"][i].permute(2, 0, 1).bool()
            one_hot_idx = torch.arange(one_hot_mask.shape[0], device=device)

            if one_hot_mask.shape[0] == 0:
                log_warn("No masks found for this image")
                continue

            # Remove empty masks. Because of flash attention bugs, we require a minimum of 32 pixels. 14 is too few.
            non_empty_mask = torch.sum(one_hot_mask, dim=[1, 2]) > 0
            non_empty_mask[1:] &= batch['valid'][i]

            if self.cfg.model.training_mask_dropout is not None and self.training:
                dropout_mask = non_empty_mask.new_full((non_empty_mask.shape[0],), False, dtype=torch.bool)

                non_empty_idx = non_empty_mask.nonzero().squeeze(1)
                num_sel_masks = torch.randint(1, non_empty_idx.shape[0] + 1, (1,)) # We randomly take 1 to n available masks
                selected_masks = non_empty_idx[torch.randperm(len(non_empty_idx))[:num_sel_masks]] # Randomly select the subset of masks
                dropout_mask[selected_masks] = True

                if self.cfg.model.dropout_foreground_only:
                    dropout_mask[self.cfg.model.background_mask_idx] = True  # We always keep the background mask
                elif self.cfg.model.dropout_background_only:
                    dropout_mask[torch.arange(dropout_mask.shape[0]) != self.cfg.model.background_mask_idx] = True

                if dropout_mask.sum().item() == 0:
                    log_warn("We would have dropped all masks but instead we preserved the background", main_process_only=False)
                    dropout_mask[self.cfg.model.background_mask_idx] = True
            else:
                dropout_mask = one_hot_mask.new_full((one_hot_mask.shape[0],), True, dtype=torch.bool)

            combined_mask = dropout_mask & non_empty_mask
            one_hot_mask = one_hot_mask[combined_mask]
            one_hot_idx = one_hot_idx[combined_mask]

            if one_hot_mask.shape[0] == 0:
                log_warn("Dropped out all masks for this image!")
                continue

            assert batch["disc_pixel_values"].shape[-1] == batch["disc_pixel_values"].shape[-2]
            feature_map_mask_ = find_true_indices_batched(original=one_hot_mask, dh=latent_dim, dw=latent_dim)
            feature_map_masks.append(feature_map_mask_)
            mask_batch_idx.append(i * feature_map_mask_.new_ones((feature_map_mask_.shape[0]), dtype=torch.long))
            mask_instance_idx.append(one_hot_idx)
            mask_dropout.append(combined_mask)

        # If the 1st image has 5 masks and the 2nd has 3 masks, we will have an integer tensor of shape (total == 8,) for 8 different cross-attns. The sequence length for each is thus the number of valid "pixels" (KVs)
        feature_map_masks = torch.cat(feature_map_masks, dim=0)  # feature_map_mask is a boolean mask of (total, h, w)
        feature_map_masks = einops.rearrange(feature_map_masks, "total h w -> total (h w)").to(device)
        mask_batch_idx = torch.cat(mask_batch_idx, dim=0).to(device)
        mask_instance_idx = torch.cat(mask_instance_idx, dim=0).to(device)
        mask_dropout = torch.stack(mask_dropout, dim=0).to(device)

        # In the simplest case, we have one query per mask, and a single feature map. However, we support multiple queries per mask 
        # [to generate tokens for different layers] and also support each query to have K/Vs  from different feature maps, as long as all feature maps 
        # have the same dimension allowing us to assume that each mask has the same number of pixels (KVs) for each feature map.
        num_queries_per_mask = self.cfg.model.num_conditioning_pairs if self.cfg.model.per_layer_queries else 1
        num_feature_maps = clip_feature_map.shape[1]
        seqlens_k = feature_map_masks.sum(dim=-1) * num_feature_maps  # We sum the number of valid "pixels" in each mask
        seqlens_k = einops.repeat(seqlens_k, 'b -> (b layers)', layers=num_queries_per_mask)

        max_seqlen_k = seqlens_k.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0, dtype=torch.torch.int32), (1, 0))

        flat_features = einops.repeat(clip_feature_map[mask_batch_idx], "masks n hw d -> (masks layers n hw) d", layers=num_queries_per_mask)
        flat_mask = einops.repeat(feature_map_masks, "masks hw -> (masks layers n hw)", layers=num_queries_per_mask, n=num_feature_maps) # Repeat mask over layers
        k_features = flat_features[flat_mask]

        # The actual query is obtained from the mapper class (a learnable tokens)        
        cu_seqlens_q = F.pad(torch.arange(seqlens_k.shape[0]).to(torch.int32) + 1, (1, 0)).to(device)
        max_seqlen_q = 1  # We are doing attention pooling so we have one query per mask

        cond.attn_dict = dict(x_kv=k_features, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)
        cond.mask_batch_idx = mask_batch_idx
        cond.mask_instance_idx = mask_instance_idx
        cond.mask_dropout = mask_dropout
        if self.cfg.model.clip_shift_scale_conditioning:
            forward_shift_scale(self, cond, clip_feature_map, latent_dim)

        return cond

    def compute_mask_tokens(self, batch: InputData, cond: ConditioningData):
        """
        Generates mask tokens by adding segmentation if necessary, then setting up and calling the cross attention module
        """
        batch = self.check_add_segmentation(batch)
        cond = self.add_cross_attn_params(batch, cond)

        queries = einops.repeat(self.mapper.learnable_token, "layers d -> (masks layers) d", masks=cond.mask_batch_idx.shape[0])

        cond.attn_dict["x"] = queries.to(self.dtype)
        pred_mask_tokens = self.mapper.cross_attn(cond).to(self.dtype)
        cond.mask_tokens = pred_mask_tokens

        if self.cfg.model.per_layer_queries: # Break e.g., 1024 -> 16 x 64
            cond.mask_tokens = einops.rearrange(cond.mask_tokens, "(masks layers) d -> masks (layers d)", masks=cond.mask_batch_idx.shape[0])

        elif self.cfg.model.layer_specialization: # Break e.g., 1024 -> 16 x 64
            layerwise_mask_tokens = rearrange("b (layers d) -> b layers d", cond.mask_tokens, l=self.cfg.model.num_conditioning_pairs)
            layerwise_mask_tokens = self.mapper.layer_specialization(layerwise_mask_tokens)  # Batched 64 -> 1024
            cond.mask_tokens = rearrange("b layers d -> b (layers d)", layerwise_mask_tokens).to(self.dtype)

        return cond

    def update_hidden_state_with_mask_tokens(
        self,
        batch: InputData,
        cond: ConditioningData,
    ):
        bs = batch["input_ids"].shape[0]

        if self.cfg.model.layer_specialization:
            cond.encoder_hidden_states = rearrange("b t d -> b t (n d)", cond.encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)

        batch["formatted_input_ids"] = batch["input_ids"].clone()
        for b in range(bs):
            cur_ids = batch["input_ids"][b]
            token_is_padding = (cur_ids == self.pad_token_id).nonzero()  # Everything after EOS token should also be a pad token
            assert (token_is_padding.shape[0] == (batch["input_ids"].shape[1] - token_is_padding[0])).item()

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

            batch["formatted_input_ids"][b] = torch.cat((start_of_prompt, masks_prompt, end_of_prompt, eos_token), dim=0)[:cur_ids.shape[0]]

        # Overwrite mask locations
        cond.learnable_idxs = (batch["formatted_input_ids"] == self.placeholder_token_id).nonzero(as_tuple=True)
        cond.encoder_hidden_states[cond.learnable_idxs[0], cond.learnable_idxs[1]] = cond.mask_tokens.to(cond.encoder_hidden_states)

        if self.cfg.model.layer_specialization:
            cond.unet_kwargs["cross_attention_kwargs"]['attn_meta'].update(dict(
                layer_idx=0,
                num_layers=self.cfg.model.num_conditioning_pairs * 2,
                num_cond_vectors=self.cfg.model.num_conditioning_pairs,
                add_pos_emb=self.cfg.model.add_pos_emb,
            ))

            if self.cfg.model.gated_cross_attn:
                cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["gate_scale"] = 1
                with torch.no_grad():
                    frozen_enc = self.text_encoder(input_ids=_get_tokens(self.tokenizer, "A photo")[None].to(cond.encoder_hidden_states.device))[0].to(dtype=self.dtype)
                cond.encoder_hidden_states = rearrange("b t (layers d), () t d -> b t ((layers + 1) d)", cond.encoder_hidden_states, frozen_enc)
                cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["frozen_dim"] = frozen_enc.shape[-1]

        if self.cfg.model.clip_shift_scale_conditioning:
            assert not self.cfg.model.layer_specialization
            cond.encoder_hidden_states[:] = shift_scale_uncond_hidden_states(self)

        return cond

    def get_hidden_state(
        self,
        batch: InputData,
        add_conditioning: bool = True,
        cond: Optional[ConditioningData] = None,  # We can optionally specify mask tokens to use [e.g., for composing during inference]
    ) -> ConditioningData:
        if cond is None:
            cond = ConditioningData()
            cond.unet_kwargs["cross_attention_kwargs"] = dict(attn_meta=AttentionMetadata())

        cond.placeholder_token = self.placeholder_token_id
        with torch.no_grad():
            cond.encoder_hidden_states = self.text_encoder(input_ids=batch["input_ids"])[0].to(dtype=self.dtype)

        if add_conditioning:
            if cond.mask_tokens is None or cond.mask_batch_idx is None:
                cond = self.compute_mask_tokens(batch, cond)

            cond = self.update_hidden_state_with_mask_tokens(batch, cond)

            if self.cfg.model.attention_masking:
                attention_masking(batch, cond)

        return cond

    def get_controlnet_conditioning(self, batch):
        return batch["gen_segmentation"].permute(0, 3, 1, 2).to(dtype=self.dtype)

    def dropout_cfg(self, cond: ConditioningData):
        device: torch.device = cond.encoder_hidden_states.device
        frozen_dim: int = cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].get("frozen_dim", None)
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

    @beartype
    def forward(self, batch: InputData):
        batch = InputData(**batch)
        batch["device"] = self.device
        batch["dtype"] = self.dtype

        assert "formatted_input_ids" not in batch

        bs = batch["gen_pixel_values"].shape[0]
        batch["num_frames"] = 1
        batch["bs"] = bs

        if self.cfg.model.predict_rotation_from_n_frames:
            # We keep the tensor format as (bs, ...) even though it's really ((bs frames), ...) but specify num_frames
            assert bs % self.cfg.model.predict_rotation_from_n_frames == 0
            batch["num_frames"] = self.cfg.model.predict_rotation_from_n_frames

        batch["gen_pixel_values"] = torch.clamp(batch["gen_pixel_values"], -1, 1)

        # Sample a random timestep for each element in batch
        if self.cfg.model.use_inverted_noise_schedule:
            timesteps = ((torch.arccos(timesteps / self.noise_scheduler.config.num_train_timesteps) / (torch.pi / 2)) * self.noise_scheduler.config.num_train_timesteps).long()
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device).long()

        cond = self.get_hidden_state(batch, add_conditioning=True)

        if self.training: self.dropout_cfg(cond)

        if self.cfg.model.token_cls_pred_loss or self.cfg.model.token_rot_pred_loss:
            pred_data = TokenPredData()

            # We must call this before rotation/cls prediction to properly setup the GT and choose which mask tokens to predict
            # Make sure to also call token_rot_loss afterwards.
            pred_data = get_gt_rot(self.cfg, cond, batch, pred_data)
            if self.cfg.model.token_rot_pred_loss and not self.cfg.model.predict_rotation_from_n_frames:
                max_timesteps = self.cfg.model.rotation_diffusion_start_timestep if self.cfg.model.rotation_diffusion_start_timestep else self.cfg.model.rotation_diffusion_timestep
                pred_data.timesteps = torch.randint(0, max_timesteps, (pred_data.mask_tokens.shape[0],), device=self.device).long()
                pred_data.rot_6d_noise = torch.randn_like(pred_data.gt_rot_6d)
                pred_data.noised_rot_6d = self.rotation_scheduler.add_noise(pred_data.gt_rot_6d, pred_data.rot_6d_noise, pred_data.timesteps)
            
            pred_data = self.token_mapper(batch=batch, cond=cond, pred_data=pred_data)

        encoder_hidden_states = cond.encoder_hidden_states

        if self.cfg.model.unet:
            latents = self.vae.encode(batch["gen_pixel_values"].to(dtype=self.dtype)).latent_dist.sample() # Convert images to latent space
            latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents) # Sample noise that we'll add to the latents

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
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
            if self.cfg.model.break_a_scene_masked_loss:
                loss_mask = break_a_scene_masked_loss(cfg=self.cfg, batch=batch, cond=cond)
                model_pred, target = model_pred * loss_mask, target * loss_mask

            if self.cfg.model.weighted_object_loss:
                losses["diffusion_loss"] = evenly_weighted_mask_loss(cfg=self.cfg, batch=batch, cond=cond, pred=model_pred.float(), target=target.float())
            else:
                losses["diffusion_loss"] = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if self.cfg.model.break_a_scene_cross_attn_loss:
                losses["cross_attn_loss"] = break_a_scene_cross_attn_loss(cfg=self.cfg, batch=batch, controller=self.controller, cond=cond)

        if self.cfg.model.token_cls_pred_loss:
            losses.update(token_cls_loss(cfg=self.cfg, batch=batch, cond=cond, pred_data=pred_data))

        if self.cfg.model.token_rot_pred_loss:
            losses.update(token_rot_loss(cfg=self.cfg, pred_data=pred_data))

        return losses
