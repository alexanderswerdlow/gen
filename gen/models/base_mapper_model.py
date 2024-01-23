import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from transformers import CLIPTokenizer

from gen.configs import BaseConfig
from transformers.models.clip.modeling_clip import CLIPTextModel
from gen.utils.encoder_utils import ClipFeatureExtractor, TimmModel, BaseModel
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.trainer_utils import unwrap
from gen.models.utils import Trainable, _init_weights, find_true_indices_batched
from gen.models.conditioning_models import CrossAttn
from jaxtyping import Float
from torch import Tensor
import torch._dynamo

class Mapper(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.learnable_token = nn.Parameter(torch.randn(cfg.model.cross_attn_dim))
        self.cross_attn = CrossAttn(cfg=cfg, input_dim=self.cfg.model.cross_attn_dim, output_dim=cfg.model.token_embedding_dim)
        self.apply(_init_weights)

class BaseMapper(Trainable):
    def __init__(self, cfg: BaseConfig, init_modules: bool = True):
        super().__init__()
        self.cfg: BaseConfig = cfg
        self.weight_dtype = getattr(torch, cfg.trainer.dtype.split(".")[-1])
        self.module_device = cfg.trainer.device
        if init_modules:
            self.initialize_diffusers_models()

        self.initialize_custom_models()

    def initialize_custom_models(self):
        self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, _recursive_=True, num_from_back=3, tensor_input=True) # , compile=self.cfg.trainer.compile

        if self.cfg.model.use_dataset_segmentation is False:
            from gen.models.sam import HQSam
            self.hqsam = HQSam(model_type="vit_b")

        self.mapper = Mapper(cfg=self.cfg)

    def initialize_diffusers_models(self) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")

        # Load scheduler and models
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfg.model.revision
        )

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant
        )
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant
        )

        if self.cfg.model.controlnet:
            self.controlnet: ControlNetModel = ControlNetModel.from_unet(self.unet, conditioning_channels=2)

    def add_adapters(self):
        self.vae.to(device=self.module_device, dtype=self.weight_dtype)
        self.unet.to(device=self.module_device, dtype=self.weight_dtype)
        self.text_encoder.to(device=self.module_device, dtype=self.weight_dtype)
        if self.cfg.model.controlnet: self.controlnet.to(device=self.module_device, dtype=self.weight_dtype)

        self.set_training_mode(set_grad=True)

        assert not (self.cfg.model.freeze_unet is False and (self.cfg.model.unfreeze_unet_after_n_steps is not None or self.cfg.model.lora_unet))
        if self.cfg.model.lora_unet:
            from peft import LoraConfig

            unet_lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)

        if self.cfg.trainer.enable_xformers_memory_efficient_attention:
            import xformers

            self.unet.enable_xformers_memory_efficient_attention()
            if self.cfg.model.controlnet:
                self.controlnet.enable_xformers_memory_efficient_attention()

        if self.cfg.trainer.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.cfg.model.controlnet:
                self.controlnet.enable_gradient_checkpointing()

        if self.cfg.trainer.compile:
            self.clip: ClipFeatureExtractor = torch.compile(self.clip, mode="reduce-overhead", fullgraph=True)
            
            # TODO: Compile currently doesn't work with flash-attn apparently
            # self.unet.to(memory_format=torch.channels_last)
            # self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)
            # if self.cfg.model.controlnet:
            #     self.controlnet = self.controlnet.to(memory_format=torch.channels_last)
            #     self.controlnet: ControlNetModel = torch.compile(self.controlnet, mode="reduce-overhead", fullgraph=True)

    def set_training_mode(self, set_grad: bool = False):
        """
        Set training mode for the proper models and freeze/unfreeze them.
        We have the set_grad param as it appears that setting requires_grad after training has started can cause:

        `element 0 of tensors does not require grad and does not have a grad_fn`
        """
        if set_grad: self.vae.requires_grad_(False)

        if self.cfg.model.use_dataset_segmentation is False:
            self.hqsam.eval()
            if set_grad: self.hqsam.requires_grad_(False)

        if self.cfg.model.freeze_clip:
            if set_grad: self.clip.requires_grad_(False)
            self.clip.eval()
            log_warn("CLIP is frozen for debugging")
        else:
            if set_grad: self.clip.requires_grad_(True)
            self.clip.train()
            log_warn("CLIP is unfrozen")

        if self.cfg.model.unfreeze_last_n_clip_layers is not None:
            log_warn(f"Unfreezing last {self.cfg.model.unfreeze_last_n_clip_layers} CLIP layers")
            for block in self.clip.base_model.transformer.resblocks[-self.cfg.model.unfreeze_last_n_clip_layers :]:
                block.requires_grad_(True)

        if self.cfg.model.freeze_unet:
            if set_grad: self.unet.requires_grad_(False)
            self.unet.eval()
        else:
            if set_grad: self.unet.requires_grad_(True)
            self.unet.train()

        if self.cfg.model.freeze_text_encoder:
            if set_grad: self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            if set_grad: self.text_encoder.requires_grad_(True)
            self.text_encoder.train()

        if self.cfg.model.freeze_mapper:
            if set_grad: self.mapper.requires_grad_(False)
            self.mapper.eval()
        else:
            if set_grad: self.mapper.requires_grad_(True)
            self.mapper.train()

        if self.cfg.model.controlnet:
            if set_grad: self.controlnet.requires_grad_(True)
            self.controlnet.train()

    def set_inference_mode(self):
        # TODO: Is there ever a case where we only want to set some submodules to eval?
        self.eval()

    def get_standard_conditioning_for_inference(
        self,
        batch: dict,
        disable_conditioning: bool = False,
    ):
        hidden_state = self.get_hidden_state(batch, add_mask_conditioning=disable_conditioning is False)
        bs = batch["disc_pixel_values"].shape[0]
        input_prompt = [[x for x in self.tokenizer.convert_ids_to_tokens(batch["input_ids"][y]) if "<|" not in x] for y in range(bs)]
        pipeline_kwargs = dict(prompt_embeds=hidden_state)
        if self.cfg.model.controlnet:
            pipeline_kwargs["image"] = self.get_controlnet_conditioning(batch)

        return pipeline_kwargs, input_prompt
    
    def check_add_segmentation(self, batch: dict):
        """
        This function checks if we have segmentation for the current batch. If we do not, we add dummy segmentation or use HQSam to get segmentation.
        """
        if self.cfg.model.use_dataset_segmentation:
            return batch # We already have segmentation
        
        if self.cfg.model.encode_token_without_tl:
            original = batch["gen_segmentation"][i].new_ones((1, batch["gen_segmentation"][i].shape[0], batch["gen_segmentation"][i].shape[1]))
            assert False

        bs: int = batch["disc_pixel_values"].shape[0]

        sam_input = rearrange(
            (((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), "b c h w -> b h w c"
        )  # SAM requires NumPy [0, 255]

        gen_segmentations = []
        for i in range(bs):
            masks = self.hqsam.forward(sam_input[i])
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            max_masks = 4
            masks = masks[:max_masks]  # We only have 77 tokens
            original = torch.from_numpy(np.array([masks[i]["segmentation"] for i in range(len(masks))]))
            if original.shape[0] == 0:
                original = (
                    batch["gen_segmentation"][i].new_ones((1, batch["gen_segmentation"][i].shape[0], batch["gen_segmentation"][i].shape[1])).cpu()
                )
            else:
                # Add additional mask to capture any pixels that are not part of any mask
                original = torch.cat((original, (~original.any(dim=0))[None]), dim=0)
            if original.shape[0] != 0 and not self.training:
                gen_segmentation_ = original.permute(1, 2, 0).long().clone()
                gen_segmentations.append(
                    torch.nn.functional.pad(gen_segmentation_, (0, (max_masks + 1) - gen_segmentation_.shape[-1]), "constant", 0)
                )

        if len(gen_segmentations) > 0:
            batch["gen_segmentation"] = torch.stack(gen_segmentations, dim=0)

        return batch

    def get_mask_attn_params(self, batch):
        """
        This function sets up the attention parameters for the mask conditioning:
            - Gets CLIP features for K/V and flattens them
            - Updates our input tokens to be "A photo of mask_0 and mask_1" and so on
        """
        bs: int = batch["disc_pixel_values"].shape[0]
        device = batch["gen_pixel_values"].device
        dtype = self.weight_dtype
        
        text_encoder_dict = dict()
        clip_feature_cls_token = None

        # isinstance fails with torch dynamo
        if "resnet" in self.clip.model_name:
            clip_feature_map = self.clip(((batch["gen_pixel_values"] + 1) / 2).to(device=device, dtype=dtype))
            clip_feature_map = rearrange(clip_feature_map, "b d h w -> b (h w) d")
            clip_feature_cls_token = torch.mean(clip_feature_map, dim=1)
        else:
            clip_feature_map = self.clip(batch["disc_pixel_values"].to(device=device, dtype=dtype)).permute(1, 0, 2)
            clip_feature_map = rearrange(clip_feature_map, "l b d -> b l d")
            clip_feature_map = clip_feature_map[:, 1:, :]

            if self.cfg.model.use_cls_token_projected:
                clip_feature_cls_token = self.clip.forward_base_model(batch["disc_pixel_values"].to(device=device, dtype=dtype))
            elif self.cfg.model.use_cls_token_final_layer:
                clip_feature_cls_token = clip_feature_map[:, -1, :]
            elif self.cfg.model.use_cls_token_mean:
                clip_feature_cls_token = torch.mean(clip_feature_map, dim=1)

        latent_dim = int(math.sqrt(clip_feature_map.shape[1]))
        feature_map_masks = []
        feature_map_batch_idxs = []
        
        assert "gen_segmentation" in batch
        for i in range(bs):
            original = batch["gen_segmentation"][i].permute(2, 0, 1).bool()

            if original.shape[0] == 0:
                log_info("Warning, no masks found for this image")
                continue

            if self.cfg.model.dropout_masks is not None and self.training:
                mask = torch.rand(original.size(0)) > self.cfg.model.dropout_masks
                mask[0] = True  # We always keep the background mask
                original = original[mask]

            original = original[torch.sum(original, dim=[1, 2]) > 0]  # Remove empty masks

            if original.shape[0] == 0:
                log_info("Warning, no masks found for this image")
                continue

            assert batch["disc_pixel_values"].shape[-1] == batch["disc_pixel_values"].shape[-2]
            feature_map_mask_ = find_true_indices_batched(original=original, dh=latent_dim, dw=latent_dim)
            feature_map_masks.append(feature_map_mask_)
            feature_map_batch_idxs.append(i * feature_map_mask_.new_ones((feature_map_mask_.shape[0]), dtype=torch.long))

        # If the 1st image has 5 masks and the 2nd has 3 masks, we will have an integer tensor of shape (total == 8,) for 8 different cross-attns. The sequence length for each is thus the number of valid "pixels" (KVs)
        feature_map_masks = torch.cat(feature_map_masks, dim=0)  # feature_map_mask is a boolean mask of (total, h, w)
        feature_map_masks = rearrange(feature_map_masks, "total h w -> total (h w)").to(device)
        feature_map_batch_idxs = torch.cat(feature_map_batch_idxs, dim=0).to(device)

        # We sum the number of valid "pixels" in each mask
        seqlens_k = feature_map_masks.sum(dim=-1)  # (total,)
        max_seqlen_k = seqlens_k.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0, dtype=torch.torch.int32), (1, 0))

        flat_features = rearrange(clip_feature_map[feature_map_batch_idxs], "total (h w) d -> (total h w) d", h=latent_dim, w=latent_dim)
        flat_mask = rearrange(feature_map_masks, "total (h w) -> (total h w)", h=latent_dim, w=latent_dim)
        k_features = flat_features[flat_mask]

        # The actual query is obtained from the timestep + layer encoding later
        cu_seqlens_q = F.pad(torch.arange(seqlens_k.shape[0]).to(torch.int32) + 1, (1, 0)).to(device)
        max_seqlen_q = 1  # We are doing attention pooling so we have one query per mask

        attn_dict = dict(x_kv=k_features, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)

        placeholder_token_id = self.tokenizer(self.cfg.model.placeholder_token, add_special_tokens=False).input_ids[0]
        mask_tokens_ids = self.tokenizer(f"{self.cfg.model.placeholder_token} and", add_special_tokens=False).input_ids
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer._pad_token)

        bs = batch["input_ids"].shape[0]
        for b in range(bs):
            # Everything after 1st pad token should also be a pad token
            token_is_padding = (batch["input_ids"][b] == pad_token_id).nonzero()
            assert (token_is_padding.shape[0] == (batch["input_ids"].shape[1] - token_is_padding[0])).item()
            mask_part_of_batch = (feature_map_batch_idxs == b).nonzero().squeeze(1)
            assert token_is_padding.shape[0] >= mask_part_of_batch.shape[0]  # We need at least as many pad tokens as we have masks
            start_loc = token_is_padding[0]
            number_of_added_tokens = (mask_part_of_batch.shape[0] * len(mask_tokens_ids)) - 1
            replace_sl = slice(start_loc, start_loc + number_of_added_tokens)
            extent = len(batch["input_ids"][b, replace_sl])
            repeated_tensor = torch.tensor(mask_tokens_ids * mask_part_of_batch.shape[0])[:extent]
            batch["input_ids"][b, replace_sl] = repeated_tensor

        text_encoder_dict = dict(
            attn_dict=attn_dict,
            placeholder_token=placeholder_token_id,
            pad_token=pad_token_id,
            feature_map_batch_idxs=feature_map_batch_idxs,
            clip_feature_cls_token=clip_feature_cls_token,
        )

        return text_encoder_dict
    
    def add_mask_conditioning(self, batch: dict, encoder_hidden_states: Float[Tensor, "b d"]):
        batch = self.check_add_segmentation(batch)
        text_encoder_dict = self.get_mask_attn_params(batch)
        queries = self.mapper.learnable_token[None, :].repeat(text_encoder_dict['feature_map_batch_idxs'].shape[0], 1)
        text_encoder_dict['attn_dict']['x'] = queries.to(self.weight_dtype)
        output = self.mapper.cross_attn(**text_encoder_dict).to(self.weight_dtype)

        # Overwrite mask locations
        learnable_idxs = (batch["input_ids"] == text_encoder_dict["placeholder_token"]).nonzero(as_tuple=True)
        encoder_hidden_states[learnable_idxs[0], learnable_idxs[1]] = output
        return encoder_hidden_states
    
    def get_hidden_state(self, batch: dict, add_mask_conditioning: bool = True) -> Float[Tensor, "b d"]:
        encoder_hidden_states = self.text_encoder(input_ids=batch["input_ids"])[0].to(dtype=self.weight_dtype)

        if add_mask_conditioning:
            encoder_hidden_states = self.add_mask_conditioning(batch, encoder_hidden_states)

        return encoder_hidden_states
    
    def get_controlnet_conditioning(self, batch):
        return batch["gen_segmentation"].permute(0, 3, 1, 2).to(dtype=self.weight_dtype)

    def forward(self, batch: dict):
        batch["gen_pixel_values"] = torch.clamp(batch["gen_pixel_values"], -1, 1)

        # Convert images to latent space
        latents = self.vae.encode(batch["gen_pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        encoder_hidden_states = self.get_hidden_state(batch, add_mask_conditioning=self.cfg.model.mask_cross_attn)

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
                down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
            ).sample
        else:
            # Predict the noise residual
            # TODO: We shouldn't need to cast to FP32 here
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(torch.float32)).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss
