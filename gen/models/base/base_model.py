import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einx import add, rearrange, where

from diffusers.training_utils import EMAModel, cast_training_params
from gen.configs import BaseConfig
from gen.models.base.base_defs import AttentionMetadata, ConditioningData
from gen.models.base.base_setup import (add_unet_adapters, checkpoint, initialize_custom_models, initialize_diffusers_models, set_inference_mode,
                                        set_training_mode)
from gen.models.dustr.depth_utils import decode_xyz, encode_xyz
from gen.models.encoders.encoder import BaseModel
from gen.utils.data_defs import InputData
from gen.utils.logging_utils import log_debug, log_error, log_info, log_warn
from gen.utils.trainer_utils import Trainable, TrainingState, unwrap
from gen.utils.visualization_utils import viz_feats
from image_utils import Im


class BaseMapper(Trainable):
    def __init__(self, cfg: BaseConfig):
        super().__init__()

        self.cfg: BaseConfig = cfg

        # dtype of most intermediate tensors and frozen weights. Notably, we always use FP32 for trainable params.
        self.dtype = getattr(torch, cfg.trainer.dtype.split(".")[-1]) if isinstance(cfg.trainer.dtype, str) else cfg.trainer.dtype

        self.initialize_diffusers_models()
        self.initialize_custom_models()
        set_training_mode(cfg=self.cfg, _other=self, dtype=self.dtype, device=self.device, set_grad=True)

        if self.cfg.model.unet: self.add_unet_adapters()
        if self.cfg.trainer.compile:
            log_info("Using torch.compile()...")
            if hasattr(self, "clip"):
                self.clip = torch.compile(self.clip, mode="reduce-overhead", fullgraph=True)
        
            # TODO: Compile currently doesn't work with flash-attn apparently
            # self.unet.to(memory_format=torch.channels_last)
            # self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)

        from gen.models.base.base_inference import infer_batch
        BaseMapper.infer_batch = infer_batch

    def initialize_diffusers_models(self): initialize_diffusers_models(self)
    def initialize_custom_models(self): initialize_custom_models(self)
    def add_unet_adapters(self): add_unet_adapters(self)
    def checkpoint(self, **kwargs): checkpoint(self, **kwargs)
    def set_training_mode(self, **kwargs): set_training_mode(**kwargs)
    def set_inference_mode(self, **kwargs): set_inference_mode(self, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    def on_sync_gradients(self, state: TrainingState):
        if self.cfg.model.unet and not self.cfg.model.freeze_unet and self.cfg.model.ema:
            self.ema_unet.step(self.unet.parameters())

    def process_input(self, batch: dict, state: TrainingState) -> InputData:
        if not isinstance(batch, InputData):
            batch: InputData = InputData.from_dict(batch)

        batch.state = state
        batch.dtype = self.dtype
        return batch

    def get_feature_map(self, batch: InputData, cond: ConditioningData):
        bs: int = batch.src_enc_rgb.shape[0]
        device = batch.src_enc_rgb.device
        dtype = self.dtype

        enc_input = batch.src_pixel_values.to(device=device, dtype=dtype)

        if self.cfg.model.mask_dropped_tokens:
            for b in range(bs):
                dropped_mask = ~torch.isin(batch.src_segmentation[b], cond.mask_instance_idx[cond.mask_batch_idx == b]).any(dim=-1)
                enc_input[b, :, dropped_mask] = 0

        if batch.attach_debug_info:
            cond.encoder_input_pixel_values = enc_input.clone()

        elif self.cfg.model.stock_dino_v2:
            with torch.no_grad():
                enc_feature_map = {f'blocks.{i}':v for i,v in enumerate(self.clip.get_intermediate_layers(x=enc_input, n=24 if 'large' in self.cfg.model.encoder.model_name else 12, norm=True))}
        else:
            with torch.no_grad() if self.cfg.model.freeze_enc and self.cfg.model.unfreeze_last_n_enc_layers is None else nullcontext():
                enc_feature_map = self.clip.forward_model(enc_input)  # b (h w) d
                if self.cfg.model.norm_vit_features:
                    for k in enc_feature_map.keys():
                        if "blocks" in k:
                            enc_feature_map[k] = self.clip.base_model.norm(enc_feature_map[k])

        if self.cfg.model.debug_feature_maps and batch.attach_debug_info:            
            orig_trained = copy.deepcopy(self.clip.state_dict())
            trained_viz = viz_feats(enc_feature_map, "trained_feature_map")
            self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False).to(self.dtype).to(self.device)
            enc_feature_map = self.clip.forward_model(enc_input)
            stock_viz = viz_feats(enc_feature_map, "stock_feature_map")
            Im.concat_vertical(stock_viz, trained_viz).save(batch.metadata['name'][0])
            self.clip.load_state_dict(orig_trained)

        if isinstance(enc_feature_map, dict):
            for k in enc_feature_map.keys():
                if k != 'ln_post' and enc_feature_map[k].ndim == 3 and enc_feature_map[k].shape[1] == bs:
                    enc_feature_map[k] = rearrange("l b d -> b l d", enc_feature_map[k])

        if self.cfg.model.feature_map_keys is not None:
            enc_feature_map = torch.stack([enc_feature_map[k] for k in self.cfg.model.feature_map_keys], dim=0)
            enc_feature_map = rearrange("n b (h w) d -> b n (h w) d", enc_feature_map)  # (b, n, (h w), d)
        else:
            enc_feature_map = rearrange("b (h w) d -> b () (h w) d", enc_feature_map)  # (b, 1, (h w), d)

        enc_feature_map = enc_feature_map.to(self.dtype)
        latent_dim = self.cfg.model.encoder_latent_dim

        if enc_feature_map.shape[-2] != latent_dim**2 and "resnet" not in self.cfg.model.encoder.model_name:
            enc_feature_map = enc_feature_map[:, :, 1:, :]
            if "dino" in self.cfg.model.encoder.model_name and "reg" in self.cfg.model.encoder.model_name:
                enc_feature_map = enc_feature_map[:, :, 4:, :]

        if batch.attach_debug_info:
            cond.src_feature_map = rearrange("b n (h w) d -> b n h w d", enc_feature_map.clone(), h=latent_dim, w=latent_dim)

        return enc_feature_map

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

        return cond

    def dropout_cfg(self, cond: ConditioningData):
        pass

    def inverted_cosine(self, timesteps):
        return torch.arccos(torch.sqrt(timesteps))

    def forward(self, batch: InputData, state: TrainingState, cond: Optional[ConditioningData] = None):
        batch.src_dec_rgb = torch.clamp(batch.src_dec_rgb, -1, 1)
        batch.tgt_dec_rgb = torch.clamp(batch.tgt_dec_rgb, -1, 1)

        if self.cfg.model.diffusion_timestep_range is not None:
            timesteps = torch.randint(self.cfg.model.diffusion_timestep_range[0], self.cfg.model.diffusion_timestep_range[1], (batch.bs,), device=self.device).long()
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch.bs,), device=self.device).long()

        if self.cfg.model.duplicate_unet_input_channels:
            timesteps = einops.repeat(timesteps, 'b -> (n b)', n=2)

        if cond is None:
            cond = self.get_hidden_state(batch, add_conditioning=True)
            if self.training: self.dropout_cfg(cond)

        pred_data = None
        encoder_hidden_states = cond.encoder_hidden_states
        
        model_pred, target = None, None
        if self.cfg.model.unet and self.cfg.model.disable_unet_during_training is False:
            rgb_to_encode = torch.cat([batch.src_dec_rgb, batch.tgt_dec_rgb], dim=0).to(dtype=self.dtype)
            latents = self.vae.encode(rgb_to_encode).latent_dist.sample() # Convert images to latent space
            latents = latents * self.vae.config.scaling_factor

            if self.cfg.model.duplicate_unet_input_channels:
                input_xyz = rearrange('b h w xyz, b h w xyz -> (b + b) h w xyz', batch.src_xyz, batch.tgt_xyz)
                input_valid = rearrange('b h w, b h w -> (b + b) h w', batch.src_xyz_valid, batch.tgt_xyz_valid)
                xyz_latents, xyz_valid, normalizer = encode_xyz(input_xyz, input_valid, self.vae)
                cond.xyz_normalizer = normalizer
                cond.xyz_valid = xyz_valid

                rgb_latents = latents
                latents = xyz_latents

            noise = torch.randn_like(latents) # Sample noise that we'll add to the latents

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            attn_meta = cond.unet_kwargs.get('cross_attention_kwargs', {}).get('attn_meta', None)
            if attn_meta is None:
                if 'cross_attention_kwargs' in cond.unet_kwargs and 'attn_meta' in cond.unet_kwargs['cross_attention_kwargs']:
                    del cond.unet_kwargs['cross_attention_kwargs']['attn_meta']

            if self.cfg.model.duplicate_unet_input_channels:
                noisy_latents = torch.cat([rgb_latents, noisy_latents], dim=1)

            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(torch.float32), **cond.unet_kwargs).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        return self.compute_losses(batch=batch, cond=cond, model_pred=model_pred, target=target, state=state)

    def compute_losses(
            self,
            batch: InputData,
            cond: ConditioningData,
            model_pred: Optional[torch.FloatTensor] = None,
            target: Optional[torch.FloatTensor] = None,
            state: Optional[TrainingState] = None,
        ):
        
        losses = dict()
        if self.cfg.model.unet and model_pred is not None and target is not None:
            mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.update({"metric_diffusion_mse": mse_loss})
            losses["diffusion_loss"] = mse_loss * self.cfg.model.diffusion_loss_weight

        if self.cfg.model.duplicate_unet_input_channels:
            out = decode_xyz(model_pred, cond.xyz_valid, self.vae, cond.xyz_normalizer)
            breakpoint()

        return losses