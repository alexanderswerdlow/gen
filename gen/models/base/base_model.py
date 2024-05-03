import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

import einops
import hydra
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einx import add, rearrange, where, mean

from diffusers.training_utils import EMAModel, cast_training_params
from gen.configs import BaseConfig
from gen.models.base.base_defs import AttentionMetadata, ConditioningData
from gen.models.base.base_setup import (add_unet_adapters, checkpoint, initialize_custom_models, initialize_diffusers_models, set_inference_mode,
                                        set_training_mode)
from gen.models.dustr.depth_utils import encode_xyz, get_xyz_input
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
                self.encoder = torch.compile(self.encoder, mode="reduce-overhead", fullgraph=True)
        
            # TODO: Compile currently doesn't work with flash-attn apparently
            # self.unet.to(memory_format=torch.channels_last)
            # self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)

        from gen.models.base.base_inference import infer_batch
        BaseMapper.infer_batch = infer_batch

    def initialize_diffusers_models(self): initialize_diffusers_models(self)
    def initialize_custom_models(self): initialize_custom_models(self)
    def add_unet_adapters(self): add_unet_adapters(self)
    def checkpoint(self, *args, **kwargs): checkpoint(self, *args, **kwargs)
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
                enc_feature_map = {f'blocks.{i}':v for i,v in enumerate(self.encoder.get_intermediate_layers(x=enc_input, n=24 if 'large' in self.cfg.model.encoder.model_name else 12, norm=True))}
        else:
            with torch.no_grad() if self.cfg.model.freeze_enc and self.cfg.model.unfreeze_last_n_enc_layers is None else nullcontext():
                enc_feature_map = self.encoder.forward_model(enc_input)  # b (h w) d
                if self.cfg.model.norm_vit_features:
                    for k in enc_feature_map.keys():
                        if "blocks" in k:
                            enc_feature_map[k] = self.encoder.base_model.norm(enc_feature_map[k])

        if self.cfg.model.debug_feature_maps and batch.attach_debug_info:            
            orig_trained = copy.deepcopy(self.encoder.state_dict())
            trained_viz = viz_feats(enc_feature_map, "trained_feature_map")
            self.encoder: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder, compile=False).to(self.dtype).to(self.device)
            enc_feature_map = self.encoder.forward_model(enc_input)
            stock_viz = viz_feats(enc_feature_map, "stock_feature_map")
            Im.concat_vertical(stock_viz, trained_viz).save(batch.metadata['name'][0])
            self.encoder.load_state_dict(orig_trained)

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
        cond: Optional[ConditioningData] = None,  # We can optionally specify mask tokens to use [e.g., for composing during inference]
    ) -> ConditioningData:
        if cond is None:
            cond = ConditioningData()
        
        if len(cond.unet_kwargs) == 0:
            cond.unet_kwargs["cross_attention_kwargs"] = dict(attn_meta=AttentionMetadata())

        if self.cfg.model.joint_attention:
            cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"].joint_attention = batch.n
        
        cond.encoder_hidden_states = self.uncond_hidden_states[None].repeat(batch.bs, 1, 1)

        return cond

    def dropout_cfg(self, cond: ConditioningData):
        pass

    def get_rgb_input(self, batch: InputData):
        if batch.dec_rgb is None:
            rgb_to_encode = torch.cat([batch.src_dec_rgb, batch.tgt_dec_rgb], dim=0)
        else:
            rgb_to_encode = rearrange('b n c h w -> (n b) c h w', batch.dec_rgb)
        return rgb_to_encode
    
    def get_rgb_latents(self, batch: InputData):
        with torch.no_grad():
            rgb_to_encode = self.get_rgb_input(batch)
            rgb_to_encode = rgb_to_encode.to(dtype=self.dtype if self.cfg.model.force_fp32_pcd_vae is False else torch.float32, device=self.device)
            rgb_latents = self.vae.encode(rgb_to_encode).latent_dist.sample() # Convert images to latent space
            rgb_latents = rgb_latents * self.vae.config.scaling_factor
        return rgb_latents
    
    @property
    def uncond_hidden_states(self):
        return torch.zeros((self.cfg.model.num_decoder_cross_attn_tokens, self.cfg.model.token_embedding_dim), dtype=self.dtype, device=self.device)

    def inverted_cosine(self, timesteps):
        return torch.arccos(torch.sqrt(timesteps))
    
    def clamp_input(self, arr):
        return torch.clamp(arr, -1, 1) if arr is not None else arr

    def forward(self, batch: InputData, state: TrainingState, cond: Optional[ConditioningData] = None):
        batch.src_dec_rgb = self.clamp_input(batch.src_dec_rgb)
        batch.tgt_dec_rgb = self.clamp_input(batch.tgt_dec_rgb)
        batch.dec_rgb = self.clamp_input(batch.dec_rgb)

        if self.cfg.model.diffusion_timestep_range is not None:
            timesteps = torch.randint(self.cfg.model.diffusion_timestep_range[0], self.cfg.model.diffusion_timestep_range[1], (batch.bs,), device=self.device).long()
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch.bs,), device=self.device).long()

        if self.cfg.model.duplicate_unet_input_channels:
            timesteps = rearrange('b -> (n b)', timesteps, n=batch.n)

        if cond is None:
            cond = self.get_hidden_state(batch)
            if self.training: self.dropout_cfg(cond)

        cond.timesteps = timesteps
        encoder_hidden_states = cond.encoder_hidden_states

        if self.cfg.model.duplicate_unet_input_channels:
            input_xyz, input_valid = get_xyz_input(self.cfg, batch)
            xyz_latents, xyz_valid, normalizer = encode_xyz(self.cfg, input_xyz, input_valid, self.vae, self.dtype, batch.n, kwargs=dict(per_axis_quantile=self.cfg.model.predict_depth is False))
            
            cond.xyz_normalizer = normalizer
            cond.xyz_valid = xyz_valid
            cond.gt_xyz = input_xyz

        model_pred, target = None, None
        if self.cfg.model.unet and self.cfg.model.disable_unet_during_training is False:
            rgb_latents = self.get_rgb_latents(batch)

            latents = xyz_latents if self.cfg.model.duplicate_unet_input_channels else rgb_latents
            if self.cfg.model.only_noise_tgt:
                assert self.cfg.model.n_view_pred is False
                set_to_zero = torch.rand(timesteps.shape[0]) > self.cfg.model.dropout_src_depth
                set_to_zero[timesteps.shape[0] // 2:] = False
                if self.cfg.model.dropout_src_depth is None:
                    set_to_zero[:] = False
                timesteps[set_to_zero] = torch.randint(0, self.noise_scheduler.config.num_train_timesteps // 5, (set_to_zero.sum().item(),), device=self.device).long()

            noise = torch.randn_like(latents) # Sample noise that we'll add to the latents

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

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
            
        if self.cfg.model.unfreeze_vae_decoder:
            to_decode_xyz_latents = (1 / self.vae.config.scaling_factor) * xyz_latents[:self.cfg.model.vae_decoder_batch_size]
            if self.cfg.model.separate_xyz_encoding:
                to_decode_xyz_latents = rearrange('b (xyz c) h w -> (b xyz) c h w', to_decode_xyz_latents, xyz=3)
            with torch.autocast(device_type="cuda", enabled=self.cfg.model.force_fp32_pcd_vae is False):
                gt_decoder_xyz = self.vae.decode(to_decode_xyz_latents.to(torch.float32).detach(), return_dict=False)[0]
                if self.cfg.model.separate_xyz_encoding:
                    gt_decoder_xyz = mean('(b xyz) [c] h w -> b xyz h w', gt_decoder_xyz, xyz=3)
                gt_decoder_xyz, gt_decoder_valid = normalizer.denormalize(gt_decoder_xyz)
                cond.gt_decoder_xyz = gt_decoder_xyz
                cond.gt_decoder_valid = gt_decoder_valid
        
        return self.compute_losses(
            batch=batch,
            cond=cond,
            model_pred=model_pred,
            target=target,
            state=state
        )

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
            if self.cfg.model.use_valid_xyz_loss_mask:
                loss_mask = F.max_pool2d(rearrange('b h w -> b c h w', cond.xyz_valid.float(), c=model_pred.shape[1]), kernel_size=self.cfg.model.decoder_resolution // self.cfg.model.decoder_latent_dim) > 0.5
            else:
                loss_mask = None

            if self.cfg.model.snr_gamma is None:
                mse_loss = F.mse_loss(model_pred.float()[loss_mask], target.float()[loss_mask], reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                from diffusers.training_utils import compute_snr
                snr = compute_snr(self.noise_scheduler, cond.timesteps)
                mse_loss_weights = torch.stack([snr, self.cfg.model.snr_gamma * torch.ones_like(cond.timesteps)], dim=1).min(
                    dim=1
                )[0]
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                
                if self.cfg.model.use_valid_xyz_loss_mask:
                    mse_loss = []
                    for b in range(model_pred.shape[0]):
                        _loss = F.mse_loss(model_pred[b].float()[loss_mask[b]], target[b].float()[loss_mask[b]], reduction="mean") * mse_loss_weights[b]
                        if not torch.isnan(_loss): mse_loss.append(_loss)
                        
                    mse_loss = torch.stack(mse_loss).mean()
                else:
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape)))) * mse_loss_weights
                    mse_loss = mse_loss.mean()
                
            losses.update({"metric_diffusion_mse": mse_loss})
            losses["diffusion_loss"] = mse_loss * self.cfg.model.diffusion_loss_weight
            if not torch.isfinite(mse_loss): breakpoint()

        if self.cfg.model.unfreeze_vae_decoder:
            # losses['l2_scale_shift_inv_decoder_loss_v2'] = get_dustr_loss(batch, cond.gt_decoder_xyz, cond.gt_decoder_valid)
            losses['vae_decoder_mse_loss'] = 1e-3 * F.mse_loss(cond.gt_decoder_xyz.float()[cond.gt_decoder_valid], cond.gt_xyz.float()[:self.cfg.model.vae_decoder_batch_size][cond.gt_decoder_valid], reduction="mean")


        return losses