from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from gen.configs import BaseConfig
from gen.configs.models import ModelConfig
from gen.models.neti.net_clip_text_embedding import NeTIBatch
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.neti_mapper import UNET_LAYERS, NeTIMapper
from gen.models.neti.xti_attention_processor import XTIAttenProc
from gen.models.sam import HQSam, find_true_indices_batched
from gen.utils.encoder_utils import ClipFeatureExtractor

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class BaseMapper(nn.Module):
    def __init__(self, cfg: BaseConfig, init_modules: bool = True):
        super(BaseMapper, self).__init__()
        self.cfg: BaseConfig = cfg
        if init_modules:
            self.get_base_mapper_model()

        self.init_pretrained()

    def init_pretrained(self):
        self.clip = ClipFeatureExtractor()
        self.clip.train()

        if self.cfg.model.freeze_clip:
            self.clip.requires_grad_(False)
            warnings.warn('Warning, CLIP is frozen for debugging!!!!!')
        
        if self.cfg.model.unfreeze_last_n_clip_layers is not None:
            for block in self.clip.base_model.transformer.resblocks[-self.cfg.model.unfreeze_last_n_clip_layers:]:
                block.requires_grad_(True)

        self.hqsam = HQSam(model_type='vit_b')
        self.hqsam.eval()
        self.hqsam.requires_grad_(False)


    def get_base_mapper_model(self) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = NeTICLIPTextModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfg.model.revision)

        self.vae = AutoencoderKL.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant)
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant)

        self.token_embeds, self.placeholder_token_id = self._add_concept_token_to_tokenizer()
        neti_mapper, self.loaded_iteration = self._init_neti_mapper()
        self.text_encoder.text_model.embeddings.set_mapper(neti_mapper)

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        # Make sure to train the mapper
        self.text_encoder.text_model.embeddings.mapper.requires_grad_(True)
        self.text_encoder.text_model.embeddings.mapper.train()

        if self.cfg.trainer.enable_xformers_memory_efficient_attention:
            import xformers
            self.unet.enable_xformers_memory_efficient_attention()
        self.unet.set_attn_processor(XTIAttenProc())

    def pre_train_setup_base_mapper(self, weight_dtype: torch.dtype, accelerator: Accelerator, bypass_dtype_check: bool = False):
        self.weight_dtype = weight_dtype
        self.text_encoder = accelerator.prepare(self.text_encoder)

        if self.cfg.trainer.gradient_checkpointing:
            self.text_encoder.enable_gradient_checkpointing()

        if not bypass_dtype_check and accelerator.unwrap_model(self.text_encoder).dtype != torch.float32:
            raise ValueError(f"text_encoder loaded as datatype {accelerator.unwrap_model(self.text_encoder).dtype}.")

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        if self.cfg.trainer.compile:
            self.unet.to(memory_format=torch.channels_last)
            self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)

            self.clip: ClipFeatureExtractor = torch.compile(self.clip, mode="reduce-overhead", fullgraph=True)

        self.clip.to(accelerator.device, dtype=weight_dtype)
        self.hqsam.to(accelerator.device, dtype=weight_dtype)

    def get_text_conditioning(self, input_ids: torch.Tensor, timesteps: torch.Tensor, device: torch.device, **kwargs) -> Dict:
        """ Compute the text conditioning for the current batch of images using our text encoder over-ride. """
        _hs = {"this_idx": 0}
        for layer_idx, unet_layer in enumerate(UNET_LAYERS):
            neti_batch = NeTIBatch(
                input_ids=input_ids,
                placeholder_token_id=self.placeholder_token_id,
                timesteps=timesteps,
                unet_layers=torch.tensor(layer_idx, device=device).repeat(timesteps.shape[0])
            )
            layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=neti_batch, **kwargs)
            layer_hidden_state = layer_hidden_state[0].to(dtype=self.weight_dtype) # TODO: indexing a dataclass like this is very bad practice
            _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state
            if layer_hidden_state_bypass is not None:
                layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.weight_dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass
        return _hs
    
    def get_text_conditioning_per_timestep(self, unet_layers, input_ids: torch.Tensor, timesteps: torch.Tensor, device: torch.device, truncation_idx: Optional[int], num_images_per_prompt: int = 1, **kwargs) -> Dict:
        warnings.warn(f"Computing embeddings over {len(timesteps)} timesteps and {len(unet_layers)} U-Net layers.")
        hidden_states_per_timestep = []
        for timestep in tqdm(timesteps, leave=False):
            _hs = {"this_idx": 0}.copy()
            for layer_idx, unet_layer in enumerate(unet_layers):
                neti_batch = NeTIBatch(
                    input_ids=input_ids.to(device=self.text_encoder.device),
                    placeholder_token_id=self.placeholder_token_id,
                    timesteps=timestep.unsqueeze(0).to(device=self.text_encoder.device),
                    unet_layers=torch.tensor(layer_idx, device=self.text_encoder.device).unsqueeze(0),
                    truncation_idx=truncation_idx
                )
                layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=neti_batch)
                layer_hidden_state = layer_hidden_state[0].to(dtype=self.dtype)
                _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state.repeat(num_images_per_prompt, 1, 1)
                if layer_hidden_state_bypass is not None:
                    layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.dtype)
                    _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass.repeat(num_images_per_prompt, 1, 1)
            hidden_states_per_timestep.append(_hs)
    
    def _add_concept_token_to_tokenizer(self) -> Tuple[torch.Tensor, int]:
        """
        Adds the concept token to the tokenizer and initializes it with the embeddings of the super category token.
        The super category token will also be used for computing the norm for rescaling the mapper output.
        """
        num_added_tokens = self.tokenizer.add_tokens(self.cfg.model.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {self.cfg.model.placeholder_token}. "
                f"Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the super_category_token, placeholder_token to ids
        token_ids = self.tokenizer.encode(self.cfg.model.super_category_token, add_special_tokens=False)

        # Check if super_category_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The super category token must be a single token.")

        super_category_token_id = token_ids[0]
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.cfg.model.placeholder_token)
        self.cfg.model.placeholder_token_id = placeholder_token_id

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize the newly added placeholder token with the embeddings of the super category token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[super_category_token_id].clone()

        # Compute the norm of the super category token embedding for scaling mapper output
        self.cfg.model.target_norm = None
        if self.cfg.model.normalize_mapper_output:
            self.cfg.model.target_norm = token_embeds[super_category_token_id].norm().item()

        return token_embeds, placeholder_token_id

    def _init_neti_mapper(self) -> Tuple[NeTIMapper, Optional[int]]:
        loaded_iteration = None
        neti_mapper = NeTIMapper(
            output_dim=768,
            use_nested_dropout=self.cfg.model.use_nested_dropout,
            nested_dropout_prob=self.cfg.model.nested_dropout_prob,
            norm_scale=self.cfg.model.target_norm,
            use_positional_encoding=self.cfg.model.use_positional_encoding,
            num_pe_time_anchors=self.cfg.model.num_pe_time_anchors,
            pe_sigmas=self.cfg.model.pe_sigmas,
            output_bypass=self.cfg.model.output_bypass,
            cfg=self.cfg
        )
        return neti_mapper, loaded_iteration
    
    def get_hidden_state(self, batch, timesteps, dtype, device, per_timestep: bool = False):
        bs: int = batch['disc_pixel_values'].shape[0]

        clip_feature_map = self.clip(batch['disc_pixel_values'].to(device=device, dtype=dtype))['stage23']

        def viz():
            from image_utils import Im, pca, get_layered_image_from_binary_mask, calculate_principal_components
            principal_components = calculate_principal_components(clip_feature_map.reshape(-1, clip_feature_map.shape[-1]))
            clip_feature_map = self.clip(batch['disc_pixel_values'].to(device=device, dtype=dtype))['stage23']
            outmap = pca(clip_feature_map[1:, ...].permute(1, 2, 0).reshape(4, 1024, 16, 16).permute(0, 2, 3, 1).reshape(-1, 1024).float(), principal_components=principal_components).reshape(4, 16, 16, 3).permute(0, 3, 1, 2)
            outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
            outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
            outmap = (outmap - outmap_min) / (outmap_max - outmap_min)
            Im(outmap).save('pca')
            sam_input = rearrange((((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), 'b c h w -> b h w c')
            Im(sam_input).save('rgb')
            Im(get_layered_image_from_binary_mask(original.permute(1, 2, 0))).save('masks')

        # viz()
        
        clip_feature_map = rearrange(clip_feature_map, 'l b d -> b l d')
        clip_feature_cls_token = clip_feature_map[:, 0, :] # We take the cls token
        clip_feature_map = clip_feature_map[:, 1:, :]

        # SAM requires NumPy [0, 255]
        sam_input = rearrange((((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), 'b c h w -> b h w c')

        latent_dim = batch['disc_pixel_values'].shape[-1] // 14
        feature_map_masks = []
        feature_map_batch_idxs = []
        for i in range(bs):
            if 'gen_segmentation' in batch: # We have gt masks
                original = batch['gen_segmentation'][i].permute(2, 0, 1).bool()
            else:
                masks = self.hqsam.forward(sam_input[i])
                masks = masks[:23]
                num_masks = len(masks)
                if num_masks == 0: # Hack to deal with images with no masks. We attend to a single token only
                    tmp_ = torch.zeros((1, latent_dim, latent_dim), dtype=torch.bool)
                    tmp_[:, 0, 0] = True
                    feature_map_masks.append(tmp_)
                    feature_map_batch_idxs.append(i * torch.ones((1), dtype=torch.long))
                    continue

                original = torch.from_numpy(np.array([masks[i]['segmentation'] for i in range(num_masks)]))

            assert batch['disc_pixel_values'].shape[-1] == batch['disc_pixel_values'].shape[-2]
            feature_map_mask_ = find_true_indices_batched(original=original, dh=latent_dim, dw=latent_dim)
            feature_map_masks.append(feature_map_mask_)
            feature_map_batch_idxs.append(i * feature_map_mask_.new_ones((feature_map_mask_.shape[0]), dtype=torch.long))

        # If the 1st image has 5 masks and the 2nd has 3 masks, we will have an integer tensor of shape (total == 8,) for 8 different cross-attns. The sequence length for each is thus the number of valid "pixels" (KVs)
        feature_map_masks = torch.cat(feature_map_masks, dim=0) # feature_map_mask is a boolean mask of (total, h, w)
        feature_map_masks = rearrange(feature_map_masks, 'total h w -> total (h w)').to(device)
        feature_map_batch_idxs = torch.cat(feature_map_batch_idxs, dim=0).to(device)

        # We sum the number of valid "pixels" in each mask
        seqlens_k = feature_map_masks.sum(dim=-1) # (total,)
        max_seqlen_k = seqlens_k.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0, dtype=torch.torch.int32), (1, 0))

        flat_features = rearrange(clip_feature_map[feature_map_batch_idxs], 'total (h w) d -> (total h w) d', h=latent_dim, w=latent_dim)
        flat_mask = rearrange(feature_map_masks, 'total (h w) -> (total h w)', h=latent_dim, w=latent_dim)
        k_features = flat_features[flat_mask]
            
        # The actual query is obtained from the timestep + layer encoding later
        cu_seqlens_q = F.pad(torch.arange(seqlens_k.shape[0]).to(torch.int32) + 1, (1, 0)).to(device)
        max_seqlen_q = 1 # We are doing attention pooling so we have one query per mask

        attn_dict = dict(x_kv=k_features, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)

        placeholder_token_id = self.tokenizer(self.cfg.model.placeholder_token, add_special_tokens=False).input_ids[0]
        mask_tokens_ids = self.tokenizer(f"{self.cfg.model.placeholder_token} and", add_special_tokens=False).input_ids
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer._pad_token)

        bs = batch['input_ids'].shape[0]
        for b in range(bs):
            # Everything after 1st pad token should also be a pad token
            token_is_padding = (batch['input_ids'][b] == pad_token_id).nonzero()
            assert (token_is_padding.shape[0] == (batch['input_ids'].shape[1] - token_is_padding[0])).item()
            mask_part_of_batch = (feature_map_batch_idxs == b).nonzero().squeeze(1)
            assert token_is_padding.shape[0] >= mask_part_of_batch.shape[0] # We need at least as many pad tokens as we have masks
            extent = len(batch['input_ids'][b, token_is_padding[0]:token_is_padding[0]+(mask_part_of_batch.shape[0] * len(mask_tokens_ids))])
            batch['input_ids'][b, token_is_padding[0]:token_is_padding[0]+(mask_part_of_batch.shape[0] * len(mask_tokens_ids))] = torch.tensor(mask_tokens_ids * mask_part_of_batch.shape[0])[:extent]

        text_encoder_dict = dict(
            attn_dict=attn_dict,
            placeholder_token=placeholder_token_id,
            pad_token=pad_token_id,
            feature_map_batch_idxs=feature_map_batch_idxs
        )

        if per_timestep:
            return batch['input_ids'], text_encoder_dict

        _hs = self.get_text_conditioning(input_ids=batch['input_ids'], timesteps=timesteps, device=device, **text_encoder_dict)

        return _hs

    def forward(self, batch, noisy_latents, timesteps, weight_dtype):
        _hs = self.get_hidden_state(batch, timesteps, device=noisy_latents.device, dtype=weight_dtype)

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, _hs).sample

        return model_pred