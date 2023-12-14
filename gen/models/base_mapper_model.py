import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from packaging import version
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from gen.configs import BaseConfig
import wandb

from gen.models.neti.net_clip_text_embedding import NeTIBatch
from gen.models.neti.neti_clip_text_encoder import NeTICLIPTextModel
from gen.models.neti.neti_mapper import UNET_LAYERS, NeTIMapper
from gen.models.neti.xti_attention_processor import XTIAttenProc

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
    




import torch
import torch.nn as nn

class BaseMapper(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(BaseMapper, self).__init__()
        self.cfg = cfg
        self.get_base_mapper_model()

    def get_base_mapper_model(self) -> tuple[AutoTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.cfg.model.revision)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = NeTICLIPTextModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfg.model.revision)

        self.vae = AutoencoderKL.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision, variant=self.cfg.model.variant)
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision, variant=self.cfg.model.variant)

        self.token_embeds, self.placeholder_token_id = self._add_concept_token_to_tokenizer()
        neti_mapper, self.loaded_iteration = self._init_neti_mapper()
        self.text_encoder.text_model.embeddings.set_mapper(neti_mapper)

        import torch
        from PIL import Image
        import open_clip

        return_nodes = {
            'transformer.resblocks.0': 'stage0',
            'transformer.resblocks.5': 'stage5',
            'transformer.resblocks.11': 'stage1',
            'transformer.resblocks.17': 'stage17',
            'transformer.resblocks.23': 'stage23',
            'ln_post': 'ln_post',
        }
        from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
        # train_nodes, eval_nodes = get_graph_node_names(self.clip.visual)
        
        self.clip = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[0]
        self.clip = create_feature_extractor(self.clip.visual, return_nodes=return_nodes)
        # tokenizer = open_clip.get_tokenizer('ViT-L-14') # 
        # self.clip(batch['disc_pixel_values'])

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


    def pre_train_setup_base_mapper(self, weight_dtype: torch.dtype, accelerator: Accelerator):
        self.weight_dtype = weight_dtype
        self.text_encoder = accelerator.prepare(self.text_encoder)

        if self.cfg.trainer.gradient_checkpointing:
            self.text_encoder.enable_gradient_checkpointing()

        if accelerator.unwrap_model(self.text_encoder).dtype != torch.float32:
            raise ValueError(f"text_encoder loaded as datatype {accelerator.unwrap_model(self.text_encoder).dtype}.")

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        if self.cfg.trainer.compile:
            self.unet.to(memory_format=torch.channels_last)
            self.unet: UNet2DConditionModel = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)

    def get_text_conditioning(self, input_ids: torch.Tensor, timesteps: torch.Tensor, device: torch.device) -> Dict:
        """ Compute the text conditioning for the current batch of images using our text encoder over-ride. """
        _hs = {"this_idx": 0}
        for layer_idx, unet_layer in enumerate(UNET_LAYERS):
            neti_batch = NeTIBatch(
                input_ids=input_ids,
                placeholder_token_id=self.placeholder_token_id,
                timesteps=timesteps,
                unet_layers=torch.tensor(layer_idx, device=device).repeat(timesteps.shape[0])
            )
            layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=neti_batch)
            layer_hidden_state = layer_hidden_state[0].to(dtype=self.weight_dtype)
            _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state
            if layer_hidden_state_bypass is not None:
                layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.weight_dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass
        return _hs
    
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
            output_bypass=self.cfg.model.output_bypass
        )
        return neti_mapper, loaded_iteration

    def forward(self, batch, noisy_latents, timesteps, weight_dtype):
        # Get the text embedding for conditioning

        batch['input_ids'][:, 0] = self.tokenizer.convert_tokens_to_ids(self.cfg.model.placeholder_token) # TODO: HACK!!!
        _hs = self.get_text_conditioning(input_ids=batch['input_ids'], timesteps=timesteps, device=noisy_latents.device)

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, _hs).sample

        return model_pred