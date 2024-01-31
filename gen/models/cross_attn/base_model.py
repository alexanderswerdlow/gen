import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import einx
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from beartype import beartype
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline, StableDiffusionPipeline, UNet2DConditionModel
from einops import rearrange
from jaxtyping import Bool, Float, Integer
from omegaconf import OmegaConf
from torch import Tensor
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel

from gen.configs import BaseConfig
from gen.models.cross_attn.break_a_scene import register_attention_control
from gen.models.cross_attn.losses import break_a_scene_cross_attn_loss, break_a_scene_masked_loss, evenly_weighted_mask_loss
from gen.models.cross_attn.modules import Mapper
from gen.models.encoders.encoder import BaseModel, ClipFeatureExtractor
from gen.models.utils import find_true_indices_batched, positionalencoding2d
from gen.utils.diffusers_utils import load_stable_diffusion_model
from gen.utils.logging_utils import log_info, log_warn
from gen.utils.tokenization_utils import get_uncond_tokens
from gen.utils.trainer_utils import Trainable, TrainingState


class InputData(TypedDict):
    gen_pixel_values: Float[Tensor, "b c h w"]
    gen_segmentation: Integer[Tensor, "b h w c"]
    disc_pixel_values: Float[Tensor, "b c h w"]
    input_ids: Integer[Tensor, "b l"]
    state: Optional[TrainingState]


@dataclass
class ConditioningData:
    placeholder_token: Optional[int] = None
    attn_dict: Optional[dict[str, Tensor]] = None
    clip_feature_map: Optional[Float[Tensor, "b d h w"]] = None
    mask_tokens: Optional[Float[Tensor, "n d"]] = None
    mask_batch_idx: Optional[Integer[Tensor, "n"]] = None
    mask_instance_idx: Optional[Integer[Tensor, "n"]] = None
    batch_cond_dropout: Optional[Bool[Tensor, "b"]] = None
    input_prompt: Optional[list[str]] = None

    # These are passed to the U-Net or pipeline
    encoder_hidden_states: Optional[Float[Tensor, "b d"]] = None
    unet_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)


class AttentionMetadata(TypedDict):
    layer_idx: int
    num_layers: int
    num_cond_vectors: int
    add_pos_emb: bool
    attention_mask: Optional[Float[Tensor, "b d h w"]]


class BaseMapper(Trainable):
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg: BaseConfig = cfg
        self.dtype = getattr(
            torch, cfg.trainer.dtype.split(".")[-1]
        )  # dtype of most intermediate tensors and frozen weights. Notably, we always use FP32 for trainable params.

        self.initialize_diffusers_models()
        self.initialize_custom_models()
        self.add_adapters()

        from gen.models.cross_attn.base_inference import infer_batch

        BaseMapper.infer_batch = infer_batch

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_custom_models(self):
        self.mapper = Mapper(cfg=self.cfg).to(self.cfg.trainer.device)
        self.clip: BaseModel = hydra.utils.instantiate(self.cfg.model.encoder)

        if self.cfg.model.use_dataset_segmentation is False:
            from gen.models.encoders.sam import HQSam

            self.hqsam = HQSam(model_type="vit_b")

        if self.cfg.model.clip_shift_scale_conditioning:

            def zero_module(module):
                for p in module.parameters():
                    nn.init.zeros_(p)
                return module

            clip_proj_layers = []
            all_layer_data = [(k, v.shape[0]) for k, v in dict(self.unet.named_parameters()).items() if "resnets.0" in k and "conv1.weight" in k]
            dims = []
            dims.extend([v for k, v in all_layer_data if "down_blocks" in k])
            dims.extend([v for k, v in all_layer_data if "mid_block" in k])
            dims.extend([v for k, v in all_layer_data if "up_blocks" in k])

            for idx, dim in enumerate(dims):
                clip_proj_layers.append(nn.Conv2d(self.cfg.model.encoder_dim, dim * 2, kernel_size=1))
            self.clip_proj_layers = nn.ModuleList(clip_proj_layers)
            self.clip_proj_layers = zero_module(self.clip_proj_layers)

    def initialize_diffusers_models(self) -> tuple[CLIPTokenizer, DDPMScheduler, AutoencoderKL, UNet2DConditionModel]:
        # Load the tokenizer
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.cfg.model.revision, use_fast=False
        )

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
        self.set_training_mode(set_grad=True)

        assert not (self.cfg.model.freeze_unet is False and (self.cfg.model.unfreeze_unet_after_n_steps is not None or self.cfg.model.unet_lora))
        if self.cfg.model.unet_lora:
            from peft import LoraConfig

            unet_lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=self.cfg.model.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)
            from diffusers.training_utils import cast_training_params

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
            assert num_cross_attn_layers == 2 * self.cfg.model.num_conditioning_pairs

    def set_training_mode(self, set_grad: bool = False):
        """
        Set training mode for the proper models and freeze/unfreeze them.

        We set the weights to weight_dtype only if they are frozen. Otherwise, they are left in FP32.

        We have the set_grad param as it appears that setting requires_grad after training has started can cause:
        `element 0 of tensors does not require grad and does not have a grad_fn`
        """
        if set_grad:
            self.vae.to(device=self.device, dtype=self.dtype)
            self.vae.requires_grad_(False)

        if self.cfg.model.use_dataset_segmentation is False:
            self.hqsam.eval()
            if set_grad:
                self.hqsam.requires_grad_(False)

        if self.cfg.model.freeze_clip:
            if set_grad:
                self.clip.requires_grad_(False)
            self.clip.eval()
            log_warn("CLIP is frozen for debugging")
        else:
            if set_grad:
                self.clip.requires_grad_(True)
            self.clip.train()
            log_warn("CLIP is unfrozen")

        if self.cfg.model.unfreeze_last_n_clip_layers is not None:
            log_warn(f"Unfreezing last {self.cfg.model.unfreeze_last_n_clip_layers} CLIP layers")
            for block in self.clip.base_model.transformer.resblocks[-self.cfg.model.unfreeze_last_n_clip_layers :]:
                if set_grad:
                    block.requires_grad_(True)
                block.train()

        if self.cfg.model.unfreeze_resnet:
            for module in list(self.clip.base_model.children())[:6]:
                if set_grad:
                    module.requires_grad_(True)
                module.train()

        if self.cfg.model.freeze_unet:
            if set_grad:
                self.unet.to(device=self.device, dtype=self.dtype)
                self.unet.requires_grad_(False)
            self.unet.eval()
        else:
            if set_grad:
                self.unet.requires_grad_(True)
            self.unet.train()

        if self.cfg.model.freeze_text_encoder:
            if set_grad:
                self.text_encoder.to(device=self.device, dtype=self.dtype)
                self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            if set_grad:
                self.text_encoder.requires_grad_(True)
            if set_grad and self.cfg.model.freeze_text_encoder_except_token_embeddings:
                self.text_encoder.text_model.encoder.requires_grad_(False)
                self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
                self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            self.text_encoder.train()

        if self.cfg.model.freeze_mapper:
            if set_grad:
                self.mapper.requires_grad_(False)
            self.mapper.eval()
        else:
            if set_grad:
                self.mapper.requires_grad_(True)
            self.mapper.train()

        if self.cfg.model.controlnet:
            if set_grad:
                self.controlnet.requires_grad_(True)
            self.controlnet.train()

        if self.cfg.model.clip_shift_scale_conditioning:
            if set_grad:
                self.clip_proj_layers.requires_grad_(True)
            self.clip_proj_layers.train()

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

    def set_inference_mode(self):
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

        self.eval()
        if self.cfg.model.break_a_scene_cross_attn_loss:
            self.controller.reset()

    def get_param_groups(self):
        if not self.cfg.model.finetune_variable_learning_rate:
            return None

        unet_params = dict(self.unet.named_parameters())

        def get_params(params, keys):
            group = {k: v for k, v in params.items() if all([key in k for key in keys])}
            for k, p in group.items():
                del params[k]
            return group

        return [  # Order matters here
            {"params": get_params(unet_params, ("attn2",)).values(), "lr": self.cfg.trainer.learning_rate},
            {"params": unet_params.values(), "lr": self.cfg.trainer.learning_rate / 10},
            {"params": self.mapper.parameters(), "lr": self.cfg.trainer.learning_rate},
        ]

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

    def check_add_segmentation(self, batch: InputData):
        """
        This function checks if we have segmentation for the current batch. If we do not, we add dummy segmentation or use HQSam to get segmentation.
        """
        if self.cfg.model.use_dataset_segmentation:
            return batch  # We already have segmentation

        if self.cfg.model.use_dummy_mask:
            original = batch["gen_segmentation"][i].new_ones((1, batch["gen_segmentation"][i].shape[0], batch["gen_segmentation"][i].shape[1]))
            assert False

        # SAM requires NumPy [0, 255]
        sam_input = rearrange((((batch["gen_pixel_values"] + 1) / 2) * 255).to(torch.uint8).cpu().detach().numpy(), "b c h w -> b h w c")
        gen_segmentations = []
        bs: int = batch["disc_pixel_values"].shape[0]

        for i in range(bs):
            masks = self.hqsam.forward(sam_input[i])
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            max_masks = 4
            masks = masks[:max_masks]  # We only have 77 tokens
            original = torch.from_numpy(np.array([masks[i]["segmentation"] for i in range(len(masks))]))

            if original.shape[0] == 0:  # Make dummy mask with all ones
                original = (
                    batch["gen_segmentation"][i].new_ones((1, batch["gen_segmentation"][i].shape[0], batch["gen_segmentation"][i].shape[1])).cpu()
                )
            else:  # Add additional mask to capture any pixels that are not part of any mask
                original = torch.cat(((~original.any(dim=0))[None], original), dim=0)

            if original.shape[0] != 0 and not self.training:  # During inference, we update batch with the SAM masks for later visualization
                gen_segmentation_ = original.permute(1, 2, 0).long().clone()
                gen_segmentations.append(
                    torch.nn.functional.pad(gen_segmentation_, (0, (max_masks + 1) - gen_segmentation_.shape[-1]), "constant", 0)
                )

        if len(gen_segmentations) > 0:
            batch["gen_segmentation"] = torch.stack(gen_segmentations, dim=0)

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

    def add_cross_attn_params(self, batch, cond: ConditioningData):
        """
        This function sets up the attention parameters for the mask conditioning:
            - Gets CLIP features for K/V and flattens them
            - Updates our input tokens to be "A photo of mask_0 and mask_1" and so on
        """
        bs: int = batch["disc_pixel_values"].shape[0]
        device = batch["gen_pixel_values"].device
        dtype = self.dtype

        # isinstance fails with torch dynamo
        clip_feature_map = self.clip(batch["disc_pixel_values"].to(device=device, dtype=dtype))  # b l d

        if "resnet" not in self.clip.model_name:
            clip_feature_map = clip_feature_map[:, 1:, :]
            if "dino" in self.clip.model_name and "reg" in self.clip.model_name:
                clip_feature_map = clip_feature_map[:, 4:, :]

        if self.cfg.model.add_pos_emb:
            h, w = round(math.sqrt(clip_feature_map.shape[1])), round(math.sqrt(clip_feature_map.shape[1]))
            pos_emb = positionalencoding2d(clip_feature_map.shape[-1], h, w).to(clip_feature_map)
            clip_feature_map = clip_feature_map + rearrange(pos_emb, "d h w -> () (h w) d")

        latent_dim = round(math.sqrt(clip_feature_map.shape[1]))
        feature_map_masks = []
        mask_batch_idx = []
        mask_instance_idx = []

        assert "gen_segmentation" in batch
        for i in range(bs):
            one_hot_mask: Bool[Tensor, "d h w"] = batch["gen_segmentation"][i].permute(2, 0, 1).bool()
            one_hot_idx = torch.arange(one_hot_mask.shape[0], device=device)

            if one_hot_mask.shape[0] == 0:
                log_warn("No masks found for this image")
                continue

            if self.cfg.model.dropout_masks is not None and self.training:
                dropout_mask = torch.rand(one_hot_mask.size(0)).to(device) > self.cfg.model.dropout_masks
                if self.cfg.model.dropout_foreground_only:
                    dropout_mask[self.cfg.model.background_mask_idx] = True  # We always keep the background mask
                elif self.cfg.model.dropout_background_only:
                    dropout_mask[torch.arange(dropout_mask.shape[0]) != self.cfg.model.background_mask_idx] = True

                if dropout_mask.sum().item() == 0:
                    log_warn("We would have dropped all masks but instead we preserved the background", main_process_only=False)
                    dropout_mask[self.cfg.model.background_mask_idx] = True
            else:
                dropout_mask = one_hot_mask.new_full((one_hot_mask.shape[0],), True, dtype=torch.bool)

            # Remove empty masks. Because of flash attention bugs, we require a minimum of 64 pixels. 14 is too few.
            empty_mask = torch.sum(one_hot_mask, dim=[1, 2]) > 64

            if (dropout_mask & empty_mask).sum().item() == 0:
                dropout_mask[(~dropout_mask).nonzero(as_tuple=True)[0]] = True

            combinbed_mask = dropout_mask & empty_mask
            one_hot_mask = one_hot_mask[combinbed_mask]
            one_hot_idx = one_hot_idx[combinbed_mask]

            if one_hot_mask.shape[0] == 0:
                log_warn("Dropped out all masks for this image!")
                continue

            assert batch["disc_pixel_values"].shape[-1] == batch["disc_pixel_values"].shape[-2]
            feature_map_mask_ = find_true_indices_batched(original=one_hot_mask, dh=latent_dim, dw=latent_dim)
            feature_map_masks.append(feature_map_mask_)
            mask_batch_idx.append(i * feature_map_mask_.new_ones((feature_map_mask_.shape[0]), dtype=torch.long))
            mask_instance_idx.append(one_hot_idx)

        # If the 1st image has 5 masks and the 2nd has 3 masks, we will have an integer tensor of shape (total == 8,) for 8 different cross-attns. The sequence length for each is thus the number of valid "pixels" (KVs)
        feature_map_masks = torch.cat(feature_map_masks, dim=0)  # feature_map_mask is a boolean mask of (total, h, w)
        feature_map_masks = rearrange(feature_map_masks, "total h w -> total (h w)").to(device)
        mask_batch_idx = torch.cat(mask_batch_idx, dim=0).to(device)
        mask_instance_idx = torch.cat(mask_instance_idx, dim=0).to(device)

        # We sum the number of valid "pixels" in each mask
        seqlens_k = feature_map_masks.sum(dim=-1)  # (total,)
        max_seqlen_k = seqlens_k.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens_k, dim=0, dtype=torch.torch.int32), (1, 0))

        flat_features = rearrange(clip_feature_map[mask_batch_idx], "total (h w) d -> (total h w) d", h=latent_dim, w=latent_dim)
        flat_mask = rearrange(feature_map_masks, "total (h w) -> (total h w)", h=latent_dim, w=latent_dim)
        k_features = flat_features[flat_mask]

        # The actual query is obtained from the timestep + layer encoding later
        cu_seqlens_q = F.pad(torch.arange(seqlens_k.shape[0]).to(torch.int32) + 1, (1, 0)).to(device)
        max_seqlen_q = 1  # We are doing attention pooling so we have one query per mask

        cond.attn_dict = dict(x_kv=k_features, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k)

        cond.mask_batch_idx = mask_batch_idx
        cond.mask_instance_idx = mask_instance_idx
        if self.cfg.model.clip_shift_scale_conditioning:
            clip_feature_maps = []
            for _, layer in enumerate(self.clip_proj_layers):
                proj_feature_map = layer(rearrange(clip_feature_map, "b (h w) d -> b d h w", h=latent_dim, w=latent_dim))
                clip_feature_maps.append(proj_feature_map)

            cond.unet_kwargs["femb"] = clip_feature_maps

        return cond

    def compute_mask_tokens(self, batch: InputData, cond: ConditioningData):
        """
        Generates mask tokens by adding segmentation if necessary, then setting up and calling the cross attention module
        """
        batch = self.check_add_segmentation(batch)
        cond = self.add_cross_attn_params(batch, cond)

        queries = self.mapper.learnable_token[None, :].repeat(cond.mask_batch_idx.shape[0], 1)
        cond.attn_dict["x"] = queries.to(self.dtype)
        cond.mask_tokens = self.mapper.cross_attn(cond).to(self.dtype)

        if self.cfg.model.layer_specialization:
            cond.encoder_hidden_states = einx.rearrange("b t d -> b t (n d)", cond.encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)

            # Break e.g., 1024 -> 16 x 64
            layerwise_mask_tokens = einx.rearrange("b (l c) -> b l c", cond.mask_tokens, l=self.cfg.model.num_conditioning_pairs)  
            layerwise_mask_tokens = self.mapper.layer_specialization(layerwise_mask_tokens)  # Batched 64 -> 1024
            cond.mask_tokens = einx.rearrange("b l c -> b (l c)", layerwise_mask_tokens).to(self.dtype)

        return cond

    def update_hidden_state_with_mask_tokens(
        self,
        batch: InputData,
        cond: ConditioningData,
    ):
        bs = batch["input_ids"].shape[0]

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
            additional_eos_token = torch.tensor([self.eos_token_id]).to(self.device)

            batch["formatted_input_ids"][b] = torch.cat((start_of_prompt, masks_prompt, end_of_prompt, additional_eos_token), dim=0)[:cur_ids.shape[0]]

        # Overwrite mask locations
        learnable_idxs = (batch["formatted_input_ids"] == self.placeholder_token_id).nonzero(as_tuple=True)
        cond.encoder_hidden_states[learnable_idxs[0], learnable_idxs[1]] = cond.mask_tokens.to(cond.encoder_hidden_states)

        if self.cfg.model.layer_specialization:
            cond.unet_kwargs["cross_attention_kwargs"] = dict(
                attn_meta=dict(
                    layer_idx=0,
                    num_layers=self.cfg.model.num_conditioning_pairs * 2,
                    num_cond_vectors=self.cfg.model.num_conditioning_pairs,
                    add_pos_emb=self.cfg.model.add_pos_emb
                )
            )


        if self.cfg.model.clip_shift_scale_conditioning:
            assert not self.cfg.model.layer_specialization
            cond.encoder_hidden_states[:] = self.shift_scale_uncond_hidden_states

        return cond

    def get_hidden_state(
        self,
        batch: InputData,
        add_conditioning: bool = True,
        cond: Optional[ConditioningData] = None,  # We can optionally specify mask tokens to use [e.g., for composing during inference]
    ) -> ConditioningData:
        if cond is None:
            cond = ConditioningData()

        cond.placeholder_token = self.placeholder_token_id
        cond.encoder_hidden_states = self.text_encoder(input_ids=batch["input_ids"])[0].to(dtype=self.dtype)

        if add_conditioning:
            if cond.mask_tokens is None or cond.mask_batch_idx is None:
                cond = self.compute_mask_tokens(batch, cond)

            cond = self.update_hidden_state_with_mask_tokens(batch, cond)

            if self.cfg.model.attention_masking:
                self.attention_masking(batch, cond)

        return cond

    def get_controlnet_conditioning(self, batch):
        return batch["gen_segmentation"].permute(0, 3, 1, 2).to(dtype=self.dtype)

    @cached_property
    def uncond_hidden_states(self):
        uncond_input_ids = get_uncond_tokens(self.tokenizer).to(self.device)
        uncond_encoder_hidden_states = self.text_encoder(input_ids=uncond_input_ids[None]).last_hidden_state.to(dtype=self.dtype).squeeze(0)
        return uncond_encoder_hidden_states

    @cached_property
    def shift_scale_uncond_hidden_states(self):
        uncond_input_ids = get_uncond_tokens(self.tokenizer, "A photo of").to(self.device)
        uncond_encoder_hidden_states = self.text_encoder(input_ids=uncond_input_ids[None]).last_hidden_state.to(dtype=self.dtype).squeeze(0)
        return uncond_encoder_hidden_states

    def dropout_cfg(self, cond: ConditioningData):
        # We dropout the entire conditioning [all-layers] for a subset of batches.
        if self.cfg.model.training_cfg_dropout is not None:
            uncond_encoder_hidden_states = self.uncond_hidden_states

            if self.cfg.model.layer_specialization:  # If we have different embeddings per-layer, we need to repeat the uncond embeddings
                uncond_encoder_hidden_states = einx.rearrange("t d -> t (n d)", uncond_encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)

            dropout_idx = torch.rand(cond.encoder_hidden_states.shape[0]) < self.cfg.model.training_cfg_dropout
            cond.encoder_hidden_states[dropout_idx] = uncond_encoder_hidden_states
            cond.batch_cond_dropout = dropout_idx

            if self.cfg.model.attention_masking:
                attn_mask = cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["attention_mask"]
                attn_mask[dropout_idx] = torch.ones((77, 64, 64)).to(device=cond.encoder_hidden_states.device, dtype=torch.bool)

        # We also might dropout only specific pairs of layers. In this case, we use the same uncond embeddings.
        # Note that there is a rare chance that we could dropout all layers but still compute loss.
        if self.cfg.model.layer_specialization and self.cfg.model.training_layer_dropout is not None:
            dropout_idx = torch.rand(self.cfg.model.num_conditioning_pairs) < self.cfg.model.training_layer_dropout
            if dropout_idx.sum().item() == dropout_idx.shape[0]:
                dropout_idx[torch.randint(high=dropout_idx.shape[0], size=(1,)).to(dropout_idx.device)] = False

            cond.encoder_hidden_states = einx.rearrange("b t (n d) -> b n t d", cond.encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)
            cond.encoder_hidden_states[:, dropout_idx] = self.uncond_hidden_states
            cond.encoder_hidden_states = einx.rearrange("b n t d -> b t (n d)", cond.encoder_hidden_states, n=self.cfg.model.num_conditioning_pairs)


    def attention_masking(self, batch: InputData, cond: Optional[ConditioningData]):
        assert "cross_attention_kwargs" in cond.unet_kwargs
        assert "attn_meta" in cond.unet_kwargs["cross_attention_kwargs"]

        batch_size: int = batch["disc_pixel_values"].shape[0]
        device: torch.device = batch["disc_pixel_values"].device
        gen_seg_ = rearrange(batch["gen_segmentation"], "b h w c -> b c () h w").float()
        learnable_idxs = (batch["formatted_input_ids"] == cond.placeholder_token).nonzero(as_tuple=True)
        h, w = 64, 64 # Max latent size in U-Net

        all_masks = []
        for batch_idx in range(batch_size):
            cur_batch_mask = learnable_idxs[0] == batch_idx  # Find the masks for this batch
            token_indices = learnable_idxs[1][cur_batch_mask]
            segmentation_indices = cond.mask_instance_idx[cur_batch_mask] 

            GT_masks = F.interpolate(input=gen_seg_[batch_idx][segmentation_indices], size=(h, w)).squeeze(1) > 0.5
            tensor_nhw = torch.zeros(batch["formatted_input_ids"].shape[-1], h, w, dtype=torch.bool)

            for i, d in enumerate(token_indices):
                tensor_nhw[d] = GT_masks[i]

            tensor_nhw[:, (~tensor_nhw.any(dim=0))] = True
            all_masks.append(tensor_nhw.to(device))

        cond.unet_kwargs["cross_attention_kwargs"]["attn_meta"]["attention_mask"] = torch.stack(all_masks, dim=0).to(dtype=torch.bool)

    @beartype
    def forward(self, batch: InputData):
        batch = InputData(**batch)

        assert "formatted_input_ids" not in batch

        batch["gen_pixel_values"] = torch.clamp(batch["gen_pixel_values"], -1, 1)

        # Convert images to latent space
        latents = self.vae.encode(batch["gen_pixel_values"].to(dtype=self.dtype)).latent_dist.sample()
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

        cond = self.get_hidden_state(batch, add_conditioning=True)
        if self.training:
            self.dropout_cfg(cond)

        encoder_hidden_states = cond.encoder_hidden_states

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

        if "femb" in cond.unet_kwargs:
            (metadata_dict, clip_feature_maps) = cond.unet_kwargs["femb"]
            metadata_dict["layer_idx"] = 0

        if self.cfg.model.break_a_scene_masked_loss:
            loss_mask = break_a_scene_masked_loss(cfg=self.cfg, batch=batch, cond=cond)
            model_pred, target = model_pred * loss_mask, target * loss_mask

        losses = dict()
        if self.cfg.model.weighted_object_loss:
            losses["diffusion_loss"] = evenly_weighted_mask_loss(batch=batch, cond=cond, pred=model_pred.float(), target=target.float())
        else:
            losses["diffusion_loss"] = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.cfg.model.break_a_scene_cross_attn_loss:
            losses["cross_attn_loss"] = break_a_scene_cross_attn_loss(cfg=self.cfg, batch=batch, controller=self.controller, cond=cond)

        return losses
