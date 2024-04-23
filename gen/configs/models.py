from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store, store_child_config
from gen.models.encoders.encoder import BaseModel, ViTFeatureExtractor
from gen.models.encoders.extra_encoders import ClipFeatureExtractor


class ModelType(Enum):
    BASE_MAPPER = 0

@dataclass
class ModelConfig:
    name: ClassVar[str] = "model"

    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    token_embedding_dim: int = 768
    num_decoder_cross_attn_tokens: int = 77
    decoder_latent_dim: int = 64  # Resolution after VAE [input to U-Net]. For SD at 512x512, this is 64 x 64
    encoder_resolution: int = 224
    decoder_resolution: int = 512

    revision: Optional[str] = None
    variant: Optional[str] = None
    model_type: Optional[ModelType] = None

    model_type: ModelType = ModelType.BASE_MAPPER
    
    unet: bool = True  # Allow not loading a U-Net [e.g., for comparison exps without diffusion]
    controlnet: bool = False  # Add a controlnet on top of the main U-Net
    freeze_unet: bool = True
    unfreeze_single_unet_layer: bool = False # Lets us debug on a 12GB GPU
    unet_lora: bool = False  # Perform low-rank adaptation on the main U-Net
    lora_rank: int = 4

    freeze_clip: bool = True  # We freeze the CLIP Vision encoder by default
    unfreeze_last_n_clip_layers: Optional[int] = None  # Unfreeze specific clip layers
    unfreeze_unet_after_n_steps: Optional[int] = None

    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)
    encoder_dim: int = 1024  # Dim of each token from encoder [E.g., CLIP]

    ema: bool = False
    autoencoder_slicing: bool = True
    encoder_latent_dim: int = 16 # resolution // patch_size

    fused_mlp: bool = True
    fused_bias_fc: bool = True

    lr_finetune_version: int = 0

    diffusion_timestep_range: Optional[tuple[int, int]] = None
    diffusion_loss_weight: float = 1.0
    disable_unet_during_training: bool = False
    
    add_grid_to_input_channels: bool = False

    use_sd_15_tokenizer_encoder: bool = False
    clip_lora: bool = False  # Perform low-rank adaptation on the main U-Net
    clip_lora_rank: int = 16
    clip_lora_alpha: int = 8
    clip_lora_dropout: float = 0.1
    debug_feature_maps: bool = False
    feature_map_keys: Optional[tuple[str]] = None  # Use multiple feature maps for attention pooling
    finetune_unet_with_different_lrs: bool = False


auto_store(
    ModelConfig,
    model_type=ModelType.BASE_MAPPER,
    name="basemapper",
    encoder=builds(ClipFeatureExtractor, return_only="ln_post", populate_full_signature=False),
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="basemapper_vit",
    encoder=builds(ViTFeatureExtractor, return_only="norm", populate_full_signature=False),
    encoder_dim=768,
)