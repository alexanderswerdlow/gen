from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store
from gen.models.encoders.encoder import BaseModel, ViTFeatureExtractor


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
    unet_lora_rank: int = 4
    use_dora: bool = True

    freeze_enc: bool = True  # We freeze the CLIP Vision encoder by default
    unfreeze_last_n_enc_layers: Optional[int] = None  # Unfreeze specific clip layers

    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)
    encoder_dim: int = 1024  # Dim of each token from encoder [E.g., CLIP]

    ema: bool = False
    autoencoder_slicing: bool = True
    encoder_latent_dim: int = 16 # resolution // patch_size

    lr_finetune_version: int = 0
    finetune_unet_with_different_lrs: bool = False

    diffusion_timestep_range: Optional[tuple[int, int]] = None
    diffusion_loss_weight: float = 1.0
    disable_unet_during_training: bool = False
    use_sd_15_tokenizer_encoder: bool = False

    enc_lora: bool = False  # Perform low-rank adaptation on the main U-Net
    enc_lora_rank: int = 16
    enc_lora_alpha: int = 8
    enc_lora_dropout: float = 0.1

    stock_dino_v2: bool = False
    debug_feature_maps: bool = False
    feature_map_keys: Optional[tuple[str]] = None
    
    fused_mlp: bool = True
    fused_bias_fc: bool = True

    duplicate_unet_input_channels: bool = False
    separate_xyz_encoding: bool = False
    dual_attention: bool = False
    joint_attention: bool = False
    enable_encoder: bool = False
    force_fp32_pcd_vae: bool = False
    snr_gamma: Optional[float] = None # Use 5.0 if enabled
    predict_depth: bool = False
    use_valid_xyz_loss_mask: bool = False
    fill_invalid_regions: bool = False
    unfreeze_vae_decoder: bool = False
    vae_decoder_batch_size: int = 1
    vae: bool = True
    xyz_min_max_quantile: float = 0.02
    only_noise_tgt: bool = False
    dropout_src_depth: Optional[float] = None
    freeze_self_attn: bool = False
    n_view_pred: bool = False
    add_cross_attn_pos_emb: Optional[int] = None
    


auto_store(
    ModelConfig,
    model_type=ModelType.BASE_MAPPER,
    name="basemapper",
    encoder=builds(ViTFeatureExtractor, return_only="norm", populate_full_signature=False),
    encoder_dim=768,
)