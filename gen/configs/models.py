from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store, store_child_config
from gen.models.cross_attn.decoder import DecoderTransformer
from gen.models.encoders.encoder import BaseModel, ClipFeatureExtractor, ViTFeatureExtractor


class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1
    SODA = 2


@dataclass
class ModelConfig:
    name: ClassVar[str] = "model"

    pretrained_model_name_or_path: Optional[str] = "runwayml/stable-diffusion-v1-5"
    token_embedding_dim: int = 768

    resolution: int = 512

    revision: Optional[str] = None
    variant: Optional[str] = None
    model_type: Optional[ModelType] = None

    model_type: ModelType = ModelType.BASE_MAPPER

    controlnet: bool = False # Add a controlnet on top of the main U-Net
    freeze_unet: bool = True
    lora_unet: bool = False # Perform low-rank adaptation on the main U-Net
    freeze_mapper: bool = False # Freezes the cross-attention mapper itself. Useful for debugging.
    freeze_text_encoder: bool = True # We freeze the CLIP Text encoder by default
    freeze_text_encoder_except_token_embeddings: bool = False # We can choose to only train the token embeddings like in break-a-scene
    freeze_clip: bool = True # We freeze the CLIP Vision encoder by default
    unfreeze_last_n_clip_layers: Optional[int] = None # Unfreeze specific clip layers
    unfreeze_unet_after_n_steps: Optional[int] = None

    dropout_masks: Optional[float] = None # We can randomly dropout object masks during training. The background is always preserved.
    per_timestep_conditioning: bool = True # Switches to the NeTI-style conditioning scheme where we get an embedding per T+L

    use_dataset_segmentation: bool = True # Determines if we use the dataset GT or SAM
    cross_attn_dim: int = 1024
    
    decoder_transformer: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)
    encoder_dim: int = 1024 # Dim of each token from encoder [E.g., CLIP]

    lora_rank: int = 4
    break_a_scene_cross_attn_loss: bool = False
    break_a_scene_cross_attn_loss_weight: float = 1e-2
    break_a_scene_masked_loss: bool = False
    background_mask_idx: int = 0 # Used for the break-a-scene mask loss to not count loss for the background mask
    placeholder_token: str = "masks"

    training_cfg_dropout: Optional[float] = None # Whether to use dropout in the training cfg
    training_layer_dropout: Optional[float] = None # Whether to use dropout in the training cfg

    # Quick experiment configs
    break_a_scene_cross_attn_loss_second_stage: bool = False
    dropout_foreground_only: bool = False
    dropout_background_only: bool = False
    layer_specialization: bool = False # Give each layer has its own embedding
    num_conditioning_pairs: int = 8 # Number of cross-attentions between U-Net latents and text-tokens

    clip_shift_scale_conditioning: bool = False # Whether to use the CLIP shift and scale embeddings as conditioning
    add_pos_emb_after_clip: bool = False
    use_dummy_mask: bool = False # Maps single token to (2 * token_embedding_dim) instead of T+L mapping
    weighted_object_loss: bool = False


@dataclass
class ControlNetConfig(ModelConfig):
    model_type: ModelType = ModelType.CONTROLNET
    controlnet_model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None


auto_store(ControlNetConfig, name="controlnet")
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