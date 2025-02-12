from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional

from hydra_zen import builds
from hydra_zen.typing import Builds

from gen.configs.utils import auto_store, store_child_config
from gen.models.cross_attn.decoder import DecoderTransformer
from gen.models.encoders.encoder import BaseModel, ResNetFeatureExtractor, ViTFeatureExtractor
from gen.models.encoders.extra_encoders import ClipFeatureExtractor, ViTWithExtraChannelsFeatureExtractor


class ModelType(Enum):
    BASE_MAPPER = 0
    CONTROLNET = 1
    SODA = 2


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
    
    mask_token_conditioning: bool = True  # Whether to condition the U-Net with mask tokens
    unet: bool = True  # Allow not loading a U-Net [e.g., for comparison exps without diffusion]
    controlnet: bool = False  # Add a controlnet on top of the main U-Net
    freeze_unet: bool = True
    unfreeze_single_unet_layer: bool = False # Lets us debug on a 12GB GPU
    unet_lora: bool = False  # Perform low-rank adaptation on the main U-Net
    lora_rank: int = 4
    freeze_mapper: bool = False  # Freezes the cross-attention mapper itself. Useful for debugging.
    freeze_text_encoder: bool = True  # We freeze the CLIP Text encoder by default
    freeze_text_encoder_except_token_embeddings: bool = False  # We can choose to only train the token embeddings like in break-a-scene
    freeze_clip: bool = True  # We freeze the CLIP Vision encoder by default
    unfreeze_last_n_clip_layers: Optional[int] = None  # Unfreeze specific clip layers
    unfreeze_unet_after_n_steps: Optional[int] = None
    unfreeze_gated_cross_attn: bool = False
    freeze_token_encoder: bool = False

    training_mask_dropout: Optional[float] = None  # Randomly dropout object masks during training.
    training_layer_dropout: Optional[float] = None  # Randomly dropout layer conditioning during training.
    training_cfg_dropout: Optional[float] = None  # Randomly dropout all conditioning during training.

    use_dataset_segmentation: bool = True  # Determines if we use the dataset GT or SAM
    cross_attn_dim: int = 1024

    decoder_transformer: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    encoder: Builds[type[BaseModel]] = builds(BaseModel, populate_full_signature=False)
    encoder_dim: int = 1024  # Dim of each token from encoder [E.g., CLIP]

    break_a_scene_masked_loss: bool = False
    background_mask_idx: int = 0  # Used for the break-a-scene mask loss to not count loss for the background mask
    placeholder_token: str = "masks"

    finetune_unet_with_different_lrs: bool = False
    layer_specialization: bool = False  # Give each layer has its own embedding
    per_layer_queries: bool = False  # Give each layer has its own queries instead of splitting a single token
    num_conditioning_pairs: int = 8  # Number of paired cross-attentions between U-Net latents and text-tokens [e.g., half the true number]
    add_pos_emb: bool = False  # Adds positional embeddings to encoder feature maps and U-Net queries
    feature_map_keys: Optional[tuple[str]] = None  # Use multiple feature maps for attention pooling

    # The dataset returns [H, W] int64 segmentation maps from -1 to N-1. -1 is ignored and 0 is the background.
    # This value should equal N. In datasets with a maximum number of instances, this is then num_instances + 1
    segmentation_map_size: int = -1

    # Quick experiment configs
    break_a_scene_cross_attn_loss: bool = False
    break_a_scene_cross_attn_loss_weight: float = 1e-2
    break_a_scene_cross_attn_loss_second_stage: bool = False
    dropout_foreground_only: bool = False
    dropout_background_only: bool = False

    ema: bool = False
    autoencoder_slicing: bool = True
    encoder_latent_dim: int = 16 # resolution // patch_size

    fused_mlp: bool = True
    fused_bias_fc: bool = True

    token_head_dim: int = "${model.encoder_dim}"
    lr_finetune_version: int = 0
    num_layer_queries: int = 1

    weighted_object_loss: bool = False
    clip_shift_scale_conditioning: bool = False  # Whether to use the CLIP shift and scale embeddings as conditioning
    use_dummy_mask: bool = False  # Maps single token to (2 * token_embedding_dim) instead of T+L mapping
    unfreeze_resnet: bool = False  # Unfreeze a resnet encoder instead of CLIP during training
    attention_masking: bool = False  # Mode which only allows U-Net Queries to attend to their own mask [e.g., instead of pos embs]
    gated_cross_attn: bool = False
    viz: bool = False
    use_inverted_noise_schedule: bool = False  # TODO: Implement properly

    # Rotation/Cls Prediction
    token_cls_pred_loss: bool = False
    token_rot_pred_loss: bool = False
    num_token_cls: int = 17
    detach_features_before_cross_attn: bool = False
    rotation_diffusion_timestep: int = 100
    rotation_diffusion_parameterization: str = "epsilon"
    rotation_diffusion_start_timestep: Optional[int] = None
    use_timestep_mask_film: bool = False
    discretize_rot_bins_per_axis: int = 8
    discretize_rot_pred: bool = False
    token_rot_transformer_head: bool = False  # WIP
    predict_rotation_from_n_frames: Optional[int] = None
    diffusion_timestep_range: Optional[tuple[int, int]] = None
    diffusion_loss_weight: float = 1.0
    token_cls_loss_weight: float = 0.1
    disable_unet_during_training: bool = False
    
    # SODA
    add_grid_to_input_channels: bool = False
    dropout_grid_conditioning: Optional[float] = 0.15

    # Misc
    masked_self_attention: bool = False
    use_pad_mask_loss: bool = True

    # Eschernet
    custom_conditioning_map: bool = False
    eschernet: bool = False
    eschernet_6dof: bool = False
    use_sd_15_tokenizer_encoder: bool = False
    modulate_src_tokens_with_tgt_pose: bool = False
    modulate_src_tokens_with_mlp: bool = False
    modulate_src_tokens_with_film: bool = False
    modulate_src_tokens_with_vanilla_transformer: bool = False
    modulate_src_tokens_loss_after_layer_specialization: bool = False
    modulate_src_feature_map: bool = False
    
    token_modulator: Builds[type[DecoderTransformer]] = builds(DecoderTransformer, populate_full_signature=True)
    add_text_tokens: bool = True

    clip_lora: bool = False  # Perform low-rank adaptation on the main U-Net
    clip_lora_rank: int = 16
    clip_lora_alpha: int = 8
    clip_lora_dropout: float = 0.1

    return_mean_pooled_mask_tokens: bool = False
    mean_pooled_mask_token_layer: int = -1
    mean_pooled_mask_tokens_in_bbox: bool = False
    mean_pooled_mask_tokens_bbox_enlarge: int = 0

    debug_feature_maps: bool = False

    encode_tgt_enc_norm: bool = False
    encode_src_twice: bool = False
    src_tgt_consistency_loss_weight: Optional[float] = None
    src_tgt_feature_map_consistency_loss_weight: Optional[float] = None
    only_encode_shared_tokens: bool = False

    mask_dropped_tokens: bool = False
    less_token_dropout: bool = False

    max_num_training_masks: Optional[int] = None
    num_feature_map_pos_emb: Optional[int] = None
    custom_cross_attn_output_dim: Optional[int] = None
    custom_dino_v2: bool = False
    add_learned_pos_emb_to_feature_map: bool = False
    merge_feature_maps: bool = False
    use_euler_camera_emb: bool = False
    custom_token_modulator_input_dim: Optional[int] = None

    inject_token_positional_information: bool = False
    tgt_positional_information_from_lang: bool = False
    pos_emb_dim: int = 192
    src_tgt_pos_emb_consistency_loss_weight: Optional[float] = None
    src_tgt_pos_emb_loss: bool = False
    norm_vit_features: bool = False
    positional_information_pred_dim: Optional[int] = None
    predict_only_pos_emb_from_lang: bool = True
    cosine_similarity_loss: bool = False
    cosine_loss_weight: Optional[float] = 1.0
    src_tgt_start_loss_step: Optional[int] = 1000
    add_vgg_to_pos_emb: bool = False
    use_cosine_similarity_src_tgt_token_consistency: bool = False
    use_t5_text_encoder_for_token_pred: bool = True
    text_encoder_lora: bool = False
    

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
auto_store(
    ModelConfig,
    model_type=ModelType.BASE_MAPPER,
    name="basemapper_clip_multiscale",
    encoder=builds(ClipFeatureExtractor, return_only=None, populate_full_signature=False),
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="basemapper_vit",
    encoder=builds(ViTFeatureExtractor, return_only="norm", populate_full_signature=False),
    encoder_dim=768,
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper",
    child="basemapper_resnet",
    encoder=builds(ResNetFeatureExtractor, return_only="layer2", pretrained=False, populate_full_signature=False),
    encoder_dim=128,
    unfreeze_resnet=True,
)
store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper_vit",
    child="basemapper_vit_scratch",
    encoder=builds(
        ViTFeatureExtractor,
        model_name="vit_tiny_patch16_224",
        num_classes=0,
        img_size=None,
        return_only=None,
        pretrained=False,
        gradient_checkpointing=True,
        return_nodes={
            "blocks": "blocks",
            "norm": "norm",
            "fc_norm": "fc_norm",
        },
        populate_full_signature=False,
    ),
    encoder_dim=192,
)

store_child_config(
    cls=ModelConfig,
    group="model",
    parent="basemapper_vit_scratch",
    child="basemapper_vit_extra_channels",
    encoder=builds(
        ViTWithExtraChannelsFeatureExtractor,
        num_total_input_channels=5,
        pretrained=True,
        model_name='vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        return_only=None,
        img_size=224,
        return_nodes={
            "blocks.5": "blocks.5",
            "norm": "norm",
        },
        populate_full_signature=False,
        num_classes=0, 
        global_pool=''
    ),
    encoder_dim=192,
)
