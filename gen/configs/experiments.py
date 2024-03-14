from functools import partial

import torch
from hydra_zen import builds

from gen import (IMAGENET_PATH, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH,
                 MOVI_OVERFIT_DATASET_PATH)
from gen.configs.datasets import get_datasets
from gen.configs.old_configs import get_deprecated_experiments
from gen.configs.utils import mode_store
from gen.models.cross_attn.base_inference import compose_two_images, interpolate_latents
from gen.models.encoders.encoder import ResNetFeatureExtractor, ViTFeatureExtractor


def get_override_dict(**kwargs):
    return dict(
        train_dataset=dict(**kwargs),
        validation_dataset=dict(**kwargs),
    )


def get_inference_experiments():
    mode_store(
        name="compose_two_images",
        inference=dict(
            inference_func=compose_two_images
        ),
    )

def get_experiments():
    get_datasets()
    get_inference_experiments()
    get_deprecated_experiments()

    mode_store(name="resnet", model=dict(encoder=builds(ResNetFeatureExtractor, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="sm",
        model=dict(
            decoder_transformer=dict(fused_mlp=False, fused_bias_fc=False),
            fused_mlp=False,
            fused_bias_fc=False,
        ),
        trainer=dict(use_fused_adam=False),
        dataset=dict(train_dataset=dict(batch_size=2, num_workers=0), validation_dataset=dict(batch_size=1, num_workers=0)),
    )

    mode_store(
        name="small_gpu",
        dataset=dict(train_dataset=dict(batch_size=1, num_workers=0), validation_dataset=dict(batch_size=1, num_workers=0)),
        model=dict(
            decoder_transformer=dict(fused_mlp=False, fused_bias_fc=False),
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            token_embedding_dim=768,
            fused_mlp=False,
            fused_bias_fc=False,
        ),
        trainer=dict(enable_xformers_memory_efficient_attention=True, compile=False, eval_on_start=False, gradient_accumulation_steps=1),
        inference=dict(
            visualize_attention_map=False,
            infer_new_prompts=False,
            max_batch_size=1,
            num_masks_to_remove=2,
            num_images_per_prompt=1,
            vary_cfg_plot=False,
        ),
        debug=True,
    )

    mode_store(
        name="sd_15",
        model=dict(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            token_embedding_dim=768,
        )
    )

    mode_store(
        name="unet_finetune",
        dataset=dict(train_dataset=dict(batch_size=8)),
        trainer=dict(learning_rate=1e-7),
        model=dict(freeze_unet=False, unet_lora=False),
    )

    mode_store(
        name="unet_lora",
        model=dict(freeze_unet=True, unet_lora=True, lora_rank=256, gated_cross_attn=False, unfreeze_gated_cross_attn=False),
        trainer=dict(learning_rate=1e-6),
        dataset=dict(train_dataset=dict(batch_size=20)),
    )

    mode_store(
        name="unet_no_lora",
        model=dict(unet_lora=False),
        trainer=dict(learning_rate=1e-5),
        dataset=dict(train_dataset=dict(batch_size=18)),
    )

    mode_store(
        name="multiscale",
        model=dict(
            per_layer_queries=True,
            encoder=dict(return_only=None),
            feature_map_keys=(
                "stage1",
                "stage6",
                "stage12",
                "stage18",
                "stage24",
            ),
        ),
    )

    mode_store(
        name="low_res",
        model=dict(encoder_resolution=224, decoder_resolution=256, decoder_latent_dim=32),
    )

    mode_store(
        name="cur_exp",
        hydra_defaults=["multiscale", "low_res", "movi_medium"],
    )

    mode_store(
        name="gated_cross_attn",
        model=dict(add_pos_emb=True, gated_cross_attn=True, unfreeze_gated_cross_attn=True, lora_rank=4),
        hydra_defaults=["unet_no_lora", "cur_exp"],
    )

    mode_store(
        name="unet_finetune_with_pos_emb",
        model=dict(add_pos_emb=True, finetune_unet_with_different_lrs=True, decoder_transformer=dict(depth=4)),
        hydra_defaults=["unet_finetune", "cur_exp"],
    )

    mode_store(
        name="vit_tiny_scratch",
        model=dict(
            encoder=dict(model_name="vit_tiny_patch16_224"),
            encoder_dim=192,
            per_layer_queries=True,
            feature_map_keys=(
                "blocks",
                "norm",
            ),
            freeze_clip=False,
        ),
        hydra_defaults=[
            "_self_",
            {"override /model": "basemapper_vit_scratch"},
        ],
    )

    mode_store(
        name="vit_small_scratch",
        model=dict(
            encoder=dict(
                model_name="vit_small_patch16_224",
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.6": "blocks.6",
                    "norm": "norm",
                },
            ),
            encoder_dim=384,
            feature_map_keys=(
                "blocks.0",
                "blocks.6",
                "norm",
            ),
        ),
        hydra_defaults=["vit_tiny_scratch"],
    )

    mode_store(
        name="vit_base_scratch",
        model=dict(
            encoder=dict(
                model_name="vit_base_patch16_224",
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.6": "blocks.6",
                    "norm": "norm",
                },
            ),
            encoder_dim=768,
            feature_map_keys=(
                "blocks.0",
                "blocks.6",
                "norm",
            ),
        ),
        hydra_defaults=["vit_small_scratch"],
    )

    mode_store(
        name="debug_vit_base_scratch",
        model=dict(
            encoder_resolution=448,
            encoder=dict(
                pretrained=False,
                model_name="vit_base_patch16_clip_384",
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.6": "blocks.6",
                    "norm": "norm",
                },
                img_size=448
            ),
            encoder_dim=768,
            feature_map_keys=(
                "blocks.0",
                "blocks.6",
                "norm",
            ),
        ),
        hydra_defaults=["vit_base_scratch"],
    )

    mode_store(
        name="debug_vit_base_clip",
        model=dict(
            encoder=dict(
                model_name="vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
                pretrained=True,
            ),
        ),
        hydra_defaults=["debug_vit_base_scratch"],
    )

    mode_store(
        name="coco_recon_only",
        model=dict(
            num_token_cls=133,
            layer_specialization=True,
            num_conditioning_pairs=8,
            num_layer_queries=2,
            custom_conditioning_map=True,
            per_layer_queries=True,
            unet=True, 
            gated_cross_attn=False,
            unfreeze_gated_cross_attn=False,
            unet_lora=False,
            freeze_unet=False,
            freeze_clip=True,
            encoder=dict(
                return_only=None,
                img_size=384,
                return_nodes={
                    "blocks.5": "blocks.5",
                    "norm": "norm",
                },
            ),
            feature_map_keys=(
                "blocks.5",
                "norm",
            ),
            decoder_latent_dim=64,
            decoder_resolution=512,
            encoder_resolution=384,
            encoder_latent_dim=24,
            decoder_transformer=dict(
                embed_dim=1024
            ),
            lr_finetune_version=2,
            finetune_unet_with_different_lrs=False,
            unfreeze_last_n_clip_layers=6,
            # pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            # token_embedding_dim=768,
            # use_sd_15_tokenizer_encoder=True,
        ),
        dataset=dict(
            reset_validation_dataset_every_epoch=True,
            train_dataset=dict(
                batch_size=36,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=1, 
            learning_rate=1e-4, 
            scale_lr_gpus_grad_accum=False, 
            scale_lr_batch_size=False, 
            checkpointing_steps=5000, 
            eval_every_n_steps=2000, 
            max_train_steps=1000000,
            validate_training_dataset=True,
            compile=True,
            use_fused_adam=False,
            use_8bit_adam=False,
        ),
        hydra_defaults=["no_movi_augmentation", "multiscale", "low_res",  "coco_panoptic", "debug_vit_base_clip"], # "sd_15"
    )

    mode_store(
        name="new_unet_finetune",
        dataset=dict(train_dataset=dict(batch_size=9)),
        trainer=dict(learning_rate=1e-7),
        model=dict(freeze_unet=False, unet_lora=False),
    )

    mode_store(
        name="new_unet_lora",
        model=dict(freeze_unet=True, unet_lora=True, lora_rank=256),
        trainer=dict(learning_rate=1e-6),
        dataset=dict(train_dataset=dict(batch_size=20)),
    )

    mode_store(
        name="disable_all_cond",
        model=dict(layer_specialization=False, per_layer_queries=False, mask_token_conditioning=False, freeze_clip=False, num_token_cls=2),
        trainer=dict(learning_rate=1e-4, eval_on_start=False, max_train_steps=10, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False, profiler_active_steps=2),
        dataset=dict(train_dataset=dict(batch_size=2)),
        hydra_defaults=["new_unet_lora", "sd_15"],
    )

    mode_store(
        name="profiler",
        trainer=dict(learning_rate=1e-4, eval_on_start=False, max_train_steps=10, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False, profiler_active_steps=2),
    )

    mode_store(
        name="eschernet",
        model=dict(
            eschernet=True,
            layer_specialization=False,
            num_conditioning_pairs=1,
            custom_conditioning_map=False,
            per_layer_queries=False,
            freeze_clip=False,
            unfreeze_last_n_clip_layers=None,
            encoder=dict(
                model_name='vit_small_patch16_224.augreg_in21k_ft_in1k',
                pretrained=True,
                return_only=None,
                img_size=256,
                return_nodes={
                    "blocks.5": "blocks.5",
                    "norm": "norm",
                },
            ),
            feature_map_keys=(
                "blocks.5",
                "norm",
            ),
            decoder_resolution=256,
            encoder_resolution=256,
            encoder_latent_dim=16,
            decoder_latent_dim=32,
            decoder_transformer=dict(
                embed_dim=512,
                depth=1,
            ),
            lr_finetune_version=2,
            finetune_unet_with_different_lrs=False,
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            token_embedding_dim=768,
            use_sd_15_tokenizer_encoder=True,
        ),
        inference=dict(
            visualize_attention_map=False
        ),
        trainer=dict(
            eval_every_n_steps=500,
            learning_rate=7.5e-5,
            enable_dynamic_grad_accum=False,
        ),
        dataset=dict(train_dataset=dict(batch_size=128)),
        hydra_defaults=["coco_recon_only", "objaverse", "vit_small_scratch"],
    )

    mode_store(
        name="soda_coco",
        model=dict(
            add_pos_emb=False,
            add_grid_to_input_channels=True,
            encoder=dict(
                img_size=224,
                num_total_input_channels=5,
            ),
            feature_map_keys=(
                "norm",
            ),
            decoder_resolution=256,
            encoder_resolution=224,
            encoder_latent_dim=14,
            decoder_latent_dim=32,
            unfreeze_last_n_clip_layers=None,
            freeze_clip=False,
        ),
        dataset=dict(
            train_dataset=dict(
                batch_size=36,
                augmentation=dict(
                    different_source_target_augmentation=True,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    source_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
                    target_random_scale_ratio=((0.5, 0.9), (0.8, 1.2)),
                    enable_rand_augment=False,
                    enable_rotate=True,
                )
            ),
            validation_dataset=dict(
                augmentation=dict(
                    different_source_target_augmentation=True,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    source_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
                    target_random_scale_ratio=((0.5, 0.9), (0.8, 1.2)),
                    enable_rand_augment=False,
                    enable_rotate=True,
                )
            ),
        ),
        trainer=dict(
            eval_every_n_steps=500,
        ),
        hydra_defaults=["coco_recon_only", "sam_coco_masks", {"override /model": "basemapper_vit_extra_channels"}],
    )

    mode_store(
        name="high_res_coco",
        model=dict(
            encoder=dict(
                model_name="vit_base_patch14_reg4_dinov2.lvd142m",
                img_size=518,
            ),
            encoder_latent_dim=37,
            encoder_resolution=518,
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            token_embedding_dim=768,
            use_sd_15_tokenizer_encoder=True,
            # We pretrained by fine-tuning all layers.
            # unfreeze_last_n_clip_layers=None,
            # freeze_clip=False,
            freeze_clip=True,
            unfreeze_last_n_clip_layers=2,
            masked_self_attention=True
        ),
        trainer=dict(
            learning_rate=1e-4,
            lr_warmup_steps=2000,
            compile=False,
            use_8bit_adam=True,
            use_fused_adam=False,
            eval_every_n_steps=1000,
        ),
        dataset=dict(
            train_dataset=dict(
                batch_size=24,
            ),
        ),
        inference=dict(
            guidance_scale=5.0
        )
    )

    mode_store(
        name="high_res_hypersim",
        model=dict(
            use_sd_15_tokenizer_encoder=True,
            freeze_clip=True,
            unfreeze_last_n_clip_layers=3,
            masked_self_attention=False
        ),
        trainer=dict(
            eval_every_n_steps=1000,
            compile=False,
        ),
        dataset=dict(
            train_dataset=dict(
                batch_size=24,
            ),
        ),
        inference=dict(
            guidance_scale=5.0,
            infer_new_prompts=False,
            vary_cfg_plot=False,
        ),
        hydra_defaults=["coco_recon_only", "high_res_coco", "hypersim"],
    )

    mode_store(
        name="memory",
        trainer=dict(
            fast_eval=True,
            profile_memory=True,
            max_train_steps=3,
            enable_dynamic_grad_accum=False,
            validate_training_dataset=False,
            gradient_accumulation_steps=1,
        ),
    )
