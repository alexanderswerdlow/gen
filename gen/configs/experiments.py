from calendar import leapdays
from functools import partial
from gc import enable
from operator import le
from pathlib import Path

import torch
from hydra_zen import builds

from gen import (IMAGENET_PATH, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH,
                 MOVI_OVERFIT_DATASET_PATH)
from gen.configs.datasets import get_datasets
from gen.configs.old_configs import get_deprecated_experiments
from gen.configs.utils import mode_store
from gen.datasets.augmentation.kornia_augmentation import Augmentation
from gen.datasets.hypersim.hypersim import Hypersim
from gen.metrics.compute_token_features import compute_token_features
from gen.models.cross_attn.base_inference import compose_two_images, interpolate_frames, interpolate_latents
from gen.models.encoders.encoder import ResNetFeatureExtractor, ViTFeatureExtractor
from functools import partial
from gen.datasets.scannetpp.run_sam import scannet_run_sam
from accelerate.utils import PrecisionType

from gen.models.encoders.extra_encoders import IntermediateViT


def get_override_dict(**kwargs):
    return dict(
        train=dict(**kwargs),
        val=dict(**kwargs),
    )


def get_inference_experiments():
    mode_store(
        name="compose_two_images",
        inference=dict(
            inference_func=compose_two_images
        ),
    )

    mode_store(
        name="compute_mask_token_features",
        run_inference=True,
        debug=True,
        inference=dict(
            inference_func=compute_token_features,
            set_seed=True,
            infer_train_dataset=True,
            infer_val_dataset=True,
            gather_results=False,
        ),
        model=dict(
            return_mean_pooled_mask_tokens=True,
        ),
        dataset=dict(
            train=dict(
                batch_size=16,
                subset_size=None,
                num_workers=16,
            ),
            val=dict(
                batch_size=16,
                subset_size=None,
                num_workers=16,
            )
        )
    )

def get_experiments():
    get_datasets()
    get_inference_experiments()
    get_deprecated_experiments()

    mode_store(name="resnet", model=dict(encoder=builds(ResNetFeatureExtractor, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="sm",
        debug=True,
        model=dict(
            decoder_transformer=dict(
                fused_mlp=False,
                fused_bias_fc=False
            ),
            fused_mlp=False,
            fused_bias_fc=False,
            token_modulator=dict(
                fused_mlp=False,
                fused_bias_fc=False
            ),
        ),
        trainer=dict(use_fused_adam=False, fast_eval=True),
        dataset=dict(train=dict(batch_size=2, num_workers=0), val=dict(batch_size=1, num_workers=0)),
    )

    mode_store(
        name="fast",
        model=dict(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            token_embedding_dim=768,
            freeze_unet=True,
            unfreeze_single_unet_layer=True,
            unfreeze_last_n_clip_layers=1,
        ),
        trainer=dict(
            compile=False,
            fast_eval=True,
            enable_dynamic_grad_accum=False,
            gradient_accumulation_steps=1,
            backward_pass=False,
        ),
        inference=dict(
            visualize_attention_map=False,
            infer_new_prompts=False,
            vary_cfg_plot=False,
            max_batch_size=1,
            num_masks_to_remove=2,
            num_images_per_prompt=1,
        ),
        hydra_defaults=["sm"],
    )

    mode_store(
        name="24G",
        debug=True,
        model=dict(
            freeze_unet=True,
            unfreeze_single_unet_layer=True,
        ),
        trainer=dict(
            enable_dynamic_grad_accum=False,
        ),
        dataset=dict(
            train=dict(
                batch_size=8,
            )
        )
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
        dataset=dict(train=dict(batch_size=8)),
        trainer=dict(learning_rate=1e-7),
        model=dict(freeze_unet=False, unet_lora=False),
    )

    mode_store(
        name="unet_lora",
        model=dict(freeze_unet=True, unet_lora=True, lora_rank=256, gated_cross_attn=False, unfreeze_gated_cross_attn=False),
        trainer=dict(learning_rate=1e-6),
        dataset=dict(train=dict(batch_size=20)),
    )

    mode_store(
        name="unet_no_lora",
        model=dict(unet_lora=False),
        trainer=dict(learning_rate=1e-5),
        dataset=dict(train=dict(batch_size=18)),
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
        name="dino_v2_large",
        model=dict(
            encoder=dict(
                model_name="vit_large_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
            ),
            encoder_dim=1024,
            # clip_lora=True,
            # clip_lora_rank=1024,
            freeze_clip=True
        ),
    )

    mode_store(
        name="vit_small_dino",
        model=dict(
            encoder=dict(
                model_name="vit_small_patch14_reg4_dinov2",
                img_size=224,
                return_nodes={
                    "blocks.5": "blocks.5",
                    "norm": "norm",
                },
            ),
            feature_map_keys=(
                "blocks.5",
                "norm",
            ),
            encoder_resolution=224,
            encoder_dim=384,
            encoder_latent_dim=16,
        ),
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
            reset_val_dataset_every_epoch=True,
            train=dict(
                batch_size=36,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=1, 
            learning_rate=1e-4, 
            scale_lr_gpus_grad_accum=False, 
            scale_lr_batch_size=False, 
            checkpointing_steps=2000, 
            eval_every_n_steps=1000, 
            max_train_steps=1000000,
            validate_training_dataset=True,
            compile=True,
            use_8bit_adam=True,
            use_fused_adam=False,
        ),
        hydra_defaults=["no_movi_augmentation", "multiscale", "low_res",  "coco_panoptic", "debug_vit_base_clip"], # "sd_15"
    )

    mode_store(
        name="new_unet_finetune",
        dataset=dict(train=dict(batch_size=9)),
        trainer=dict(learning_rate=1e-7),
        model=dict(freeze_unet=False, unet_lora=False),
    )

    mode_store(
        name="new_unet_lora",
        model=dict(freeze_unet=True, unet_lora=True, lora_rank=256),
        trainer=dict(learning_rate=1e-6),
        dataset=dict(train=dict(batch_size=20)),
    )

    mode_store(
        name="disable_all_cond",
        model=dict(layer_specialization=False, per_layer_queries=False, mask_token_conditioning=False, freeze_clip=False, num_token_cls=2),
        trainer=dict(learning_rate=1e-4, eval_on_start=False, max_train_steps=10, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False, profiler_active_steps=2),
        dataset=dict(train=dict(batch_size=2)),
        hydra_defaults=["new_unet_lora", "sd_15"],
    )

    mode_store(
        name="profiler",
        profile=True,
        trainer=dict(eval_on_start=False, max_train_steps=10, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False, profiler_active_steps=2),
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
        dataset=dict(train=dict(batch_size=128)),
        hydra_defaults=["coco_recon_only", "objaverse", "vit_small_scratch"],
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
            freeze_clip=True,
            unfreeze_last_n_clip_layers=2,
            masked_self_attention=True
        ),
        trainer=dict(
            learning_rate=1e-4,
            lr_warmup_steps=2000,
            compile=False,
            eval_every_n_steps=1000,
        ),
        dataset=dict(
            train=dict(
                batch_size=24,
            ),
        ),
        inference=dict(
            guidance_scale=6.5
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
            train=dict(
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
        dataset=dict(
            train=dict(
                shuffle=False,
            ),
            val=dict(
                shuffle=False,
            )
        ),
        model=dict(
            training_mask_dropout=None,
        )
    )

    mode_store(
        name="eschernet_hypersim",
        model=dict(
            eschernet=True,
            eschernet_6dof=True,
        ),
        inference=dict(
            visualize_attention_map=False,
            num_masks_to_remove=None,
        ),
        trainer=dict(
            eval_every_n_steps=500,
            enable_dynamic_grad_accum=False,
            gradient_accumulation_steps=1,
            cudnn_benchmark=True,
        ),
        dataset=dict(train=dict(batch_size=20), val=dict(num_workers=0)),
        hydra_defaults=["high_res_hypersim", "hypersim_multiview"],
    )

    mode_store(
        name="eschernet_hypersim_low_res",
        model=dict(
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
            encoder_dim=384,
            encoder_resolution=256,
            encoder_latent_dim=16,
            decoder_resolution=256,
            decoder_latent_dim=32,
            decoder_transformer=dict(
                embed_dim=512,
                depth=2,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=2,
        ),
        dataset=dict(
            train=dict(
                batch_size=96,
                camera_trajectory_window=32,
                bbox_overlap_threshold=0.85,
                bbox_area_threshold=1.5,
            ),
            val=dict(
                camera_trajectory_window=32,
                bbox_overlap_threshold=0.85,
                bbox_area_threshold=1.5,
            ),
        ),
        hydra_defaults=["eschernet_hypersim"],
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
            train=dict(
                batch_size=176,
                num_workers=8,
                augmentation=dict(
                    different_src_tgt_augmentation=True,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    enable_rotate=True,
                    enable_rand_augment=False,
                    src_random_scale_ratio=((0.7, 1.0), (0.9, 1.1)),
                    tgt_random_scale_ratio=((0.7, 1.0), (0.9, 1.1)),
                    return_grid=True,
                )
            ),
            val=dict(
                augmentation=dict(
                    different_src_tgt_augmentation=True,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    enable_rotate=True,
                    enable_rand_augment=False,
                    src_random_scale_ratio=((0.7, 1.0), (0.9, 1.1)),
                    tgt_random_scale_ratio=((0.7, 1.0), (0.9, 1.1)),
                    return_grid=True
                )
            ),
        ),
        trainer=dict(
            eval_every_n_steps=500,
        ),
        hydra_defaults=["coco_recon_only", "sam_coco_masks",  "gt_coco_masks", {"override /model": "basemapper_vit_extra_channels"},],
    )

    mode_store(
        name="soda_coco_same_src_tgt",
        dataset=dict(
            train=dict(
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    src_random_scale_ratio=None,
                )
            ),
            val=dict(
                augmentation=dict(
                    different_src_tgt_augmentation=False,
                    src_random_scale_ratio=None,
                )
            ),
        ),
    )

    mode_store(
        name="multiview_scannet",
        hydra_defaults=["eschernet_hypersim", "scannetpp_multiview_dataset"],
        dataset=dict(
            train=dict(
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                num_overlapping_masks=6,
                augmentation=dict(
                    reorder_segmentation=False,
                )
            ),
            val=dict(
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
                top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                num_overlapping_masks=6,
                augmentation=dict(
                    reorder_segmentation=False,
                )
            ),
        ),
        model=dict(
            add_text_tokens=False,
            segmentation_map_size=77,
        )
    )

    mode_store(
        name="concat_hypersim_scannet",
        trainer=dict(
            learning_rate=1e-5,
        ),
        model=dict(
            eschernet=True,
            modulate_src_tokens_with_tgt_pose=True,
            encode_tgt=True,
            src_tgt_consistency_loss_weight=0.1,
        ),
        dataset=dict(
            train=dict(
                return_encoder_normalized_tgt=True,
                num_workers=10,
            ),
            val=dict(
                return_encoder_normalized_tgt=True,
            ),
            additional_train=(
                builds(
                    Hypersim, 
                    populate_full_signature=True,
                    zen_partial=True,
                    repeat_n=50,
                    return_encoder_normalized_tgt="${model.return_encoder_normalized_tgt}",
                    camera_trajectory_window=32,
                    return_different_views=True,
                    bbox_overlap_threshold=0.75,
                    bbox_area_threshold=0.75,
                    object_ignore_threshold=0.0,
                    top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                    num_overlapping_masks=1,
                    augmentation=builds(
                        Augmentation,
                        different_src_tgt_augmentation=False,
                        enable_square_crop=True,
                        center_crop=True,
                        enable_random_resize_crop=False, 
                        enable_horizontal_flip=False,
                        enable_rand_augment=False,
                        enable_rotate=False,
                        src_random_scale_ratio=None,
                        tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                        initial_resolution=512,
                        src_resolution=None,
                        tgt_resolution=None,
                        src_transforms="${get_src_transform:model}",
                        tgt_transforms="${get_tgt_transform:model}",
                        populate_full_signature=True,
                    )
                ),
            ),
            additional_val=(
                builds(
                    Hypersim, 
                    populate_full_signature=True,
                    zen_partial=True,
                    repeat_n=50,
                    return_encoder_normalized_tgt="${model.return_encoder_normalized_tgt}",
                    camera_trajectory_window=32,
                    return_different_views=True,
                    bbox_overlap_threshold=0.75,
                    bbox_area_threshold=0.75,
                    object_ignore_threshold=0.0,
                    top_n_masks_only="${eval:'${model.segmentation_map_size} - 1'}",
                    num_overlapping_masks=1,
                    augmentation=builds(
                        Augmentation,
                        different_src_tgt_augmentation=False,
                        enable_square_crop=True,
                        center_crop=True,
                        enable_random_resize_crop=False, 
                        enable_horizontal_flip=False,
                        enable_rand_augment=False,
                        enable_rotate=False,
                        src_random_scale_ratio=None,
                        tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                        initial_resolution=512,
                        src_resolution=None,
                        tgt_resolution=None,
                        src_transforms="${get_src_transform:model}",
                        tgt_transforms="${get_tgt_transform:model}",
                        populate_full_signature=True,
                    ),
                ),
            ),
        ),
        hydra_defaults=["multiview_scannet"],
    )

    mode_store(
        name="noconcat_hypersim_scannet",
        dataset=dict(
            additional_train=None,
            additional_val=None
        ),
        hydra_defaults=["concat_hypersim_scannet"],
    )

    mode_store(
        name="disable_debug_scannet",
        model=dict(
            encode_tgt=False,
            modulate_src_tokens_with_tgt_pose=False,
            segmentation_map_size=16,
            eschernet=False,
            encode_src_twice=True,
            mask_dropped_tokens=True,
            src_tgt_consistency_loss_weight=1.0,
        ),
        dataset=dict(
            train=dict(
                return_encoder_normalized_tgt=False,
                src_eq_tgt=True,
                distance_threshold = (0.3, 0.1, 0.12, 0.7),
                num_overlapping_masks=1,
            ),
            val=dict(
                return_encoder_normalized_tgt=False,
                src_eq_tgt=True,
                distance_threshold = (0.3, 0.1, 0.12, 0.7),
                num_overlapping_masks=1,
            ),
        ),
        trainer=dict(
            eval_every_n_steps=500,
            learning_rate=5e-6,
        ),
        inference=dict(
            num_single_token_gen=5,
            num_masks_to_remove=5,
            visualize_attention_map=True,
        ),
    )

    mode_store(
        name="base_clip_lora",
        model=dict(
            encoder=builds(
                IntermediateViT,
                populate_full_signature=False,
                model_name="vit_base_patch14_reg4_dinov2.lvd142m",
                img_size=518,
                return_nodes=None,
            ),
            feature_map_keys=(
                "blocks.5",
                "blocks.11",
            ),
            encoder_dim=768,
            encoder_resolution=518,
            encoder_latent_dim=37,
            clip_lora=True,
            freeze_clip=True,
            unfreeze_last_n_clip_layers=None,
            segmentation_map_size=256,
        ),
        dataset=dict(
            train=dict(use_new_seg=True, scenes_slice=None, frames_slice=None),
            val=dict(use_new_seg=True, scenes_slice=None, frames_slice=None),
        ),
        trainer=dict(
            fsdp=True,
            checkpointing_steps=1000,
        ),
        inference=dict(
            num_masks_to_remove=10,
            num_single_token_gen=10,
        )
    )

    mode_store(
        name="high_res_lora",
        model=dict(
             encoder_dim=1024,
             encoder=dict(
                model_name="vit_large_patch14_reg4_dinov2.lvd142m",
             ),
             feature_map_keys=(
                "blocks.11",
                "blocks.23",
            ),
        ),
    )

    mode_store(
        name="multi_frame_interpolation",
        debug=True,
        dataset=dict(
            train=dict(
                src_eq_tgt=False,
                return_encoder_normalized_tgt=True,
                distance_threshold = (0.25, 0.1, 0.1, 0.5),
            ),
            val=dict(
                src_eq_tgt=False,
                return_encoder_normalized_tgt=True,
                distance_threshold = (0.25, 0.1, 0.1, 0.5),
            ),
        ),
        inference=dict(
            inference_func=interpolate_frames,
        )
    )

    mode_store(
        name="run_sam_scannetpp", # To generate SAM masks
        dataset=dict(
            train=dict(
                return_raw_dataset_image=True,
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
            ),
            val=dict(
                return_raw_dataset_image=True,
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
            )
        ),
    )

    mode_store(
        name="scannetpp_local_debug", # For local debugging
        dataset=dict(
            train=dict(
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
                use_segmentation=True,
                return_encoder_normalized_tgt="${model.return_encoder_normalized_tgt}",
                single_scene_debug=True,
            ),
            val=dict(
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
                use_segmentation=True,
                return_encoder_normalized_tgt="${model.return_encoder_normalized_tgt}",
                single_scene_debug=True,
            )
        ),
        model=dict(
            encode_tgt=False,
            modulate_src_tokens_with_tgt_pose=True,
        ),
    )

    mode_store(
        name="nfs",
        run_dataloader_only=True,
        debug=True,
        dataset=dict(
            train=dict(
                scratch_only=False,
                batch_size=32,
                num_workers=4,
            ),
        ),
        trainer=dict(
            mixed_precision=PrecisionType.NO,
        )
    )

    mode_store(
        name="coco_updated_03_26",
        model=dict(
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            token_embedding_dim=768,
            use_sd_15_tokenizer_encoder=True,
            add_text_tokens=False,
            masked_self_attention=False,
            segmentation_map_size=77,
        ),
        inference=dict(
            infer_new_prompts=False,
        )
    )

    mode_store(
        name="inference_aug",
        dataset=dict(
            train=dict(
                augmentation=dict(
                    center_crop=True,
                    reorder_segmentation=False,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=False,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                    different_src_tgt_augmentation=False,
                )
            ),
            val=dict(
                augmentation=dict(
                    center_crop=True,
                    reorder_segmentation=False,
                    enable_random_resize_crop=True,
                    enable_horizontal_flip=False,
                    enable_square_crop=True,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
                    different_src_tgt_augmentation=False,
                )
            ),
        ),
    )

    mode_store(
        name="debug_feature_maps",
        model=dict(
            debug_feature_maps=True,
            encoder=dict(
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.1": "blocks.1",
                    "blocks.2": "blocks.2",
                    "blocks.3": "blocks.3",
                    "blocks.4": "blocks.4",
                    "blocks.5": "blocks.5",
                    "blocks.6": "blocks.6",
                    "blocks.7": "blocks.7",
                    "blocks.8": "blocks.8",
                    "blocks.9": "blocks.9",
                    "blocks.10": "blocks.10",
                    "blocks.11": "blocks.11",
                    "norm": "norm",
                    "blocks": "blocks",
                },
            )
        )
    )

    mode_store(
        name="tmp_disable",
        debug=True,
        model=dict(
            training_mask_dropout=None,
            encode_src_twice=False,
            mask_dropped_tokens=True,
            break_a_scene_masked_loss=True,
        ),
        inference=dict(
            num_masks_to_remove=None,
            num_single_token_gen=None,
        ),
        trainer=dict(
            enable_dynamic_grad_accum=False,
            checkpointing_steps=None,
        )
    )

    mode_store(
        name="tmp_disable_v2",
        debug=True,
        model=dict(
            encode_src_twice=False,
            mask_dropped_tokens=True,
            break_a_scene_masked_loss=False,
        ),
        inference=dict(
            num_masks_to_remove=8,
            num_single_token_gen=8,
        ),
        trainer=dict(
            enable_dynamic_grad_accum=False,
            checkpointing_steps=1000,
        )
    )

    mode_store(
        name="merge_extra_masks",
        debug=True,
        dataset=dict(
            train=dict(merge_masks=True),
            val=dict(merge_masks=True),
        ),
    )

    mode_store(
        name="old_data",
        dataset=dict(
            train=dict(
                use_new_seg=False,
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
            ),
            val=dict(
                use_new_seg=False,
                scenes_slice=(0, None, 4),
                frames_slice=(0, None, 5),
            ),
        ),
    )

    mode_store(
        name="vit_small_dino_finetune",
        model=dict(
            encoder=builds(
                ViTFeatureExtractor,
                num_classes=0,
                return_only=None,
                pretrained=False,
                gradient_checkpointing=True,
                model_name="vit_small_patch14_reg4_dinov2",
                img_size=224,
                return_nodes={
                    "blocks.5": "blocks.5",
                    "norm": "norm",
                },
                populate_full_signature=False,
            ),
            feature_map_keys=(
                "blocks.5",
                "norm",
            ),
            encoder_resolution=224,
            encoder_dim=384,
            encoder_latent_dim=16,
            freeze_clip=False,
            clip_lora=False,
            unfreeze_last_n_clip_layers=None,
        ),
        hydra_defaults=[
            "_self_",
            {"override /model": "basemapper_vit_scratch"},
        ],
    )

    mode_store(
        name="efficient_training",
        trainer=dict(
            fsdp=True,
            checkpointing_steps=1000,
            enable_dynamic_grad_accum=False,
        ),
        inference=dict(
            num_masks_to_remove=10,
            num_single_token_gen=10,
        ),
    )

    mode_store(
        name="sam_segmentaton",
        model=dict(
            segmentation_map_size=255,
        ),
        dataset=dict(
            train=dict(scenes_slice=None, frames_slice=None),
            val=dict(scenes_slice=None, frames_slice=None),
        ),
    )

    mode_store(
        name="instance_segmentation",
        model=dict(
            segmentation_map_size=255,
        ),
        dataset=dict(
            train=dict(
                allow_instance_seg=True,
                return_only_instance_seg=True,
                scenes_slice=None,
                frames_slice=None,
                distance_threshold=(0.25, 0.35, 0.25, 0.0),
            ),
            val=dict(
                allow_instance_seg=True,
                return_only_instance_seg=True,
                scenes_slice=None,
                frames_slice=None,
                distance_threshold=(0.25, 0.35, 0.25, 0.0)
            ),
        ),
    )

    mode_store(
        name="exp_1_base",
        debug=True,
        model=dict(
            segmentation_map_size=255,
            max_num_training_masks=16,
            
            mask_dropped_tokens=False,
            break_a_scene_masked_loss=False,

            encode_src_twice=False,
            encode_tgt=False,
            src_tgt_consistency_loss_weight=None,
            
            only_encode_shared_tokens=True,

            modulate_src_tokens_with_tgt_pose=True,
            modulate_src_tokens_with_mlp=True,
            
            less_token_dropout=True,
        ),
        dataset=dict(
            train=dict(
                src_eq_tgt=False,
            ),
            val=dict(
                src_eq_tgt=False,
            ),
        ),
        trainer=dict(
            fsdp=True,
            checkpointing_steps=1000,
            enable_dynamic_grad_accum=False,
        ),
        inference=dict(
            num_masks_to_remove=10,
            num_single_token_gen=10,
        ),
        hydra_defaults=[
           "noconcat_hypersim_scannet",
           "disable_debug_scannet",
        ],
    )

    mode_store(
        name="exp_0_8",
        model=dict(
            only_encode_shared_tokens=False,
            training_mask_dropout=None,
        ),
        dataset=dict(
            train=dict(
                distance_threshold=(0.20, 0.1, 0.10, 0.3),
            ),
            val=dict(
                distance_threshold=(0.20, 0.1, 0.10, 0.3),
            ),
        ),
        hydra_defaults=[
           "exp_1_base",
           "sam_segmentaton",
        ],
    )

    mode_store(
        name="exp_0_9",
        model=dict(
            only_encode_shared_tokens=True,
            training_mask_dropout=None,
        ),
        dataset=dict(
            train=dict(
                distance_threshold=(0.8, 0.70, 0.25, 0.0),
                use_colmap_poses=True,
            ),
            val=dict(
                distance_threshold=(0.8, 0.70, 0.25, 0.0),
                use_colmap_poses=True,
            ),
        ),
        hydra_defaults=[
           "exp_1_base",
           "instance_segmentation",
        ],
    )

    mode_store(
        name="exp_0_9_1",
        model=dict(
            only_encode_shared_tokens=True,
            training_mask_dropout=None,
            encode_tgt=True,
            src_tgt_consistency_loss_weight=100.0,
            modulate_src_tokens_with_film=True,
            modulate_src_tokens_with_mlp=False,
        ),
        dataset=dict(
            train=dict(
                distance_threshold=(0.5, 0.50, 0.25, 0.0),
                use_colmap_poses=True,
                return_encoder_normalized_tgt=True,
            ),
            val=dict(
                distance_threshold=(0.5, 0.50, 0.25, 0.0),
                use_colmap_poses=True,
                return_encoder_normalized_tgt=True,
            ),
        ),
        hydra_defaults=[
           "exp_1_base",
           "instance_segmentation",
        ],
    )

    mode_store(
        name="freeze_exp",
        model=dict(
            freeze_unet=True,
            unfreeze_last_n_clip_layers=None,
            freeze_token_encoder=True,
        ),
        dataset=dict(
            train=dict(
                batch_size=32
            ),
        ),
        trainer=dict(
            eval_every_n_steps=500,
            learning_rate=1e-4,
            log_gradients=50
        )
    )

    mode_store(
        name="load_ckpt",
        trainer=dict(
            ckpt=Path("/projects/katefgroup/aswerdlo/gen/checkpoints/debug_2024-04-01_17_27_31/yRVLCngRkN/checkpoint_28000/state/pytorch_model.bin"),
            strict_load=False,
        ),
    )