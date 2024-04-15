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
from gen.datasets.calvin.calvin import CalvinDataset
from gen.datasets.coco.coco_panoptic import CocoPanoptic
from gen.datasets.hypersim.hypersim import Hypersim
from gen.datasets.kubrics.movi_dataset import MoviDataset
from gen.datasets.scannetpp.scannetpp import ScannetppIphoneDataset
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
        hydra_defaults=["no_movi_augmentation", "multiscale", "low_res",  "coco_panoptic", "debug_vit_base_clip"],
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
            encode_tgt_enc_norm=True,
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
            additional_train=dict(
                hypersim=builds(
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
            additional_val=dict(
                hypersim=builds(
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
            encode_tgt_enc_norm=False,
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
            additional_train=None,
            additional_val=None,
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
            encode_tgt_enc_norm=False,
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
            encode_tgt_enc_norm=False,
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
            checkpointing_steps=None,
            enable_dynamic_grad_accum=False,
        ),
        inference=dict(
            num_masks_to_remove=10,
            num_single_token_gen=10,
            compute_quantitative_token_metrics=True,
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
            encode_tgt_enc_norm=True,
            src_tgt_consistency_loss_weight=1.0,
            modulate_src_tokens_with_film=True,
            modulate_src_tokens_with_mlp=False,
        ),
        dataset=dict(
            train=dict(
                batch_size=12,
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
        name="exp_0_9_2",
        debug=True,
        model=dict(
            max_num_training_masks=None,
            disable_unet_during_training=True,
            src_tgt_consistency_loss_weight=1.0,
            diffusion_loss_weight=0.0,
            modulate_src_tokens_with_film=False,
            modulate_src_tokens_with_mlp=False,
            token_modulator=dict(final_norm=False, depth=2)
        ),
        dataset=dict(
            train=dict(
                batch_size=6,
            ),
            val=dict(
                subset_size=64,
            )
        ),
        trainer=dict(
            eval_on_start=True,
            custom_inference_every_n_steps=2000,
            eval_every_n_steps=4000,
            learning_rate=5e-6,
            log_gradients=100,
            set_even_batches_false=True,
        ),
        inference=dict(
            num_masks_to_remove=None,
            visualize_attention_map=False,
            visualize_embeds=False,
            infer_new_prompts=False,
            save_prompt_embeds=False,
            num_single_token_gen=None,
            vary_cfg_plot=False,
        ),
        hydra_defaults=[
           "exp_0_9_1",
           "freeze_exp",
           "load_ckpt"
        ],
    )


    mode_store(
        name="exp_0_9_2_single",
        dataset=dict(
            train=dict(
                batch_size=1,
                overfit_subset_size=1,
            ),
            val=dict(
                subset_size=1,
            )
        ),
        hydra_defaults=[
           "exp_0_9_2",
        ],
    )

    mode_store(
        name="exp_0_9_2_unfrozen_enc",
        debug=True,
        model=dict(
            unfreeze_last_n_clip_layers=8,
            freeze_token_encoder=False,
            disable_unet_during_training=False,

            src_tgt_consistency_loss_weight=1.0,
            diffusion_loss_weight=0.01,
        ),
        hydra_defaults=[
           "exp_0_9_2",
           "subset_exp"
        ],
    )

    mode_store(
        name="subset_exp",
        dataset=dict(
            train=dict(
                overfit_subset_size=12,
            ),
        ),
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
                batch_size=48
            ),
        ),
        trainer=dict(
            eval_every_n_steps=500,
            learning_rate=2e-5,
            log_gradients=100
        )
    )

    mode_store(
        name="feature_map_warp",
        model=dict(
            freeze_unet=True,
            freeze_token_encoder=True,
            modulate_src_tokens_with_tgt_pose=False,
            modulate_src_feature_map=True,
            src_tgt_consistency_loss_weight=None,
            src_tgt_feature_map_consistency_loss_weight=1.0,
            unfreeze_last_n_clip_layers=None,
        ),
        dataset=dict(
            train=dict(
                batch_size=4,
                overfit_subset_size=4,
            ),
        ),
        trainer=dict(
            eval_every_n_steps=2000,
            strict_load=False,
        ),
        hydra_defaults=[
           "exp_0_9_2",
           "custom_vit"
        ],
    )

    mode_store(
        name="feature_map_warp_unfreeze",
        debug=True,
        model=dict(
            freeze_token_encoder=False,
            disable_unet_during_training=False,
            diffusion_loss_weight=1.0,
        ),
        hydra_defaults=[
           "feature_map_warp",
        ],
    )

    mode_store(
        name="load_ckpt",
        trainer=dict(
            ckpt=Path("/projects/katefgroup/aswerdlo/gen/checkpoints/debug_2024-04-01_17_27_31/yRVLCngRkN/checkpoint_28000/state/pytorch_model.bin"),
            strict_load=False,
        ),
    )


    mode_store(
        name='custom_vit',
        model=dict(
            feature_map_keys=('blocks.5', "norm"),
            num_feature_map_pos_emb=2,
            encoder=dict(
                num_classes=0,
                return_only=None,
                pretrained=True,
                gradient_checkpointing=True,
                model_name="vit_base_patch14_reg4_dinov2.lvd142m",
                img_size=518,
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.5": "blocks.5",
                    "blocks.6": "blocks.6",
                    "norm": "norm",
                    "mid_blocks": "mid_blocks",
                    "final_norm": "final_norm",
                },
            ),
        )
    )


    def get_train_aug():
        return builds(
            Augmentation,
            reorder_segmentation=False,
            different_src_tgt_augmentation="${dataset.train.augmentation.different_src_tgt_augmentation}",
            enable_square_crop="${dataset.train.augmentation.enable_square_crop}",
            center_crop="${dataset.train.augmentation.center_crop}",
            enable_random_resize_crop="${dataset.train.augmentation.enable_random_resize_crop}", 
            enable_horizontal_flip="${dataset.train.augmentation.enable_horizontal_flip}",
            enable_rand_augment="${dataset.train.augmentation.enable_rand_augment}",
            enable_rotate="${dataset.train.augmentation.enable_rotate}",
            src_random_scale_ratio="${dataset.train.augmentation.src_random_scale_ratio}",
            tgt_random_scale_ratio="${dataset.train.augmentation.tgt_random_scale_ratio}",
            initial_resolution=512,
            src_resolution=None,
            tgt_resolution=None,
            src_transforms="${get_src_transform:model}",
            tgt_transforms="${get_tgt_transform:model}",
            rotation_range="${dataset.train.augmentation.rotation_range}",
            populate_full_signature=True,
        )
    
    def get_val_aug():
        return builds(
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


    mode_store(
        name="single_image_pretraining",
        model=dict(
            encoder=builds(
                ViTFeatureExtractor,
                img_size=518,
                pretrained=False,
                model_name="vit_large_patch14_dinov2.lvd142m",
                return_only=None,
                return_nodes={
                    "blocks.0": "blocks.0",
                    "blocks.11": "blocks.11",
                    "blocks.17": "blocks.17",
                    "blocks.23": "blocks.23",
                },
            ),
            encoder_dim=4096,
            decoder_latent_dim=32,
            decoder_resolution=256,
            encoder_resolution=518,
            encoder_latent_dim=37,

 
            custom_conditioning_map=False,
            unet=True,
            gated_cross_attn=False,
            unfreeze_gated_cross_attn=False,
            unet_lora=False,
            freeze_unet=False,
            freeze_clip=True,
            per_layer_queries=False,
            custom_dino_v2=True,
            feature_map_keys=(
                "blocks.0",
                "blocks.11",
                "blocks.17",
                "blocks.23",
            ),
            decoder_transformer=dict(
                add_self_attn=False,
                add_cross_attn=True,
                depth=4,
                num_heads=16,
            ),
            lr_finetune_version=2,
            finetune_unet_with_different_lrs=False,
            unfreeze_last_n_clip_layers=None,
            cross_attn_dim=2048,
            use_sd_15_tokenizer_encoder=True,
            masked_self_attention=False,
            add_text_tokens=False,
            eschernet=False,
            modulate_src_tokens_with_tgt_pose=False,
            encode_tgt_enc_norm=False,
            segmentation_map_size=255,
    
            revision="v2.0",
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            token_embedding_dim=768,

            max_num_training_masks=12,
            only_encode_shared_tokens=False,
            less_token_dropout=True,
            break_a_scene_masked_loss=False,
            mask_dropped_tokens=True,

            encode_src_twice=False,
            src_tgt_consistency_loss_weight=None,
            add_pos_emb=False,
            add_learned_pos_emb_to_feature_map=False,
            merge_feature_maps=True,
            num_layer_queries=1,

            # custom_cross_attn_output_dim=4096,
            # layer_specialization=True,
            # num_conditioning_pairs=8,
            
            custom_cross_attn_output_dim=768,
            layer_specialization=False,
            num_conditioning_pairs=1,
        ),
        dataset=dict(
            reset_val_dataset_every_epoch=True,
            train=builds(
                CocoPanoptic,
                populate_full_signature=True,
                repeat_single_dataset_n_times=10,
                num_workers=8,
                batch_size=36,
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                use_preprocessed_masks=True,
                scratch_only=False,
                num_overlapping_masks=1,
                preprocessed_mask_type="custom_postprocessed",
                return_encoder_normalized_tgt=False,
                augmentation=builds(
                    Augmentation,
                    reorder_segmentation=False,
                    different_src_tgt_augmentation=False,
                    enable_square_crop=True,
                    center_crop=False,
                    enable_random_resize_crop=True, 
                    enable_horizontal_flip=True,
                    enable_rand_augment=False,
                    enable_rotate=False,
                    src_random_scale_ratio=None,
                    tgt_random_scale_ratio=((0.7, 1.0), (0.9, 1.1)),
                    initial_resolution=512,
                    src_resolution=None,
                    tgt_resolution=None,
                    src_transforms="${get_src_transform:model}",
                    tgt_transforms="${get_tgt_transform:model}",
                    rotation_range=0,
                    populate_full_signature=True,
                ),
                allowed_keys=("tgt_pixel_values", "src_pixel_values", "tgt_mask", "src_mask", "src_segmentation", "tgt_segmentation", "input_ids", "metadata", "valid", "src_valid", "has_global_instance_ids", "tgt_enc_norm_pixel_values", "tgt_enc_norm_segmentation", "tgt_enc_norm_valid"),
            ),
            val=builds(
                CocoPanoptic,
                populate_full_signature=True,
                repeat_single_dataset_n_times=10,
                batch_size=1,
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                use_preprocessed_masks=True,
                scratch_only=False,
                num_overlapping_masks=1,
                preprocessed_mask_type="custom_postprocessed",
                return_encoder_normalized_tgt=False,
                augmentation=get_val_aug(),
                allowed_keys=("tgt_pixel_values", "src_pixel_values", "tgt_mask", "src_mask", "src_segmentation", "tgt_segmentation", "input_ids", "metadata", "valid", "src_valid", "has_global_instance_ids", "tgt_enc_norm_pixel_values", "tgt_enc_norm_segmentation", "tgt_enc_norm_valid"),
            ),
            additional_train=dict(
                scannetpp=builds(
                    ScannetppIphoneDataset,
                    zen_partial=True,
                    image_pairs_per_scene=16384,
                    top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    src_eq_tgt=True,
                    distance_threshold=(0.3, 0.1, 0.12, 0.7),
                    num_overlapping_masks=1,
                    augmentation=get_train_aug(),
                ),
                hypersim=builds(
                    Hypersim, 
                    populate_full_signature=True,
                    zen_partial=True,
                    repeat_n=50,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    camera_trajectory_window=32,
                    return_different_views=False,
                    bbox_overlap_threshold=0.75,
                    bbox_area_threshold=0.75,
                    object_ignore_threshold=0.0,
                    top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                    num_overlapping_masks=1,
                    augmentation=get_train_aug(),
                ),
                kubrics=builds(
                    MoviDataset,
                    populate_full_signature=True,
                    zen_partial=True,
                    dataset="movi_e",
                    path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
                    num_objects=23,
                    num_frames=24,
                    num_cameras=1,
                    fake_return_n=10,
                    multi_camera_format=True,
                    cache_in_memory=True,
                    cache_instances_in_memory=False,
                    num_subset=None,
                    return_tensorclass=True,
                    return_multiple_frames=None,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    augmentation=get_train_aug(),
                ),
                calvin=builds(
                    CalvinDataset,
                    populate_full_signature=True,
                    zen_partial=True,
                    src_eq_tgt=True,
                    fake_return_n=200000,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    augmentation=get_train_aug(),
                ),
            ),
            additional_val=dict(
                scannetpp=builds(
                    ScannetppIphoneDataset,
                    image_pairs_per_scene=16384,
                    top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    src_eq_tgt=True,
                    distance_threshold=(0.3, 0.1, 0.12, 0.7),
                    num_overlapping_masks=1,
                    augmentation=get_val_aug(),
                    zen_partial=True,
                ),
                hypersim=builds(
                    Hypersim, 
                    populate_full_signature=True,
                    zen_partial=True,
                    repeat_n=10,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    camera_trajectory_window=32,
                    return_different_views=False,
                    bbox_overlap_threshold=0.75,
                    bbox_area_threshold=0.75,
                    object_ignore_threshold=0.0,
                    top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                    num_overlapping_masks=1,
                    augmentation=get_val_aug(),
                ),
                kubrics=builds(
                    MoviDataset,
                    populate_full_signature=True,
                    zen_partial=True,
                    dataset="movi_e",
                    path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
                    num_objects=23,
                    num_frames=24,
                    num_cameras=1,
                    fake_return_n=10,
                    multi_camera_format=True,
                    cache_in_memory=True,
                    cache_instances_in_memory=False,
                    num_subset=None,
                    return_tensorclass=True,
                    return_multiple_frames=None,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    augmentation=get_val_aug(),
                ),
                calvin=builds(
                    CalvinDataset,
                    populate_full_signature=True,
                    zen_partial=True,
                    src_eq_tgt=True,
                    fake_return_n=20000,
                    return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                    augmentation=get_val_aug(),
                ),
            ),
        ),
        trainer=dict(
            scale_lr_gpus_grad_accum=False, 
            scale_lr_batch_size=False, 
            checkpointing_steps=2000, 
            eval_every_n_steps=1000, 
            max_train_steps=1000000,
            validate_training_dataset=True,
            use_8bit_adam=True,
            use_fused_adam=False,
            lr_warmup_steps=5000,
            compile=False,
            enable_dynamic_grad_accum=False,
            gradient_accumulation_steps=4,
            cudnn_benchmark=True,
            learning_rate=5e-6,
            fsdp=True,
        ),
        inference=dict(
            guidance_scale=7.0,
            infer_new_prompts=False,
            vary_cfg_plot=True,
            visualize_attention_map=False,
            num_single_token_gen=4,
            num_masks_to_remove=6,
        ),
        hydra_defaults=[
           {"override /model": "basemapper_vit_scratch"},
           {"override /dataset": "coco_panoptic"},
        ],
    )

    mode_store(
        name="single_image_pretraining_v1",
        model=dict(
            encode_src_twice=True,
            src_tgt_consistency_loss_weight=25.0,
            src_tgt_start_loss_step=0,
            max_num_training_masks=16,
            use_cosine_similarity_src_tgt_token_consistency=True,
            only_encode_shared_tokens=False,
        ),
        trainer=dict(
            checkpointing_steps=1000,
        ),
        inference=dict(
            num_images_per_prompt=4
        ),
        hydra_defaults=["single_image_pretraining"]
    )

    mode_store(
        name="single_image_pretraining_v1_1",
        model=dict(
            mask_dropped_tokens=True,
        ),
        hydra_defaults=["single_image_pretraining_v1"]
    )

    def get_large_train_aug():
        return builds(
            Augmentation,
            different_src_tgt_augmentation=True,
            enable_random_resize_crop=True, 
            enable_horizontal_flip=True,
            enable_rand_augment=False,
            enable_rotate=True,
            rotation_range=5,
            src_random_scale_ratio=((0.6, 1.0), (0.9, 1.1)),
            tgt_random_scale_ratio=((0.6, 1.0), (0.9, 1.1)),
            src_transforms="${get_src_transform:model}",
            tgt_transforms="${get_tgt_transform:model}",
        )

    mode_store(
        # Here we randomly augment src/tgt and enforce that the tokens have the same representation with some injected positional offset
        name="single_image_pretraining_v2",
        model=dict(
            encode_src_twice=False,
            encode_tgt_enc_norm=True,
            only_encode_shared_tokens=True,
            inject_token_positional_information=True,
            src_tgt_consistency_loss_weight=1.0,
            src_tgt_start_loss_step=4000,
            use_cosine_similarity_src_tgt_token_consistency=True,
        ),
        inference=dict(
            visualize_attention_map=False,
            visualize_positional_control=True,
        ),
        dataset=dict(
            train=dict(return_encoder_normalized_tgt=True, augmentation=get_large_train_aug()),
            val=dict(return_encoder_normalized_tgt=True, augmentation=get_val_aug()),
        ),
        trainer=dict(
            checkpointing_steps=1000,
        ),
        hydra_defaults=["single_image_pretraining_v1"]
    )

    mode_store(
        name="single_image_pretraining_v3",
        model=dict(
            tgt_positional_information_from_lang=True,
            token_modulator=dict(final_norm=True, num_heads=8, depth=4),
            pos_emb_dim=64,
            positional_information_pred_dim=768+64,
            src_tgt_pos_emb_consistency_loss_weight=10.0,
            src_tgt_consistency_loss_weight=100.0,
            cosine_loss_weight=50.0,
            training_mask_dropout=None,
            modulate_src_tokens_loss_after_layer_specialization=False,
            weighted_object_loss=False,
            src_tgt_start_loss_step=1000,
            src_tgt_pos_emb_loss=True,
            predict_only_pos_emb_from_lang=True,
            use_t5_text_encoder_for_token_pred=True,
        ),
        inference=dict(
            guidance_scale=7.5,
        ),
        trainer=dict(
            gradient_accumulation_steps=2,
        ),
        hydra_defaults=["single_image_pretraining_v2"]
    )

    mode_store(
        name="single_image_pretraining_v4",
        model=dict(
            src_tgt_pos_emb_loss=False,
            src_tgt_consistency_loss_weight=None,
        ),
        hydra_defaults=["single_image_pretraining_v3"]
    )

    mode_store(
        name="single_image_pretraining_v5",
        model=dict(
            predict_only_pos_emb_from_lang=False,
            inject_token_positional_information=False,
            positional_information_pred_dim=768,
            use_t5_text_encoder_for_token_pred=False,
            token_modulator=dict(final_norm=False),
            src_tgt_consistency_loss_weight=25,
            text_encoder_lora=True,
            freeze_unet=True,
            freeze_token_encoder=True,
        ),
        inference=dict(
            visualize_positional_control=False,
        ),
        hydra_defaults=["single_image_pretraining_v4"]
    )

    mode_store(
        name="hypersim_nvs",
        model=dict(
            only_encode_shared_tokens=True,
            modulate_src_tokens_with_tgt_pose=True,
            lr_finetune_version=3,
            finetune_unet_with_different_lrs=True,
            use_euler_camera_emb=True,
            custom_token_modulator_input_dim=4224,
            token_modulator=dict(
                num_heads=22,
            )
        ),
        dataset=dict(
            reset_val_dataset_every_epoch=True,
            train=builds(
                Hypersim, 
                populate_full_signature=True,
                zen_partial=True,
                batch_size=64,
                repeat_n=75,
                return_encoder_normalized_tgt=True,
                camera_trajectory_window=32,
                return_different_views=True,
                bbox_overlap_threshold=0.75,
                bbox_area_threshold=0.75,
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                num_overlapping_masks=1,
                augmentation=get_val_aug(),
            ),
            val=builds(
                Hypersim, 
                populate_full_signature=True,
                zen_partial=True,
                batch_size=1,
                repeat_n=50,
                return_encoder_normalized_tgt=True,
                camera_trajectory_window=32,
                return_different_views=True,
                bbox_overlap_threshold=0.75,
                bbox_area_threshold=0.75,
                object_ignore_threshold=0.0,
                top_n_masks_only="${eval:'${model.segmentation_map_size}'}",
                num_overlapping_masks=1,
                augmentation=get_val_aug(),
            ),
            additional_train=None,
            additional_val=None,
        ),
        trainer=dict(strict_load=False),
        hydra_defaults=["single_image_pretraining", {"override /dataset": "hypersim"}],
    )

    mode_store(
        name="freeze_nvs",
        model=dict(
            finetune_unet_with_different_lrs=False,
            freeze_unet=True,
            freeze_token_encoder=True,
        ),
    )

    def get_zoom_val_aug():
        return builds(
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
            initial_resolution=144,
            enable_zoom_crop=True,
            src_resolution=None,
            tgt_resolution=None,
            src_transforms="${get_src_transform:model}",
            tgt_transforms="${get_tgt_transform:model}",
            populate_full_signature=True,
        )

    mode_store(
        name="calvin",
        dataset=dict(
            reset_val_dataset_every_epoch=False,
            train=builds(
                CalvinDataset,
                populate_full_signature=True,
                zen_partial=True,
                num_workers=6,
                batch_size=32,
                src_eq_tgt=False,
                return_encoder_normalized_tgt=True,
                augmentation=get_zoom_val_aug(),
            ),
            val=builds(
                CalvinDataset,
                populate_full_signature=True,
                zen_partial=True,
                batch_size=1,
                src_eq_tgt=False,
                return_encoder_normalized_tgt="${dataset.train.return_encoder_normalized_tgt}",
                augmentation=get_zoom_val_aug(),
            ),
            additional_train=None,
            additional_val=None,
        ),
        hydra_defaults=[{"override /dataset": "calvin"}],
    )

    mode_store(
        name="kubrics",
        dataset=dict(
            reset_val_dataset_every_epoch=True,
            train=builds(
                MoviDataset,
                populate_full_signature=True,
                zen_partial=True,
                num_workers=6,
                batch_size=32,
                dataset="movi_e",
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
                num_objects=23,
                num_frames=24,
                num_cameras=1,
                multi_camera_format=True,
                cache_in_memory=True,
                cache_instances_in_memory=False,
                num_subset=None,
                return_tensorclass=True,
                return_multiple_frames=None,
                return_encoder_normalized_tgt=True,
                augmentation=get_val_aug(),
            ),
            val=builds(
                MoviDataset,
                populate_full_signature=True,
                zen_partial=True,
                batch_size=1,
                dataset="movi_e",
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
                num_objects=23,
                num_frames=24,
                num_cameras=1,
                multi_camera_format=True,
                cache_in_memory=True,
                cache_instances_in_memory=False,
                num_subset=None,
                return_tensorclass=True,
                return_multiple_frames=None,
                return_encoder_normalized_tgt=True,
                augmentation=get_val_aug(),
            ),
            additional_train=None,
            additional_val=None,
        ),
        hydra_defaults=[{"override /dataset": "movi_e"}],
    )

    mode_store(
        name='kubrics_multiview',
        model=dict(
            encode_tgt_enc_norm=True,
            src_tgt_consistency_loss_weight=1000,
            modulate_src_tokens_loss_after_layer_specialization=False,
            training_mask_dropout=None,
        ),
        dataset=dict(
            train=dict(
                return_multiple_frames=2,
            ),
            val=dict(
                return_multiple_frames=2,
            ),
        ),
        hydra_defaults=["kubrics"],
    )