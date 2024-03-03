from hydra_zen import builds

from gen import (
    IMAGENET_PATH,
    MOVI_DATASET_PATH,
    MOVI_MEDIUM_PATH,
    MOVI_MEDIUM_SINGLE_OBJECT_PATH,
    MOVI_OVERFIT_DATASET_PATH,
    MOVI_MEDIUM_TWO_OBJECTS_PATH,
)
from gen.configs.utils import mode_store, store_child_config
from gen.models.encoders.encoder import ResNetFeatureExtractor, ViTFeatureExtractor
import torch

def get_override_dict(**kwargs):
    return dict(
        train_dataset=dict(**kwargs),
        validation_dataset=dict(**kwargs),
    )


shared_overfit_movi_args = dict(
    custom_split="train",
    path=MOVI_OVERFIT_DATASET_PATH,
    num_objects=1,
    augmentation=dict(minimal_source_augmentation=True, enable_crop=False),
)

shared_movi_args = dict(
    path=MOVI_DATASET_PATH,
    num_objects=23,
    augmentation=dict(minimal_source_augmentation=True, enable_crop=True, enable_horizontal_flip=False),
)


def get_deprecated_experiments():

    mode_store(
        name="break_a_scene",
        model=dict(
            lora_rank=128,
            break_a_scene_cross_attn_loss=True,
            break_a_scene_masked_loss=True,
        ),
        trainer=dict(enable_xformers_memory_efficient_attention=False, eval_every_n_steps=250, max_train_steps=10000, checkpointing_steps=1000),
        dataset=dict(
            train_dataset=dict(batch_size=6, augmentation=dict(enable_crop=False, enable_horizontal_flip=True)),
            validation_dataset=dict(augmentation=dict(enable_crop=False, enable_horizontal_flip=True)),
        ),
        inference=dict(infer_new_prompts=True, num_masks_to_remove=6, save_prompt_embeds=False),
    )

    mode_store(
        name="break_a_scene_two_stage",
        model=dict(unet_lora=False, unfreeze_unet_after_n_steps=100),
        trainer=dict(
            finetune_learning_rate=2e-6,
            learning_rate=5e-4,
            lr_warmup_steps=50,
            enable_xformers_memory_efficient_attention=True,
            eval_every_n_steps=50,
            max_train_steps=10000,
            checkpointing_steps=1000,
            log_gradients=25,
        ),
        dataset=dict(train_dataset=dict(batch_size=20)),
        hydra_defaults=["break_a_scene"],
    )


def get_datasets():  # TODO: These do not need to be global configs
    mode_store(
        name="movi",
        dataset=dict(
            train_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False), multi_camera_format=False),
            validation_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False), multi_camera_format=False),
        ),
        model=dict(
            segmentation_map_size=24,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "movi_e"},
        ],
    )

    mode_store(
        name="movi_full",
        dataset=dict(
            train_dataset=dict(custom_split="train", **shared_movi_args),
            validation_dataset=dict(custom_split="validation", **shared_movi_args),
        ),
        hydra_defaults=["movi"],
    )

    mode_store(
        name="movi_overfit",
        dataset=dict(
            train_dataset=dict(batch_size=20, subset_size=None, **shared_overfit_movi_args),
            validation_dataset=dict(subset_size=4, **shared_overfit_movi_args),
            overfit=True,
        ),
        hydra_defaults=["movi"],
    )

    mode_store(
        name="movi_single_scene",
        dataset=dict(train_dataset=dict(subset=("video_0015",), fake_return_n=8), validation_dataset=dict(subset=("video_0015",), fake_return_n=8)),
        hydra_defaults=["movi_overfit"],
    )

    mode_store(
        name="movi_single_frame",
        dataset=dict(train_dataset=dict(num_dataset_frames=1, fake_return_n=256), validation_dataset=dict(num_dataset_frames=1, fake_return_n=256)),
        hydra_defaults=["movi_single_scene"],
    )

    mode_store(
        name="movi_augmentation",
        dataset=dict(
            train_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=True)),
            validation_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=True)),
        ),
    )

    mode_store(
        name="no_movi_augmentation",
        dataset=dict(
            train_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False)),
            validation_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False)),
        ),
    )

    mode_store(
        name="movi_validate_single_scene",
        dataset=dict(
            validation_dataset=dict(
                subset_size=4,
                subset=("video_0018",),
                fake_return_n=8,
                random_subset=False,
                augmentation=dict(enable_horizontal_flip=False, enable_crop=False, minimal_source_augmentation=True),
            ),
        ),
        hydra_defaults=["movi_single_scene"],
    )

    mode_store(
        name="movi_medium",
        dataset=dict(
            train_dataset=dict(
                custom_split="train",
                augmentation=dict(target_resolution=256, enable_horizontal_flip=False, enable_crop=False, minimal_source_augmentation=True),
                path=MOVI_MEDIUM_PATH,
                num_objects=23,
                num_frames=8,
                num_cameras=2,
                multi_camera_format=True,
            ),
            validation_dataset=dict(
                custom_split="validation",
                subset_size=8,
                random_subset=True,
                path=MOVI_MEDIUM_PATH,
                num_objects=23,
                num_frames=8,
                num_cameras=2,
                multi_camera_format=True,
                augmentation=dict(target_resolution=256, enable_horizontal_flip=False, enable_crop=False, minimal_source_augmentation=True),
            ),
        ),
        model=dict(
                segmentation_map_size=24,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "movi_e"},
        ],
    )

    mode_store(
        name="movi_medium_two_objects",
        dataset=dict(
            train_dataset=dict(
                num_cameras=1,
                path=MOVI_MEDIUM_TWO_OBJECTS_PATH,
            ),
            validation_dataset=dict(
                num_cameras=1,
                path=MOVI_MEDIUM_TWO_OBJECTS_PATH,
            ),
        ),
        hydra_defaults=["movi_medium"],
    )

    mode_store(
        name="single_scene",
        dataset=dict(
            train_dataset=dict(subset=("000001",), fake_return_n=8), validation_dataset=dict(subset=("000001",), fake_return_n=8), overfit=True
        ),
    )

    mode_store(
        name="movi_medium_single_object",
        dataset=dict(
            train_dataset=dict(
                num_cameras=1,
                num_frames=24,
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
            ),
            validation_dataset=dict(
                num_cameras=1,
                num_frames=24,
                path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
            ),
        ),
        hydra_defaults=["movi_medium"],
    )

    mode_store(
        name="imagenet",
        dataset=dict(
            train_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=False, enable_horizontal_flip=False)),
            validation_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=False, enable_horizontal_flip=False)),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagenet"},
        ],
    )

    mode_store(
        name="coco_panoptic",
        dataset=dict(
            train_dataset=dict(),
            validation_dataset=dict(),
        ),
        model=dict(
            segmentation_map_size=134,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "coco_panoptic"},
        ],

    )


def get_experiments():
    get_datasets()
    get_deprecated_experiments()

    mode_store(name="resnet", model=dict(encoder=builds(ResNetFeatureExtractor, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="sm",
        model=dict(
            decoder_transformer=dict(fused_mlp=False, fused_bias_fc=False),
            fused_mlp=False,
            fused_bias_fc=False,
        ),
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
        model=dict(unet_lora=True, lora_rank=256, gated_cross_attn=False, unfreeze_gated_cross_attn=False),
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
        name="dino",
        model=dict(
            encoder=builds(ViTFeatureExtractor, return_only=None, populate_full_signature=False),
            encoder_dim=768,
            per_layer_queries=True,
            feature_map_keys=(
                "blocks",
                "norm",
            ),
        ),
    )

    mode_store(
        name="low_res",
        dataset=dict(
            train_dataset=dict(
                augmentation=dict(
                    source_resolution="${model.encoder_resolution}", 
                    target_resolution="${model.decoder_resolution}"
                )
            ),
            validation_dataset=dict(
                augmentation=dict(
                    source_resolution="${model.encoder_resolution}", 
                    target_resolution="${model.decoder_resolution}"
                )
            ),
        ),
        model=dict(encoder_resolution=224, decoder_resolution=256, latent_dim=32),
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
        name="token_pred",
        model=dict(token_cls_pred_loss=True, token_rot_pred_loss=True),
        trainer=dict(custom_inference_every_n_steps=100),
        dataset=dict(train_dataset=dict(drop_last=False), validation_dataset=dict(drop_last=False)),
        hydra_defaults=["cur_exp"],
    )

    mode_store(
        name="discrete_token_pred",
        model=dict(
            discretize_rot_pred=True,
            discretize_rot_bins_per_axis=36,
            predict_rotation_from_n_frames=None,
            training_mask_dropout=None,
            add_pos_emb=True,
            decoder_transformer=dict(depth=4),
            unet=False,
            rotation_diffusion_start_timestep=100,
            num_conditioning_pairs=1,
            per_layer_queries=True,
            feature_map_keys=(
                "stage1",
                "stage18",
                # "stage24",
                "ln_post",
            ),
        ),
        dataset=dict(
            train_dataset=dict(
                subset_size=None,
                cache_in_memory=True,
                num_workers=2,
                num_subset=None,
                return_multiple_frames=None,
                fake_return_n=None,
                batch_size=24,
                cache_instances_in_memory=False,
            ),
            validation_dataset=dict(
                subset_size=None, return_multiple_frames=None, num_workers=2, cache_instances_in_memory=False, cache_in_memory=True
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=1,
            eval_every_n_steps=1000,
            custom_inference_every_n_steps=1000,
            learning_rate=5e-6,
            custom_inference_batch_size=24,
            custom_inference_dataset_size=256,
            custom_inference_fixed_shuffle=True,
            scale_lr_batch_size=False,
            lr_warmup_steps=1000,
            eval_on_start=True,
            log_parameters=True,
            log_gradients=1000,
        ),
        inference=dict(
            num_images_per_prompt=1,
            save_prompt_embeds=False,
            infer_new_prompts=False,
            vary_cfg_plot=False,
        ),
        hydra_defaults=["token_pred", "movi_medium_single_object", "no_movi_augmentation"],
    )

    mode_store(
        name="relative_token_pred",
        model=dict(
            predict_rotation_from_n_frames=2,
        ),
        dataset=dict(
            train_dataset=dict(return_multiple_frames=2),
            validation_dataset=dict(return_multiple_frames=2),
        ),
    )

    mode_store(
        name="overfit_tokens",
        dataset=dict(
            overfit=True,
            train_dataset=dict(
                num_subset=None,
                subset_size=200,
                random_subset=False,
            ),
        ),
        trainer=dict(
            custom_inference_dataset_size=64,
            eval_every_n_steps=1000,
            custom_inference_every_n_steps=250,
        ),
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
        name="large_model",
        dataset=dict(train_dataset=dict(batch_size=8, cache_in_memory=False), validation_dataset=dict(cache_in_memory=False)),
        model=dict(
            num_conditioning_pairs=8,
            diffusion_loss_weight=1.0,
            lr_finetune_version=1,
        ),
        trainer=dict(
            custom_inference_batch_size=None,
            learning_rate=4e-6,
        ),
        hydra_defaults=["discrete_token_pred"],
    )
    
    mode_store(
        name="sgd",
        trainer=dict(
            optimizer_cls=torch.optim.SGD,
            momentum=0.0,
            weight_decay=0.0,
            learning_rate=1e-6,
        ),
    )

    mode_store(
        name="coco_debug",
        model=dict(
            token_rot_pred_loss=False,
            num_token_cls=133, 
            encoder_resolution=448, 
            decoder_resolution=512,
            latent_dim=64,
            diffusion_timestep_range=(500, 1000),
            training_mask_dropout=None,
            training_layer_dropout=None,
            training_cfg_dropout=None,
            detach_features_before_cross_attn=True,
        ),
        dataset=dict(
            train_dataset=dict(
                batch_size=16, 
                cache_in_memory=False,
                augmentation=dict(
                    source_resolution="${model.encoder_resolution}", 
                    target_resolution="${model.decoder_resolution}"
                )
            ),
            validation_dataset=dict(
                cache_in_memory=False,
                augmentation=dict(
                    source_resolution="${model.encoder_resolution}",
                    target_resolution="${model.decoder_resolution}"
                )
            )
        ),
        hydra_defaults=["large_model", "coco_panoptic"],
    )

    mode_store(
        name="disable_reconstruction",
        model=dict(
            unet=False,
            freeze_mapper=False,
            detach_features_before_cross_attn=False,
            freeze_clip=True,
        )
    )

    mode_store(
        name="enable_reconstruction",
        model=dict(
            unet=True,
            freeze_mapper=False,
            freeze_unet=False,
            unet_lora=False,
            finetune_unet_with_different_lrs=True,
            freeze_clip=False,
        )
    )

    mode_store(
        name="03_01_no_recon",
        model=dict(
            layer_specialization=False,
            num_conditioning_pairs=1,
            per_layer_queries=False,
            diffusion_timestep_range=(250, 750),
            detach_features_before_cross_attn=False,
            gated_cross_attn=False,
            unfreeze_gated_cross_attn=False,
            unet=False,
            freeze_mapper=False,
            freeze_clip=False,
            encoder=dict(
                model_name="vit_small_patch16_384",
            ),
            encoder_dim=384,
        ),
        trainer=dict(
            learning_rate=1e-6,
            momentum=0.9,
        ),
        dataset=dict(
            train_dataset=dict(batch_size=16),
        ),
        inference=dict(
            visualize_attention_map=False
        ),
        hydra_defaults=["coco_debug", "debug_vit_base_scratch", "sgd"],
    )

    mode_store(
        name="03_01_recon",
        model=dict(unet=True, unet_lora=True, finetune_unet_with_different_lrs=False, lora_rank=256),
        hydra_defaults=["03_01_no_recon"],
    )

    mode_store(
        name="coco_recon_only",
        model=dict(
            num_token_cls=133,
            layer_specialization=True,
            num_conditioning_pairs=8,
            per_layer_queries=True,
            unet=True, 
            gated_cross_attn=False,
            unfreeze_gated_cross_attn=False,
            unet_lora=True,
            lora_rank=512,
            freeze_unet=True,
            encoder=dict(return_only=None),
            feature_map_keys=(
                "stage12",
                "stage18",
                "stage24",
            ),
        ),
        dataset=dict(
            train_dataset=dict(batch_size=20),
        ),
        trainer=dict(learning_rate=1e-6),
        hydra_defaults=["no_movi_augmentation", "multiscale", "low_res", "coco_panoptic"],
    )

    mode_store(
        name="overfit_coco",
        dataset=dict(
            overfit=False,
            train_dataset=dict(
                subset_size=10,
                random_subset=False,
                repeat_dataset_n_times=100,
            ),
        ),
        trainer=dict(
            eval_every_n_steps=100,
            validate_training_dataset=True
        ),
        hydra_defaults=["coco_recon_only"],
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
        model=dict(layer_specialization=False, per_layer_queries=False, num_token_cls=2),
        trainer=dict(learning_rate=1e-4, eval_on_start=False, max_train_steps=10, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False, profiler_active_steps=2),
        hydra_defaults=["sd_15"],
    )