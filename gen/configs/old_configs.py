import torch
from hydra_zen import builds

from gen.configs.utils import mode_store
from gen.models.cross_attn.base_inference import compose_two_images, interpolate_latents
from gen.models.encoders.encoder import ResNetFeatureExtractor, ViTFeatureExtractor


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
            train=dict(batch_size=6, augmentation=dict(enable_random_resize_crop=False, enable_horizontal_flip=True)),
            val=dict(augmentation=dict(enable_random_resize_crop=False, enable_horizontal_flip=True)),
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
        dataset=dict(train=dict(batch_size=20)),
        hydra_defaults=["break_a_scene"],
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
            train=dict(batch_size=16),
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
        name="profile_high_res_coco",
        model=dict(
            encoder=dict(
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                img_size=392,
                gradient_checkpointing=True,
            ),
            encoder_latent_dim=28,
            encoder_resolution=392,
            encoder_dim=384,
            freeze_clip=True,
            unfreeze_last_n_clip_layers=4,
            pretrained_model_name_or_path="lambdalabs/sd-image-variations-diffusers",
            token_embedding_dim=768,
            use_sd_15_tokenizer_encoder=True,
            decoder_transformer=dict(
                embed_dim=512,
            ),
        ),
        trainer=dict(
            learning_rate=1e-4,
            lr_warmup_steps=2000,
            enable_timing=True,
            fast_eval=True,
            profile_memory=True,
            eval_on_start=False,
            eval_every_n_steps=1,
            max_train_steps=3,
            enable_dynamic_grad_accum=False,
            validate_training_dataset=False,
            gradient_accumulation_steps=1,
        ),
        inference=dict(
            visualize_embeds=False,
            infer_new_prompts=False,
            save_prompt_embeds=False,
            num_images_per_prompt=1,
            num_masks_to_remove=None,
            vary_cfg_plot=False,
            visualize_attention_map=False,
        ),
        dataset=dict(
            train=dict(
                batch_size=36,
                augmentation=dict(
                    reorder_segmentation=False,
                )
            ),
            val=dict(
                augmentation=dict(
                    reorder_segmentation=False,
                )
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
        name="token_pred",
        model=dict(token_cls_pred_loss=True, token_rot_pred_loss=True),
        trainer=dict(custom_inference_every_n_steps=100),
        dataset=dict(train=dict(drop_last=False), val=dict(drop_last=False)),
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
            train=dict(
                subset_size=None,
                cache_in_memory=True,
                num_workers=2,
                num_subset=None,
                return_multiple_frames=None,
                fake_return_n=None,
                batch_size=24,
                cache_instances_in_memory=False,
            ),
            val=dict(
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
            train=dict(return_multiple_frames=2),
            val=dict(return_multiple_frames=2),
        ),
    )

    mode_store(
        name="overfit_tokens",
        dataset=dict(
            overfit=True,
            train=dict(
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
        name="coco_recon_small",
        model=dict(
            decoder_latent_dim=32,
            decoder_resolution=256,
        ),
        hydra_defaults=["coco_recon_only"],
    )

    mode_store(
        name="debug_vit_large_clip",
        model=dict(
            encoder=dict(
                model_name="vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
                pretrained=True,
                img_size=448,
            ),
            encoder_dim=1024,
        ),
        hydra_defaults=["debug_vit_base_scratch"],
    )

    mode_store(
        name="coco_debug",
        model=dict(
            token_rot_pred_loss=False,
            num_token_cls=133, 
            encoder_resolution=448, 
            decoder_resolution=512,
            decoder_latent_dim=64,
            diffusion_timestep_range=(500, 1000),
            training_mask_dropout=None,
            training_layer_dropout=None,
            training_cfg_dropout=None,
            detach_features_before_cross_attn=True,
        ),
        dataset=dict(
            train=dict(
                batch_size=16, 
                cache_in_memory=False,
            ),
            val=dict(
                cache_in_memory=False,
            )
        ),
        hydra_defaults=["large_model", "coco_panoptic"],
    )

    mode_store(
        name="large_model",
        dataset=dict(train=dict(batch_size=8, cache_in_memory=False), val=dict(cache_in_memory=False)),
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
        name="overfit_coco",
        dataset=dict(
            overfit=False,
            train=dict(
                subset_size=10,
                random_subset=False,
                repeat_dataset_n_times=100,
            ),
        ),
        trainer=dict(
            eval_every_n_steps=100,
            validate_training_dataset=True
        ),
    )
