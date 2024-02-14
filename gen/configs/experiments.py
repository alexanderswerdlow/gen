from hydra_zen import builds

from gen import IMAGENET_PATH, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_OVERFIT_DATASET_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH
from gen.configs.utils import mode_store, store_child_config
from gen.models.encoders.encoder import ResNetFeatureExtractor, ViTFeatureExtractor


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
                augmentation=dict(target_resolution=256, enable_horizontal_flip=False, enable_crop=True, minimal_source_augmentation=True),
                path=MOVI_MEDIUM_PATH,
                num_objects=23,
                num_frames=8,
                num_cameras=2,
                multi_camera_format=True
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
        hydra_defaults=[
            "movi_medium"
        ],
    )

    mode_store(
        name="movi_medium_single_scene",
        dataset=dict(train_dataset=dict(subset=("000001",), fake_return_n=8), validation_dataset=dict(subset=("000001",), fake_return_n=8), overfit=True),
        hydra_defaults=["movi_medium"],
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
        hydra_defaults=[
            "movi_medium"
        ],
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


def get_experiments():
    get_datasets()
    get_deprecated_experiments()

    mode_store(name="resnet", model=dict(encoder=builds(ResNetFeatureExtractor, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="small_gpu",
        dataset=dict(train_dataset=dict(batch_size=1, num_workers=0), validation_dataset=dict(batch_size=1, num_workers=0)),
        model=dict(decoder_transformer=dict(fused_mlp=False, fused_bias_fc=False), single_fuser_layer=True, pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5', token_embedding_dim=768),
        trainer=dict(enable_xformers_memory_efficient_attention=True, compile=False, eval_on_start=False, gradient_accumulation_steps=1),
        inference=dict(
            visualize_attention_map=False,
            infer_new_prompts=False,
            max_batch_size=1,
            num_masks_to_remove=None,
            num_images_per_prompt=1,
            vary_cfg_plot=False,
        ),
        debug=True,
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
                augmentation=dict(target_resolution=256),
            ),
            validation_dataset=dict(
                augmentation=dict(target_resolution=256),
            ),
        ),
        model=dict(resolution=256, latent_dim=32),
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
        trainer=dict(base_model_custom_validation_steps=100),
        dataset=dict(train_dataset=dict(drop_last=False), validation_dataset=dict(drop_last=False)),
        hydra_defaults=["cur_exp"],
    )

    mode_store(
        name="debug_token_pred",
        model=dict(
            add_pos_emb=True, 
            decoder_transformer=dict(depth=2),
            unet=False,
            rotation_diffusion_start_timestep=10,
            num_conditioning_pairs=1
        ),
        trainer=dict(
            learning_rate=1e-6,
            lr_warmup_steps=100,
            eval_every_n_steps=100,
            base_model_custom_validation_steps=100,
            eval_on_start=False,
        ),
        dataset=dict(
            train_dataset=dict(batch_size=16, cache_in_memory=True, num_workers=4, num_subset=5),
            validation_dataset=dict(cache_in_memory=True)
        ),
        hydra_defaults=["token_pred", "movi_medium_single_object", "no_movi_augmentation"],
    )