from hydra_zen import builds
from gen import IMAGENET_PATH, MOVI_OVERFIT_DATASET_PATH
from gen.configs.utils import inherit_mode_store, mode_store, store_child_config
from gen.utils.encoder_utils import ResNet50
from accelerate.utils import PrecisionType


def get_override_dict(**kwargs):
    return dict(
        train_dataset=dict(**kwargs),
        validation_dataset=dict(**kwargs),
    )


shared_overfit_movi_args = dict(
    custom_split="train",
    path=MOVI_OVERFIT_DATASET_PATH,
    num_objects=1,
    legacy_transforms=False,
    augmentation=dict(minimal_source_augmentation=True, enable_crop=False),
)


def get_experiments():
    # Tmp debugging params
    mode_store(name="resnet", model=dict(encoder=builds(ResNet50, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="overfit_exp",
        debug=True,
        inference=dict(num_masks_to_remove=None, visualize_attention_map=True, empty_string_cfg=True, guidance_scale=7.5),
        trainer=dict(
            gradient_accumulation_steps=4, num_train_epochs=10000, eval_every_n_steps=500, learning_rate=1e-3, log_gradients=100
        ),
        dataset=dict(
            train_dataset=dict(batch_size=8, subset_size=None, **shared_overfit_movi_args),
            validation_dataset=dict(subset_size=4, **shared_overfit_movi_args),
            overfit=True,
        ),
        model=dict(use_dataset_segmentation=True),
    )

    mode_store(
        name="movi",
        dataset=dict(
            train_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False)),
            validation_dataset=dict(augmentation=dict(enable_horizontal_flip=False, enable_crop=False)),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "movi_e"},
        ],
    )

    mode_store(
        name="movi_single_scene",
        dataset=dict(train_dataset=dict(subset=("video_0015",), fake_return_n=8), validation_dataset=dict(subset=("video_0015",), fake_return_n=8)),
    )

    mode_store(
        name="movi_single_frame",
        dataset=dict(train_dataset=dict(num_dataset_frames=1, fake_return_n=256), validation_dataset=dict(num_dataset_frames=1, fake_return_n=256)),
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
        name="overfit_debug",
        trainer=dict(scale_lr_batch_size=True, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False),
        model=dict(dropout_masks=0.0, enable_norm_scale=True, use_timestep_layer_encoding=False, encode_token_without_tl=True),
    )

    mode_store(
        name="overfit_neti",
        model=dict(
            enable_norm_scale=True,
            use_timestep_layer_encoding=False,
            nested_dropout_prob=0.5,
            mask_cross_attn=False,
            output_bypass=True,
            enable_neti=True,
            encode_token_without_tl=False,
        ),
    )

    mode_store(
        name="controlnet",
        model=dict(controlnet=True),
        trainer=dict(learning_rate=5e-6, scale_lr_batch_size=True),
        dataset=dict(
            train_dataset=dict(batch_size=4),
        ),
    )

    mode_store(
        name="neti_training",
        trainer=dict(
            learning_rate=1e-3,
            enable_dynamic_grad_accum=False,
            gradient_accumulation_steps=4,
            scale_lr_batch_size=True,
            mixed_precision=PrecisionType.NO,
            log_parameters=True,
            eval_every_n_steps=100,
            log_gradients=10,
            tracker_project_name="textual_inversion",
        ),
        dataset=dict(train_dataset=dict(batch_size=2, fake_return_n=8), validation_dataset=dict(subset_size=8)),
        model=dict(tmp_revert_to_neti_logic=True),
        inference=dict(visualize_attention_map=False),
    )


    mode_store(
        name="unet_finetune",
        dataset=dict(train_dataset=dict(batch_size=6)),
        trainer=dict(
            learning_rate=1e-3,
            enable_dynamic_grad_accum=True,
            gradient_accumulation_steps=4,
            scale_lr_batch_size=True,
            log_parameters=False,
            log_gradients=50,
            gradient_checkpointing=True,
            finetune_learning_rate=2e-8,
        ),
        model=dict(freeze_unet=True, unfreeze_unet_after_n_steps=500),
    )

    mode_store(
        name="train_cross_attn_only",
        model=dict(mask_cross_attn=True),
        trainer=dict(max_train_steps=10000, checkpointing_steps=500, save_accelerator_format=True)
    )

    mode_store(
        name="finetune_cross_attn_unet",
        model=dict(mask_cross_attn=True, freeze_unet=False, freeze_text_encoder=True),
        trainer=dict(gradient_checkpointing=True, learning_rate=4e-7, compile=False, save_accelerator_format=True, lr_scheduler="constant_with_warmup"),
        dataset=dict(train_dataset=dict(batch_size=6)),
    )

    mode_store(
        name="lora_disable_timestep_layer_encoding",
        model=dict(decoder_transformer=dict(add_self_attn=False), per_timestep_conditioning=False, freeze_mapper=False, freeze_unet=True, lora_unet=True, use_timestep_layer_encoding=False, use_custom_position_encoding=True, output_bypass=False),
        inference=(dict(use_custom_pipeline=False)),
    )

    mode_store(
        name="lora_new_model",
        model=dict(decoder_transformer=dict(add_self_attn=False), per_timestep_conditioning=False, freeze_mapper=False, freeze_unet=True, lora_unet=True, use_timestep_layer_encoding=False),
        inference=(dict(use_custom_pipeline=False)),
    )

    mode_store(
        name="small_gpu",
        dataset=dict(train_dataset=dict(batch_size=1), validation_dataset=dict(batch_size=1)),
        model=dict(decoder_transformer=dict(fused_mlp=False, fused_bias_fc=False)),
        trainer=dict(enable_xformers_memory_efficient_attention=True),
        inference=dict(visualize_attention_map=False)
    )