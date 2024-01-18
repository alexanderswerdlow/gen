from hydra_zen import builds
from gen import IMAGENET_PATH, MOVI_OVERFIT_DATASET_PATH
from gen.configs.utils import mode_store
from gen.utils.encoder_utils import ResNet50


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
    mode_store(
        name="overfit_movi",
        debug=True,
        inference=dict(num_masks_to_remove=None, visualize_attention_map=True, empty_string_cfg=True, guidance_scale=7.5),
        trainer=dict(
            gradient_accumulation_steps=4, num_train_epochs=10000, eval_every_n_steps=100, learning_rate=1e-3, eval_on_start=False, log_gradients=100
        ),
        dataset=dict(
            train_dataset=dict(batch_size=8, random_subset=None, **shared_overfit_movi_args),
            validation_dataset=dict(random_subset=4, evenly_spaced_subset=False, **shared_overfit_movi_args),
            overfit=False,
        ),
    )


    mode_store(
        name="overfit_movi_single_frame",
        dataset=dict(
            train_dataset=dict(num_dataset_frames=1, subset=("video_0015",)),
            validation_dataset=dict(num_dataset_frames=1, subset=("video_0015",)),
        ),
    )

    mode_store(
        name="overfit_debug",
        trainer=dict(scale_lr_batch_size=True, gradient_accumulation_steps=1, enable_dynamic_grad_accum=False),
        dataset=dict(
            train_dataset=dict(subset=("video_0015",), augmentation=dict(enable_horizontal_flip=False)),
            validation_dataset=dict(subset=("video_0015",), augmentation=dict(enable_horizontal_flip=False), random_subset=2),
        ),
        model=dict(dropout_masks=0.0, single_token=True, enable_norm_scale=True, use_fixed_position_encoding=False),
    )

    mode_store(
        name="overfit_neti",
        model=dict(enable_norm_scale=True, use_fixed_position_encoding=False, nested_dropout_prob=0.5, mask_cross_attn=False, output_bypass=True),
    )

    mode_store(
        name="overfit_imagenet",
        model=dict(use_dataset_segmentation=False),
        dataset=dict(
            train_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=True)),
            validation_dataset=dict(path=IMAGENET_PATH, augmentation=dict(enable_crop=False, enable_horizontal_flip=False), random_subset=4),
            overfit=False,
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagenet"},
        ],
    )

    mode_store(
        name="controlnet",
        model=dict(controlnet=True),
        trainer=dict(learning_rate=5e-6, scale_lr_batch_size=True),
        dataset=dict(
            train_dataset=dict(batch_size=4),
        ),
    )

    mode_store(name="resnet", model=dict(encoder=builds(ResNet50, populate_full_signature=False), cross_attn_dim=256))

    mode_store(
        name="should_work",
        inference=dict(guidance_scale=0.0),
        model=dict(use_dataset_segmentation=True, mask_cross_attn=False, use_cls_token_only=True),
        trainer=dict(learning_rate=1e-5),
    )
