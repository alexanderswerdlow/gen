from pathlib import Path
from pyexpat import model

from hydra_zen import builds
from gen.configs.datasets import get_datasets, get_default_augmentation
from gen.configs.utils import mode_store
from gen.datasets.hypersim.hypersim import Hypersim
from gen.datasets.imagefolder.imagefolder import ImagefolderDataset
from gen.datasets.imagefolder.videofolder import VideofolderDataset


def get_override_dict(**kwargs):
    return dict(
        train=dict(**kwargs),
        val=dict(**kwargs),
    )


def get_experiments():
    get_datasets()
    mode_store(
        name="sm",
        debug=True,
        model=dict(
            fused_mlp=False,
            fused_bias_fc=False,
        ),
        trainer=dict(use_fused_adam=False, fast_eval=True),
        dataset=dict(
            train=dict(batch_size=2, num_workers=0),
            val=dict(batch_size=1, num_workers=0)
        ),
    )

    mode_store(
        name="24G",
        debug=True,
        trainer=dict(
            enable_dynamic_grad_accum=False,
        ),
        dataset=dict(
            train=dict(
                batch_size=2,
            ),
            val=dict(
                batch_size=2,
            ),
        )
    )

    mode_store(
        name="test_depth",
        debug=True,
        dataset=dict(
            train=dict(
                batch_size=1,
                num_workers=0,
                root='data/depth',
                load_depth=True,
            ),
            val=dict(
                batch_size=1,
                num_workers=0,
                root='data/depth',
                load_depth=True,
            ),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "imagefolder"},
        ],
    )

    mode_store(
        name="exp_v0",
        debug=True,
        model=dict(
            freeze_unet=True,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
            duplicate_unet_input_channels=True,
            dual_attention=True,
            token_embedding_dim=1024,
            decoder_resolution=256,
            decoder_latent_dim=32,
            force_fp32_pcd_vae=True,
            fill_invalid_regions=True,
            use_valid_xyz_loss_mask=True,
            snr_gamma=5.0,
            xyz_min_max_quantile=0.02,
        ),
        dataset=dict(
            train=dict(
                batch_size=36,
                resolution="${model.decoder_resolution}",
                fill_invalid_regions=True,
                mask_bg=True,
                inpaint=False,
            ),
            val=dict(
                batch_size=32,
                subset_size="${eval:'${dataset.val.batch_size} * 8'}",
                resolution="${model.decoder_resolution}",
                fill_invalid_regions=True,
                mask_bg=True,
                inpaint=False,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=4,
            ckpt_steps=1000,
            eval_steps=500,
            fsdp=True,
            param_dtype_exception_prefixes=["vae."],
            enable_dynamic_grad_accum=False,
        ),
        inference=dict(
            guidance_scale=1
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "co3d"},
        ],
    )

    mode_store(
        name="high_res",
        model=dict(
            decoder_resolution=512,
            decoder_latent_dim=64,
        ),
        dataset=dict(
            train=dict(
                batch_size=12,
            ),
            val=dict(
                batch_size=8,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=8,
        ),
    )

    mode_store(
        name="full_scene",
        model=dict(
            separate_xyz_encoding=True,
            xyz_min_max_quantile=0.1,
        ),
        dataset=dict(
            train=dict(
                mask_bg=False,
                inpaint=False,
                fill_invalid_regions=True,
            ),
            val=dict(
                mask_bg=False,
                inpaint=False,
                fill_invalid_regions=True,
                batch_size=16,
            ),
        ),
    )

    mode_store(
        name="smaller_gpu",
        dataset=dict(
            train=dict(
                batch_size=12
            ),
            val=dict(
                batch_size=12
            ),
        ),
    )

    mode_store(
        name="inpaint",
        dataset=dict(
            train=dict(
                mask_bg=False,
                inpaint=True,
                fill_invalid_regions=False,
            ),
            val=dict(
                mask_bg=False,
                inpaint=True,
                fill_invalid_regions=False,
            ),
        ),
    )

    mode_store(
        name="finetune_vae_decoder",
        model=dict(
            separate_xyz_encoding=True,
            unet=False,
            unfreeze_vae_decoder=True,
        ),
    )

    mode_store(
        name="exp_v1",
        debug=True,
        model=dict(
            freeze_unet=True,
            pretrained_model_name_or_path="prs-eth/marigold-v1-0",
            duplicate_unet_input_channels=True,
            dual_attention=False,
            joint_attention=True,
            predict_depth=True,
            token_embedding_dim=1024,
            decoder_resolution=256,
            decoder_latent_dim=32,
            force_fp32_pcd_vae=True,
            snr_gamma=5.0,
            only_noise_tgt=False,
            use_valid_xyz_loss_mask=False,
            dropout_src_depth=0.5,
            num_cross_attn_views=2,
        ),
        dataset=dict(
            train=dict(
                batch_size=36,
                return_different_views=True,
                augmentation=dict(
                    src_resolution="${model.decoder_resolution}",
                    tgt_resolution="${model.decoder_resolution}",
                    src_transforms="${get_tgt_transform:model}", # This is intentional
                    tgt_transforms="${get_tgt_transform:model}",
                ),
            ),
            val=dict(
                batch_size=28,
                subset_size="${eval:'${dataset.val.batch_size} * 6'}",
                random_subset=False,
                return_different_views=True,
                augmentation=dict(
                    src_resolution="${model.decoder_resolution}",
                    tgt_resolution="${model.decoder_resolution}",
                    src_transforms="${get_tgt_transform:model}", # This is intentional
                    tgt_transforms="${get_tgt_transform:model}",
                ),
            ),
            additional_val=dict(
                sevenscenes=builds(
                    VideofolderDataset, 
                    populate_full_signature=True,
                    root=Path('data/depth_data/video'),
                    zen_partial=True,
                    camera_trajectory_window=100,
                    return_n_views=2,
                    load_depth=True,
                    augmentation=get_default_augmentation(**dict(
                        src_resolution="${model.decoder_resolution}",
                        tgt_resolution="${model.decoder_resolution}",
                        src_transforms="${get_tgt_transform:model}", # This is intentional
                        tgt_transforms="${get_tgt_transform:model}",
                    )),
                    batch_size="${dataset.val.batch_size}",
                    subset_size="${dataset.val.subset_size}",
                    random_subset=False,
                ),
            )
        ),
        trainer=dict(
            gradient_accumulation_steps=4,
            ckpt_steps=1000,
            eval_steps=1000,
            fsdp=True,
            param_dtype_exception_prefixes=["vae."],
            enable_dynamic_grad_accum=False,
            additional_val_datasets_seperate_inference=True,
        ),
        inference=dict(
            guidance_scale=1
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "hypersim"},
        ],
    )

    mode_store(
        name="fast_hypersim",
        debug=True,
        dataset=dict(
            train=dict(
                uniform_sampler=True,
                camera_trajectory_window=6,
                augmentation=dict(
                    enable_horizontal_flip=True,
                    enable_square_crop=True,
                    enable_zoom_crop=False,
                    enable_random_resize_crop=False,
                    tgt_random_scale_ratio=None,
                    initial_resolution=768,
                )
            ),
            val=dict(
                uniform_sampler=True,
                camera_trajectory_window=6,
            ),
        ),
     )

    mode_store(
        name="exp_v1_1",
        debug=True,
        model=dict(
            freeze_self_attn=True,
            n_view_pred=True,
        ),
        hydra_defaults=["exp_v1", "fast_hypersim"],
    )

    mode_store(
        name="exp_v1_1_multiview",
        debug=True,
        model=dict(
            n_view_pred=True,
            add_cross_attn_pos_emb="${model.num_cross_attn_views}",
            shuffle_every_layer=True,
            num_cross_attn_views=4,
        ),
        dataset=dict(
            train=dict(
                uniform_sampler=True,
                return_n_views=4,
                camera_trajectory_window=16,
                batch_size=22
            ),
            val=dict(
                uniform_sampler=True,
                return_n_views=4,
                camera_trajectory_window=16,
                batch_size=16
            ),
            additional_val=dict(
                hypersim_double_views=builds(
                    Hypersim,
                    populate_full_signature=True,
                    zen_partial=True,
                    return_different_views=True,
                    uniform_sampler=True,
                    return_n_views="${eval:'${dataset.val.return_n_views} * 2'}",
                    camera_trajectory_window=16,
                    batch_size="${eval:'${dataset.val.batch_size} // 3'}",
                    augmentation=get_default_augmentation(**dict(
                        src_resolution="${model.decoder_resolution}",
                        tgt_resolution="${model.decoder_resolution}",
                        src_transforms="${get_tgt_transform:model}", # This is intentional
                        tgt_transforms="${get_tgt_transform:model}",
                    )),
                ),
                sevenscenes=dict(
                    camera_trajectory_window=500,
                    return_n_views="${dataset.val.return_n_views}",
                    batch_size="${eval:'${dataset.val.batch_size} // 3'}",
                    subset_size="${dataset.val.subset_size}",
                ),
                sevenscenes_double_views=builds(
                    VideofolderDataset, 
                    populate_full_signature=True,
                    root=Path('data/depth_data/video'),
                    zen_partial=True,
                    camera_trajectory_window=500,
                    return_n_views="${eval:'${dataset.val.return_n_views} * 2'}",
                    load_depth=True,
                    augmentation=get_default_augmentation(**dict(
                        src_resolution="${model.decoder_resolution}",
                        tgt_resolution="${model.decoder_resolution}",
                        src_transforms="${get_tgt_transform:model}", # This is intentional
                        tgt_transforms="${get_tgt_transform:model}",
                    )),
                    batch_size="${eval:'${dataset.val.batch_size} // 3'}",
                    subset_size="${dataset.val.subset_size}",
                    random_subset=False,
                ),
            ),
        ),
        hydra_defaults=["exp_v1_1"],
    )

    mode_store(
        name="high_res_multiview",
        model=dict(
            decoder_resolution=512,
            decoder_latent_dim=64,
        ),
        dataset=dict(
            train=dict(
                batch_size=5,
            ),
            val=dict(
                batch_size=4,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=8,
        ),
    )

    mode_store(
        name="replica_dataset",
        dataset=dict(
            train=builds(
                VideofolderDataset, 
                populate_full_signature=True,
                root=Path('/home/aswerdlo/repos/lib/SplaTAM/data/Replica'),
                zen_partial=True,
                camera_trajectory_window=100,
                return_n_views="${model.num_input_views}",
                load_depth=True,
                augmentation=get_default_augmentation(**dict(
                    src_resolution="${model.decoder_resolution}",
                    tgt_resolution="${model.decoder_resolution}",
                    src_transforms="${get_tgt_transform:model}", # This is intentional
                    tgt_transforms="${get_tgt_transform:model}",
                )),
                random_subset=True,
                postfix='results',
                rgb_prefix='frame'
            ),
            val=builds(
                VideofolderDataset, 
                populate_full_signature=True,
                root=Path('/home/aswerdlo/repos/lib/SplaTAM/data/Replica'),
                zen_partial=True,
                camera_trajectory_window=100,
                return_n_views="${model.num_input_views}",
                load_depth=True,
                augmentation=get_default_augmentation(**dict(
                    src_resolution="${model.decoder_resolution}",
                    tgt_resolution="${model.decoder_resolution}",
                    src_transforms="${get_tgt_transform:model}", # This is intentional
                    tgt_transforms="${get_tgt_transform:model}",
                )),
                random_subset=False,
                postfix='results',
                rgb_prefix='frame'
            ),
            additional_val=dict(
                replica_16_views=builds(
                    VideofolderDataset, 
                    populate_full_signature=True,
                    root=Path('/home/aswerdlo/repos/lib/SplaTAM/data/Replica'),
                    zen_partial=True,
                    camera_trajectory_window=400,
                    return_n_views="${eval:'${dataset.val.return_n_views} * 4'}",
                    load_depth=True,
                    augmentation=get_default_augmentation(**dict(
                        src_resolution="${model.decoder_resolution}",
                        tgt_resolution="${model.decoder_resolution}",
                        src_transforms="${get_tgt_transform:model}", # This is intentional
                        tgt_transforms="${get_tgt_transform:model}",
                    )),
                    batch_size="${eval:'${dataset.val.batch_size} // 6'}",
                    random_subset=False,
                    postfix='results',
                    rgb_prefix='frame'
                ),
                sevenscenes=None,
                sevenscenes_double_views=None,
                hypersim_double_views=None,
            ),
        ),
        hydra_defaults=["_self_", {"override /dataset": "videofolder"}],
    )

    mode_store(
        name="exp_v1_2",
        model=dict(
            add_cross_attn_pos_emb=96,
            shuffle_embedding_per_layer=True,
        ),
        hydra_defaults=["exp_v1_1_multiview"],
    )

    mode_store(
        name="exp_v1_3",
        model=dict(
            unfreeze_last_n_unet_layer=2,
            fill_invalid_with_max=True,
            disable_quantile=True,
        ),
        hydra_defaults=["exp_v1_2"],
    )

    mode_store(
        name="exp_v1_4",
        model=dict(
            unfreeze_last_n_unet_layer=1,
            fill_invalid_with_max=True,
            disable_quantile=True,
            num_cross_attn_views=4,
            num_input_views=32,
            shuffle_every_layer=True,
            xyz_min_max_quantile=0.0,
            fix_current_view_during_shuffle=True
        ),
        dataset=dict(
            train=dict(
                batch_size=1,
            ),
            val=dict(
                batch_size=1,
            ),
        ),
        trainer=dict(
            gradient_accumulation_steps=8,
        ),
        hydra_defaults=["exp_v1_3"],
    )