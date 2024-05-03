from pyexpat import model
from gen.configs.datasets import get_datasets
from gen.configs.utils import mode_store


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
                batch_size=8,
            ),
            val=dict(
                batch_size=10,
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
            dropout_src_depth=0.5
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
        ),
        trainer=dict(
            gradient_accumulation_steps=4,
            ckpt_steps=1000,
            eval_steps=1000,
            fsdp=True,
            param_dtype_exception_prefixes=["vae."],
            enable_dynamic_grad_accum=False,
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
            add_cross_attn_pos_emb=4,
        ),
        dataset=dict(
            train=dict(
                uniform_sampler=True,
                return_n_views=4,
                camera_trajectory_window=12,
                batch_size=22
            ),
            val=dict(
                uniform_sampler=True,
                return_n_views=4,
                camera_trajectory_window=12,
                batch_size=16
            ),
        ),
        hydra_defaults=["exp_v1_1"],
    )