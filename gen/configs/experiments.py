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
