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
        name="demo",
        debug=True,
        model=dict(
            freeze_unet=True,
            unfreeze_single_unet_layer=True,
        ),
        dataset=dict(
            train=dict(
                batch_size=2,
                num_workers=0,
                repeat_dataset_n_times=1000,
                root='/home/aswerdlo/repos/gen/archive/imagefolder_test',
            ),
            val=dict(
                batch_size=1,
                num_workers=0,
                repeat_dataset_n_times=1000,
                root='/home/aswerdlo/repos/gen/archive/imagefolder_test',
            ),
        ),
    )


    mode_store(
        name="exp_v0",
        debug=True,
        model=dict(
            freeze_unet=True,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
            duplicate_unet_input_channels=True,
            dual_attention=True,
        ),
        dataset=dict(
            train=dict(
                batch_size=2,
            ),
            val=dict(
                batch_size=1,
            ),
        ),
        hydra_defaults=[
            "_self_",
            {"override /dataset": "co3d"},
        ],
    )
