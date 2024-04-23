from calendar import leapdays
from functools import partial
from gc import enable
from operator import le
from pathlib import Path

import torch
from hydra_zen import builds

from gen import (IMAGENET_PATH, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH,
                 MOVI_OVERFIT_DATASET_PATH, SCANNETPP_DATASET_PATH)
from gen.configs.datasets import get_datasets
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
