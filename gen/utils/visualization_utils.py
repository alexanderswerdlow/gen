from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from torchvision.transforms.functional import InterpolationMode, resize

from gen.utils.decoupled_utils import to_numpy
from image_utils import Im

if TYPE_CHECKING:
    from gen.utils.data_defs import InputData


def viz_feats(cfg, batch, feats, name):
    from gen.utils.data_defs import undo_normalization_given_transforms
    bot_imgs = []
    for k,v in feats.items():
        if 'fc_norm' not in k:
            assert v.shape[0] == 1
            orig_viz = get_dino_pca(to_numpy(v[0, 5:, :]), patch_h=cfg.model.encoder_latent_dim, patch_w=cfg.model.encoder_latent_dim, threshold=0.6, object_larger_than_bg=True, return_all=True)
            bot_imgs.append(Im(orig_viz).scale(10, resampling_mode=InterpolationMode.NEAREST).add_border(50, color=(255, 255, 255)).write_text(f"{k}"))
    
    src_rgb = undo_normalization_given_transforms(cfg.dataset.val.augmentation.src_transforms, batch.src_pixel_values)
    return Im.concat_horizontal(src_rgb, Im.concat_vertical(Im.concat_horizontal(bot_imgs))).write_text(f"{name}_feature_maps")

def get_dino_pca(features: np.ndarray, patch_h: int, patch_w: int, threshold: float, object_larger_than_bg: bool, return_all: bool = False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    
    pca = PCA(n_components=3)
    scaler = MinMaxScaler(clip=True)
    pca_features = pca.fit_transform(features)
    pca_features = scaler.fit_transform(pca_features)

    if object_larger_than_bg:
        pca_features_bg = pca_features[:, 0] > threshold
    else: 
        pca_features_bg = pca_features[:, 0] < threshold
        
    pca_features_fg = ~pca_features_bg

    pca_features_fg_seg = pca.fit_transform(features[pca_features_fg])

    pca_features_fg_seg = scaler.fit_transform(pca_features_fg_seg)

    pca_features_rgb = np.zeros((patch_h * patch_w, 3))
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_fg_seg
    pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)

    if return_all:
        pca_features_rgb = pca_features
        pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)
        
    return pca_features_rgb