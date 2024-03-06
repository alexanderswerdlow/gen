import os
from typing import Any, Callable, List

import kornia.augmentation as K
import torch
from attr import dataclass
from git import Optional
from image_utils import Im
from kornia.augmentation.auto.base import SUBPLOLICY_CONFIG
from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.container import AugmentationSequential
import torchvision.transforms.v2 as transforms

from gen.datasets.augmentation.utils import get_keypoints, get_viz_keypoints, process_output_keypoints, process_output_segmentation, viz
from gen.datasets.utils import get_simple_transform, get_stable_diffusion_transforms
import warnings

randaug_policy: List[SUBPLOLICY_CONFIG] = [
    [("auto_contrast", 0, 1)],
    [("equalize", 0, 1)],
    [("invert", 0, 1)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    [("solarize_add", 0.0, 0.43)],
    [("color", 0.1, 1.9)],
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("rotate", -30.0, 30.0)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    [("translate_x", -0.1, 0.1)],
    [("translate_x", -0.1, 0.1)],
]


@dataclass
class Data:
    image: Optional[torch.Tensor] = None
    segmentation: Optional[torch.Tensor] = None
    grid: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    image_only: bool = False


class Augmentation:
    def __init__(
        self,
        source_resolution: int = 224,
        target_resolution: int = 512,
        different_source_target_augmentation: bool = False,
        enable_rand_augment: bool = False,
        enable_random_resize_crop: bool = True,
        enable_horizontal_flip: bool = True,
        random_scale_ratio: Optional[tuple[tuple[float, float], tuple[float, float]]] = ((0.7, 1.3), (0.7, 1.3)),
        source_normalization: Optional[Callable] = None,
        target_normalization: Optional[Callable] = None,
    ):
        self.source_resolution = source_resolution
        self.source_normalization = source_normalization or get_simple_transform()
        self.target_normalization = target_normalization or get_stable_diffusion_transforms(target_resolution)
        self.different_source_target_augmentation = different_source_target_augmentation

        source_transforms = []

        if enable_random_resize_crop:
            resize_resolution = source_resolution
            if different_source_target_augmentation is False:
                resize_resolution = max(source_resolution, target_resolution)
            
            scale, ratio = random_scale_ratio
            source_transforms.append(K.RandomResizedCrop(size=(resize_resolution, resize_resolution), scale=scale, ratio=ratio, p=1.0))  # For logistical reasons

        if enable_horizontal_flip:
            source_transforms.append(K.RandomHorizontalFlip(p=0.95))

        if enable_rand_augment:
            assert enable_random_resize_crop
            source_transforms.append(RandAugment(n=2, m=10, policy=randaug_policy))

        target_transforms = []
        if different_source_target_augmentation:
            target_transforms.extend([K.Resize(size=(target_resolution, target_resolution)), K.RandomHorizontalFlip(p=0.95)])

        self.source_transform = AugmentationSequential(*source_transforms) if len(source_transforms) > 0 else None
        self.target_transform = AugmentationSequential(*target_transforms) if len(target_transforms) > 0 else None
        self.has_warned = False

    def kornia_augmentations_enabled(self) -> bool:
        return self.source_transform is not None or self.target_transform is not None

    def __call__(self, source_data, target_data):
        if not self.has_warned:
            warnings.warn(f"Source image is being resized to {self.source_normalization.transforms[0].size}")
            warnings.warn(f"Target image is being resized to {self.target_normalization.transforms[0].size}")
            self.has_warned = True

        assert source_data.image.shape == target_data.image.shape, f"Source and target image shapes do not match: {source_data.image.shape} != {target_data.image.shape}"
        initial_h = source_data.image.shape[-2]
        initial_w = source_data.image.shape[-1]

        if self.source_transform is not None:
            # When we augment the source we need to also augment the target
            source_params = self.source_transform.forward_parameters(batch_shape=source_data.image.shape)
            source_data = process(aug=self.source_transform, params=source_params, input_data=source_data)
            if self.different_source_target_augmentation is False:
                target_data = process(aug=self.source_transform, params=source_params, input_data=target_data)

        if self.different_source_target_augmentation and self.target_transform is not None:
            target_params = self.target_transform.forward_parameters(batch_shape=target_data.image.shape)
            target_data = process(aug=self.target_transform, params=target_params, input_data=target_data)
        
        source_data.image = self.source_normalization(source_data.image)
        target_data.image = self.target_normalization(target_data.image)

        # TODO: Ideally we should apply all transforms through Kornia.
        # However, torchvision transforms are the defacto standard and this allows us to directly take the normalization from e.g., timm
        # The below code applies the same transform to the segmentation mask (generally a resize and crop) and skips the normalization but is not ideal
        if source_data.segmentation is not None:
            source_data.segmentation = apply_non_image_normalization_transforms(source_data.segmentation, self.source_normalization).long()
        
        if source_data.grid is not None:
            source_data.grid = apply_non_image_normalization_transforms(source_data.grid.permute(0, 3, 1, 2), self.source_normalization).float()
            source_data.grid[:, 0][source_data.grid[:, 0] >= 0] /= (initial_h - 1)
            source_data.grid[:, 1][source_data.grid[:, 1] >= 0] /= (initial_w - 1)

        if target_data.segmentation is not None:
            target_data.segmentation = apply_non_image_normalization_transforms(target_data.segmentation, self.target_normalization).long()
        
        if target_data.grid is not None:
            target_data.grid = apply_non_image_normalization_transforms(target_data.grid.permute(0, 3, 1, 2), self.target_normalization).float()
            target_data.grid[:, 0][target_data.grid[:, 0] >= 0] /= (initial_h - 1)
            target_data.grid[:, 1][target_data.grid[:, 1] >= 0] /= (initial_w - 1)
        
        try:
            mask = target_data.grid >= 0
            assert 0 <= target_data.grid[mask].min() <= target_data.grid.max() <= 1 and torch.all(target_data.grid[~mask] == -1)
        except:
            pass

        return source_data, target_data

def apply_non_image_normalization_transforms(data, normalization_transform):
    if normalization_transform is None:
        return data

    data = data + 1
    for transform_ in normalization_transform.transforms:
        if transform_.__class__.__name__ not in ("Resize", "Normalize", "CenterCrop", "Compose"):
            raise ValueError(f"Transform {transform_.__class__.__name__} not supported")
        
        if transform_.__class__.__name__ in ("Normalize", "Compose", "CenterCrop"): 
            continue
        elif transform_.__class__.__name__ == "Resize":
            transform_ = transforms.Resize(size=transform_.size, max_size=transform_.max_size, antialias=transform_.antialias, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

        data = transform_(data)

    data = data - 1
    return data

def process(aug: AugmentationSequential, input_data: Data, should_viz: bool = False, params: Optional[List[Any]] = None, **kwargs):
    """
    Args:
        input_image: (B, C, H, W)
        input_segmentation_mask: (B, H, W)
    """

    if not input_data.image_only:
        B, _, H, W = input_data.image.shape
        keypoints = get_keypoints(B, H, W)
        assert input_data.segmentation.ndim == 3
        input_data.segmentation = input_data.segmentation.unsqueeze(1).float()

    # output_tensor and output_segmentation are affected by resize but output_keypoints are not
    output_data = Data(image_only=input_data.image_only)
    if input_data.image_only:
        output_data.image = aug(input_data.image, data_keys=["image"], params=params)
    else:
        output_data.image, output_data.segmentation, output_keypoints = aug(
            input_data.image, input_data.segmentation, keypoints, data_keys=["image", "mask", "keypoints"], params=params
        )

    if not input_data.image_only:
        output_data.grid, output_data.mask = process_output_keypoints(output_keypoints, B, H, W, output_data.image.shape[-2], output_data.image.shape[-1])
        output_data.segmentation = process_output_segmentation(output_keypoints, output_data.segmentation.squeeze(1), output_data.image.shape[-2], output_data.image.shape[-1], -1)

    if should_viz:
        Im(input_image.permute(0, 2, 3, 1)[:, output_data.grid[..., 0], output_data.grid[..., 1]][0]).save()
        keypoints_viz = get_viz_keypoints(B, H, W)
        output_keypoints_viz = aug(keypoints_viz, data_keys=["keypoints"], params=aug._params)
        viz(input_image, keypoints_viz, output_data.image, output_keypoints_viz)

    return output_data


if __name__ == "__main__":
    aug = AugmentationSequential(
        K.RandomResizedCrop(size=(192, 192), scale=(0.7, 1.3), ratio=(0.7, 1.3)),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        RandAugment(n=2, m=10, policy=randaug_policy),
        data_keys=["input", "keypoints", "keypoints"],
        random_apply=True,
    )
    image = Im("https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png").pil
    input_image = (
        Im(image.crop(((image.size[0] - image.size[1]) // 2, 0, image.size[0] - (image.size[0] - image.size[1]) // 2, image.size[1])))
        .resize(224, 224)
        .torch.repeat(10, 1, 1, 1)
    )
    input_segmentation = torch.randint(0, 2, (10, 224, 224))
    process(aug=aug, input_data=Data(image=input_image, segmentation=input_segmentation), should_viz=True)
