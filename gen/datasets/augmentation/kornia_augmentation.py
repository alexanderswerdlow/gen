from dataclasses import dataclass
import os
from typing import Any, Callable, List
from einops import rearrange

import kornia.augmentation as K
from kornia.constants import Resample
import torch
from git import Optional
from image_utils import Im
from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.container import AugmentationSequential
import torchvision.transforms.v2 as transforms

from gen.datasets.augmentation.utils import get_keypoints, get_viz_keypoints, process_output_keypoints, process_output_segmentation, viz
import warnings
import random

randaug_policy: List[Any] = [
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
        initial_resolution: int = 256,
        different_source_target_augmentation: bool = False,
        return_grid: bool = True,
        center_crop: bool = True,
        enable_square_crop: bool = True,
        enable_rand_augment: bool = False,
        enable_random_resize_crop: bool = True,
        enable_horizontal_flip: bool = True,
        source_random_scale_ratio: Optional[tuple[tuple[float, float], tuple[float, float]]] = ((0.7, 1.3), (0.7, 1.3)),
        target_random_scale_ratio: Optional[tuple[tuple[float, float], tuple[float, float]]] = ((0.7, 1.3), (0.7, 1.3)),
        kornia_resize_mode: str = "BICUBIC",
        source_resolution: Optional[int] = None, # By default, we keep initial_resolution and let source_normalization resize further if enabled
        target_resolution: Optional[int] = None, # By default, we keep initial_resolution and let target_normalization resize further if enabled
        source_normalization: Optional[Callable] = None,
        target_normalization: Optional[Callable] = None,
    ):
        self.source_resolution = source_resolution
        self.source_normalization = source_normalization
        self.target_resolution = target_resolution
        self.target_normalization = target_normalization
        self.initial_resolution = initial_resolution
        self.kornia_resize_mode = kornia_resize_mode
        self.enable_square_crop = enable_square_crop
        self.return_grid = return_grid
        self.center_crop = center_crop

        if self.return_grid: assert self.enable_square_crop, "Grids only seem to work on square images for now."

        if self.source_normalization is None or self.target_normalization is None:
            print("Warning: source_normalization and target_normalization are None. This is not recommended.")
        
        self.different_source_target_augmentation = different_source_target_augmentation

        main_transforms = []
        if enable_random_resize_crop:
            resize_resolution = self.target_resolution if different_source_target_augmentation and target_resolution is not None else self.initial_resolution
            target_scale, target_ratio = target_random_scale_ratio
            main_transforms.append(K.RandomResizedCrop(size=(resize_resolution, resize_resolution), scale=target_scale, ratio=target_ratio, resample=self.kornia_resize_mode, p=1.0))  # For logistical reasons

        if enable_horizontal_flip:
            main_transforms.append(K.RandomHorizontalFlip(p=0.5))

        if enable_rand_augment:
            assert enable_random_resize_crop
            main_transforms.append(RandAugment(n=2, m=10, policy=randaug_policy))

        # When we augment source/target differently, we want the target to be more augmented [e.g., a smaller crop]
        if different_source_target_augmentation:
            source_scale, source_ratio = source_random_scale_ratio
            resize_resolution = self.source_resolution if self.source_resolution is not None else initial_resolution
            source_transforms = [K.RandomResizedCrop(size=(resize_resolution, resize_resolution), scale=source_scale, ratio=source_ratio, resample=self.kornia_resize_mode, p=1.0)]
            target_transforms = main_transforms
        else:
            # If source == target, we use the target augmentations as they are generally heavier
            assert source_random_scale_ratio is None
            source_transforms = main_transforms
            target_transforms = []

        self.source_transform = AugmentationSequential(*source_transforms) if len(source_transforms) > 0 else None
        self.target_transform = AugmentationSequential(*target_transforms) if len(target_transforms) > 0 else None
        self.has_warned = False

    def kornia_augmentations_enabled(self) -> bool:
        return self.source_transform is not None or self.target_transform is not None

    def __call__(self, source_data, target_data):
        if not self.has_warned and self.source_normalization is not None and self.target_normalization is not None:
            warnings.warn(f"Source image is being resized to {self.source_normalization.transforms[0].size}")
            warnings.warn(f"Target image is being resized to {self.target_normalization.transforms[0].size}")
            self.has_warned = True

        assert source_data.image.shape == target_data.image.shape, f"Source and target image shapes do not match: {source_data.image.shape} != {target_data.image.shape}"
        initial_h, initial_w = source_data.image.shape[-2], source_data.image.shape[-1]

        if self.enable_square_crop:
            assert source_data.mask is None and target_data.mask is None, "Square crop is not supported with masks"
            assert source_data.grid is None and target_data.grid is None, "Square crop is not supported with grids"
            source_data.image, source_data.segmentation, target_data.image, target_data.segmentation = crop_to_square(source_data.image, source_data.segmentation, target_data.image, target_data.segmentation, center=self.center_crop)

        if self.source_transform is not None:
            # When we augment the source we need to also augment the target
            source_params = self.source_transform.forward_parameters(batch_shape=source_data.image.shape)
            source_data = process(aug=self.source_transform, params=source_params, input_data=source_data)
            if self.different_source_target_augmentation is False:
                target_data = process(aug=self.source_transform, params=source_params, input_data=target_data)

        if self.different_source_target_augmentation and self.target_transform is not None:
            target_params = self.target_transform.forward_parameters(batch_shape=target_data.image.shape)
            target_data = process(aug=self.target_transform, params=target_params, input_data=target_data)
        

        source_data = apply_normalization_transforms(source_data, self.source_normalization, initial_h, initial_w)
        target_data = apply_normalization_transforms(target_data, self.target_normalization, initial_h, initial_w)
        
        try:
            mask = target_data.grid >= 0
            assert 0 <= target_data.grid[mask].min() <= target_data.grid.max() <= 1 and torch.all(target_data.grid[~mask] == -1)
        except:
            pass

        if not self.return_grid:
            source_data.grid = None
            target_data.grid = None

        return source_data, target_data
    
def crop_to_square(*args, center=True):
    """
    Crop a 4D tensor representing an image to a square shape.
    
    Parameters:
    - tensor (torch.Tensor): Input tensor with shape [1, 3, height, width].
    - center (bool): If True, crop the center. If False, crop a random part.
    
    Returns:
    - torch.Tensor: Cropped tensor with shape [1, 3, min_dim, min_dim].
    """
    height, width = args[0].shape[-2:]
    min_dim = min(height, width)
    
    if center:
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
    else:
        start_x = random.randint(0, width - min_dim)
        start_y = random.randint(0, height - min_dim)
    
    return [tensor[..., start_y:start_y+min_dim, start_x:start_x+min_dim] for tensor in args]

def santitize_and_normalize_grid_values(tensor, patch_size = 4):
    tensor[tensor < 0] = torch.nan
    B, C, H, W = tensor.shape
    if H == 518:
        patch_size = 7
    assert H // patch_size == W // patch_size
    assert H % patch_size == 0 and W % patch_size == 0
    n_patches = H // patch_size
    patch_size = H // n_patches
    patches = rearrange(tensor, 'b c (h p1) (w p2) -> b c (h w) p1 p2', p1=patch_size, p2=patch_size) # Reshape into patches
    patch_means = torch.nanmean(patches, dim=(-1, -2), keepdim=True) # Compute the mean, ignoring NaNs
    patch_means[torch.isnan(patch_means)] = 0 # Where the patch is entirely NaN, set the mean to zero (or another default value)
    patches_filled = torch.where(torch.isnan(patches), patch_means, patches) # Replace NaNs in the original patches with their respective patch means
    tensor = rearrange(patches_filled, 'b c (h w) p1 p2 -> b c (h p1) (w p2)', h=n_patches, w=n_patches)
    assert 0 <= tensor.min() <= tensor.max() <= 1
    return (2 * tensor) - 1 # Normalize from -1 to 1

def apply_normalization_transforms(data: Data, normalization_transform, initial_h, initial_w):
    if data.grid is not None:
        data.grid = data.grid.permute(0, 3, 1, 2).float()

    if normalization_transform is not None:
        if normalization_transform.transforms[0].interpolation != transforms.InterpolationMode.BICUBIC:
            if data.image.is_floating_point() and -128 <= data.image.min() <= data.image.max() <= 128:
                data.image = torch.clamp(data.image, 0, 1)

        data.image = normalization_transform(data.image)

        # TODO: Ideally we should apply all transforms through Kornia.
        # However, torchvision transforms are the defacto standard and this allows us to directly take the normalization from e.g., timm
        # The below code applies the same transform to the segmentation mask (generally a resize and crop) and skips the normalization but is not ideal
        if data.segmentation is not None:
            data.segmentation = apply_non_image_normalization_transforms(data.segmentation, normalization_transform).long()
        
        if data.grid is not None:
            data.grid = apply_non_image_normalization_transforms(data.grid, normalization_transform)

    data.grid[:, 0][data.grid[:, 0] >= 0] /= (initial_h - 1)
    data.grid[:, 1][data.grid[:, 1] >= 0] /= (initial_w - 1)
    data.grid = santitize_and_normalize_grid_values(data.grid)

    return data

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

def apply_non_image_normalization_transforms(tensor_, normalization_transform):
    if normalization_transform is None:
        return tensor_

    tensor_ = tensor_ + 1
    for transform_ in normalization_transform.transforms:
        if transform_.__class__.__name__ not in ("Resize", "Normalize", "CenterCrop", "Compose"):
            raise ValueError(f"Transform {transform_.__class__.__name__} not supported")
        
        if transform_.__class__.__name__ in ("Normalize", "Compose", "CenterCrop"): 
            continue
        elif transform_.__class__.__name__ == "Resize":
            transform_ = transforms.Resize(size=transform_.size, max_size=transform_.max_size, antialias=transform_.antialias, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

        tensor_ = transform_(tensor_)

    tensor_ = tensor_ - 1
    return tensor_

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
