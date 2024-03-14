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

from gen.utils.logging_utils import log_warn

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
    """
    TODO: One known issue with this class has to due with interpolation modes. Specifically, if the source and target normalization have different modes 
    then we might first resize with kornia_resize_mode, and then in some cases not resize (if we already have the desired resolution.)
    """
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
        enable_rotate: bool = False,
        reorder_segmentation: bool = False,
        source_random_scale_ratio: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
        target_random_scale_ratio: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
        kornia_resize_mode: str = "BICUBIC",
        source_resolution: Optional[int] = None, # By default, we keep initial_resolution and let source_transforms resize further if enabled
        target_resolution: Optional[int] = None, # By default, we keep initial_resolution and let target_transforms resize further if enabled
        source_transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
    ):
        self.source_resolution = source_resolution
        self.source_transforms = source_transforms
        self.target_resolution = target_resolution
        self.target_transforms = target_transforms
        self.initial_resolution = initial_resolution
        self.kornia_resize_mode = kornia_resize_mode
        self.enable_square_crop = enable_square_crop
        self.return_grid = return_grid
        self.center_crop = center_crop
        self.enable_rotate = enable_rotate
        self.reorder_segmentation = reorder_segmentation

        if self.return_grid: assert self.enable_square_crop, "Grids only seem to work on square images for now."

        if self.source_transforms is None or self.target_transforms is None:
            log_warn("Warning: source_transforms and target_transforms are None. This is not recommended.")
        
        self.different_source_target_augmentation = different_source_target_augmentation

        main_transforms = []

        if enable_rotate:
            main_transforms.append(K.RandomRotation(degrees=(-60, 60), p=0.5))

        if enable_random_resize_crop:
            resize_resolution = self.target_resolution if different_source_target_augmentation and target_resolution is not None else self.initial_resolution
            target_scale, target_ratio = target_random_scale_ratio
            main_transforms.append(K.RandomResizedCrop(size=(resize_resolution, resize_resolution), scale=target_scale, ratio=target_ratio, resample=self.kornia_resize_mode, p=1.0))

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
        if not self.has_warned and self.source_transforms is not None and self.target_transforms is not None:
            self.has_warned = True
            log_warn(f"Source image is being resized to {self.source_transforms.transforms[0].size}")
            log_warn(f"Target image is being resized to {self.target_transforms.transforms[0].size}")
        return self.source_transform is not None or self.target_transform is not None

    def __call__(self, source_data, target_data):
        assert source_data.image.shape == target_data.image.shape, f"Source and target image shapes do not match: {source_data.image.shape} != {target_data.image.shape}"
        initial_h, initial_w = source_data.image.shape[-2], source_data.image.shape[-1]

        if self.enable_square_crop:
            assert source_data.mask is None and target_data.mask is None, "Square crop is not supported with masks"
            assert source_data.grid is None and target_data.grid is None, "Square crop is not supported with grids"
            source_data.image, source_data.segmentation, target_data.image, target_data.segmentation = crop_to_square(source_data.image, source_data.segmentation, target_data.image, target_data.segmentation, center=self.center_crop)

        if self.reorder_segmentation:
            # This may result in a different ordering of the segmentation channels
            source_data.segmentation = reorder_segmentation(source_data.segmentation.clone().contiguous())
            target_data.segmentation = reorder_segmentation(target_data.segmentation.clone().contiguous())
            
        if self.source_transform is not None:
            # When we augment the source we need to also augment the target
            source_params = self.source_transform.forward_parameters(batch_shape=source_data.image.shape)
            source_data = process(aug=self.source_transform, params=source_params, input_data=source_data)
            if self.different_source_target_augmentation is False:
                target_data = process(aug=self.source_transform, params=source_params, input_data=target_data)

        if self.different_source_target_augmentation and self.target_transform is not None:
            target_params = self.target_transform.forward_parameters(batch_shape=target_data.image.shape)
            target_data = process(aug=self.target_transform, params=target_params, input_data=target_data)
        

        source_data = apply_normalization_transforms(source_data, self.source_transforms, initial_h, initial_w)
        target_data = apply_normalization_transforms(target_data, self.target_transforms, initial_h, initial_w)
        
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

def reorder_segmentation(tensor):
    # Re-order segmentation channels (except background). This is because during cropping, large segmentation masks may no longer be in view.
    if tensor.is_floating_point():
        tensor = tensor.long()
    
    try:
        unique, _, counts = tensor.unique(return_inverse=True, return_counts=True)
        _, indices_sorted = counts.sort(descending=True)
        new_ranks = torch.zeros(tensor.max() + 1, dtype=torch.long, device=tensor.device)  # Adjusted for direct indexing

        # We always keep the 0th channel at 0.
        zero_mask = unique != 0
        new_ranks[unique[indices_sorted][zero_mask]] = torch.arange((~zero_mask).sum().item(), len(unique), device=tensor.device)
    except Exception as e:
        print(e)
        print(tensor.min(), tensor.max(), unique)
        print(zero_mask)
        print(indices_sorted)
        print(unique)
        print(new_ranks)
        breakpoint()
    
    tensor = new_ranks[tensor]
    assert tensor.max() == len(unique) - 1
    
    return tensor.to(torch.float)

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
        keypoints._data = keypoints._data.to(input_data.image.device)
        keypoints._valid_mask = keypoints._valid_mask.to(input_data.image.device)
        
        if input_data.segmentation.ndim == 3:
            input_data.segmentation = input_data.segmentation.unsqueeze(1)

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
        output_data.segmentation = process_output_segmentation(output_keypoints, output_data.segmentation, output_data.image.shape[-2], output_data.image.shape[-1], -1)

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
