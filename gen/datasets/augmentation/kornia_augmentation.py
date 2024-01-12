import autoroot

from typing import Any, List

import kornia.augmentation as K
import torch
from attr import dataclass
from git import Optional
from image_utils import Im
from kornia.augmentation.auto.base import SUBPLOLICY_CONFIG
from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.container import AugmentationSequential

from gen.datasets.augmentation.utils import get_keypoints, get_viz_keypoints, process_output_keypoints, process_output_segmentation, viz
from gen.datasets.utils import get_open_clip_transforms_v2, get_stable_diffusion_transforms

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


class Augmentation:
    def __init__(
        self,
        source_resolution: int = 224,
        target_resolution: int = 512,
        source_only_augmentation: bool = True,
        minimal_source_augmentation: bool = False,
        enable_crop: bool = True,
        enable_horizontal_flip: bool = True,
    ):
        self.source_resolution = source_resolution
        self.source_normalization = get_open_clip_transforms_v2()
        self.target_normalization = get_stable_diffusion_transforms(resolution=target_resolution)

        source_transforms = []

        if enable_crop:
            source_transforms.append(K.RandomResizedCrop(size=(source_resolution, source_resolution), scale=(0.7, 1.3), ratio=(0.7, 1.3), p=1.0))  # For logistical reasons

        if enable_horizontal_flip:
            source_transforms.append(K.RandomHorizontalFlip(p=0.95))

        if not minimal_source_augmentation:
            source_transforms.append(RandAugment(n=2, m=10, policy=randaug_policy))

        target_transforms = [K.Resize(size=(target_resolution, target_resolution))]
        if not source_only_augmentation:
            target_transforms.append(
                K.RandomHorizontalFlip(p=0.95),
            )

        self.source_transform = AugmentationSequential(*source_transforms)
        self.target_transform = AugmentationSequential(*target_transforms)

    def set_validation(self):
        pass

    def __call__(self, source_data, target_data):
        if self.source_transform is not None:
            # When we augment the source we need to also augment the target
            source_params = self.source_transform.forward_parameters(batch_shape=source_data.image.shape)
            source_data = process(aug=self.source_transform, params=source_params, input_data=source_data)
            target_data = process(aug=self.source_transform, params=source_params, input_data=target_data)

        if self.target_transform is not None:
            target_params = self.target_transform.forward_parameters(batch_shape=target_data.image.shape)
            target_data = process(aug=self.target_transform, params=target_params, input_data=target_data)

        source_data.image = self.source_normalization(source_data.image)
        target_data.image = self.target_normalization(target_data.image)

        return source_data, target_data


def process(aug: AugmentationSequential, input_data: Data, should_viz: bool = False, params: Optional[List[Any]] = None, **kwargs):
    """
    Args:
        input_image: (B, C, H, W)
        input_segmentation_mask: (B, H, W)
    """

    B, _, H, W = input_data.image.shape
    keypoints = get_keypoints(B, H, W)

    assert input_data.segmentation.ndim == 3
    input_data.segmentation = input_data.segmentation.unsqueeze(1).float()

    # output_tensor and output_segmentation are affected by resize but output_keypoints are not
    output_data = Data()
    output_data.image, output_data.segmentation, output_keypoints = aug(
        input_data.image, input_data.segmentation, keypoints, data_keys=["image", "mask", "keypoints"], params=params
    )

    output_data.grid, output_data.mask = process_output_keypoints(output_keypoints, B, H, W, output_data.image.shape[-2], output_data.image.shape[-1])
    output_data.segmentation = process_output_segmentation(output_data.segmentation.squeeze(1), output_data.mask, H, W)

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
