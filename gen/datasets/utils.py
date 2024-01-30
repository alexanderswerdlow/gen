import warnings
import torch

def _convert_to_rgb(image):
    return image.convert("RGB")


def get_stable_diffusion_transforms(resolution):
    import torchvision.transforms.v2 as transforms

    return transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(resolution),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def get_open_clip_transforms_v1():
    import torchvision.transforms as transforms

    warnings.warn("Using Hardcoded OpenCLIP transforms. Make sure it is the same as the one used in the model.")
    return transforms.Compose(
        [
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(size=(224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def get_open_clip_transforms_v2():
    import torchvision.transforms.v2 as transforms

    warnings.warn("Using Hardcoded OpenCLIP transforms. Make sure it is the same as the one used in the model.")
    return transforms.Compose(
        [
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(size=(224, 224)),
            # _convert_to_rgb,
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )

def get_simple_transform():
    import torchvision.transforms.v2 as transforms
    return transforms.Compose(
        [
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

