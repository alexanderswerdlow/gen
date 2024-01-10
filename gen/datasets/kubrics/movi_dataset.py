import autoroot

import os
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import open_clip
import torch
import torchvision
import webdataset as wds
from einops import rearrange
from ipdb import set_trace as st
from torch.utils.data import Dataset

from gen import MOVI_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.datasets.utils import get_open_clip_transforms_v2, get_stable_diffusion_transforms

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask


@inherit_parent_args
class MoviDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        tokenizer: Optional[Any] = None,
        path: Path = MOVI_DATASET_PATH,
        resolution: int = 512,
        override_text: bool = True,
        dataset: str = "movi_e",
        num_frames: int = 24,
        augment: bool = False,
        num_dataset_frames: int = 24,
        num_objects: int = 23,
        legacy_transforms: bool = False,
        augmentation: Optional[Augmentation] = None,
        **kwargs,
    ):
        # Note: The super __init__ is handled by inherit_parent_args
        self.tokenizer = tokenizer
        self.root = path  # Path to the dataset containing folders of "movi_a", "movi_e", etc.
        self.dataset = dataset  # str of dataset name (e.g. "movi_a")
        self.resolution = resolution
        self.legacy_transforms = legacy_transforms
        self.root_dir = self.root / self.dataset / ("train" if self.split == Split.TRAIN else "validation")
        self.files = os.listdir(self.root_dir)
        self.files.sort()

        self.num_dataset_frames = num_dataset_frames
        self.num_frames = num_frames
        self.augment = augment
        self.num_classes = num_objects

        if self.legacy_transforms:
            self.gen_image_transforms = get_stable_diffusion_transforms(resolution)
            self.disc_image_transforms = get_open_clip_transforms_v2()
        else:
            self.augmentation = augmentation

        self.override_text = override_text
        if self.override_text:
            warnings.warn("Overriding text captions with 'A photo of'")

    def get_dataset(self):
        return self

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)

    def __getitem__(self, index):
        video_idx = index // self.num_dataset_frames
        frame_idx = index % self.num_dataset_frames

        try:
            path = self.files[video_idx]
        except IndexError:
            print(f"Index {video_idx} is out of bounds for dataset of size {len(self.files)} for dir: {self.root_dir}")
            raise

        rgb = os.path.join(self.root_dir, os.path.join(path, "rgb.npy"))
        instance = os.path.join(self.root_dir, os.path.join(path, "segment.npy"))
        bbx = os.path.join(self.root_dir, os.path.join(path, "bbox.npy"))

        rgb = np.load(rgb)
        bbx = np.load(bbx)
        instance = np.load(instance)

        # For returning videos
        # rand_id = random.randint(0, 24 - self.num_frames)
        # real_idx = [rand_id + j for j in range(self.num_frames)]

        rgb = rgb[frame_idx]
        bbx = bbx[frame_idx]
        instance = instance[frame_idx]

        bbx[..., [0, 1]] = bbx[..., [1, 0]]
        bbx[..., [2, 3]] = bbx[..., [3, 2]]

        bbx[..., [0, 2]] *= rgb.shape[1]
        bbx[..., [1, 3]] *= rgb.shape[0]

        bbx = torch.from_numpy(bbx)

        assert rgb.shape[0] == rgb.shape[1]

        rgb = rearrange(rgb, "... h w c -> ... c h w") / 255.0  # [0, 1]

        if self.legacy_transforms:
            bounding_boxes = BoundingBoxes(bbx, format=BoundingBoxFormat.XYXY, canvas_size=rgb.shape[-2:])
            instance = Mask(instance.squeeze(-1))

            gen_rgb, gen_bbx, gen_instance = self.gen_image_transforms(Image(rgb), bounding_boxes, instance)
            gen_instance = torch.nn.functional.one_hot(gen_instance.long(), num_classes=self.num_classes + 1).numpy()

            gen_bbx[..., [0, 2]] /= gen_rgb.shape[1]
            gen_bbx[..., [1, 3]] /= gen_rgb.shape[2]
            assert gen_bbx.min() >= 0 and gen_bbx.max() <= 1

            disc_rgb, disc_bbx, disc_instance = self.disc_image_transforms(Image(rgb), bounding_boxes, instance)
            disc_instance = torch.nn.functional.one_hot(disc_instance.long(), num_classes=self.num_classes + 1).numpy()
            disc_bbx[..., [0, 2]] /= disc_rgb.shape[1]
            disc_bbx[..., [1, 3]] /= disc_rgb.shape[2]
            assert disc_bbx.min() >= 0 and disc_bbx.max() <= 1

            ret = {
                "gen_pixel_values": gen_rgb,
                "gen_bbox": gen_bbx,
                "gen_segmentation": gen_instance,
                "disc_pixel_values": disc_rgb,
                "disc_bbox": disc_bbx,
                "disc_segmentation": disc_instance,
                "input_ids": self.tokenizer(
                    "A photo of", max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
            }

        else:
            source_data, target_data = self.augmentation(
                source_data=Data(image=torch.from_numpy(rgb).float(), segmentation=torch.from_numpy(instance.squeeze(-1)).float()),
                target_data=Data(image=torch.from_numpy(rgb).float(), segmentation=torch.from_numpy(instance.squeeze(-1)).float()),
            )

            ret = {
                "gen_pixel_values": source_data.image,
                "gen_grid": source_data.grid,
                "gen_segmentation": source_data.segmentation,
                "disc_pixel_values": target_data.image,
                "disc_grid": target_data.grid,
                "disc_segmentation": target_data.segmentation,
                "input_ids": self.tokenizer(
                    "A photo of", max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
            }

        return ret

    def __len__(self):
        return len(self.files) * self.num_dataset_frames


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = MoviDataset(
        cfg=None,
        split=Split.VALIDATION,
        num_workers=0,
        batch_size=2,
        shuffle=True,
        random_subset=None,
        dataset="movi_e",
        augment=False,
        num_frames=24,
        tokenizer=tokenizer,
    )
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        print(batch)
