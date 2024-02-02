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

from gen import DEFAULT_PROMPT, MOVI_DATASET_PATH, MOVI_OVERFIT_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.datasets.utils import get_open_clip_transforms_v2, get_stable_diffusion_transforms

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask
from gen.utils.tokenization_utils import get_tokens

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
        num_dataset_frames: int = 24,
        num_objects: int = 23,
        augmentation: Optional[Augmentation] = Augmentation(),
        custom_split: Optional[str] = None,
        subset: Optional[tuple[str]] = None,
        return_video: bool = False,
        fake_return_n: Optional[int] = None,
        use_single_mask: bool = False, # Force using a single mask with all 1s
        new_format: bool = False,
        **kwargs,
    ):
        # Note: The super __init__ is handled by inherit_parent_args
        self.tokenizer = tokenizer
        self.root = path  # Path to the dataset containing folders of "movi_a", "movi_e", etc.
        self.dataset = dataset  # str of dataset name (e.g. "movi_a")
        self.resolution = resolution
        self.return_video = return_video
        self.fake_return_n = fake_return_n
        self.use_single_mask = use_single_mask
        self.new_format = new_format
        local_split = ("train" if self.split == Split.TRAIN else "validation")
        local_split = local_split if custom_split is None else custom_split
        self.root_dir = self.root / self.dataset / local_split

        if subset is not None:
            self.files = subset
        else:
            self.files = os.listdir(self.root_dir)
            self.files.sort()

        self.num_dataset_frames = num_dataset_frames
        self.num_frames = num_frames
        self.num_classes = num_objects

        self.augmentation = augmentation
        if self.split == Split.VALIDATION:
            self.augmentation.set_validation()

        self.override_text = override_text
        if self.override_text:
            warnings.warn(f"Overriding text captions with {DEFAULT_PROMPT}")

    def get_dataset(self):
        return self

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)

    def __getitem__(self, index):
        if self.fake_return_n:
            video_idx = 0
            frame_idx = index % self.num_dataset_frames
        else:
            video_idx = index // self.num_dataset_frames
            frame_idx = index % self.num_dataset_frames

        try:
            path = self.files[video_idx]
        except IndexError:
            print(f"Index {video_idx} is out of bounds for dataset of size {len(self.files)} for dir: {self.root_dir}")
            raise

        # if self.num_dataset_frames == 1 and rgb.shape[0] > 1:
        #     # Get middle frame
        #     frame_idx = rgb.shape[0] // 2
        
        if self.new_format:
            breakpoint()
        else:
            rgb = os.path.join(self.root_dir, os.path.join(path, "rgb.npy"))
            instance = os.path.join(self.root_dir, os.path.join(path, "segment.npy"))
            bbx = os.path.join(self.root_dir, os.path.join(path, "bbox.npy"))

            rgb = np.load(rgb)
            bbx = np.load(bbx)
            instance = np.load(instance)

            rgb = rgb[frame_idx]
            instance = instance[frame_idx]

            bbx = bbx[frame_idx]
            bbx[..., [0, 1]] = bbx[..., [1, 0]]
            bbx[..., [2, 3]] = bbx[..., [3, 2]]

            bbx[..., [0, 2]] *= rgb.shape[1]
            bbx[..., [1, 3]] *= rgb.shape[0]
            bbx = torch.from_numpy(bbx)

        assert rgb.shape[0] == rgb.shape[1]

        rgb = rearrange(rgb, "... h w c -> ... c h w") / 255.0  # [0, 1]

        source_data, target_data = self.augmentation(
            source_data=Data(image=torch.from_numpy(rgb[None]).float(), segmentation=torch.from_numpy(instance[None].squeeze(-1)).float()),
            target_data=Data(image=torch.from_numpy(rgb[None]).float(), segmentation=torch.from_numpy(instance[None].squeeze(-1)).float()),
        )

        # We have -1 as invalid so we simply add 1 to all the labels to make it start from 0 and then later remove the 1st channel
        source_data.image = source_data.image.squeeze(0)
        source_data.segmentation = torch.nn.functional.one_hot(source_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]
        target_data.image = target_data.image.squeeze(0)
        target_data.segmentation = torch.nn.functional.one_hot(target_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]

        if self.use_single_mask:
            source_data.segmentation = torch.ones_like(source_data.segmentation)[..., [0]]
            target_data.segmentation = torch.ones_like(target_data.segmentation)[..., [0]]

        ret = {
            "gen_pixel_values": target_data.image,
            "gen_grid": target_data.grid,
            "gen_segmentation": target_data.segmentation,
            "disc_pixel_values": source_data.image,
            "disc_grid": source_data.grid,
            "disc_segmentation": source_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
        }

        if self.return_video:
            ret["video"] = path

        return ret

    def __len__(self):
        if self.fake_return_n is not None:
            return len(self.files) * self.num_dataset_frames * self.fake_return_n
        return len(self.files) * self.num_dataset_frames


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = MoviDataset(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        subset_size=None,
        dataset="movi_e",
        num_frames=24,
        tokenizer=tokenizer,
        path=MOVI_OVERFIT_DATASET_PATH,
        num_objects=1,
        augmentation=Augmentation(minimal_source_augmentation=True, enable_crop=True, enable_horizontal_flip=True),
        return_video=True
    )
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        from image_utils import Im, get_layered_image_from_binary_mask
        gen_ = Im.concat_vertical(Im((batch['gen_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['gen_segmentation'].squeeze(0))))
        disc_ = Im.concat_vertical(Im((batch['disc_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['disc_segmentation'].squeeze(0))))
        print(batch['gen_segmentation'].sum() / batch['gen_segmentation'][0, ..., 0].numel(), batch['disc_segmentation'].sum() / batch['disc_segmentation'][0, ..., 0].numel())
        Im.concat_horizontal(gen_, disc_).save(batch['video'][0])

