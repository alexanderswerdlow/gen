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
from scipy.spatial.transform import Rotation as R

from gen import DEFAULT_PROMPT, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_OVERFIT_DATASET_PATH
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
        num_objects: int = 23,
        augmentation: Optional[Augmentation] = Augmentation(),
        custom_split: Optional[str] = None,
        subset: Optional[tuple[str]] = None,
        return_video: bool = False,
        fake_return_n: Optional[int] = None,
        use_single_mask: bool = False, # Force using a single mask with all 1s
        num_cameras: int = 1,
        multi_camera_format: bool = False,
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
        self.multi_camera_format = multi_camera_format
        self.num_cameras = num_cameras

        if num_cameras > 1: assert multi_camera_format

        local_split = ("train" if self.split == Split.TRAIN else "validation")
        local_split = local_split if custom_split is None else custom_split
        self.root_dir = self.root / self.dataset / local_split

        if subset is not None:
            self.files = subset
        else:
            self.files = os.listdir(self.root_dir)
            self.files.sort()

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
    
    def map_idx(self, idx):
        file_idx = idx // (self.num_cameras * self.num_frames)
        camera_idx = (idx % (self.num_cameras * self.num_frames)) // self.num_frames
        frame_idx = idx % self.num_frames
        return file_idx, camera_idx, frame_idx

    def __getitem__(self, index):
        file_idx, camera_idx, frame_idx = self.map_idx(index)
        if self.fake_return_n:
            file_idx = 0

        try:
            path = self.files[file_idx]
        except IndexError:
            print(f"Index {file_idx} is out of bounds for dataset of size {len(self.files)} for dir: {self.root_dir}")
            raise

        ret = {}
        
        if self.multi_camera_format:
            data = np.load(self.root_dir / path / "data.npz")
            rgb = data["rgb"][camera_idx, frame_idx]
            instance = data["segment"][camera_idx, frame_idx]

            quaternions = data["quaternions"][camera_idx, frame_idx] # (23, 4)
            positions = data["positions"][camera_idx, frame_idx] # (23, 3)
            valid = data["valid"][camera_idx, :].squeeze(0) # (23, )
            categories = data["categories"][camera_idx, :].squeeze(0) # (23, )

            if 'camera_quaternions' in data:
                camera_quaternion = data['camera_quaternions'][camera_idx, frame_idx] # (4, )
                quaternions[~valid] = 1 # Set invalid quaternions to 1 to avoid 0 norm.
                quaternions = (R.from_quat(quaternions) * R.from_quat(camera_quaternion).inv()).as_quat()
                quaternions[~valid] = 0
            else:
                raise NotImplementedError("Camera quaternions not found in data.npz")
            
            ret.update({
                "quaternions": quaternions,
                "positions": positions,
                "valid": valid,
                "categories": categories,
            })
        else:
            assert self.num_cameras == 1 and camera_idx == 0
            rgb = os.path.join(self.root_dir, os.path.join(path, "rgb.npy"))
            instance = os.path.join(self.root_dir, os.path.join(path, "segment.npy"))
            bbx = os.path.join(self.root_dir, os.path.join(path, "bbox.npy"))

            rgb = np.load(rgb)
            bbx = np.load(bbx)
            instance = np.load(instance)

            if self.num_frames == 1 and rgb.shape[0] > 1:
                # Get middle frame
                frame_idx = rgb.shape[0] // 2

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

        ret.update({
            "gen_pixel_values": target_data.image,
            "gen_grid": target_data.grid,
            "gen_segmentation": target_data.segmentation,
            "disc_pixel_values": source_data.image,
            "disc_grid": source_data.grid,
            "disc_segmentation": source_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
        })

        if self.return_video:
            ret["video"] = path

        return ret

    def __len__(self):
        init_size = len(self.files) * self.num_frames * self.num_cameras
        return init_size * self.fake_return_n if self.fake_return_n else init_size


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
    new_dataset = MoviDataset(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        subset_size=None,
        dataset="movi_e",
        tokenizer=tokenizer,
        path=MOVI_MEDIUM_PATH,
        num_objects=23,
        num_frames=8,
        num_cameras=1, 
        augmentation=Augmentation(target_resolution=256, minimal_source_augmentation=True, enable_crop=True, enable_horizontal_flip=True),
        return_video=True,
        multi_camera_format=True,
    )
    dataloader = new_dataset.get_dataloader()
    for batch in dataloader:
        from image_utils import Im, get_layered_image_from_binary_mask
        gen_ = Im.concat_vertical(Im((batch['gen_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['gen_segmentation'].squeeze(0))))
        disc_ = Im.concat_vertical(Im((batch['disc_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['disc_segmentation'].squeeze(0))))
        print(batch['gen_segmentation'].sum() / batch['gen_segmentation'][0, ..., 0].numel(), batch['disc_segmentation'].sum() / batch['disc_segmentation'][0, ..., 0].numel())
        Im.concat_horizontal(gen_, disc_).save(batch['video'][0])

