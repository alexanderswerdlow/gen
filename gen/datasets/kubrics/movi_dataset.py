import autoroot

import os
import warnings
from functools import cached_property
from itertools import chain, permutations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import open_clip
import torch
import torchvision
import webdataset as wds
from einops import rearrange
from einx import roll
from ipdb import set_trace as st
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from gen import (DEFAULT_PROMPT, MOVI_DATASET_PATH, MOVI_MEDIUM_PATH, MOVI_MEDIUM_SINGLE_OBJECT_PATH, MOVI_MEDIUM_TWO_OBJECTS_PATH,
                 MOVI_OVERFIT_DATASET_PATH)
from gen.configs.utils import inherit_parent_args
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.utils.decoupled_utils import load_tensor_dict, save_tensor_dict

torchvision.disable_beta_transforms_warning()
import io

from gen.utils.tokenization_utils import get_tokens


@inherit_parent_args
class MoviDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        tokenizer: Optional[Any] = None,
        path: Path = MOVI_DATASET_PATH,
        resolution: int = 512,
        dataset: str = "movi_e",
        num_frames: int = 24,
        num_objects: int = 23, # We need to return a consistent segmentation mask with this many channels + 1 (for the background)
        augmentation: Optional[Augmentation] = Augmentation(),
        custom_split: Optional[str] = None, # Specifies train or validation
        subset: Optional[tuple[str]] = None, # Specifies an optional subset of scenes to use
        fake_return_n: Optional[int] = None, # Fake that we have n times more elements. Useful for debugging on small datasets.
        use_single_mask: bool = False, # Force using a single mask with all 1s
        num_cameras: int = 1,
        multi_camera_format: bool = False, # The default format that supports multiple cameras and does not require an intermediate TFDS conversion
        cache_in_memory: bool = False, # Cache the entire dataset from file to memory in a compressed format
        cache_instances_in_memory: bool = False, # Cache each intermediate result after processing in memory in a compressed format
        num_subset: Optional[int] = None,
        return_multiple_frames: Optional[int] = None,
        **kwargs,
    ):
        # Note: The super __init__ is handled by inherit_parent_args
        self.tokenizer = tokenizer
        self.root = path  # Path to the dataset containing folders of "movi_a", "movi_e", etc.
        self.dataset = dataset  # str of dataset name (e.g. "movi_a")
        self.resolution = resolution
        self.fake_return_n = fake_return_n
        self.use_single_mask = use_single_mask
        self.multi_camera_format = multi_camera_format
        self.num_cameras = num_cameras
        self.cache_in_memory = cache_in_memory
        self.cache_instances_in_memory = cache_instances_in_memory
        self.return_multiple_frames = return_multiple_frames

        if num_cameras > 1: assert multi_camera_format

        local_split = ("train" if self.split == Split.TRAIN else "validation")
        local_split = local_split if custom_split is None else custom_split
        self.root_dir = self.root / self.dataset / local_split

        if subset is not None:
            self.files = subset
        else:
            self.files = os.listdir(self.root_dir)
            self.files.sort()

        if num_subset is not None:
            self.files = self.files[:num_subset]
            print(f"Using subset of {num_subset} files: {self.files}")

        self.num_frames = num_frames
        self.num_classes = num_objects
        self.augmentation = augmentation

        if self.cache_in_memory:
            self.cache = {}

        if self.cache_instances_in_memory:
            self.index_cache = {}

        if self.return_multiple_frames is not None:
            assert self.augmentation.kornia_augmentations_enabled() is False

    def get_dataset(self):
        return self

    def collate_fn(self, batch):
        if self.return_multiple_frames:
            # When we return multiple frames, we return a list [which is batched, resulting in a list of lists].
            # For simplicity, we will then return (b frames) ... for all elements
            batch = list(chain.from_iterable(batch))
        return torch.utils.data.default_collate(batch)
    
    def num_unique_scene_elements(self):
        """
        Returns either how many individual frames in a single camera OR how many unique frame permutations we can have [if we are returning multiple frames at a time].
        """
        return len(self.get_frame_permutations) if self.return_multiple_frames else (self.num_cameras * self.num_frames)
    
    def map_idx(self, idx):
        unique_frames = self.num_unique_scene_elements()
        file_idx = idx // unique_frames
        camera_idx = (idx % unique_frames) // self.num_frames # This value is ignored when we have return_multiple_frames
        frame_idx = idx % (unique_frames if self.return_multiple_frames else self.num_frames)
        return file_idx, camera_idx, frame_idx
    
    @cached_property
    def get_frame_permutations(self):
        return list(permutations(range(self.num_frames * self.num_cameras), self.return_multiple_frames))

    def __getitem__(self, index):
        file_idx, camera_idx, frame_idx = self.map_idx(index)        

        if self.fake_return_n: # Fake that we have n times more elements so we module the real file size
            file_idx = file_idx % len(self.files)

        try:
            path = self.files[file_idx]
        except IndexError:
            print(f"Index {file_idx} is out of bounds for dataset of size {len(self.files)} for dir: {self.root_dir}")
            raise
            
        data = None
        if self.multi_camera_format:
            file = self.root_dir / path / "data.npz"
            if self.cache_in_memory:
                # We cache the file in the compressed form (as raw BytesIO) as the uncompressed size is far too large.
                if file_idx in self.cache:
                    file = self.cache[file_idx]
                else:
                    with open(file, 'rb') as f:
                        file = io.BytesIO(f.read())
                        self.cache[file_idx] = file
                file.seek(0)

            data = np.load(file)
            

        if self.return_multiple_frames:
            # In this context, frame_idx actually refers to the specific permutation of frames we want to return.
            true_frame_idxs = self.get_frame_permutations[frame_idx]
            combined_frame_data = []
            for true_frame_idx in true_frame_idxs:
                frame_data = self.fetch_data((file_idx, true_frame_idx // self.num_frames, true_frame_idx % self.num_frames), path=path, data=data)
                combined_frame_data.append(frame_data)

            return combined_frame_data

        return self.fetch_data((file_idx, camera_idx, frame_idx), path=path, data=data)


    def fetch_data(self, combined_idx, path=None, data=None):
        file_idx, camera_idx, frame_idx = combined_idx

        ret = {
            "metadata": {
                "id": str(path),
                "path": str(self.root_dir / path / "data.npz"),
                "file_idx": file_idx,
                "frame_idx": frame_idx,
                "camera_idx": camera_idx
            },
        }

        if self.cache_instances_in_memory and combined_idx in self.index_cache:
            bytes_io = self.index_cache[combined_idx]
            start_time = time.time()
            bytes_io.seek(0)
            ret.update(load_tensor_dict(bytes_io, object_keys=['asset_id']))
            print(f"Time taken to load from cache: {time.time() - start_time}")
            ret['asset_id'] = [str(x) for x in ret['asset_id']]
                    
            return ret
        
        if self.multi_camera_format:
            rgb = data["rgb"][camera_idx, frame_idx]
            instance = data["segment"][camera_idx, frame_idx]

            object_quaternions = data["quaternions"][camera_idx, frame_idx] # (23, 4)
            object_quaternions = roll('objects [wxyz]', object_quaternions, shift=(-1,))
            raw_object_quaternions = object_quaternions.copy()
            positions = data["positions"][camera_idx, frame_idx] # (23, 3)
            valid = data["valid"][camera_idx, :].squeeze(0) # (23, )
            categories = data["categories"][camera_idx, :].squeeze(0) # (23, )
            asset_id = data["asset_ids"][camera_idx, :].squeeze(0)

            camera_quaternion = data['camera_quaternions'][camera_idx, frame_idx] # (4, )
            camera_quaternion = roll('[wxyz]', camera_quaternion, shift=(-1,))
            camera_quaternion = R.from_quat(camera_quaternion)

            object_quaternions[~valid] = 1 # Set invalid quaternions to 1 to avoid 0 norm.
            object_quaternions = R.from_quat(object_quaternions)
            object_quaternions = camera_quaternion.inv() * object_quaternions
            object_quaternions = object_quaternions.as_quat()
            object_quaternions[~valid] = 0
            
            ret.update({
                "quaternions": object_quaternions,
                "raw_object_quaternions": raw_object_quaternions,
                "camera_quaternions": camera_quaternion.as_quat(),
                "positions": positions,
                "valid": valid,
                "categories": categories,
                "asset_id": [str(x) for x in asset_id],
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
            "gen_segmentation": target_data.segmentation,
            "disc_pixel_values": source_data.image,
            "disc_segmentation": source_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
        })

        if source_data.grid is not None: ret["disc_grid"] = source_data.grid
        if target_data.grid is not None: ret["gen_grid"] = target_data.grid

        if self.cache_instances_in_memory:
            bytes_io = io.BytesIO()
            save_tensor_dict({k:v for k,v in ret.items() if not isinstance(v, dict)}, bytes_io)
            self.index_cache[combined_idx] = bytes_io

        return ret

    def __len__(self):
        init_size = len(self.files) * self.num_unique_scene_elements()
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
    )
    new_dataset = MoviDataset(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=24,
        shuffle=True,
        subset_size=None,
        dataset="movi_e",
        tokenizer=tokenizer,
        path=MOVI_MEDIUM_SINGLE_OBJECT_PATH,
        num_objects=23,
        num_frames=8,
        num_cameras=2,
        augmentation=Augmentation(target_resolution=256, minimal_source_augmentation=True, enable_crop=False, enable_horizontal_flip=False),
        multi_camera_format=True,
        cache_in_memory=True,
        cache_instances_in_memory=False,
        num_subset=None,
        return_multiple_frames=2,
    )
    dataloader = new_dataset.get_dataloader()
    import time
    start_time = time.time()
    for batch in dataloader:
        # from ipdb import set_trace as st; st()
        print(f'Time taken: {time.time() - start_time}')
        start_time = time.time()
        # from image_utils import Im, get_layered_image_from_binary_mask
        # gen_ = Im.concat_vertical(Im((batch['gen_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['gen_segmentation'].squeeze(0))))
        # disc_ = Im.concat_vertical(Im((batch['disc_pixel_values'][0] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['disc_segmentation'].squeeze(0))))
        # print(batch['gen_segmentation'].sum() / batch['gen_segmentation'][0, ..., 0].numel(), batch['disc_segmentation'].sum() / batch['disc_segmentation'][0, ..., 0].numel())
        # Im.concat_horizontal(gen_, disc_).save(batch['metadata']['id'][0])

