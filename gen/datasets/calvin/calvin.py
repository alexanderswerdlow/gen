# Taken from : https://github.com/michaelnoi/scene_nvs/blob/main/scene_nvs/data/dataset.py
from collections import defaultdict
import autoroot

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import lmdb
import msgpack
import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F
from einops import rearrange
from joblib import Memory
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.ops import nms
from tqdm import tqdm

from gen import CALVIN_V0_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.imagefolder.run_sam import get_to_process
from gen.datasets.run_dataloader import MockTokenizer
from gen.datasets.scannetpp.scene_data import test_scenes, train_scenes, val_scenes
from gen.utils.data_defs import integer_to_one_hot, one_hot_to_integer, visualize_input_data
from gen.utils.decoupled_utils import (breakpoint_on_error, get_device, get_rank, get_time_sync, hash_str_as_int, sanitize_filename,
                                       set_timing_builtins, to_numpy)
from gen.utils.file_utils import get_available_path, sync_data
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import _get_tokens, get_tokens
from image_utils.standalone_image_utils import integer_to_color, onehot_to_color
from gen.datasets.imagefolder.run_sam import save_type_names


def coco_decode_rle(compressed_rle) -> np.ndarray:
    if isinstance(compressed_rle['counts'], str):
        compressed_rle['counts'] = compressed_rle['counts'].encode()

    binary_mask = mask_utils.decode(compressed_rle).astype(np.bool_)
    return binary_mask

def im_to_numpy(im):
    im.load()
    # unpack data
    e = Image._getencoder(im.mode, 'raw', im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast('B', (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data

@inherit_parent_args
class CalvinDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Path = CALVIN_V0_DATASET_PATH,
        augmentation: Optional[Augmentation] = None,
        return_encoder_normalized_tgt: bool = False,
        src_eq_tgt: bool = False,
        tokenizer = None,
        specific_scenes: Optional[list[str]] = None,

        # TODO: All these params are not actually used but needed because of a quick with hydra_zen
        image_pairs_per_scene: int = 16384,
        distance_threshold: tuple[float] = (0.30, 0.1, 0.12, 0.8),
        depth_map_type: str = "gt",
        depth_map: bool = False,
        scenes_slice: Optional[tuple] = None,
        frames_slice: Optional[tuple] = None,
        scenes: Optional[List[str]] = None,
        top_n_masks_only: int = 34,
        sync_dataset_to_scratch: bool = False,
        return_raw_dataset_image: bool = False,
        num_overlapping_masks: int = 6,
        single_scene_debug: bool = False,
        use_segmentation: bool = True,
        only_preprocess_seg: bool = False,
        scratch_only: bool = False,
        image_files: Optional[list] = None,
        use_new_seg: bool = False,
        no_filtering: bool = False,
        dummy_mask: bool = False,
        merge_masks: bool = False,
        num_objects=-1,
        resolution=-1,
        custom_split=None, # TODO: Needed for hydra
        path=None, # TODO: Needed for hydra
        num_frames=None, # TODO: Needed for hydra
        num_cameras=None, # TODO: Needed for hydra
        multi_camera_format=None, # TODO: Needed for hydra
        subset=None, # TODO: Needed for hydra
        fake_return_n=None, # TODO: Needed for hydra
        use_single_mask=None,# TODO: Needed for hydra
        cache_in_memory=None, # TODO: Needed for hydra
        cache_instances_in_memory= None, # TODO: Needed for hydra
        num_subset=None, # TODO: Needed for hydra
        object_ignore_threshold=None, # TODO: Needed for hydra
        use_preprocessed_masks=None, # TODO: Needed for hydra
        preprocessed_mask_type=None, # TODO: Needed for hydra
        erode_dialate_preprocessed_masks=None, # TODO: Needed for hydra
        camera_trajectory_window=None, # TODO: Needed for hydra
        return_different_views=None, # TODO: Needed for hydra
        bbox_area_threshold=None, # TODO: Needed for hydra
        bbox_overlap_threshold=None, # TODO: Needed for hydra
        custom_data_root=None, # TODO: Needed for hydra
        semantic_only=None, # TODO: Needed for hydra
        ignore_stuff_in_offset=None, # TODO: Needed for hydra
        small_instance_area=None, # TODO: Needed for hydra
        small_instance_weight=None, # TODO: Needed for hydra
        enable_orig_coco_augmentation=None, # TODO: Needed for hydra
        enable_orig_coco_processing=None, # TODO: Needed for hydra
        single_return=None, # TODO: Needed for hydra
        merge_with_background=None, # TODO: Needed for hydra
        return_multiple_frames=None, # TODO: Needed for hydra
        **kwargs
    ):
        
        self.root = root / ("training" if self.split == Split.TRAIN else "validation")
        # self.root = root / "validation"
        self.specific_scenes = specific_scenes
        self.return_encoder_normalized_tgt = return_encoder_normalized_tgt
        self.src_eq_tgt = src_eq_tgt
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        
        self.scene_paths = [folder for folder in Path(self.root).iterdir() if folder.is_dir() and (folder / "metadata.json").exists()]
        self.scene_names = [scene.name for scene in self.scene_paths]
        self.metadata = [list(json.load((folder / 'metadata.json').open()).values()) for folder in self.scene_paths]
        print(f"Initial num pairs: {sum(len(task['start_end_ids']) for scene in self.metadata for task in scene)}")
        for i, scene in enumerate(self.metadata):
            for task in scene:
                frame_ids = set()
                for start, end in task['start_end_ids']:
                    frame_ids.add(start)
                    frame_ids.add(end)

                episode_numbers = sorted(int(e.split('_')[1]) for e in frame_ids)
                n = 40
                task['start_end_ids'] = [[f'episode_{i}', f'episode_{i + n}'] for i in episode_numbers if i + n in episode_numbers]

        print(f"Final num pairs: {sum(len(task['start_end_ids']) for scene in self.metadata for task in scene)} on {self.split.name.lower()}")

    def __len__(self) -> int:
        return len(self.metadata) * 10000

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for i in range(30):
            try:
                idx %= len(self.metadata)
                return self.get_paired_data(idx)
            except Exception as e:
                print(e)
                idx = np.random.randint(len(self.metadata))

        raise Exception("Failed to get data")
            
    def get_image(self, image_path: Path):
        image = np.asarray(Image.open(image_path))
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

        return image
    
    
    def get_paired_data(self, idx: int):
        metadata, (idx, task_id, pair_idx) = self.get_metadata(idx)

        scene_id = idx
        scene_name = self.scene_names[scene_id]
        src_img_name, tgt_img_name = self.metadata[idx][task_id]['start_end_ids'][pair_idx]
        instruction = self.metadata[idx][task_id]['instruction']

        def process_name(filename, padding_length=7):
            return f"{filename[:-len(str(int(filename.split('_')[-1])))]}{int(filename.split('_')[-1]):0{padding_length}d}"
        
        src_img_name = process_name(src_img_name)
        tgt_img_name = process_name(tgt_img_name)

        if self.src_eq_tgt:
            tgt_img_name = src_img_name
        
        src_path = self.root / scene_name / "image" / f"{src_img_name}.png"
        tgt_path = self.root / scene_name / "image" / f"{tgt_img_name}.png"

        def get_seg(src_seg_path_):
            src_seg = torch.from_numpy(im_to_numpy(Image.open(src_seg_path_)))
            src_seg = src_seg.float().unsqueeze(0).unsqueeze(0)
            src_seg[src_seg == 255] = -1
            return src_seg

        src_seg_path = self.root / scene_name / "segmentation" / f"{src_img_name}.png"
        src_seg = get_seg(src_seg_path)

        if self.src_eq_tgt:
            tgt_seg = src_seg.clone()
        else:
            tgt_seg_path = self.root / scene_name / "segmentation" / f"{tgt_img_name}.png"
            tgt_seg = get_seg(tgt_seg_path)

        src_img = self.get_image(src_path)
        if self.src_eq_tgt:
            tgt_img = src_img.clone()
        else:
            tgt_img = self.get_image(tgt_path)

        ret = {}
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=src_img.to(self.device), segmentation=src_seg.to(self.device)),
            tgt_data=Data(image=tgt_img.to(self.device), segmentation=tgt_seg.to(self.device)),
            use_keypoints=False, 
            return_encoder_normalized_tgt=self.return_encoder_normalized_tgt
        )

        if self.return_encoder_normalized_tgt:
            tgt_data, tgt_data_src_transform = tgt_data

        def process_data(data_: Data):
            data_.image = data_.image.squeeze(0)
            data_.segmentation = rearrange(data_.segmentation, "() c h w -> h w c")
            assert data_.segmentation.max() < 255
            data_.segmentation[data_.segmentation == -1] = 255
            data_.pad_mask = ~(data_.segmentation < 255).any(dim=-1)
            return data_

        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        if self.return_encoder_normalized_tgt:
            tgt_data_src_transform = process_data(tgt_data_src_transform)
            ret.update({
                "tgt_enc_norm_pixel_values": tgt_data_src_transform.image,
                "tgt_enc_norm_segmentation": tgt_data_src_transform.segmentation.to(torch.uint8),
                "tgt_enc_norm_valid": torch.full((255,), True, dtype=torch.bool),
            })
        
        ret.update({
            "tgt_pad_mask": tgt_data.pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation.to(torch.uint8),
            "src_pad_mask": src_data.pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation.to(torch.uint8),
            "input_ids": _get_tokens(self.tokenizer, instruction, max_length=24),
            "valid": torch.full((254,), True, dtype=torch.bool),
            "src_valid": torch.full((255,), True, dtype=torch.bool),
            "tgt_valid": torch.full((255,), True, dtype=torch.bool),
            **metadata
        })

        return ret
    
    def get_metadata(self, idx):
        scene_name = self.scene_names[idx]

        num_scene_tasks = len(self.metadata[idx])
        task_id = np.random.randint(num_scene_tasks)
        task_data = self.metadata[idx][task_id]
        num_task_pairs = len(task_data['start_end_ids'])

        if num_task_pairs == 0:
            return self.get_metadata((idx + 1) % len(self.metadata))
        
        pair_idx = np.random.randint(num_task_pairs)
        src_img_idx, tgt_img_idx = task_data['start_end_ids'][pair_idx]

        frame_name = f"{src_img_idx}-{tgt_img_idx}"
        frame_idxs = (0, 0)
        name = f"{scene_name}_{task_id}_{frame_name}"

        ret = {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "has_global_instance_ids": torch.tensor(True),
            "metadata": {
                "dataset": "calvin",
                "name": name,
                "scene_id": scene_name,
                "camera_trajectory": str(task_id), # Dummy value
                "camera_frame": frame_name,
                "frame_idxs": frame_idxs,
                "index": idx,
            },
        }

        return ret, (idx, task_id, pair_idx)

    def get_dataset(self):
        return self


import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    num_workers: int = 0,
    batch_size: int = 1,
    viz: bool = True,
    steps: Optional[int] = None,
    breakpoint_on_start: bool = False,
    return_tensorclass: bool = True,
):
    with breakpoint_on_error():
        from gen.datasets.utils import get_stable_diffusion_transforms
        from image_utils import library_ops
        augmentation=Augmentation(
            initial_resolution=512,
            enable_square_crop=True,
            center_crop=True,
            different_src_tgt_augmentation=False,
            enable_random_resize_crop=False,
            enable_horizontal_flip=False,
            tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
            enable_rand_augment=False,
            enable_rotate=False,
            reorder_segmentation=False,
            return_grid=False,
            src_transforms=get_stable_diffusion_transforms(resolution=256),
            tgt_transforms=get_stable_diffusion_transforms(resolution=256),
        )
        dataset = CalvinDataset(
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            tokenizer=MockTokenizer(),
            augmentation=augmentation,
            return_tensorclass=return_tensorclass,
            use_cuda=False,
        )

        subset_range = None
        dataloader = dataset.get_dataloader(pin_memory=False, subset_range=subset_range)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if breakpoint_on_start: breakpoint()
            if viz: visualize_input_data(batch, show_overlapping_masks=True, remove_invalid=False)
            if steps is not None and i >= steps - 1: break

if __name__ == "__main__":
    with breakpoint_on_error():
        app()