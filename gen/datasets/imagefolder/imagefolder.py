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

from gen import GLOBAL_CACHE_PATH, SCANNETPP_CUSTOM_DATA_PATH, SCANNETPP_DATASET_PATH
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
from gen.utils.tokenization_utils import get_tokens
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
class ImagefolderDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Path = SCANNETPP_DATASET_PATH,
        augmentation: Optional[Augmentation] = None,
        return_encoder_normalized_tgt: bool = False,
        src_eq_tgt: bool = False,
        tokenizer = None,
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
        # TODO: All these params are not actually used but needed because of a quick with hydra_zen
        num_objects=None,
        resolution=None,
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
        **kwargs
    ):
        
        self.root = root
        from datetime import datetime
        current_datetime = datetime.now()
        image_files, saved_data = get_to_process(current_datetime, self.root, save_type_names, return_raw_data=True)
        self.saved_scene_frames = defaultdict(set)
        for scene_id, frame_id in saved_data:
            self.saved_scene_frames[scene_id].add(frame_id)

        self.return_encoder_normalized_tgt = return_encoder_normalized_tgt
        self.src_eq_tgt = src_eq_tgt
        self.augmentation = augmentation
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.saved_scene_frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            return self.get_paired_data(idx)
        except Exception as e:
            log_warn(f"Failed to load image {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
            
    def get_image(self, image_path: Path):
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=False)
        
        with open(image_path, 'rb', buffering=100*1024) as fp: data = fp.read()
        image = simplejpeg.decode_jpeg(data)
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

        return image
    
    def get_raw_image(self, image_path: Path):
        import torchvision
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=False)

        target_file = torchvision.io.read_file(str(image_path))
        image = torchvision.io.decode_jpeg(target_file, device=self.device)

        return image
    
    def get_paired_data(self, idx: int):
        metadata = self.get_metadata(idx)

        scene_id = metadata['metadata']['scene_id']
        src_img_idx, tgt_img_idx = metadata['metadata']['frame_idxs']

        if self.src_eq_tgt:
            tgt_img_idx = src_img_idx
        
        src_path = self.root / save_type_names[0] / scene_id / f"{src_img_idx}.jpg"
        tgt_path = self.root / save_type_names[0] / scene_id / f"{tgt_img_idx}.jpg"

        def get_seg(src_seg_path_):
            src_seg = torch.from_numpy(im_to_numpy(Image.open(src_seg_path_)))
            src_seg = src_seg.float().unsqueeze(0).unsqueeze(0)
            src_seg[src_seg == 255] = -1
            return src_seg

        src_seg_path = self.root / save_type_names[2] / scene_id / f"{src_img_idx}.png"
        src_seg = get_seg(src_seg_path)

        if self.src_eq_tgt:
            tgt_seg = src_seg.clone()
        else:
            tgt_seg_path = self.root / save_type_names[2] / scene_id / f"{tgt_img_idx}.png"
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
            data_.segmentation[data_.segmentation >= 8] = 255
            data_.pad_mask = ~(data_.segmentation < 255).any(dim=-1)
            return data_

        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        pixels = src_data.segmentation.long().contiguous().view(-1)
        pixels = pixels[(pixels < 255) & (pixels >= 0)]
        src_bincount = torch.bincount(pixels, minlength=256)
        valid = src_bincount > 0

        if self.return_encoder_normalized_tgt:
            tgt_data_src_transform = process_data(tgt_data_src_transform)
            ret.update({
                "tgt_enc_norm_pixel_values": tgt_data_src_transform.image,
                "tgt_enc_norm_segmentation": tgt_data_src_transform.segmentation.to(torch.uint8),
            })
        
        ret.update({
            "tgt_pad_mask": tgt_data.pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation.to(torch.uint8),
            "src_pad_mask": src_data.pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation.to(torch.uint8),
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid[..., 1:],
            **metadata
        })

        return ret
    
    def get_metadata(self, idx):
        scene_name = list(self.saved_scene_frames.keys())[idx]
        allowed_frames = self.saved_scene_frames[scene_name]

        src_img_idx, tgt_img_idx = np.random.choice(list(allowed_frames), size=2, replace=False)
        frame_name = f"{src_img_idx}-{tgt_img_idx}"
        frame_idxs = (src_img_idx, tgt_img_idx)

        name = f"{scene_name}_{frame_name}"
        return {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "has_global_instance_ids": torch.tensor(False),
            "metadata": {
                "dataset": "imagefolder",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": idx,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
            },
        }

    def get_dataset(self):
        return self


import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    root: Path,
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
            enable_random_resize_crop=True,
            enable_horizontal_flip=False,
            tgt_random_scale_ratio=((1.0, 1.0), (1.0, 1.0)),
            enable_rand_augment=False,
            enable_rotate=False,
            reorder_segmentation=False,
            return_grid=False,
            src_transforms=get_stable_diffusion_transforms(resolution=512),
            tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        )
        dataset = ImagefolderDataset(
            root=root,
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            tokenizer=MockTokenizer(),
            augmentation=augmentation,
            return_tensorclass=return_tensorclass,
            top_n_masks_only=128,
            num_overlapping_masks=1,
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