# Taken from : https://github.com/michaelnoi/scene_nvs/blob/main/scene_nvs/data/dataset.py
import autoroot

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from joblib import Memory
import lmdb
import msgpack
import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

from gen import SCANNETPP_CUSTOM_DATA_PATH, SCANNETPP_DATASET_PATH, SCRATCH_CACHE_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.run_dataloader import MockTokenizer
from gen.datasets.scannetpp.scene_data import test_scenes, train_scenes, val_scenes
from gen.utils.data_defs import integer_to_one_hot, one_hot_to_integer, visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, get_device, get_rank, hash_str_as_int, sanitize_filename, set_timing_builtins
from gen.utils.file_utils import get_available_path, sync_data
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import get_tokens
from image_utils.standalone_image_utils import integer_to_color, onehot_to_color

def coco_decode_rle(compressed_rle) -> np.ndarray:
    if isinstance(compressed_rle['counts'], str):
        compressed_rle['counts'] = compressed_rle['counts'].encode()

    binary_mask = mask_utils.decode(compressed_rle).astype(np.bool_)
    return binary_mask

def get_distance_matrix_vectorized(poses: np.ndarray) -> torch.Tensor:
    rotations = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    translations = poses[:, :3, 3]
    rotational_distances = 2 * np.arccos(np.clip(np.einsum('ij,kj->ik', rotations, rotations), -1.0, 1.0))
    translational_distances = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1)
    return torch.from_numpy(rotational_distances), torch.from_numpy(translational_distances)

def get_split_data(split, directory, distance_threshold, frames_slice) -> List[Dict[str, Union[str, torch.Tensor]]]:
    # Load data (Image + Camera Poses)
    image_folder = os.path.join(directory, "rgb")
    image_names = sorted(os.listdir(image_folder))[frames_slice]

    if len(image_names) == 0:
        log_warn(f"Skipping {directory} because no images found")
        return None

    try:
        with open(os.path.join(directory, "pose_intrinsic_imu.json")) as f:
            poses = json.load(f)
    except:
        log_warn(f"Skipping {directory} because loading pose JSON failed")
        return None

    frame_names = [frame for frame in poses.keys() if frame + ".jpg" in image_names]
    poses_c2w = np.stack([pose["aligned_pose"] for key, pose in poses.items() if key in frame_names])
    intrinsics = np.stack([pose["intrinsic"] for key, pose in poses.items() if key in frame_names])

    rotational_distances, translational_distances = get_distance_matrix_vectorized(poses_c2w)
    assert torch.isnan(rotational_distances).sum() == 0 and torch.isnan(translational_distances).sum() == 0

    mask = torch.logical_or(torch.logical_and(rotational_distances < distance_threshold[0], translational_distances < distance_threshold[1]), torch.logical_and(rotational_distances < distance_threshold[2], translational_distances < distance_threshold[3]))
    candidate_indicies = torch.argwhere(mask)

    return candidate_indicies, frame_names, poses_c2w, intrinsics

def get_scene_data(split, directory, distance_threshold, frames_slice) -> List[Dict[str, Union[str, torch.Tensor]]]:
    image_folder = os.path.join(directory, "rgb")
    output = get_split_data(split, directory, distance_threshold, frames_slice)

    if output is None: return None

    indices_arr, frame_names, poses_c2w, intrinsics = output

    assert len(frame_names) == len(poses_c2w) == len(intrinsics)
    
    image_files = [os.path.join(image_folder, frame_name + ".jpg") for frame_name in frame_names]

    return indices_arr, image_files, torch.from_numpy(poses_c2w), torch.from_numpy(intrinsics)

@inherit_parent_args
class ScannetppIphoneDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Path = SCANNETPP_DATASET_PATH,
        augmentation: Optional[Augmentation] = None,
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
        return_encoder_normalized_tgt: bool = False,
        only_preprocess_seg: bool = False,
        scratch_only: bool = False,
        src_eq_tgt: bool = False,
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
        self.root: str = root
        self.scenes = scenes
        self.scenes_slice = scenes_slice if scenes_slice is not None else slice(None)
        self.frames_slice = frames_slice if frames_slice is not None else slice(None)
        self.scenes_slice = self.scenes_slice if isinstance(self.scenes_slice, slice) else slice(*self.scenes_slice)
        self.frames_slice = self.frames_slice if isinstance(self.frames_slice, slice) else slice(*self.frames_slice)
        self.distance_threshold = distance_threshold
        self.depth_map = depth_map
        self.depth_map_type = depth_map_type
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.top_n_masks_only = top_n_masks_only
        self.image_pairs_per_scene = image_pairs_per_scene
        self.return_raw_dataset_image = return_raw_dataset_image
        self.num_overlapping_masks = num_overlapping_masks
        self.use_segmentation = use_segmentation and not self.return_raw_dataset_image
        self.return_encoder_normalized_tgt = return_encoder_normalized_tgt
        self.only_preprocess_seg = only_preprocess_seg
        self.scratch_only = scratch_only
        self.sync_dataset_to_scratch = sync_dataset_to_scratch
        self.src_eq_tgt = src_eq_tgt

        default_split_scenes: List[str] = train_scenes if self.split == Split.TRAIN else (val_scenes if self.split == Split.VALIDATION else test_scenes)
        self.scenes = default_split_scenes if (self.scenes is None or len(self.scenes) == 0) else self.scenes
        if self.scenes_slice is not None:
            self.scenes = self.scenes[self.scenes_slice]

        if single_scene_debug:
            self.scenes = ["712dc47104"]

        seg_data_path = SCANNETPP_CUSTOM_DATA_PATH / "all_pose_data"
        if self.sync_dataset_to_scratch:
            sync_data(
                nfs_path=SCANNETPP_DATASET_PATH, 
                sync=self.num_workers == 0,
                run_space_check=False,
                options=["--archive", "--progress", "--include=*/", "--include=rgb/***", "--exclude=*"]
            )
            sync_data(nfs_path=seg_data_path, sync=self.num_workers == 0)

        scenes_hash = hashlib.md5(("_".join(self.scenes)).encode()).hexdigest()
        cache_key = sanitize_filename(f"{scenes_hash}_{str(self.root).replace('/', '_')}_{distance_threshold}")
        cache_path = get_available_path(seg_data_path / f"{cache_key}.pt")

        if cache_path.exists():
            log_info(f"Loading all scene data from {cache_path}")
            self.dataset_idx_to_frame_pair, self.image_files, self.frame_poses, self.intrinsics, self.frame_scene_indices = torch.load(cache_path)
        else:
            log_info(f"Cache not found at {cache_path}, creating...")
            self.dataset_idx_to_frame_pair, self.image_files, self.frame_poses, self.intrinsics, self.frame_scene_indices = torch.zeros((0, 2), dtype=torch.int32), [], [], [], []
            for idx, scene in tqdm(enumerate(self.scenes), desc="Loading scenes", total=len(self.scenes)):
                output = get_scene_data(self.split, os.path.join(root, scene, "iphone"), self.distance_threshold, self.frames_slice)
                if output is None: continue

                scene_candidate_indices, scene_image_files, scene_poses_c2w, scene_intrinsics = output
                indices = torch.randint(0, scene_candidate_indices.size(0), (image_pairs_per_scene,))
                scene_candidate_indices = scene_candidate_indices[indices].to(torch.int32)

                assert 0 <= scene_candidate_indices.min() <= scene_candidate_indices.max() < len(scene_image_files)

                self.dataset_idx_to_frame_pair = torch.cat((self.dataset_idx_to_frame_pair, scene_candidate_indices + len(self.image_files)), dim=0)
                self.image_files.extend(scene_image_files)
                self.frame_poses.append(scene_poses_c2w)
                self.intrinsics.append(scene_intrinsics)
                self.frame_scene_indices.append(torch.full((len(scene_candidate_indices),), idx, dtype=torch.long))

            self.frame_poses = torch.cat(self.frame_poses, dim=0).to(torch.float32)
            self.intrinsics = torch.cat(self.intrinsics, dim=0).to(torch.float32)
            self.frame_scene_indices = torch.cat(self.frame_scene_indices, dim=0).to(torch.int32)
            log_info(f"Saving all scene data to {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((self.dataset_idx_to_frame_pair, self.image_files, self.frame_poses, self.intrinsics, self.frame_scene_indices), cache_path)

        log_info(f"Scannet++ has {len(self)} samples")

    def open_lmdb(self):
        if self.use_segmentation:
            seg_data_path = SCANNETPP_CUSTOM_DATA_PATH / "segmentation" / 'v2'
            if self.sync_dataset_to_scratch:
                sync_data(nfs_path=seg_data_path, sync=self.num_workers == 0)
            cache_path = get_available_path(seg_data_path, return_scratch_only=self.scratch_only)
            self.seg_env = lmdb.open(str(cache_path), readonly=True, map_size=1099511627776)
    
    def __len__(self) -> int:
        return len(self.image_files) if self.return_raw_dataset_image or self.only_preprocess_seg else len(self.dataset_idx_to_frame_pair)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'seg_env'):
            self.open_lmdb()
        if self.return_raw_dataset_image:
            return self.get_raw_data(idx)
        elif self.only_preprocess_seg:
            metadata = self.get_metadata(idx)
            src_path = self.image_files[idx]
            scene_id = metadata['metadata']['scene_id']
            src_key = f"{scene_id}_{Path(src_path).stem}"
            src_masks_msgpack = self.get_seg(src_key)
            return torch.zeros((1,))
        else:
            try:
                return self.get_paired_data(idx)
            except Exception as e:
                log_warn(f"Failed to load image {idx}: {e}")
                return self.__getitem__((idx + 1) % len(self))
            
    def get_image(self, image_path: Path):
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=self.scratch_only)
        
        with open(image_path, 'rb', buffering=100*1024) as fp: data = fp.read()
        image = simplejpeg.decode_jpeg(data)
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

        # target_file = torchvision.io.read_file(str(image_path))
        # image = torchvision.io.decode_jpeg(target_file, device=self.device)[None] / 255

        return image

    def get_raw_data(self, idx: int):
        return {
            "tgt_pixel_values": self.get_image(self.image_files[idx]),
            "tgt_segmentation": torch.zeros((5)),
            "src_pixel_values": torch.zeros((5)),
            "src_segmentation": torch.zeros((5)),
            "input_ids": torch.zeros((5)),
            **self.get_metadata(idx)
        }

    def get_seg(self, key):
        seg_data_path = SCANNETPP_CUSTOM_DATA_PATH / "segmentation" / 'npz_v0' / f"{key}.npz"
        cache_path = get_available_path(seg_data_path, return_scratch_only=self.scratch_only)
        try:
            if not cache_path.exists(): raise FileNotFoundError()
            seg = torch.from_numpy(np.load(cache_path)['arr_0'])
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                log_warn(f"Failed to load seg data from {cache_path}: {e}")
            with self.seg_env.begin() as txn:
                masks_msgpack = txn.get(key.encode('ascii'))
                if masks_msgpack is None:
                    return None
            _, masks = msgpack.unpackb(masks_msgpack, raw=False)
            seg = torch.from_numpy(np.stack([coco_decode_rle(mask['segmentation']) for mask in masks])).to(self.device)
            seg = one_hot_to_integer(seg.permute(1, 2, 0), self.num_overlapping_masks, assert_safe=False).permute(2, 0, 1)[None]
            np.savez_compressed(cache_path, seg.cpu().numpy())
            log_info(f"Saved seg data to {cache_path}")
        return seg
    
    def get_paired_data(self, idx: int):
        metadata = self.get_metadata(idx)

        src_img_idx, tgt_img_idx = metadata['metadata']['frame_idxs']

        if self.src_eq_tgt:
            tgt_img_idx = src_img_idx
        
        src_path = self.image_files[src_img_idx]
        tgt_path = self.image_files[tgt_img_idx]

        src_pose = self.frame_poses[src_img_idx]
        tgt_pose = self.frame_poses[tgt_img_idx]

        src_intrinsics = self.intrinsics[src_img_idx]
        tgt_intrinsics = self.intrinsics[tgt_img_idx]

        src_img = self.get_image(src_path)
        tgt_img = self.get_image(tgt_path)
            
        if self.use_segmentation:
            throw_error = False
            scene_id = metadata['metadata']['scene_id']

            src_key = f"{scene_id}_{Path(src_path).stem}"
            src_masks_msgpack = self.get_seg(src_key)

            tgt_key = f"{scene_id}_{Path(tgt_path).stem}"
            tgt_masks_msgpack = self.get_seg(tgt_key)

            if src_masks_msgpack is not None and tgt_masks_msgpack is not None:
                def process_seg(seg):
                    seg = F.interpolate(seg, size=(seg.shape[-2] * 2, seg.shape[-1] * 2), mode="nearest-exact")
                    seg = seg.long()
                    seg[seg == 255] = -1
                    seg = seg.float()
                    return seg

                src_seg = process_seg(src_masks_msgpack)
                tgt_seg = process_seg(tgt_masks_msgpack)
            else:
                throw_error = True

            if throw_error: raise KeyError(f"Key not found in LMDB: {src_key}")
            
        else:
            src_seg = src_img.new_zeros((1, 1, *src_img.shape[-2:]))
            tgt_seg = tgt_img.new_zeros((1, 1, *tgt_img.shape[-2:]))

        src_pose[:3, 3] /= 10
        tgt_pose[:3, 3] /= 10

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
            data_.segmentation[data_.segmentation >= self.top_n_masks_only] = 0
            assert data_.segmentation.max() < 255
            data_.segmentation[data_.segmentation == -1] = 255
            data_.pad_mask = ~(data_.segmentation < 255).any(dim=-1)
            return data_

        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        pixels = src_data.segmentation.long().contiguous().view(-1)
        pixels = pixels[(pixels < 255) & (pixels >= 0)]
        src_bincount = torch.bincount(pixels, minlength=self.top_n_masks_only + 1)
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
            "src_pose": src_pose,
            "tgt_pose": tgt_pose,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid[..., 1:],
            **metadata
        })

        return ret
    
    def get_metadata(self, idx):
        if self.return_raw_dataset_image or self.only_preprocess_seg:
            frame_path = Path(self.image_files[idx])
            frame_name = frame_path.stem
            scene_name = frame_path.parent.parent.parent.stem
            frame_idxs = (idx,)
        else:
            src_img_idx, tgt_img_idx = self.dataset_idx_to_frame_pair[idx, 0], self.dataset_idx_to_frame_pair[idx, 1]
            first_frame_path = Path(self.image_files[src_img_idx])
            second_frame_path = Path(self.image_files[tgt_img_idx])
            frame_name = f"{first_frame_path.stem}-{second_frame_path.stem}"
            scene_name = first_frame_path.parent.parent.parent.stem
            frame_idxs = (src_img_idx.item(), tgt_img_idx.item())

        name = f"{scene_name}_{frame_name}"
        return {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "has_global_instance_ids": torch.tensor(False),
            "metadata": {
                "dataset": "scannetpp",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": idx,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
            },
        }

    def cartesian_to_spherical(self, xyz: np.ndarray) -> np.ndarray:
        # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

        # ptsnew = np.hstack((xyz, np.zeros(xyz.shape))) #what is this for?
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        # for elevation angle defined from Z-axis down
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
        # # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT: np.ndarray, cond_RT: np.ndarray) -> torch.Tensor:
        # https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L318

        R, T = target_RT[:3, :3], target_RT[:3, -1]  # double check this
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:3, -1]  # double check this
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor(
            [
                d_theta.item(),
                math.sin(d_azimuth.item()),
                math.cos(d_azimuth.item()),
                d_z.item(),
            ]
        )

        return d_T
    

    def get_dataset(self):
        return self


import typer
typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    num_workers: int = 0,
    batch_size: int = 32,
    scenes: Optional[list[str]] = None,
    viz: bool = False,
    steps: int = 30,
    sync_dataset_to_scratch: bool = False,
    return_raw_dataset_image: bool = False,
    breakpoint_on_start: bool = False,
    use_segmentation: bool = True,
    single_scene_debug: bool = False,
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
        dataset = ScannetppIphoneDataset(
            shuffle=False,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            tokenizer=MockTokenizer(),
            augmentation=augmentation,
            return_tensorclass=False,
            scenes=scenes,
            sync_dataset_to_scratch=sync_dataset_to_scratch,
            return_raw_dataset_image=return_raw_dataset_image,
            scenes_slice=slice(0, None, 4),
            frames_slice=slice(0, None, 5),
            top_n_masks_only=128,
            num_overlapping_masks=6,
            use_segmentation=use_segmentation,
            single_scene_debug=single_scene_debug,
            use_cuda=False,
            only_preprocess_seg=False,
        )
        dataloader = dataset.get_dataloader(pin_memory=False)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if breakpoint_on_start: breakpoint()
            if viz: visualize_input_data(batch, show_overlapping_masks=True)
            if i >= steps - 1: break

if __name__ == "__main__":
    with breakpoint_on_error():
        app()