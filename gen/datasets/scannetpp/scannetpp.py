# Taken from : https://github.com/michaelnoi/scene_nvs/blob/main/scene_nvs/data/dataset.py
import autoroot

import json
import math
import multiprocessing
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from gen import SCANNETPP_CUSTOM_DATA_PATH, SCANNETPP_DATASET_PATH, SCRATCH_CACHE_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.run_dataloader import MockTokenizer
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, get_device, get_rank, hash_str_as_int
from gen.utils.file_utils import get_available_path, sync_data
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import get_tokens
from gen.datasets.scannetpp.scene_data import train_scenes, val_scenes, test_scenes

def get_distance_matrix_vectorized(poses: np.ndarray) -> torch.Tensor:
    rotations = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    translations = poses[:, :3, 3]
    
    rotational_distances = 2 * np.arccos(np.clip(np.einsum('ij,kj->ik', rotations, rotations), -1.0, 1.0))
    translational_distances = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1)
    
    distance_matrix = np.stack((rotational_distances, translational_distances), axis=-1)
    distance_matrix = np.sqrt(np.sum(distance_matrix**2, axis=-1))
    
    return torch.from_numpy(distance_matrix)

def get_split_data(split, directory, distance_threshold) -> List[Dict[str, Union[str, torch.Tensor]]]:
    scene_pose_data_path = SCANNETPP_CUSTOM_DATA_PATH / "scene_pose_data"
    cache_key = f"{split.name}_{directory.replace('/', '_')}_{distance_threshold}"
    cache_path = get_available_path(scene_pose_data_path / f"{cache_key}.pt")

    if cache_path.exists():
        return torch.load(cache_path)

    # Load data (Image + Camera Poses)
    image_folder = os.path.join(directory, "rgb")
    image_names = sorted(os.listdir(image_folder))

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
    poses_c2w = np.stack([pose["aligned_pose"] for _, pose in poses.items()])
    K = np.stack([pose["intrinsic"] for _, pose in poses.items()])

    distance_matrix = get_distance_matrix_vectorized(poses_c2w).to(torch.float64)
    maximum = torch.max(distance_matrix[~torch.isnan(distance_matrix)]) # get max
    distance_matrix = distance_matrix / maximum # scale to 0-1
    mask = torch.logical_and(distance_matrix > 0, distance_matrix <= distance_threshold)

    candidate_indicies = torch.argwhere(mask)        
    candidate_viewpoint_metric = distance_matrix[mask] # get corresponding viewpoint metric for the candidate indicies

    bins = np.linspace(0, distance_threshold, 5) # bin the viewpoint metric into 10 bins to stratify the data
    binned_viewpoint_metric = np.digitize(candidate_viewpoint_metric, bins)
    learn, test = train_test_split(
        candidate_indicies,
        test_size=0.2,
        stratify=binned_viewpoint_metric,
        random_state=42,
    )

    # shape of splits: (n, 2)
    train, val = train_test_split(
        learn,
        test_size=0.2,
        stratify=binned_viewpoint_metric[learn],
        random_state=42,
    )

    indices_arr = (train if split == Split.TRAIN else (val if split == Split.VALIDATION else test))
    pose_data = np.concatenate((poses_c2w.reshape(poses_c2w.shape[0], -1), K.reshape(K.shape[0], -1)), axis=1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((indices_arr, frame_names, pose_data), cache_path)
    log_info(f"Saved data to {cache_path}")

    return indices_arr, frame_names, pose_data

def get_scene_data(split, directory, distance_threshold, num_instances) -> List[Dict[str, Union[str, torch.Tensor]]]:
    image_folder = os.path.join(directory, "rgb")

    output = get_split_data(split, directory, distance_threshold)
    if output is None: return None

    indices_arr, frame_names, pose_data = output
    poses_c2w, intrinsics = pose_data[:, :16].reshape(-1, 4, 4), pose_data[:, 16:].reshape(-1, 3, 3)

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
        distance_threshold: float = 0.1,
        depth_map_type: str = "gt",
        depth_map: bool = False,
        scenes_slice: Optional[slice] = None,
        num_processes: int = 1,
        top_n_masks_only: int = 34,
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
        num_overlapping_masks=None, # TODO: Needed for hydra
        **kwargs
    ):
        self.root: str = root
        self.scenes: List[str] = train_scenes if self.split == Split.TRAIN else (val_scenes if self.split == Split.VALIDATION else test_scenes)
        self.scenes_slice = scenes_slice
        self.distance_threshold = distance_threshold
        self.depth_map = depth_map
        self.depth_map_type = depth_map_type
        self.num_processes = num_processes
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.top_n_masks_only = top_n_masks_only
        self.image_pairs_per_scene = image_pairs_per_scene
        
        if scenes_slice is not None:
            self.scenes = self.scenes[scenes_slice]

        sync_data(nfs_path=SCANNETPP_CUSTOM_DATA_PATH, sync=self.num_workers == 0)

        scene_pose_data_path = SCANNETPP_CUSTOM_DATA_PATH / "scene_pose_data"
        cache_key = f"{self.split.name}_{str(self.root).replace('/', '_')}_{distance_threshold}"
        cache_path = get_available_path(scene_pose_data_path / f"{cache_key}.pt")

        if cache_path.exists():
            log_info(f"Loading all scene data from {cache_path}")
            self.frame_indices, self.image_files, self.poses, self.intrinsics, self.scene_idx = torch.load(cache_path)
        else:
            log_info(f"Cache not found at {cache_path}, creating...")
            self.frame_indices, self.image_files, self.poses, self.intrinsics, self.scene_idx = torch.zeros((0, 2), dtype=torch.int32), [], [], [], []
            for idx, scene in tqdm(enumerate(self.scenes), desc="Loading scenes"):
                output = get_scene_data(self.split, os.path.join(root, scene, "iphone"), self.distance_threshold, image_pairs_per_scene)
                if output is None: continue

                indices_arr_, image_files_, poses_c2w_, K_ = output

                indices = torch.randint(0, indices_arr_.size(0), (image_pairs_per_scene,))
                indices_arr_ = indices_arr_[indices].to(torch.int32)

                self.frame_indices = torch.cat((self.frame_indices, indices_arr_ + len(self.image_files)), dim=0)
                self.image_files.extend(image_files_)
                self.poses.append(poses_c2w_)
                self.intrinsics.append(K_)
                self.scene_idx.append(torch.full((len(indices_arr_),), idx, dtype=torch.long))

            self.poses = torch.cat(self.poses, dim=0).to(torch.float32)
            self.intrinsics = torch.cat(self.intrinsics, dim=0).to(torch.float32)
            self.scene_idx = torch.cat(self.scene_idx, dim=0).to(torch.int32)
            log_info(f"Saving all scene data to {cache_path}")
            torch.save((self.frame_indices, self.image_files, self.poses, self.intrinsics, self.scene_idx), cache_path)
    
    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_dict = {
            "path_cond": self.image_files[self.frame_indices[idx, 0]],
            "path_target": self.image_files[self.frame_indices[idx, 1]],
            "pose_cond": self.poses[self.frame_indices[idx, 0]],
            "pose_target": self.poses[self.frame_indices[idx, 1]],
            "K_cond": self.intrinsics[self.frame_indices[idx, 0]],
            "K_target": self.intrinsics[self.frame_indices[idx, 1]],

        }

        image_cond = torchvision.io.decode_jpeg(torchvision.io.read_file(data_dict["path_cond"]))[None] / 255
        image_target = torchvision.io.decode_jpeg(torchvision.io.read_file(data_dict["path_target"]))[None] / 255
        
        T = self.get_relative_pose(data_dict["pose_target"], data_dict["pose_cond"])  # shape [7]

        src_pose, tgt_pose = data_dict["pose_cond"], data_dict["pose_target"]

        scene_extent = self.poses[self.frame_indices[:, 0][(self.scene_idx == self.scene_idx[self.frame_indices[idx, 0]])]][:, :3, 3].max()
        src_pose[:3, 3] /= scene_extent
        tgt_pose[:3, 3] /= scene_extent

        ret = {}
        if self.depth_map:
            depth_map_path = (data_dict["path_target"].replace("rgb", "depth").replace("jpg", "png"))
            depth_map = Image.open(depth_map_path)
            h, w = depth_map.size
            depth_map = torchvision.transforms.CenterCrop(min(h, w))(depth_map) # ensure that the depth image corresponds to the target image
            depth_map = torchvision.transforms.ToTensor()(depth_map)
            ret["depth_map"] = depth_map.float()

        seg = image_cond.new_zeros((1, 1, *image_cond.shape[-2:]))
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=image_cond, segmentation=seg),
            tgt_data=Data(image=image_target, segmentation=seg),
        )

        src_data.image = src_data.image.squeeze(0)
        src_data.segmentation = rearrange(src_data.segmentation, "() c h w -> h w c")
        tgt_data.image = tgt_data.image.squeeze(0)
        tgt_data.segmentation = rearrange(tgt_data.segmentation, "() c h w -> h w c")

        src_data.segmentation[src_data.segmentation >= self.top_n_masks_only] = 0
        tgt_data.segmentation[tgt_data.segmentation >= self.top_n_masks_only] = 0

        assert tgt_data.segmentation.max() < 255 and src_data.segmentation.max() < 255
        src_data.segmentation[src_data.segmentation == -1] = 255
        tgt_data.segmentation[tgt_data.segmentation == -1] = 255
        src_pad_mask = ~(src_data.segmentation < 255).any(dim=-1)
        tgt_pad_mask = ~(tgt_data.segmentation < 255).any(dim=-1)

        pixels = src_data.segmentation.long().contiguous().view(-1)
        pixels = pixels[(pixels < 255) & (pixels >= 0)]
        src_bincount = torch.bincount(pixels, minlength=self.top_n_masks_only + 1)
        valid = src_bincount > 0

        # We convert to uint8 to save memory.
        src_data.segmentation = src_data.segmentation.to(torch.uint8)
        tgt_data.segmentation = tgt_data.segmentation.to(torch.uint8)

        name = str(idx)
        ret.update({
            "tgt_pad_mask": tgt_pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation,
            "src_pad_mask": src_pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation,
            "src_pose": src_pose,
            "tgt_pose": tgt_pose,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid[..., 1:],
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "name": name,
                # "scene_id": metadata[0],
                # "camera_trajectory": metadata[1],
                # "camera_frame": metadata[2],
                "index": idx,
            },
        })

        return ret

    def read_depth_map(self, depth_map_path: str) -> torch.Tensor:
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH) # make sure to read the image as 16 bit
        depth_map = depth_map.astype(np.int16) # convert to int16, hacky, but depth shouldn't exceed 32.767 m

        return torch.from_numpy(depth_map).unsqueeze(0).float()

    def _truncate_data(self, n: int) -> None: # truncate the data to n points (for debugging)
        self.data = self.data[:n]
        log_info("Truncated data to length: " + str(self.__len__()))

    def get_rotational_difference(
        self, rotation_1: Rotation, rotation_2: Rotation
    ) -> np.ndarray:
        # https://stackoverflow.com/questions/22157435/difference-between-the-two-quaternions
        # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=Using%20quaternions%C2%B6&text=The%20difference%20rotation%20quaternion%20that,quaternion%20r%20%3D%20p%20q%20%E2%88%97%20.

        return rotation_2.as_quat() * rotation_1.inv().as_quat()

    def get_translational_difference(
        self, translation_1: np.ndarray, translation_2: np.ndarray
    ) -> np.ndarray:
        return translation_1 - translation_2

    def get_relative_pose(self, pose_1: np.ndarray, pose_2: np.ndarray) -> np.ndarray:
        rotation_1 = Rotation.from_matrix(pose_1[:3, :3])
        rotation_2 = Rotation.from_matrix(pose_2[:3, :3])
        translation_1 = pose_1[:3, 3]
        translation_2 = pose_2[:3, 3]

        return np.concatenate(
            [
                self.get_rotational_difference(rotation_1, rotation_2),
                self.get_translational_difference(translation_1, translation_2),
            ]
        )

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

if __name__ == "__main__":
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
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=0,
            batch_size=32,
            tokenizer=MockTokenizer(),
            augmentation=augmentation,
            return_tensorclass=True,
            num_processes=1,
        )
        dataloader = dataset.get_dataloader(pin_memory=False)
        import time
        start_time = time.time()
        for i, batch in tqdm(enumerate(dataloader)):
            end_time = time.time()
            print(f"Time per batch: {end_time - start_time}")
            start_time = end_time
            # visualize_input_data(batch, show_overlapping_masks=True)
            exit()