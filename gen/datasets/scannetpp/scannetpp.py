# Taken from : https://github.com/michaelnoi/scene_nvs/blob/main/scene_nvs/data/dataset.py

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

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
from gen.datasets.run_dataloader import MockTokenizer
from gen.datasets.scannetpp.colmap import read_cameras_text, read_images_text
from gen.datasets.scannetpp.run_sam_dask import get_to_process
from gen.datasets.scannetpp.scene_data import test_scenes, train_scenes, val_scenes
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int, sanitize_filename, to_numpy
from gen.utils.file_utils import get_available_path, sync_data
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import get_tokens

def get_distance_matrix_vectorized(poses: np.ndarray) -> torch.Tensor:
    poses = to_numpy(poses)
    rotations = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    translations = poses[:, :3, 3]
    rotational_distances = 2 * np.arccos(np.clip(np.einsum('ij,kj->ik', rotations, rotations), -1.0, 1.0))
    translational_distances = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1)
    return torch.from_numpy(rotational_distances), torch.from_numpy(translational_distances)

memory = Memory(GLOBAL_CACHE_PATH, verbose=0)

@memory.cache()
def get_scene_data(
        directory,
        distance_threshold,
        frames_slice,
        valid_scene_frames=None
) -> List[Dict[str, Union[str, torch.Tensor]]]:
    image_folder = os.path.join(directory, "rgb")
    image_names = set(os.listdir(image_folder)[frames_slice])
    image_names = set(filter(lambda x: x.removesuffix('.jpg') in valid_scene_frames, image_names)) if valid_scene_frames is not None else image_names

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

    assert len(frame_names) == len(poses_c2w) == len(intrinsics)
    
    image_files = [os.path.join(image_folder, frame_name + ".jpg") for frame_name in frame_names]

    return candidate_indicies, image_files, torch.from_numpy(poses_c2w), torch.from_numpy(intrinsics)

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

def get_instance_scene_data(
        _metadata,
        directory,
        distance_threshold,
        use_colmap_poses,
        valid_scene_frames=None,
) -> List[Dict[str, Union[str, torch.Tensor]]]:
    
    scene_id = Path(directory).parent.stem
    frames = _metadata[scene_id]
    
    image_folder = Path(os.path.join(directory, "rgb"))
    image_files = [image_folder / f"{frame}" for frame in frames]

    if len(image_files) == 0:
        log_warn(f"Skipping {directory} because no images found")
        return None

    try:
        with open(os.path.join(directory, "pose_intrinsic_imu.json")) as f:
            poses = json.load(f)
    except:
        log_warn(f"Skipping {directory} because loading pose JSON failed")
        return None
    
    image_data = read_images_text(Path(directory) / "colmap" / "images.txt")
    camera_data = read_cameras_text(Path(directory) / "colmap" / "cameras.txt")

    poses_c2w = np.stack([v.world_to_camera for i, (k,v) in enumerate(image_data.items()) if v.name == frames[i]])
    assert len(poses_c2w) == len(frames)

    intrinsics = np.stack([poses[frame_name.removesuffix('.jpg')]["intrinsic"] for frame_name in frames])
    # poses_c2w = np.stack([poses[frame_name.removesuffix('.jpg')]["aligned_pose"] for frame_name in frames])

    _camera_frustrum = torch.from_numpy(_metadata[f"{scene_id}_camera_frustrum"])
    _instance_pixel_overlap = torch.from_numpy(_metadata[f"{scene_id}_instance_pixel_overlap"])
    _num_overapping_instances = torch.from_numpy(_metadata[f"{scene_id}_num_overapping_instances"])

    mask = _camera_frustrum > distance_threshold[0]
    
    if _num_overapping_instances.sum() <= 1:
        log_error(f"No overlapping instances found on {scene_id}.")
        return None
        mask = _camera_frustrum > 0.75
    else:
        mask = torch.logical_and(
            _camera_frustrum > distance_threshold[0],
            torch.logical_and(
                _instance_pixel_overlap >= distance_threshold[1],
                _num_overapping_instances >= 
                (distance_threshold[2] * _num_overapping_instances[torch.arange(_num_overapping_instances.shape[0]), torch.arange(_num_overapping_instances.shape[0])]) 
            )
        )
    
    candidate_indicies = torch.argwhere(mask)
    return candidate_indicies, image_files, torch.from_numpy(poses_c2w), torch.from_numpy(intrinsics)


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
        single_scene_debug: Optional[str] = None,
        use_segmentation: bool = True,
        return_encoder_normalized_tgt: bool = False,
        scratch_only: bool = False,
        src_eq_tgt: bool = False,
        image_files: Optional[list] = None,
        no_filtering: bool = False,
        dummy_mask: bool = False,
        merge_masks: bool = False,
        return_only_instance_seg: bool = False,
        allow_instance_seg: bool = False,
        use_colmap_poses: bool = False,
        # TODO: All these params are not actually used but needed because of a quick with hydra_zen
        only_preprocess_seg: bool = False,
        use_new_seg: bool = False,
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
        self.scratch_only = scratch_only
        self.sync_dataset_to_scratch = sync_dataset_to_scratch
        self.src_eq_tgt = src_eq_tgt
        self.image_files = image_files
        self.no_filtering = no_filtering
        self.dummy_mask = dummy_mask
        self.merge_masks = merge_masks
        self.return_only_instance_seg = return_only_instance_seg
        self.allow_instance_seg = allow_instance_seg
        self.use_colmap_poses = use_colmap_poses

        if self.return_raw_dataset_image and self.image_files is not None:
            return 
        
        if self.use_colmap_poses: assert self.return_only_instance_seg
        if self.return_only_instance_seg: assert self.allow_instance_seg

        default_split_scenes: List[str] = train_scenes if self.split == Split.TRAIN else (val_scenes if self.split == Split.VALIDATION else test_scenes)
        self.scenes = default_split_scenes if (self.scenes is None or len(self.scenes) == 0) else self.scenes
        if self.scenes_slice is not None:
            self.scenes = self.scenes[self.scenes_slice]

        saved_data = None
        saved_scene_frames = None
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y_%m_%d_00_00_00')

        save_type_names = ["rgb", "masks"] + (["instance_seg_v1"] if self.return_only_instance_seg else ["seg_v1"])
        image_files, saved_data = get_to_process(formatted_datetime, save_type_names, return_raw_data=True)

        gt_scene_count = defaultdict(int)
        for scene_id, frame_id in image_files:
            gt_scene_count[scene_id] += 1
        
        saved_scene_count = defaultdict(int)
        saved_scene_frames = defaultdict(set)
        for scene_id, frame_id in saved_data:
            saved_scene_count[scene_id] += 1
            saved_scene_frames[scene_id].add(frame_id)

        self.new_scenes = []
        for scene_id in saved_scene_count.keys():
            if scene_id not in self.scenes: continue
            if saved_scene_count[scene_id] >= int(gt_scene_count[scene_id] * 1/100):
                self.new_scenes.append(scene_id)
            else:
                print(f"Skipping {scene_id} because {saved_scene_count[scene_id]} out of {gt_scene_count[scene_id]} frames saved")

        log_info(f"Using {len(self.new_scenes)} scenes out of {len(self.scenes)}")
        self.scenes = sorted(self.new_scenes)

        if single_scene_debug is not None:
            self.scenes = [single_scene_debug]

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
        cache_key = sanitize_filename(f"{scenes_hash}_{str(self.root).replace('/', '_')}_{distance_threshold}_{use_colmap_poses}")
        cache_path = get_available_path(seg_data_path / f"{cache_key}.pt")

        if self.return_only_instance_seg:
            _metadata = np.load(SCANNETPP_CUSTOM_DATA_PATH /  "data_v1" / "overlap_data.npz")

        if cache_path.exists():
            log_info(f"Loading all scene data from {cache_path}")
            self.dataset_idx_to_frame_pair, self.image_files, self.frame_poses, self.intrinsics, self.frame_scene_indices = torch.load(cache_path)
        else:
            log_info(f"Cache not found at {cache_path}, creating...")
            self.dataset_idx_to_frame_pair, self.image_files, self.frame_poses, self.intrinsics, self.frame_scene_indices = torch.zeros((0, 2), dtype=torch.int32), [], [], [], []
            for idx, scene_id in tqdm(enumerate(self.scenes), desc="Loading scenes", total=len(self.scenes)):
                _iphone_path = os.path.join(root, scene_id, "iphone")
                valid_scene_frames = None
                if saved_scene_frames is not None:
                    assert len(saved_scene_frames[scene_id]) > 0
                    valid_scene_frames = saved_scene_frames[scene_id]

                if self.return_only_instance_seg:
                    if scene_id not in _metadata: continue
                    output = get_instance_scene_data(_metadata, _iphone_path, self.distance_threshold, self.use_colmap_poses, valid_scene_frames)
                else:
                    output = get_scene_data(_iphone_path, self.distance_threshold, self.frames_slice, valid_scene_frames)

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

        log_info(f"Scannet++ has {len(self)} samples with {len(self.scenes)} scenes and {len(self.image_files)} images")

    def __len__(self) -> int:
        return len(self.image_files) if (self.return_raw_dataset_image or self.src_eq_tgt) else len(self.dataset_idx_to_frame_pair)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.return_raw_dataset_image:
            return self.get_raw_data(idx)
        else:
            for _ in range(60):
                try:
                    return self.get_paired_data(idx)
                except Exception as e:
                    # log_info(f"Failed to load image {idx}: {e}")
                    idx = torch.randint(0, len(self), (1,)).item()

            raise Exception(f"Failed to load image {idx}")
            
    def get_image(self, image_path: Path):
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=self.scratch_only)
        
        with open(image_path, 'rb', buffering=100*1024) as fp: data = fp.read()
        image = simplejpeg.decode_jpeg(data)
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)

        return image
    
    def get_raw_image(self, image_path: Path):
        import torchvision
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=self.scratch_only)

        target_file = torchvision.io.read_file(str(image_path))
        image = torchvision.io.decode_jpeg(target_file, device=self.device)

        return image

    def get_raw_data(self, idx: int):
        return {
            "tgt_pixel_values": self.get_raw_image(self.image_files[idx]),
            "tgt_segmentation": torch.zeros((5)),
            "src_pixel_values": torch.zeros((5)),
            "src_segmentation": torch.zeros((5)),
            "input_ids": torch.zeros((5)),
            "tgt_path": str(self.image_files[idx]),
            **self.get_metadata(idx)
        }
    
    def open_seg(self, sam_path: Path):
        if "instance" in sam_path.parent.parent.stem:
            src_seg = torch.from_numpy(im_to_numpy(Image.open(sam_path)))
            src_seg[src_seg == 65535] = -1
            assert src_seg.max().item() < 255
            src_seg = src_seg.float().unsqueeze(0).unsqueeze(0)
            src_seg = F.interpolate(src_seg, size=(720, 960), mode="nearest")
        else:
            if self.return_only_instance_seg:
                raise Exception()
            
            src_seg = torch.from_numpy(im_to_numpy(Image.open(sam_path)))
            src_seg = src_seg.float().unsqueeze(0).unsqueeze(0)
            src_seg[src_seg == 255] = -1
        return src_seg

    
    def get_paired_data(self, idx: int):
        if self.src_eq_tgt:
            idx = torch.randint(0, len(self.dataset_idx_to_frame_pair), (1,)).item()
            
        metadata = self.get_metadata(idx)

        src_img_idx, tgt_img_idx = metadata['metadata']['frame_idxs']

        if self.src_eq_tgt:
            tgt_img_idx = src_img_idx
        
        src_rgb_path = Path(self.image_files[src_img_idx])
        tgt_rgb_path = Path(self.image_files[tgt_img_idx])

        src_pose = self.frame_poses[src_img_idx]
        tgt_pose = self.frame_poses[tgt_img_idx]

        src_intrinsics = self.intrinsics[src_img_idx]
        tgt_intrinsics = self.intrinsics[tgt_img_idx]

        scene_id = metadata['metadata']['scene_id']

        save_data_path = SCANNETPP_CUSTOM_DATA_PATH / "data_v1"
        src_rgb_path = save_data_path / "rgb" / scene_id / f"{src_rgb_path.stem}.jpg"
        tgt_rgb_path = save_data_path / "rgb" / scene_id / f"{tgt_rgb_path.stem}.jpg"
        
        if self.dummy_mask:
            src_seg = torch.zeros((1, 1, 960, 720))
            tgt_seg = torch.zeros((1, 1, 960, 720))
        else:
            src_seg_path = save_data_path / "instance_seg_v1" / scene_id / f"{src_rgb_path.stem}.png"
            if not src_seg_path.exists():
                src_seg_path = save_data_path / "seg_v1" / scene_id / f"{src_rgb_path.stem}.png"
            
            src_seg = self.open_seg(src_seg_path)

            if self.src_eq_tgt:
                tgt_seg = src_seg.clone()
            else:
                tgt_seg_path = save_data_path / "instance_seg_v1" / scene_id / f"{tgt_rgb_path.stem}.png"
                if not tgt_seg_path.exists():
                    tgt_seg_path = save_data_path / "seg_v1" / scene_id / f"{tgt_rgb_path.stem}.png"
                tgt_seg = self.open_seg(tgt_seg_path)

        src_img = self.get_image(src_rgb_path)
        if self.src_eq_tgt:
            tgt_img = src_img.clone()
        else:
            tgt_img = self.get_image(tgt_rgb_path)

        # if scene_id != "5fb5d2dbf2":
        #     from image_utils.standalone_image_utils import integer_to_color
        #     from image_utils import Im
        #     Im.concat_vertical(src_img, integer_to_color((src_seg.long() + 1)[0, 0])).save(str(idx))

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

        def process_data(_data: Data):
            _data.image = _data.image.squeeze(0)
            _data.segmentation = rearrange(_data.segmentation, "() c h w -> h w c")
            assert _data.segmentation.max() < 255
            _data.segmentation[_data.segmentation == -1] = 255
            
            if self.merge_masks:
                _data.segmentation[_data.segmentation >= 8] = 0

            _data.pad_mask = ~(_data.segmentation < 255).any(dim=-1)

            pixels = _data.segmentation.long().contiguous().view(-1)
            pixels = pixels[(pixels < 255) & (pixels >= 0)]
            src_bincount = torch.bincount(pixels, minlength=255)
            _data.valid = src_bincount > 0

            return _data

        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        if len(torch.unique(src_data.segmentation)) <= 2:
            raise Exception()

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
            "src_pose": src_pose,
            "tgt_pose": tgt_pose,
            "input_ids": get_tokens(self.tokenizer),
            "src_valid": torch.full((255,), True, dtype=torch.bool),
            "tgt_valid": torch.full((255,), True, dtype=torch.bool),
            "valid": src_data.valid[..., 1:],
            **metadata
        })

        return ret
    
    def get_metadata(self, idx):
        if self.return_raw_dataset_image:
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
            "has_global_instance_ids": torch.tensor(True) if self.return_only_instance_seg else torch.tensor(False),
            "metadata": {
                "dataset": "scannetpp",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": idx,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
                "split": self.split.name.lower(),
            },
        }

    def get_dataset(self):
        return self


import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

@app.command()
def main(
    num_workers: int = 0,
    batch_size: int = 2,
    scenes: Optional[list[str]] = None,
    viz: bool = True,
    steps: Optional[int] = None,
    sync_dataset_to_scratch: bool = False,
    return_raw_dataset_image: bool = False,
    breakpoint_on_start: bool = False,
    use_segmentation: bool = True,
    single_scene_debug: bool = False,
    total_size: Optional[int] = None,
    index: Optional[int] = None,
    return_tensorclass: bool = True,
    no_filtering: bool = False,
    return_only_instance_seg: bool = False,
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
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            tokenizer=MockTokenizer(),
            augmentation=augmentation,
            return_tensorclass=return_tensorclass,
            scenes=scenes,
            sync_dataset_to_scratch=sync_dataset_to_scratch,
            return_raw_dataset_image=return_raw_dataset_image,
            top_n_masks_only=128,
            num_overlapping_masks=1,
            use_segmentation=use_segmentation,
            use_cuda=False,
            no_filtering=no_filtering,
            return_only_instance_seg=return_only_instance_seg
        )

        subset_range = None
        if total_size is not None:
            subset_range = list(split_range(range(len(dataset)), total_size))[index]

        dataloader = dataset.get_dataloader(pin_memory=False, subset_range=subset_range)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if breakpoint_on_start: breakpoint()
            if viz: visualize_input_data(batch, show_overlapping_masks=True, remove_invalid=False)
            if steps is not None and i >= steps - 1: break

if __name__ == "__main__":
    with breakpoint_on_error():
        app()