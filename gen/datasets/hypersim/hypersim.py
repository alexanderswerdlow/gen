
import autoroot

import os
import pickle
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
from triton import MockTensor
import typer
from einops import rearrange
from nicr_scene_analysis_datasets.pytorch import Hypersim as BaseHypersim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from gen import HYPERSIM_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.run_dataloader import MockTokenizer
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int, set_global_breakpoint
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import get_tokens

torchvision.disable_beta_transforms_warning()

class ModifiedHypersim(BaseHypersim):
    pass

@inherit_parent_args
class Hypersim(AbstractDataset, Dataset):
    def __init__(
            self,
            *,
            root: Path = HYPERSIM_DATASET_PATH,
            augmentation: Optional[Augmentation] = None,
            top_n_masks_only: int = 255,
            add_background: bool = False,
            tokenizer = None,
            return_different_views: bool = False,
            return_raw_dataset_image: bool = False,
            camera_trajectory_window: int = 24,
            bbox_overlap_threshold: float = 0.65,
            bbox_area_threshold: float = 0.75,
            num_overlapping_masks: int = 1,
            return_encoder_normalized_tgt: bool = False,
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
            **kwargs
        ):

        self.top_n_masks_only = top_n_masks_only
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.add_background = add_background
        self.return_different_views = return_different_views
        self.return_raw_dataset_image = return_raw_dataset_image
        self.camera_trajectory_window = camera_trajectory_window
        self.bbox_overlap_threshold = bbox_overlap_threshold
        self.bbox_area_threshold = bbox_area_threshold
        self.num_overlapping_masks = num_overlapping_masks
        self.return_encoder_normalized_tgt = return_encoder_normalized_tgt
        self.root = root

        self.hypersim = ModifiedHypersim(
            dataset_path=root,
            split='train' if self.split == Split.TRAIN else 'valid',
            sample_keys=("rgb", "instance", "identifier", "extrinsics", "3d_boxes"),
        )

        self.scene_cam_map = defaultdict(list)
        for idx, filename in enumerate(self.hypersim._filenames):
            parts = filename.split(os.sep)
            scene_name, camera_trajectory, camera_frame = parts
            key = (scene_name, camera_trajectory)
            self.scene_cam_map[key].append((idx, camera_frame))

        self.camera_trajectory_extents = pickle.load(open(root / 'extents.pkl', "rb"))
        log_info(f"Loaded {len(self.hypersim)} frames from hypersim dataset with root: {self.root}", main_process_only=False)
    
    def __len__(self):
        return len(self.hypersim)
    
    def collate_fn(self, batch):
        return super().collate_fn(batch)
    
    def __getitem__(self, index):
        if self.return_different_views:
            assert self.augmentation.reorder_segmentation is False
            assert self.augmentation.return_grid is False
            camera_trajectory_frames = self.scene_cam_map[self.hypersim._load_identifier(index)[:2]]
            window_size = min(max(1, self.camera_trajectory_window), len(camera_trajectory_frames))

            if window_size <= 2:
                log_warn(f"Camera trajectory {camera_trajectory_frames[0][1]} has less than {window_size} frames. Returning a random frame.")
                return self.__getitem__(random.randint(0, len(self) - 1))

            while True:
                first_frame_idx = random.randint(0, len(camera_trajectory_frames) - 1)
                lower_bound = max(0, first_frame_idx - window_size)
                upper_bound = min(len(camera_trajectory_frames) - 1, first_frame_idx + window_size)
                possible_indices = list(range(lower_bound, upper_bound + 1))
                second_frame_idx = random.choice(possible_indices)
                if self.sum_largest_face_areas(first_frame_idx, second_frame_idx):
                    break
            
            frame_data = [self.get_single_frame(camera_trajectory_frames[idx][0]) for idx in [first_frame_idx, second_frame_idx]]
            ret = frame_data[0]
            for k, v in frame_data[1].items():
                if 'tgt' in k:
                    ret[k] = v

            return ret
        else:
            return self.get_single_frame(index)

    def get_single_frame(self, index):
        try:
            data = self.hypersim.__getitem__(index)
        except Exception as e:
            log_error(e)
            return self.__getitem__(random.randint(0, len(self))) # Very hacky but might save us in case of an error with a single instance.

        ret = {}

        if self.return_raw_dataset_image: ret["raw_dataset_image"] = data["rgb"].copy()

        rgb, seg, metadata = torch.from_numpy(data['rgb']), torch.from_numpy(data['instance']), data["identifier"]
        rgb = rearrange(rgb / 255, "h w c -> () c h w")
        seg = rearrange(seg.to(torch.float32), "h w -> () () h w")
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=rgb, segmentation=seg),
            tgt_data=Data(image=rgb, segmentation=seg),
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
            data_.segmentation = torch.cat([data_.segmentation, data_.segmentation.new_full((*data_.segmentation.shape[:-1], self.num_overlapping_masks - 1), 255)], dim=-1)
            data_.pad_mask = ~(data_.segmentation < 255).any(dim=-1)
            return data_
        
        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        if self.return_encoder_normalized_tgt:
            tgt_data_src_transform = process_data(tgt_data_src_transform)
            ret.update({
                "tgt_enc_norm_pixel_values": tgt_data_src_transform.image,
                "tgt_enc_norm_segmentation": tgt_data_src_transform.segmentation.to(torch.uint8),
            })

        pixels = src_data.segmentation.long().contiguous().view(-1)
        pixels = pixels[(pixels < 255) & (pixels >= 0)]
        src_bincount = torch.bincount(pixels, minlength=self.top_n_masks_only + 1)
        valid = src_bincount > 0

        name = "_".join(metadata)

        extrinsics = data['extrinsics']
        rot = R.from_quat((extrinsics['quat_x'], extrinsics['quat_y'], extrinsics['quat_z'], extrinsics['quat_w']))

        # Normalize translation
        # camera_trajectory_extent = self.camera_trajectory_extents[(metadata[0], metadata[1])]
        T = torch.tensor([extrinsics['x'], extrinsics['y'], extrinsics['z']]).view(3, 1) / 50
        
        RT = torch.cat((torch.from_numpy(rot.as_matrix()), T), dim=1)
        RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]])), dim=0)

        ret.update({
            "tgt_pad_mask": tgt_data.pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation.to(torch.uint8),
            "src_pad_mask": src_data.pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation.to(torch.uint8),
            "src_pose": RT,
            "tgt_pose": RT,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid[..., 1:],
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "has_global_instance_ids": torch.tensor(True),
            "metadata": {
                "dataset": "hypersim",
                "name": name,
                "scene_id": metadata[0],
                "camera_frame": metadata[2],
                "index": index,
                "camera_trajectory": metadata[1],
                "frame_idxs": (0, 0) # Dummy value
            },
        })

        if src_data.grid is not None: ret["src_grid"] = src_data.grid.squeeze(0)
        if tgt_data.grid is not None: ret["tgt_grid"] = tgt_data.grid.squeeze(0)

        return ret

    def get_dataset(self):
        return self

    def sum_largest_face_areas(self, first_frame_idx, second_frame_idx):
        frame1 = self.hypersim._load_3d_boxes(first_frame_idx)
        frame2 = self.hypersim._load_3d_boxes(second_frame_idx)

        assert isinstance(frame1, dict) and isinstance(frame2, dict), "Inputs must be dictionaries."

        total_boxes_frame1 = len(frame1)
        total_boxes_frame2 = len(frame2)
        shared_keys = frame1.keys() & frame2.keys()  # Get the set of keys present in both frames
        total_shared_boxes = len(shared_keys)

        if total_shared_boxes < self.bbox_overlap_threshold * max(total_boxes_frame1, total_boxes_frame2):
            return False

        total_area = 0

        for key in shared_keys:
            extents1 = sorted(frame1[key]['extents'], reverse=True)  # Sort extents in descending order
            extents2 = sorted(frame2[key]['extents'], reverse=True)  # Sort extents for the second frame

            # The largest face area of a box is the product of its two largest extents
            area1 = extents1[0] * extents1[1]
            area2 = extents2[0] * extents2[1]

            total_area += max(area1, area2)  # Sum the larger of the two areas for each shared box

        return total_area > self.bbox_area_threshold

    def make_video(self, camera_scene, camera_traj):
        from image_utils import Im
        from image_utils.standalone_image_utils import get_color
        frames = self.scene_cam_map[(camera_scene, camera_traj)]
        rgb = []
        prev_ids = None
        all_ids = set()
        for frame_idx, frame_name in frames:
            data = self.hypersim.__getitem__(frame_idx)
            unique_ids = set(np.unique(data['instance']).tolist())
            if prev_ids is not None:
                print(len(unique_ids.symmetric_difference(prev_ids)), len(unique_ids), len(prev_ids))
            prev_ids = unique_ids
            all_ids.update(unique_ids)
            rgb.append(data["rgb"])

        colors = get_color(max(list(all_ids)) + 1)
        color_tensor = np.array(colors, dtype=np.uint8)  # [N, 3]

        instance = []
        for frame_idx, frame_name in frames:
            data = self.hypersim.__getitem__(frame_idx)
            instance.append(color_tensor[data['instance']])

        Im(np.stack(instance)).save_video(f"{camera_scene}_{camera_traj}_instance.mp4")
        Im(np.stack(rgb)).save_video(f"{camera_scene}_{camera_traj}.mp4")

        # frame1 = self.hypersim._load_3d_boxes(0)
        # frame2 = self.hypersim._load_3d_boxes(2)
        # sum_largest_face_areas(frame1, frame2)



typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main():
    from image_utils import library_ops
    from gen.datasets.utils import get_stable_diffusion_transforms
    soda_augmentation=Augmentation(
        return_grid=False,
        enable_square_crop=True,
        center_crop=False,
        different_src_tgt_augmentation=True,
        enable_random_resize_crop=True, 
        enable_horizontal_flip=True,
        src_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
        tgt_random_scale_ratio=((0.3, 0.6), (0.8, 1.2)),
        enable_rand_augment=False,
        enable_rotate=True,
        src_transforms=get_stable_diffusion_transforms(resolution=512),
        tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        reorder_segmentation=False
    )
    dataset = Hypersim(
        shuffle=True,
        cfg=None,
        split=Split.TRAIN,
        num_workers=2,
        batch_size=32,
        tokenizer=MockTokenizer(),
        augmentation=soda_augmentation,
        return_tensorclass=True,
        return_different_views=True,
    )

    import time
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator, pin_memory=False)

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        names = [f'{batch.metadata["scene_id"][i]}_{batch.metadata["camera_trajectory"][i]}_{batch.metadata["camera_frame"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        # visualize_input_data(batch, names=names, show_overlapping_masks=True)
        start_time = time.time()
        print(batch.src_pose[:, :3, 3].min(), batch.src_pose[:, :3, 3].max())
        if step > 20:
            break
      
if __name__ == "__main__":
    with breakpoint_on_error():
        app()