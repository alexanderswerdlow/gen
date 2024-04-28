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
            repeat_n: int = 1,
            scratch_only: bool = True,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
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
        self.repeat_n = repeat_n
        self.scratch_only = scratch_only
        assert self.augmentation.reorder_segmentation is False

        self.hypersim = ModifiedHypersim(
            dataset_path=root,
            split='train' if self.split == Split.TRAIN else 'valid',
            sample_keys=("rgb", "instance", "identifier", "rgb_intrinsics", "depth_intrinsics", "extrinsics", "3d_boxes", "depth"),
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
        return len(self.hypersim) * self.repeat_n
    
    def collate_fn(self, batch):
        return super().collate_fn(batch)
    
    def __getitem__(self, index):
        for i in range(60):
            try:
                return self.getitem(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)
                if i == 59:
                    log_error(e)

        raise Exception(f'Hypersim Failed to get item for index {index}')
    
    def getitem(self, index):
        index = index % len(self.hypersim)
        if self.return_different_views:
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
            
            ret = self.get_two_frames(camera_trajectory_frames[first_frame_idx][0], camera_trajectory_frames[second_frame_idx][0])
            return ret
        else:
            return self.get_two_frames(index, index)

    def get_two_frames(self, src_index, tgt_index):
        init_src_data = self.hypersim.__getitem__(src_index)

        if src_index == tgt_index:
            init_tgt_data = init_src_data
        else:
            init_tgt_data = self.hypersim.__getitem__(tgt_index)

        ret = {}

        def _process(_data):
            rgb, seg, metadata = torch.from_numpy(_data['rgb']).to(self.device), torch.from_numpy(_data['depth'].astype(np.int32)).to(self.device), _data["identifier"]
            if seg.max().item() < 0: raise Exception(f"Segmentation mask has only one unique value for index {src_index}")
            rgb = rearrange(rgb / 255, "h w c -> () c h w")
            seg = rearrange(seg.to(torch.float32), "h w -> () () h w")
            return rgb, seg, metadata
        
        src_rgb, src_seg, src_metadata = _process(init_src_data)
        tgt_rgb, tgt_seg, tgt_metadata = _process(init_tgt_data)

        src_data, tgt_data = self.augmentation(
            src_data=Data(image=src_rgb, segmentation=src_seg),
            tgt_data=Data(image=tgt_rgb, segmentation=tgt_seg),
            use_keypoints=False, 
            return_encoder_normalized_tgt=self.return_encoder_normalized_tgt
        )

        def process_data(data_: Data):
            data_.image = data_.image.squeeze(0)
            data_.segmentation = data_.segmentation.squeeze(0).squeeze(0).to(torch.int32)
            return data_
        
        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)
        
        name = "_".join(src_metadata) + "_" + "_".join(tgt_metadata)

        def get_rt(_data):
            extrinsics = _data['extrinsics']
            rot = R.from_quat((extrinsics['quat_x'], extrinsics['quat_y'], extrinsics['quat_z'], extrinsics['quat_w']))
            T = torch.tensor([extrinsics['x'], extrinsics['y'], extrinsics['z']]).view(3, 1) / 50
            RT = torch.cat((torch.from_numpy(rot.as_matrix()), T), dim=1)
            RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]])), dim=0)
            return RT

        def get_intrinsics_matrix(intrinsics_dict):
            return torch.tensor([
                [intrinsics_dict['fx'], 0, intrinsics_dict['cx']],
                [0, intrinsics_dict['fy'], intrinsics_dict['cy']],
                [0, 0, 1]
            ])

        ret.update({
            "src_dec_rgb": src_data.image,
            "tgt_dec_rgb": tgt_data.image,
            "src_dec_depth": src_data.segmentation,
            "tgt_dec_depth": tgt_data.segmentation,
            "src_xyz_valid": torch.ones_like(src_data.segmentation, dtype=torch.bool),
            "tgt_xyz_valid": torch.ones_like(tgt_data.segmentation, dtype=torch.bool),
            "src_extrinsics": get_rt(init_src_data),
            "tgt_extrinsics": get_rt(init_tgt_data),
            "src_intrinsics": get_intrinsics_matrix(init_src_data['depth_intrinsics']),
            "tgt_intrinsics": get_intrinsics_matrix(init_tgt_data['depth_intrinsics']),
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "dataset": "hypersim",
                "name": name,
                "scene_id": src_metadata[0],
                "camera_frame": src_metadata[2],
                "index": src_index,
                "camera_trajectory": src_metadata[1],
                "split": self.split.name.lower(),
                "frame_idxs": (0, 0) # Dummy value
            },
        })

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
    augmentation=Augmentation(
        return_grid=False,
        enable_square_crop=True,
        center_crop=True,
        different_src_tgt_augmentation=False,
        enable_random_resize_crop=False, 
        enable_horizontal_flip=False,
        src_random_scale_ratio=None,
        tgt_random_scale_ratio=None,
        enable_rand_augment=False,
        enable_rotate=False,
        src_transforms=get_stable_diffusion_transforms(resolution=512),
        tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        reorder_segmentation=False
    )
    dataset = Hypersim(
        shuffle=True,
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=32,
        tokenizer=MockTokenizer(),
        augmentation=augmentation,
        return_tensorclass=True,
        return_different_views=True,
        bbox_overlap_threshold=0.9,
        bbox_area_threshold=0.5,
    )

    import time
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator, pin_memory=False)

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        if step == 2: exit()
        print(f'Time taken: {time.time() - start_time}')
        names = [f'{batch.metadata["scene_id"][i]}_{batch.metadata["camera_trajectory"][i]}_{batch.metadata["camera_frame"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        visualize_input_data(batch, names=names)
      
if __name__ == "__main__":
    with breakpoint_on_error():
        app()