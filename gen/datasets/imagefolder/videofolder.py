import autoroot
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int
from gen.utils.file_utils import get_available_path

import os
import pickle
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Optional, Tuple

from joblib import Memory
import numpy as np
import torch
import torchvision
from triton import MockTensor
import typer
from einops import rearrange
from nicr_scene_analysis_datasets.pytorch import Hypersim as BaseHypersim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from gen import GLOBAL_CACHE_PATH, HYPERSIM_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.run_dataloader import MockTokenizer
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int, set_global_breakpoint
from gen.utils.logging_utils import log_error, log_info, log_warn

@inherit_parent_args
class VideofolderDataset(AbstractDataset, Dataset):
    def __init__(
            self,
            *,
            root: Path = HYPERSIM_DATASET_PATH,
            augmentation: Optional[Augmentation] = None,
            top_n_masks_only: int = 255,
            add_background: bool = False,
            tokenizer = None,
            return_different_views: bool = False,
            return_n_views: int = 2,
            return_raw_dataset_image: bool = False,
            camera_trajectory_window: int = 28,
            bbox_overlap_threshold: float = 0.65,
            bbox_area_threshold: float = 0.75,
            segmentation_overlap_threshold: float = 0.65,
            num_overlapping_masks: int = 1,
            return_encoder_normalized_tgt: bool = False,
            repeat_n: int = 1,
            scratch_only: bool = True,
            return_tgt_only: bool = True,
            legacy_format: bool = False,
            uniform_sampler: bool = False,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            **kwargs
        ):

        self.top_n_masks_only = top_n_masks_only
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.root = root
        self.return_n_views = return_n_views
        assert self.augmentation.reorder_segmentation is False

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
        return self.getitem(index)
    
    def getitem(self, index):
        index = index % len(self.hypersim)
        if self.return_different_views:
            assert self.augmentation.return_grid is False
            camera_trajectory_frames = self.scene_cam_map[self.hypersim._load_identifier(index)[:2]]
            window_size = min(max(1, self.camera_trajectory_window), len(camera_trajectory_frames))

            if window_size < self.return_n_views:
                log_warn(f"Camera trajectory {camera_trajectory_frames[0][1]} has less than {window_size} frames. Returning a random frame.")
                return self.__getitem__(random.randint(0, len(self) - 1))

            if self.uniform_sampler:
                frame_idxs = random.sample(range(len(camera_trajectory_frames)), self.return_n_views)
            else:
                frame_idxs = self.nonuniform_sampler(camera_trajectory_frames, window_size)
            
            ret = self.get_two_frames((camera_trajectory_frames[_idx][0] for _idx in frame_idxs))
            return ret
        else:
            return self.get_two_frames((index, index))

    def get_two_frames(self, indices):
        frame_data = [self.hypersim.__getitem__(tgt_index) for tgt_index in indices]
        ret = {}
        def _process(_data):
            rgb, seg, metadata = torch.from_numpy(_data['rgb']).to(self.device), torch.from_numpy(_data['depth'].astype(np.int32)).to(self.device), _data["identifier"]
            if seg.max().item() < 0: raise Exception(f"Segmentation mask has only one unique value for index")
            rgb = rearrange(rgb / 255, "h w c -> () c h w")
            seg = rearrange(seg, "h w -> () () h w")
            return rgb, seg, metadata
        
        rgb, seg, metadata = zip(*[_process(frame_data_) for frame_data_ in frame_data])
        rgb = torch.cat(rgb, dim=0)
        seg = torch.cat(seg, dim=0).to(torch.float32)

        src_data, _ = self.augmentation(
            src_data=Data(image=rgb, segmentation=seg),
            tgt_data=None,
            use_keypoints=False, 
            return_encoder_normalized_tgt=False
        )

        def process_data(data_: Data):
            data_.segmentation = data_.segmentation.squeeze(1).to(torch.int32)
            return data_
        
        src_data = process_data(src_data)
        name = "__".join(("_".join(_meta) for _meta in metadata))
        
        if self.legacy_format:
            ret.update({
                "src_dec_rgb": src_data.image[0],
                "tgt_dec_rgb": src_data.image[1],
                "src_dec_depth": src_data.segmentation[0],
                "tgt_dec_depth": src_data.segmentation[1],
                "src_xyz_valid": src_data.segmentation[0] > 0,
                "tgt_xyz_valid": src_data.segmentation[1] > 0,
                "src_extrinsics": get_rt(frame_data[0]),
                "tgt_extrinsics": get_rt(frame_data[1]),
                "src_intrinsics": get_intrinsics_matrix(frame_data[0]['depth_intrinsics']),
                "tgt_intrinsics": get_intrinsics_matrix(frame_data[1]['depth_intrinsics']),
            })

        ret.update({
            "dec_rgb": src_data.image,
            "dec_depth": src_data.segmentation,
            "xyz_valid": src_data.segmentation > 0,
            "extrinsics": torch.stack([get_rt(_frame_data) for _frame_data in frame_data]),
            "intrinsics": torch.stack([get_intrinsics_matrix(_frame_data['depth_intrinsics']) for _frame_data in frame_data]),
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "dataset": "hypersim",
                "name": name,
                "scene_id": metadata[0][0],
                "camera_frame": metadata[0][2],
                "index": 0,
                "camera_trajectory": metadata[0][1],
                "split": self.split.name.lower(),
                "frame_idxs": (0, 0) # Dummy value
            },
        })

        return ret

    def get_dataset(self):
        return self

        
    def get_metadata(self, idx):
        scene_name = list(self.saved_scene_frames.keys())[idx]
        allowed_frames = self.saved_scene_frames[scene_name]

        src_img_idx, tgt_img_idx = np.random.choice(list(allowed_frames), size=2, replace=False)
        frame_name = f"{src_img_idx}-{tgt_img_idx}"
        frame_idxs = (hash_str_as_int(src_img_idx), hash_str_as_int(tgt_img_idx))

        name = f"{scene_name}_{frame_name}"
        return {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "dataset": "imagefolder",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": idx,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
                "split": self.split.name.lower(),
            },
        }, src_img_idx, tgt_img_idx



typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main():
    from image_utils import library_ops
    from gen.datasets.utils import get_stable_diffusion_transforms
    resolution = 256
    augmentation=Augmentation(
        return_grid=False,
        different_src_tgt_augmentation=False,
        src_random_scale_ratio=None,
        enable_rand_augment=False,
        enable_rotate=False,
        src_transforms=get_stable_diffusion_transforms(resolution=resolution),
        tgt_transforms=get_stable_diffusion_transforms(resolution=resolution),
        reorder_segmentation=False,
        enable_horizontal_flip=True,
        enable_square_crop=True,
        enable_zoom_crop=False,
        enable_random_resize_crop=None,
        tgt_random_scale_ratio=None,
        initial_resolution=768,
    )
    dataset = Hypersim(
        shuffle=True,
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        tokenizer=MockTokenizer(),
        augmentation=augmentation,
        return_tensorclass=True,
        batch_size=28,
        subset_size=28*28,
        random_subset=True,
        return_different_views=True,
        return_n_views=2,
        window_size=10,
        legacy_format=True,
        uniform_sampler=True,
    )

    import time
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator, pin_memory=False)

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        if step == 10: exit()
        print(f'Time taken: {time.time() - start_time}')
        start_time = time.time()
        names = [f'{batch.metadata["scene_id"][i]}_{batch.metadata["camera_trajectory"][i]}_{batch.metadata["camera_frame"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        visualize_input_data(batch, names=names)
      
if __name__ == "__main__":
    with breakpoint_on_error():
        app()


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
    load_depth: bool = False,
    save_images: Optional[Path] = None,
):
    with breakpoint_on_error():
        from gen.datasets.utils import get_stable_diffusion_transforms
        from image_utils import library_ops
        augmentation=Augmentation(
            initial_resolution=512,
            enable_square_crop=True,
            center_crop=True,
            src_transforms=get_stable_diffusion_transforms(resolution=224),
            tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        )
        dataset = ImagefolderDataset(
            root=root,
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            augmentation=augmentation,
            return_tensorclass=return_tensorclass,
            use_cuda=False,
            load_depth=load_depth,
        )

        subset_range = None
        dataloader = dataset.get_dataloader(pin_memory=False, subset_range=subset_range)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if breakpoint_on_start: breakpoint()
            if viz: visualize_input_data(batch)
            if steps is not None and i >= steps - 1: break
            if save_images is not None: save_data(batch, save_images)

if __name__ == "__main__":
    with breakpoint_on_error():
        app()