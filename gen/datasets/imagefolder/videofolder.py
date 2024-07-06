import autoroot

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import typer
from einops import rearrange
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from gen import HYPERSIM_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.imagefolder.imagefolder import get_depth_image, get_rgb_image
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int
from gen.utils.logging_utils import log_warn

@inherit_parent_args
class VideofolderDataset(AbstractDataset, IterableDataset):
    def __init__(
            self,
            *,
            root: Optional[Path] = None,
            augmentation: Optional[Augmentation] = None,
            top_n_masks_only: int = 255,
            tokenizer = None,
            return_n_views: int = 2,
            load_depth: bool = False,
            camera_trajectory_window: int = 4,
            legacy_format: bool = False,
            postfix: Optional[str] = None,
            rgb_prefix: str = "rgb",
            depth_prefix: str = "depth",
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            return_different_views: bool = False,
            uniform_sampler: bool = False,
            **kwargs,
        ):
        self.top_n_masks_only = top_n_masks_only
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.root = root
        self.return_n_views = return_n_views
        assert self.augmentation.reorder_segmentation is False
        self.pairs = [x.name for x in root.iterdir() if x.is_dir()]
        if postfix is not None:
            self.pairs = [x + "/" + postfix for x in self.pairs]
        self.saved_scene_frames = defaultdict(list)
        self.load_depth = load_depth
        self.legacy_format = legacy_format
        self.camera_trajectory_window = camera_trajectory_window
        self.postfix = postfix
        self.rgb_prefix = rgb_prefix
        self.depth_prefix = depth_prefix
        for scene in self.pairs:
            self.saved_scene_frames[scene] = sorted([x.name for x in (root / scene).iterdir() if self.rgb_prefix in x.stem and x.suffix in ('.png', '.jpg')])
    
    def collate_fn(self, batch):
        return super().collate_fn(batch)
    
    def __len__(self):
        return 10000000000
    
    def __iter__(self):
        while True:
            assert self.augmentation.return_grid is False
            scene_name = self.pairs[random.randint(0, len(self.pairs) - 1)]
            camera_trajectory_frames = self.saved_scene_frames[scene_name]
            window_size = min(max(1, self.camera_trajectory_window), len(camera_trajectory_frames))

            if window_size < self.return_n_views:
                log_warn(f"Camera trajectory has less than {window_size} frames. Returning a random frame.")
                return self.__getitem__(random.randint(0, len(self) - 1))

            first_frame_idx = random.randint(0, len(camera_trajectory_frames) - 1)
            lower_bound = max(0, first_frame_idx - window_size)
            upper_bound = min(len(camera_trajectory_frames) - 1, first_frame_idx + window_size)
            possible_indices = list(range(lower_bound, upper_bound + 1))
            frame_idxs = random.sample(possible_indices, self.return_n_views)
            ret = self.get_two_frames(scene_name, [self.saved_scene_frames[scene_name][idx] for idx in frame_idxs])
            yield ret

    def get_two_frames(self, scene_name, indices):
        ret = {}
        def _process(_data):
            rgb = get_rgb_image(self.root / scene_name / _data)
            seg = get_depth_image(self.root / scene_name / _data, rgb_prefix=self.rgb_prefix, depth_prefix=self.depth_prefix)
            if seg.max().item() < 0: raise Exception(f"Segmentation mask has only one unique value for index")
            return rgb, seg
        
        rgb, seg = zip(*[_process(frame_data_) for frame_data_ in indices])
        rgb = torch.cat(rgb, dim=0)
        seg = torch.cat(seg, dim=0).to(torch.float32)

        src_data, _ = self.augmentation(
            src_data=Data(image=rgb, segmentation=seg),
            tgt_data=None,
            use_keypoints=False,
            return_encoder_normalized_tgt=False,
        )

        def process_data(data_: Data):
            data_.segmentation = data_.segmentation.squeeze(1).to(torch.int32)
            return data_
        
        src_data = process_data(src_data)
        
        if self.legacy_format:
            ret.update({
                "src_dec_rgb": src_data.image[0],
                "tgt_dec_rgb": src_data.image[1],
                "src_dec_depth": src_data.segmentation[0],
                "tgt_dec_depth": src_data.segmentation[1],
                "src_xyz_valid": src_data.segmentation[0] > 0,
                "tgt_xyz_valid": src_data.segmentation[1] > 0,
            })

        ret.update({
            "dec_rgb": src_data.image,
            "dec_depth": src_data.segmentation,
            "xyz_valid": src_data.segmentation > 0,
            "id": torch.tensor([hash_str_as_int('a')], dtype=torch.long),
            **self.get_metadata(scene_name, indices)
        })

        return ret

    def get_dataset(self):
        return self
        
    def get_metadata(self, scene_name, indices):
        frame_name = "_".join(indices)
        frame_idxs = (hash_str_as_int(indices[0]), hash_str_as_int(indices[1]))
        name = f"{scene_name}_{frame_name}"
        return {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "dataset": "videofolder",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": 0,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
                "split": self.split.name.lower(),
            },
        }

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    root: Optional[Path] = None,
    num_workers: int = 0,
    batch_size: int = 1,
    viz: bool = True,
    steps: Optional[int] = None,
    breakpoint_on_start: bool = False,
    return_tensorclass: bool = True,
    load_depth: bool = False,
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
        dataset = VideofolderDataset(
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
            legacy_format=True,
            camera_trajectory_window=24,
        )
        subset_range = None
        dataloader = dataset.get_dataloader(pin_memory=False, subset_range=subset_range)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if breakpoint_on_start: breakpoint()
            if viz: visualize_input_data(batch)
            if steps is not None and i >= steps - 1: break

if __name__ == "__main__":
    with breakpoint_on_error():
        app()