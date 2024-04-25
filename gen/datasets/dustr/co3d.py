import autoroot

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import typer
from torch.utils.data import Dataset
from tqdm import tqdm

from gen import DUSTR_REPO_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.models.dustr.depth_utils import xyz_to_depth
from gen.models.dustr.geometry import depthmap_to_absolute_camera_coordinates
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int

@inherit_parent_args
class Co3d(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        augmentation: Optional[Augmentation] = None,
        tokenizer = None,
        resolution: int = 512,
        **kwargs
    ):
        sys.path.append(str(DUSTR_REPO_PATH))
        from dust3r.datasets.co3d import Co3d as DustrCo3d
        ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        _split = 'train' if self.split == Split.TRAIN else 'test'
        self.dataset = DustrCo3d(transform=ColorJitter, split=_split, ROOT=str(DUSTR_REPO_PATH / 'data/co3d_subset_processed'), aug_crop=16, mask_bg='rand', resolution=[(resolution, resolution)])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.get_paired_data(idx)
    
    def get_paired_data(self, idx: int):
        metadata = self.get_metadata(idx)
        left, right = self.dataset[idx]
                
        ret = {}
        ret.update({
            "src_enc_rgb": left["img"],
            "src_dec_rgb": left["img"],
            "src_xyz": left["pts3d"],
            "src_xyz_valid": left["valid_mask"],
            "src_intrinsics": left['camera_intrinsics'],
            "src_extrinsics": left['camera_pose'],
            "src_dec_depth": left["depthmap"],
            "tgt_enc_rgb": right["img"],
            "tgt_dec_rgb": right["img"],
            "tgt_xyz": right["pts3d"],
            "tgt_xyz_valid": right["valid_mask"],
            "tgt_intrinsics": right['camera_intrinsics'],
            "tgt_extrinsics": right['camera_pose'],
            "tgt_dec_depth": right["depthmap"],
            **metadata
        })

        return ret
    
    def get_metadata(self, idx):
        # TODO: Add real values here

        scene_name = str(idx)
        frame_name = str(idx)
        frame_idxs = (idx, idx)
        name = f"{scene_name}_{frame_name}"

        return {
            "id": torch.tensor([hash_str_as_int(name)], dtype=torch.long),
            "metadata": {
                "dataset": "co3d",
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
        dataset = Co3d(
            shuffle=True,
            cfg=None,
            split=Split.TRAIN,
            num_workers=num_workers,
            batch_size=batch_size,
            augmentation=None,
            return_tensorclass=return_tensorclass,
            use_cuda=False,
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