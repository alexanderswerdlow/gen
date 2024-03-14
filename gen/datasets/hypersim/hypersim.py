import os
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
import typer
from einops import rearrange
from nicr_scene_analysis_datasets.pytorch import Hypersim as BaseHypersim
from torch.utils.data import Dataset

from gen import HYPERSIM_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.utils.decoupled_utils import breakpoint_on_error, set_global_breakpoint
from gen.utils.logging_utils import log_error
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
            return_multiple_frames: bool = False,
            return_raw_dataset_image: bool = False,
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

        self.top_n_masks_only = top_n_masks_only
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.add_background = add_background
        self.return_multiple_frames = return_multiple_frames
        self.return_raw_dataset_image = return_raw_dataset_image

        self.hypersim = ModifiedHypersim(
            dataset_path=root,
            split='train' if self.split == Split.TRAIN else 'valid',
            sample_keys=("rgb", "instance", "identifier", "extrinsics"),
        )

        self.scene_cam_map = defaultdict(list)
        for idx, filename in enumerate(self.hypersim._filenames):
            parts = filename.split(os.sep)
            scene_name, camera_trajectory, camera_frame = parts
            key = (scene_name, camera_trajectory)
            self.scene_cam_map[key].append((idx, camera_frame))
    
    def __len__(self):
        return len(self.hypersim)
    
    def collate_fn(self, batch):
        if self.return_multiple_frames:
            batch = list(chain.from_iterable(batch))
        
        return super().collate_fn(batch)
    
    def __getitem__(self, index):
        if self.return_multiple_frames:
            camera_trajectory_frames = self.scene_cam_map[self.hypersim._load_identifier(index)[:2]]
            assert len(camera_trajectory_frames) > 1, "Need at least 2 frames for multiple frame mode."
            sampled_frames = random.sample(camera_trajectory_frames, 2)
            return [self.get_single_frame(idx) for idx, _ in sampled_frames]
        else:
            return self.get_single_frame(index)

    def get_single_frame(self, index):
        try:
            data = self.hypersim.__getitem__(index)
        except Exception as e:
            log_error(e)
            return self.__getitem__(random.randint(0, len(self))) # Very hacky but might save us in case of an error with a single instance.

        rgb, seg, metadata = torch.from_numpy(data['rgb']), torch.from_numpy(data['instance']), data["identifier"]
        rgb = rearrange(rgb / 255, "h w c -> () c h w")
        seg = rearrange(seg.to(torch.float32), "h w -> () () h w")
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=rgb, segmentation=seg),
            tgt_data=Data(image=rgb, segmentation=seg),
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
        
        ret = {
            "tgt_pad_mask": tgt_pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation,
            "src_pad_mask": src_pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid[..., 1:],
            "metadata": {
                "scene_id": metadata[0],
                "camera_trajectory": metadata[1],
                "camera_frame": metadata[2],
                "index": index,
            },
        }

        if src_data.grid is not None: ret["src_grid"] = src_data.grid.squeeze(0)
        if tgt_data.grid is not None: ret["tgt_grid"] = tgt_data.grid.squeeze(0)
        if self.return_raw_dataset_image: ret["raw_dataset_image"] = data["rgb"]


        return ret

    def get_dataset(self):
        return self


typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main():
    from transformers import AutoTokenizer

    from image_utils import library_ops
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    from gen.datasets.utils import get_stable_diffusion_transforms
    soda_augmentation=Augmentation(
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
        reorder_segmentation=True
    )
    dataset = Hypersim(
        shuffle=True,
        cfg=None,
        split=Split.TRAIN,
        num_workers=4,
        batch_size=32,
        tokenizer=tokenizer,
        augmentation=soda_augmentation,
        return_tensorclass=True,
        return_multiple_frames=True,
    )

    import time
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator, pin_memory=False) #.to(torch.device('cuda:0'))

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        names = [f'{batch.metadata["scene_id"][i]}_{batch.metadata["camera_trajectory"][i]}_{batch.metadata["camera_frame"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        # visualize_input_data(batch, names=names, show_overlapping_masks=True)
        start_time = time.time()
        if step > 100:
            break
      
if __name__ == "__main__":
    with breakpoint_on_error():
        app()