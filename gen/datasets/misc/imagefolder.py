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

from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.utils.data_defs import visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, hash_str_as_int
from gen.utils.file_utils import get_available_path

@inherit_parent_args
class ImagefolderDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        augmentation: Optional[Augmentation] = None,
        tokenizer = None,
        **kwargs
    ):
        self.root = root
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.pairs = [x.name for x in root.iterdir() if x.is_dir()]
        self.saved_scene_frames = defaultdict(list)
        for scene in self.pairs:
            self.saved_scene_frames[scene] = sorted([int(x.stem) for x in (root / scene).iterdir()])

    def __len__(self) -> int:
        return len(self.saved_scene_frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.get_paired_data(idx)
            
    def get_image(self, image_path: Path):
        image_path = get_available_path(image_path, resolve=False, return_scratch_only=False)
        with open(image_path, 'rb', buffering=100*1024) as fp:
            data = fp.read()
        image = simplejpeg.decode_jpeg(data)
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        return image
    
    def get_paired_data(self, idx: int):
        metadata = self.get_metadata(idx)

        scene_id = metadata['metadata']['scene_id']
        src_img_idx, tgt_img_idx = metadata['metadata']['frame_idxs']
        
        src_path = self.root / scene_id / f"{src_img_idx}.jpg"
        tgt_path = self.root / scene_id / f"{tgt_img_idx}.jpg"

        src_img = self.get_image(src_path)
        tgt_img = self.get_image(tgt_path)

        ret = {}
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=src_img.to(self.device)),
            tgt_data=Data(image=tgt_img.to(self.device)),
            use_keypoints=False, 
            return_encoder_normalized_tgt=True
        )

        tgt_data_dec_transform, tgt_data_enc_transform = tgt_data
        src_data_enc_transform, src_data_dec_transform = src_data

        ret.update({
            "src_enc_rgb": src_data_enc_transform.image.squeeze(),
            "tgt_enc_rgb": tgt_data_enc_transform.image.squeeze(),
            "src_dec_rgb": src_data_dec_transform.image.squeeze(),
            "tgt_dec_rgb": tgt_data_dec_transform.image.squeeze(),
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
        }

    def get_dataset(self):
        return self


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