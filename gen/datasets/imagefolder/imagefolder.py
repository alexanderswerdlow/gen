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

def get_jpeg_image(image_path: Path):
    image_path = get_available_path(image_path, resolve=False, return_scratch_only=False)
    with open(image_path, 'rb', buffering=100*1024) as fp:
        data = fp.read()
    image = simplejpeg.decode_jpeg(data)
    image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    return image

def get_rgb_image(image_path: Path):
    if image_path.suffix == '.jpg':
        return get_jpeg_image(image_path)
    else:
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        return image

def get_depth_image(rgb_image_path: Path, rgb_prefix: str = "rgb", depth_prefix: str = "depth"):
    depth_image_path = rgb_image_path.with_name(rgb_image_path.name.replace(rgb_prefix, depth_prefix))
    if not depth_image_path.exists():
        depth_image_path = depth_image_path.with_suffix('.png')
    return torch.from_numpy(np.asarray(Image.open(depth_image_path)).copy().astype(np.float32))[None, None]

save_image_idx = 0
def save_data(batch, save_image_path):
    global save_image_idx
    from PIL import Image
    for b in range(batch.bs):
        save_path = save_image_path / str(save_image_idx)
        save_path.mkdir(parents=True, exist_ok=True)
        src_depth_img = batch.src_dec_depth[b]
        tgt_depth_img = batch.tgt_dec_depth[b]

        src_rgb_img = (((batch.src_dec_rgb[b] + 1) / 2) * 255).to(torch.uint8).permute(1, 2, 0)
        tgt_rgb_img = (((batch.tgt_dec_rgb[b] + 1) / 2) * 255).to(torch.uint8).permute(1, 2, 0)

        Image.fromarray(src_depth_img.numpy().astype(np.uint16)).save(save_path / "src_depth.png")
        Image.fromarray(tgt_depth_img.numpy().astype(np.uint16)).save(save_path / "tgt_depth.png")
        Image.fromarray(src_rgb_img.numpy().astype(np.uint8)).save(save_path / "src_rgb.png")
        Image.fromarray(tgt_rgb_img.numpy().astype(np.uint8)).save(save_path / "tgt_rgb.png")

        save_image_idx += 1
        print(save_image_idx)

@inherit_parent_args
class ImagefolderDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        augmentation: Optional[Augmentation] = None,
        tokenizer = None,
        load_depth: bool = False,
        return_different_views=None,
        **kwargs
    ):
        self.root = root
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.pairs = [x.name for x in root.iterdir() if x.is_dir()]
        self.saved_scene_frames = defaultdict(list)
        self.load_depth = load_depth
        for scene in self.pairs:
            self.saved_scene_frames[scene] = sorted([x.name for x in (root / scene).iterdir() if 'depth' not in x.stem])

    def __len__(self) -> int:
        return len(self.saved_scene_frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.get_paired_data(idx)

    def get_paired_data(self, idx: int):
        metadata, src_img_idx, tgt_img_idx = self.get_metadata(idx)

        scene_id = metadata['metadata']['scene_id']
        
        src_path = self.root / scene_id / src_img_idx
        tgt_path = self.root / scene_id / tgt_img_idx
        
        src_img = get_rgb_image(src_path)
        tgt_img = get_rgb_image(tgt_path)

        ret = {}

        src_seg_img, tgt_seg_img = None, None
        if self.load_depth:
            src_seg_img = get_depth_image(src_path).to(self.device)
            tgt_seg_img = get_depth_image(tgt_path).to(self.device)
        
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=src_img.to(self.device), segmentation=src_seg_img),
            tgt_data=Data(image=tgt_img.to(self.device), segmentation=tgt_seg_img),
            use_keypoints=False, 
            return_encoder_normalized_tgt=True
        )

        tgt_data_dec_transform, tgt_data_enc_transform = tgt_data
        src_data_enc_transform, src_data_dec_transform = src_data

        ret.update({
            "src_dec_rgb": src_data_dec_transform.image.squeeze(),
            "tgt_dec_rgb": tgt_data_dec_transform.image.squeeze(),
            **metadata
        })

        ret.update({
            "dec_rgb": torch.stack([ret['src_dec_rgb'], ret['tgt_dec_rgb']], dim=0),
        })

        if self.load_depth:
            ret.update({
                "src_dec_depth": src_data_dec_transform.segmentation.squeeze(0).squeeze(0).to(torch.int32),
                "tgt_dec_depth": tgt_data_dec_transform.segmentation.squeeze(0).squeeze(0).to(torch.int32),
            })

            ret.update({
                "src_xyz_valid": ret['src_dec_depth'] > 0,
                "tgt_xyz_valid": ret['tgt_dec_depth'] > 0,
            })
            
            ret.update({
                "xyz_valid": torch.stack([ret['src_xyz_valid'], ret['tgt_xyz_valid']], dim=0),
                "dec_depth": torch.stack([ret['src_dec_depth'], ret['tgt_dec_depth']], dim=0),
            })

        return ret
    
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