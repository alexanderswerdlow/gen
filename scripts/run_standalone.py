from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from gen.configs.utils import get_cfg
from gen.datasets.augmentation.kornia_augmentation import Data
from gen.models.utils import get_model_from_cfg
from gen.utils.decoupled_utils import breakpoint_on_error
from gen.utils.trainer_utils import TrainingState, load_from_ckpt
from einops import rearrange
import hydra
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from torch import nn
    from gen.configs.base import BaseConfig
    from jaxtyping import Bool, Float, Integer
    from torch import Tensor

def setup_model(cfg: BaseConfig):
    model = get_model_from_cfg(cfg)

    if cfg.trainer.ckpt is not None:
        load_from_ckpt(cfg=cfg, accelerator=None, model=model, load_model=True)

    return model

def setup_inference(device: Optional[torch.device] = "cuda", dtype: Optional[torch.dtype] = torch.bfloat16):
    cfg = get_cfg(overrides=[
        "+experiment=gen",
        "+modes=['single_image_pretraining_v3','calvin']",
        "inference=susie_inference",
        "trainer.ckpt='/home/aswerdlo/data/checkpoints/2024-04-13_02_19_02/pTuFZyYREb/checkpoint_31000/state/pytorch_model.bin'"
    ])
    cfg.trainer.device = device
    assert cfg.trainer.dtype == dtype

    model = setup_model(cfg)
    model.to(device=cfg.trainer.device, dtype=cfg.trainer.dtype)
    model.set_inference_mode()

    return cfg, model

def process_dict(cfg: BaseConfig, batch: dict):
    dtype = cfg.trainer.dtype
    device = cfg.trainer.device

    def convert_seg_to_tensor(arr):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        assert arr.ndim == 3 and arr.dtype == torch.uint8
        return rearrange(arr.to(device), "b h w -> b () h w").to(device=device, dtype=torch.float32)
    
    def convert_img_to_tensor(arr):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        assert arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == torch.uint8
        return rearrange(arr.to(device) / 255., "b h w c -> b c h w").to(device=device, dtype=torch.float32)

    img: Integer[Tensor, "b h w c"] = batch["src_pixel_values"] # We expect uint8 [0-255]
    seg: Integer[Tensor, "b h w"] = batch["src_segmentation"] # We expect uint8 [0-255]. 255 will be ignored.
    input_ids: list[str] = batch["input_ids"] # List of scene captions

    img = convert_img_to_tensor(img)
    seg = convert_seg_to_tensor(seg)
    return_encoder_normalized_tgt = True

    augmentation = hydra.utils.instantiate(cfg.dataset.val.augmentation)
    src_data, tgt_data = augmentation(
        src_data=Data(image=img, segmentation=seg),
        tgt_data=Data(image=img, segmentation=seg),
        use_keypoints=False, 
        return_encoder_normalized_tgt=return_encoder_normalized_tgt
    )

    if return_encoder_normalized_tgt:
        tgt_data, tgt_data_src_transform = tgt_data
    
    def process_data(data_: Data):
        data_.segmentation = rearrange(data_.segmentation, "b c h w -> b h w c")
        assert data_.segmentation.max() < 255
        data_.segmentation[data_.segmentation == -1] = 255
        return data_

    src_data = process_data(src_data)
    tgt_data = process_data(tgt_data)

    ret = {}

    if return_encoder_normalized_tgt:
        tgt_data_src_transform = process_data(tgt_data_src_transform)
        ret.update({
            "tgt_enc_norm_pixel_values": tgt_data_src_transform.image.to(dtype),
            "tgt_enc_norm_segmentation": tgt_data_src_transform.segmentation.to(torch.uint8),
        })
    
    ret.update({
        "tgt_pixel_values": tgt_data.image.to(dtype),
        "tgt_segmentation": tgt_data.segmentation.to(torch.uint8),
        "src_pixel_values": src_data.image.to(dtype),
        "src_segmentation": src_data.segmentation.to(torch.uint8),
        "input_ids": input_ids
    })
    
    return ret


def run_inference(cfg: BaseConfig, model: nn.Module, batch: dict, state: Optional[TrainingState] = None, use_saved_data: bool = False):
    state = TrainingState(0, 0, 0, 0, 0)
    batch = process_dict(cfg, batch)
    if use_saved_data:
        batch = torch.load("/projects/katefgroup/aswerdlo/gen/tmp_data/calvin_batch.pt")
    batch = model.process_input(batch=batch, state=state)
    batch = batch.to(device=cfg.trainer.device)
    output: Image = model.run_inference(batch=batch, state=state)
    output.save("output.png")
    return output

if __name__ == '__main__':
    with breakpoint_on_error():
        cfg, model = setup_inference()
        batch = {
            "src_pixel_values": np.zeros((2, 256, 256, 3), dtype=np.uint8),
            "src_segmentation": np.zeros((2, 256, 256), dtype=np.uint8),
            "input_ids": ["This is a test sentence", "This is another test sentence"],
        }
        run_inference(cfg, model, batch)
        
