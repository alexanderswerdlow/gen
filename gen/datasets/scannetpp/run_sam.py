from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import hydra
import lmdb
import msgpack
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen import SCANNETPP_CUSTOM_DATA_PATH
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.run_dataloader import MockTokenizer
from gen.models.encoders.sam import HQSam
from gen.utils.data_defs import InputData, visualize_input_data
from gen.utils.decoupled_utils import get_time_sync, is_main_process, profile_memory_decorator, set_timing_builtins, show_memory_usage
from gen.utils.file_utils import get_available_path
from gen.utils.logging_utils import log_info
from image_utils import Im

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

def coco_decode_rle(compressed_rle) -> np.ndarray:
    from typing import Any, Dict

    import numpy as np
    from pycocotools import mask as mask_utils  # type: ignore

    if isinstance(compressed_rle['counts'], str):
        compressed_rle['counts'] = compressed_rle['counts'].encode()

    binary_mask = mask_utils.decode(compressed_rle)
    return binary_mask

def show_anns(image, anns, output_path: Path = Path("outputs/sam_hq.png")):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    if isinstance(sorted_anns[0]['segmentation'], dict):
        for ann in sorted_anns:
            ann['segmentation'] = coco_decode_rle(ann['segmentation']).astype(np.bool_)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = np.concatenate([np.random.random(3), [0.85]])
    ax.imshow(img)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()


def downsample(tensor, scale_factor):
    """
    Downsample a tensor while preserving aspect ratio.
    
    Args:
    - tensor (torch.Tensor): Input tensor with shape [B, H, W, C] and dtype uint8.
    - scale_factor (float): Scale factor for downsampling. Must be > 0 and < 1.
    
    Returns:
    - torch.Tensor: Downsampled tensor with the same dtype (uint8).
    """
    B, H, W, C = tensor.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    tensor_permuted = tensor.permute(0, 3, 1, 2).float()  # Change to [B, C, H, W] and convert to float
    downsampled = F.interpolate(tensor_permuted, size=(new_H, new_W), mode='bilinear', align_corners=False)
    downsampled_uint8 = downsampled.permute(0, 2, 3, 1).to(torch.uint8)  # Convert back to uint8 and [B, H, W, C]
    return downsampled_uint8

def signal_handler(signum, frame):
    raise KeyboardInterrupt

# @profile_memory_decorator
def scannet_run_sam(cfg: BaseConfig, accelerator: Accelerator, run_train: bool = True, points_per_batch: int = 256, process_batch_size: int = 1, model_type: str = "vit_h", max_masks: int = 128, viz: bool = False):
    signal.signal(signal.SIGINT, signal_handler)

    cfg.dataset.train.num_workers = 2
    cfg.dataset.val.num_workers = 2

    cfg.dataset.train.return_tensorclass = True
    cfg.dataset.val.return_tensorclass = True

    cfg.dataset.train.batch_size = 1
    cfg.dataset.val.batch_size = 1

    cfg.dataset.train.subset_size = None
    cfg.dataset.val.subset_size = None

    cfg.dataset.train.shuffle = False
    cfg.dataset.val.shuffle = False

    set_timing_builtins(False, True)

    model = HQSam(model_type=model_type, process_batch_size=process_batch_size, points_per_batch=points_per_batch, points_per_side=32, output_mode="coco_rle",)
    model.requires_grad_(False)
    
    g = torch.Generator()
    g.manual_seed(int(time.time()))
    
    train: AbstractDataset = hydra.utils.instantiate(cfg.dataset.train, _recursive_=True)(
        cfg=cfg, split=Split.TRAIN, tokenizer=MockTokenizer()
    )
    val: AbstractDataset = hydra.utils.instantiate(cfg.dataset.train, _recursive_=True)(
        cfg=cfg, split=Split.VALIDATION, tokenizer=MockTokenizer()
    )
    
    dataset = train if run_train else val
    dataloader = dataset.get_dataloader(generator=g)
    dataloader, model = accelerator.prepare(dataloader, model)
    
    seg_data_path = SCANNETPP_CUSTOM_DATA_PATH / "segmentation" / 'v2'
    cache_path = get_available_path(seg_data_path, return_scratch_only=True)
    if is_main_process():
        env = lmdb.open(str(cache_path), map_size=1099511627776, readonly=False)  # 1 TB
    accelerator.wait_for_everyone()
    if not is_main_process():
        env = lmdb.open(str(cache_path), map_size=1099511627776, readonly=False)

    batch: InputData
    try:
        for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process(), total=len(dataloader)):
            with get_time_sync(enable=False):
                imgs = downsample(batch.tgt_pixel_values, 0.5).cpu().numpy()
                for b in range(batch.bs):
                    key = batch.metadata['name'][b]

                    with env.begin(write=False) as txn:
                        cursor = txn.cursor()
                        exists = cursor.set_key(key.encode())
                        if exists:
                            log_info(f"Key already exists: {key} Skipping...", main_process_only=False)
                            continue 

                    img = imgs[b]
                    masks = model(img)
                    masks = sorted(masks, key=lambda d: d["area"], reverse=True)
                    masks = masks[:max_masks]  # We only have 77 tokens

                    if viz:
                        show_anns(img, masks, (Path('output') / f'{i}_{b}').with_suffix('.png'))
                        Im(img).save(f'{i}_{b}_orig')
                    
                    metadata = {k:v[0] for k,v in batch.metadata.items() if isinstance(v[0], str)}
                    masks_msgpack = msgpack.packb((metadata, masks), use_bin_type=True)

                    with env.begin(write=True) as txn:
                        txn.put(key.encode('ascii'), masks_msgpack)

                    log_info(f"Processed {key} ({i}/{len(dataloader)})", main_process_only=False)
                    
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected. Cleaning up...", main_process_only=False)
        txn.commit()
        env.close()
        sys.exit(0)
    finally:
        env.close()