from __future__ import annotations

import autoroot

from pathlib import Path
import time
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.run_dataloader import MockTokenizer
from gen.datasets.scannetpp.run_sam_dask import get_all_gt_images
from gen.utils.data_defs import InputData, visualize_input_data
from gen.utils.decoupled_utils import breakpoint_on_error, get_time_sync, is_main_process, profile_memory_decorator, set_timing_builtins, show_memory_usage
from image_utils import Im
from gen.models.encoders.sam import HQSam
import numpy as np

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

def coco_decode_rle(compressed_rle) -> np.ndarray:
    from pycocotools import mask as mask_utils  # type: ignore
    from typing import Any, Dict
    import numpy as np

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

    Im(np.stack([ann['segmentation'] for ann in sorted_anns])[..., None]).scale(0.5).grid(pad_value=0.5).save(output_path.parent / f"{output_path.stem}_masks.png")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    

def scannet_run_sam():
    set_timing_builtins(False, True)

    params = (
        dict(points_per_side=32, stability_score_thresh=0.98, pred_iou_thresh=0.96, box_nms_thresh=0.6),
        dict(points_per_side=32)
    )

    for j, _param in enumerate(params):
        model = HQSam(model_type="vit_h", process_batch_size=1, points_per_batch=128, output_mode="coco_rle",  **_param)
        model.requires_grad_(False)
        model = model.cuda()
        
        g = torch.Generator()
        g.manual_seed(int(time.time()))
        max_masks = 256
        filenames = get_all_gt_images("a")

        batch: InputData
        for i, filename in tqdm(enumerate(filenames), leave=False, disable=not is_main_process()):
            img = Im.open(filename).scale(0.5).np
            masks = model(img)
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            masks = masks[:max_masks]
            show_anns(img, masks, (Path('output_sam') / f'{j}_{i}').with_suffix('.png'))
            if i == 10:
                break

if __name__ == "__main__":
    with breakpoint_on_error():
        scannet_run_sam()