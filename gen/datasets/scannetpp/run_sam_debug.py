from __future__ import annotations

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
from gen.utils.data_defs import InputData, visualize_input_data
from gen.utils.decoupled_utils import get_time_sync, is_main_process, profile_memory_decorator, set_timing_builtins, show_memory_usage
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

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    
# @profile_memory_decorator
def scannet_run_sam(cfg: BaseConfig, accelerator: Accelerator):
    cfg.dataset.train.num_workers = 0
    cfg.dataset.val.num_workers = 0

    cfg.dataset.train.return_tensorclass = True
    cfg.dataset.val.return_tensorclass = True

    cfg.dataset.train.batch_size = 2
    cfg.dataset.val.batch_size = 2

    cfg.dataset.train.subset_size = None
    cfg.dataset.val.subset_size = None

    cfg.dataset.train.shuffle = True
    cfg.dataset.val.shuffle = True

    set_timing_builtins(False, True)

    model = HQSam(model_type="vit_b", points_per_side=28, process_batch_size=2, points_per_batch=784, output_mode="coco_rle")
    model.requires_grad_(False)
    
    g = torch.Generator()
    g.manual_seed(int(time.time()))
    
    train: AbstractDataset = hydra.utils.instantiate(cfg.dataset.train, _recursive_=True)(
        cfg=cfg, split=Split.TRAIN, tokenizer=MockTokenizer()
    )
    val: AbstractDataset = hydra.utils.instantiate(cfg.dataset.train, _recursive_=True)(
        cfg=cfg, split=Split.VALIDATION, tokenizer=MockTokenizer()
    )
    dataset = train
    dataloader = dataset.get_dataloader(generator=g)
    dataloader, model = accelerator.prepare(dataloader, model)

    max_masks = 36

    batch: InputData
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        with get_time_sync(enable=False):
            for b in range(batch.bs):
                start_timing("Forward Pass")
                img = Im(batch.tgt_pixel_values[b].cpu().numpy()).scale(0.5).np
                masks = model(img)
                end_timing()
                masks = sorted(masks, key=lambda d: d["area"], reverse=True)
                masks = masks[:max_masks]  # We only have 77 tokens
                print(len(masks))
                show_anns(img, masks, (Path('output') / f'{i}_{b}').with_suffix('.png'))
                Im(img).save(f'{i}_{b}_orig')
        if i == 10:
            break