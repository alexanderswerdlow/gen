from __future__ import annotations

import time
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.utils.data_defs import InputData, visualize_input_data
from gen.utils.decoupled_utils import is_main_process
from image_utils import Im

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig


class MockTokenizer:
    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length

    def __call__(self, prompt, max_length=None, padding="max_length", truncation=True, return_tensors="pt"):
        input_ids = torch.randint(320, 40000, (1, self.model_max_length), dtype=torch.int64).squeeze(0)
        TokenizedOutput = namedtuple('TokenizedOutput', ['input_ids'])
        return TokenizedOutput(input_ids=input_ids)
    
def coco_colormap_viz(batch: InputData):
    pass

def iterate_dataloader(cfg: BaseConfig, accelerator: Accelerator):
    run_train = False

    cfg.dataset.train.return_tensorclass = True
    cfg.dataset.val.return_tensorclass = True

    cfg.dataset.train.subset_size = None
    cfg.dataset.val.subset_size = None

    cfg.dataset.train.shuffle = True
    cfg.dataset.val.shuffle = True
    
    g = torch.Generator()
    g.manual_seed(int(time.time()))
    
    additional_datasets = None
    if run_train:
        train: AbstractDataset = hydra.utils.instantiate(cfg.dataset.train, _recursive_=True)(
            cfg=cfg, split=Split.TRAIN, tokenizer=MockTokenizer()
        )

        if exists(cfg.dataset.additional_train):
            additional_datasets = [instantiate(dataset_cfg)(cfg=cfg, split=Split.TRAIN, tokenizer=MockTokenizer()) for dataset_cfg in cfg.dataset.additional_train]

        dataset = train
    else:
        val: AbstractDataset = hydra.utils.instantiate(cfg.dataset.val, _recursive_=True)(
            cfg=cfg, split=Split.VALIDATION, tokenizer=MockTokenizer()
        )

        if exists(cfg.dataset.additional_val):
            additional_datasets = [instantiate(dataset_cfg)(cfg=cfg, split=Split.VALIDATION, tokenizer=MockTokenizer()) for dataset_cfg in cfg.dataset.additional_val]

        dataset = val

    dataloader = dataset.get_dataloader(generator=g, additional_datasets=additional_datasets)

    batch: InputData
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        names = [f'{batch.metadata["dataset"][i]}_{batch.metadata["name"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        visualize_input_data(batch, names=names, show_overlapping_masks=True, remove_invalid=False, cfg=cfg)
        # print(batch.metadata['dataset'])
        # print(batch.src_pose[:, :3, 3].min(), batch.src_pose[:, :3, 3].max())
        # print('')