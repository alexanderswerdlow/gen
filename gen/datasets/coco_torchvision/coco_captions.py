from typing import Optional

import warnings
from enum import Enum
from pathlib import Path

import open_clip
import torch
import webdataset as wds
from ipdb import set_trace as st
from torch.utils.data import DataLoader
from torchvision import transforms

from gen import COCO_CAPTIONS_FILES, DEFAULT_PROMPT
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset

@inherit_parent_args
class CocoCaptions(AbstractDataset):
    def __init__(
            self,
            *,
            tokenizer,
            path: str = COCO_CAPTIONS_FILES,
            resolution: int = 512, 
            override_text: bool = True,
            **kwargs
        ):
        # Note: The super __init__ is handled by inherit_parent_args
        self.allow_shuffle = False
        self.allow_subset = False
        self.tokenizer = tokenizer
        self.path = path
        self.tgt_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # TODO: This is super inefficient as we load the entire model just to get the transforms!
        self.src_image_transforms = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[-1]
        self.override_text = override_text
        if self.override_text:
            warnings.warn(f"Overriding text captions with {DEFAULT_PROMPT}")

    def get_dataset(self):
        # See: https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md
        dataset = wds.WebDataset(self.path)
        if self.shuffle:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode("pil").map(self.make_sample)
        if self.subset_size:
            dataset = dataset.with_length(self.subset_size)
        return dataset

    def collate_fn(self, batch):
        pixel_values = torch.stack([example[0] for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        src_pixel_values = torch.stack([example[1] for example in batch])
        src_pixel_values = src_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example[2] for example in batch]).squeeze(1)
        return {"tgt_pixel_values": pixel_values, "input_ids": input_ids, "src_pixel_values": src_pixel_values}

    def make_sample(self, sample, val=False):
        input_text = DEFAULT_PROMPT if self.override_text else sample['txt']
        inputs = self.tokenizer(input_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return self.tgt_image_transforms(sample["jpg"]), self.src_image_transforms(sample["jpg"]), inputs.input_ids