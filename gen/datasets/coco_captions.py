from enum import Enum

import autoroot
import open_clip
import torch
import webdataset as wds
from ipdb import set_trace as st
from torch.utils.data import DataLoader
from torchvision import transforms

from gen.configs import BaseConfig
from gen.datasets.base_dataset import AbstractDataset, Split


class CocoCaptions(AbstractDataset):
    def __init__(self, cfg: BaseConfig, tokenizer, override_text: bool = False, **kwargs):
        super().__init__(cfg)
        self.tokenizer = tokenizer
        self.gen_image_transforms = transforms.Compose([
            transforms.Resize(self.cfg.dataset.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.cfg.dataset.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # TODO: This is super inefficient as we load the entire model just to get the transforms!
        self.disc_image_transforms = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[-1]
        self.override_text = override_text
        if self.override_text:
            print("Overriding text captions with 'A photo of '")

    def get_dataset(self, split: Split):
        pass

    def collate_fn(self, batch):
        pixel_values = torch.stack([example[0] for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        disc_pixel_values = torch.stack([example[1] for example in batch])
        disc_pixel_values = disc_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example[2] for example in batch]).squeeze(1)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "disc_pixel_values": disc_pixel_values}

    def make_sample(self, sample, val=False):
        input_text = 'A photo of ' if sample['txt'] else self.override_text
        inputs = self.tokenizer(input_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return self.gen_image_transforms(sample["jpg"]), self.disc_image_transforms(sample["jpg"]), inputs.input_ids
    
    def get_dataloader(self, split: Enum):
        if split != Split.TRAIN:
            raise NotImplementedError(f"Dataset split {split} not implemented.")
        
        # See: https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md
        dataset = wds.WebDataset('/home/aswerdlow/research/lib/img2dataset/mscoco/{00000..00059}.tar')
        dataset = dataset.shuffle(0).decode("pil").map(self.make_sample).with_length(100)
        loader = DataLoader(dataset, batch_size=self.cfg.dataset.train_batch_size, collate_fn=self.collate_fn, num_workers=self.cfg.dataset.dataloader_num_workers)

        return loader