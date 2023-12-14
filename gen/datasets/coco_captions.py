from enum import Enum
import autoroot
from ipdb import set_trace as st
import webdataset as wds
from gen.configs import BaseConfig
from gen.datasets.base_dataset import AbstractDataset, Split
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import open_clip
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
class CocoCaptions(AbstractDataset):
    def __init__(self, cfg: BaseConfig, tokenizer, **kwargs):
        super().__init__(cfg)
        self.tokenizer = tokenizer
        self.disc_image_transforms = transforms.Compose([
            transforms.Resize(self.cfg.dataset.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.cfg.dataset.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.gen_image_transforms = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[-1]

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

        inputs = self.tokenizer(sample['txt'], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return self.gen_image_transforms(sample["jpg"]), self.disc_image_transforms(sample["jpg"]), inputs.input_ids
    
    def get_dataloader(self, split: Enum):
        if split != Split.TRAIN:
            raise NotImplementedError(f"Dataset split {split} not implemented.")
        
        # See: https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md
        dataset = wds.WebDataset('/home/aswerdlow/research/lib/img2dataset/mscoco/{00000..00059}.tar')
        dataset = dataset.shuffle(0).decode("pil").map(self.make_sample).with_length(100)
        loader = DataLoader(dataset, batch_size=self.cfg.dataset.train_batch_size, collate_fn=self.collate_fn, num_workers=self.cfg.dataset.dataloader_num_workers)

        return loader