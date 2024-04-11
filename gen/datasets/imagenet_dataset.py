import glob
import itertools
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional

import PIL
import torch
import torchvision
from ipdb import set_trace as st
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision.tv_tensors import Image

from gen import DEFAULT_PROMPT, IMAGENET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.utils import (get_open_clip_transforms_v2,
                                get_stable_diffusion_transforms)
from gen.utils.logging_utils import log_info, log_warn

torchvision.disable_beta_transforms_warning()
from ipdb import set_trace as st

DIR = os.path.dirname(os.path.realpath(__file__))
IMAGENET_CLASS_INDEX_PATH = os.path.join(DIR, "imagenet_class_index.json")
from gen.utils.tokenization_utils import get_tokens

class ImageNetBase(VisionDataset):
    def __init__(
        self,
        root: str,
    ):
        super().__init__(root)
        self.classes = None
        self.class_names = []

        glob_path = self._set_glob_path()
        if isinstance(glob_path, list):
            self._images = sorted(itertools.chain.from_iterable(glob.glob(path) for path in glob_path))
        else:
            self._images = sorted(glob.glob(glob_path))

        print(f"Found {len(self._images)} images given glob path: {glob_path}")

        self.do_objectnet = False

        # use a fixed shuflled order
        class_name = self.__class__.__name__

        # Find class index mappings
        class_index_mapping = json.load(open(IMAGENET_CLASS_INDEX_PATH, "r"))
        new_class_index_mapping = {}
        index = 0
        for key, val in class_index_mapping.items():
            new_class_index_mapping[val[0]] = [key, val[1]]
            assert int(key) == index
            self.class_names.append(val[1])
            index += 1

        self.class_index_mapping = new_class_index_mapping

        self.classes = [int(new_class_index_mapping[val.split("/")[-1]][0]) for val in glob.glob(self.root + "/n*")]
        self.classes.sort()

    def _set_glob_path(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        filepath = self._images[index]
        class_index = self.class_index_mapping[filepath.split("/")[-2]][0]
        class_index = int(class_index)
        return {
            "image": PIL.Image.open(filepath).convert("RGB"),
            "class_idx": class_index,
            "filepath": filepath,
            "index": index,
        }


class ImageNetDataset(ImageNetBase):
    def __init__(
        self,
        root: str,
        tokenizer: Optional[Any] = None,
        resolution: int = 512,
        override_text: bool = True,
        legacy_transforms: bool = False,
        augmentation: Optional[Augmentation] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.legacy_transforms = legacy_transforms

        if self.legacy_transforms:
            self.tgt_image_transforms = get_stable_diffusion_transforms(resolution)
            self.src_image_transforms = get_open_clip_transforms_v2()
        else:
            self.augmentation = augmentation

            import torchvision.transforms.v2 as transforms

            self.to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

        self.override_text = override_text
        if self.override_text:
            warnings.warn(f"Overriding text captions with {DEFAULT_PROMPT}")

        super().__init__(root)

    def _set_glob_path(self):
        return os.path.join(self.root, "*/*.JPEG")

    def __getitem__(self, index):
        ret = super().__getitem__(index)

        if self.legacy_transforms:
            tgt_rgb = self.tgt_image_transforms(Image(ret["image"]))
            src_rgb = self.src_image_transforms(Image(ret["image"]))
            ret = {
                "tgt_pixel_values": tgt_rgb,
                "src_pixel_values": src_rgb,
                "input_ids": get_tokens(self.tokenizer),
            }

        else:
            src_data, tgt_data = self.augmentation(
                src_data=Data(image=self.to_tensor(ret["image"])[None], image_only=True),
                tgt_data=Data(image=self.to_tensor(ret["image"])[None], image_only=True),
            )

            ret = {
                "tgt_pixel_values": tgt_data.image.squeeze(0),
                "src_pixel_values": src_data.image.squeeze(0),
                "input_ids": get_tokens(self.tokenizer),
            }

        # We make dummy segmentation maps to make things easier for now
        ret["tgt_segmentation"] = torch.ones((ret["tgt_pixel_values"].shape[1], ret["tgt_pixel_values"].shape[2], 1), dtype=torch.long)
        ret["src_segmentation"] = torch.ones((ret["src_pixel_values"].shape[1], ret["src_pixel_values"].shape[2], 1), dtype=torch.long)

        return ret


@inherit_parent_args
class ImageNetCustomDataset(AbstractDataset):
    def __init__(
        self,
        *,
        path: Path = IMAGENET_PATH,
        tokenizer: Optional[Any] = None,
        resolution: int = 512,
        override_text: bool = True,
        legacy_transforms: bool = False,
        augmentation: Optional[Augmentation] = None,
        custom_split=None,
        num_objects=-1,
        subset=None,
        **kwargs,
    ):
        # Note: The super __init__ is handled by inherit_parent_args
        local_split = "train" if self.split == Split.TRAIN else "val"
        root_dir = path / local_split
        self.dataset = ImageNetDataset(
            root=str(root_dir),
            tokenizer=tokenizer,
            resolution=resolution,
            override_text=override_text,
            legacy_transforms=legacy_transforms,
            augmentation=augmentation,
            **kwargs,
        )

    def get_dataset(self):
        log_info(f"Returning ImageNet dataset from {self.dataset.root}")
        return self.dataset

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = ImageNetCustomDataset(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=10,
        shuffle=True,
        subset_size=None,
        tokenizer=tokenizer,
        augmentation=Augmentation(enable_rand_augment=True),
    )
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        from image_utils import Im

        tgt_ = Im((batch["tgt_pixel_values"] + 1) / 2)
        src_ = Im(batch["src_pixel_values"]).denormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        st()
