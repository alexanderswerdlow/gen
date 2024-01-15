import autoroot

import glob
import itertools
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision
from ipdb import set_trace as st
import PIL
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image

from gen import DEFAULT_PROMPT, IMAGENET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.base_dataset import AbstractDataset, Split
from gen.datasets.utils import get_open_clip_transforms_v2, get_stable_diffusion_transforms

torchvision.disable_beta_transforms_warning()
from ipdb import set_trace as st

DIR = os.path.dirname(os.path.realpath(__file__))
IMAGENET_CLASS_INDEX_PATH = os.path.join(DIR, "imagenet_class_index.json")


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
        augmentation: Optional[Augmentation] = Augmentation(),
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.legacy_transforms = legacy_transforms

        if self.legacy_transforms:
            self.gen_image_transforms = get_stable_diffusion_transforms(resolution)
            self.disc_image_transforms = get_open_clip_transforms_v2()
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
            gen_rgb = self.gen_image_transforms(Image(ret["image"]))
            disc_rgb = self.disc_image_transforms(Image(ret["image"]))
            ret = {
                "gen_pixel_values": gen_rgb,
                "disc_pixel_values": disc_rgb,
                "input_ids": self.tokenizer(
                    DEFAULT_PROMPT, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
            }

        else:
            source_data, target_data = self.augmentation(
                source_data=Data(image=self.to_tensor(ret["image"])[None], image_only=True),
                target_data=Data(image=self.to_tensor(ret["image"])[None], image_only=True),
            )

            ret = {
                "gen_pixel_values": target_data.image.squeeze(0),
                "disc_pixel_values": source_data.image.squeeze(0),
                "input_ids": self.tokenizer(
                    DEFAULT_PROMPT, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
            }

        # We make dummy segmentation maps to make things easier for now
        ret["gen_segmentation"] = torch.ones((ret["gen_pixel_values"].shape[1], ret["gen_pixel_values"].shape[2], 2), dtype=torch.long)
        ret["disc_segmentation"] = torch.ones((ret["disc_pixel_values"].shape[1], ret["disc_pixel_values"].shape[2], 2), dtype=torch.long)

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
        augmentation: Optional[Augmentation] = Augmentation(),
        custom_split=None,
        num_objects=None,
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
        batch_size=1,
        shuffle=True,
        random_subset=None,
        tokenizer=tokenizer,
    )
    dataloader = dataset.get_dataloader()
    for batch in dataloader:
        from image_utils import Im

        st()
        Im((batch["gen_pixel_values"][0] + 1) / 2).save()
