import random
from typing import Any, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_dataset
from torchvision import transforms
import open_clip
from gen import DEFAULT_PROMPT
from gen.configs.utils import inherit_parent_args

from gen.datasets.abstract_dataset import AbstractDataset, Split



def collate_fn(examples):
    pixel_values = torch.stack([example["tgt_pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    src_pixel_values = torch.stack([example["src_pixel_values"] for example in examples])
    src_pixel_values = src_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "tgt_pixel_values": pixel_values,
        "src_pixel_values": pixel_values,
        "src_pixel_values": src_pixel_values,
        "input_ids": input_ids,
    }

@inherit_parent_args
class ControlnetDataset(AbstractDataset):
    def __init__(
            self,
            *,
            tokenizer: Optional[Any] = None,
            accelerator: Optional[Any] = None,
            resolution: int = 512, 
            dataset_class: str = 'controlnet_dataset',
            dataset_name: Optional[str] = "fusing/fill50k",
            dataset_config_name: Optional[str] = None,
            dataset_split: Optional[tuple[str]] = None,
            train_data_dir: Optional[str] = None,
            image_column: Optional[str] = "image",
            conditioning_image_column: Optional[str] = "conditioning_image",
            caption_column: Optional[str] = "text",
            proportion_empty_prompts: float = 0,
            validation_prompt: Optional[str] = None,
            validation_image: Optional[tuple[str]] = None,
            cache_dir: Optional[str] = None,
            override_text: bool = True,
            **kwargs
        ):

        # Note: The super __init__ is handled by inherit_parent_args
        self.allow_subset = False
        self.allow_shuffle = False
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.dataset_split = dataset_split
        self.train_data_dir = train_data_dir
        self.image_column = image_column
        self.conditioning_image_column = conditioning_image_column
        self.caption_column = caption_column
        self.proportion_empty_prompts = proportion_empty_prompts
        self.validation_prompt = validation_prompt
        self.validation_image = validation_image
        self.cache_dir = cache_dir
        self.override_text = override_text
        self.src_image_transforms = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[-1]
        self.tgt_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def make_train_dataset(self, tokenizer, accelerator):
        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                cache_dir=self.cache_dir,
                split=self.dataset_split,
            )
        else:
            if self.train_data_dir is not None:
                dataset = load_dataset(
                    self.train_data_dir,
                    cache_dir=self.cache_dir,
                    split=self.dataset_split,
                )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        if self.image_column is None:
            image_column = column_names[0]
            log(f"image column defaulting to {image_column}")
        else:
            image_column = self.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{self.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            log(f"caption column defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.conditioning_image_column is None:
            conditioning_image_column = column_names[2]
            log(f"conditioning image column defaulting to {conditioning_image_column}")
        else:
            conditioning_image_column = self.conditioning_image_column
            if conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{self.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if self.override_text:
                    captions.append(DEFAULT_PROMPT)
                elif random.random() < self.proportion_empty_prompts:
                    captions.append("")
                elif isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            images = [self.tgt_image_transforms(image) for image in images]

            conditioning_images = [image.convert("RGB") for image in examples[image_column]]
            conditioning_images = [self.src_image_transforms(image) for image in conditioning_images]

            examples["tgt_pixel_values"] = images
            examples["src_pixel_values"] = conditioning_images
            examples["input_ids"] = tokenize_captions(examples)

            return examples

        with accelerator.main_process_first():
            if self.shuffle:
                dataset["train"] = dataset["train"].shuffle(seed=self.cfg.trainer.seed)
            if self.subset_size is not None:
                dataset["train"] = dataset["train"].select(range(self.subset_size))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        return train_dataset

    def get_dataset(self):
        return self.make_train_dataset(self.tokenizer, self.accelerator)

    def collate_fn(self, batch):
        return collate_fn(batch)