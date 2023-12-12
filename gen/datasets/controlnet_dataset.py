import random

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from datasets import load_dataset
from torchvision import transforms

from config.configs.configs import BaseConfig

logger = get_logger(__name__)

def make_train_dataset(cfg: BaseConfig, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if cfg.dataset.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset.dataset_name,
            cfg.dataset.dataset_config_name,
            cache_dir=cfg.dataset.cache_dir,
            split=cfg.dataset.dataset_split,
        )
    else:
        if cfg.dataset.train_data_dir is not None:
            dataset = load_dataset(
                cfg.dataset.train_data_dir,
                cache_dir=cfg.dataset.cache_dir,
                split=cfg.dataset.dataset_split,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if cfg.dataset.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = cfg.dataset.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{cfg.dataset.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.dataset.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = cfg.dataset.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{cfg.dataset.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if cfg.dataset.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = cfg.dataset.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{cfg.dataset.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < cfg.dataset.proportion_empty_prompts:
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

    image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.dataset.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.dataset.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.dataset.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.dataset.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if cfg.trainer.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=cfg.trainer.seed).select(range(cfg.trainer.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }