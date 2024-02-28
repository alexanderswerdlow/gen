import autoroot

import json
import os
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from gen import COCO_DATASET_PATH, COCO_TRAIN_ID_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.coco.build import build_transforms
from gen.datasets.coco.coco_utils import _COCO_PANOPTIC_INFORMATION, _COCO_PANOPTIC_THING_LIST, _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID, COCO_CATEGORIES
from gen.datasets.coco.pre_augmentation_transforms import Resize
from gen.datasets.coco.target_transforms import PanopticTargetGenerator, SemanticTargetGenerator
from gen.utils.tokenization_utils import get_tokens

torchvision.disable_beta_transforms_warning()

@inherit_parent_args
class CocoPanoptic(AbstractDataset, Dataset):
    """
    Written by Bowen Cheng (bcheng9@illinois.edu)
    COCO panoptic segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
        semantic_only: Bool, only use semantic segmentation label.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
    """
    def __init__(
            self,
            *,
            resolution,
            root=COCO_DATASET_PATH,
            train_id_root=COCO_TRAIN_ID_PATH,
            min_resize_value=641,
            max_resize_value=641,
            resize_factor=32,
            mirror=True,
            min_scale=0.5,
            max_scale=2.,
            scale_step_size=0.25,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            semantic_only=False,
            ignore_stuff_in_offset=False,
            small_instance_area=0,
            small_instance_weight=1,
            augmentation: Optional[Augmentation] = Augmentation(),
            enable_train_augmentation: bool = True,
            tokenizer=None,
            object_ignore_threshold: float = 0.1,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            custom_split=None, # TODO: Needed for hydra
            path=None, # TODO: Needed for hydra
            num_objects=None, # TODO: Needed for hydra
            num_frames=None, # TODO: Needed for hydra
            num_cameras=None, # TODO: Needed for hydra
            multi_camera_format=None, # TODO: Needed for hydra
            subset=None, # TODO: Needed for hydra
            fake_return_n=None, # TODO: Needed for hydra
            use_single_mask=None,# TODO: Needed for hydra
            cache_in_memory=None, # TODO: Needed for hydra
            cache_instances_in_memory= None, # TODO: Needed for hydra
            num_subset=None, # TODO: Needed for hydra
            return_multiple_frames=None, # TODO: Needed for hydra
            **kwargs
        ):

        # We follow https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html which is different 
        # from the original coco structure: https://github.com/cocodataset/cocoapi/blob/master/README.txt
        
        self.root = root
        self.train_id_root = train_id_root
        self.coco_split = 'train2017' if self.split == Split.TRAIN else 'val2017'
        self.is_train = self.split == Split.TRAIN

        crop_size = (resolution, resolution)
        self.crop_h, self.crop_w = crop_size

        self.mirror = mirror
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

        self.mean = mean
        self.std = std

        self.pad_value = tuple([int(v * 255) for v in self.mean])

        # ======== override the following fields ========
        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.label_dtype = 'uint8'

        # list of image filename (required)
        self.img_list = []
        # list of label filename (required)
        self.ann_list = []
        # list of instance dictionary (optional)
        self.ins_list = []

        self.has_instance = False
        self.label_divisor = 1000

        self.raw_label_transform = None
        self.pre_augmentation_transform = None
        self.transform = None
        self.target_transform = None

        assert self.coco_split in _COCO_PANOPTIC_INFORMATION.splits_to_sizes.keys()

        self.num_classes = _COCO_PANOPTIC_INFORMATION.num_classes
        self.ignore_label = _COCO_PANOPTIC_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 256
        self.label_dtype = np.float32
        self.thing_list = _COCO_PANOPTIC_THING_LIST
        self.enable_train_augmentation = enable_train_augmentation
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.object_ignore_threshold = object_ignore_threshold
        assert self.augmentation.kornia_augmentations_enabled() is False

        # Get image and annotation list.
        if 'test' in self.coco_split:
            self.img_list = []
            self.ann_list = None
            self.ins_list = None
            json_filename = os.path.join(self.root, 'annotations', 'image_info_{}.json'.format(self.coco_split))
            dataset = json.load(open(json_filename))
            for img in dataset['images']:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(self.root, 'test2017', img_file_name))
        else:
            self.img_list = []
            self.ann_list = []
            self.ins_list = []
            json_filename = os.path.join(self.train_id_root, 'panoptic_{}_trainId.json'.format(self.coco_split))
            dataset = json.load(open(json_filename))
            # First sort by image id.
            images = sorted(dataset['images'], key=lambda i: i['id'])
            annotations = sorted(dataset['annotations'], key=lambda i: i['image_id'])
            for img in images:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(self.root, self.coco_split, img_file_name))
            for ann in annotations:
                ann_file_name = ann['file_name']
                self.ann_list.append(os.path.join(
                    self.root, 'panoptic_{}'.format(self.coco_split), ann_file_name))
                self.ins_list.append(ann['segments_info'])

        assert len(self) == _COCO_PANOPTIC_INFORMATION.splits_to_sizes[self.coco_split]

        self.pre_augmentation_transform = Resize(min_resize_value, max_resize_value, resize_factor)
        self.transform = build_transforms(self, self.is_train and augmentation)
        if semantic_only:
            self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _COCO_PANOPTIC_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    @staticmethod
    def train_id_to_eval_id():
        return _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in COCO panoptic benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i, color in enumerate(COCO_CATEGORIES):
            colormap[i+1] = color['color'] # see Line 300, ignored label is mapped to index 0
        return colormap

    @staticmethod
    def read_image(file_name, format=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    @staticmethod
    def read_label(file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        label = Image.open(file_name)
        return np.asarray(label, dtype=dtype)
    
    @staticmethod
    def train_id_to_eval_id():
        return None
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        image = self.read_image(self.img_list[index], 'RGB')
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype)
        else:
            label = None
        raw_label = label.copy()
        if self.raw_label_transform is not None:
            raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
        if not self.is_train:
            # Do not save this during training
            dataset_dict['raw_label'] = raw_label
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        dataset_dict['name'] = os.path.splitext(os.path.basename(self.ann_list[index]))[0]

        # Resize and pad image to the same size before data augmentation.
        if self.pre_augmentation_transform is not None:
            image, label = self.pre_augmentation_transform(image, label)
            size = image.shape
            dataset_dict['size'] = np.array(size)
        else:
            dataset_dict['size'] = dataset_dict['raw_size']

        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label)

        dataset_dict['image'] = image
        if not self.has_instance:
            dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
            return dataset_dict

        # Generate training target.
        if self.target_transform is not None:
            label_dict = self.target_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        # At this point, we have 133 categories. For some reason the range is 0-132 + 255
        # 0-132 are valid [e.g., 0 is person]. 255 is invalid/ignore. We treat this as background. The convention in this codebase is that the 0th segmentation mask is the background.
        # However, the class category remains unchanged (internally we always do seg_idx + 1 == cls_idx)
        mask_ = dataset_dict['semantic'] == 255
        dataset_dict['semantic'] += 1
        dataset_dict['semantic'][mask_] = 0
        
        rgb = dataset_dict['image']
        instance = dataset_dict['semantic']

        # Very inefficient but works.
        source_data, target_data = self.augmentation(
            source_data=Data(image=rgb[None].float(), segmentation=instance[None].squeeze(-1).float()),
            target_data=Data(image=rgb[None].float(), segmentation=instance[None].squeeze(-1).float()),
        )

        # We have -1 as invalid so we simply add 1 to all the labels to make it start from 0 and then later remove the 1st channel
        source_data.image = source_data.image.squeeze(0)
        source_data.segmentation = torch.nn.functional.one_hot(source_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]
        target_data.image = target_data.image.squeeze(0)
        target_data.segmentation = torch.nn.functional.one_hot(target_data.segmentation.squeeze(0).long() + 1, num_classes=self.num_classes + 2)[..., 1:]

        valid = (torch.sum(source_data.segmentation[..., 1:], dim=[0, 1]) > (source_data.segmentation.shape[0] * self.object_ignore_threshold)**2)
        categories = torch.full((valid.shape), fill_value=-1)
        categories[valid] = torch.arange(valid.shape[0])[valid]
        ret = {
            "gen_pixel_values": target_data.image,
            "gen_segmentation": target_data.segmentation,
            "disc_pixel_values": source_data.image,
            "disc_segmentation": source_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid,
            "categories": categories,
        }

        return ret

    
    def get_dataset(self):
        return self


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = CocoPanoptic(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        max_resize_value=512,
        min_resize_value=512,
        resize_factor=None,
        crop_size=(512, 512),
        scale_step_size=0.1,
        ignore_stuff_in_offset=True,
        tokenizer=tokenizer,
        resolution=512,
        augmentation=Augmentation(minimal_source_augmentation=True, enable_crop=False, enable_horizontal_flip=False),
    )

    import time
    start_time = time.time()
    dataloader = dataset.get_dataloader()
    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        start_time = time.time()
        from image_utils import Im, get_layered_image_from_binary_mask
        for b in range(batch['gen_pixel_values'].shape[0]):            
            gen_ = Im.concat_vertical(Im((batch['gen_pixel_values'][b] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['gen_segmentation'][b].squeeze(0))))
            disc_ = Im.concat_vertical(Im((batch['disc_pixel_values'][b] + 1) / 2), Im(get_layered_image_from_binary_mask(batch['disc_segmentation'][b].squeeze(0))))
            print(batch['gen_segmentation'].sum() / batch['gen_segmentation'][b, ..., 0].numel(), batch['disc_segmentation'].sum() / batch['disc_segmentation'][b, ..., 0].numel())
            Im.concat_horizontal(gen_, disc_).save(f'coco_{step}_{b}.png')

        if step > 1:
            break