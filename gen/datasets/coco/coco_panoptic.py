import autoroot

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from gen import COCO_DATASET_PATH, COCO_TRAIN_ID_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.coco.build import build_transforms
from gen.datasets.coco.coco_utils import _COCO_PANOPTIC_INFORMATION, _COCO_PANOPTIC_THING_LIST, _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID, COCO_CATEGORIES
from gen.datasets.coco.pre_augmentation_transforms import Resize
from gen.datasets.coco.target_transforms import PanopticTargetGenerator, SemanticTargetGenerator
from gen.models.encoders.sam import show_anns
from gen.utils.data_defs import get_one_hot_channels, visualize_input_data
from gen.utils.tokenization_utils import get_tokens

torchvision.disable_beta_transforms_warning()

def one_hot_to_integer_np(one_hot_mask):
    indices = np.argmax(one_hot_mask, axis=-1)
    values = np.max(one_hot_mask, axis=-1)
    return np.where(values > 0, indices, -1)

def process_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    mask_np = mask.numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask_np, kernel, iterations=1) # Erosion to remove thin lines
    dilated = cv2.dilate(eroded, kernel, iterations=1) # Dilation to restore the eroded main objects
    processed_mask = torch.from_numpy(dilated).to(torch.bool)
    return processed_mask

@inherit_parent_args
class CocoPanoptic(AbstractDataset, Dataset):
    """
    Written by Bowen Cheng (bcheng9@illinois.edu)
    COCO panoptic segmentation dataset.
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/data/datasets/coco_panoptic.py
    https://github.com/facebookresearch/detectron2/blob/main/projects/Panoptic-DeepLab/panoptic_deeplab/panoptic_seg.py
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/datasets/prepare_coco_panoptic_trainid.py
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
            root=COCO_DATASET_PATH,
            train_id_root=COCO_TRAIN_ID_PATH,
            tokenizer=None,
            semantic_only=False,
            ignore_stuff_in_offset=False,
            small_instance_area=0,
            small_instance_weight=1,
            augmentation: Optional[Augmentation] = None,
            enable_orig_coco_augmentation: bool = True,
            enable_orig_coco_processing: bool = False,
            object_ignore_threshold: float = 0.2,
            single_return: bool = False,
            top_n_masks_only: Optional[int] = 8,
            load_custom_masks: bool = False,
            num_objects: int = 133,
            postprocess: bool = False,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            resolution=None,
            custom_split=None, # TODO: Needed for hydra
            path=None, # TODO: Needed for hydra
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
        self.single_return = single_return
        if self.single_return:
            augmentation.source_transform = None
            augmentation.target_transform = None
            enable_orig_coco_augmentation = False

        self.train_id_root = train_id_root
        self.coco_split = 'train2017' if self.split == Split.TRAIN else 'val2017'
        self.is_train = self.split == Split.TRAIN
        
        # Only used if enable_orig_coco_processing is True
        self.min_resize_value=641
        self.max_resize_value=641
        self.resize_factor=32
        self.mirror=True
        self.min_scale=0.5
        self.max_scale=2.
        self.scale_step_size=0.25
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        crop_size = (resolution, resolution)
        self.crop_h, self.crop_w = crop_size

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
        self.label_pad_value = (-1, -1, -1)
        self.post_label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 256
        self.label_dtype = np.float32
        self.thing_list = _COCO_PANOPTIC_THING_LIST
        self.enable_orig_coco_augmentation = enable_orig_coco_augmentation
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.object_ignore_threshold = object_ignore_threshold
        self.enable_orig_coco_processing = enable_orig_coco_processing
        self.top_n_masks_only = top_n_masks_only
        self.load_custom_masks = load_custom_masks
        self.num_objects = num_objects
        self.postprocess = postprocess

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

        if self.enable_orig_coco_processing:
            self.pre_augmentation_transform = Resize(self.min_resize_value, self.max_resize_value, self.resize_factor)
            self.transform = build_transforms(self, self.is_train and enable_orig_coco_augmentation)

        if semantic_only:
            self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _COCO_PANOPTIC_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

        if self.load_custom_masks:
            if self.postprocess:
                coco_path_ = os.path.join(self.train_id_root, 'custom_postprocessed_{}.json'.format(self.coco_split))
            else:
                coco_path_ = os.path.join(self.train_id_root, 'custom_{}.json'.format(self.coco_split))
            self.coco = COCO(coco_path_)

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
        if self.single_return:
            index = 0
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

        pad_mask = (label == np.array(self.label_pad_value).reshape(1, 1, 3)).all(axis=-1)
        label[pad_mask] = self.post_label_pad_value

        dataset_dict['image'] = image
        if not self.has_instance:
            dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
            return dataset_dict

        # Generate training target.
        if self.target_transform is not None:
            label_dict = self.target_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        for key in dataset_dict.keys():
            if (key == 'semantic' or key == 'image') and isinstance(dataset_dict[key], np.ndarray):
                dataset_dict[key] = torch.from_numpy(dataset_dict[key].copy())
                if key == 'image':
                    dataset_dict[key] = dataset_dict[key].permute(2, 0, 1) / 255.0
        
        # At this point, we have 133 categories. For some reason the range is 0-132 + 255
        # 0-132 are valid [e.g., 0 is person]. 255 is invalid/ignore. We treat this as background. The convention in this codebase is that the 0th segmentation mask is the background.
        # However, the class category remains unchanged (internally we always do seg_idx + 1 == cls_idx)
        mask_ = ((dataset_dict['semantic'] == 255) & (~pad_mask)).bool()
        dataset_dict['semantic'][mask_] = 0
        rgb = dataset_dict['image']

        if 'instance' in dataset_dict.keys():
            dataset_dict['instance'][mask_] = 0 # We want 0 to be only background. 255 means either background or ignore.
            instance = dataset_dict['instance']
        else:
            instance = dataset_dict['semantic']

        semantic = dataset_dict['semantic']

        # Aside from using rgb, we can represent panoptic labels in terms of
        # semantic_label * label_divisor + instance_label
        lab_divisor = instance.max() + 1
        unique_panoptic = torch.unique(semantic * lab_divisor + instance)
        unique_instance = unique_panoptic % lab_divisor
        unique_semantic = unique_panoptic // lab_divisor

        # Very inefficient but works.
        instance_with_pad_mask = torch.where(
            torch.as_tensor(pad_mask), -torch.ones_like(instance), instance
        )

        if self.load_custom_masks:
            try:
                annIds = self.coco.getAnnIds(imgIds=[int(Path(self.ann_list[index]).stem)])
                anns = self.coco.loadAnns(ids=annIds)
                if self.postprocess:
                    anns = [ann for ann in anns if len(ann['segmentation']) > 0][:self.num_objects]
                else:
                    anns = [ann for ann in anns if len(ann['segmentation']) > 0 and maskUtils.area(ann['segmentation']) > 10][:self.num_objects]

                masks_ = np.stack([self.coco.annToMask(anns[i]) for i in range(len(anns))]).astype(np.bool_).transpose(1, 2, 0)
                instance_with_pad_mask = torch.from_numpy(one_hot_to_integer_np(masks_) + 1) # Instances are 1...., 0 is background
            except Exception as e:
                print(e)
                print(f'Error with {int(Path(self.ann_list[index]).stem)}')

        # -1 is ignore, 0 is background
        source_data, target_data = self.augmentation(
            source_data=Data(image=rgb[None].float(), segmentation=instance_with_pad_mask[None].squeeze(-1).float()),
            target_data=Data(image=rgb[None].float(), segmentation=instance_with_pad_mask[None].squeeze(-1).float()),
        )

        source_data.image = source_data.image.squeeze(0)
        source_data.segmentation = source_data.segmentation.squeeze(0).long()
        target_data.image = target_data.image.squeeze(0)
        target_data.segmentation = target_data.segmentation.squeeze(0).long()

        # We have -1 as invalid so we simply add 1 to all the labels to make it start from 0 and then later remove the 1st channel
        source_pad_mask = source_data.segmentation == -1
        target_pad_mask = target_data.segmentation == -1

        # We handle -1 by adding 1 and then removing the 1st element
        source_bincount = torch.bincount(source_data.segmentation.squeeze(0).long().view(-1) + 1, minlength=self.num_classes + 2)[1:]

        if self.postprocess is False:
            # We remove instance masks that are too small
            valid = source_bincount > (source_data.segmentation.shape[0] * self.object_ignore_threshold)**2
            
            # We optionally only take the largest n masks [if previously valid]
            if self.top_n_masks_only is not None:
                remove_smaller_masks = torch.argsort(source_bincount)[:-self.top_n_masks_only]
                valid[remove_smaller_masks] = False

            # For all pixels that belong to a instance that is too small, we set the pixel to 0 (background)
            too_small_instance_pixels = get_one_hot_channels(source_data.segmentation, indices=(~valid).nonzero()[:, 0]).any(dim=-1)
            source_data.segmentation[too_small_instance_pixels] = 0

            too_small_instance_pixels_ = get_one_hot_channels(target_data.segmentation, indices=(~valid).nonzero()[:, 0]).any(dim=-1)
            target_data.segmentation[too_small_instance_pixels_] = 0
        else:
            valid = source_bincount > 0
            initial_one_hot_background = target_data.segmentation == 0
            one_hot_background = process_mask(initial_one_hot_background, kernel_size=int(target_data.segmentation.shape[0] * 24/512))
            target_data.segmentation[initial_one_hot_background] = -1
            target_data.segmentation[one_hot_background] = 0

        categories = torch.full((valid.shape), fill_value=-1)
        categories[unique_instance] = unique_semantic - 1
        categories[~valid] = -1
        valid = valid[..., 1:]
        categories = categories[..., 1:]
        
        ret = {
            "gen_pad_mask": target_pad_mask,
            "gen_pixel_values": target_data.image,
            "gen_segmentation": target_data.segmentation,
            "disc_pad_mask": source_pad_mask,
            "disc_pixel_values": source_data.image,
            "disc_segmentation": source_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid,
            "categories": categories,
        }

        if source_data.grid is not None: ret["disc_grid"] = source_data.grid.squeeze(0)
        if target_data.grid is not None: ret["gen_grid"] = target_data.grid.squeeze(0)

        return ret

    
    def get_dataset(self):
        return self


def run_sam(dataloader):
    from einops import rearrange

    from gen.models.encoders.sam import HQSam
    from image_utils import Im
    hqsam = HQSam(model_type='vit_b')
    hqsam = hqsam.to('cuda')
    for step, batch in enumerate(dataloader):
        images = rearrange(((batch.gen_pixel_values + 1) / 2) * 255, "b c h w -> b h w c").to(torch.uint8).cpu().detach().numpy()
        for i in range(batch.bs):
            image = images[i]
            torch.cuda.synchronize()
            start_time = time.time()
            masks = hqsam.forward(image)
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            masks = masks[:32]  # We only have 77 tokens
            masks = np.array([masks[i]["segmentation"] for i in range(len(masks))]).transpose(1, 2, 0)
            masks = one_hot_to_integer_np(masks)
            batch.gen_segmentation[i] = torch.from_numpy(masks).long()
            torch.cuda.synchronize()
            print(f'Time taken per image: {time.time() - start_time}')
            show_anns(image, masks, output_path=Path(f"output/{step}_{i}.png"))

        visualize_input_data(batch, name=f'coco_sam_{step}')


    image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').pil
    image = Im(image.crop(((image.size[0]-image.size[1]) // 2, 0, image.size[0] - (image.size[0]-image.size[1]) // 2, image.size[1]))).resize(224, 224).np
    masks = hqsam.forward(image)
    bs = len(masks)
    original = torch.from_numpy(np.array([masks[i]['segmentation'] for i in range(bs)]))
    from ipdb import set_trace; set_trace()
    Im(rearrange(original[:, None].repeat(1, 3, 1, 1) * 1.0, 'b c h w -> b h w c')).save('high_res_mask')

    show_anns(image, masks)

if __name__ == "__main__":
    from transformers import AutoTokenizer

    from image_utils import library_ops
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    from gen.datasets.utils import get_stable_diffusion_transforms
    dataset = CocoPanoptic(
        shuffle=False,
        cfg=None,
        split=Split.VALIDATION,
        num_workers=0,
        batch_size=16,
        tokenizer=tokenizer,
        resolution=256,
        augmentation=Augmentation(
            different_source_target_augmentation=False,
            enable_random_resize_crop=True, 
            enable_horizontal_flip=True,
            source_random_scale_ratio=None, # ((0.8, 1.0), (0.9, 1.1)),
            target_random_scale_ratio=((0.9, 0.9), (0.8, 1.2)),
            enable_rand_augment=False,
            target_normalization=get_stable_diffusion_transforms(resolution=512)
        ),
        single_return=False,
        object_ignore_threshold=0.0,
        top_n_masks_only=100,
        enable_orig_coco_processing=False,
        return_tensorclass=True,
        load_custom_masks=True,
        postprocess=True,
    )

    import time
    start_time = time.time()
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator)

    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        visualize_input_data(batch, name=f'coco_orig_{step}', show_background_foreground_only=True)
        start_time = time.time()

        if step > 1:
            break