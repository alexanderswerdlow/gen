import autoroot

import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
from einops import rearrange
from image_utils.im import Im
from image_utils.standalone_image_utils import integer_to_color
import numpy as np
import torch
import torchvision
from joblib import Memory
from PIL import Image, ImageOps
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from gen import COCO_CUSTOM_PATH, COCO_DATASET_PATH, SCRATCH_CACHE_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.datasets.coco.build import build_transforms
from gen.datasets.coco.coco_utils import _COCO_PANOPTIC_INFORMATION, _COCO_PANOPTIC_THING_LIST, _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID, COCO_CATEGORIES
from gen.datasets.coco.pre_augmentation_transforms import Resize
from gen.datasets.coco.target_transforms import PanopticTargetGenerator, SemanticTargetGenerator
from gen.models.encoders.sam import show_anns
from gen.utils.data_defs import get_one_hot_channels, one_hot_to_integer, visualize_input_data
from gen.utils.logging_utils import log_error, log_info, log_warn
from gen.utils.tokenization_utils import get_tokens

torchvision.disable_beta_transforms_warning()

# memory = Memory(SCRATCH_CACHE_PATH, verbose=0)
# @memory.cache(ignore=["coco"])
def process(coco_split, coco, erode_dialate_preprocessed_masks, num_masks, num_overlapping_masks, image_id):
    annIds = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ids=annIds)
    anns = [(maskUtils.area(ann['segmentation']), ann) for ann in anns if len(ann['segmentation']) > 0]
    anns = sorted(anns, key=lambda d: d[0], reverse=True)
    anns = [ann[1] for ann in anns][:num_masks]

    if len(anns) == 0:
        log_warn(f"Image {image_id} has no annotations")
        return None
    else:
        masks = torch.from_numpy(np.stack([coco.annToMask(anns[i]) for i in range(len(anns))])).bool().permute(1, 2, 0)

        maximum_mask_threshold, min_coverage_threshold = 0.9, 0.4
        num_pixels = masks[..., 0].numel()

        remove_large_masks = masks.sum(dim=[0, 1]) < (num_pixels * maximum_mask_threshold)
        masks = masks[..., remove_large_masks]

        mask_coverage = torch.sum(torch.sum(masks, dim=-1) >= 1) / (num_pixels)

        if mask_coverage < min_coverage_threshold:
            log_warn(f"Mask coverage {mask_coverage:.2f} is less than {min_coverage_threshold} for image {image_id}")
            return None
        
        instance_with_pad_mask = one_hot_to_integer(masks, num_overlapping_masks, assert_safe=False).permute(2, 0, 1) # Instances are 1...., 0 is background

    if erode_dialate_preprocessed_masks:
        initial_tgt_one_hot_background = instance_with_pad_mask == 255
        one_hot_background = process_mask(initial_tgt_one_hot_background, kernel_size=int(instance_with_pad_mask.shape[0] * 18/512))
        instance_with_pad_mask[initial_tgt_one_hot_background] = -1
        instance_with_pad_mask[one_hot_background] = 0
        raise NotImplementedError
    
    return instance_with_pad_mask

def process_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    mask_np = mask.numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask_np, kernel, iterations=1) # Erosion to remove thin lines
    dilated = cv2.dilate(eroded, kernel, iterations=1) # Dilation to restore the eroded main objects
    processed_mask = torch.from_numpy(dilated).to(torch.bool)
    return processed_mask

def new_init(self, annotation_file=None):
    """
    Constructor of Microsoft COCO helper class for reading and visualizing annotations.
    :param annotation_file (str): location of annotation file
    :param image_folder (str): location to the folder that hosts images.
    :return:
    """
    # load dataset
    self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
    self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
    if not annotation_file == None:
        print('loading annotations into memory...')
        tic = time.time()
        if isinstance(annotation_file, dict):
            dataset = annotation_file
        else:
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()


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
            custom_data_root=COCO_CUSTOM_PATH,
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
            use_preprocessed_masks: bool = False,
            preprocessed_mask_type: Optional[str] = None,
            erode_dialate_preprocessed_masks: bool = False,
            num_overlapping_masks: int = 1,
            merge_with_background: bool = False,
            scratch_only: bool = True,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            num_objects=None,
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
        self.scratch_only = scratch_only
        if self.single_return:
            augmentation.src_transform = None
            augmentation.tgt_transform = None
            enable_orig_coco_augmentation = False

        self.custom_data_root = custom_data_root
        self.coco_split = 'train2017' if self.split == Split.TRAIN else 'val2017'
        self.is_train = self.split == Split.TRAIN
        
        # Only used if enable_orig_coco_processing is True
        self.min_resize_value = 641
        self.max_resize_value = 641
        self.resize_factor = 32
        self.mirror = True
        self.min_scale = 0.5
        self.max_scale = 2.
        self.scale_step_size=0.25
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        crop_size = (self.min_resize_value, self.min_resize_value)
        self.crop_h, self.crop_w = crop_size
        self.pad_value = tuple([int(v * 255) for v in self.mean])
        # End

        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.label_dtype = 'uint8'
        self.img_list = [] # list of image filename (required)
        self.ann_list = [] # list of label filename (required)
        self.ins_list = [] # list of instance dictionary (optional)
        self.raw_label_transform = None
        self.pre_augmentation_transform = None
        self.transform = None
        self.tgt_transform = None
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
        self.use_preprocessed_masks = use_preprocessed_masks
        self.preprocessed_mask_type = preprocessed_mask_type
        self.erode_dialate_preprocessed_masks = erode_dialate_preprocessed_masks
        self.num_overlapping_masks = num_overlapping_masks
        self.merge_with_background = merge_with_background

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
            json_filename = os.path.join(self.custom_data_root, 'panoptic_{}_trainId.json'.format(self.coco_split))
            dataset = json.load(open(json_filename))
            # First sort by image id.
            images = sorted(dataset['images'], key=lambda i: i['id'])
            annotations = sorted(dataset['annotations'], key=lambda i: i['image_id'])
            for img in images:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(self.root, self.coco_split, img_file_name))
            for ann in annotations:
                ann_file_name = ann['file_name']
                self.ann_list.append(os.path.join(self.root, 'panoptic_{}'.format(self.coco_split), ann_file_name))
                self.ins_list.append(ann['segments_info'])

        assert len(self) == _COCO_PANOPTIC_INFORMATION.splits_to_sizes[self.coco_split]

        if self.enable_orig_coco_processing:
            self.pre_augmentation_transform = Resize(self.min_resize_value, self.max_resize_value, self.resize_factor)
            self.transform = build_transforms(self, self.is_train and enable_orig_coco_augmentation)

        if semantic_only:
            self.tgt_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.tgt_transform = PanopticTargetGenerator(
                self.ignore_label, self.rgb2id, _COCO_PANOPTIC_THING_LIST, sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                small_instance_area=small_instance_area, small_instance_weight=small_instance_weight
            )
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id) # Generates semantic label for evaluation.

        if self.preprocessed_mask_type is not None:
            self.custom_coco_json_path = Path(self.custom_data_root) / f'{self.preprocessed_mask_type}_{self.coco_split}.json'
            if self.custom_coco_json_path.with_suffix('.json.gz').exists(): # We want to load the gzipped version if it exists
                self.custom_coco_json_path = self.custom_coco_json_path.with_suffix('.json.gz')
                with gzip.open(self.custom_coco_json_path, 'rt') as file:
                    annotation_file = json.load(file)
            else:
                annotation_file = self.custom_coco_json_path
                log_warn(f'Could not find gzipped version of {self.custom_coco_json_path}')

            log_info(f'Loading custom masks from {self.custom_coco_json_path}')
            COCO.__init__ = new_init
            self.coco = COCO(annotation_file=annotation_file)

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
    
    def get_coco_label(self, index):
        dataset_dict = {}
        if self.ann_list is not None:
            assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype)
        else:
            label = None

        raw_label = label.copy()
        if self.raw_label_transform is not None:
            raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']

        dataset_dict['name'] = os.path.splitext(os.path.basename(self.ann_list[index]))[0]

        # Resize and pad image to the same size before data augmentation.
        if self.pre_augmentation_transform is not None:
            image, label = self.pre_augmentation_transform(image, label)
            size = image.shape
            dataset_dict['size'] = np.array(size)

        if self.transform is not None: # Apply data augmentation.
            image, label = self.transform(image, label)

        pad_mask = (label == np.array(self.label_pad_value).reshape(1, 1, 3)).all(axis=-1)
        label[pad_mask] = self.post_label_pad_value

        if self.pre_augmentation_transform is not None or  self.transform is not None:
            dataset_dict['image'] = image
            
        if not self.has_instance:
            dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
            return dataset_dict

        # Generate training target.
        if self.tgt_transform is not None:
            label_dict = self.tgt_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        for key in dataset_dict.keys():
            if key == 'semantic' and isinstance(dataset_dict[key], np.ndarray):
                dataset_dict[key] = torch.from_numpy(dataset_dict[key].copy())
        
        # At this point, we have 133 categories. For some reason the range is 0-132 + 255
        # 0-132 are valid [e.g., 0 is person]. 255 is invalid/ignore. We treat this as background. The convention in this codebase is that the 0th segmentation mask is the background.
        # However, the class category remains unchanged (internally we always do seg_idx + 1 == cls_idx)
                
        # mask_ = ((dataset_dict['semantic'] == 255) & (~pad_mask)).bool()
        # dataset_dict['semantic'][mask_] = 0
        # dataset_dict['instance'][mask_] = 0 # We want 0 to be only background. 255 means either background or ignore.

        instance = dataset_dict.get('instance', dataset_dict['semantic'])
        semantic = dataset_dict['semantic']

        # Im(integer_to_color((instance == 0).long())).save(f'instance_{index}')
        # Im(integer_to_color((semantic == 0).long())).save(f'semantic_{index}')

        # Aside from using rgb, we can represent panoptic labels in terms of
        # semantic_label * label_divisor + instance_label
        lab_divisor = instance.max() + 1
        unique_panoptic = torch.unique(semantic * lab_divisor + instance)
        unique_instance = unique_panoptic % lab_divisor
        unique_semantic = unique_panoptic // lab_divisor

        instance[instance == 0] = -1

        # Very inefficient but works.
        instance_with_pad_mask = torch.where(
            torch.as_tensor(pad_mask), -torch.ones_like(instance), instance
        )

        return instance_with_pad_mask, unique_instance, unique_semantic

    def __getitem__(self, index):
        if self.single_return:
            index = 2186
        
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        rgb = self.read_image(self.img_list[index], 'RGB').copy()
        rgb = torch.from_numpy(rgb).permute(2, 0, 1) / 255.0

        num_masks = min(self.top_n_masks_only, self.num_classes)
        image_id = int(Path(self.ann_list[index]).stem)
        if self.use_preprocessed_masks:
            try:
                image_id = int(Path(self.ann_list[index]).stem)
                instance_with_pad_mask = process(self.coco_split, self.coco, self.erode_dialate_preprocessed_masks, num_masks, self.num_overlapping_masks, image_id)
                if instance_with_pad_mask is None:
                    raise Exception(f'No masks found for image_id {image_id}')
                
            except Exception as e:
                log_error(e)
                return self.__getitem__((index + 1) % len(self)) # Very hacky but might save us in case of an error with a single instance.
        else:
            instance_with_pad_mask, unique_instance, unique_semantic = self.get_coco_label(index)

        # -1 is ignore, 0 is background
        src_data, tgt_data = self.augmentation(
            src_data=Data(image=rgb[None].float(), segmentation=instance_with_pad_mask[None].float()),
            tgt_data=Data(image=rgb[None].float(), segmentation=instance_with_pad_mask[None].float()),
            use_keypoints=False
        )

        def process_data(data_: Data):
            data_.image = data_.image.squeeze(0)
            data_.segmentation = rearrange(data_.segmentation, "() c h w -> h w c")
            data_.segmentation[data_.segmentation >= self.top_n_masks_only] = 0 if self.merge_with_background else 255
            assert data_.segmentation.max() <= 255
            data_.segmentation[data_.segmentation == -1] = 255
            return data_

        src_data = process_data(src_data)
        tgt_data = process_data(tgt_data)

        pixels = src_data.segmentation.squeeze(0).long().contiguous().view(-1)
        src_bincount = torch.bincount(pixels[(pixels < 255) & (pixels >= 0)], minlength=num_masks + 1)

        valid = src_bincount > (src_data.segmentation.shape[0] * self.object_ignore_threshold)**2 # We remove instance masks that are too small

        if self.use_preprocessed_masks is False:
            # We optionally only take the largest n masks [if previously valid]
            if self.top_n_masks_only is not None:
                remove_smaller_masks = torch.argsort(src_bincount)[:-self.top_n_masks_only]
                valid[remove_smaller_masks] = False

            if self.merge_with_background:
                # For all pixels that belong to a instance that is too small, we set the pixel to 0 (background)
                too_small_instance_pixels = get_one_hot_channels(src_data.segmentation, indices=(~valid).nonzero()[:, 0]).any(dim=-1)
                src_data.segmentation[too_small_instance_pixels] = 0

                too_small_instance_pixels_ = get_one_hot_channels(tgt_data.segmentation, indices=(~valid).nonzero()[:, 0]).any(dim=-1)
                tgt_data.segmentation[too_small_instance_pixels_] = 0


        src_pad_mask = (src_data.segmentation < 255).any(dim=-1)
        tgt_pad_mask = (tgt_data.segmentation < 255).any(dim=-1)

        # We convert to uint8 to save memory.
        src_data.segmentation = src_data.segmentation.to(torch.uint8)
        tgt_data.segmentation = tgt_data.segmentation.to(torch.uint8)

        ret = {}

        if self.use_preprocessed_masks is False:
            categories = torch.full((valid.shape), fill_value=-1, dtype=torch.long)
            unique_instance, unique_semantic = unique_instance[:num_masks], unique_semantic[:num_masks]
            categories[unique_instance] = unique_semantic
            categories[~valid] = -1
            categories = categories[..., 1:]
            ret["categories"] = categories

        valid = valid[..., 1:]
        
        ret.update({
            "tgt_pad_mask": tgt_pad_mask,
            "tgt_pixel_values": tgt_data.image,
            "tgt_segmentation": tgt_data.segmentation,
            "src_pad_mask": src_pad_mask,
            "src_pixel_values": src_data.image,
            "src_segmentation": src_data.segmentation,
            "input_ids": get_tokens(self.tokenizer),
            "valid": valid,
            "metadata" : {
                "name": str(image_id),
                "scene_id": str(image_id),
                "split": self.split.name.lower(),
            }
        })

        if src_data.grid is not None: ret["src_grid"] = src_data.grid.squeeze(0)
        if tgt_data.grid is not None: ret["tgt_grid"] = tgt_data.grid.squeeze(0)

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
        images = rearrange(((batch.tgt_pixel_values + 1) / 2) * 255, "b c h w -> b h w c").to(torch.uint8).cpu().detach().numpy()
        for i in range(batch.bs):
            image = images[i]
            torch.cuda.synchronize()
            start_time = time.time()
            masks = hqsam.forward(image)
            masks = sorted(masks, key=lambda d: d["area"], reverse=True)
            masks = masks[:32]  # We only have 77 tokens
            masks = np.array([masks[i]["segmentation"] for i in range(len(masks))]).transpose(1, 2, 0)
            masks = one_hot_to_integer(masks)
            batch.tgt_segmentation[i] = torch.from_numpy(masks).long()
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
    default_augmentation=Augmentation(
        initial_resolution=512,
        center_crop=False,
        enable_random_resize_crop=True,
        enable_horizontal_flip=True,
        tgt_random_scale_ratio=((0.5, 1), (1.0, 1.0)),
        src_transforms=get_stable_diffusion_transforms(resolution=512),
        tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        reorder_segmentation=False,
    )
    soda_augmentation=Augmentation(
        different_src_tgt_augmentation=True,
        enable_random_resize_crop=True, 
        enable_horizontal_flip=True,
        src_random_scale_ratio=((0.8, 1.0), (0.9, 1.1)),
        tgt_random_scale_ratio=((0.3, 0.6), (0.8, 1.2)),
        enable_rand_augment=False,
        enable_rotate=True,
        tgt_transforms=get_stable_diffusion_transforms(resolution=512),
        reorder_segmentation=True
    )
    dataset = CocoPanoptic(
        shuffle=False,
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=4,
        tokenizer=tokenizer,
        augmentation=default_augmentation,
        single_return=False,
        return_tensorclass=True,
        object_ignore_threshold=0.0,
        top_n_masks_only=77,
        num_overlapping_masks=3,
        # use_preprocessed_masks=True,
        # postprocess=True,
        # preprocessed_mask_type="custom_postprocessed",
        # erode_dialate_preprocessed_masks=False,
        # num_overlapping_masks=2,
    )

    import time
    generator = torch.Generator().manual_seed(0)
    dataloader = dataset.get_dataloader(generator=generator, pin_memory=False) #.to(torch.device('cuda:0'))

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        names = [f'{batch.metadata["scene_id"][i]}_{dataset.split.name.lower()}' for i in range(batch.bs)]
        visualize_input_data(batch, names=names, show_overlapping_masks=True, remove_invalid=False)
        start_time = time.time()
        if step > 10:
            break