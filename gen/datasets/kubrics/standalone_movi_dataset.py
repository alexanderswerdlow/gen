import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from gen import MOVI_DATASET_PATH

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
from image_utils import Im, onehot_to_color
from torchvision.tv_tensors import (BoundingBoxes, BoundingBoxFormat, Image,
                                    Mask)


def draw_boxes(img, bbx):
    import cv2
    boxes = bbx.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[0]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[1]
    for box in boxes:
        if box.sum() == 0:
            continue
        x1, y1, x2, y2 = torch.round(box).to(torch.int).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)

    return Im(img)
    
class StandaloneMoviDataset(Dataset):
    def __init__(self, root: Path, dataset: str, split='train', num_frames = 24, augment=False, num_dataset_frames=24, resolution=(512, 512), normalize_img: bool = True, **kwargs):
        super(MoviDataset, self).__init__()
        self.root = Path(root) # Path to the dataset containing folders of "movi_a", "movi_e", etc.
        self.dataset = dataset # str of dataset name (e.g. "movi_a")
        self.split = split # str of split name (e.g. "train", "validation")
        self.resolution = tuple(resolution)
        self.root_dir = self.root / self.dataset / split
        self.files = os.listdir(self.root_dir)
        self.files.sort()
        self.num_dataset_frames = num_dataset_frames
        self.num_frames = num_frames
        self.augment = augment
        self.normalize_img = normalize_img

        if self.augment:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(self.resolution, scale=(0.5, 1.0), antialias=True)])
        else:
            self.transform = transforms.Compose([transforms.Resize(self.resolution, antialias=True)])


    def __getitem__(self, index):
        video_idx = index // len(self.files)
        frame_idx = index % self.num_dataset_frames

        path = self.files[video_idx]
        rgb = os.path.join(self.root_dir, os.path.join(path, 'rgb.npy'))
        instance = os.path.join(self.root_dir, os.path.join(path, 'segment.npy'))
        bbx = os.path.join(self.root_dir, os.path.join(path, 'bbox.npy'))

        rgb = np.load(rgb)
        bbx = np.load(bbx)
        instance = np.load(instance)

        # For returning videos
        # rand_id = random.randint(0, 24 - self.num_frames)
        # real_idx = [rand_id + j for j in range(self.num_frames)]

        rgb = rgb[frame_idx]
        bbx = bbx[frame_idx]
        instance = instance[frame_idx]

        bbx[..., [0, 1]] = bbx[..., [1, 0]]
        bbx[..., [2, 3]] = bbx[..., [3, 2]]

        bbx[..., [0, 2]] *= rgb.shape[1]
        bbx[..., [1, 3]] *= rgb.shape[0]

        bbx = torch.from_numpy(bbx)
        rgb = torch.Tensor(rgb).float()
        rgb = (rgb / 255.0)

        assert rgb.shape[0] == rgb.shape[1]
        assert rgb.min() >= 0 and rgb.max() <= 1

        if self.normalize_img:
            rgb = (rgb * 2) - 1

        rgb = rearrange(rgb, '... h w c -> ... c h w')
        bounding_boxes = BoundingBoxes(bbx, format=BoundingBoxFormat.XYXY, canvas_size=rgb.shape[-2:])
        instance = Mask(instance.squeeze(-1))
        rgb, bbx, instance = self.transform(Image(rgb), bounding_boxes, instance)
        rgb = rearrange(rgb, '... c h w -> ... h w c')

        bbx[..., [0, 2]] /= rgb.shape[0]
        bbx[..., [1, 3]] /= rgb.shape[1]
        assert bbx.min() >= 0 and bbx.max() <= 1

        instance = torch.nn.functional.one_hot(instance.long(), num_classes=21).numpy()

        ret = {
            "image": rgb,
            'bbox': bbx,
            'segmentation': instance,
        }

        return ret
            
    
    def __len__(self):
        return len(self.files) * self.num_dataset_frames


if __name__ == "__main__":
    dataset = MoviDataset(root=MOVI_DATASET_PATH, dataset='movi_e', split='validation', augment=False, num_frames=24)
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        Im.concat_horizontal(draw_boxes(Im((data['image'] + 1) / 2).np, data['bbox']), Im(onehot_to_color(data['segmentation']))).save(f'{i}')
        
    from ipdb import set_trace; set_trace()