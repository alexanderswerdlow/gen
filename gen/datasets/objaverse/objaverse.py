import autoroot

import json
import os
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from gen import OBJAVERSE_DATASET_PATH
from gen.configs.utils import inherit_parent_args
from gen.datasets.abstract_dataset import AbstractDataset, Split
from gen.datasets.augmentation.kornia_augmentation import Augmentation, Data
from gen.utils.data_defs import integer_to_one_hot, one_hot_to_integer
from gen.utils.tokenization_utils import get_tokens

torchvision.disable_beta_transforms_warning()

import os
import math
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys

class ObjaverseDataLoader():
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [torchvision.transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])

def get_pose(target_RT):
    target_RT = target_RT[:3, :]
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    # assert if z_target is out of range
    if z_target.item() < 1.5 or z_target.item() > 2.2:
        # print('z_target out of range 1.5-2.2', z_target.item())
        z_target = np.clip(z_target.item(), 1.5, 2.2)
    # with log scale for radius
    target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.5))/(np.log(2.2)-np.log(1.5)) * torch.pi, torch.tensor(0)])
    assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
    return target_T.numpy()


@inherit_parent_args
class ObjaverseData(AbstractDataset, Dataset):
    def __init__(
            self,
            *,
            root_dir=OBJAVERSE_DATASET_PATH, # '.objaverse/hf-objaverse-v1/views',
            total_view=12,
            T_in=1,
            T_out=1,
            fix_sample=False,
            augmentation: Optional[Augmentation] = None,
            tokenizer=None,
            # TODO: All these params are not actually used but needed because of a quick with hydra_zen
            resolution=None,
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
            root=None,
            custom_data_root=None,
            min_resize_value=None,
            max_resize_value=None,
            resize_factor=None,
            mirror=None,
            min_scale=None,
            max_scale=None,
            scale_step_size=None,
            mean=None,
            std=None,
            semantic_only=False,
            ignore_stuff_in_offset=False,
            small_instance_area=None,
            small_instance_weight=None,
            enable_orig_coco_augmentation=None,
            enable_orig_coco_processing=None,
            object_ignore_threshold=None,
            single_return=None,
            top_n_masks_only=None,
            **kwargs
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.total_view = total_view
        self.T_in = T_in
        self.T_out = T_out
        self.fix_sample = fix_sample

        self.augmentation = augmentation
        self.tokenizer = tokenizer
        validation = (self.split == Split.VALIDATION)

        self.paths = []
        # # include all folders
        for folder in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, folder)):
                self.paths.append(folder)

        # load ids from .npy so we have exactly the same ids/order
        # self.paths = np.load("../scripts/obj_ids.npy")
        # # only use 100K objects for ablation study
        # self.paths = self.paths[:100000]
        # assert total_objects == 790152, 'total objects %d' % total_objects
                
        total_objects = len(self.paths)
        
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))

        downscale = 512 / 256.
        self.fx = 560. / downscale
        self.fy = 560. / downscale
        self.intrinsic = torch.tensor([[self.fx, 0, 128., 0, self.fy, 128., 0, 0, 1.]], dtype=torch.float64).view(3, 3)

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def get_pose(self, target_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        # assert if z_target is out of range
        if z_target.item() < 1.5 or z_target.item() > 2.2:
            # print('z_target out of range 1.5-2.2', z_target.item())
            z_target = np.clip(z_target.item(), 1.5, 2.2)
        # with log scale for radius
        target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.5))/(np.log(2.2)-np.log(1.5)) * torch.pi, torch.tensor(0)])
        assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
        return target_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = 12

        if self.fix_sample:
            if self.T_out > 1:
                indexes = range(total_view)
                index_targets = list(indexes[:2]) + list(indexes[-(self.T_out-2):])
                index_inputs = indexes[1:self.T_in+1]   # one overlap identity
            else:
                indexes = range(total_view)
                index_targets = indexes[:self.T_out]
                index_inputs = indexes[self.T_out-1:self.T_in+self.T_out-1] # one overlap identity
        else:
            assert self.T_in + self.T_out <= total_view
            # training with replace, including identity
            indexes = np.random.choice(range(total_view), self.T_in+self.T_out, replace=True)
            index_inputs = indexes[:self.T_in]
            index_targets = indexes[self.T_in:]
        filename = os.path.join(self.root_dir, self.paths[index])

        color = [1., 1., 1., 1.]

        try:
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(input_RT))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(target_RT))
        except:
            print('error loading data ', filename)
            filename = os.path.join(self.root_dir, '3fa344c4fbc74e68a07b139d9920772f')  # this one we know is valid
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            # very hacky solution, sorry about this
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(input_RT))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(target_RT))

        source_data, target_data = self.augmentation(
            source_data=Data(image=torch.stack(input_ims, dim=0)),
            target_data=Data(image=torch.stack(target_ims, dim=0)),
        )

        # stack to batch
        assert source_data.image.shape[0] == 1
        assert target_data.image.shape[0] == 1
        data['disc_pixel_values'] = source_data.image[0]
        data['gen_pixel_values'] = target_data.image[0]
        data['disc_segmentation'] = data['disc_pixel_values'].new_zeros((data['disc_pixel_values'].shape[1], data['disc_pixel_values'].shape[2]), dtype=torch.long)
        data['gen_segmentation'] = data['gen_pixel_values'].new_zeros((data['gen_pixel_values'].shape[1], data['gen_pixel_values'].shape[2]), dtype=torch.long)
        data['gen_pose_out'] = torch.stack(target_Ts, dim=0)
        data['disc_pose_in'] = torch.stack(cond_Ts, dim=0)
        data['input_ids'] = get_tokens(self.tokenizer)
        data['valid'] = torch.full((1,), True, dtype=torch.bool)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        im = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.
        return im
    
    def get_dataset(self):
        return self
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = ObjaverseData(
        cfg=None,
        split=Split.TRAIN,
        num_workers=0,
        batch_size=2,
        shuffle=True,
        subset_size=None,
        tokenizer=tokenizer,
        augmentation=Augmentation(enable_rand_augment=False, enable_random_resize_crop=False, enable_horizontal_flip=False),
    )
    import time
    start_time = time.time()
    dataloader = dataset.get_dataloader()
    for step, batch in enumerate(dataloader):
        print(f'Time taken: {time.time() - start_time}')
        start_time = time.time()
        from image_utils import Im, onehot_to_color
        gen_seg = integer_to_one_hot(batch["gen_segmentation"], 1)
        disc_seg = integer_to_one_hot(batch["disc_segmentation"], 1)
        for b in range(batch['gen_pixel_values'].shape[0]):        
            gen_ = Im.concat_vertical(Im((batch['gen_pixel_values'][b] + 1) / 2), Im(onehot_to_color(gen_seg[b].squeeze(0))))
            disc_ = Im.concat_vertical(Im((batch['disc_pixel_values'][b] + 1) / 2), Im(onehot_to_color(disc_seg[b].squeeze(0))))
            print(gen_seg.sum() / gen_seg[b, ..., 0].numel(), disc_seg.sum() / disc_seg[b, ..., 0].numel())
            Im.concat_horizontal(gen_, disc_).save(f'objaverse_{step}_{b}.png')

        if step > 1:
            break