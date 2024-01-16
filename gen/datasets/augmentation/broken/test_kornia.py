from typing import List
from image_utils import Im
from einops import rearrange

from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.container import AugmentationSequential
from kornia.geometry.keypoints import Keypoints
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

import kornia.augmentation as K
from kornia.augmentation.auto.base import SUBPLOLICY_CONFIG, PolicyAugmentBase
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

custom_policy: List[SUBPLOLICY_CONFIG] = [
    [("auto_contrast", 0, 1)],
    [("equalize", 0, 1)],
    [("invert", 0, 1)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    [("solarize_add", 0.0, 0.43)],
    [("color", 0.1, 1.9)],
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    # [("rotate", -30.0, 30.0)],
    # [("shear_x", -0.3, 0.3)],
    # [("shear_y", -0.3, 0.3)],
    # [("translate_x", -0.1, 0.1)],
    # [("translate_x", -0.1, 0.1)],
]

aug = AugmentationSequential(
    K.RandomResizedCrop(size=(192, 192), scale=(0.7, 1.3), ratio=(0.7, 1.3)),
    K.RandomRotation(degrees=45.0, p=1.0),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    K.RandomTranslate((0.3, 0.3), p=1.0),
    K.RandomShear((-25., 25.), p=1.0),
    K.RandomAffine((-15., 20.), p=1.0),
    # RandAugment(n=2, m=10, policy=custom_policy),
    data_keys=["input", "keypoints", "keypoints"],
    random_apply=False
)
# in_tensor = Im('https://raw.githubusercontent.com/albumentations-team/albumentations_examples/master/images/original_parrot.jpg').torch.repeat(10, 1, 1, 1)
image = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').pil
in_tensor = Im(image.crop(((image.size[0]-image.size[1]) // 2, 0, image.size[0] - (image.size[0]-image.size[1]) // 2, image.size[1]))).resize(224, 224).torch.repeat(10, 1, 1, 1)

B, C, H, W = in_tensor.shape
y_coords, x_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
coords = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
keypoints = Keypoints(coords.float(), image_size=torch.tensor(((W, H))))

# Viz
step = 4
y_coords_viz = torch.linspace(0, H - 1, (H - 1) // step + 1)
x_coords_viz = torch.linspace(0, W - 1, (W - 1) // step + 1)
x_coords_viz, y_coords_viz = torch.meshgrid(x_coords_viz, y_coords_viz, indexing='ij')
viz_coords = torch.stack((x_coords_viz, y_coords_viz), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
keypoints_viz = Keypoints(viz_coords.float(), image_size=torch.tensor(((W, H))))

import time
start_time = time.time()
output_tensor, output_keypoints, output_keypoints_viz = aug(in_tensor, keypoints, keypoints_viz)
print(f'Augmentation time: {time.time() - start_time}')

# output_keypoints_viz_ = rearrange(output_keypoints_viz.to_tensor(), 'b (h w) ... -> b h w ...', h=viz_H, w=viz_W) # (B, H, W, 2)
# trans = aug.get_transformation_matrix(in_tensor, params=aug._params)

num_keypoints = keypoints_viz.to_tensor().shape[1]
colors = plt.cm.get_cmap('flag', num_keypoints)

def draw_keypoints_with_circles(idx, image, keypoints, colors, radius=1):
    pil_img = to_pil_image(image)
    draw = ImageDraw.Draw(pil_img)
    for i, (x, y) in enumerate(keypoints.to_tensor()[idx]):
        if keypoints.visibility[idx][i] and 0 <= y <= image.shape[1] and 0 <= x <= image.shape[2]:
            color = tuple(int(c*255) for c in colors(i)[:3])
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)

    return pil_img


imgs = []
for j in range(B):
    input_with_keypoints = draw_keypoints_with_circles(j, in_tensor[j], keypoints_viz, colors)
    output_with_keypoints = draw_keypoints_with_circles(j, output_tensor[j], output_keypoints_viz, colors)
    imgs.append(Im.concat_horizontal(Im(input_with_keypoints).add_border(5, color=(255, 255, 255)), Im(output_with_keypoints).add_border(5, color=(255, 255, 255))))
    
Im.concat_vertical(*imgs).save('transform')