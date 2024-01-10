import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
from image_utils import Im

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig('test.png')

image = Im('https://raw.githubusercontent.com/albumentations-team/albumentations_examples/master/images/original_parrot.jpg').np
image0 = image.copy()

keypoints = [
    (100, 100),
    (720, 410),
    (1100, 400),
    (1700, 30), 
    (300, 650),
    (1570, 590),
    (560, 800),
    (1300, 750), 
    (900, 1000),
    (910, 780),
    (670, 670),
    (830, 670), 
    (1000, 670),
    (1150, 670),
    (820, 900),
    (1000, 900),
]


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(p=0.2),
    ],
    additional_targets={'image0': 'image', 'image1': 'image'},
    keypoint_params=A.KeypointParams(format='xy')
)
random.seed(42)

transformed = transform(image=image, image0=image0, keypoints=keypoints)

visualize(transformed['image'])