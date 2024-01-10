import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
from kornia.augmentation.container import AugmentationSequential
from kornia.geometry.keypoints import Keypoints
from PIL import ImageDraw, Image, ImageOps
from torchvision.transforms.functional import to_pil_image

def draw_bounding_boxes(pil_img, boxes, colors, width=1):
    draw = ImageDraw.Draw(pil_img)
    
    for i, box in enumerate(boxes):
        # Ensure box coordinates are within image dimensions
        if all(0 <= coord <= max(pil_img.size[1:]) for coord in box.flatten()):
            color = tuple(int(c*255) for c in colors(i)[:3])
            # Convert box tensor to a flat list of coordinates
            box_coords = tuple(coord.item() for coord in box.flatten())
            draw.line(box_coords + box_coords[:2], fill=color, width=width)

    return pil_img

# Setup Vis Utils
def draw_keypoints_with_circles(image, keypoints, colors, radius=2):
    pil_img = to_pil_image(image)
    draw = ImageDraw.Draw(pil_img)
    for i, (x, y) in enumerate(keypoints):
        if 0 <= y <= image.shape[1] and 0 <= x <= image.shape[2]:
            color = tuple(int(c*255) for c in colors(i)[:3])
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)

    return pil_img

def add_border(im):
    return ImageOps.expand(im, border=5, fill=(0, 255, 0))

# From StackOverflow
def concat_images_vertically(*images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    composite = Image.new('RGB', (width, height))
    y = 0
    for image in images:
        composite.paste(image, (0, y))
        y += image.height
    return composite

B, C, H, W = (1, 3, 224, 224)
in_tensor = torch.ones((B, C, H, W))

# Setup grid of spaced out keypoints for visualization
step = 16
y_coords_viz = torch.linspace(0, H - 1, (H - 1) // step + 1)
x_coords_viz = torch.linspace(0, W - 1, (W - 1) // step + 1)
x_coords_viz, y_coords_viz = torch.meshgrid(x_coords_viz, y_coords_viz, indexing='ij')
viz_coords = torch.stack((x_coords_viz, y_coords_viz), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
input_keypoints = Keypoints(viz_coords.float())

rng = torch.manual_seed(0)
aug1 = AugmentationSequential(K.RandomResizedCrop(size=(128, 128)), data_keys=["input", "keypoints"])
aug2 = AugmentationSequential(K.RandomTranslate(0.5, 0.5), data_keys=["input", "keypoints"])

output_tensor_1, output_keypoints_1 = aug1(in_tensor, input_keypoints)
output_tensor_2, output_keypoints_2 = aug2(output_tensor_1, output_keypoints_1)
print(aug1._params)
print(aug2._params)

from einops import rearrange
input_0 = rearrange(input_keypoints.to_tensor(), 'b (w h) c -> b w h c', w=x_coords_viz.shape[0], h=x_coords_viz.shape[1])
input_1 = rearrange(output_keypoints_1.to_tensor(), 'b (w h) c -> b w h c', w=x_coords_viz.shape[0], h=x_coords_viz.shape[1])
input_2 = rearrange(output_keypoints_2.to_tensor(), 'b (w h) c -> b w h c', w=x_coords_viz.shape[0], h=x_coords_viz.shape[1])

num_keypoints = output_keypoints_2.to_tensor().shape[1]
colors = plt.cm.get_cmap('flag', num_keypoints)

imgs = []
for j in range(B):
    input_with_keypoints = draw_keypoints_with_circles(in_tensor[j], input_keypoints.to_tensor()[j], colors)
    input_with_keypoints = draw_bounding_boxes(input_with_keypoints, aug1._params[0].data['src'][j][None], colors)
    output_1_with_keypoints = draw_keypoints_with_circles(output_tensor_1[j], output_keypoints_1.to_tensor()[j], colors)
    output_2_with_keypoints = draw_keypoints_with_circles(output_tensor_2[j], output_keypoints_2.to_tensor()[j], colors)
    
    imgs.append(add_border(input_with_keypoints))
    imgs.append(add_border(output_1_with_keypoints))
    imgs.append(add_border(output_2_with_keypoints))
    
concat_images_vertically(*imgs).save('transform.png')