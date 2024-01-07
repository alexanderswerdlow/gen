import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
from kornia.augmentation.container import AugmentationSequential
from kornia.geometry.keypoints import Keypoints
from PIL import ImageDraw, Image, ImageOps
from torchvision.transforms.functional import to_pil_image

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

B, C, H, W = (10, 3, 224, 224)
in_tensor = torch.ones((B, C, H, W))

# Setup grid of spaced out keypoints for visualization
step = 16
y_coords_viz = torch.linspace(0, H - 1, (H - 1) // step + 1)
x_coords_viz = torch.linspace(0, W - 1, (W - 1) // step + 1)
y_coords_viz, x_coords_viz = torch.meshgrid(y_coords_viz, x_coords_viz, indexing='ij')
viz_coords = torch.stack((x_coords_viz, y_coords_viz), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
input_keypoints = Keypoints(viz_coords.float())

aug = AugmentationSequential(
    K.RandomResizedCrop(size=(128, 128)),
    K.RandomTranslate(0.5, 0.5),
    data_keys=["input", "keypoints"],
    random_apply=False
)
output_tensor, output_keypoints = aug(in_tensor, input_keypoints)

num_keypoints = output_keypoints.to_tensor().shape[1]
colors = plt.cm.get_cmap('flag', num_keypoints)

imgs = []
for j in range(B):
    input_with_keypoints = draw_keypoints_with_circles(in_tensor[j], input_keypoints.to_tensor()[j], colors)
    output_with_keypoints = draw_keypoints_with_circles(output_tensor[j], output_keypoints.to_tensor()[j], colors)
    
    imgs.append(add_border(input_with_keypoints))
    imgs.append(add_border(output_with_keypoints))
    
concat_images_vertically(*imgs).save('transform.png')