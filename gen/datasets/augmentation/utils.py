from einops import rearrange
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from image_utils import Im
import matplotlib.pyplot as plt
import torch
from kornia.geometry.keypoints import Keypoints
import warnings


def get_keypoints(B, H, W):
    # warnings.warn("Kornia uses XY order for Keypoints. Be Careful.")
    y_coords, x_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
    coords = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
    keypoints = Keypoints(coords.float())
    return keypoints


def get_viz_keypoints(B, H, W, step=4):
    y_coords_viz = torch.linspace(0, H - 1, (H - 1) // step + 1)
    x_coords_viz = torch.linspace(0, W - 1, (W - 1) // step + 1)
    x_coords_viz, y_coords_viz = torch.meshgrid(x_coords_viz, y_coords_viz, indexing="ij")
    viz_coords = torch.stack((x_coords_viz, y_coords_viz), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
    keypoints_viz = Keypoints(viz_coords.float())
    return keypoints_viz


def process_output_keypoints(keypoints: Keypoints, B: int, input_H: int, input_W: int, output_H: int, output_W: int, fill_value: int = -1):
    y_coords, x_coords = torch.meshgrid(torch.arange(input_W), torch.arange(input_H), indexing="ij")
    coords = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)  # YX Order

    # Round valid keypoints to nearest pixel and mask out invalid keypoints
    arr = torch.round(keypoints.to_tensor()[keypoints.valid_mask]).int()  # Coordinates for the target image
    coords_arr = coords[keypoints.valid_mask]  # Coordinates for the source image

    inverted_arr = torch.full((B, output_H, output_W, 2), fill_value, dtype=torch.long) # TODO: BUG HERE
    # warnings.warn("Kornia uses XY order for Keypoints. Be Careful.")
    inverted_arr[torch.nonzero(keypoints.valid_mask)[:, 0], arr[:, 1], arr[:, 0]] = coords_arr
    inverted_arr = torch.flip(inverted_arr, dims=(-1,))

    output_mask = ~(inverted_arr == -1).all(dim=-1)
    return inverted_arr, output_mask


def filter_bounds(coords, h, w):
    # Filter coordinates within image bounds (0 <= y < h and 0 <= x < w)
    in_bounds = (coords[:, 0] < h) & (coords[:, 0] >= 0) & \
                (coords[:, 1] < w) & (coords[:, 1] >= 0)
    return in_bounds

def process_output_segmentation(keypoints: Keypoints, output_segmentation: torch.Tensor, h: int, w: int, fill_value: float = -1):
    output_segmentation = output_segmentation.clone()
    # old : output_segmentation[~output_mask] = fill_value

    # TODO: Verify this is correct
    arr = torch.round(keypoints.to_tensor()[~keypoints.valid_mask]).int()  # Coordinates for the target image
    filter_mask = filter_bounds(arr, h, w)
    arr = arr[filter_mask]

    output_segmentation[torch.nonzero(~keypoints.valid_mask)[filter_mask][:, 0], arr[:, 1], arr[:, 0]] = fill_value
    return output_segmentation


def draw_keypoints_with_circles(idx, image, keypoints, colors, radius=1):
    pil_img = to_pil_image(image)
    draw = ImageDraw.Draw(pil_img)
    for i, (x, y) in enumerate(keypoints.to_tensor()[idx]):
        if keypoints.valid_mask[idx][i] and 0 <= y <= image.shape[1] and 0 <= x <= image.shape[2]:
            color = tuple(int(c * 255) for c in colors(i)[:3])
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)

    return pil_img


def viz(in_tensor, keypoints_viz, output_tensor, output_keypoints_viz):
    num_keypoints = keypoints_viz.to_tensor().shape[1]
    colors = plt.cm.get_cmap("flag", num_keypoints)
    B, C, H, W = in_tensor.shape

    imgs = []
    for j in range(B):
        input_with_keypoints = draw_keypoints_with_circles(j, in_tensor[j], keypoints_viz, colors)
        output_with_keypoints = draw_keypoints_with_circles(j, output_tensor[j], output_keypoints_viz, colors)
        imgs.append(
            Im.concat_horizontal(Im(input_with_keypoints).add_border(5, color=(255, 255, 255)), Im(output_with_keypoints).add_border(5, color=(255, 255, 255)))
        )

    Im.concat_vertical(*imgs).save("transform")
