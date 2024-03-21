from pathlib import Path
from typing import List, Tuple
import autoroot

import sys
from argparse import ArgumentParser
from os import path

from einops import rearrange
from image_utils.standalone_image_utils import onehot_to_color
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from gen.models.encoders.sam import get_sam_model
from gen.utils.decoupled_utils import breakpoint_on_error

detic_path = Path('/home/aswerdlo/repos/lib/Detic/')

sys.path.insert(0, str(detic_path))
sys.path.insert(1, str(detic_path / 'third_party/'))
sys.path.insert(2, str(detic_path / 'third_party/CenterNet2/'))

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detic.config import add_detic_config
from centernet.config import add_centernet_config
from segment_anything_fast import SamAutomaticMaskGenerator, SamPredictor, calculate_stability_score, MaskData

def setup_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(detic_path / 'configs' / 'Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.MODEL.WEIGHTS = str(detic_path / 'models' / 'Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = str(detic_path / 'datasets/metadata/lvis_v1_clip_a+cname.npy')
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(detic_path / 'datasets/metadata/lvis_v1_train_cat_info.json')
    cfg.freeze()
    return cfg

def get_unique_masks(predictions):

    raw_boxes = predictions["instances"].pred_boxes.tensor
    unique_masks = [predictions["instances"].pred_masks[0]]
    mask_labels = [predictions["instances"].pred_classes[0]]
    for idx in range(1, len(raw_boxes)):
        if not torch.allclose(raw_boxes[idx], raw_boxes[idx-1]):
            unique_masks.append(predictions["instances"].pred_masks[idx])
            mask_labels.append(predictions["instances"].pred_classes[idx])

    unique_masks = torch.stack(unique_masks)
    mask_labels = torch.stack(mask_labels)

    return unique_masks, mask_labels

@torch.no_grad()
def forward_detic(detector, img, device, return_masks: bool = False, viz: bool = False):
    """
    Takes np HWC uint8, 0-255
    """
    height, width = img.shape[:2]
    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]
    predictions = detector(inputs)[0]
    
    if return_masks:
        if len(predictions["instances"].pred_masks) > 0:
            unique_masks, mask_labels = get_unique_masks(predictions)
        else:
            unique_masks = torch.zeros((0, height, width), device=device)
            mask_labels = []

        detections = unique_masks.unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

    if viz:
        breakpoint()
        img = img[:, :, ::-1]
        from detectron2.utils.visualizer import ColorMode, Visualizer
        visualizer = Visualizer(img)
        instances = predictions["instances"].to(torch.device('cpu'))
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        Im(vis_output.get_image())


    return predictions['instances'].pred_boxes.tensor

def forward_sam(sam_predictor, image, boxes, device):
    sam_predictor.set_image(image)
    im_size = image.shape[:2]
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes.to(device, non_blocking=True), im_size)
    masks, iou_predictions, low_res_masks = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    Im(onehot_to_color(rearrange(masks[:, 0], 'c h w -> h w c'))).save()
    batch_data = process_batch(sam_predictor, masks, iou_predictions)
    Im(onehot_to_color(rearrange(batch_data['masks'], 'c h w -> h w c'))).save()
    breakpoint()

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def process_batch(
        predictor,
        masks: torch.Tensor,
        iou_preds: torch.Tensor,
) -> MaskData:
    # Serialize predictions and store in MaskData
    data = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
    )
    del masks

    pred_iou_thresh: float = 0.75
    stability_score_offset: float = 1.0
    stability_score_thresh: float = 0.95

    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

    # # Calculate stability score
    # mask_threshold = 0.0
    # stability_score_offset = 0.0
    # data["stability_score"] = calculate_stability_score(data["masks"], 0.2, stability_score_offset)
    # if stability_score_thresh > 0.0:
    #     keep_mask = data["stability_score"] >= stability_score_thresh
    #     data.filter(keep_mask)

    # # Threshold masks and calculate boxes
    # data["masks"] = data["masks"] > self.predictor.model.mask_threshold
    # data["boxes"] = batched_mask_to_box(data["masks"])

    # Filter boxes that touch crop boundaries
    # keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
    # if not torch.all(keep_mask):
    #     data.filter(keep_mask)

    # # Compress to RLE
    # data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)

    return data



if __name__ == "__main__":
    from image_utils import Im
    with breakpoint_on_error():
        device = torch.device('cuda:0')

        cfg = setup_cfg()
        detector = build_model(cfg)
        detector = detector.to(device)
        detector.eval()

        sam = get_sam_model('vit_b')
        sam = sam.to(device)
        sam_predictor = SamPredictor(sam)

        img = Im('https://raw.githubusercontent.com/SysCV/sam-hq/main/demo/input_imgs/example8.png').np
        boxes = forward_detic(detector, img, device, viz=True)
        forward_sam(sam_predictor, img, boxes, device)
        

