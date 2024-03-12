import json
from pathlib import Path
import cv2
import numpy as np
import os
from pycocotools import mask as maskUtils

def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}],
    }

def mask_to_poly(masks):
  N = masks.shape[0]
  polys = []
  for i in range(N):
    contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
      # Valid polygons have >= 6 coordinates (3 points)
      if contour.size >= 6:
        segmentation.append(contour.flatten().tolist())
    polys.append(segmentation)
  return polys

def update_with_inference_single(masks, image_id, annotations, annotation_id, image_size):
    category_id = 1
    contours = mask_to_poly(masks.cpu().numpy())
    for j, contour in enumerate(contours):
        annotation = create_annotation_format(contour, image_id, category_id, annotation_id, image_size)
        annotations.append(annotation)
        annotation_id += 1
    return annotations, annotation_id

def create_annotation_format(contour, image_id_, category_id, annotation_id, image_size, use_rle: bool = True):
    height, width = image_size
    area = -1
    if use_rle:
        try:
            if len(contour) > 0:
                contour = maskUtils.frPyObjects(contour, height, width)
                contour = maskUtils.merge(contour)
                area = maskUtils.area(contour)
                contour['counts'] = str(contour['counts'], "utf-8")

        except:
            print(f"Error in maskUtils.frPyObjects, countour is empty: {contour}")

    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "segmentation": contour,
        "area": int(area),
        # "bbox": cv2.boundingRect(contour),
    }

def update_with_inference_batch(inference_batch, annotations, annotation_id, original_sizes):
    for i in range(len(inference_batch)):
        category_id = 1
        masks = inference_batch[i][1]
        contours = mask_to_poly(masks.cpu().numpy())
        image_size = original_sizes[i]
        for j, contour in enumerate(contours):
            annotation = create_annotation_format(contour, inference_batch[i][0], category_id, annotation_id, image_size)
            annotations.append(annotation)
            annotation_id += 1

    return annotations, annotation_id

def save_json(coco_root_dir, coco_slice_name, annotations, split_idx, prefix='sam'):
    custom_data_root = f"{str(coco_root_dir)}/annotations"
    json_data = json.load(open(os.path.join(custom_data_root, 'instances_{}.json'.format(coco_slice_name))))
    del json_data["annotations"]
    json_data["annotations"] = annotations
    output_dir = Path(coco_root_dir)

    with open(output_dir / "annotations" / f"{prefix}_split_{split_idx}_{coco_slice_name}.json", "w") as outfile:
        json.dump(json_data, outfile, sort_keys=True)