import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import typer
from pycocotools import mask as maskUtils
from tqdm import tqdm

from image_utils import Im

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def annToRLE(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    rle = ann['segmentation']
    return rle

def annToMask(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann)
    m = maskUtils.decode(rle)
    return m
    
def postprocess_custom(anns):
    from segment_anything_fast.utils.amg import batched_mask_to_box
    from torchvision.ops.boxes import batched_nms

    anns = [ann for ann in anns if len(ann['segmentation']) > 0]
    try:
        masks_ = np.stack([annToMask(anns[i]) for i in range(len(anns))]).astype(np.bool_)
    except Exception as e:
        print(e)
        breakpoint()

    new_masks = []
    scores = []
    min_hole_area = 64
    min_island_area = 64
    nms_thresh = 0.95
    for mask in masks_:
        mask, changed = remove_small_regions(mask, min_hole_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_island_area, mode="islands")
        unchanged = unchanged and not changed
        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),  # categories
        iou_threshold=nms_thresh,
    )
    
    masks_ = masks[keep_by_nms].bool()
    return masks_.numpy()

def get_file_list(**kwargs):
    [f for f in get_files(**kwargs)]


def get_files(path: Path, recursive: bool = False, return_folders: bool = False, allowed_extensions=None):
    path = Path(path)

    if allowed_extensions or recursive:
        glob_str = "*" if allowed_extensions is None else f"*.[{''.join(allowed_extensions)}]*"
        iterator = path.rglob(glob_str) if recursive else path.glob(glob_str)
    else:
        iterator = path.iterdir()

    for file in iterator:
        if file.is_file() or return_folders:
            yield file

def create_annotation_format(contour, image_id_, category_id, annotation_id, image_size, use_rle: bool = True):
    height, width = image_size
    area = -1
    if use_rle:
        try:
            if len(contour) > 0:
                contour = maskUtils.encode(np.asfortranarray(contour))
                area = maskUtils.area(contour)
                contour['counts'] = str(contour['counts'], "utf-8")
        except:
            breakpoint()
            print(f"Error in maskUtils.frPyObjects, countour is empty: {contour}")

    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id_,
        "category_id": category_id,
        "segmentation": contour,
        "area": int(area),
    }

@app.command()
def main(path: Path, splits: Optional[str] = None):
    json_paths = get_files(path, allowed_extensions=['json'])
    for split in ([splits] if splits is not None else ['train2017', 'val2017']):
        merged_json_path = path / f'custom_merged_{split}.json'
        print(f"Postprocessing {split}, merging jsons...")
        if not merged_json_path.exists():
            merged_json_data = {'annotations': []}
            for json_path_ in json_paths:
                if f'custom_split' in json_path_.stem and split in json_path_.stem:
                    json_data_ = json.load(open(json_path_))
                    merged_json_data['annotations'].extend(json_data_['annotations'])
                    del json_data_['annotations']
                    merged_json_data.update(json_data_)

            with open(merged_json_path, "w") as outfile:
                json.dump(merged_json_data, outfile, sort_keys=True, indent=4)

        else:
            merged_json_data = json.load(open(merged_json_path))

        print(f"Postprocessing {split}, creating dict...")
        img_id_to_annotations = defaultdict(list)
        for ann in merged_json_data['annotations']:
            img_id_to_annotations[ann['image_id']].append(ann)

        print(f"Postprocessing {split}, creating new json...")
        new_annotation_json = copy.deepcopy(merged_json_data)
        new_annotation_json['annotations'] = []
        for img_id in tqdm(img_id_to_annotations.keys()):
            masks = postprocess_custom(img_id_to_annotations[img_id])
            for mask in masks:
                new_ann = create_annotation_format(mask, img_id, 1, len(new_annotation_json["annotations"]), (mask.shape[0], mask.shape[1]))
                new_annotation_json["annotations"].append(new_ann)

        new_json_path = path / f'custom_postprocessed_{split}.json'
        with open(new_json_path, "w") as outfile:
            json.dump(new_annotation_json, outfile, sort_keys=True, indent=4)

if __name__ == "__main__":
    app()
