import copy
from functools import partial
import gzip
import json
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import typer
from pycocotools import mask as maskUtils
from tqdm import tqdm

from image_utils import Im, library_ops, hist

from gen.utils.data_defs import integer_to_one_hot
from image_utils import integer_to_color

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
    
def postprocess_custom(anns, idx: int, run_nms: bool = True, run_remove_holes: bool = False, viz: bool = False):
    from segment_anything_fast.utils.amg import batched_mask_to_box
    from torchvision.ops.boxes import batched_nms

    anns = [ann for ann in anns if len(ann['segmentation']) > 0]
    try:
        anns = [(maskUtils.area(ann['segmentation']), ann) for ann in anns if len(ann['segmentation']) > 0]
        anns = sorted(anns, key=lambda d: d[0], reverse=True)
        anns = [ann[1] for ann in anns]

        masks = np.stack([annToMask(anns[i]) for i in range(len(anns))]).astype(np.bool_)
    except Exception as e:
        print(e)
        breakpoint()

    if viz:
        initial_num_classes = masks.sum(axis=0).max() + 1
        initial_image = integer_to_color(masks.sum(axis=0), colormap='hot', num_classes=initial_num_classes, ignore_empty=False)
        first_masks = Im(torch.from_numpy(masks).unsqueeze(-1)).grid(pad_value=0.5)
        first_hist = hist(np.sum(masks, axis=0).reshape(-1), save=False)
    
    if run_remove_holes:
        new_masks = []
        scores = []
        min_hole_area = 100
        min_island_area = 100
        for mask in masks:
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
    else:
        scores = [float(1)] * len(masks)

    if viz:
        island_image = integer_to_color(masks.sum(axis=0), colormap='hot', num_classes=initial_num_classes, ignore_empty=False)
        second_masks = Im(masks.unsqueeze(-1)).grid(pad_value=0.5)
        second_hist = hist(torch.sum(masks, dim=0).reshape(-1), save=False)

    nms_thresh = 0.9
    if run_nms:
        if isinstance(masks, np.ndarray):
            masks = torch.as_tensor(masks)

        try:
            boxes = batched_mask_to_box(masks)
            keep_by_nms = batched_nms(
                boxes.float(),
                torch.as_tensor(scores),
                torch.zeros_like(boxes[:, 0]),  # categories
                iou_threshold=nms_thresh,
            )
            masks = masks[keep_by_nms]
        except:
            print(type(masks), masks.dtype, masks.shape)
            print(scores, torch.as_tensor(scores).shape, torch.as_tensor(scores).dtype)

    if viz:
        nms_image = integer_to_color(masks.sum(axis=0), colormap='hot', num_classes=initial_num_classes, ignore_empty=False)
        third_masks = Im(masks.unsqueeze(-1)).grid(pad_value=0.5)
        third_hist = hist(torch.sum(masks, dim=0).reshape(-1), save=False)

    if viz:
        Im.concat_vertical(
            Im.concat_horizontal(initial_image, island_image, nms_image),
            first_hist.scale(3).write_text("Number of masks per pixel distribution before removing small regions"),
            first_masks,
            second_hist.scale(3).write_text("Number of masks per pixel distribution after removing small regions"),
            second_masks,
            third_hist.scale(3).write_text(f"Number of masks per pixel distribution after box NMS with {nms_thresh:.2f} threshold"),
            third_masks,
            spacing=120,
            fill=(128, 128, 128)
        ).save(f"postprocess_{idx}.png")

    return masks.bool().numpy()

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
def main(
    path: Path, 
    splits: Optional[str] = None, 
    num_chunks: int = 16, 
    prefix: str = "", 
    subset: Optional[int] = None, 
    merge_only: bool = False, 
    output: Optional[Path] = None, 
    override_merged: bool = False,
    run_nms: bool = True,
    run_remove_holes: bool = True,
    ):
    json_paths = list(get_files(path, allowed_extensions=['json']))
    print(json_paths)
    for split in ([splits] if splits is not None else ['train2017', 'val2017']):
        merged_json_path = path / f'merged_{prefix}_{split}.json'
        print(f"Postprocessing {split}")
        if not merged_json_path.exists() or override_merged:
            print(f"Postprocessing {split}, merging jsons...")
            merged_json_data = {'annotations': []}
            for json_path_ in json_paths:
                if prefix in json_path_.stem and split in json_path_.stem and 'merged' not in json_path_.stem:
                    print(f"Postprocessing {split}, merging {json_path_}...")
                    json_data_ = json.load(open(json_path_))
                    merged_json_data['annotations'].extend(json_data_['annotations'])
                    del json_data_['annotations']
                    merged_json_data.update(json_data_)

            if len(merged_json_data['annotations']) == 0:
                exit(f"No annotations found for {split} in {path}.")

            with open(merged_json_path, "w") as outfile:
                json.dump(merged_json_data, outfile, sort_keys=True, indent=4)

        else:
            print(f"Postprocessing {split}, merged json already exists, skipping...")
            merged_json_data = json.load(open(merged_json_path))


        if merge_only:
            new_annotation_json = merged_json_data
        else:
            print(f"Postprocessing {split}, Finished merging, creating image_id dict...")
            global img_id_to_annotations
            img_id_to_annotations = defaultdict(list)
            for ann in merged_json_data['annotations']:
                img_id_to_annotations[ann['image_id']].append(ann)
            
            img_ids = list(img_id_to_annotations.keys())
            if subset is not None:
                img_ids = img_ids[:subset]

            # Split img_ids into roughly equal chunks
            chunks = [img_ids[i::num_chunks] for i in range(num_chunks)]

            if len(chunks) > 1:
                print(f"Postprocessing {split}, creating new json in parallel...")
                with Pool(processes=num_chunks) as pool:
                    results = list(tqdm(pool.imap(partial(process_chunk, run_nms=run_nms, run_remove_holes=run_remove_holes), chunks), total=num_chunks))
            else:
                print(f"Postprocessing {split}, creating new json in serial...")
                results = [process_chunk(chunks[0])]

            new_annotation_json = copy.deepcopy(merged_json_data)
            new_annotation_json['annotations'] = [ann for sublist in results for ann in sublist]

        try:
            # The ids are duplicates because we processed the chunks separately, so we need to reassign them
            for i, ann in enumerate(new_annotation_json['annotations']):
                ann.update({'id': i})

            if output is not None:
                new_json_path = output
            else:
                new_json_path = Path(path) / f'{prefix}_{split}.json'
            print(f"Postprocessing {split}, saving new json to {new_json_path}...")
            with open(new_json_path, "w") as outfile:
                json.dump(new_annotation_json, outfile, sort_keys=True)

            print(f"Postprocessing {split}, compressing new json to {new_json_path.with_suffix('.json.gz')}...")
            with open(new_json_path, 'rb') as f_in, gzip.open(new_json_path.with_suffix('.json.gz'), 'wb', compresslevel=4) as f_out:
                shutil.copyfileobj(f_in, f_out)

        except Exception as e:
            print(e)
            breakpoint()

def process_chunk(chunk, run_nms, run_remove_holes):
    processed_annotations = []
    for idx, img_id in tqdm(enumerate(chunk)):
        masks = postprocess_custom(img_id_to_annotations[img_id], idx, run_nms=run_nms, run_remove_holes=run_remove_holes, viz=False)
        for mask in masks:
            new_ann = create_annotation_format(mask, img_id, 1, len(processed_annotations), (mask.shape[0], mask.shape[1]))
            processed_annotations.append(new_ann)
    return processed_annotations

if __name__ == "__main__":
    app()
