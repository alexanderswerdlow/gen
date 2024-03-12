import os
import autoroot
import sys

from tqdm import tqdm
# add to start of path
hipie_path = os.getenv("HIPIE_PATH", "/home/aswerdlo/repos/lib/HIPIE")
sys.path.insert(0, hipie_path)
import cv2
import numpy as np
import requests
import torch
from detectron2.data.detection_utils import convert_PIL_to_numpy, read_image
from detectron2.projects.hipie.data.coco_dataset_mapper_uni import get_openseg_labels
from detectron2.projects.hipie.demo_lib.part_segm_demo import PartSegmDemo
from detectron2.utils.visualizer import Visualizer
from fairscale.nn.checkpoint import checkpoint_wrapper
from matplotlib import pyplot as plt
from PIL import Image
import skimage.io as io
import fire
from pycocotools.coco import COCO
from gen.datasets.coco.sam.save_custom_json import save_json, update_with_inference_single
from image_utils import Im

COCO_OPENSEG_LABELS = get_openseg_labels("coco_panoptic")
coco_labels = dict(
    things_labels=[x["name"] for x in get_openseg_labels("coco_panoptic")[:80]],
    stuff_labels=[x["name"] for x in get_openseg_labels("coco_panoptic")[80:]],
)
from PIL import Image, ImageOps

def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image

def setup_coco_img_ids(coco_root_dir, coco_slice_name, coco_category_names, img_id):
    annFile = "{}/annotations/instances_{}.json".format(coco_root_dir, coco_slice_name)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_cat = {cat["id"]: cat for cat in cats}

    if coco_category_names is not None:
        catIds = coco.getCatIds(catNms=coco_category_names)
    else:
        catIds = coco.getCatIds()

    if img_id is not None:
        coco_img_ids = [img_id]
    elif coco_category_names is None:
        coco_img_ids = coco.getImgIds()
    else:
        coco_img_ids = coco.getImgIds(catIds=catIds)

    return coco_img_ids, cat_id_to_cat, catIds, coco


def get_model():
    device = "cuda:0"
    config_file = f"{hipie_path}/projects/HIPIE/configs/training/r50_parts.yaml"
    ckpt = f"{hipie_path}/weights/r50_parts.pth"
    print(f"config_file: {config_file}")
    uninext_demo = PartSegmDemo(config_file=config_file, weight=ckpt, device=device)
    return uninext_demo


def integer_to_one_hot(int_tensor, num_classes):
    one_hot = torch.nn.functional.one_hot(int_tensor, num_classes)
    return one_hot

@torch.no_grad()
def forward_model(model, image):
    mask = model.foward_panoptic(image, do_part=True, instance_thres=0.49, **coco_labels) # (600, 600, 3), 0-255
    integer_seg = mask["panoptic_seg"][0].long()
    integer_one_hot = torch.nn.functional.one_hot(integer_seg, integer_seg.max() + 1).permute(2, 0, 1)

    orig_one_hot = integer_one_hot.clone()
    other_one_hot = None
    last_one_hot = None
    if len(mask["output_refined"][0]) > 0:
        print("No other masks")
        other_one_hot = torch.stack(mask["output_refined"][0])
        integer_one_hot = torch.cat([integer_one_hot, other_one_hot], dim=0)

    # if len(mask["output_coarse"][0]) > 0:
    #     last_one_hot = torch.stack(mask["output_coarse"][0])
    #     integer_one_hot = torch.cat([integer_one_hot, other_one_hot], dim=0)

    # if len(mask['instances'].pred_masks) > 0:
    #     last_one_hot = mask['instances'].pred_masks
    #     integer_one_hot = torch.cat([integer_one_hot, last_one_hot], dim=0)

    return integer_one_hot, mask, (orig_one_hot, other_one_hot, last_one_hot)


def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def run(
    coco_root_dir,
    coco_slice_name,
    limit=None,
    split_size=1,
    split_idx=0,
    viz=False,
):
    coco_category_names, img_id = None, None
    coco_img_ids, cat_id_to_cat, catIds, coco = setup_coco_img_ids(coco_root_dir, coco_slice_name, coco_category_names, img_id)
    limit = len(coco_img_ids) if limit is None else limit
    all_idx = split_range(range(limit), split_size)
    idx = list(all_idx)[split_idx]

    model = get_model()
    annotations = []
    annotation_id = 0
    for imgId in tqdm(coco_img_ids[slice(idx.start, idx.stop)]):
        img = coco.loadImgs(imgId)[0]
        file_location = f'{coco_root_dir}/{coco_slice_name}/{img["file_name"]}'
        img_ = read_image(file_location, format="BGR")
        try:
            seg_, mask, split_masks = forward_model(model, img_)
        except Exception as e:
            print(e)
            breakpoint()

        if viz:
            from image_utils import Im, hist
            visualizer = Visualizer(img_, metadata=mask['meta_data'])
            visualizer.draw_panoptic_seg(mask['panoptic_seg'][0].cpu(),mask['panoptic_seg'][1])
            if len(mask['instances']) > 0:
                visualizer.draw_instance_predictions(mask['instances'].to('cpu'))
            if len(mask['output_refined'][0]) > 0:
                visualizer.overlay_instances(masks=torch.stack(mask['output_refined'][0]),labels=mask['output_refined'][1])
            
            Im.concat_vertical(Im(visualizer.get_output().get_image()).write_text(str(seg_.shape), size=0.5), hist(torch.sum(seg_, dim=0).reshape(-1), save=False).torch[:3, :, :]).save(f'{imgId}')

            masks = seg_
            first_hist = hist(np.sum(masks.cpu().numpy(), axis=0).reshape(-1), save=False).scale(3).torch[:3, :, :]
            first_masks = Im(split_masks[0].unsqueeze(-1)).grid()
            second_masks = Im(split_masks[1].unsqueeze(-1)).grid() if split_masks[1] is not None else Im(torch.zeros_like(first_masks.torch))
            third_masks = Im(split_masks[2].unsqueeze(-1)).grid() if split_masks[2] is not None else Im(torch.zeros_like(first_masks.torch))
            Im.concat_vertical(
                first_hist,
                first_masks,
                second_masks,
                third_masks,
                spacing=120,
                fill=(128, 128, 128)
            ).save(f"{imgId}_masks.png")

        annotations, annotation_id = update_with_inference_single(seg_, imgId, annotations, annotation_id, img_.shape[:2])

    save_json(coco_root_dir, coco_slice_name, annotations, split_idx, prefix='hipie')

if __name__ == "__main__":
    fire.Fire(run)
