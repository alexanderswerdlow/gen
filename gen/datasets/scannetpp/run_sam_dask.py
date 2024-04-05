from __future__ import annotations

import sys

from altair import Optional

sys.path.insert(0, "/home/aswerdlo/repos/gen")
import io
import os
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import hydra
import msgpack
import numpy as np
import torch
import torch.nn.functional as F
from distributed import Client, get_worker, print
from joblib import Memory
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.ops import nms
from tqdm import tqdm

from gen import GLOBAL_CACHE_PATH, SCANNETPP_CUSTOM_DATA_PATH, SCANNETPP_DATASET_PATH, SCRATCH_CACHE_PATH
from gen.models.encoders.sam import HQSam
from gen.utils.data_defs import InputData, one_hot_to_integer
from gen.utils.decoupled_utils import breakpoint_on_error, get_time_sync, is_main_process, set_timing_builtins
from gen.utils.logging_utils import log_info
from image_utils import Im

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

def coco_decode_rle(compressed_rle) -> np.ndarray:
    from typing import Any, Dict

    import numpy as np

    if isinstance(compressed_rle['counts'], str):
        compressed_rle['counts'] = compressed_rle['counts'].encode()

    binary_mask = mask_utils.decode(compressed_rle)
    return binary_mask

def postprocess(masks):
    from pycocotools import mask as mask_utils
    from segment_anything_fast.utils.amg import batched_mask_to_box, remove_small_regions
    from torchvision.ops.boxes import batched_nms, box_area

    rles = [x['segmentation'] for x in masks]
    masks = torch.cat([torch.from_numpy(mask_utils.decode(rle)).unsqueeze(0) for rle in rles], dim=0)
    
    avg_size = (masks.shape[1] + masks.shape[2]) / 2
    min_area = int(avg_size**2 * 0.0001)

    nms_thresh = 0.93
    new_masks = []
    scores = []

    for rle in rles:
        mask = mask_utils.decode(rle)
        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        scores.append(float(unchanged))

    masks = torch.cat(new_masks, dim=0)

    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),
        iou_threshold=nms_thresh,
    )
    masks = masks[keep_by_nms]

    small_masks = torch.sum(masks, dim=(1, 2))
    masks = masks[small_masks > min_area]
    return masks

def signal_handler(signum, frame):
    raise KeyboardInterrupt

memory = Memory(GLOBAL_CACHE_PATH, verbose=0)

save_data_path = SCANNETPP_CUSTOM_DATA_PATH / "data_v1"
save_type_names = ("rgb", "masks", "seg_v1")

@memory.cache()
def get_all_gt_images():
    log_info(f"Fetching all GT images from {SCANNETPP_DATASET_PATH}...")
    return list(SCANNETPP_DATASET_PATH.glob('*/iphone/rgb/*.jpg'))

@memory.cache()
def get_all_saved_data(timestamp, subdir_names):
    log_info(f"Fetching saved data from {save_data_path}...")
    return [list(save_data_path.glob(f'{typename}/*/*')) for typename in subdir_names]

@memory.cache()
def get_to_process(timestamp, subdir_names, return_raw_data=False):
    orig_image_files = get_all_gt_images()
    image_files = [(frame_path.parent.parent.parent.stem, frame_path.stem) for frame_path in orig_image_files]
    log_info(f"Got a total of {len(orig_image_files)} GT images ")

    saved_data = get_all_saved_data(timestamp, subdir_names)
    saved_data = [set([(frame_path.parent.stem, frame_path.stem) for frame_path in saved_files]) for saved_files in saved_data]
    for typename, saved_data_ in zip(subdir_names, saved_data):
        log_info(f"Got a total of {len(saved_data_)} saved {typename} images")

    image_files = set(image_files)
    saved_data = set.intersection(*saved_data)
    if return_raw_data:
        return image_files, saved_data
    
    to_process = image_files - saved_data

    unique_scene_ids = set([scene_id for (scene_id, frame_name) in to_process])
    for type_name in subdir_names:
        for scene_id in unique_scene_ids:
            scene_path = save_data_path / type_name / scene_id
            scene_path.mkdir(parents=True, exist_ok=True)

    to_process = sorted([frame_path for frame_path in orig_image_files if (frame_path.parent.parent.parent.stem, frame_path.stem) in to_process])

    log_info(f"Got a total of {len(to_process)} images to process.")
    
    return to_process

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        frame_path = Path(self.image_files[index])
        frame_name = frame_path.stem
        scene_name = frame_path.parent.parent.parent.stem
        frame_idxs = (index,)
        name = f"{scene_name}_{frame_name}"
        return {
            "tgt_path": str(self.image_files[index]),
            "metadata": {
                "dataset": "scannetpp",
                "name": name,
                "scene_id": scene_name,
                "camera_frame": frame_name,
                "index": index,
                "camera_trajectory": "0", # Dummy value
                "frame_idxs": frame_idxs,
            },
        }

def get_dataset(**kwargs):
    dataset = CustomDataset(**kwargs)
    return dataset


def train(image_files, indices, only_keyframes=False):
    log_info(f"Initializing...", main_process_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        job_id = os.getenv('SLURM_JOB_ID')
        addr = get_worker().address if hasattr(get_worker(), 'address') else None
        info_str = f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}"
        log_info(f"Starting inference on {info_str}", main_process_only=False)
    except:
        pass

    dataset = get_dataset(image_files=image_files)

    model_kwargs = {}
    result = subprocess.check_output("nvidia-smi -L", shell=True).decode()

    points_per_batch: int = 256
    process_batch_size: int = 1
    model_type: str = "vit_h"
    max_masks: int = 256
    num_overlapping_masks: int = 1
    batch_size = 1

    if "A5500" in result:
        points_per_batch: int = 128
        # raise ValueError(f"Unallowed GPU: {result}")
    elif "A100" in result or "6000 Ada" in result:
        raise ValueError(f"Unallowed GPU: {result}")
    elif "V100" in result:
        model_kwargs = {"compile_mode": "default", "dtype": torch.float16}
        points_per_batch = 256
        process_batch_size = 1
    elif "2080" in result:
        model_kwargs = {"compile_mode": "default", "dtype": torch.float16}
        points_per_batch = 90
        process_batch_size = 1
    elif "1080" in result or "TITAN" in result or "P40" in result:
        model_kwargs = {"compile_mode": "default", "dtype": torch.float16}
        points_per_batch = 86
        process_batch_size = 1
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    else:
        raise ValueError(f"Unknown GPU type: {result}")

    set_timing_builtins(False, True)

    model = None
    def get_model():
        _model = HQSam(
            model_type=model_type,
            process_batch_size=process_batch_size,
            points_per_batch=points_per_batch,
            points_per_side=32,
            output_mode="coco_rle",
            model_kwargs=model_kwargs,
        )
        _model.requires_grad_(False)
        _model.eval()
        _model = _model.to(device)
        return _model
    
    g = torch.Generator()
    g.manual_seed(int(time.time()))
    dataset = torch.utils.data.Subset(dataset, indices)
    batch: InputData
    try:
        for i, batch in tqdm(enumerate(dataset), leave=False, disable=not is_main_process(), total=len(dataset)):
            with get_time_sync(enable=False):
                for b in range(batch_size):
                    scene_id = batch["metadata"]['scene_id']
                    frame_name = batch["metadata"]['camera_frame']
                    rgb_path = (save_data_path / "rgb" / scene_id / frame_name).with_suffix(".jpg")

                    if not rgb_path.exists():
                        if not Path(batch["tgt_path"]).exists():
                            raise FileNotFoundError(f"File not found: {batch['tgt_path']}")

                        try:
                            img = Image.open(Path(batch["tgt_path"]))
                        except Exception as e:
                            log_info(f"Error opening {batch['tgt_path']}: {e}")
                            continue

                        img = img.resize((img.width // 2, img.height // 2), Image.BICUBIC)
                        img.save(rgb_path)
                        if only_keyframes: continue
                    else:
                        img = Image.open(rgb_path)

                    if only_keyframes: continue

                    mask_path = (save_data_path / "masks" / scene_id / frame_name).with_suffix(".msgpack")
                    masks = None
                    if not mask_path.exists():
                        if model is None:
                            model = get_model()
                        masks = model(np.asarray(img))
                        masks = sorted(masks, key=lambda d: d["area"], reverse=True)
                        masks = masks[:max_masks]
                        
                        metadata = {k:v for k,v in batch["metadata"].items() if isinstance(v, str)}
                        masks_msgpack = msgpack.packb((metadata, masks), use_bin_type=True)

                        with open(mask_path, "wb") as f:
                            f.write(masks_msgpack)

                    seg_path = (save_data_path / "seg_v1" / scene_id / frame_name).with_suffix(".png")
                    if not seg_path.exists():
                        if masks is None:
                            with open(mask_path, "rb") as f:
                                _, masks = msgpack.unpackb(f.read(), raw=False)
                        
                        rles = [x['segmentation'] for x in masks if x['area'] > 256]
                        if len(rles) == 0:
                            log_info(f"No masks found for {scene_id} {frame_name} ({i}/{len(dataset)})", main_process_only=False)
                            log_info(f"Area is {sum([x['area'] for x in masks])}")
                        else:
                            seg = torch.cat([torch.from_numpy(mask_utils.decode(rle)).unsqueeze(0) for rle in rles], dim=0)
                            seg = one_hot_to_integer(seg.permute(1, 2, 0), num_overlapping_masks, assert_safe=False).permute(2, 0, 1)
                            Image.fromarray(seg[0].cpu().numpy(), mode='L').save(seg_path)

                    log_info(f"Processed {scene_id} {frame_name} ({i}/{len(dataset)})", main_process_only=False)
                    
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected. Cleaning up...", main_process_only=False)
        sys.exit(0)
    finally:
        pass

    log_info("Finished processing all images.", main_process_only=False)

def tail_log_file(log_file_path, glob_str):
    max_retries = 60
    retry_interval = 2

    for _ in range(max_retries):
        if len(list(log_file_path.glob(glob_str))) > 0:
            proc = subprocess.Popen(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"], stdout=subprocess.PIPE)
            try:
                for line in iter(proc.stdout.readline, b''):
                    print(line.decode('utf-8'), end='')
            except KeyboardInterrupt:
                proc.terminate()
            break
        else:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")


node_gpus = {
    "matrix-0-16": "titanx",
    "matrix-0-18": "titanx",
    "matrix-0-24": "P40,volta",
    "matrix-0-26": "titanx",
    "matrix-0-36": "2080Ti",
    "matrix-1-1": "volta",
    "matrix-1-6": "2080Ti",
    "matrix-1-10": "2080Ti",
    "matrix-1-14": "volta",
    "matrix-1-16": "volta",
    "matrix-1-18": "titanx",
    "matrix-1-22": "2080Ti",
    "matrix-1-24": "volta",
    "matrix-2-1": "2080Ti",
    "matrix-2-25": "A100",
    "matrix-2-29": "A100",
    "matrix-3-18": "6000ADA",
    "matrix-3-22": "6000ADA",
    "matrix-0-34": "2080Ti",
    "matrix-0-22": "titanx",
    "matrix-0-28": "titanx",
    "matrix-0-38": "titanx",
    "matrix-1-4": "2080Ti",
    "matrix-1-8": "2080Ti",
    "matrix-1-12": "2080Ti",
    "matrix-1-20": "titanx",
    "matrix-2-3": "2080Ti",
    "matrix-2-5": "2080Ti",
    "matrix-2-7": "2080Ti",
    "matrix-2-9": "2080Ti",
    "matrix-2-11": "2080Ti",
    "matrix-2-13": "2080Ti",
    "matrix-2-15": "2080Ti",
    "matrix-2-17": "2080Ti",
    "matrix-2-19": "2080Ti",
    "matrix-2-21": "2080Ti",
    "matrix-2-23": "2080Ti",
    "matrix-3-13": "1080Ti",
    "matrix-2-33": "3090",
    "matrix-2-37": "3090",
    "matrix-3-26": "A5500",
    "matrix-3-28": "A5500"
}

def run_slurm(num_chunks, num_workers, current_datetime, partition):
    log_info(f"Running slurm job with {num_chunks} chunks and {num_workers} workers...")
    from simple_slurm import Slurm

    kwargs = dict()
    if partition == 'all':
        # "volta", 
        kwargs['exclude'] = ",".join([x for x in node_gpus.keys() if not any(s in node_gpus[x] for s in ("2080Ti", "titanx", "P40", "1080Ti"))])

    print(kwargs)
    slurm = Slurm(
        "--requeue=10",
        job_name='seg_parallel',
        cpus_per_task=2,
        mem='16g',
        export='ALL',
        gres=['gpu:1'],
        output=f'outputs/dask/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if 'kate' in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=f"0-{num_chunks-1}%{num_workers}",
        partition=partition,
        **kwargs
    )

    job_id = slurm.sbatch(f"python gen/datasets/scannetpp/run_sam_dask.py --is_slurm_task --slurm_task_datetme={current_datetime} --slurm_task_index=$SLURM_ARRAY_TASK_ID")
    log_info(f"Submitted job {job_id} with {num_chunks} tasks and {num_workers} workers...")
    tail_log_file(Path(f"outputs/dask"), f"{job_id}*")

import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    num_workers: int = 1,
    use_slurm: bool = False,
    is_slurm_task: bool = False,
    slurm_task_datetme: str = None,
    slurm_task_index: int = None,
    max_chunk_size: int = 2000,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    partition: str = 'all',
    only_keyframes: bool = False,
    keyframe_path: Optional[Path] = '/home/aswerdlo/repos/gen/gen/datasets/scannetpp/scripts/all_image_file_paths.txt',
):
    if only_keyframes:
        with open(keyframe_path, 'r') as f:
            image_files = [line.strip() for line in f.readlines()]
    else:
        current_datetime = datetime.now()
        datetime_up_to_hour = current_datetime.strftime('%Y_%m_%d_%H_00_00') if use_slurm else current_datetime.strftime('%Y_%m_%d_00_00_00')
        image_files = get_to_process(slurm_task_datetme if is_slurm_task else datetime_up_to_hour, save_type_names)

    dataset = get_dataset(image_files=image_files)
    
    submission_list = list(range(len(dataset)))
    if shuffle:
        random.seed(shuffle_seed)
        random.shuffle(submission_list)

    chunk_size = len(submission_list) // num_workers  # Adjust this based on the number of workers
    chunks = [submission_list[i:i + min(chunk_size, max_chunk_size)] for i in range(0, len(submission_list), min(chunk_size, max_chunk_size))]
    assert sum([len(chunk) for chunk in chunks]) == len(submission_list)

    if is_slurm_task:
        data_chunks = chunks[slurm_task_index]
        log_info(f"Running slurm task {slurm_task_index} with {len(data_chunks)} images...")
        train(image_files, data_chunks, only_keyframes=only_keyframes)
        exit()

    if use_slurm:
        num_workers = min(num_workers, len(chunks))
        run_slurm(len(chunks), num_workers, datetime_up_to_hour, partition)
        exit()
    else:
        random.shuffle(submission_list)

    with breakpoint_on_error():
        train(image_files, submission_list, only_keyframes=only_keyframes)

if __name__ == '__main__':
    app()