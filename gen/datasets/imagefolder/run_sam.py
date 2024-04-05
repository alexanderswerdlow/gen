from __future__ import annotations

import sys

from gen.configs.matrix_configs import get_excluded_nodes

sys.path.insert(0, "/home/aswerdlo/repos/gen")
import io
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import hydra
import msgpack
import numpy as np
import torch
import torch.nn.functional as F
from distributed import Client, get_worker, print
from joblib import Memory
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
import time

from gen import GLOBAL_CACHE_PATH, SCANNETPP_CUSTOM_DATA_PATH, SCANNETPP_DATASET_PATH, SCRATCH_CACHE_PATH
from gen.models.encoders.sam import HQSam
from gen.utils.data_defs import InputData, one_hot_to_integer
from gen.utils.decoupled_utils import breakpoint_on_error, get_time_sync, is_main_process, set_timing_builtins
from gen.utils.logging_utils import log_info
from image_utils import Im

if TYPE_CHECKING:
    from gen.configs.base import BaseConfig

import torch
from torchvision.ops import nms

def coco_decode_rle(compressed_rle) -> np.ndarray:
    from typing import Any, Dict

    import numpy as np

    if isinstance(compressed_rle['counts'], str):
        compressed_rle['counts'] = compressed_rle['counts'].encode()

    binary_mask = mask_utils.decode(compressed_rle)
    return binary_mask

from PIL import Image

from PIL import Image

def crop_and_resize(image, target_width=960, target_height=720):
    # Calculate the aspect ratios
    image_aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    if image_aspect_ratio > target_aspect_ratio:
        # Image is wider than the target aspect ratio
        # Resize based on height and crop the width
        new_height = target_height
        new_width = int(image.width * (new_height / image.height))
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)

        # Calculate the cropping coordinates
        left = (new_width - target_width) // 2
        right = left + target_width
        cropped_image = resized_image.crop((left, 0, right, target_height))
    else:
        # Image is taller than the target aspect ratio
        # Resize based on width and crop the height
        new_width = target_width
        new_height = int(image.height * (new_width / image.width))
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)

        # Calculate the cropping coordinates
        top = (new_height - target_height) // 2
        bottom = top + target_height
        cropped_image = resized_image.crop((0, top, target_width, bottom))

    return cropped_image

def signal_handler(signum, frame):
    raise KeyboardInterrupt

memory = Memory(GLOBAL_CACHE_PATH, verbose=0)

save_type_names = ("rgb", "masks", "seg_v0")

def get_all_gt_images(timestamp, data_path):
    return list((data_path / "orig_rgb").glob('*/*.jpg')) + list((data_path / "orig_rgb").glob('*/*.png'))

@memory.cache()
def get_all_saved_data(timestamp, data_path, subdir_names):
    log_info(f"Fetching saved data from {data_path}...")
    return [list(data_path.glob(f'{typename}/*/*')) for typename in subdir_names]

@memory.cache()
def get_to_process(timestamp, data_path, subdir_names, return_raw_data=False):
    orig_image_files = get_all_gt_images(timestamp, data_path)
    image_files = [(frame_path.parent.stem, frame_path.stem) for frame_path in orig_image_files]
    log_info(f"Got a total of {len(orig_image_files)} GT images ")

    saved_data = get_all_saved_data(timestamp, data_path, subdir_names)
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
            scene_path = data_path / type_name / scene_id
            scene_path.mkdir(parents=True, exist_ok=True)

    to_process = sorted([frame_path for frame_path in orig_image_files if (frame_path.parent.stem, frame_path.stem) in to_process])

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
        scene_name = frame_path.parent.stem
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

def rgba_to_foreground_background(image):
    image_array = np.array(image)
    
    alpha = image_array[:, :, 3]
    foreground = (alpha > 0).astype(np.uint8)
    
    background = 1 - foreground
    output = np.stack((foreground, background), axis=0)
    
    return output

def train(data_path, image_files, indices):
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
    elif "A100" in result or "6000 Ada" in result:
        raise ValueError(f"Unallowed GPU: {result}")
    elif "V100" in result:
        model_kwargs = {"compile_mode": "default", "dtype": torch.float16}
        points_per_batch = 256
        process_batch_size = 1
    elif "2080" in result:
        model_kwargs = {"compile_mode": "default", "dtype": torch.float16}
        points_per_batch = 96
        process_batch_size = 1
    else:
        raise ValueError(f"Unknown GPU type: {result}")

    set_timing_builtins(False, True)

    model = HQSam(
        model_type=model_type,
        process_batch_size=process_batch_size,
        points_per_batch=points_per_batch,
        points_per_side=12,
        output_mode="coco_rle",
        model_kwargs=model_kwargs,
    )
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    
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

                    rgb_path = (data_path / save_type_names[0] / scene_id / frame_name).with_suffix(".jpg")
                    mask_path = (data_path / save_type_names[1] / scene_id / frame_name).with_suffix(".msgpack")
                    seg_path = (data_path / save_type_names[2] / scene_id / frame_name).with_suffix(".png")

                    if not rgb_path.exists():
                        img = Image.open(Path(batch["tgt_path"]))
                        img = crop_and_resize(img, target_width=960, target_height=720)
                        if 'A' in img.getbands() or 'a' in img.getbands():
                            _masks = rgba_to_foreground_background(img).argmax(axis=0).astype(np.uint8)
                            Image.fromarray(_masks, mode='L').save(seg_path)
                            mask_path.touch()
                            img = img.convert("RGB")
                            img.save(rgb_path)
                            continue

                        img = img.convert("RGB")
                        img.save(rgb_path)
                    else:
                        img = Image.open(rgb_path)

                    masks = None
                    if not mask_path.exists():
                        masks = model(np.asarray(img))
                        masks = sorted(masks, key=lambda d: d["area"], reverse=True)
                        masks = masks[:max_masks]
                        
                        metadata = {k:v for k,v in batch["metadata"].items() if isinstance(v, str)}
                        masks_msgpack = msgpack.packb((metadata, masks), use_bin_type=True)

                        with open(mask_path, "wb") as f:
                            f.write(masks_msgpack)

                    
                    if not seg_path.exists():
                        if masks is None:
                            with open(mask_path, "rb") as f:
                                _, masks = msgpack.unpackb(f.read(), raw=False)
                        
                        rles = [x['segmentation'] for x in masks if x['area'] > 512]
                        seg = torch.cat([torch.from_numpy(mask_utils.decode(rle)).unsqueeze(0) for rle in rles], dim=0)
                        seg = one_hot_to_integer(seg.permute(1, 2, 0), num_overlapping_masks, assert_safe=False).permute(2, 0, 1)
                        Image.fromarray(seg[0].cpu().numpy(), mode='L').save(seg_path)

                    log_info(f"Processed {scene_id} {frame_name} ({i}/{len(dataset)})", main_process_only=False)
                    
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected. Cleaning up...", main_process_only=False)
        sys.exit(0)
    finally:
        pass

def tail_log_file(log_file_path, glob_str):
    max_retries = 60
    retry_interval = 2

    for _ in range(max_retries):
        try:
            if len(list(log_file_path.glob(glob_str))) > 0:
                try:
                    proc = subprocess.Popen(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"], stdout=subprocess.PIPE)
                    print(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"])
                    for line in iter(proc.stdout.readline, b''):
                        print(line.decode('utf-8'), end='')
                except:
                    proc.terminate()
        except:
            log_info(f"Tried to glob: {log_file_path}, {glob_str}")
        finally:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")


def run_slurm(data_path, num_chunks, num_workers, current_datetime, partition, chunk_size):
    log_info(f"Running slurm job with {num_chunks} chunks and {num_workers} workers...")
    from simple_slurm import Slurm

    kwargs = dict()
    if partition == 'all':
        kwargs['exclude'] = get_excluded_nodes("volta", "2080Ti")

    print(kwargs)
    slurm = Slurm(
        "--requeue=10",
        job_name='image_folder_parallel',
        cpus_per_task=8,
        mem='16g',
        export='ALL',
        gres=['gpu:1'],
        output=f'outputs/dask/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if 'kate' in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=f"0-{num_chunks-1}%{num_workers}",
        partition=partition,
        **kwargs
    )
    job_id = slurm.sbatch(f"python gen/datasets/imagefolder/run_sam.py {data_path} --is_slurm_task --slurm_task_datetme={current_datetime} --slurm_task_index=$SLURM_ARRAY_TASK_ID --chunk_size={chunk_size}")
    log_info(f"Submitted job {job_id} with {num_chunks} tasks and {num_workers} workers...")
    tail_log_file(Path(f"outputs/dask"), f"{job_id}*")

import typer
typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    data_path: Path,
    num_workers: int = 1,
    use_slurm: bool = False,
    is_slurm_task: bool = False,
    slurm_task_datetme: str = None,
    slurm_task_index: int = None,
    max_chunk_size: int = 20000,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    invalidate_cache: bool = False,
    partition: str = 'all',
    chunk_size: Optional[int] = None
):
    current_datetime = datetime.now()
    datetime_up_to_hour = current_datetime.strftime('%Y_%m_%d_%H_00_00') if use_slurm else current_datetime.strftime('%Y_%m_%d_00_00_00')
    _timestamp = slurm_task_datetme if is_slurm_task else datetime_up_to_hour
    if invalidate_cache or use_slurm:
        _timestamp = current_datetime.strftime('%Y_%m_%d_%H_%M_00')
    image_files = get_to_process(_timestamp, data_path, save_type_names)
    dataset = get_dataset(image_files=image_files)
    
    submission_list = list(range(len(dataset)))
    if len(submission_list) == 0:
        log_info("No images to process. Exiting...")
        exit()
    if shuffle:
        import random
        random.seed(shuffle_seed)
        random.shuffle(submission_list)

    if chunk_size is None:
        chunk_size = min(len(submission_list) // num_workers, max_chunk_size)  # Adjust this based on the number of workers
    chunks = [submission_list[i:i + chunk_size] for i in range(0, len(submission_list), chunk_size)]
    assert sum([len(chunk) for chunk in chunks]) == len(submission_list)

    if is_slurm_task:
        data_chunks = chunks[slurm_task_index]
        log_info(f"Running slurm task {slurm_task_index} with {len(data_chunks)} images...")
        train(data_path, image_files, data_chunks)
        exit()

    if use_slurm:
        run_slurm(data_path, len(chunks), num_workers, datetime_up_to_hour, partition, chunk_size)
        exit()
    else:
        import random
        random.shuffle(submission_list)

    with breakpoint_on_error():
        train(data_path, image_files, submission_list)
    
if __name__ == '__main__':
    app()