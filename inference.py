from io import BytesIO
import pickle
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Union

import hydra
import numpy as np
from accelerate import Accelerator
from image_utils import Im
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import wandb
from gen.configs.base import BaseConfig
from gen.datasets.abstract_dataset import Split
from gen.models.utils import get_model_from_cfg
from gen.utils.decoupled_utils import all_gather, get_num_gpus, get_rank, is_main_process, sanitize_filename, save_tensor_dict
from gen.utils.logging_utils import log_debug, log_error, log_info
from gen.utils.trainer_utils import Trainable, TrainingState, load_from_ckpt, unwrap
import itertools
import time
import cv2
import traceback

tqdm.monitor_interval = 0 # May not be necessary

def inference(cfg: BaseConfig, accelerator: Accelerator):
    model = get_model_from_cfg(cfg)

    if not (cfg.debug and cfg.trainer.ckpt is None): load_from_ckpt(cfg=cfg, accelerator=accelerator, model=model, load_model=True)
    model = accelerator.prepare(model)

    if cfg.inference.infer_val_dataset: run_inference_split(cfg, accelerator, model, Split.VALIDATION, cfg.dataset.val)

    if cfg.inference.infer_train_dataset: run_inference_split(cfg, accelerator, model, Split.TRAIN, cfg.dataset.train)


def run_inference_split(cfg: BaseConfig, accelerator: Accelerator, model: Trainable, split: Split, dataset_cfg):
    g = torch.Generator()
    if cfg.inference.set_seed:
        g.manual_seed(42 + get_rank())
    else:
        g.manual_seed(int(time.time()) + get_rank())

    dataloader = hydra.utils.instantiate(dataset_cfg, _recursive_=True)(
        cfg=cfg, split=split, tokenizer=unwrap(model).tokenizer
    ).get_dataloader(generator=g)

    dataloader = accelerator.prepare(dataloader)

    run_inference_dataloader(
        cfg=cfg, accelerator=accelerator, state=TrainingState(0, 0, 0, 0, 0), dataloader=dataloader, model=model, output_path=cfg.output_dir, prefix=f'inference_{split.name.lower()}/'
    )


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


@torch.no_grad()
def run_inference_dataloader(
    accelerator: Optional[Accelerator],
    dataloader: DataLoader,
    model: Trainable,
    output_path: Path,
    state: TrainingState,
    prefix: Optional[str] = None,
    cfg: Optional[BaseConfig] = None,
    **kwargs,
):
    assert len(dataloader) > 0, "Dataloader is empty."
    
    output_path.mkdir(exist_ok=True, parents=True)
    log_debug(f"Setting inference mode.", main_process_only=False)
    unwrap(model).set_inference_mode(**kwargs)

    model.eval()
    log_info(f"Running inference w/prefix: {prefix}, Dataloder size: {len(dataloader)}", main_process_only=True)
    
    outputs = []
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process(), total=len(dataloader)):
        inference_state = TrainingState(
            epoch_step=i,
            num_epoch_steps=len(dataloader),
            epoch=state.epoch,
            global_step=state.global_step,
            true_step=state.true_step,
            split=Split.TRAIN if 'train' in prefix else Split.VALIDATION,
        )
        try:
            batch = unwrap(model).process_input(batch, state)
            batch = batch.to(accelerator.device)
            output = unwrap(model).run_inference(batch=batch, state=inference_state)
            outputs.append(output)
        except Exception as e:
            if get_num_gpus() > 1 and state.global_step > 100:
                log_error(f"Error during validation. Continuing...", main_process_only=False)
                traceback.print_exc()
                torch.cuda.empty_cache()
            else:
                raise

    log_debug(f"Before all_gather.", main_process_only=False)
    if cfg is None or cfg.inference.gather_results:
        del model; torch.cuda.empty_cache()
        outputs = all_gather(outputs)  # Combine over GPUs.
        outputs = flatten(outputs)  # Concat outputs from each GPU

    if is_main_process() or (cfg is not None and cfg.inference.gather_results is False):
        new_dict = defaultdict(list)
        for d in outputs:
            for k, v in d.items():
                new_dict[k].append(v)

        if prefix: new_dict = {f"{prefix}{k}":v for k,v in new_dict.items()}
        for k, v in sorted(new_dict.items()):
            if "wandb_" in k:
                _k = k.replace("wandb_", "")
                if isinstance(v[0], wandb.Histogram):
                    continue
                if "hist_" in k:
                    v_ = torch.cat(v, dim=0).numpy()
                    log_obj = wandb.Histogram(v_)
                else:
                    log_obj = sum(v) / len(v)
                
                accelerator.log({f"metrics_{_k}": log_obj}, step=state.global_step)
            elif "pred_" in k:
                if v[0].ndim == 0:
                    v_ = torch.stack(v).mean()
                else:
                    v_ = torch.cat(v, dim=0).mean()
                accelerator.log({k: v_}, step=state.global_step)
            elif 'knn' in k:
                v_ = {}
                for k_ in v[0].keys(): 
                    v_[k_] = torch.cat([x[k_] for x in v], dim=0)

                from tensordict import TensorDict
                tensordict = TensorDict(v_, [v_[k_].shape[0]])
                tensordict.memmap(output_path / k / str(get_rank()), num_threads=16)
                log_info(f"Saved mmap tensordict to {output_path / k}")
            elif isinstance(v[0], dict):
                v_ = {}
                for i in range(len(v)):
                    v_[str(i)] = v[i]
                save_tensor_dict(v_, path=output_path / sanitize_filename(f"{k}_{state.global_step}.npz"))
            elif isinstance(v[0], Iterable) and isinstance(next(iter(v[0])), BytesIO):
                log_video_with_accelerator(
                    accelerator=accelerator,
                    videos=list(itertools.chain(*v)),
                    save_folder=output_path,
                    name=k,
                    global_step=(state.global_step if state.global_step is not None else i),
                )
            else:
                output_images = [Im(im) for im in v]
                log_with_accelerator(
                    accelerator=accelerator,
                    images=output_images,
                    save_folder=output_path,
                    name=k,
                    global_step=(state.global_step if state.global_step is not None else i),
                    spacing=25,
                )
    log_info(f"Saved to {output_path}")

def log_video_with_accelerator(
    accelerator: Optional[Accelerator], videos: List[BytesIO], global_step: int, name: str, save_folder: Optional[Path] = None,
):
    save_folder.parent.mkdir(exist_ok=True)
    for i, v in enumerate(videos):
        save_path = save_folder / sanitize_filename(f"{name}_{global_step}_{i}.mp4")
        with open(save_path, "wb") as f:
            f.write(v.getbuffer())
    try:
        if accelerator is not None:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                        tracker.log({f"{name}_{i}": wandb.Video(v, format='mp4') for i,v in enumerate(videos)}, step=global_step)
    except Exception as e:
        print(e)
        log_info(f"Failed to log video to wandb: {save_folder}, name {name} with exception: {e}")


def log_with_accelerator(
    accelerator: Optional[Accelerator], images: List[Image.Image], global_step: int, name: str, save_folder: Optional[Path] = None, spacing: int = 15
):
    save_folder.parent.mkdir(exist_ok=True)

    if len(images) == 1:
        save_path = save_folder / sanitize_filename(f"{name}_{global_step}.png")
        Im(images[0]).save(save_path)
    else:
        for i in range(len(images)):
            try:
                save_path = save_folder / sanitize_filename(f"{name}_{global_step}_{i}.png")
                Im(images[i]).save(save_path)
            except Exception as e:
                print(e)
                log_info(f"Failed to save image: {save_path}, name {name} with exception: {e}")
    try:
        if accelerator is not None:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(name, np_images, global_step, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log({name: wandb.Image(Im.concat_horizontal(images, spacing=spacing, fill=(255, 255, 255)).pil)}, step=global_step)
    except Exception as e:
        print(e)
        log_info(f"Failed to log image: {save_path}, name {name} with exception: {e}")
