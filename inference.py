import pickle
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Union

import hydra
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from image_utils import Im
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from gen.configs.base import BaseConfig
from gen.datasets.base_dataset import Split
from gen.models.utils import get_model_from_cfg
from gen.utils.decoupled_utils import get_rank, is_main_process, save_tensor_dict
from gen.utils.logging_utils import log_info
from gen.utils.trainer_utils import Trainable, TrainingState, load_from_ckpt, unwrap


def inference(cfg: BaseConfig, accelerator: Accelerator):
    model = get_model_from_cfg(cfg)

    validation_dataloader = hydra.utils.instantiate(cfg.dataset.validation_dataset, _recursive_=True)(
        cfg=cfg, split=Split.VALIDATION, tokenizer=model.tokenizer
    ).get_dataloader()

    load_from_ckpt(cfg=cfg, accelerator=accelerator, model=model)

    validation_dataloader, model = accelerator.prepare(validation_dataloader, model)

    run_inference_dataloader(
        accelerator=accelerator, state=TrainingState(0, 0, 0, 0), dataloader=validation_dataloader, model=model, output_path=cfg.output_dir
    )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def run_inference_dataloader(
    accelerator: Optional[Accelerator],
    dataloader: DataLoader,
    model: Trainable,
    output_path: Path,
    state: TrainingState,
):
    output_path.mkdir(exist_ok=True, parents=True)
    unwrap(model).set_inference_mode()
    log_info(f"Running inference. Dataloder size: {len(dataloader)}")
    outputs = []
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        output = unwrap(model).run_inference(batch=batch, state=state)
        outputs.append(output)

    outputs = all_gather(outputs)  # Combine over GPUs.
    outputs = flatten(outputs)  # Concat outputs from each GPU

    if is_main_process():
        new_dict = defaultdict(list)
        for d in outputs:
            for k, v in d.items():
                new_dict[k].append(v)
        for k, v in sorted(new_dict.items()):
            if isinstance(v[0], dict):
                for i in range(len(v)):
                    save_tensor_dict(v[i], path=output_path / f"{k}_{state.global_step}_{i}.npz")
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


def log_with_accelerator(
    accelerator: Optional[Accelerator], images: List[Image.Image], global_step: int, name: str, save_folder: Optional[Path] = None, spacing: int = 15
):
    save_folder.parent.mkdir(exist_ok=True)

    if len(images) == 1:
        save_path = save_folder / f"{name}_{global_step}.png"
        Im(images[0]).save(save_path)
    else:
        for i in range(len(images)):
            save_path = save_folder / f"{name}_{global_step}_{i}.png"
            Im(images[i]).save(save_path)

    if accelerator is not None:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(name, np_images, global_step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log({name: wandb.Image(Im.concat_horizontal(images, spacing=spacing, fill=(255, 255, 255)).pil)}, step=global_step)
