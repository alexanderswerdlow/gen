from pathlib import Path
from typing import Iterable, List, Optional, Union

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import gather as accelerate_gather
from accelerate.utils import gather_object as accelerate_gather_object
from image_utils import Im
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen.configs.base import BaseConfig
from gen.datasets.base_dataset import Split
from gen.models.utils import get_model_from_cfg
from gen.utils.decoupled_utils import is_main_process
from gen.utils.logging_utils import log_info
from gen.utils.trainer_utils import Trainable, TrainingState


def inference(cfg: BaseConfig, accelerator: Accelerator):
    model = get_model_from_cfg(cfg)

    validation_dataloader = hydra.utils.instantiate(cfg.dataset.validation_dataset, _recursive_=False)(
        cfg=cfg, split=Split.VALIDATION, tokenizer=model.tokenizer, accelerator=accelerator
    ).get_dataloader()

    validation_dataloader, model = accelerator.prepare(validation_dataloader, model)

    run_inference_dataloader(
        accelerator=accelerator, state=TrainingState(0, 0, 0, 0), dataloader=validation_dataloader, model=model, output_path=cfg.output_dir
    )


def run_inference_dataloader(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: Trainable,
    output_path: Path,
    state: TrainingState,
):
    
    output_path.mkdir(exist_ok=True, parents=True)
    model.set_inference_mode()
    log_info(f"Running inference. Dataloder size: {len(dataloader)}")
    for i, batch in tqdm(enumerate(dataloader), leave=False, disable=not is_main_process()):
        output = model.run_inference(batch=batch, state=state)

    device = batch["gen_pixel_values"].device
    accelerator.wait_for_everyone()
    for k, v in sorted(output.items()):
        output_images = gather(device, v)
        log_with_accelerator(
            accelerator=accelerator,
            images=output_images,
            save_folder=output_path,
            name=k,
            global_step=(state.global_step if state.global_step is not None else i),
            spacing=25,
        )
        log_info(f"Saved to {output_path}")


def gather(device: torch.device, img: Union[Im, Iterable[Im]], gather_different: bool = False):
    if gather_different:
        ret = accelerate_gather_object(img)
    else:
        if isinstance(img, Iterable):
            tensor = torch.cat([img_.torch.to(device).unsqueeze(0) for img_ in img], dim=0)
        else:
            tensor = img.torch.to(device).unsqueeze(0)

        concat_tensor = accelerate_gather(tensor)

        try:
            ret = [Im(concat_tensor[i]) for i in range(concat_tensor.shape[0])]
        except:
            print(concat_tensor.shape, concat_tensor.dtype, concat_tensor.device)
            raise

    return ret


def log_with_accelerator(
    accelerator: Accelerator, images: List[Image.Image], global_step: int, name: str, save_folder: Optional[Path] = None, spacing: int = 15
):
    save_folder.parent.mkdir(exist_ok=True)

    if len(images) == 1:
        save_path = save_folder / f"{name}_{global_step}.png"
        Im(images[0]).save(save_path)
    else:
        for i in range(len(images)):
            save_path = save_folder / f"{name}_{global_step}_{i}.png"
            Im(images[i]).save(save_path)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(name, np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({name: wandb.Image(Im.concat_horizontal(images, spacing=spacing, fill=(255, 255, 255)).pil)}, step=global_step)
