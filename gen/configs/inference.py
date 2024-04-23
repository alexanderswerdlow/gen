from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from gen.configs.utils import auto_store, store_child_config
from gen.datasets.run_dataloader import iterate_dataloader
from gen.models.cross_attn.base_inference import run_qualitative_inference
from gen.utils.trainer_utils import Trainable


@dataclass
class InferenceConfig:
    name: ClassVar[str] = "inference"

    num_denoising_steps: int = 50
    guidance_scale: float = 7.5
    resolution: int = "${model.decoder_resolution}"
    
    num_images_per_prompt: int = 1 # Only applies to the primary generation
    num_masks_to_remove: Optional[int] = 4
    vary_cfg_plot: bool = False
    max_batch_size: int = 16
   
    set_seed: bool = False
    batched_cfg: bool = False # WARNING: This may silently break things
    use_ddim: bool = False

    use_custom_pipeline: bool = True
    inference_func: Callable[[Trainable], None] = run_qualitative_inference
    infer_val_dataset: bool = True
    infer_train_dataset: bool = False
    dataloader_only_func: Callable[..., None] = iterate_dataloader
    gather_results: bool = True
    tta: bool = False


auto_store(InferenceConfig, name="basemapper")