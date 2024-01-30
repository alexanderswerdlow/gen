from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from gen.configs.utils import auto_store, store_child_config
from gen.utils.trainer_utils import Trainable
from gen.models.cross_attn.base_inference import run_inference, run_custom_inference
from functools import partial

@dataclass
class InferenceConfig:
    name: ClassVar[str] = "inference"

    num_denoising_steps: int = 50
    guidance_scale: float = 7.5
    resolution: int = "${model.resolution}"
    
    num_images_per_prompt: int = 1 # Only applies to the primary generation
    num_masks_to_remove: Optional[int] = 4
    visualize_attention_map: bool = False
    visualize_embeds: bool = False
    infer_new_prompts: bool = False
    save_prompt_embeds: bool = False
    vary_cfg_plot: bool = False
    max_batch_size: int = 16

    set_seed: bool = False
    batched_cfg: bool = False # WARNING: This may silently break things
    empty_string_cfg: bool = True
    use_ddim: bool = False

    use_custom_pipeline: bool = True
    inference_func: Callable[[Trainable], None] = run_inference


auto_store(InferenceConfig, name="basemapper")
store_child_config(
    cls=InferenceConfig,
    group="inference",
    parent="basemapper",
    child="composable",
    inference_func = partial(run_custom_inference)
)
