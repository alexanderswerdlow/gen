from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from gen.configs.utils import auto_store


@dataclass
class InferenceConfig:
    name: ClassVar[str] = "inference"

    num_masks_to_remove: Optional[int] = 4
    visualize_attention_map: bool = False
    visualize_embeds: bool = False
    num_denoising_steps: int = 50
    guidance_scale: float = 7.5
    empty_string_cfg: bool = True
    resolution: int = "${model.resolution}"
    set_seed: bool = False
    batched_cfg: bool = False # WARNING: This may silently break things
    use_custom_pipeline: bool = True

    # Legacy Args
    # Specifies which checkpoint iteration we want to load
    iteration: Optional[str] = None
    # The input directory containing the saved models and embeddings
    input_dir: Optional[Path] = None
    # Where the save the inference results to
    inference_dir: Optional[Path] = None
    # Specific path to the mapper you want to load, overrides `input_dir`
    mapper_checkpoint_path: Optional[Path] = None
    # Specific path to the embeddings you want to load, overrides `input_dir`
    learned_embeds_path: Optional[Path] = None
    # List of prompts to run inference on
    prompts: Optional[List[str]] = None
    # Text file containing a prompts to run inference on (one prompt per line), overrides `prompts`
    prompts_file_path: Optional[Path] = None
    # List of random seeds to run on
    seeds: List[int] = field(default_factory=lambda: [42])
    # If you want to run with dropout at inference time, this specifies the truncation indices for applying dropout.
    # None indicates that no dropout will be performed. If a list of indices is provided, will run all indices.
    truncation_idxs: Optional[Union[int, List[int]]] = field(default_factory=lambda: [None])


auto_store(InferenceConfig, name="basemapper")
