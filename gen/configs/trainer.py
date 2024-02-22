from dataclasses import dataclass
from typing import ClassVar, Optional

from accelerate.utils import PrecisionType
from accelerate.utils.dataclasses import DynamoBackend, LoggerType

from gen.configs.utils import auto_store
import torch

@dataclass
class TrainerConfig:
    name: ClassVar[str] = "trainer"
    log_with: Optional[LoggerType] = LoggerType.WANDB
    tracker_project_name: str = "controlnet"  # wandb project name
    mixed_precision: PrecisionType = PrecisionType.BF16
    dynamo_backend: DynamoBackend = DynamoBackend.NO

    gradient_accumulation_steps: int = 1
    
    seed: int = 42
    num_train_epochs: int = 10
    max_train_steps: Optional[int] = None # Global step

    checkpointing_steps: int = 10000
    checkpoints_total_limit: Optional[int] = None
    ckpt: Optional[str] = None
    resume: bool = False
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr_batch_size: bool = False
    scale_lr_gpus_grad_accum: bool = True
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    set_grads_to_none: bool = True
    eval_every_n_steps: Optional[int] = 100
    eval_every_n_epochs: Optional[int] = None
    eval_on_start: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    compile: bool = False
    profiler_active_steps: int = 2
    log_gradients: Optional[int] = None
    log_parameters: bool = False

    dynamic_grad_accum_default_gpus: int = 4
    enable_dynamic_grad_accum: bool = False

    save_accelerator_format: bool = False
    load_weights_only_no_state: bool = False
    finetune_learning_rate: Optional[float] = None
    detect_anomaly: bool = False
    find_unused_parameters: bool = False
    validate_training_dataset: bool = False # Whether to also run inference on the training dataset after the validation dataset

    custom_inference_every_n_steps: Optional[int] = None
    custom_inference_fixed_shuffle: bool = False
    custom_inference_batch_size: Optional[int] = None
    custom_inference_dataset_size: Optional[int] = 512

    # Set in code
    num_gpus: Optional[int] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    initial_learning_rate: Optional[float] = None


auto_store(TrainerConfig, name="base")