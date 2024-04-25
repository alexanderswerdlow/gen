from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Type

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from accelerate.utils import PrecisionType
from accelerate.utils.dataclasses import DynamoBackend, LoggerType

from gen.configs.utils import auto_store
import torch
from torch.optim import Optimizer

if TYPE_CHECKING:
    from train import Trainer
    from gen.models.base.base_trainer import BaseTrainer

@dataclass
class TrainerConfig:
    trainer_cls: str = "gen.models.base.base_trainer.BaseTrainer"
    name: ClassVar[str] = "trainer"
    log_with: Optional[LoggerType] = LoggerType.WANDB
    tracker_project_name: str = "controlnet"  # wandb project name
    mixed_precision: PrecisionType = PrecisionType.BF16
    dynamo_backend: DynamoBackend = DynamoBackend.NO

    gradient_accumulation_steps: int = 1
    
    seed: int = 42
    num_train_epochs: int = 10
    max_train_steps: Optional[int] = None # Global step

    eval_steps: Optional[int] = 100
    eval_epochs: Optional[int] = None
    eval_on_start: bool = True
    ckpt_steps: Optional[int] = 10000
    checkpoints_total_limit: Optional[int] = None
    ckpt: Optional[str] = None
    resume: bool = False
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr_batch_size: bool = False
    scale_lr_gpus_grad_accum: bool = True
    optimizer_cls: Optimizer = torch.optim.AdamW
    use_fused_adam: bool = False
    use_8bit_adam: bool = True
    momentum: Optional[float] = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    set_grads_to_none: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    compile: bool = False
    profiler_active_steps: int = 2
    profiler_warmup_steps: int = 5
    profile_memory: bool = False
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
    validate_validation_dataset: bool = True
    strict_load: bool = True
    wandb_log_code: bool = True

    custom_inference_every_n_steps: Optional[int] = None
    custom_inference_fixed_shuffle: bool = False
    custom_inference_batch_size: Optional[int] = None
    custom_inference_dataset_size: Optional[int] = 512

    compose_inference: bool = False

    enable_timing: bool = False
    enable_timing_sync: bool = True
    fast_eval: bool = False
    eval_decay_steps: bool = True
    cudnn_benchmark: bool = True
    init_pipeline_inference: bool = True
    backward_pass: bool = True
    fsdp: bool = False
    load_accelerator_state: bool = True
    ignore_clip_weights: bool = False
    set_even_batches_false: bool = False

    # Set in code
    num_gpus: Optional[int] = None
    dtype: Optional[torch.dtype] = "${get_dtype:trainer}"
    device: Optional[str] = None
    initial_learning_rate: Optional[float] = None


auto_store(TrainerConfig, name="base")