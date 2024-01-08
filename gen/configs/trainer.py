from dataclasses import dataclass
from typing import ClassVar, Optional

from accelerate.utils import PrecisionType
from accelerate.utils.dataclasses import DynamoBackend, LoggerType

from gen.configs.utils import auto_store


@dataclass
class TrainerConfig:
    name: ClassVar[str] = 'trainer'
    mixed_precision: PrecisionType = PrecisionType.BF16
    gradient_accumulation_steps: int = 1
    log_with: Optional[LoggerType] = LoggerType.WANDB
    seed: int = 42
    num_train_epochs: int = 10
    limit_num_checkpoints: int = 1
    dynamo_backend: DynamoBackend = DynamoBackend.NO
    max_train_steps: Optional[int] = None
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    gradient_accumulation_steps: int = 1
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
    max_train_samples: Optional[int] = None
    eval_every_n_steps: Optional[int] = 100
    eval_every_n_epochs: Optional[int] = None
    enable_xformers_memory_efficient_attention: bool = True
    compile: bool = False
    profiler_active_steps: int = 2
    log_gradients: Optional[int] = None

auto_store(TrainerConfig, name="base")

