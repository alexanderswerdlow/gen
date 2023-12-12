from hydra.core.config_store import ConfigStore
from dataclasses import field, dataclass
from typing import List, Optional

from accelerate.utils import PrecisionType
from accelerate.utils.dataclasses import LoggerType, DynamoBackend

@dataclass
class TrainerConfig:
    mixed_precision: PrecisionType = PrecisionType.BF16
    gradient_accumulation_steps: int = 1
    log_with: Optional[LoggerType] = LoggerType.WANDB
    seed: int = 42
    max_steps: int = 4000
    num_epochs: int = 10
    validate_steps: int = 100
    eval_on_start: bool = True
    save_steps: int = 100
    limit_num_checkpoints: int = 1
    dynamo_backend: DynamoBackend = DynamoBackend.NO
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    set_grads_to_none: bool = False
    max_train_samples: Optional[int] = None
    validation_steps: int = 100
    enable_xformers_memory_efficient_attention: bool = True
    compile: bool = False

cs = ConfigStore.instance()
cs.store(group="trainer", name="base", node=TrainerConfig)