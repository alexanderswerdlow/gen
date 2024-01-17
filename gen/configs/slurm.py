from dataclasses import MISSING, dataclass
from typing import ClassVar, Optional
from dataclasses import dataclass
from gen.configs.utils import auto_store

@dataclass
class SlurmConfig:
    name: ClassVar[str] = "slurm"
    cmd: Optional[str] = None
    program: str = "main.py"
    use_accelerate: bool = True

    job_name: str = "gen"
    partition: str = "kate_reserved" # kate_reserved, deepaklong
    time: str = "24:00:00"
    gpus: int = 1
    mem_gb: str = "48GB"
    cpus_per_task: int = 8
    n_processes: int = 1
    n_nodes: int = 1
    max_num_timeout: int = 100
    use_deepspeed: bool = False
    exclude: Optional[str] = None
    constraint: str = "A100|6000ADA|A5500"

auto_store(SlurmConfig, name="default")