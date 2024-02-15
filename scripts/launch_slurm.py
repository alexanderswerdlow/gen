from typing import Optional

import os
import random
import subprocess
import sys
from pathlib import Path

import submitit
from gen import REPO_DIR

from dataclasses import dataclass
from typing import Optional
from rich.pretty import pprint

@dataclass(kw_only=True)
class SlurmConfig:
    init_cmds: Optional[str] = None
    output_dir: Optional[Path] = None
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
    max_num_timeout: int = 5
    use_deepspeed: bool = False
    exclude: Optional[str] = None
    constraint: str = "A100|6000ADA|A5500"
    env_vars: Optional[dict[str, str]] = None
    nodelist: Optional[str] = None
    comment: Optional[str] = None

def is_a5500_gpu():
    try:
        result = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        return "A5500" in result
    except subprocess.CalledProcessError:
        return False
    
def nvidia_smi_gpu_memory_stats():
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    try:
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        out_dict = {}
        for item in out_list:
            if " MiB" in item:
                gpu_idx, mem_used = item.split(",")
                gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
                out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024
    except FileNotFoundError:
        print("Failed to find the 'nvidia-smi' executable for printing GPU stats")
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def get_nvidia_smi_gpu_memory_stats_str():
    return f"nvidia-smi stats: {nvidia_smi_gpu_memory_stats()}"


DEEPSPEED_MULTINODE = "<is_deepspeed_multinode>"


class Task(submitit.helpers.Checkpointable):
    def __init__(self, cfg: SlurmConfig):
        self.cfg = cfg

    def __call__(self, *args, **kwargs):
        print(f"Running task on slurm")
        pprint(self.cfg)

        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        rng = random.Random(dist_env._job_env.job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env = dist_env.export()
        os.environ.update(
            **{
                "CUDA_LAUNCH_BLOCKING": "0",
                "OMP_NUM_THREADS": "1",
                "NCCL_P2P_DISABLE": "0",
                "PROJECT_ROOT": str(REPO_DIR),
                #"NCCL_DEBUG": "info",
            }
        )

        if self.cfg.env_vars is not None:
            os.environ.update(**self.cfg.env_vars)
        
        if is_a5500_gpu():
            os.environ["NCCL_P2P_DISABLE"] = "1"
            print("Running on A5500, setting NCCL_P2P_DISABLE=1")

        print(nvidia_smi_gpu_memory_stats())
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        print("Running training script")
        print(f"Local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"Removing CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']=}")
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        with open(Path(self.cfg.output_dir) / "env_vars.txt", "w") as file:
            for key, value in os.environ.items():
                file.write(f"{key}={value}\n")

        print(self.cfg)
        print(get_nvidia_smi_gpu_memory_stats_str())
        output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        print(output)

        num_processes = self.cfg.gpus * self.cfg.n_nodes
        machine_rank = dist_env.rank // self.cfg.n_processes
        deepspeed_str = ""
        if self.cfg.use_deepspeed:
            deepspeed_str = f" {DEEPSPEED_MULTINODE} --use_deepspeed"

        if self.cfg.use_accelerate:
            init_cmd = f"accelerate launch --dynamo_backend no --num_processes {num_processes} --gpu_ids all {deepspeed_str} --num_machines {self.cfg.n_nodes} --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port}"
        else:
            init_cmd = f"python"

        shell_init = f"{self.cfg.init_cmds} && " if self.cfg.init_cmds else ""
        cmd = f"{shell_init}{init_cmd} {self.cfg.program} {self.cfg.cmd}"

        if self.cfg.n_nodes > 1:
            hostfile_dir = "hostfiles"
            os.makedirs(hostfile_dir, exist_ok=True)
            hostfile = os.path.realpath(f"{hostfile_dir}/{dist_env._job_env.job_id}.txt")
            if dist_env.rank == 0:
                with open(hostfile, "w") as f:
                    for host in dist_env._job_env.hostnames:
                        f.write(f"{host} slots={self.cfg.n_processes}\n")
                print(f"Created hostfile: {hostfile}")
            cmd = cmd.replace(DEEPSPEED_MULTINODE, f"--deepspeed_hostfile {hostfile} --deepspeed_multinode_launcher standard")
        else:
            cmd = cmd.replace(DEEPSPEED_MULTINODE, "")

        if dist_env.local_rank == 0:
            print(f"Running command: {cmd}")
            exit_code = os.system(cmd)
        else:
            exit_code = 0
            print("Waiting for master to finish")
        if exit_code != 0:
            raise RuntimeError(f"Command {cmd} failed with exit code {exit_code}")

    def checkpoint(self, *args, **kwargs):
        print("Checkpointing task on slurm")
        print(args)
        print(kwargs)
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


global_jobs = []

def add_job(cfg: SlurmConfig):
    """
    This script uses submitit to execute a training run on a SLURM cluster. Although it takes a BaseConfig it only actually uses the SLURM submodule. The rest of the config is ignored.
    """
    cfg.cpus_per_task = cfg.cpus_per_task * cfg.gpus
    cfg.mem_gb = f"{int(cfg.mem_gb[:-2]) * cfg.gpus}{cfg.mem_gb[-2:]}"
    print(f"Autoscaling to {cfg.cpus_per_task} CPUs and {cfg.mem_gb}GB of memory")

    executor = submitit.AutoExecutor(folder=cfg.output_dir, max_num_timeout=cfg.max_num_timeout)

    slurm_additional_parameters = {
        "ntasks_per_node": cfg.n_processes,
        "constraint": cfg.constraint,
    }

    if cfg.nodelist is not None:
        slurm_additional_parameters["nodelist"] = cfg.nodelist

    print(f"SLURM additional parameters: {slurm_additional_parameters}")
    
    slurm_kwargs = {
        "slurm_job_name": cfg.job_name,
        "slurm_partition": cfg.partition,
        "slurm_nodes": cfg.n_nodes,
        "slurm_additional_parameters": slurm_additional_parameters,
        "slurm_cpus_per_task": cfg.cpus_per_task,
        "slurm_time": cfg.time,
        "slurm_exclude": cfg.exclude if cfg.exclude else "",
        "stderr_to_stdout": True,
        "slurm_mem": cfg.mem_gb,
        "slurm_gres": f"gpu:{cfg.gpus}",
        "slurm_comment": cfg.comment,
    }
    executor.update_parameters(**slurm_kwargs)

    task = Task(cfg)
    job = executor.submit(task, resume=False)
    global_jobs.append(job)
    return job

def watch():
    submitit.helpers.monitor_jobs(global_jobs)


if __name__ == "__main__":
    sys.exit(main())
