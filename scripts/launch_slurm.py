import autoroot

import os
import random
import subprocess
import sys
from pathlib import Path

import hydra
import rich.syntax
import rich.tree
import submitit
from omegaconf import DictConfig, OmegaConf, open_dict
from gen import REPO_DIR
from gen.configs.base import BaseConfig
from gen.configs.slurm import SlurmConfig

from gen.utils.logging_utils import log_info


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
        log_info("Failed to find the 'nvidia-smi' executable for printing GPU stats")
    except subprocess.CalledProcessError as e:
        log_info(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def get_nvidia_smi_gpu_memory_stats_str():
    return f"nvidia-smi stats: {nvidia_smi_gpu_memory_stats()}"


def print_config(cfg: DictConfig):
    style = "bright"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    fields = cfg.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


DEEPSPEED_MULTINODE = "<is_deepspeed_multinode>"


class Task:
    def __init__(self, output_dir: Path, cfg: SlurmConfig):
        self.output_dir = output_dir
        self.cfg = cfg

    def __call__(self):
        print("Running task on slurm")
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        rng = random.Random(dist_env._job_env.job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env = dist_env.export()
        os.environ.update(
            **{
                "CUDA_LAUNCH_BLOCKING": "0",
                #    "NCCL_DEBUG": "info",
                "OMP_NUM_THREADS": "1",
                "NCCL_P2P_DISABLE": "0",
                "PROJECT_ROOT": str(REPO_DIR),
            }
        )
        print(nvidia_smi_gpu_memory_stats())
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        print("Running training script")
        print(f"Local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        with open(Path(self.output_dir) / "env_vars.txt", "w") as file:
            for key, value in os.environ.items():
                file.write(f"{key}={value}\n")

        print_config(self.cfg)
        print(get_nvidia_smi_gpu_memory_stats_str())
        output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        print(output)

        num_processes = self.cfg.n_processes * self.cfg.n_nodes
        machine_rank = dist_env.rank // self.cfg.n_processes
        deepspeed_str = ""
        if self.cfg.use_deepspeed:
            deepspeed_str = f" {DEEPSPEED_MULTINODE} --use_deepspeed"

        if self.cfg.use_accelerate:
            init_cmd = f"accelerate launch --dynamo_backend no --num_processes {num_processes} --gpu_ids all {deepspeed_str} --num_machines {self.cfg.n_nodes} --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port}"
        else:
            init_cmd = f"python"

        cmd = f"{init_cmd} {self.cfg.program} {self.cfg.cmd}"

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

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: BaseConfig):
    """
    This script uses submitit to execute a training run on a SLURM cluster. Although it takes a BaseConfig it only actually uses the SLURM submodule. The rest of the config is ignored.
    """
    if cfg.output_dir is None:
        with open_dict(cfg):
            cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    executor = submitit.AutoExecutor(folder=cfg.output_dir, max_num_timeout=cfg.slurm.max_num_timeout)

    print_config(cfg.slurm)

    slurm_additional_parameters = {
        "ntasks_per_node": cfg.slurm.n_processes,
        "constraint": cfg.slurm.constraint,
    }

    print(f"SLURM additional parameters: {slurm_additional_parameters}")

    slurm_kwargs = {
        "slurm_job_name": cfg.slurm.job_name,
        "slurm_partition": cfg.slurm.partition,
        "slurm_nodes": cfg.slurm.n_nodes,
        "slurm_additional_parameters": slurm_additional_parameters,
        "slurm_cpus_per_task": cfg.slurm.cpus_per_task,
        "slurm_time": cfg.slurm.time,
        "slurm_exclude": cfg.slurm.exclude if cfg.slurm.exclude else "",
        "stderr_to_stdout": True,
        "slurm_mem": cfg.slurm.mem_gb,
        "slurm_gres": f"gpu:{cfg.slurm.gpus}",
    }
    executor.update_parameters(**slurm_kwargs)

    task = Task(cfg.output_dir, cfg.slurm)
    job = executor.submit(task)
    submitit.helpers.monitor_jobs([job])


if __name__ == "__main__":
    sys.exit(main())
