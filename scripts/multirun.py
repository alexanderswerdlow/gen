import random
import string
import autoroot

import itertools
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from gen import CONDA_ENV, REPO_DIR
from scripts.launch_slurm import add_job, SlurmConfig, watch
from rich.pretty import pprint

app = typer.Typer(pretty_exceptions_show_locals=False)
typer.main.get_command_name = lambda name: name


def sanitize_filename(filename: str) -> str:
    return "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-"))


def int_to_lowercase_letter(integer):
    return chr(ord("a") + integer)


@app.command()
def main(
    init_cmds: str = f"""cd {REPO_DIR} && source {Path.home() / 'anaconda3/bin/activate'} {CONDA_ENV}""",
    env_var: Annotated[Optional[List[str]], typer.Option()] = None,
    args: Annotated[Optional[List[str]], typer.Argument()] = None,
    prod: Annotated[Optional[List[str]], typer.Option()] = None,
    output_dir: Path = REPO_DIR / "outputs" / "slurm",
    name: str = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
    gpus: Optional[int] = None,
    big_gpu: bool = False,
    long_job: bool = False,
    dry_run: bool = False,
    nodelist: Optional[str] = None,
    constraint: Optional[str] = None,
    watch_run: bool = True,
):
    """
    This script is used to run experiments in parallel on a SLURM cluster. It is a wrapper around launch_slurm.py to support hyperparameter sweeps.
    """

    # TODO: The naming and directories are needlessly complicated. We should make things more uniform and orthogonal.

    is_sweep = os.environ.get("SWEEP") or (prod is not None and len(prod) > 0)
    multirun_name = os.environ.get("SWEEP") or name
    
    if not os.environ.get("SWEEP"):
        multirun_name = f'{multirun_name}_{"".join(random.choices(string.ascii_letters, k=2))}_{datetime.now().strftime("%m_%d")}'

    multirun_dir = output_dir / multirun_name
    regular_args = " ".join(args)

    if prod is None:
        data = {None: (None,)}
    else:
        data = {item.split("=")[0]: item.split("=")[1].split(",") for item in prod}
        keys = list(data.keys())

    env_vars = {var.split('=')[0]:var.split('=')[1] for var in env_var} if (env_var is not None and len(env_var) > 1) else {}

    # Generating and printing all combinations
    num_jobs = 0
    for idx, combination in enumerate(itertools.product(*data.values())):
        if len(combination) == 0:
            prod_args = ""
            if is_sweep:
                run_id = f'{name}_{datetime.now().strftime("%H%M")}_{"".join(random.choices(string.ascii_letters, k=2))}'
        else:
            prod_args = " " + " ".join([f"{keys[i]}={value}" for i, value in enumerate(combination)])
            run_id = sanitize_filename("_".join([f"{keys[i]}={value}" for i, value in enumerate(combination)])) + f'_{datetime.now().strftime("%H%M")}_{"".join(random.choices(string.ascii_letters, k=2))}'

        
        output_dir_ = multirun_dir / run_id if is_sweep else multirun_dir
        output_dir_.mkdir(parents=True, exist_ok=True)

        sweep_args = f"sweep_id={multirun_name} sweep_run_id={run_id} " if is_sweep else (f"exp={multirun_name} ")
        output_file_ = output_dir_ / "submitit.log"

        job = SlurmConfig(
            job_name=multirun_name,
            output_dir=output_dir_,
            cmd=f"{sweep_args}reference_dir={output_dir_} {regular_args}{prod_args}",
            env_vars=env_vars,
            init_cmds=init_cmds,
            nodelist=nodelist,
            comment=f"{run_id}",
        )

        if constraint is not None:
            job.constraint = constraint
        
        if gpus is not None:
            job.gpus = gpus

        if big_gpu:
            job.constraint = "A100|6000ADA"
        
        if long_job:
            job.time = "72:00:00"

        if not dry_run:
            with open(output_file_, "a") as f:
                f.write(f"Running: {repr(job)}\n")

            added_job = add_job(job)
            num_jobs += 1
        
        print("\n\n\n")
        print(f"Running Job {idx}: ")
        pprint(job)
        print("\n")
        log_file = output_dir_ / f"{added_job.job_id}_{added_job.task_id}_log.out"
        print(f"Output file: {log_file}")
        print(f"To view live output: tail -f {log_file}")
        
    if watch_run: 
        if num_jobs == 1:
            tail_log_file(log_file)
        else:
            watch()

import time
def tail_log_file(log_file_path):
    max_retries = 60
    retry_interval = 2

    for _ in range(max_retries):
        if os.path.exists(log_file_path):
            proc = subprocess.Popen(['tail', '-f', "-n", "+1", log_file_path], stdout=subprocess.PIPE)
            try:
                for line in iter(proc.stdout.readline, b''):
                    print(line.decode('utf-8'), end='')
            except KeyboardInterrupt:
                proc.terminate()
            break
        else:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")

if __name__ == "__main__":
    app()
