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
    name: str = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}',
    gpus: Optional[int] = None,
    big_gpu: bool = False,
    long_job: bool = False,
    dry_run: bool = False,
    nodelist: Optional[str] = None,
    constraint: Optional[str] = None,
):
    """
    This script is used to run experiments in parallel on a SLURM cluster. It is a wrapper around launch_slurm.py to support hyperparameter sweeps.
    """

    is_sweep = os.environ.get("SWEEP") is not None or (prod is not None and len(prod) > 0)
    name = os.environ.get("SWEEP", name + ("_sweep" if is_sweep else ""))
    sweep_dir = output_dir / name
    if sweep_dir.exists():
        sweep_dir = output_dir / (f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_' + name)
        name = f"{name}_" + "".join(random.choices(string.ascii_letters, k=3))
    regular_args = " ".join(args)

    if prod is None:
        data = {None: (None,)}
    else:
        data = {item.split("=")[0]: item.split("=")[1].split(",") for item in prod}
        keys = list(data.keys())

    env_vars = {var.split('=')[0]:var.split('=')[1] for var in env_var} if (env_var is not None and len(env_var) > 1) else {}

    # Generating and printing all combinations
    for idx, combination in enumerate(itertools.product(*data.values())):
        if len(combination) == 0:
            prod_args = ""
            if is_sweep:
                run_id = "".join(random.choices(string.ascii_letters, k=3))
        else:
            prod_args = " " + " ".join([f"{keys[i]}={value}" for i, value in enumerate(combination)])
            run_id = sanitize_filename("__".join([f"{keys[i]}={value}" for i, value in enumerate(combination)]))

        output_dir_ = sweep_dir
        if is_sweep:
            output_dir_ = output_dir_ / run_id
        output_dir_.mkdir(parents=True, exist_ok=True)

        sweep_args = f"sweep_id={name} sweep_run_id={run_id} " if is_sweep else (f"exp={name} ")
        output_file_ = output_dir_ / "submitit.log"
        append_run_id = f"_{run_id}" if is_sweep else ""

        job = SlurmConfig(
            job_name=f"{name}{append_run_id}",
            output_dir=output_dir_,
            cmd=f"{sweep_args}reference_dir={output_dir_} {regular_args}{prod_args}",
            env_vars=env_vars,
            init_cmds=init_cmds,
            nodelist=nodelist,
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

        print(f"Running Job {idx}: ")
        pprint(job)
        print("\n")
        log_file = output_dir_ / f"{added_job.job_id}_{added_job.task_id}_log.out"
        print(f"Output file: {log_file}")
        print("\n\n\n")

    watch()

if __name__ == "__main__":
    app()
