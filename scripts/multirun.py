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
    slurm: Annotated[Optional[List[str]], typer.Option()] = None,
    output_dir: Path = REPO_DIR / "outputs" / "slurm",
    job_name: str = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}',
    gpus: Optional[int] = None,
    big_gpu: bool = False,
    long_job: bool = False,
    dry_run: bool = False,
):
    """
    This script is used to run experiments in parallel on a SLURM cluster. It is a wrapper around launch_slurm.py to support hyperparameter sweeps.
    """

    is_sweep = os.environ.get("SWEEP") is not None or prod is not None
    job_name = os.environ.get("SWEEP", job_name + ("_sweep" if is_sweep else ""))
    sweep_dir = output_dir / job_name
    regular_args = " ".join(args)

    if prod is None:
        data = {None: (None,)}
    else:
        data = {item.split("=")[0]: item.split("=")[1].split(",") for item in prod}
        keys = list(data.keys())

    custom_slurm_cmd = " ".join((f"slurm.{k}" for k in slurm)) if slurm is not None else ""
    gpu_args = f"slurm.gpus={gpus} " if gpus is not None else ""
    constraint_args = 'slurm.constraint="A100|6000ADA" ' if big_gpu else ""
    timeout_args = 'slurm.time="72:00:00" ' if long_job else ""
    launch_args = f"""{gpu_args}{constraint_args}{timeout_args} 'slurm.job_name="{job_name}"' {custom_slurm_cmd}"""
    env_vars = (" && ".join((f"export {var}=\'{os.environ[var]}\'" for var in env_var))) + " && " if (env_var is not None and len(env_var) > 1) else ""

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

        sweep_args = f"sweep_id={job_name} sweep_run_id={run_id} " if is_sweep else ""
        output_file_ = output_dir_ / "submitit.log"
        command = f"""{env_vars}{init_cmds} && python scripts/launch_slurm.py {launch_args} 'output_dir="{output_dir_}"' 'slurm.cmd="{sweep_args}reference_dir={output_dir_} {regular_args}{prod_args}"' """

        if dry_run:
            print(command)
        else:
            with open(output_file_, "a") as f:
                f.write(f"Running: {command}\n")
            with open(output_file_, "a") as file:
                # Note: Processes are started and will run concurrently. Their outputs will be appended to the output file.
                process = subprocess.Popen(command, shell=True, stdout=file, stderr=subprocess.STDOUT)

        print(f"Running: {command}")
        print(f"Output: {output_file_}")
        print("\n\n")


if __name__ == "__main__":
    app()
