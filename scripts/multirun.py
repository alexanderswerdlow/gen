import autoroot
import itertools
import subprocess
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated
from datetime import datetime
from gen import CONDA_ENV, REPO_DIR
import random
import string

app = typer.Typer(pretty_exceptions_show_locals=False)
typer.main.get_command_name = lambda name: name

def sanitize_filename(filename: str) -> str:
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-', '='))

@app.command()
def main(
    init_cmds: str = f"""cd {REPO_DIR} && source {Path.home() / 'anaconda3/bin/activate'} {CONDA_ENV}""",
    args: Annotated[Optional[List[str]], typer.Argument()] = None,
    prod: Annotated[Optional[List[str]], typer.Option()] = None,
    output_dir: Path = REPO_DIR / "outputs" / "slurm",
    sweep_name: str = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
    dry_run: bool = False,
):
    """
    This script is used to run experiments in parallel on a SLURM cluster. It is a wrapper around launch_slurm.py to support hyperparameter sweeps.
    """
    sweep_dir = output_dir / sweep_name
    regular_args = " ".join(args)

    if prod is None:
        data = {None: (None,)}
    else:
        data = {item.split("=")[0]: item.split("=")[1].split(",") for item in prod}
        keys = list(data.keys())

    # Generating and printing all combinations
    for idx, combination in enumerate(itertools.product(*data.values())):
        if len(combination) == 0:
            prod_args = ""
            run_id = idx
            # run_id = "".join(random.choices(string.ascii_letters, k=3))
        else:
            prod_args = " " + " ".join([f"{keys[i]}={value}" for i, value in enumerate(combination)])
            run_id = sanitize_filename(" ".join([f"{keys[i]}={value}" for i, value in enumerate(combination)]))

        run_id = f"sweep_{run_id}"
        output_dir_ = sweep_dir / run_id
        output_dir_.mkdir(parents=True, exist_ok=True)
        output_file_ = output_dir_ / "submitit.log"
        command = f"""{init_cmds} && python scripts/launch_slurm.py output_dir={output_dir_} 'slurm.cmd="sweep_id={sweep_name} sweep_run_id={run_id} reference_dir={output_dir_} {regular_args}{prod_args}"' """

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

if __name__ == "__main__":
    app()
