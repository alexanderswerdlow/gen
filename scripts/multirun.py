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


@app.command()
def main(
    init_cmds: str = f"""cd {REPO_DIR} && source {Path.home() / 'anaconda3/bin/activate'} {CONDA_ENV}""",
    args: Annotated[Optional[List[str]], typer.Argument()] = None,
    prod: Annotated[Optional[List[str]], typer.Option()] = None,
    output_dir: Path = REPO_DIR / "outputs" / "slurm" / f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
    dry_run: bool = False,
):
    regular_args = " ".join(args)

    if prod is None:
        data = {None: (None,)}
    else:
        data = {item.split("=")[0]: item.split("=")[1].split(",") for item in prod}
        keys = list(data.keys())

    # Generating and printing all combinations
    for combination in itertools.product(*data.values()):
        if len(combination) == 0:
            prod_args = ""
        else:
            prod_args = " " + " ".join([f"{keys[i]}={value}" for i, value in enumerate(combination)])

        run_id = "".join(random.choices(string.ascii_letters, k=5))
        output_dir_ = output_dir.parent / (output_dir.name + f"_{run_id}")
        output_dir_.mkdir(parents=True, exist_ok=True)
        output_file_ = output_dir_ / "submitit.log"
        command = f"""{init_cmds} && python scripts/launch_slurm.py output_dir={output_dir_} 'slurm.cmd="reference_dir={output_dir_} {regular_args}{prod_args}"' """

        if dry_run:
            print(command)
        else:
            with open(output_file_, "a") as f:
                f.write(f"Running: {command}\n")
            with open(output_file_, "a") as file:
                process = subprocess.Popen(command, shell=True, stdout=file, stderr=subprocess.STDOUT)

        print(f"Running: {command}")
        print(f"Output: {output_file_}")
        # Note: Processes are started and will run concurrently. Their outputs will be appended to the output file.


if __name__ == "__main__":
    app()
