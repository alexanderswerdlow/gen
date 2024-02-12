import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import typer
from torch import Tensor

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)


def save_tensor_dict(tensor_dict: dict, path: Path):
    output_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, Tensor):
            if v.dtype == torch.float16 or v.dtype == torch.bfloat16:
                output_dict[k] = v.to(dtype=torch.float32).detach().cpu().numpy()
            else:
                output_dict[k] = v.detach().cpu().numpy()
        else:
            output_dict[k] = v
    np.savez_compressed(path, **output_dict)


def find_rgba_files(base_dir):
    for path in Path(base_dir).rglob("visual_geometry.obj"):
        if path.is_file():
            yield path, (path.parent)


@app.command()
def main(root_dir: Path = Path("data")):
    ret = {}
    for file_path, parent_dir in find_rgba_files(root_dir):
        mesh = o3d.io.read_triangle_mesh(str(file_path), enable_post_processing=True)
        mesh.paint_uniform_color([1, 0.706, 0])
        data = json.load(open(parent_dir / "data.json", "r"))
        ret[data["id"]] = np.asarray(mesh.vertices)

    save_tensor_dict(ret, Path("output"))


if __name__ == "__main__":
    app()
