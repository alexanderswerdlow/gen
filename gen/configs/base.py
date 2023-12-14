from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict, Optional
from omegaconf import DictConfig, MISSING
from gen.configs.datasets import DatasetConfig
from gen.configs.models import ModelConfig
from gen.configs.trainer import TrainerConfig

def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def instantiate_with_cfg(cfg: DictConfig, **kwargs):
    target = _locate(cfg._target_)
    return target(cfg, **kwargs)


defaults = [
    {"trainer": "base"},
    {"dataset": "huggingface"},
    {"model": "basemapper"},
]

@dataclass
class BaseConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: TrainerConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    top_level_output_path: Optional[Path] = Path("outputs")
    logging_dir: Optional[Path] = Path("logs") # Folder inside the experiment folder
    exp: Optional[str] = None
    debug: bool = False
    overfit: bool = False
    tracker_project_name: str = "controlnet" # wandb project name
    tags: Optional[tuple[str]] = None
    attach: bool = False
    profile: bool = False

    # These are set in code
    output_dir: Optional[Path] = None
    run_name: Optional[str] = None
    cwd: Optional[Path] = None