import os
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

COCO_CAPTIONS_FILES = os.getenv("COCO_CAPTIONS_PATH", "/projects/katefgroup/aswerdlo/mscoco/{00000..00059}.tar")
MOVI_DATASET_PATH = Path(os.getenv("MOVI_DATASET_PATH", "/projects/katefgroup/datasets/movi"))
MOVI_OVERFIT_DATASET_PATH = Path(os.getenv("MOVI_OVERFIT_DATASET_PATH", "/projects/katefgroup/aswerdlo/movi"))
MOVI_MEDIUM_PATH = Path(os.getenv("MOVI_DATASET_PATH", "/projects/katefgroup/datasets/movi"))
IMAGENET_PATH = Path(os.getenv("IMAGENET_PATH", "/projects/katefgroup/datasets/ImageNet"))
PLACEHOLDER_TOKEN: str = os.getenv("PLACEHOLDER_TOKEN", "masks")
DEFAULT_PROMPT: str = os.getenv("DEFAULT_PROMPT", f"A photo of {PLACEHOLDER_TOKEN}")
CONDA_ENV = Path(os.getenv("CONDA_ENV", "gen_nightly"))
