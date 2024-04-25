
import os
from pathlib import Path
from gen.utils.logging_utils import log_error, log_info

if os.getenv("IS_SSD", "1") == "0": log_error("Running on HDD. Matrix is irrevocably broken. Exiting."); exit()

os.environ["EINX_WARN_ON_RETRACE"] = "25"
os.environ["IMAGE_UTILS_DISABLE_WARNINGS"] = "1"

REPO_DIR = Path(__file__).parent.parent
PROJECTS_PREFIX = str(os.getenv("PROJECTS_PREFIX", ""))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
GLOBAL_CACHE_PATH = Path(os.getenv("GLOBAL_CACHE_PATH", "/home/aswerdlo/data/cache"))
SCRATCH_CACHE_PATH = Path(os.getenv("SCRATCH_CACHE_PATH", "/scratch/aswerdlo/cache"))

COCO_DATASET_PATH = Path(os.getenv("COCO_DATASET_PATH", f"{PROJECTS_PREFIX}/projects/katefgroup/datasets/coco"))
SCRATCH_COCO_PATH = Path(os.getenv("SCRATCH_COCO_PATH", "/scratch/aswerdlo/coco"))
COCO_CUSTOM_PATH = Path(os.getenv("COCO_CUSTOM_PATH", f"{PROJECTS_PREFIX}/projects/katefgroup/aswerdlo/datasets/coco/annotations"))

DUSTR_REPO_PATH = Path(os.getenv("DUSTR_REPO_PATH", Path.home() / "repos" / "dust3r"))

if SCRATCH_COCO_PATH.exists(): 
    COCO_DATASET_PATH = SCRATCH_COCO_PATH
    COCO_CUSTOM_PATH = SCRATCH_COCO_PATH / "annotations"
    log_info(f"Using scratch coco path: {COCO_DATASET_PATH}")


