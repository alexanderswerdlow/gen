import os
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

COCO_CAPTIONS_FILES = os.getenv("COCO_CAPTIONS_PATH", "/projects/katefgroup/aswerdlo/mscoco/{00000..00059}.tar")
COCO_DATASET_PATH = Path(os.getenv("COCO_DATASET_PATH", "/projects/katefgroup/datasets/coco"))
SCRATCH_COCO_PATH = Path(os.getenv("SCRATCH_COCO_PATH", "/scratch/aswerdlo/coco"))
if SCRATCH_COCO_PATH.exists(): 
    COCO_DATASET_PATH = SCRATCH_COCO_PATH
    print(f"Using scratch coco path: {COCO_DATASET_PATH}")
COCO_TRAIN_ID_PATH = Path(os.getenv("COCO_TRAIN_ID_PATH", "/projects/katefgroup/aswerdlo/datasets/coco"))
MOVI_DATASET_PATH = Path(os.getenv("MOVI_DATASET_PATH", "/projects/katefgroup/datasets/movi"))
MOVI_OVERFIT_DATASET_PATH = Path(os.getenv("MOVI_OVERFIT_DATASET_PATH", "/projects/katefgroup/aswerdlo/movi"))
MOVI_MEDIUM_PATH = Path(os.getenv("MOVI_MEDIUM_PATH", "/projects/katefgroup/aswerdlo/datasets/movi_medium"))
MOVI_MEDIUM_TWO_OBJECTS_PATH = Path(os.getenv("MOVI_MEDIUM_TWO_OBJECTS_PATH", "/projects/katefgroup/aswerdlo/datasets/movi_medium_two_objects"))
MOVI_MEDIUM_SINGLE_OBJECT_PATH = Path(os.getenv("MOVI_MEDIUM_SINGLE_OBJECT_PATH", "/projects/katefgroup/aswerdlo/datasets/single_object_rotating"))
IMAGENET_PATH = Path(os.getenv("IMAGENET_PATH", "/projects/katefgroup/datasets/ImageNet"))
PLACEHOLDER_TOKEN: str = os.getenv("PLACEHOLDER_TOKEN", "masks")
DEFAULT_PROMPT: str = os.getenv("DEFAULT_PROMPT", f"A photo of {PLACEHOLDER_TOKEN}")
CONDA_ENV = Path(os.getenv("CONDA_ENV", "gen"))
GSO_PCD_PATH = Path(os.getenv("GSO_PCD_PATH", "/projects/katefgroup/aswerdlo/gen/gso_pcd.npz"))
OBJAVERSE_DATASET_PATH = Path(os.getenv("OBJAVERSE_DATASET_PATH", "/projects/katefgroup/aswerdlo/datasets/objaverse"))
SCRATCH_OBJAVERSE_PATH = Path(os.getenv("SCRATCH_OBJAVERSE_PATH", "/scratch/aswerdlo/objaverse"))
if SCRATCH_OBJAVERSE_PATH.exists():
    OBJAVERSE_DATASET_PATH = SCRATCH_OBJAVERSE_PATH
    print(f"Using scratch objaverse path: {OBJAVERSE_DATASET_PATH}")

os.environ["EINX_WARN_ON_RETRACE"] = "25"
os.enviorn["IMAGE_UTILS_DISABLE_WARNINGS"] = "1"