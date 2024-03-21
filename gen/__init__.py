
import os
from pathlib import Path
from gen.utils.logging_utils import log_info

os.environ["EINX_WARN_ON_RETRACE"] = "25"
os.environ["IMAGE_UTILS_DISABLE_WARNINGS"] = "1"

REPO_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

COCO_CAPTIONS_FILES = os.getenv("COCO_CAPTIONS_PATH", "/projects/katefgroup/aswerdlo/mscoco/{00000..00059}.tar")
COCO_DATASET_PATH = Path(os.getenv("COCO_DATASET_PATH", "/projects/katefgroup/datasets/coco"))
SCRATCH_COCO_PATH = Path(os.getenv("SCRATCH_COCO_PATH", "/scratch/aswerdlo/coco"))
COCO_CUSTOM_PATH = Path(os.getenv("COCO_CUSTOM_PATH", "/projects/katefgroup/aswerdlo/datasets/coco/annotations"))

if SCRATCH_COCO_PATH.exists(): 
    COCO_DATASET_PATH = SCRATCH_COCO_PATH
    COCO_CUSTOM_PATH = SCRATCH_COCO_PATH / "annotations"
    log_info(f"Using scratch coco path: {COCO_DATASET_PATH}")

SCRATCH_CACHE_PATH = Path(os.getenv("SCRATCH_CACHE_PATH", "/scratch/aswerdlo/cache"))
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
    log_info(f"Using scratch objaverse path: {OBJAVERSE_DATASET_PATH}")

OMNI3D_PATH = Path(os.getenv("OMNI3D_PATH", "/projects/katefgroup/aswerdlo/datasets/omni3d"))
OMNI3D_DATASETS_PATH = os.getenv("OMNI3D_DATASETS_PATH") or OMNI3D_PATH / "datasets"
OMNI3D_DATASET_OMNI3D_PATH = os.getenv("OMNI3D_DATASET_OMNI3D_PATH") or OMNI3D_DATASETS_PATH / "Omni3D"

HYPERSIM_DATASET_PATH = Path(os.getenv("HYPERSIM_DATASET_PATH", "/projects/katefgroup/aswerdlo/datasets/hypersim"))
SCRATCH_HYPERSIM_PATH = Path(os.getenv("SCRATCH_HYPERSIM_PATH", "/scratch/aswerdlo/hypersim"))
if SCRATCH_HYPERSIM_PATH.exists():
    HYPERSIM_DATASET_PATH = SCRATCH_HYPERSIM_PATH
    log_info(f"Using scratch hypersim path: {HYPERSIM_DATASET_PATH}")

SCANNETPP_DATASET_PATH =  Path(os.getenv("SCANNETPP_DATASET_PATH", "/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/data"))
SCANNETPP_CUSTOM_DATA_PATH =  Path(os.getenv("SCANNETPP_CUSTOM_DATA_PATH", "/projects/katefgroup/language_grounding/SCANNET_PLUS_PLUS/custom"))