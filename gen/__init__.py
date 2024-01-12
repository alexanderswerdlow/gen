import os
from pathlib import Path

COCO_CAPTIONS_FILES = os.getenv('COCO_CAPTIONS_PATH', '/projects/katefgroup/aswerdlo/mscoco/{00000..00059}.tar')
MOVI_DATASET_PATH = Path(os.getenv('MOVI_DATASET_PATH', '/projects/katefgroup/datasets/movi'))
MOVI_OVERFIT_DATASET_PATH = Path(os.getenv('MOVI_OVERFIT_DATASET_PATH', '/projects/katefgroup/aswerdlo/movi'))

DEFAULT_PROMPT: str = "A photo of"