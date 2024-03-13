# Copyright (c) Meta Platforms, Inc. and affiliates
from pathlib import Path
from typing import Any, Optional
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg

def get_cfg_defaults(cfg):

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

    # The category names which will be treated as ignore
    # e.g., not counting as background during training
    # or as false positives during evaluation.
    cfg.DATASETS.IGNORE_NAMES = []

    # Should the datasets appear with the same probabilty
    # in batches (e.g., the imbalance from small and large
    # datasets will be accounted for during sampling)
    cfg.DATALOADER.BALANCE_DATASETS = False

    # The thresholds for when to treat a known box
    # as ignore based on too heavy of truncation or 
    # too low of visibility in the image. This affects
    # both training and evaluation ignores.
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, 
    # or if the full 3D projected boxes should be used.
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, 
    # or if the 3D full projected boxes should be used.
    cfg.DATASETS.TRUNC_2D_BOXES = True

    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0
    cfg.INPUT.RANDOM_FLIP = "horizontal"



def get_static_cfg(args: Optional[Any] = None, config_file: Path = Path("gen/datasets/omni3d/configs/Base_Omni3D.yaml")):
    cfg = get_cfg()

    get_cfg_defaults(cfg)
    cfg.merge_from_file(config_file)

    if args is not None:
        cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ('ARKitScenes_val',)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    return cfg