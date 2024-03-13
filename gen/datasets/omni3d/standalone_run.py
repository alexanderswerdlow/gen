
import autoroot

import os
import sys
import traceback
from pathlib import Path
from tqdm import tqdm

import typer
from detectron2.data import DatasetCatalog, MetadataCatalog

from gen import OMNI3D_DATASET_OMNI3D_PATH
from gen.datasets.omni3d.build import build_detection_train_loader
from gen.datasets.omni3d.config import get_static_cfg
from gen.datasets.omni3d.dataset_mapper import DatasetMapper3D
from gen.datasets.omni3d.datasets import Omni3D, get_filter_settings_from_cfg, register_and_store_model_metadata, simple_register
from gen.datasets.omni3d.util import file_parts, load_json
from gen.utils.decoupled_utils import set_global_breakpoint
from image_utils import Im, library_ops

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def get_dataloader(cfg, eval_only: bool = False):
    filter_settings = get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True, datasets_root_path=OMNI3D_DATASET_OMNI3D_PATH)
    
    dataset_names_test = cfg.DATASETS.TEST

    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(dataset_name, filter_settings, filter_empty=False, datasets_root_path=OMNI3D_DATASET_OMNI3D_PATH)

    
    metadata = load_json(OMNI3D_DATASET_OMNI3D_PATH / 'category_meta.json')
    thing_classes = metadata['thing_classes']
    id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
    MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map
            
    if eval_only:
        metadata = load_json(OMNI3D_DATASET_OMNI3D_PATH / 'category_meta.json')
        thing_classes = metadata['thing_classes']
        id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
        MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map
    else:
        # setup and join the data.
        dataset_paths = [OMNI3D_DATASET_OMNI3D_PATH / (name + '.json') for name in cfg.DATASETS.TRAIN]
        datasets = Omni3D(dataset_paths, filter_settings=filter_settings)

        # determine the meta data given the datasets used. 
        register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        
        '''
        It may be useful to keep track of which categories are annotated/known
        for each dataset in use, in case a method wants to use this information.
        '''

        infos = datasets.dataset['info']

        if type(infos) == dict:
            infos = [datasets.dataset['info']]

        dataset_id_to_unknown_cats = {}
        possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
        
        dataset_id_to_src = {}

        for info in infos:
            dataset_id = info['id']
            known_category_training_ids = set()

            if not dataset_id in dataset_id_to_src:
                dataset_id_to_src[dataset_id] = info['source']

            for id in info['known_category_ids']:
                if id in dataset_id_to_contiguous_id:
                    known_category_training_ids.add(dataset_id_to_contiguous_id[id])
            
            # determine and store the unknown categories.
            unknown_categories = possible_categories - known_category_training_ids
            dataset_id_to_unknown_cats[dataset_id] = unknown_categories

            # log the per-dataset categories
            print('Available categories for {}'.format(info['name']))
            print([thing_classes[i] for i in (possible_categories & known_category_training_ids)])
    
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src)
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats
    return data_loader

@app.command()
def main(
    path: Path = Path("output")
):
    cfg = get_static_cfg()
    data_loader = get_dataloader(cfg)

    for batch in tqdm(data_loader):
        breakpoint()
      
if __name__ == "__main__":
    set_global_breakpoint()

    try:
        app()
    except Exception as e:
        print("Exception...", e)
        traceback.print_exc()
        breakpoint(traceback=e.__traceback__)
        sys.exit(1)
        raise
    finally:
        pass



