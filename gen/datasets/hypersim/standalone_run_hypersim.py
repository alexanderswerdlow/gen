from pathlib import Path
from nicr_scene_analysis_datasets import Hypersim

sample_keys = (
    'identifier',    # helps to know afterwards which sample was loaded
    'rgb', 'depth',    # camera data
    'rgb_intrinsics', 'depth_intrinsics', 'extrinsics',    # camera parameters
    'semantic', 'instance', 'orientations', '3d_boxes', 'scene', 'normal'    # annotations
)

# list available sample keys
print(Hypersim.sample_keys) # get_available_sample_keys(split='train')

dataset_path = Path("/home/aswerdlow/datasets/hypersim_processed")

dataset_train = Hypersim(
    dataset_path=dataset_path,
    split='train',
    sample_keys=sample_keys
)

# finally, you can iterate over the dataset
for sample in dataset_train:
    breakpoint()
    print(sample)

# note: for usage along with pytorch, simply change the import above to
from nicr_scene_analysis_datasets.pytorch import Hypersim