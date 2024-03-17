#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import h5py
import os
from pathlib import Path
import pandas as pd

dir_path = Path("hypersim_raw")

extents = {}
for scene_dir in Path(dir_path).iterdir():
    if scene_dir.is_dir():
        for camera_trajectory_dir in Path(scene_dir / "_detail").iterdir():
            if camera_trajectory_dir.is_dir() and camera_trajectory_dir.stem.startswith("cam_"):
                camera_keyframe_positions_hdf5_file         = os.path.join(camera_trajectory_dir, "camera_keyframe_positions.hdf5")
                camera_keyframe_orientations_hdf5_file      = os.path.join(camera_trajectory_dir, "camera_keyframe_orientations.hdf5")

                with h5py.File(camera_keyframe_positions_hdf5_file,     "r") as f: camera_keyframe_positions     = f["dataset"][:]
                with h5py.File(camera_keyframe_orientations_hdf5_file,  "r") as f: camera_keyframe_orientations  = f["dataset"][:]

                assets_to_meters_path = os.path.join(Path(camera_trajectory_dir).parent, 'metadata_scene.csv')
                assets_to_meters_csv = pd.read_csv(assets_to_meters_path)
                assets_to_meters = assets_to_meters_csv['parameter_value'][0]
                camera_keyframe_positions *= assets_to_meters

                extents[(scene_dir.stem, camera_trajectory_dir.stem)] = camera_keyframe_positions.max()


# save dict as json
import pickle
with open(dir_path / 'extents.pkl', 'wb') as handle:
    pickle.dump(extents, handle, protocol=pickle.HIGHEST_PROTOCOL)


