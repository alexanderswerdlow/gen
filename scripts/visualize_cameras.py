
import autoroot


import torch
import collections
import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from ipdb import set_trace as st
import torch
from image_utils import Im
import numpy as np
from einops import rearrange
import hickle as hkl
import hdf5plugin

import numpy as np
import pyviz3d.visualizer as viz
from pyquaternion import Quaternion
from image_utils import library_ops
from scipy.spatial.transform import Rotation as R
import http.server
import socketserver
import os
import socket
from pathlib import Path

def start_web_server(directory):
    os.chdir(directory)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port provided by the OS
        port = s.getsockname()[1]  # Get the chosen port

    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    print("")
    print(
        "************************************************************************"
    )
    print("2) Open in browser:")
    print("    http://0.0.0.0:" + str(port))
    print(
        "************************************************************************"
    )

    httpd.serve_forever()

def add_cam(v, name, quaternion, position):
    v.add_points(f'{name}_Position', position.reshape(1, 3), np.array([[255, 0, 0]]), point_size=100)
    rotation_matrix = Quaternion(quaternion).rotation_matrix
    scale = 1.0

    xyz_name = ["X", "Y", "Z"]
    for i, vec in enumerate(rotation_matrix.T):
        start_point = position
        end_point = position + vec * scale
        color = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])][i]  # Red, Green, Blue
        v.add_arrow(name=f"{name}_{xyz_name[i]}_axis", start=start_point, end=end_point, color=color)

def viz_pcd():
    v = viz.Visualizer()
    # from ipdb import set_trace as st; st()
    path = Path("/home/aswerdlow/research/tmp/custom_kubric/output/singleview_new_format/processed/movi_e/train/000002/data.npz")
    data = np.load(path)

    print(list(data.keys()))
    num_frames = data["quaternions"].shape[1]
    
    frame_idx = 0
    camera_idx = 0

    camera_position = data['camera_positions'][camera_idx, frame_idx]
    camera_quaternion = data['camera_quaternions'][camera_idx, frame_idx]

    add_cam(v, 'Super Long Name Here camera', camera_quaternion, camera_position)

    valid = data['valid'][camera_idx].squeeze(0)
    num_objects = valid.sum()

    for frame_idx in range(num_frames):
        for object_idx in range(num_objects):
            quaternions = data["quaternions"][camera_idx, frame_idx, object_idx] # (23, 4)
            positions = data["positions"][camera_idx, frame_idx, object_idx] # (23, 3)
            add_cam(v, f'f_{frame_idx}_o_{object_idx}', quaternions, positions)

    for frame_idx in range(num_frames):
        for object_idx in range(num_objects):
            quaternions = data["quaternions"][camera_idx, frame_idx, object_idx] # (23, 4)
            positions = data["positions"][camera_idx, frame_idx, object_idx] # (23, 3)
            quaternions = (R.from_quat(quaternions) * R.from_quat(camera_quaternion).inv()).as_quat()
            add_cam(v, f'mf_{frame_idx}_o_{object_idx}', quaternions, positions)
    
    Im(data['rgb'][camera_idx]).save_video(Path(f'output_{path.parent.name}.mp4'), fps=4)

    v.save('example', verbose=False)

    start_web_server('example')

if __name__ == '__main__':
    viz_pcd()