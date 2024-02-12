import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from einx import rearrange, roll
from scipy.spatial.transform import Rotation as R

from image_utils import Im

path = "data.npz"
pcds = np.load("output.npz")

data = np.load(path)
camera_idx = 0

frames = 24
np.set_printoptions(suppress=True)


def get_data(frame_idx: int):
    rgb = data["rgb"][camera_idx, frame_idx]
    instance = data["segment"][camera_idx, frame_idx]
    quaternions = data["quaternions"][camera_idx, frame_idx]  # (23, 4)
    quaternions = roll("objects [wxyz]", quaternions, shift=(-1,))
    positions = data["positions"][camera_idx, frame_idx]  # (23, 3)
    valid = data["valid"][camera_idx, :].squeeze(0)  # (23, )
    categories = data["categories"][camera_idx, :].squeeze(0)  # (23, )
    asset_id = data["asset_ids"][camera_idx, :].squeeze(0)  # (23, )
    scale = data["scales"][camera_idx, :].squeeze(0)  # (23, )

    camera_quaternion = data["camera_quaternions"][camera_idx, frame_idx]  # (4, )
    camera_quaternion = roll("[wxyz]", camera_quaternion, shift=(-1,))
    camera_position = data["camera_positions"][camera_idx, frame_idx]  # (3, )

    return camera_quaternion, camera_position, quaternions, positions, valid, asset_id, rgb, scale


width, height = 256, 256
focal_length, sensor_width = 35, 32
sensor_height = sensor_width

f_x = focal_length / sensor_width * width
f_y = focal_length / sensor_height * height
p_x = width / 2.0
p_y = height / 2.0

K = torch.tensor([[f_x, 0, p_x], [0, f_y, p_y], [0, 0, 1]])

device = torch.device("cpu")

vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height)
frame_imgs = []

for frame_idx in range(frames):
    frame_img = []
    for j in range(5):
        vis.clear_geometries()

        camera_quaternion, camera_position, object_quaternions, positions, valid, asset_id, rgb, scale = get_data(frame_idx)

        object_quaternions[~valid] = 1  # Set invalid quaternions to 1 to avoid 0 norm.
        if j == 0:
            # Render as in original sim
            pass
        elif j == 1:
            # Puts the object in canonical pose. Fixed in world space as the camera moves.
            object_quaternions = np.array([[0, 0, 0, 1]]).repeat(len(object_quaternions), axis=0)
        elif j == 2:
            # Object frame -> world frame -> camera frame.
            # The pose of the object is now in the camera frame. Keeps it locked w.r.t the camera.
            object_quaternions = (R.from_quat(camera_quaternion) * R.from_quat(object_quaternions).inv()).as_quat()
        elif j == 3:
            # Camera frame -> World frame -> object frame
            # This is what we want to predict. This should be the delta between the object and the object if it had the same pose as the camera.
            object_quaternions = (R.from_quat(object_quaternions) * R.from_quat(camera_quaternion).inv()).as_quat()
        elif j == 4:
            # Camera frame -> World frame -> object frame
            # This is what we want to predict. This should be the delta between the object and the object if it had the same pose as the camera.
            object_quaternions = (
                (R.from_quat(object_quaternions) * R.from_quat(camera_quaternion).inv()).inv() * R.from_quat(object_quaternions)
            ).as_quat()
        object_quaternions[~valid] = 0

        rot = torch.from_numpy(R.from_quat(camera_quaternion).as_matrix()).to(device)
        obj_rot = torch.from_numpy(R.from_quat(object_quaternions[valid]).as_matrix()).to(device)

        T = torch.from_numpy(camera_position).float().to(device)
        image_size = torch.tensor([(width, height)])

        verts = []
        rgbs = []
        for i in range(valid.sum()):
            obj_pcd = obj_rot[i] @ (scale[i] * rearrange("n xyz -> xyz n", pcds[asset_id[i]]))
            obj_pcd = rearrange("xyz n -> n xyz", obj_pcd) + positions[i]
            verts.append(torch.Tensor(obj_pcd).to(device))
            rgbs.append(verts[-1].new_ones(verts[-1].shape[0], 4))

        rot = -torch.eye(3).float() @ rot[:3, :3].float()
        E = torch.cat([rot.T, (-rot.T.double() @ T.double())[:, None]], dim=1).float()

        XYZ = torch.cat([obj_pcd, torch.ones(obj_pcd.shape[0], 1)], dim=1).permute(1, 0).float()

        xy = K @ E @ XYZ
        xy = xy[:2, :] / xy[2, :]

        rgbs = torch.cat(rgbs, dim=0).cpu().numpy()
        verts = torch.cat(verts, dim=0).cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        ctr = vis.get_view_control()

        vis.add_geometry(pcd)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

        # doesn't work until after add_geometry...  see Visualizer.cpp line 408 - ResetViewPoint() in Visualizer::AddGeometry
        params = ctr.convert_to_pinhole_camera_parameters()

        camera_position = params.extrinsic[:3, 3]

        vis.update_geometry(pcd)
        vis.update_renderer()

        extrinsic = np.eye(4)
        extrinsic[:3, :] = E.numpy()

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic
        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params, True)

        vis.update_renderer()
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer()
        frame_img.append(Im.concat_horizontal(Im(np.asarray(image)), Im(rgb)).write_text(str(j)).torch)
    frame_imgs.append(Im.concat_vertical(*frame_img).torch)

Im(torch.stack(frame_imgs, dim=0)).save_video(Path("output.mp4"), fps=12)

vis.destroy_window()
