import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from einx import rearrange, roll
from scipy.spatial.transform import Rotation as R

from image_utils import Im
import typer
from torch import Tensor

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

pcds = np.load("data/output.npz")
device = torch.device("cpu")

def get_data(data, camera_idx: int, frame_idx: int):
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


def find_files(base_dir, ext):
    for path in Path(base_dir).rglob(ext):
        if path.is_file():
            yield path


@app.command()
def main(root_dir: Path):
    """
    On Linux, you must start an X11 server to run.
    """

    for i, file in enumerate(find_files(root_dir, "data.npz")):
        create_video(file)
        if i > 4:
            break

def create_video(path):
    print(f"Creating video for {path}...")
    data = np.load(path)
    camera_idx = 0

    frames = 24
    np.set_printoptions(suppress=True)

    width, height = 256, 256
    focal_length, sensor_width = 35, 32
    sensor_height = sensor_width

    f_x = focal_length / sensor_width * width
    f_y = focal_length / sensor_height * height
    p_x = width / 2.0
    p_y = height / 2.0

    K = torch.tensor([[f_x, 0, p_x], [0, f_y, p_y], [0, 0, 1]])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    frame_imgs = []

    for frame_idx in range(frames):
        frame_img = []
        for j in [0, 5, 4]:
            vis.clear_geometries()

            camera_quaternion, camera_position, object_quaternions, positions, valid, asset_id, rgb, scale = get_data(data, camera_idx, frame_idx)

            object_quaternions[~valid] = 1  # Set invalid quaternions to 1 to avoid 0 norm.
            object_quaternions = R.from_quat(object_quaternions)
            camera_quaternion = R.from_quat(camera_quaternion)

            # breakpoint()

            if j == 0:
                # Render as in original sim
                object_quaternions = object_quaternions
            elif j == 1:
                # Camera frame -> World frame -> object frame
                # This is what we want to predict. This should be the delta between the object and the object if it had the same pose as the camera.
                object_quaternions = (object_quaternions * camera_quaternion.inv())
            elif j == 2:
                # Puts the object in canonical pose. Fixed in world space as the camera moves.
                object_quaternions = R.from_quat(np.array([[0, 0, 0, 1]]).repeat(len(object_quaternions), axis=0))
            elif j == 4:
                # Sets the object frame to be the camera frame [e.g., object is not in relative canonical pose]
                object_quaternions = R.from_quat(camera_quaternion.as_quat()[None].repeat(len(object_quaternions), axis=0))
            elif j == 5:
                # We have the delta pose between the camera and object. We put it in the camera frame.
                # (object_quaternions * camera_quaternion.inv()) -> x
                # camera_quaternion -> identity
                object_quaternions = camera_quaternion.inv() * (object_quaternions * camera_quaternion.inv())

            object_quaternions = object_quaternions.as_quat()
            object_quaternions[~valid] = 0

            rot = torch.from_numpy(camera_quaternion.as_matrix()).to(device)
            obj_rot = torch.from_numpy(R.from_quat(object_quaternions[valid]).as_matrix()).to(device)

            T = torch.from_numpy(camera_position).float().to(device)

            verts = []
            rgbs = []
            for i in range(valid.sum()):
                obj_pcd = rearrange("n xyz -> xyz n", pcds[asset_id[i]])
                obj_pcd = obj_rot[i] @ (scale[i] * obj_pcd)
                obj_pcd = rearrange("xyz n -> n xyz", obj_pcd) + positions[i]
                verts.append(torch.Tensor(obj_pcd).to(device))
                rgb_ = verts[-1].new_ones(verts[-1].shape[0], 3) * 0.4
                rgbs.append(rgb_)

            rot = -torch.eye(3).float() @ rot[:3, :3].float() # Convert between Kubrics and Open3D coordinate systems. See: https://github.com/google-research/kubric/pull/307/files
            E = torch.cat([rot.T, (-rot.T.double() @ T.double())[:, None]], dim=1).float()

            # Project to 2D. Not currently used.
            xyz_homogenous = torch.cat([obj_pcd, torch.ones(obj_pcd.shape[0], 1)], dim=1).permute(1, 0).float()
            xy = K @ E @ xyz_homogenous
            xy = xy[:2, :] / xy[2, :]

            rgbs = torch.cat(rgbs, dim=0).cpu().numpy()
            verts = torch.cat(verts, dim=0).cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.colors = o3d.utility.Vector3dVector(rgbs)

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
            text_map = {
                0: "Original",
                1: "GT Pred Rot",
                2: "World Canonical Pose",
                3: "Object frame -> world frame -> camera frame",
                4: "Camera Relative Canonical Pose",
                5: "test",
                6: "test2",
            }
            frame_img.append(Im(np.asarray(image)).write_text(text_map[j], size=0.5).torch)
        frame_imgs.append(Im.concat_horizontal(*frame_img, Im(rgb)).torch)

    Im(torch.stack(frame_imgs, dim=0)).save_video(Path(f"{path.parent.name}.mp4"), fps=12)
    vis.destroy_window()


if __name__ == "__main__":
    app()
