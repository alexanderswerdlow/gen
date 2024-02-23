import io

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from PIL import Image

from image_utils import Im

from gen.utils.pytorch3d_transforms import matrix_to_quaternion

matplotlib.use("agg")
matplotlib.rcParams["figure.dpi"] = 128

from scipy.spatial.transform import Rotation as R

def visualize_rotations(R_ref, R_pred):
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define the origin
    origin = np.array([[0, 0, 0]])

    # Axes vectors
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Transform axes through the rotation matrices
    ref_axes = np.dot(R_ref, axes.T).T
    pred_axes = np.dot(R_pred, axes.T).T

    length = 1.0

    # Plot reference axes
    for i in range(3):
        ax.quiver(*origin.T, *ref_axes[i], length=length, color=["r", "g", "b"][i], label="Ref" if i == 0 else "")

    # Plot prediction axes, slightly offset to differentiate
    for i in range(3):
        ax.quiver(*origin.T, *pred_axes[i], length=length, color=["r", "g", "b"][i], label="Pred" if i == 0 else "")

    # Setting the legend and the limits of the plot
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf)
    plt.close("all")

    return im


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return max([x_range, y_range, z_range])


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def visualize_rotations_pcds(ref, pred, pcd, legends=[], markers=[], save=False):
    fig = plt.figure()
    canvas = fig.canvas
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    min_vals = np.min(pcd, axis=0)
    max_vals = np.max(pcd, axis=0)
    scale_factor = (max_vals - min_vals).max()
    pcd = (pcd - min_vals) / scale_factor

    cur_vis_pcd_ = pcd.copy()
    cur_vis_pcd_ -= np.mean(cur_vis_pcd_, axis=0)
    cur_vis_pcd_[:, 1] -= 0.5
    cur_vis_pcd_[:, 2] += 0.5

    cur_vis_rgb = np.zeros_like(pcd)
    cur_vis_rgb[:, 1] = 1.0

    ax.scatter(cur_vis_pcd_[:, 0], cur_vis_pcd_[:, 1], cur_vis_pcd_[:, 2], c=cur_vis_rgb[:], s=1, label="GT Canonical Pose")

    cur_vis_pcd_ = pcd.copy()
    cur_vis_pcd_ = ((np.linalg.inv(ref) @ pred) @ cur_vis_pcd_.transpose(1, 0)).transpose(1, 0)
    cur_vis_pcd_ -= np.mean(cur_vis_pcd_, axis=0)
    cur_vis_pcd_[:, 1] += 0.5
    cur_vis_pcd_[:, 2] += 0.5

    cur_vis_rgb = np.zeros_like(pcd)
    cur_vis_rgb[:, 0] = 1.0

    ax.scatter(cur_vis_pcd_[:, 0], cur_vis_pcd_[:, 1], cur_vis_pcd_[:, 2], c=cur_vis_rgb[:], s=1, label="Pred Pose")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plot_bounds = set_axes_equal(ax)

    arrow_length = plot_bounds / 2

    ax.arrow3D(0, 0, 0, arrow_length, 0, 0, mutation_scale=20, ec="green", fc="red")
    ax.arrow3D(0, 0, 0, 0, arrow_length, 0, mutation_scale=20, ec="green", fc="green")
    ax.arrow3D(0, 0, 0, 0, 0, arrow_length, mutation_scale=20, ec="green", fc="blue")

    ax.text(0.0, 0.0, -0.1, r"$0$")
    ax.text(arrow_length + 0.1, 0, 0, r"$x$")
    ax.text(0, arrow_length + 0.1, 0, r"$y$")
    ax.text(0, 0, arrow_length + 0.1, r"$z$")

    fig.tight_layout()
    images = []
    for elev, azim in zip([0, 40, 15], [0, 30, 75]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110]  # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate(images[:5], axis=1)
    if save:
        Image.fromarray(images, mode="RGB").save("diff_traj.png")

    plt.close()

    return Image.fromarray(images)


def normalize_vector(v, return_mag=False):
    device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out  # batch*3


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def get_ortho6d_from_rotation_matrix(matrix):
    # Noe the orhto6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
    return ortho6d


def get_quat_from_discretized_zyx(zyx: np.ndarray | torch.Tensor, num_bins: int):
    """
    Takes [N, 3] discretized zyx and returns [N, 4] quat, xyzw
    """
    if isinstance(zyx, torch.Tensor):
        zyx = zyx.detach().float().cpu().numpy()

    zyx = zyx / (num_bins - 1)
    zyx %= 1 # Wraparound
    zyx[:, [0, 2]] = (zyx[:, [0, 2]] * (2 * np.pi)) - (np.pi)
    zyx[:, 1] = (zyx[:, 1] * (np.pi)) - (np.pi / 2)
    quat = R.from_euler(seq="zyx", angles=zyx, degrees=False).as_quat()
    return quat

def get_discretized_zyx_from_quat(quat: torch.Tensor, num_bins: int, return_unquantized=False):
    """
    Takes [N, 4] quat, xyzw and returns [N, 3] discretized zyx

    From SciPy:
    First angle belongs to [-180, 180] degrees (both inclusive)
    Second angle belongs to: [-90, 90] degrees if all axes are different (like xyz)
    Third angle belongs to [-180, 180] degrees (both inclusive)
    """
    device = quat.device
    max_range = num_bins - 1
    zyx = R.from_quat(quat.float().cpu().numpy()).as_euler(seq="zyx", degrees=False)
    zyx[:, [0, 2]] = (zyx[:, [0, 2]] + np.pi) / (2 * np.pi) # Normalize from 0 to 1
    zyx[:, 1] = (zyx[:, 1] + (np.pi / 2)) / (np.pi)
    discretized_zyx = np.round(zyx * max_range).astype(int)
    discretized_zyx = torch.clamp(torch.from_numpy(discretized_zyx).to(device), 0, max_range)

    if return_unquantized:
        return discretized_zyx, torch.from_numpy(zyx * max_range).to(dtype=torch.float, device=device)
    
    return discretized_zyx

def quat_l1_loss(rot1, rot2):
    mat1 = compute_rotation_matrix_from_ortho6d(rot1)
    quat1 = matrix_to_quaternion(mat1)

    mat2 = compute_rotation_matrix_from_ortho6d(rot2)
    quat2 = matrix_to_quaternion(mat2)

    quat_l1 = (quat1 - quat2).abs().sum(-1)
    quat_l1_ = (quat1 + quat2).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_
    return quat_l1