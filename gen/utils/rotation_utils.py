import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
from image_utils import Im
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib

matplotlib.use("agg")
matplotlib.rcParams["figure.dpi"] = 128
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL.Image as Image
import cv2


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


def visualize_rotations_pcds(ref, pred, cur_vis_pcd, legends=[], markers=[], save=False):
    # np.linalg.inv(ref) @ pred

    fig = plt.figure()
    canvas = fig.canvas
    # ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    cur_vis_pcd_ = cur_vis_pcd.copy()
    cur_vis_rgb = np.zeros_like(cur_vis_pcd)
    cur_vis_rgb[:, 1] = 1.0
    ax.scatter(cur_vis_pcd_[:, 0], cur_vis_pcd_[:, 1], cur_vis_pcd_[:, 2], c=cur_vis_rgb[:], s=1)

    cur_vis_rgb = np.zeros_like(cur_vis_pcd)
    cur_vis_rgb[:, 0] = 1.0
    cur_vis_pcd_ = cur_vis_pcd.copy()
    cur_vis_pcd_ = ((np.linalg.inv(ref) @ pred) @ cur_vis_pcd_.transpose(1, 0)).transpose(1, 0)
    x_offset = (cur_vis_pcd_[:, 0].max() - cur_vis_pcd_[:, 0].min()) * 1.5
    cur_vis_pcd_[:, 0] += x_offset
    ax.scatter(cur_vis_pcd_[:, 0], cur_vis_pcd_[:, 1], cur_vis_pcd_[:, 2], c=cur_vis_rgb[:], s=1)

    ax.legend()
    set_axes_equal(ax)
    fig.tight_layout()
    images = []
    for elev, azim in zip([10, 15, 20, 25, 30, 25, 20, 15, 45, 90], [0, 45, 90, 135, 180, 225, 270, 315, 360, 360]):
        ax.view_init(elev=elev, azim=azim, roll=0)
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        image = image[60:, 110:-110]  # HACK <>
        image = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
        images.append(image)
    images = np.concatenate([np.concatenate(images[:5], axis=1), np.concatenate(images[5:10], axis=1)], axis=0)
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
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
    return ortho6d
