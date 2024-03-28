# 4DoF CaPE
import einops
import torch

def rotate_every_two(x):
    x = einops.rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return einops.rearrange(x, '... d j -> ... (d j)')

def cape(x, p):
    d, l, n = x.shape[-1], p.shape[-2], p.shape[-1]
    assert d % (2 * n) == 0
    m = einops.repeat(p, 'b l n -> b l (n k)', k=d // n)
    return m

def cape_embed_4dof(p1, p2, qq, kk):
    """
    Embed camera position encoding into attention map
    Args:
        p1: query pose  b, l_q, pose_dim
        p2: key pose    b, l_k, pose_dim
        qq: query feature map   b, l_q, feature_dim
        kk: key feature map    b, l_k, feature_dim

    Returns: cape embedded attention map    b, l_q, l_k

    """
    assert p1.shape[-1] == p2.shape[-1]
    assert qq.shape[-1] == kk.shape[-1]
    assert p1.shape[0] == p2.shape[0] == qq.shape[0] == kk.shape[0]
    assert p1.shape[1] == qq.shape[1]
    assert p2.shape[1] == kk.shape[1]

    m1 = cape(qq, p1)
    m2 = cape(kk, p2)

    q = (qq * m1.cos()) + (rotate_every_two(qq) * m1.sin())
    k = (kk * m2.cos()) + (rotate_every_two(kk) * m2.sin())

    return q, k

def cape_embed_6dof(f, P):
    # f is feature vector of shape [..., d]
    # P is 4x4 transformation matrix
    f = einops.rearrange(f, '... (d k) -> ... d k', k=4)
    return einops.rearrange(f @ P.to(f), '... d k -> ... (d k)', k=4)

from scipy.spatial.transform import Rotation
import numpy as np
def get_relative_pose(pose_1, pose_2):
    orig_dtype, orig_device = pose_1.dtype, pose_1.device
    pose_1 = pose_1.cpu().numpy()
    pose_2 = pose_2.cpu().numpy()
    rotation_1 = Rotation.from_matrix(pose_1[..., :3, :3])
    rotation_2 = Rotation.from_matrix(pose_2[..., :3, :3])
    translation_1 = pose_1[..., :3, 3]
    translation_2 = pose_2[..., :3, 3]

    final_tensor = np.zeros((pose_1.shape[0], 4, 4))

    final_tensor[:, :3, :3] = Rotation.from_quat(rotation_2.as_quat() * rotation_1.inv().as_quat()).as_matrix()
    final_tensor[:, :3, 3] = translation_1 - translation_2
    final_tensor[:, 3, 3] = 1

    return torch.from_numpy(final_tensor)