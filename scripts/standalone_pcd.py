import autoroot

import itertools

import numpy as np
import pyviz3d.visualizer as viz
import torch
import torch.nn.functional as F
import typer
from einx import rearrange

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from gen.models.dustr.geometry import depthmap_to_absolute_camera_coordinates
from gen.models.dustr.marigold import NearFarMetricNormalizer
from gen.utils.decoupled_utils import breakpoint_on_error

data = np.load('/projects/katefgroup/share_alex/view1.npz')
device = torch.device('cuda')
dtype = torch.float32
vae = AutoencoderKL.from_pretrained('prs-eth/marigold-v1-0', subfolder="vae").to(device=device, dtype=torch.float32)

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def get_decoded_pcd(gt_points, valid_mask, min_max_quantile: float = 0.001, kwargs = None):
    """
    Args:
        gt_points: [B, 3, H, W]
        valid_mask: [B, H, W]
        min_max_quantile: float
    Returns:
        decoded_points: [B, 3, H, W]
        final_mask: [B, H, W]
    """
    b, _, h, w = gt_points.shape

    if kwargs is None:
        kwargs = dict(valid_mask=valid_mask, clip=True, per_axis_quantile=True)

    normalizer = NearFarMetricNormalizer(min_max_quantile=min_max_quantile)
    _pre_enc, outside_range_pre = normalizer(gt_points, **kwargs)
    latents = vae.encode(_pre_enc).latent_dist.sample() * vae.config.scaling_factor
    latents = (1 / vae.config.scaling_factor) * latents # Obviously this is a no-op
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            decoded_points = vae.decode(latents.to(torch.float32), return_dict=False)[0]
            decoded_points, outside_range_post = normalizer.denormalize(decoded_points)

        outside_range_post = outside_range_post.to(device=device)
        _mask = torch.from_numpy(rearrange('b h w -> (b h w)', valid_mask)).to(device)
        _final_mask = ((~outside_range_pre) & _mask) & (~outside_range_post)

        _final_mask = rearrange('(b h w) -> b h w', _final_mask, b=b, h=h, w=w)

    return decoded_points, _final_mask

@app.command()
def main():
    v = viz.Visualizer()
    output_strings = []

    for i, _args in enumerate(itertools.product(
        [ dict(per_axis_quantile=True), dict(per_axis_quantile=False), ], # 
        [dict(valid_mask=data['valid_mask']),], # dict(valid_mask=None),
        [dict(min_max_quantile=0.1), dict(min_max_quantile=0.05), dict(min_max_quantile=0.02), dict(min_max_quantile=0.01), dict(min_max_quantile=0.001),],
        [dict(clip=True),], # dict(clip=False), 
        [dict(index=None),  dict(index=0), dict(index=1), dict(index=2), ] #  
    )):
        __args = {k: v for d in _args for k, v in d.items()}
        orig_args = __args.copy()

        orig = data['pts3d'].copy()
        _index = __args.pop('index')
        if _index is not None:
            _orig = np.zeros_like(orig)
            _orig[:, :, :, :] = orig[..., _index, None]
        else:
            _orig = orig
        gt_points = rearrange('b h w xyz -> b xyz h w', torch.from_numpy(_orig).to(device=device, dtype=torch.float32))

        _min_max_quantile = __args.pop('min_max_quantile')
        decoded_points, final_mask = get_decoded_pcd(gt_points, data['valid_mask'], _min_max_quantile, __args)

        final_mask = rearrange('b h w -> (b h w)', final_mask)
        _gt = rearrange('b xyz h w -> (b h w) xyz', gt_points)
        _dec = rearrange('b xyz h w -> (b h w) xyz', decoded_points)

        _mask = torch.from_numpy(rearrange('b h w -> (b h w)', data['valid_mask'])).to(device)
        err = F.mse_loss(_gt[_mask], _dec[_mask])
        err_inside_range = F.mse_loss(_gt[final_mask], _dec[final_mask])

        gt_min, gt_max = torch.min(_gt), torch.max(_gt)
        _gt_norm = (_gt - gt_min) / (gt_max - gt_min)
        _pred_norm = (_dec - gt_min) / (gt_max - gt_min)
        err_inside_range = F.mse_loss(_gt_norm, _pred_norm)
        
        if orig_args['valid_mask'] is not None:
            orig_args['valid_mask'] = True
            
        _gt = _gt[final_mask]
        _gt = _gt.float().cpu().numpy() / 100
        _gt_colors = np.ones_like(_gt) * np.array([0, 255, 0])

        _dec = _dec[final_mask]
        _dec = _dec.float().cpu().numpy() / 100
        _dec_colors = np.ones_like(_dec) * np.array([255, 0, 0])
        output_strings.append((err, err_inside_range, f"MSE Error: {err:.5f}, Within dist: {err_inside_range:.5f} {orig_args}", _gt, _dec, _gt_colors, _dec_colors))

    output_strings.sort(key=lambda x: x[1])
    for s in output_strings:
        print(s[2])

    _, _, _, _gt, _dec, _gt_colors, _dec_colors = output_strings[0]

    v.add_points(f"{i} GT PCD",  _gt, _gt_colors, point_size=1)
    v.add_points(f"{i} Decoded PCD", _dec, _dec_colors, point_size=1)
    v.save('output/sensor')
    breakpoint()

        
if __name__ == "__main__":
    with breakpoint_on_error():
        app()






# depthmap = data['depthmap'].squeeze(0)
# camera_intrinsics = np.zeros((3, 3))
# _camera_intrinsics = data['camera_intrinsics'].squeeze(0)
# camera_intrinsics[0, 0] = _camera_intrinsics[0, 1]
# camera_intrinsics[1, 1] = _camera_intrinsics[1, 0]
# camera_intrinsics[:2, 2] = _camera_intrinsics[:2, 2]
# camera_pose = data['camera_pose'].squeeze(0)
# depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose)