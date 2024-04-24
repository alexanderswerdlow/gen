from einx import rearrange
from gen.models.dustr.marigold import NearFarMetricNormalizer
import torch

def encode_xyz(gt_points, init_valid_mask, vae, min_max_quantile: float = 0.1, kwargs = None):
    if kwargs is None:
        kwargs = dict(valid_mask=init_valid_mask, clip=True, per_axis_quantile=True)

    normalizer = NearFarMetricNormalizer(min_max_quantile=min_max_quantile)
    pre_enc, post_enc_valid_mask = normalizer(gt_points, **kwargs)
    post_enc_valid_mask = (~post_enc_valid_mask & init_valid_mask).to(torch.bool)
    latents = vae.encode(pre_enc).latent_dist.sample() * vae.config.scaling_factor

    return latents, post_enc_valid_mask, normalizer

def decode_xyz(pred_latents, post_enc_valid_mask, vae, normalizer):
    pred_latents = (1 / vae.config.scaling_factor) * pred_latents
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            decoded_points = vae.decode(pred_latents.to(torch.float32), return_dict=False)[0]
            decoded_points, outside_range_post = normalizer.denormalize(decoded_points)

        outside_range_post = outside_range_post.to(device=pred_latents.device)
        final_mask = post_enc_valid_mask & (~outside_range_post)

    return decoded_points, final_mask

if __name__ == "__main__":
    pass