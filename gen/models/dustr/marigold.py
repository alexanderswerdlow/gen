import torch
from einops import rearrange
class DepthNormalizerBase:
    is_relative = None
    far_plane_at_max = None

    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class NearFarMetricNormalizer(DepthNormalizerBase):
    """
    depth in [0, d_max] -> [-1, 1]
    """

    is_relative = True
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True, num_views=1,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip
        self._min = None
        self._max = None
        self.num_views = num_views

    def __call__(self, depth_linear, valid_mask=None, clip: bool = False, per_axis_quantile=False):
        clip = clip if clip is not None else self.clip

        shape = depth_linear.shape

        depth_linear = rearrange(depth_linear, '(num_views b) h w xyz -> b (num_views h w) xyz', num_views=self.num_views)
        if valid_mask is not None:
            valid_mask = rearrange(valid_mask, '(num_views b) h w -> b (num_views h w)', num_views=self.num_views)
        else:
            valid_mask = torch.full_like(depth_linear[..., 0], True, dtype=torch.bool)

        bs = depth_linear.shape[0]

        # Take quantiles as min and max
        all_min, all_max = [], []
        for b in range(bs):
            _min, _max = torch.quantile(
                depth_linear[b][valid_mask[b]].float(),
                torch.tensor([self.min_quantile, self.max_quantile]).to(depth_linear.device),
                dim=(0 if per_axis_quantile else None),
            )
            all_min.append(_min)
            all_max.append(_max)

        self._min = torch.stack(all_min)[:, None]
        self._max = torch.stack(all_max)[:, None]

        if per_axis_quantile is False:
            self._min =  self._min[:, None]
            self._max =  self._max[:, None]


        denom = torch.clamp(self._max - self._min, min=1e-8)

        # scale and shift
        depth_norm_linear = (depth_linear - self._min) / denom * self.norm_range + self.norm_min
        outside_range = ((depth_norm_linear < self.norm_min) | (depth_norm_linear > self.norm_max)).any(dim=-1)

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        depth_norm_linear = rearrange(depth_norm_linear, 'b (num_views h w) xyz -> (num_views b) xyz h w', b=bs, h=shape[1], w=shape[2], num_views=self.num_views)
        outside_range = rearrange(outside_range, 'b (num_views h w) -> (num_views b) h w', b=bs, h=shape[1], w=shape[2], num_views=self.num_views)
        return depth_norm_linear, outside_range

    def scale_back(self, depth_norm, warn: bool = False):
        shape = depth_norm.shape
        depth_norm = rearrange(depth_norm, '(num_views b) xyz h w -> b (num_views h w) xyz', num_views=self.num_views)
        depth_linear = (depth_norm / 2 + 0.5)
        bs = depth_linear.shape[0]

        if warn and (depth_linear.min() <= (0 - 1e-8) or depth_linear.max() >= (1 + 1e-8)):
            print(f"Warning: depth_linear out of range: {depth_linear.min():.3f}, {depth_linear.max():.3f}")

        outside_range = ((depth_linear < 0) | (depth_linear > 1)).any(dim=-1)
        depth_linear = depth_linear * (
            self._max - self._min
        ) + self._min

        depth_linear = rearrange(depth_linear, 'b (num_views h w) xyz -> (num_views b) h w xyz', b=bs, h=shape[2], w=shape[3], num_views=self.num_views)
        outside_range = rearrange(outside_range, 'b (num_views h w) -> (num_views b) h w', b=bs, h=shape[2], w=shape[3], num_views=self.num_views)
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        return self.scale_back(depth_norm=depth_norm)
