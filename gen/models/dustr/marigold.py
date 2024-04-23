import torch
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
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip
        self._min = None
        self._max = None

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        # Take quantiles as min and max
        self._min, self._max = torch.quantile(
            depth_linear,
            torch.tensor([self.min_quantile, self.max_quantile]).to(depth_linear.device),
        )

        # scale and shift
        depth_norm_linear = (depth_linear - self._min) / (
            self._max - self._min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear


    def scale_back(self, depth_norm):
        depth_linear = (depth_norm / 2 + 0.5).clamp(0, 1)
        depth_linear = depth_linear * (
            self._max - self._min
        ) + self._min
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        return self.scale_back(depth_norm=depth_norm)
