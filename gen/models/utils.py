from typing import List, Optional
import math
import hydra
import torch
import torch.nn as nn

from gen.configs.base import BaseConfig
from gen.configs.models import ModelType
import types

def get_model_from_cfg(cfg: BaseConfig):
    from gen.models.cross_attn.base_model import BaseMapper

    match cfg.model.model_type:
        case ModelType.BASE_MAPPER:
            model = BaseMapper(cfg)

            inference_func = hydra.utils.instantiate(cfg.inference.inference_func)  # Instantiate the function (e.g., partials)
            model.run_inference = types.MethodType(inference_func, model)
            return model
        case _:
            raise ValueError(f"Unknown model type: {cfg.model.model_type}")


def _init_weights(m):
    initializer_range = 0.02
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=initializer_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=initializer_range)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def sinusoidal_pos_emb(x, dim, scale: Optional[float] = None):
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    if scale is not None:
        scale *= 2 * torch.pi * scale
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale: Optional[float] = None):  #  0.0001
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        return sinusoidal_pos_emb(x, self.dim, self.scale)

class FourierPositionalEncodingNDims(nn.Module):
    """
    Implementation of n-dim Fourier mapping from
    https://github.com/tancik/fourier-feature-networks

    This is the class we use.
    """

    def __init__(self, sigmas: List[float], dim: int = 128, normalize=False, seed=0):
        super().__init__()
        # some config
        self.sigmas = sigmas
        self.dim = dim
        nfeats = len(sigmas)

        # generate the random features
        self.w = torch.randn((dim // 2, nfeats))
        for i in range(nfeats):
            self.w[:, i] *= sigmas[i]

        self.w = nn.Parameter(self.w)
        self.normalize = normalize

    def forward(self, x: torch.Tensor):
        """
        Maps the given time and layer input into a 2048-dimensional vector.
        The neti pos encoding does normalization, but the OG
        """
        # check its in range [-1,1]
        # assert torch.all(x >= -1) and torch.all(x <= 1), "inputs should be in [-1,1]"

        if x.ndim == 1:
            x = x.unsqueeze(1)  # (bs,1)

        x = x.T  # (1,bs)
        x = x.cuda()
        v = torch.cat([torch.sin(self.w.detach() @ x), torch.cos(self.w.detach() @ x)])  # (dim, bs)

        if self.normalize:
            v = v / v.norm(dim=0)

        v = v.T  # (bs,dim)
        return v


def find_true_indices_batched(original, dh, dw):
    # Get dimensions
    masks, h, w = original.shape

    # Reshape and unfold to align with the downscaled dimensions
    reshaped = original.unfold(1, h // dh, h // dh).unfold(2, w // dw, w // dw)
    reshaped = reshaped.reshape(masks, dh, dw, -1)

    # Check for any True values in the corresponding blocks
    result = reshaped.any(dim=3)

    # Get indices where result is True
    # indices = [torch.nonzero(r, as_tuple=False) for r in result]

    return result


if __name__ == "__main__":
    x = torch.rand(100)
    pos_emb = SinusoidalPosEmb(512)
    y = pos_emb(x)
    print(y.shape)
