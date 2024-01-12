from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale: Optional[float] = None):  #  0.0001
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        import math

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        if self.scale is not None:
            self.scale *= 2 * torch.pi * self.scale
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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
        torch.manual_seed(seed)

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


if __name__ == "__main__":
    x = torch.rand(100)
    pos_emb = SinusoidalPosEmb(512)
    y = pos_emb(x)
    print(y.shape)
