import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale: float = 0.0001):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * (2 * torch.pi * self.scale)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
if __name__ == "__main__":
    x = torch.rand(100)
    pos_emb = SinusoidalPosEmb(512)
    y = pos_emb(x)
    print(y.shape)