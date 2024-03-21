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
            set_default_inference_func(model, cfg)
            return model
        case _:
            raise ValueError(f"Unknown model type: {cfg.model.model_type}")
        
def set_inference_func(model, inference_func):
    model.run_inference = types.MethodType(inference_func, model)

def set_default_inference_func(model, cfg: BaseConfig):
    inference_func = hydra.utils.instantiate(cfg.inference.inference_func)  # Instantiate the function (e.g., partials)
    model.run_inference = types.MethodType(inference_func, model)

def _init_weights(m):
    initializer_range = 0.02
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=initializer_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=initializer_range)

def positionalencoding2d(d_model, height, width, device, dtype, scale: Optional[float] = 200, normalize: bool = True):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.empty(d_model, height, width, device=device, dtype=dtype)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=device, dtype=dtype) * -(math.log(10000.0) / d_model))

    pos_w = torch.arange(0., width, device=device, dtype=dtype).unsqueeze(1)
    pos_h = torch.arange(0., height, device=device, dtype=dtype).unsqueeze(1)

    if normalize:
        pos_w = (pos_w / width) * 2 - 1
        pos_h = (pos_h / height) * 2 - 1

    if scale is not None:
        div_term *= 2 * torch.pi * scale

    # alternates between sin and cos up to halfway for w and the second half for h (half of the dimension is for each)
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

class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels
        """
        super(FourierEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

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
    # x = torch.rand(100)
    # pos_emb = SinusoidalPosEmb(512)
    # y = pos_emb(x)
    # print(y.shape)

    import torch.utils.benchmark as benchmark
    globals_dict = {
        'd_model': 1024,
        'height': 256,
        'width': 256,
        'device': 'cuda:0',
        'dtype': torch.bfloat16
    }
    for scale in [15, 20, 25, 30, 35, 40, 100, 150, 200, 1000]:
        output = positionalencoding2d(**globals_dict, normalize=True, scale=scale) # scale=0.0001, 
        emb_sin = output[output.shape[0] // 2:, :, 0][0::2]
        emb_cos = output[output.shape[0] // 2:, :, 0][1::2]
        embs = [emb_sin, emb_cos]
        # Plotting the positional encodings
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 15))
        for j in range(2):
            plt.subplot(1, 2, j + 1)
            plt.imshow(embs[j].float().cpu().numpy(), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'Scale: {scale}')
        plt.tight_layout()
        plt.savefig(f'test_{scale:.4f}.png')
        plt.close()
    breakpoint()

    t1 = benchmark.Timer(
        stmt='positionalencoding2d_slow(d_model, height, width, device, dtype)',
        setup='from __main__ import positionalencoding2d_slow',
        globals=globals_dict)
    
    t0 = benchmark.Timer(
        stmt='positionalencoding2d(d_model, height, width, device, dtype)',
        setup='from __main__ import positionalencoding2d',
        globals=globals_dict)

    t2 = benchmark.Timer(
        stmt='positionalencoding2d_(d_model, height, width, device, dtype)',
        setup='from __main__ import positionalencoding2d_',
        globals=globals_dict)

    print(t0.timeit(100))
    print(t1.timeit(100))
    print(t2.timeit(100))

    assert positionalencoding2d(**globals_dict).allclose(positionalencoding2d_(**globals_dict))
    assert positionalencoding2d(**globals_dict).allclose(positionalencoding2d_slow(**globals_dict))

    breakpoint()