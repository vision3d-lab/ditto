import torch as th
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.ditto.utils import default


def activation(name):
    return {
        "relu": lambda: nn.ReLU(inplace=True),
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(inplace=True),
    }[name.lower()]()


class LayerNormBCN(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # input: b c ...
        x = rearrange(x, "b c ... -> b ... c")
        x = super().forward(x)
        x = rearrange(x, "b ... c -> b c ...")
        return x


def normalization(name, dim):
    return {
        "bn1": lambda: nn.BatchNorm1d(dim),
        "bn2": lambda: nn.BatchNorm2d(dim),
        "bn3": lambda: nn.BatchNorm3d(dim),
        "bn": lambda: nn.BatchNorm2d(dim),
        "gn": lambda: nn.GroupNorm(32, dim),
        "ln": lambda: LayerNormBCN(dim),
        None: lambda: nn.Identity(),
        "": lambda: nn.Identity(),
    }[name]()


class ResidualMLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_hidden=None, act_fn="relu"):
        dim_out = default(dim_out, dim_in)
        dim_hidden = default(dim_hidden, min(dim_in, dim_out))
        super().__init__()
        self.fc_0 = nn.Linear(dim_in, dim_hidden)
        self.fc_1 = nn.Linear(dim_hidden, dim_out)
        self.actvn = activation(act_fn)

        self.shortcut = nn.Identity()
        if dim_in != dim_out:
            self.shortcut = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        h = self.fc_0(self.actvn(x))
        h = self.fc_1(self.actvn(h))
        out = h + self.shortcut(x)
        return out


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding
