import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PAdaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_kernels=1):
        """Plane-Adaptive Convolution"""
        if n_kernels > 1:
            super().__init__(in_channels, n_kernels * out_channels, kernel_size, stride, padding)

            self.plane_emb = nn.Embedding(3, in_channels)
            self.weight_enc = nn.Linear(in_channels, n_kernels)
        else:
            super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.n_kernels = n_kernels

    def forward(self, x):
        """
        - x: (3b c_in h w) or 3 x (b c_in h w)
        - return: (3b c_out h w) or 3 x (b c_out h w)
        """
        if self.n_kernels > 1:
            is_tuple = isinstance(x, (list, tuple))
            if is_tuple:
                assert len(x) == 3, len(x)
                x = th.cat(x)  # 3b c_in h w

            assert x.size(0) % 3 == 0, x.shape

            b3, b = x.size(0), x.size(0) // 3

            weight1 = x.mean(dim=(2, 3)) + repeat(self.plane_emb.weight, "x c -> (x b) c", b=b)  # 3b c_in
            # weight1 = self.weight_enc(weight1.softmax(dim=-1))  # 3b n_k
            weight1 = self.weight_enc(weight1).softmax(dim=-1)  # 3b n_k

            bias = rearrange(self.bias, "(nk co) -> 1 nk co", nk=self.n_kernels) * weight1[..., None]  # 3b n_k c_out
            bias = bias.sum(dim=1).flatten().contiguous()  # (3b x c_out)

            weight1 = rearrange(weight1, "b nk -> b nk 1 1 1 1")  # 3b n_k 1 1 1 1
            weight2 = rearrange(self.weight, "(nk co) ci k1 k2 -> 1 nk co ci k1 k2", nk=self.n_kernels)  # 1 n_k c_out c_in k k

            weight = th.sum(weight1 * weight2, dim=1)  # 3b c_out c_in k k
            weight = rearrange(weight, "b co ci k1 k2 -> (b co) ci k1 k2").contiguous()

            x = rearrange(x, "b c h w -> 1 (b c) h w").contiguous()  # 1 (3b x c_in) h w

            x = F.conv2d(x, weight, bias, self.stride, self.padding, groups=b3)  # 1 (3b x c_out) h w
            x = rearrange(x, "1 (b c) h w -> b c h w", b=b3).contiguous()  # 3b c_out h w

            if is_tuple:
                return x.chunk(3)
            return x

        elif isinstance(x, (list, tuple)):
            outs = []
            for x_sub in x:
                outs.append(super().forward(x_sub))
            return outs

        else:
            return super().forward(x)


class PAdaConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_kernels=1):
        """Plane-Adaptive Transposed Convolution"""
        if n_kernels > 1:
            super().__init__(in_channels, n_kernels * out_channels, kernel_size, stride, padding)

            self.plane_emb = nn.Embedding(3, in_channels)
            self.weight_enc = nn.Linear(in_channels, n_kernels)
        else:
            super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.n_kernels = n_kernels

    def forward(self, x):
        """
        - x: (3b c_in h w) or 3 x (b c_in h w)
        - return: (3b c_out h w) or 3 x (b c_out h w)
        """
        is_tuple = isinstance(x, (list, tuple))
        if is_tuple:
            assert len(x) == 3, len(x)
            x = th.cat(x)  # 3b c_in h w
        assert x.size(0) % 3 == 0, x.shape

        if self.n_kernels > 1:
            b3, b = x.size(0), x.size(0) // 3

            weight1 = x.mean(dim=(2, 3)) + repeat(self.plane_emb.weight, "x c -> (x b) c", b=b)  # 3b c_in
            # weight1 = self.weight_enc(weight1.softmax(dim=-1))  # 3b n_k
            weight1 = self.weight_enc(weight1).softmax(dim=-1)  # 3b n_k

            bias = rearrange(self.bias, "(nk co) -> 1 nk co", nk=self.n_kernels) * weight1[..., None]  # 3b n_k c_out
            bias = bias.sum(dim=1).flatten().contiguous()  # (3b x c_out)

            weight1 = rearrange(weight1, "b nk -> b nk 1 1 1 1")  # 3b n_k 1 1 1 1
            weight2 = rearrange(self.weight, "ci (nk co) k1 k2 -> 1 nk ci co k1 k2", nk=self.n_kernels)  # 1 n_k c_in c_out k k

            weight = th.sum(weight1 * weight2, dim=1)  # 3b c_in c_out k k
            weight = rearrange(weight, "b ci co k1 k2 -> (b ci) co k1 k2").contiguous()

            x = rearrange(x, "b c h w -> 1 (b c) h w").contiguous()  # 1 (3b x c_in) h w

            x = F.conv_transpose2d(x, weight, bias, self.stride, self.padding, groups=b3)  # 1 (3b x c_out) h w
            x = rearrange(x, "1 (b c) h w -> b c h w", b=b3).contiguous()  # 3b c_out h w

            if is_tuple:
                return x.chunk(3)
            return x
        else:
            return super().forward(x)
