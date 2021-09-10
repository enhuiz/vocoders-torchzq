import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, linear_scale=5000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32)
        exponents = exponents / half_dim
        ω = linear_scale * torch.exp(-math.log(1e4) * exponents)
        ω = ω.unsqueeze(0)
        self.register_buffer("ω", ω, False)

    def forward(self, t):
        """
        Args:
            t: (... 1) or (...)
        Returns:
            pe: (... d)
        """
        shape = t.shape
        t = t.flatten().unsqueeze(-1)
        ωt = self.ω * t
        e = torch.cat([ωt.sin(), ωt.cos()], dim=-1)
        return e.view(*(shape + (self.dim,)))
