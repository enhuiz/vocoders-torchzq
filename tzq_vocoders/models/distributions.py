import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DistributionLayer(ABC, nn.Module):
    @abstractmethod
    def log_prob(self, x, y) -> torch.Tensor:
        ...

    @abstractmethod
    def neg_log_prob(self, x, y) -> torch.Tensor:
        ...

    @abstractmethod
    def sample(self, x) -> torch.Tensor:
        ...


class μLawCategoricalLayer(DistributionLayer):
    def __init__(self, input_dim, bits=9):
        super().__init__()
        self.bits = bits
        self.linear = nn.Linear(input_dim, self.num_quants)

    @property
    def num_quants(self):
        return self.bits ** 2

    @property
    def μ(self):
        return self.num_quants - 1

    def μ_law_encode(self, x):
        x = x.clamp(-1, 1)
        x = x.sign() * ((self.μ * x.abs()).log1p() / math.log1p(self.μ))
        return ((x + 1) / 2 * self.μ + 0.5).long()

    def μ_law_decode(self, x):
        x = (x - 0.5) / self.μ * 2 - 1
        x = (x.sign() / self.μ) * ((1 + self.μ) ** x.abs() - 1)
        return x.clamp(-1, 1)

    def log_prob(self, x, y):
        return self.neg_log_prob(x, y).neg()

    def neg_log_prob(self, x, y):
        logits = self.linear(x).transpose(-1, -2)  # (t c b)
        y = self.μ_law_encode(y)  # (t b)
        return F.cross_entropy(logits, y, reduction="none")

    def sample(self, x):
        logits = self.linear(x)
        if self.training:
            z = F.gumbel_softmax(logits, 1, hard=True)
        else:
            z = logits.argmax(dim=-1)
        z = self.μ_law_decode(z)
        return z
