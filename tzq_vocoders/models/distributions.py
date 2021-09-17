import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from abc import ABC, abstractmethod


class DiscretizedDistributionLayer(ABC, nn.Module):
    def __init__(self, bits=9):
        super().__init__()
        self.bits = bits

    @abstractmethod
    def log_prob(self, x, y) -> torch.Tensor:
        ...

    @abstractmethod
    def neg_log_prob(self, x, y) -> torch.Tensor:
        ...

    @abstractmethod
    def sample(self, x) -> torch.Tensor:
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_quants(self):
        return 2 ** self.bits

    @property
    def num_quants_minus_1(self):
        return self.num_quants - 1


class RawCategoricalLayer(DiscretizedDistributionLayer):
    def __init__(self, input_dim, bits=9):
        super().__init__(bits)
        self.linear = nn.Linear(input_dim, self.num_quants)

    def log_prob(self, x, y):
        return self.neg_log_prob(x, y).neg()

    def quantize(self, y):
        y = y.clamp(-1, 1)
        return (((y + 1) / 2) * self.num_quants_minus_1).long()

    def dequantize(self, y):
        return ((y / self.num_quants_minus_1) * 2 - 1).clamp(-1, 1)

    def neg_log_prob(self, x, y):
        logits = self.linear(x).transpose(-1, -2)  # (t c b)
        return F.cross_entropy(logits, self.quantize(y), reduction="none")

    def sample(self, x):
        logits = self.linear(x)
        if self.training:
            z = Categorical(logits=logits).sample()
        else:
            z = logits.argmax(dim=-1)
        z = self.dequantize(z)
        return z


class MuLawCategoricalLayer(RawCategoricalLayer):
    @property
    def μ(self):
        return self.num_quants_minus_1

    def quantize(self, y):
        y = y.clamp(-1, 1)
        y = y.sign() * ((self.μ * y.abs()).log1p() / math.log1p(self.μ))
        return ((y + 1) / 2 * self.μ + 0.5).long()

    def dequantize(self, y):
        y = (y - 0.5) / self.μ * 2 - 1
        y = (y.sign() / self.μ) * ((1 + self.μ) ** y.abs() - 1)
        return y.clamp(-1, 1)


class DiscretizedMixtureLogisticsLayer(DiscretizedDistributionLayer):
    def __init__(self, input_dim, num_mixtures=10):
        super().__init__(bits=8)
        self.num_mixtures = num_mixtures
        self.linear = nn.Linear(input_dim, num_mixtures * 3)

    def logistics_cdf(self, y, μ, s):
        return ((y - μ) / s).sigmoid()

    def log_prob(self, x, y, ε=1e-12, min_logs=-7.0):
        """
        Args:
            x: (t b d)
            y: (t b)
        Returns:
            log_prob: (t b)
        """
        half_bin_size = 1 / self.num_quants_minus_1

        y = y.unsqueeze(-1)  # (t b) -> (t b 1)
        y = y.clamp(-0.999 + half_bin_size, 0.999 - half_bin_size)

        logits, μ, logs = self.linear(x).chunk(3, dim=-1)
        logs = logs.clamp(min=min_logs)
        s = logs.exp()

        cdf_plus = self.logistics_cdf(y + half_bin_size, μ, s)
        cdf_minus = self.logistics_cdf(y - half_bin_size, μ, s)
        cdf_delta = (cdf_plus - cdf_minus).clamp(min=ε)  # will this simple clamp work?

        log_prob = cdf_delta.log() + F.log_softmax(logits, dim=-1)
        log_prob = log_prob.logsumexp(dim=-1)

        return log_prob

    def neg_log_prob(self, x, y):
        return self.log_prob(x, y).neg()

    def sample(self, x):
        logits, μ, logs = self.linear(x).chunk(3, dim=-1)

        k = Categorical(logits=logits).sample().unsqueeze(-1)  # (... 1)

        μ = μ.gather(dim=-1, index=k).squeeze(-1)
        logs = logs.gather(dim=-1, index=k).squeeze(-1)

        p = torch.rand_like(μ) * (1 - 1e-5) + 1e-5
        y = μ + logs.exp() * (p.log() - (1 - p).log())

        y = y.clamp(-1, 1)

        return y
