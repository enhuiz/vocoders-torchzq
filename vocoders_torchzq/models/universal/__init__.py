"""Universal vocoder, adapted from https://github.com/yistLin/universal-vocoder"""

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import Callable

from ..utils import tbc2bct, bct2tbc
from ..distributions import DiscretizedDistributionLayer


@dataclass(eq=False)
class UniversalVocoder(nn.Module):
    dist_factory: Callable[..., DiscretizedDistributionLayer]
    sample_rate: int
    hop_length: int
    dim_mel: int
    dim_enc: int = 128
    dim_vec: int = 256
    dim_dec: int = 896
    dim_proj: int = 512

    def __post_init__(self):
        super().__init__()
        self.mel_rnn = nn.GRU(
            self.dim_mel,
            self.dim_enc,
            num_layers=2,
            bidirectional=True,
        )

        self.wav_rnn = nn.GRU(
            2 * self.dim_enc + self.dim_vec,
            self.dim_dec,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.dim_dec, self.dim_proj),
            nn.GELU(),
        )

        self.dist = self.dist_factory(
            dim_vec=self.dim_vec,
            dim_proj=self.dim_proj,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        """
        Args:
            x: (t b d)
        Return:
            c: (t b d)
        """
        c, _ = self.mel_rnn(x)
        c = tbc2bct(c)
        size = (int(c.shape[-1] * self.hop_length),)
        assert size[0] % self.hop_length == 0
        c = F.interpolate(c, size)
        c = bct2tbc(c)
        return c

    def forward(self, x, y):
        """Generate waveform from mel-spectrogram with teacher-forcing.
        Args:
            x: conditioned, e.g. mel  (t_m b c_mel)
            y: wavefrom (t_w b), continous [-1, 1]
        Returns:
            nll, the loss
        """
        x = self.encode(x)  # (t b c)

        # w: the y as input
        # insert w0, remove the last y
        w0 = torch.zeros((1, x.shape[1]), device=self.device)
        w = torch.cat([w0, y[:-1]])  # (t b)

        e = self.dist.vectorize(w)  # (t_w b d)
        o, _ = self.wav_rnn(torch.cat([e, x], dim=-1))
        o = self.proj(o)

        # average over time, average over batch
        nll = self.dist.neg_log_prob(o, y).mean()

        return nll

    @torch.no_grad()
    def generate(self, x, verbose=True):
        """Generate waveform from x spectrogram.
        Args:
            x: list of tensor (t c)
        Returns:
            y: list of tensor (t)
        """
        xl = torch.tensor(list(map(len, x)))
        wl = (xl * self.hop_length).long()

        x = self.encode(pad_sequence(x))

        w0 = torch.zeros((x.shape[1],), device=self.device)
        w = [w0]

        et = self.dist.vectorize(w0)
        ht = None
        pbar = tqdm.tqdm(x, "Decoding ...") if verbose else x
        for xt in pbar:
            it = torch.cat([et, xt], dim=-1)
            ot, ht = self.wav_rnn(it[None], ht)
            ot = self.proj(ot.squeeze(0))
            wt, et = self.dist.sample(ot, return_vectorized=True)
            w.append(wt)

        w = torch.stack(w[1:], dim=1)  # (b t), [1:] to remove w0
        w = [wi[:li] for wi, li in zip(w, wl)]

        return w


if __name__ == "__main__":
    from ..distributions import (
        MuLawCategoricalLayer,
        DiscretizedMixtureLogisticsLayer,
        RawCategoricalLayer,
    )

    model = UniversalVocoder(
        lambda dim: MuLawCategoricalLayer(dim, 9),
        # lambda dim: DiscretizedMixtureLogisticsLayer(dim),
        # lambda dim: RawCategoricalLayer(dim),
        sample_rate=16_000,
        hop_length=256,
        dim_mel=8,
        dim_vec=10,
        dim_dec=10,
        dim_proj=10,
    )
    mel = torch.randn(1, 3, 8)
    wav = torch.rand(256, 3) * 2 - 1
    loss = model(mel, wav)
    print(loss)
    wav = model.generate(list(mel.transpose(0, 1)))
    print(wav[0].shape)
