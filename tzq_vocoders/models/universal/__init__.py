"""Universal vocoder, adapted from https://github.com/yistLin/universal-vocoder"""

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from typing import Callable

from ..utils import tbc2bct, bct2tbc
from ..distributions import DistributionLayer
from ..layers import PositionalEncoding


@dataclass(eq=False)
class UniversalVocoder(nn.Module):
    dist_fn: Callable[..., DistributionLayer]
    sample_rate: int
    hop_length: int
    dim_mel: int
    dim_enc: int = 128
    dim_emb: int = 256
    dim_dec: int = 896
    dim_affine: int = 512

    def __post_init__(self):
        super().__init__()
        self.mel_rnn = nn.GRU(
            self.dim_mel,
            self.dim_enc,
            num_layers=2,
            bidirectional=True,
        )
        self.wav_emb = PositionalEncoding(
            self.dim_emb,
        )
        self.wav_rnn = nn.GRU(
            2 * self.dim_enc + self.dim_emb,
            self.dim_dec,
        )
        self.dist = self.dist_fn(self.dim_dec)

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

        # insert y0, remove the last y, name it as "i" for input
        y0 = torch.zeros((1, x.shape[1]), device=self.device)
        w = torch.cat([y0, y[:-1]])  # (t b)
        e = self.wav_emb(w)  # (t b d)

        o, _ = self.wav_rnn(torch.cat([e, x], dim=-1))

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
        yl = (xl * self.hop_length).long()

        x = self.encode(pad_sequence(x))
        y = []

        ht = None
        wt = torch.zeros((x.shape[1],), device=self.device)
        for xt in tqdm.tqdm(x, "Decoding ...") if verbose else x:
            et = self.wav_emb(wt)
            it = torch.cat([et, xt], dim=-1)
            ot, ht = self.wav_rnn(it[None], ht)
            yt = self.dist.sample(ot)
            y.append(yt.squeeze(0))

        y = torch.stack(y, dim=1)  # (b t)
        y = [yi[:li] for yi, li in zip(y, yl)]

        return y


if __name__ == "__main__":
    from ..distributions import μLawCategoricalLayer, DiscretizedMixtureLogisticsLayer

    model = UniversalVocoder(
        lambda dim: μLawCategoricalLayer(dim, 9),
        # lambda dim: DiscretizedMixtureLogisticsLayer(dim),
        sample_rate=16_000,
        hop_length=256,
        dim_mel=8,
        dim_emb=10,
        dim_dec=10,
    )
    mel = torch.randn(1, 3, 8)
    wav = torch.rand(256, 3) * 2 - 1
    loss = model(mel, wav)
    print(loss)
    wav = model.generate(list(mel.transpose(0, 1)))
    print(wav[0].shape)
