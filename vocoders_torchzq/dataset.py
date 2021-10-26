import math
import csv
import attr
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from mmds import MultimodalDataset, MultimodalSample
from mmds.modalities.mel import MelModality
from mmds.modalities.wav import WavModality


@attr.define
class AudioSample(MultimodalSample):
    trim_randomly: bool
    trim_seconds: Optional[float] = None

    @staticmethod
    def convert_to_integer(x):
        assert x.is_integer(), "Random trim only support integer sample rate for now."
        return int(x)

    def generate_info(self):
        if self.trim_seconds is None:
            return {}

        mel_modality = self.get_modality_by_name("mel")
        wav_modality = self.get_modality_by_name("wav")

        # least common period where timesteps overlap
        # e.g. 16k & 80 hz -> greatest common rate 80hz,
        # -> 1/80 least common period
        lcp = 1 / math.gcd(
            self.convert_to_integer(wav_modality.sample_rate),
            self.convert_to_integer(mel_modality.sample_rate),
        )

        max_start = mel_modality.duration - self.trim_seconds
        possible_starts = np.arange(0, max_start, lcp)

        if len(possible_starts) == 0:
            possible_starts = [0]

        if self.trim_randomly:
            t0 = np.random.choice(possible_starts)
        else:
            # center trim
            t0 = possible_starts[len(possible_starts) // 2]

        t1 = t0 + self.trim_seconds

        ret = dict(t0=t0, t1=t1)

        return ret


class AudioDataset(MultimodalDataset, Dataset):
    def __init__(
        self,
        root,
        split,
        mel_fn,
        trim_randomly,
        trim_seconds,
        split_folder="split/110s",
        wav_folder="audio",
        wav_suffix=".wav",
        mel_folder="mel",
        mel_suffix=".npz",
    ):
        path = (root / split_folder / split).with_suffix(".csv")
        df: pd.DataFrame = pd.read_csv(path, sep="|", quoting=csv.QUOTE_NONE, dtype=str)
        df = df.fillna("")

        super().__init__(
            [
                AudioSample(str(id), trim_randomly, trim_seconds)
                for id in df["id"].tolist()
            ],
            [
                WavModality.create_factory(
                    name="wav",
                    root=root / wav_folder,
                    suffix=wav_suffix,
                    sample_rate=mel_fn.sample_rate,
                ),
                MelModality.create_factory(
                    name="mel",
                    root=root / mel_folder,
                    suffix=mel_suffix,
                    mel_fn=mel_fn,
                    base_modality_name="wav",
                ),
            ],
        )
