import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .spectrogram import LogMelSpectrogram


class Cache:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(exist_ok=True, parents=True)

    def write(self, rpath, data, validator):
        path = self.root / rpath
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(path, data, validator)

    def read(self, rpath, validator):
        ret = None
        path = self.root / rpath
        if path.exists():
            try:
                data = np.load(path)
                if validator == data["arr_1"]:
                    ret = data["arr_0"]
            except:
                pass
        return ret


class AudioDataset(Dataset):
    def __init__(self, pattern: str, mel_fn: LogMelSpectrogram):
        super().__init__()
        self.paths = sorted(glob.glob(pattern, recusive=True))
        self.mel_fn = mel_fn
        self.cache = Cache(".cache")

    def to_rpath(self, path, suffix):
        return Path(path).relative_to(".").with_suffix(suffix)

    def load_wav(self, path):
        return librosa.load(path, sr=self.mel_fn.sample_rate)[0]  # (t)

    def load_mel(self, path, wav=None):
        validator = str(self.mel_fn)
        rpath = self.to_rpath(path, ".mel.pth")
        mel = self.cache.read(rpath, validator)
        if mel is None:
            wav = self.load_wav(path) if wav is None else wav
            with torch.no_grad():
                wav = torch.from_numpy(wav)  # (t c)
                mel = self.mel_fn(wav, dim=0).numpy()  # (t c d)
            self.cache.write(rpath, mel, validator)
        return mel

    def __getitem__(self, index):
        path = self.paths[index]
        wav = self.load_wav(path)
        mel = self.load_mel(path, wav)
        return dict(wav=wav, mel=mel)

    def __len__(self):
        return len(self.paths)

    def as_dataloader(self, *args, **kwargs):
        kwargs.setdefault("collate_fn", self.collate)
        return DataLoader(self, *args, **kwargs)

    @staticmethod
    def collate(samples):
        batch = {}
        for sample in samples:
            for k, v in sample.items():
                if k not in batch:
                    batch[k] = [v]
                else:
                    batch[k].append(v)
        return batch
