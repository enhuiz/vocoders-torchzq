import random
import numpy as np
import torch
import torchzq
import torch.nn.functional as F
from functools import cached_property
from torch.nn.utils.rnn import pad_sequence

from ..dataset import AudioDataset
from ..spectrogram import LogMelSpectrogram
from .utils import plot_tensor_to_numpy


class Runner(torchzq.Runner):
    def __init__(
        self,
        training_glob: str = "data/train/*.wav",
        validation_glob: str = "data/val/*.wav",
        crop_seconds: float = 0.8,
        n_demos: int = 4,
        eval_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @cached_property
    def mel_fn(self):
        # just use the default mel fn
        return LogMelSpectrogram()

    def create_dataloader(self, mode):
        args = self.args

        if mode == mode.TRAIN:
            pattern = args.training_glob
            batch_size = args.batch_size
        else:
            pattern = args.validation_glob
            batch_size = args.eval_batch_size

        dataset = AudioDataset(pattern, self.mel_fn.cpu())

        dataloader = dataset.as_dataloader(
            batch_size=batch_size,
            num_workers=args.nj,
            shuffle=mode == mode.TRAIN,
            drop_last=mode == mode.TRAIN,
        )

        print("Dataset size:", len(dataset))

        return dataloader

    def crop_to_tensor(self, x, rate, mode):
        args = self.args
        crop_length = int(rate * args.crop_seconds)
        if len(x) < crop_length:
            x = np.pad(x, (0, crop_length - len(x)))
        if mode == mode.TRAIN:
            # random crop
            start = random.randint(0, len(x) - crop_length - 1)
        else:
            # center crop
            start = (len(x) - crop_length) // 2
        x = x[start : start + crop_length]
        x = torch.from_numpy(x)
        return x

    def prepare_batch(self, batch, mode):
        mel = batch["mel"]  # list of (t c)
        wav = batch["wav"]  # list of (t)
        mel = [self.crop_to_tensor(m, self.mel_fn.rate, mode) for m in mel]
        wav = [self.crop_to_tensor(w, self.mel_fn.sample_rate, mode) for w in wav]
        batch = dict(mel=pad_sequence(mel), wav=pad_sequence(wav))
        return super().prepare_batch(batch, mode)

    def training_step(self, batch, _):
        loss = self.model(x=batch["mel"], y=batch["wav"])
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        args = self.args

        stat_dict = super().validation_step(batch, batch_idx)

        if batch_idx * args.eval_batch_size >= args.n_demos:
            return stat_dict

        if batch_idx == 0:
            self.plot_gt = not getattr(self, "gt_plotted", False)
            self.gt_plotted = True

        logger = self.logger

        log_dict = {"epoch": self.current_epoch}

        fwav = self.model.generate(list(batch["mel"].transpose(0, 1)))
        fmel = [self.mel_fn(x.cpu(), dim=-1) for x in fwav]

        make_wav = lambda x: logger.Audio(x.cpu().detach(), args.sample_rate)
        make_mel = lambda x: logger.Image(plot_tensor_to_numpy(x))

        mel_mses = []

        for i, (wav_i, mel_i), (fwav_i, fmel_i) in zip(
            range(batch_idx * args.eval_batch_size, args.n_demos),
            zip(batch["wav"], batch["mel"]),
            zip(fwav, fmel),
        ):
            if self.plot_gt:
                log_dict |= {
                    f"demo/{i}/wav": make_wav(wav_i.t()),
                    f"demo/{i}/mel": make_mel(mel_i.t()),
                }

            log_dict |= {
                f"demo/{i}/fwav": make_wav(fwav_i.t()),
                f"demo/{i}/fmel": make_mel(fmel_i.t()),
            }

            mel_mses.append(F.mse_loss(fmel_i, mel_i.cpu()).item())

        stat_dict["mel_mse"] = np.mean(mel_mses)
        logger.log(log_dict, self.global_step)

        return stat_dict

    def testing_step(self, batch, batch_idx):
        raise NotImplementedError


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
