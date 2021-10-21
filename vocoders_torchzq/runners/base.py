import numpy as np
import torch
import torchzq
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from functools import cached_property
from torch.nn.utils.rnn import pad_sequence
from mmds.utils.spectrogram import LogMelSpectrogram


from ..dataset import AudioDataset
from .utils import plot_tensor_to_numpy


class Runner(torchzq.Runner):
    def __init__(
        self,
        # dataset
        data_root: Path = Path("data/vctk"),
        wav_folder: str = "wav48_silence_trimmed",
        wav_suffix: str = ".flac",
        # mel
        sample_rate: int = 16000,
        f_min: int = 55,
        f_max: int = 7600,
        hop_length: int = 200,
        # train
        trim_seconds: float = 0.6,
        # eval
        n_demos: int = 4,
        eval_batch_size: int = 8,
        # misc
        wandb_project: str = "vocoders",
        **kwargs,
    ):
        super().__init__(**kwargs)

    @cached_property
    def mel_fn(self):
        args = self.args
        mel_fn = LogMelSpectrogram(
            sample_rate=args.sample_rate,
            f_min=args.f_min,
            f_max=args.f_max,
            hop_length=args.hop_length,
        )
        print(mel_fn)
        return mel_fn

    def create_dataloader(self, mode):
        args = self.args

        trim_seconds = None if mode == mode.TEST else args.trim_seconds

        if mode == mode.TRAIN:
            split = "train"
            batch_size = args.batch_size
            trim_randomly = True
        else:
            split = "val" if mode == mode.VAL else "test"
            batch_size = args.eval_batch_size
            trim_randomly = False

        dataset = AudioDataset(
            args.data_root,
            split,
            self.mel_fn,
            trim_randomly,
            trim_seconds,
            wav_folder=args.wav_folder,
            wav_suffix=args.wav_suffix,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=args.nj,
            shuffle=mode == mode.TRAIN,
            drop_last=mode == mode.TRAIN,
            collate_fn=self.collate,
        )

        print("Dataset size:", len(dataset))

        return dataloader

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

    def n2pt(self, x):
        """numpy to padded tensor"""
        device = self.args.device
        return pad_sequence([torch.tensor(xi.squeeze(1), device=device) for xi in x])

    def prepare_batch(self, batch, mode):
        mel = batch["mel"]  # list of (t 1 c)
        wav = batch["wav"]  # list of (t 1)
        batch = dict(mel=self.n2pt(mel), wav=self.n2pt(wav))
        return batch

    def training_step(self, batch, _):
        args = self.args
        loss = self.model(x=batch["mel"], y=batch["wav"])
        return loss, {"loss": loss.item(), "lr": args.lr()}

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
        fmel = [self.mel_fn(x.cpu(), dim=0) for x in fwav]

        make_wav = lambda x: logger.Audio(x.cpu().detach(), self.mel_fn.sample_rate)
        make_mel = lambda x: logger.Image(plot_tensor_to_numpy(x))

        mel_mses = []

        for i, (wav_i, mel_i), (fwav_i, fmel_i) in zip(
            range(batch_idx * args.eval_batch_size, args.n_demos),
            zip(
                batch["wav"].transpose(0, 1),
                batch["mel"].transpose(0, 1),
            ),
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
