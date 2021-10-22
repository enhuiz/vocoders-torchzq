import torchzq

from ..models.distributions import (
    MuLawCategoricalLayer,
    DiscretizedMixtureLogisticsLayer,
    RawCategoricalLayer,
)
from ..models.universal import UniversalVocoder
from .base import Runner as BaseRunner


class Runner(BaseRunner):
    def __init__(self, dist: str = "μlaw", **kwargs):
        return super().__init__(**kwargs)

    def create_model(self):
        args = self.args
        if args.dist == "μlaw":
            dist_factory = MuLawCategoricalLayer.create_factory(bits=9)
        elif args.dist == "raw":
            # raw bits better than μ-law?
            # https://github.com/G-Wang/WaveRNN-Pytorch/issues/2#issuecomment-438820768
            dist_factory = RawCategoricalLayer.create_factory(bits=10)
        elif args.dist == "mol":
            dist_factory = DiscretizedMixtureLogisticsLayer.create_factory()
        else:
            raise NotImplementedError(args.dist)

        return UniversalVocoder(
            dist_factory=dist_factory,
            sample_rate=self.mel_fn.sample_rate,
            hop_length=self.mel_fn.hop_length,
            dim_mel=self.mel_fn.n_mels,
        )


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
