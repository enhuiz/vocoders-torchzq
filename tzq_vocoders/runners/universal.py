import torchzq
from torchzq.typing import Custom, Parser

from ..models.distributions import (
    μLawCategoricalLayer,
    DiscretizedMixtureLogisticsLayer,
)
from ..models.universal import UniversalVocoder
from .base import Runner as BaseRunner


class Runner(BaseRunner):
    def __init__(self, dist: str = "μlaw", **kwargs):
        return super().__init__(**kwargs)

    def create_model(self):
        args = self.args
        if args.dist == "μlaw":
            dist_fn = lambda dim: μLawCategoricalLayer(dim)
        elif args.dist == "mol":
            dist_fn = lambda dim: DiscretizedMixtureLogisticsLayer(dim)
        else:
            raise NotImplementedError(args.dist)

        return UniversalVocoder(
            dist_fn=dist_fn,
            sample_rate=self.mel_fn.sample_rate,
            hop_length=self.mel_fn.hop_length,
            dim_mel=self.mel_fn.n_mels,
        )


def main():
    torchzq.start(Runner)


if __name__ == "__main__":
    main()
