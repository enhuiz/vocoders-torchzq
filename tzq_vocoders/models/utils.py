import contextlib
from einops import rearrange, repeat
from functools import partial

bct2tbc = partial(rearrange, pattern="b c t -> t b c")
tbc2bct = partial(rearrange, pattern="t b c -> b c t")


class Savable:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_for_later = {}
        self._saving_prefix = ""

    @property
    def device(self):
        return next(self.parameters()).device

    def save_for_later(self, **kwargs):
        kwargs = {self._saving_prefix + k: v for k, v in kwargs.items()}
        twice = kwargs.keys() & self._saved_for_later.keys()
        assert not twice, f"Variable(s) with the same name saved twice. {twice}."
        self._saved_for_later.update(kwargs)

    @contextlib.contextmanager
    def saving_prefix(self, prefix=""):
        self._saving_prefix = prefix
        yield
        self._saving_prefix = ""

    def __call__(self, *args, **kwargs):
        self._saved_for_later.clear()
        # this super may not be just calling its parent
        # details: https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
        return super().__call__(*args, **kwargs)
