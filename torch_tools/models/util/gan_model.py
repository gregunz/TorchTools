from abc import abstractmethod
from typing import Any

from torch import nn


class GAN(nn.Module):
    @property
    @abstractmethod
    def input_size(self):
        return self._input_size

    @property
    @abstractmethod
    def generator(self) -> nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self) -> nn.Module:
        raise NotImplementedError

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)
