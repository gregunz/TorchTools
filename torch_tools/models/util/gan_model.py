from abc import abstractmethod
from typing import Any

from torch import nn


class GAN(nn.Module):
    @property
    @abstractmethod
    def generator(self) -> nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self) -> nn.Module:
        raise NotImplementedError

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)
