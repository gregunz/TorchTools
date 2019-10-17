from abc import abstractmethod

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
