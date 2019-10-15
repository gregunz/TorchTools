from abc import abstractmethod
from functools import reduce

from torch import nn


class AEModel(nn.Module):
    @abstractmethod
    def encode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def latent_size(self, *args):
        raise NotImplementedError

    def latent_dim(self, *args):
        return reduce(lambda x, y: x * y, self.latent_size(*args))
