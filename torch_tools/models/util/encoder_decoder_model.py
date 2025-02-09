from abc import abstractmethod, ABCMeta
from functools import reduce
from typing import Any

import torch
from torch import nn


class Encoder(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def latent_size(self, *args):
        raise NotImplementedError

    def latent_dim(self, *args):
        return reduce(lambda x, y: x * y, self.latent_size(*args))

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)


class Decoder(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def latent_size(self, *args):
        raise NotImplementedError

    def latent_dim(self, *args):
        return reduce(lambda x, y: x * y, self.latent_size(*args))

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)


class AE(Encoder, Decoder, metaclass=ABCMeta):
    def reparameterize(self, mu, logvar):
        """
        Reparameterize trick for Variational Auto Encoders

        Args:
            mu (torch.Tensor):
            logvar (torch.Tensor):

        Returns (torch.Tensor): z

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)
