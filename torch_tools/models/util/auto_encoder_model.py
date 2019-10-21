from abc import abstractmethod
from functools import reduce

import torch
from torch import nn


class AE(nn.Module):
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
