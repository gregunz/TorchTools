import copy

import torch
from torch import nn

from torch_tools.models.util import Flatten
from . import VectorCAE


class VectorCVAE(VectorCAE):
    """
    Convolutional Variational AutoEncoder with Fully Connected layer in between to create a latent vector.

    Note that latent dim is the exact number of dimension of the latent vector.
    """

    def __init__(self, input_size, latent_dim=100, n_filters=64, n_pyramid=None, **kwargs):
        super().__init__(input_size, latent_dim, n_filters, n_pyramid)
        self.fc_enc2 = copy.deepcopy(self.fc_enc)

    def encode_mu_logvar(self, x):
        h = nn.Sequential(
            self.encoder,
            nn.ReLU(),
            Flatten(),
        )(x)
        mu = self.fc_enc(h)
        logvar = self.fc_enc2(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        mu, logvar = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
