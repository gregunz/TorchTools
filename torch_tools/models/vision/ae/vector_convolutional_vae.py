import copy

from torch import nn

from torch_tools.models.util import Flatten
from . import VectorCAE


class VectorCVAE(VectorCAE):
    """
    Variational variant of `VectorCAE`, refer to `VectorCAE`'s doc for more details.
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

    def encode(self, x):
        mu, logvar = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode_mu_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
