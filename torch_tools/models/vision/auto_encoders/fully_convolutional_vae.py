import copy

from torch import nn

from torch_tools.models.vision import FCAE
from torch_tools.models.vision.util import DCEncoder


class FCVAE(FCAE):
    """
    Fully Convolutional Variational AutoEncoder
    """

    def __init__(self, input_channels, latent_channels=100, n_filters=64, n_pyramid=3, **kargs):
        super().__init__(
            input_channels=input_channels,
            latent_channels=latent_channels,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )

        # overriding encoder part (duplicate its final layer for mu and var)
        self.encoder = DCEncoder(input_channels, latent_channels=latent_channels, n_filters=n_filters,
                                 n_pyramid=n_pyramid, final_layer=False)  # without final layer
        self.encoder_out_mu = nn.Conv2d(n_filters * 2 ** n_pyramid, latent_channels, kernel_size=4, stride=1, padding=0,
                                        bias=False)
        self.encoder_out_var = copy.deepcopy(self.encoder_out_mu)

    def encode_mu_logvar(self, x):
        h = self.encoder(x)
        mu = self.encoder_out_mu(h)
        logvar = self.encoder_out_var(h)
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
