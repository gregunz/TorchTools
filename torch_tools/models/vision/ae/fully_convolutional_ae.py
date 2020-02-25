from argparse import ArgumentParser

from torch_tools.models.util import AE
from torch_tools.models.vision.util import DCEncoder, DCDecoder

_lc = 128
_nf = 64
_np = 3


class FCAE(AE):
    """
    Fully Convolutional AutoEncoder
    """

    def __init__(self, in_channels, latent_channels, n_filters, n_pyramid, **kargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_pyramid = n_pyramid

        self.encoder = DCEncoder(in_channels, latent_channels=latent_channels, n_filters=n_filters,
                                 n_pyramid=n_pyramid)
        self.decoder = DCDecoder(in_channels, latent_channels=latent_channels, n_filters=n_filters,
                                 n_pyramid=n_pyramid)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def latent_size(self, input_width, input_height):
        mult = 2 ** (1 + self.n_pyramid)
        return self.latent_channels, input_width // mult - 3, input_height // mult - 3

    def latent_dim(self, input_width, input_height):
        return super().latent_dim(input_width, input_height)

    @staticmethod
    def add_argz(parser: ArgumentParser):
        parser.add_argument('--latent_channels', type=int, default=_lc, help=f'latent channels')
        parser.add_argument('--n_pyramid', type=int, default=_np, help=f'number of pyramid blocks)')
        parser.add_argument('--n_filters', type=int, default=_nf, help=f'num of filters for the 1st pyramid block')
