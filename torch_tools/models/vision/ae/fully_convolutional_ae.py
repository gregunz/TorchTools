from test_tube import HyperOptArgumentParser

from torch_tools.models.util import AE
from torch_tools.models.vision.util import DCEncoder, DCDecoder


class FCAE(AE):
    """
    Fully Convolutional AutoEncoder
    """

    def __init__(self, input_channels, latent_channels, n_filters, n_pyramid, **kargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_pyramid = n_pyramid

        self.encoder = DCEncoder(input_channels, latent_channels=latent_channels, n_filters=n_filters,
                                 n_pyramid=n_pyramid)
        self.decoder = DCDecoder(input_channels, latent_channels=latent_channels, n_filters=n_filters,
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
    def add_argz(parser: HyperOptArgumentParser):
        default_latent_channels, default_latent_channels_opt = 100, (50, 100)
        parser.opt_list('--latent_channels', type=int, default=default_latent_channels,
                        options=default_latent_channels_opt, tunable=True,
                        help=f'latent channels (default: {default_latent_channels})')

        default_n_filters, default_n_filters_opt = 64, (32, 64)
        parser.opt_list('--n_filters', type=int, default=default_n_filters, options=default_n_filters_opt,
                        tunable=True, help=f'num of filters for the 1st pyramid block (default: {default_n_filters})')

        default_n_pyramid = 3
        parser.add_argument('--n_pyramid', type=int, default=default_n_pyramid,
                            help=f'number of pyramid blocks (default: {default_n_pyramid})')
