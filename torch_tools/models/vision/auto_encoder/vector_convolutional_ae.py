from functools import reduce

from torch import nn

from torch_tools.models.util import Flatten, UnFlatten, FISModel
from torch_tools.models.vision.util import DCDecoder, DCEncoder, AEModel


class VectorCAE(AEModel, FISModel):
    """
    Convolutional AutoEncoder with Fully Connected layer in between to create a latent vector.

    Note that latent dim is the exact number of dimension of the latent vector.
    """

    def __init__(self, input_size, latent_dim=100, n_filters=64, n_pyramid=None, **kwargs):
        super().__init__(input_size)
        self.latent_channels = latent_dim
        if n_pyramid is None:
            n_pyramid = 1
            while self._check_input_size(n_pyramid + 1):
                n_pyramid += 1

        self._check_input_size(n_pyramid, do_assert=True)

        mult = 2 ** (1 + n_pyramid)
        self.encoder_out_size = (latent_dim, self.input_width // mult - 3, self.input_height // mult - 3)
        self.encoder_out_dim = reduce(lambda x, y: x * y, self.encoder_out_size)

        self.encoder = DCEncoder(self.input_channels, latent_channels=latent_dim, n_filters=n_filters,
                                 n_pyramid=n_pyramid)
        self.fc_enc = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.encoder_out_dim)
        self.decoder = DCDecoder(self.input_channels, latent_channels=latent_dim, n_filters=n_filters,
                                 n_pyramid=n_pyramid)

    def encode(self, x):
        return nn.Sequential(
            self.encoder,
            nn.ReLU(),
            Flatten(),
            self.fc_enc,
        )(x)

    def decode(self, z):
        return nn.Sequential(
            self.fc_dec,
            nn.ReLU(),
            UnFlatten(*self.encoder_out_size),
            self.decoder,
        )(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def latent_size(self):
        # noinspection PyRedundantParentheses
        return (self.latent_channels,)

    @staticmethod
    def add_argz(parser):
        default_latent_dim, default_latent_size_opt = 100, (50, 100)
        parser.opt_list('--latent_dim', type=int, default=default_latent_dim, options=default_latent_size_opt,
                        tunable=True, help=f'latent dim (default: {default_latent_dim})')

        default_n_filters, default_n_filters_opt = 32, (32, 64)
        parser.opt_list('--n_filters', type=int, default=default_n_filters, options=default_n_filters_opt,
                        tunable=True,
                        help=f'num of filters for the 1st pyramid block (default: {default_n_filters})')

        default_n_pyramid = None
        parser.add_argument('--n_pyramid', type=int, default=default_n_pyramid,
                            help=f'number of pyramid blocks (default: {default_n_pyramid})')

    def _check_input_size(self, n_pyramid, do_assert=False) -> bool:
        mult = 2 ** (1 + n_pyramid)  # minimum multiple for image width and height
        is_height_mult = self.input_height % mult == 0
        if do_assert:
            assert is_height_mult, f'{self.input_height} % {mult} != 0, height should be a multiple of {mult}'
        is_width_mult = self.input_width % mult == 0
        if do_assert:
            assert is_width_mult, f'{self.input_width} % {mult} != 0, width should be a multiple of {mult}'

        min_size = 4 * mult  # minimum size
        has_min_height = self.input_height >= min_size
        if do_assert:
            assert has_min_height, f'{self.input_height} < {min_size}, height should be >= {min_size}'
        has_min_width = self.input_width >= min_size
        if do_assert:
            assert has_min_width, f'{self.input_width} < {min_size}, width should be >= {min_size}'

        return is_height_mult and is_width_mult and has_min_height and has_min_width
