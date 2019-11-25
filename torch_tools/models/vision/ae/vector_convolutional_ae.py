from functools import reduce

from torch import nn

from torch_tools.models.util import Flatten, UnFlatten, FISModel, AE
from torch_tools.models.vision.util import DCDecoder, DCEncoder

_ld = 100
_nf = 64
_np = None


class VectorCAE(AE, FISModel):
    """
    Convolutional AutoEncoder with Fully Connected layer in between to create a latent vector.

    Note that latent dim is the exact number of dimension of the latent vector (hence -> vector in the name).
    Because of that, it requires to know the input size in order to construct a fully connected layer with
    the right input size.

    `n_filters` controls the capacity of the model, it is the number of filters (kernels) used in the
    first `PyramidBlock`, then it grows exponentially with the number of `PyramidBlock` blocks.

    By default, the number of `PyramidBlock` is automatically computed to have the maximum of them
    (until the image size (width or height) cannot be divided by 2 anymore).
    This way there is only the need to provide the input size for small image dataset such as MNIST or
    CIFAR10 for which the maximum number of `PyramidBlock` is quite limited.

        Args:
            input_size (tuple): input data size
            latent_dim (int): number of dimension of the latent vector
            n_filters (int, optional): number of filter of the first `PyramidBlock`
            n_pyramid (int): number of `PyramidBlock`
            **kwargs: not used (practical when feeding **vars(args) in constructor)
    """

    def __init__(self, input_size, latent_dim=_ld, n_filters=_nf, n_pyramid=_np, **kwargs):
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
        parser.add_argument('--latent_dim', type=int, default=_ld, help=f'latent dim (default: {_ld})')
        parser.add_argument('--n_pyramid', type=int, default=_np, help=f'number of pyramid blocks (default: {_np})')
        parser.add_argument('--n_filters', type=int, default=_nf,
                            help=f'num of filters for the 1st pyramid block (default: {_nf})')

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
