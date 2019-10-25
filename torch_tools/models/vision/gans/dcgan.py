from argparse import ArgumentParser

from torch import nn

from torch_tools.models.util import GAN, FISModel
from torch_tools.models.vision.util import DCDecoder, DCEncoder

_ld = 128  # default latent_dim
_nf = 64  # default n_filters
_np = 4  # default n_pyramid
_wi = True  # default use_custom_weight_init


class DCGAN(GAN, FISModel):
    """
    DCGAN Implementation <https://arxiv.org/abs/1511.06434>

    Args:
        latent_dim: size of the dimension of the latent vector (latent_dim x 1 x 1) used for generator input.
        img_channels: number of channels of the generated images.
        n_filters: number of filters (kernels) used in the first `PyramidBlock`, then it grows exponentially
         with the number of `PyramidBlock` blocks. It controls the capacity of the model.
        n_pyramid: number of pyramid blocks, it is related to the image size (H x W). Input image must be
         squared (H = W) and powers of 2 starting at 8. `n_pyramid = log_2(H / 8)`.
        use_custom_weight_init: whether to use the weight initialization proposed in the paper.
    """

    def __init__(self, img_channels, latent_dim=_ld, n_filters=_nf, n_pyramid=_np, use_custom_weight_init=_wi,
                 **kwargs):
        super().__init__(input_size=(latent_dim, 1, 1))

        self._generator = DCDecoder(
            out_channels=img_channels,
            latent_channels=latent_dim,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )

        self._discriminator = DCEncoder(
            in_channels=img_channels,
            latent_channels=1,  # binary output (real/fake)
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )

        self.latent_dim = latent_dim

        h = 2 ** (n_pyramid + 2)
        self.image_size = (img_channels, h, h)

        if use_custom_weight_init:
            self.apply(self.weights_init)

    @property
    def generator(self) -> nn.Module:
        return self._generator

    @property
    def discriminator(self) -> nn.Module:
        return self._discriminator

    # custom weights initialization
    @staticmethod
    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    @staticmethod
    def add_argz(parser: ArgumentParser):
        parser.add_argument('--latent_dim', type=int, default=_ld, help=f'latent dim (default: {_ld})')
        parser.add_argument('--n_pyramid', type=int, default=_np, help=f'number of pyramid blocks (default: {_np})')
        parser.add_argument('--n_filters', type=int, default=_nf,
                            help=f'num of filters for the 1st pyramid block (default: {_nf})')
        parser.add_argument('--no_custom_weight_init', action='store_false', default=not _nf,
                            help=f'use this flag for not using the weight initialization proposed in the paper')
