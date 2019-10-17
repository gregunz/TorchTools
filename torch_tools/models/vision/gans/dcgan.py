from torch import nn

from torch_tools.models.util import GAN
from torch_tools.models.vision.util import DCDecoder, DCEncoder


class DCGAN(GAN):
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

    def __init__(self, latent_dim, img_channels, n_filters=64, n_pyramid=4, use_custom_weight_init=True):
        super().__init__()

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

        if use_custom_weight_init:
            self.apply(self.weights_init)

    @property
    def generator(self) -> nn.Module:
        return self._generator

    @property
    def discriminator(self) -> nn.Module:
        return self._discriminator

    @property
    def input_size(self):
        return self.latent_dim, 1, 1

    # custom weights initialization
    @staticmethod
    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)
