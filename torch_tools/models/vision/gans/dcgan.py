from torch import nn

from torch_tools.models.vision.dc_coder import DCDecoder, DCEncoder


class _Generator(DCDecoder):
    def __init__(self, in_channels, out_channels, n_filters=64, n_pyramid=3, use_custom_weight_init=True):
        super().__init__(
            out_channels=out_channels,
            latent_channels=in_channels,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )
        # if use_custom_weight_init:
        #     self.apply(weights_init)


class _Discriminator(DCEncoder):
    def __init__(self, in_channels, out_channels, n_filters=64, n_pyramid=3, use_custom_weight_init=True):
        super().__init__(
            in_channels=in_channels,
            latent_channels=out_channels,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )
        # if use_custom_weight_init:
        #     self.apply(weights_init)


class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=64, n_pyramid=3, use_custom_weight_init=True):
        super().__init__()

        self.generator = DCDecoder(
            out_channels=out_channels,
            latent_channels=in_channels,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )

        self.discriminator = DCEncoder(
            in_channels=in_channels,
            latent_channels=out_channels,
            n_filters=n_filters,
            n_pyramid=n_pyramid,
        )

        if use_custom_weight_init:
            self.apply(self.weights_init)

    # custom weights initialization
    @staticmethod
    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)
