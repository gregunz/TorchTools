from torch import nn

from . import PyramidDown, PyramidUp


class DCEncoder(nn.Module):
    """
    Deep Convolutional Encoder

    Inspired from DCGAN implementation <https://arxiv.org/abs/1511.06434>
    """
    def __init__(self, in_channels, latent_size=100, n_filters=64, n_pyramid=3):
        super().__init__()

        self.model = nn.Sequential(*[
            # check if this bias=False helps
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            *[PyramidDown(n_filters * 2 ** i) for i in range(n_pyramid)],

            # check if this bias=False helps
            nn.Conv2d(n_filters * 2 ** n_pyramid, latent_size, kernel_size=4, stride=1, padding=0, bias=False),
        ])

    def forward(self, x):
        return self.model(x)


class DCDecoder(nn.Module):
    """
    Deep Convolutional Decoder

    Inspired from DCGAN implementation <https://arxiv.org/abs/1511.06434>
    """
    def __init__(self, out_channels, latent_size=100, n_filters=64, n_pyramid=3):
        super().__init__()

        self.model = nn.Sequential(*[
            nn.ConvTranspose2d(latent_size, n_filters * 2 ** n_pyramid, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 2 ** n_pyramid),
            nn.ReLU(inplace=True),

            *[PyramidUp(n_filters * 2 ** i) for i in range(n_pyramid, 0, -1)],

            # check if this bias=False helps
            nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
        ])

    def forward(self, x):
        return self.model(x)
