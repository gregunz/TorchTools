from typing import Any

from torch import nn

from .pyramid_blocks import PyramidDown, PyramidUp


class DCEncoder(nn.Module):
    """
    Deep Convolutional Encoder

    Inspired from DCGAN implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, latent_channels=100, n_filters=64, n_pyramid=3, first_layer=True, final_layer=True,
                 factor: int = 2, max_channels=None):
        super().__init__()

        layers = []

        if first_layer:
            layers += [nn.Sequential(
                # check if this bias=False helps
                nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )]

        layers += [PyramidDown(n_filters * factor ** i, factor=factor, max_channels=max_channels)
                   for i in range(n_pyramid)]

        if final_layer:
            c = n_filters * factor ** n_pyramid
            if max_channels is not None:
                c = min(c, max_channels)
            layers += [
                # check if this bias=False helps
                nn.Conv2d(in_channels=c, out_channels=latent_channels,
                          kernel_size=4, stride=1, padding=0, bias=False)
            ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return nn.Sequential(*self.layers)(x)

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)


class DCDecoder(nn.Module):
    """
    Deep Convolutional Decoder

    Inspired from DCGAN implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, out_channels, latent_channels=100, n_filters=64, n_pyramid=3, first_layer=True,
                 final_layer=True, factor: int = 2, max_channels=None):
        super().__init__()

        layers = []

        if first_layer:
            c = n_filters * factor ** n_pyramid
            if max_channels is not None:
                c = min(c, max_channels)
            layers += [nn.Sequential(
                nn.ConvTranspose2d(in_channels=latent_channels, out_channels=c,
                                   kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )]

        layers += [PyramidUp(n_filters * factor ** i, factor=factor, max_channels=max_channels)
                   for i in range(n_pyramid, 0, -1)]

        if final_layer:
            layers += [
                # check if this bias=False helps
                nn.ConvTranspose2d(n_filters, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return nn.Sequential(*self.layers)(x)

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)
