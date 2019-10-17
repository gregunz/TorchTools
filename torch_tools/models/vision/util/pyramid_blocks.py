from torch import nn


class _PyramidBlock(nn.Module):
    def __init__(self, conv_builder, in_channels: int, out_channels: int, activation: nn.Module = nn.ReLU(True)):
        super().__init__()
        self.seq = nn.Sequential(*[
            conv_builder(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        ])
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.seq(x)

    def __repr__(self):
        return f'{self.__class__.__name__.lower()}-{self.in_channels}->{self.out_channels}'


class PyramidDown(_PyramidBlock):
    """
    This convolutional block reduces the input size (H, W) by a factor 2.

    Conv2d -> BatchNorm -> LeakyReLU

    Inspired from by DCGAN discriminator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels * 2
        super().__init__(nn.Conv2d, in_channels, out_channels, nn.LeakyReLU(negative_slope=0.2, inplace=True))


class PyramidUp(_PyramidBlock):
    """
    This convolutional block augments the input size (H, W) by a factor 2.

    ConvTranspose2d -> BatchNorm -> ReLU

    Inspired from by DCGAN generator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels // 2
        super().__init__(nn.ConvTranspose2d, in_channels, out_channels)  # using default activations: ReLU
