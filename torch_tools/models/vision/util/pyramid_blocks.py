from torch import nn


class PyramidBlock(nn.Module):
    def __init__(self, conv_builder, in_channels: int, out_channels: int, use_batch_norm: bool = True,
                 activation: nn.Module = nn.ReLU(True)):
        super().__init__()

        block_parts = [conv_builder(in_channels, out_channels, 4, 2, 1, bias=not use_batch_norm)]

        if use_batch_norm:
            block_parts += [nn.BatchNorm2d(out_channels)]
        if activation is not None:
            block_parts += [activation]

        self.seq = nn.Sequential(*block_parts)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.seq(x)

    def __repr__(self):
        return f'{self.__class__.__name__.lower()}-{self.in_channels}-to-{self.out_channels}'


class PyramidDown(PyramidBlock):
    """
    This convolutional block reduces the input size (H, W) by a factor 2.

    Conv2d -> BatchNorm -> LeakyReLU

    Inspired from by DCGAN discriminator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None, use_batch_norm=True,
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        if out_channels is None:
            out_channels = in_channels * 2
        super().__init__(
            conv_builder=nn.Conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )


class PyramidUp(PyramidBlock):
    """
    This convolutional block augments the input size (H, W) by a factor 2.

    ConvTranspose2d -> BatchNorm -> ReLU

    Inspired from by DCGAN generator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None, use_batch_norm=True, activation=nn.ReLU(inplace=True)):
        if out_channels is None:
            out_channels = in_channels // 2
        super().__init__(
            conv_builder=nn.ConvTranspose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
