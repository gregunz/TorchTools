import warnings
from typing import Any

from torch import nn


class PyramidBlock(nn.Module):
    def __init__(self, conv_op, in_channels: int, out_channels: int, use_batch_norm: bool = True,
                 activation: nn.Module = nn.ReLU(inplace=True), max_channels=None):
        super().__init__()
        if max_channels is not None:
            in_channels = min(in_channels, max_channels)
            out_channels = min(out_channels, max_channels)

        block_parts = [conv_op(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1, bias=not use_batch_norm)]

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
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)


class PyramidDown(PyramidBlock):
    """
    This convolutional block reduces the input size (H, W) by a factor 2.

    Conv2d -> BatchNorm -> LeakyReLU

    Inspired from by DCGAN discriminator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None, use_batch_norm=True,
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True), factor=2, max_channels=None):
        if out_channels is None:
            out_channels = in_channels * factor
        elif factor != 2:
            warnings.warn('when out_channels is defined, factor is ignored')

        super().__init__(
            conv_op=nn.Conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            max_channels=max_channels,
        )


class PyramidUp(PyramidBlock):
    """
    This convolutional block augments the input size (H, W) by a factor 2.

    ConvTranspose2d -> BatchNorm -> ReLU

    Inspired from by DCGAN generator implementation <https://arxiv.org/abs/1511.06434>
    """

    def __init__(self, in_channels, out_channels=None, use_batch_norm=True, activation=nn.ReLU(inplace=True), factor=2,
                 max_channels=None):
        if out_channels is None:
            out_channels = in_channels // factor
        elif factor != 2:
            warnings.warn('when out_channels is defined, factor is ignored')

        super().__init__(
            conv_op=nn.ConvTranspose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
            max_channels=max_channels,
        )
