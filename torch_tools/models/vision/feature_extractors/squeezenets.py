import torch
from torch import nn

from torch_tools.models.util import Lambda


class SqueezeNetFeatures(nn.Module):

    def __init__(self, trained_net, log=False, with_relu=True, in_place=False, rescale_size=None,
                 rescale_mode='nearest'):
        super().__init__()
        self.squeezenet = trained_net
        self.with_relu = with_relu
        self.in_place = in_place
        self.rescale = None

        self.log = Lambda(lambda x: x)
        if log:
            if not with_relu:
                raise ValueError('cannot compute log(1+x) when there is no ReLU final activation (negative values)')
            self.log = Lambda(lambda x: torch.log(x + 1))

        if self.in_place:
            self.squeezenet.classifier = nn.AdaptiveAvgPool2d((1, 1))
            if not self.with_relu:
                self.squeezenet.features[-1] = FireNoRelu(self.squeezenet.features[-1])
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        if rescale_size is not None:
            self.rescale = nn.Upsample(size=rescale_size, mode=rescale_mode)

    def forward(self, x):
        if self.rescale is not None:
            x = self.rescale(x)

        if self.in_place:
            return self.log(self.squeezenet(x))

        if self.with_relu:
            x = self.squeezenet.features(x)
        else:
            x = self.squeezenet.features[:-1](x)
            x = FireNoRelu(self.squeezenet.features[-1])(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.log(x)
        return x


class FireNoRelu(nn.Module):

    def __init__(self, fire_module):
        super().__init__()
        self.fire = fire_module

    def forward(self, x):
        x = self.fire.squeeze_activation(self.fire.squeeze(x))
        return torch.cat([
            self.fire.expand1x1(x),
            self.fire.expand3x3(x),
        ], 1)
