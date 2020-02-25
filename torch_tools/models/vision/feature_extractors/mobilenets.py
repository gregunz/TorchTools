import torch
from torch import nn
from torchvision.models import mobilenet_v2

from torch_tools.models.util import Lambda


class MobileNetFeatures(nn.Module):
    @classmethod
    def load_pretrained(cls, **kwargs):
        return cls(mobilenet_v2(pretrained=True), **kwargs)

    def __init__(self, trained_net, log=False, with_relu=True, in_place=False, rescale_size=None,
                 rescale_mode='nearest'):
        super().__init__()
        self.mobilenet = trained_net
        self.with_relu = with_relu
        self.in_place = in_place
        self.rescale = None

        self.log = Lambda(lambda x: x)
        if log:
            if not with_relu:
                raise ValueError('cannot compute log(1+x) when there is no ReLU final activation (negative values)')
            self.log = Lambda(lambda x: torch.log(x + 1))

        if self.in_place:
            self.mobilenet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.mobilenet.classifier = Lambda(lambda x: x)
            if not self.with_relu:
                self.mobilenet.features[-1][-1] = ConvBNNoReLU(self.mobilenet.features[-1][-1])
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        if rescale_size is not None:
            self.rescale = nn.Upsample(size=rescale_size, mode=rescale_mode)

    def forward(self, x):
        if self.rescale is not None:
            x = self.rescale(x)

        if self.in_place:
            return self.log(self.mobilenet(x))

        if self.with_relu:
            x = self.mobilenet.features(x)
        else:
            x = self.mobilenet.features[:-1](x)
            x = ConvBNNoReLU(self.mobilenet.features[-1])(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.log(x)
        return x


class ConvBNNoReLU(nn.Module):
    def __init__(self, conv_bn_relu):
        super().__init__()
        self.conv_bn_relu = conv_bn_relu

    def forward(self, x):
        x = self.conv_bn_relu[0](x)
        x = self.conv_bn_relu[1](x)
        # ignoring self.conv_bn_relu[2] which is a ReLU layer
        return x
