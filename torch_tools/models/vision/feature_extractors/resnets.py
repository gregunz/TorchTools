import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock

from torch_tools.models.util import Lambda


class ResNetFeatures(nn.Module):

    def __init__(self, trained_net, log=False, with_relu=True, in_place=False, rescale_size=None,
                 rescale_mode='nearest'):
        super().__init__()
        self.resnet = trained_net
        self.with_relu = with_relu
        self.in_place = in_place
        self.rescale = None

        self.log = Lambda(lambda x: x)
        if log:
            if not with_relu:
                raise ValueError('cannot compute log(1+x) when there is no ReLU final activation (negative values)')
            self.log = Lambda(lambda x: torch.log(x + 1))

        if self.in_place:
            self.resnet.fc = Lambda(lambda x: x)
            if not self.with_relu:
                self.resnet.layer4[-1] = self.to_no_relu_block(self.resnet.layer4[-1])
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        if rescale_size is not None:
            self.rescale = nn.Upsample(size=rescale_size, mode=rescale_mode)

    def forward(self, x):
        if self.rescale is not None:
            x = self.rescale(x)

        if self.in_place:
            return self.log(self.resnet(x))

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        if self.with_relu:
            x = self.resnet.layer4(x)
        else:
            # ignore last module (because it contains the ReLU)
            x = self.resnet.layer4[:-1](x)
            x = self.to_no_relu_block(self.resnet.layer4[-1])(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.log(x)
        return x

    def to_no_relu_block(self, block):
        if isinstance(block, Bottleneck):  # handling bottleneck block
            return BottleneckNoRelu(block)
        elif isinstance(block, BasicBlock):
            return BasicBlockNoRelu(block)  # handling basic block
        else:
            raise NotImplementedError(f'unknown block {type(block)}')


class BottleneckNoRelu(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        out = self.block.conv3(out)
        out = self.block.bn3(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        # we do not apply the final relu
        # out = self.relu(out)

        return out


class BasicBlockNoRelu(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        # we do not apply the final relu
        # out = self.relu(out)

        return out
