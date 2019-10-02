from torch import nn, Tensor


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.sizes = size

    def forward(self, x: Tensor) -> Tensor:
        return x.view((x.size(0),) + self.sizes)