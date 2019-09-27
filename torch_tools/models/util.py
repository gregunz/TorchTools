from torch import nn, Tensor


class Print(nn.Module):
    def __init__(self, fn=lambda x: x):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        print('[DEBUG]')
        print(f'type = {type(x)}')
        print(f'x = {self.fn(x)}')
        return x


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.sizes = size

    def forward(self, x: Tensor) -> Tensor:
        return x.view((x.size(0),) + self.sizes)


"""
from typing import List, Any, Union, Tuple

InputSize = Union[Tuple[int, int], Tuple[int, int, int]]

class Module2d(nn.Module):
    ""
    Helps keeping track of sizes of images.

    ""
    def __init__(self, input_size: InputSize):
        super().__init__()
        if len(input_size) == 2:
            input_size = (1,) + input_size
        self.n_channels, self.h, self.w = input_size
"""