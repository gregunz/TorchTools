from collections import Sequence

import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, *tensor, dim=1):
        super().__init__()
        self.dim = dim
        if len(tensor) == 1 and isinstance(tensor[0], Sequence):
            tensor = tuple(tensor[0])

        self.tensors = tensor

    def forward(self, *args) -> torch.Tensor:
        assert len(args) != 0 or self.tensors is not None, 'concat single tensor is meaningless'
        if len(args) == 1 and isinstance(args, Sequence):
            args = tuple(args[0])
        if self.tensors is not None:
            args += self.tensors
        return torch.cat(args, dim=self.dim)


class ConcatModules(nn.Module):
    def __init__(self, *module, dim=1):
        super().__init__()
        if len(module) == 1 and isinstance(module[0], Sequence):
            module = tuple(module[0])

        self.module_list = nn.ModuleList(module)
        self.concat = Concat(dim=dim)

    def forward(self, x):
        return self.concat([m(x) for m in self.module_list])
