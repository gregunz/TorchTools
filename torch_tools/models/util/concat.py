import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, *tensor, dim=1):
        super().__init__()
        self.dim = dim
        self.tensors = tuple(tensor)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            assert self.tensors is not None, 'concat single tensor is meaningless'
            x = (x,)
        if self.tensors is not None:
            x += self.tensors
        return torch.cat(x, dim=self.dim)
