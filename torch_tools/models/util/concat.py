import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, *tensor, dim=1):
        super().__init__()
        self.dim = dim
        self.tensors = tuple(tensor)

    def forward(self, *args) -> torch.Tensor:
        assert len(args) != 0 or self.tensors is not None, 'concat single tensor is meaningless'
        if self.tensors is not None:
            args += self.tensors
        return torch.cat(args, dim=self.dim)
