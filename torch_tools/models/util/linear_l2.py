from torch import Tensor
from torch import nn
from torch.nn import functional as F


class LinearL2(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        return super().forward(x)
