from torch import nn, Tensor


class Lambda(nn.Module):
    def __init__(self, lambda_fn, *modules):
        super().__init__()
        self.lambda_fn = lambda_fn
        self.modules = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.lambda_fn(x)
