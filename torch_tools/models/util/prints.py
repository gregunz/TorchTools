from torch import nn, Tensor


class Print(nn.Module):
    def __init__(self, fn=lambda x: x):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        print('[DEBUG]')
        print(f'type = {type(x)}')
        print(f'fn(x) = {self.fn(x)}')
        return x
