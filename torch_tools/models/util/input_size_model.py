from typing import Union, Tuple, Any

from torch import nn

InputSize = Union[Tuple[int, int], Tuple[int, int, int]]


class FISModel(nn.Module):
    """
    Fixed Input Size Model

    Class of models which only works with fixed input size.
    """

    def __init__(self, input_size: InputSize):
        super().__init__()
        if len(input_size) == 2:
            input_size = (1,) + input_size
        self.input_channels, self.input_height, self.input_width = input_size
        self.input_size = input_size

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)
