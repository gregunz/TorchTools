from typing import Union, Tuple

InputSize = Union[Tuple[int, int], Tuple[int, int, int]]


class FixedInputSizeModel:
    def __init__(self, input_size: InputSize):
        super().__init__()
        if len(input_size) == 2:
            input_size = (1,) + input_size
        self.input_channels, self.input_height, self.input_width = input_size
