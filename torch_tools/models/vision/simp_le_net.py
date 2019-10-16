from torch import nn

from torch_tools import utils
from ..util import Flatten, FISModel


class SimpLeNet(FISModel):
    """
    A (very) simple network inspired by LeNet architecture
    """

    def __init__(self, input_size, n_classes):
        super().__init__(input_size)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(20, 40, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        h_out, w_out = utils.cnn_out_size(self.input_height, self.input_width, self.cnn)
        self.fc = nn.Sequential(
            nn.Linear(h_out * w_out * 40, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(100, n_classes),
        )

    def forward(self, x):
        return nn.Sequential(
            self.cnn,
            Flatten(),
            self.fc,
        )(x)
