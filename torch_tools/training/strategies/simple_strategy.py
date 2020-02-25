from abc import ABCMeta
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

from torch import optim, nn

from .. import Strategy


class SimpleStrategy(Strategy, metaclass=ABCMeta):
    """
    A simple (abstract) strategy, practical for simple model training.

    It needs one model. It provides one adam optimizer on model's parameters.
    """

    def __init__(self, net: nn.Module, lr: float, betas: Tuple[float, float], weight_decay: float,
                 log_dir: Union[str, Path], **kwargs):
        super().__init__(log_dir)
        # networks
        self.net = net
        self.lr = lr
        self.betas = betas
        self.wd = weight_decay

    @staticmethod
    def add_argz(parser: ArgumentParser):
        parser.add_argument('--lr', type=float, default=0.0001, help=f'adam learning rate')
        parser.add_argument('--betas', type=float, nargs='+', default=(0.9, 0.999), help=f'betas of adam optimizer')
        parser.add_argument('--weight_decay', type=float, default=0, help=f'adam weight decay')

    def optim_schedulers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.wd)
        return optimizer
