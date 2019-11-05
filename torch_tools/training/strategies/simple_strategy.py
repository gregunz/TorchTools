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

    def __init__(self, net: nn.Module, lr: float, betas: Tuple[float, float], log_dir: Union[str, Path], **kwargs):
        super().__init__(log_dir)
        # networks
        self.net = net
        self.lr = lr
        self.betas = betas

    @staticmethod
    def add_argz(parser: ArgumentParser):
        default_lr = 0.0001
        parser.add_argument('--lr', type=float, default=default_lr, help=f'learning rate (default: {default_lr})')
        default_betas = (0.9, 0.999)
        parser.add_argument('--betas', type=float, nargs='+', default=default_betas,
                            help=f'betas of adam optimizer (default: {default_betas})')

    def optim_schedulers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        return optimizer
