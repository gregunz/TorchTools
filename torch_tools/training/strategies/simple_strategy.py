from abc import ABCMeta
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

from torch import optim, nn
from torch.utils.data import DataLoader

from .. import Strategy


class SimpleStrategy(Strategy, metaclass=ABCMeta):
    """
    A simple (abstract) strategy, practical for simple model training.

    It needs one model and (at least) one dataset.

    It provides one adam optimizer for model's parameters, shuffled dataloaders.
    """

    def __init__(self, tng_dataloader: DataLoader, net: nn.Module, lr: float, betas: Tuple[float, float],
                 log_dir: Union[str, Path], val_dataloader: DataLoader = None, tst_dataloader: DataLoader = None,
                 **kwargs):
        super().__init__(log_dir)
        self.tng_dataloader = tng_dataloader
        self.val_dataloader = val_dataloader
        self.tst_dataloader = tst_dataloader
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
        # default_tng_batch_size = 64
        # parser.add_argument('--tng_batch_size', type=int, default=default_tng_batch_size,
        #                     help=f'training batch size (default: {default_tng_batch_size})')
        # default_val_batch_size = 64
        # parser.add_argument('--val_batch_size', type=int, default=default_val_batch_size,
        #                     help=f'validation batch size (default: {default_val_batch_size})')

    def optim_schedulers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def tng_data_loader(self):
        return self.tng_dataloader

    def val_data_loader(self):
        return self.val_dataloader

    def tst_data_loader(self):
        return self.tst_dataloader
