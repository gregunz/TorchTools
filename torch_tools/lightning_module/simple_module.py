from abc import abstractmethod

import torch
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader

from . import BaseModule


class SimpleModule(BaseModule):
    def __init__(self, tng_dataset, val_dataset, net, lr, betas, tng_batch_size, val_batch_size, tst_dataset=None):
        super().__init__()
        self.tng_dataset = tng_dataset
        self.val_dataset = val_dataset
        self.tst_data = tst_dataset
        # networks
        self.net = net
        self.lr = lr
        self.betas = betas
        self.tng_batch_size = tng_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 32

    @staticmethod
    def add_args(parser: HyperOptArgumentParser):
        # opt
        default_lr, default_lr_opt = 0.0001, (0.0001, 0.0005, 0.001)
        parser.opt_list('--lr', type=float, default=default_lr, options=default_lr_opt, tunable=True,
                        help=f'learning rate (default: {default_lr})')

        default_tng_batch_size, default_tng_batch_size_opt = 64, (64, 128, 256, 512)
        parser.opt_list('--tng_batch_size', type=int, default=default_tng_batch_size,
                        options=default_tng_batch_size_opt, tunable=False,
                        help=f'training batch size (default: {default_tng_batch_size})')

        # args
        # todo: fix nargs (HyperOptArgumentParser apparently does not and list, hence we use tuple here but this only
        #  works with default arguments and not with user input)
        default_betas = (0.9, 0.999)
        parser.add_argument('--betas', type=float, nargs='+', default=default_betas,
                            help=f'betas of adam optimizer (default: {default_betas})')

        default_val_batch_size = 64
        parser.add_argument('--val_batch_size', type=int, default=default_val_batch_size,
                            help=f'validation batch size (default: {default_val_batch_size})')

    @abstractmethod
    def loss(self, output, target):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def tng_data_loader(self):
        return DataLoader(self.tng_dataset, batch_size=self.tng_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_data_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=self.num_workers)

    def tst_data_loader(self):
        return DataLoader(self.tst_data, batch_size=1, shuffle=False, num_workers=self.num_workers)
