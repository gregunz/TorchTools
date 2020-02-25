from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

from . import callback as C
from . import strategy as S


class Executor:
    """
    Executes strategies, handles the dataloader, multiple gpus, checkpointing, early stopping...

    Available implementations:
     - `LightningExecutor` (pytorch-lightning backend)
     - `SimpleExecutor` (simple minimal for loop)

    """

    def __init__(self, tng_dataloader: DataLoader, exp_name, gpus, val_dataloader=None,
                 tst_dataloader=None, callbacks: List[C.Callback] = None):
        self.tng_dataloader = tng_dataloader
        self.val_dataloader = val_dataloader
        self.tst_dataloader = tst_dataloader
        self.exp_name = exp_name
        self.gpus = gpus
        self.callbacks = callbacks
        if self.callbacks is None:
            self.callbacks: List[C.Callback] = []
        self.version = None

    @abstractmethod
    def train(self, strategy: S.Strategy, epochs: int, version=None):
        """
        Executes training procedure given a `Strategy` and a given number of epochs.

        Args:
            strategy:
            epochs:
            version:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def test(self, strategy: S.Strategy, version=None):
        """
        Executes testing procedure given a `Strategy`.

        Args:
            strategy:
            version:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def train_test(self, strategy: S.Strategy, epochs: int, version=None):
        """
        Executes testing procedure given a `Strategy`.

        Args:
            strategy:
            epochs:
            version:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def add_argz(parser: ArgumentParser):
        parser.add_argument('--epochs', type=int, default=10, help=f'number of epochs to train')
        parser.add_argument("--log_dir", type=str, default=Path('/data/logs/'),
                            help=f'directory for log outputs (tensorboard and more)')
        parser.add_argument('--model_dir', type=str, default=Path('/data/models/'), help=f'directory for model weights')
        parser.add_argument('--version', type=int, default=None, help=f'specify version continue its training')
        # this means the seed will be set by the clock (random)
        parser.add_argument('--manual_seed', type=int, default=None,
                            help=f'set the seed manually for more reproducibility')
        # represents which gpu is used in binary representation (e.g, 0 = cpu, 5 = 1010 = gpu0 and gpu2)
        parser.add_argument('--gpus', type=int, default=1, help=f'which cuda device is used in binary representation '
                                                                f'(i.e. 5 = 1010 = cuda:0 and cuda:2)')
        parser.add_argument('--n_best_or_period', type=int, default=1,
                            help=f'save model every x epoch or keep the best x model weights if a metric name is given')
        parser.add_argument('--metric_name', type=str, default=None, help=f'metric name to use for checkpointing')
