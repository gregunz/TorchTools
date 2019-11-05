from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from . import strategy as S


class Executor:
    """
    Executes strategies, handles the dataloader, multiple gpus, checkpointing, early stopping...

    Available implementations:
     - `LightningExecutor` (pytorch-lightning backend)
     - `SimpleExecutor` (simple minimal for loop)

    """

    def __init__(self, tng_dataloader: DataLoader, exp_name, model_dir, gpus, ckpt_period, val_dataloader=None,
                 tst_dataloader=None):
        self.tng_dataloader = tng_dataloader
        self.val_dataloader = val_dataloader
        self.tst_dataloader = tst_dataloader
        self.exp_name = exp_name
        self.model_dir = model_dir
        self.gpus = gpus
        self.ckpt_period = ckpt_period

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
        default_epochs = 10
        parser.add_argument('--epochs', type=int, default=default_epochs,
                            help=f'number of epochs to train (default: {default_epochs})')

        default_log_dir = Path('/data/logs/')
        parser.add_argument("--log_dir", type=str, default=default_log_dir,
                            help=f'directory for log outputs (tensorboard and more) (default: {default_log_dir})')

        default_model_dir = Path('/data/models/')
        parser.add_argument('--model_dir', type=str, default=default_model_dir,
                            help=f'directory for model weights (default: {default_model_dir})')

        default_version = None  # when None, it creates a new one
        parser.add_argument('--version', type=int, default=default_version,
                            help=f'specify version continue its training (default: {default_version})')

        default_seed = None  # this means it will be set by the clock (random)
        parser.add_argument('--manual_seed', type=int, default=default_seed,
                            help=f'set the seed manually for more reproducibility (default: {default_seed})')
        # represents which gpu is used in binary representation (e.g, 0 = cpu, 5 = 1010 = gpu0 and gpu2)
        default_gpus = 1
        parser.add_argument('--gpus', type=int, default=default_gpus,
                            help=f'which cuda device is used in binary representation '
                                 f'(i.e. 5 = 0101 = cuda:0 and cuda:2) (default: {default_gpus})')
        default_ckpt_period = 5
        parser.add_argument('--ckpt_period', type=int, default=default_ckpt_period,
                            help=f'save model every ckpt_period epoch')
