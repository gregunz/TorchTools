import warnings
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Union, List, Tuple

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .util import AggFn


class Strategy:
    """
    A training Strategy describes the meaningful parts of a typical training loop.
    It also provides an optional logger.

    Args:
        log_dir (str): path to the logs directory
    """

    def __init__(self, log_dir):
        # self.name = name
        self.log_dir = log_dir
        self._logger = None

    @abstractmethod
    def tng_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Create the `DataLoader` for the training steps

        Returns (DataLoader):

        """
        raise NotImplementedError

    #  @abstractmethod
    def val_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        [Optional] Create the `DataLoader` for the validation steps

        Returns (DataLoader):

        """
        pass  # raise NotImplementedError

    # @abstractmethod
    def tst_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        [Optional] Create the `DataLoader` for the testing steps

        Returns (DataLoader):

        """
        pass  # raise NotImplementedError

    @abstractmethod
    def optim_schedulers(self) -> Union[Optimizer, List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Create the optimizers and schedulers

        Returns: [optimizer, ...], [scheduler, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def tng_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        """
        Describe the training step. It should return a dict with at least the loss.

        Args:
            batch: data from a batch of the dataloader
            batch_idx: index of the the batch
            optimizer_idx:
            epoch_idx:

        Returns (dict): it must at least contains the loss: {
            'loss': tng_loss,
            'acc': tng_acc,
        }
        """
        raise NotImplementedError

    # @abstractmethod
    def val_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        """
        Describe the validation step. It should return a dict with at least the loss.
        The dicts will be aggregated over steps and provided as list to `val_agg_outputs`.
        Logging here might cause performance issue if a step is quickly processed.

        Args:
            batch:
            batch_idx:
            optimizer_idx:
            epoch_idx:

        Returns (dict): for example: {
            'loss': val_loss,
            'acc': val_acc,
            'gt': y,
            'logits': y_hat,
        }
        """
        pass  # raise NotImplementedError

    # @abstractmethod
    def val_agg_outputs(self, outputs: List[dict], agg_fn: AggFn, epoch_idx: int) -> dict:
        """
        This is where you have the opportunity to aggregate the outputs of the validation steps
        and log any metrics you wish.

        Args:
            outputs:
            agg_fn:
            epoch_idx:

        Returns:

        """
        pass  # raise NotImplementedError

    # @abstractmethod
    def tst_step(self, batch, batch_idx: int, optimizer_idx: int) -> dict:
        """
        Describe the testing step. It should return a dict with at least the loss.
        The dicts will be aggregated over steps and provided as list to `tst_agg_outputs`.

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns (dict): {
            'loss': test_loss,
            'acc': test_acc,
            'gt': y,
            'logits': y_hat,
        }
        """
        pass  # raise NotImplementedError

    # @abstractmethod
    def tst_agg_outputs(self, outputs: List[dict], agg_fn: AggFn) -> dict:
        """
        This is where you have the opportunity to aggregate the outputs of the testing steps
        and log any metrics you wish.


        Args:
            outputs:
            agg_fn:

        Returns:

        """
        pass  # raise NotImplementedError

    def add_graph(self) -> None:
        """
        [Optional] Log model(s) graph to tensorboard

        One can use `_add_graph` helper method
        """
        pass

    @staticmethod
    def add_argz(parser: ArgumentParser) -> None:
        pass

    @property
    def logger(self) -> SummaryWriter:
        """
        Provides a logger

        Returns:
        """
        if self._logger is None:
            # warnings.warn('Accessing logger but it is not set. Instantiating one with default arguments')
            self._logger = SummaryWriter(self.log_dir)
        else:
            try:
                from pytorch_lightning.logging import TestTubeLogger
                if isinstance(self._logger, TestTubeLogger):
                    return self._logger.experiment
            except ImportError as e:
                pass
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def log(self, metrics_dict: dict, global_step: int) -> None:
        """
        Logs a dictionary of scalars

        Args:
            metrics_dict:
            global_step:

        """
        try:
            from pytorch_lightning.logging import TestTubeLogger
            if isinstance(self._logger, TestTubeLogger):
                self._logger.log_metrics(metrics_dict, step_num=global_step)
                return
        except ImportError:
            pass
        for k, v in metrics_dict.items():
            self.logger.add_scalar(tag=k, scalar_value=v, global_step=global_step)

    def _add_graph(self, model) -> None:
        try:
            x, _ = next(iter(self.tng_data_loader()))
            self.logger.add_graph(model, x)
        except Exception as e:
            warnings.warn("Failed to save model graph: {}".format(e))
