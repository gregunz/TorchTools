import warnings
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, List, Tuple, Sequence

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from werkzeug.utils import cached_property

from .util import AggFn, Logger

OptimOrSched = Union[Optimizer, List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]

class Strategy(Logger):
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
        self._log_metrics_cache = dict()

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
    def optim_schedulers(self) -> OptimOrSched:
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
    def modules(self) -> List[nn.Module]:
        return [module for module in self.__dict__.values() if isinstance(module, nn.Module)]

    @cached_property
    def num_tng_batch(self):
        return len(self.tng_data_loader())

    @cached_property
    def num_val_batch(self):
        return len(self.val_data_loader())

    @cached_property
    def num_tst_batch(self):
        return len(self.tst_data_loader())

    @property
    def logger(self) -> SummaryWriter:
        """
        Provides a logger

        Returns:
        """
        if self._logger is None:
            # warnings.warn('Accessing logger but it is not set. Instantiating one with default arguments')
            i = 0
            log_dir = Path(self.log_dir) / f'version_{i}'
            while log_dir.exists():
                i += 1
                log_dir = Path(self.log_dir) / f'version_{i}'
            self._logger = SummaryWriter(log_dir / 'tf')
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

    def log(self, metrics_dict: dict, global_step: int, interval: int = 1) -> None:
        """
        Logs a dictionary of scalars

        Args:
            metrics_dict:
            global_step:
            interval:

        """
        for name, scalar in metrics_dict.items():
            self._log_metrics_cache[name] = self._log_metrics_cache.get(name, []) + [scalar.item()]
        if global_step % interval == 0:
            metrics_dict = {name: torch.tensor(self._log_metrics_cache[name]).mean().item()
                            for name, _ in metrics_dict.items()}
            self._log_metrics_cache = dict()
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

    @staticmethod
    def opt_sched_unpack(opt_sched):
        try:
            opt, sched = opt_sched
        except TypeError:
            opt, sched = opt_sched, []
        if not isinstance(opt, Sequence):
            opt = [opt]
        if not isinstance(sched, Sequence):
            sched = [sched]
        return opt, sched
