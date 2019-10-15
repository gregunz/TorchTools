import warnings
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Union, List

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .util import AggFn


class Strategy:
    def __init__(self, logger: SummaryWriter = None):
        # self.name = name
        self._logger = logger

    @abstractmethod
    def tng_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError

    #  @abstractmethod
    def val_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        pass  # raise NotImplementedError

    # @abstractmethod
    def tst_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        pass  # raise NotImplementedError

    @abstractmethod
    def optimizers(self):
        """

        Returns: [optimizer, ...], [scheduler, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def tng_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        """

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

        Args:
            batch:
            batch_idx:
            optimizer_idx:
            epoch_idx:

        Returns (dict): {
            'loss': val_loss,
            'acc': val_acc,
            'gt': y,
            'logits': y_hat,
        }
        """
        pass  # raise NotImplementedError

    # @abstractmethod
    def val_agg_outputs(self, outputs: List[dict], agg_fn: AggFn, epoch_idx: int) -> None:
        """
        This is where you have the opportunity to aggregate the outputs
        in order to log any metrics you wish

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
    def tst_agg_outputs(self, outputs: List[dict], agg_fn: AggFn) -> None:
        """

        Args:
            outputs:
            agg_fn:

        Returns:

        """
        pass  # raise NotImplementedError

    def has_val(self) -> bool:
        """
        Does this trainer contains a validation set? Yes=True, No=False
        Returns: bool

        """
        try:
            self.val_data_loader()
            return True
        except NotImplementedError:
            return False

    def has_tst(self) -> bool:
        """
        Does this trainer contains a test set? Yes=True, No=False
        Returns: bool

        """
        try:
            self.tst_data_loader()
            return True
        except NotImplementedError:
            return False

    @staticmethod
    def add_argz(parser: ArgumentParser):
        pass

    @property
    def logger(self) -> SummaryWriter:
        if self._logger is None:
            warnings.warn('Accessing logger but it is not set. Instantiating one with default arguments')
            self._logger = SummaryWriter()
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def log(self, metrics_dict: dict, global_step: int):
        for k, v in metrics_dict.items():
            self.logger.add_scalar(tag=k, scalar_value=v, global_step=global_step)

    def add_graph(self):
        pass

    def _add_graph(self, model):
        try:
            x, _ = next(iter(self.tng_data_loader()))
            self.logger.add_graph(model, x)
        except Exception as e:
            raise Exception("Failed to save model graph: {}".format(e))
