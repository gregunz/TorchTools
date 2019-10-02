from abc import abstractmethod
from typing import Union, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from torch_tools import utils
from .util import AggFn
from ..datasets import BaseDataset


class BaseModule(pl.LightningModule, utils.AddArgs):
    def __init__(self, tng_dataset: BaseDataset, tst_dataset: Dataset = None):
        super().__init__()
        self.tng_dataset = tng_dataset
        self.tst_data = tst_dataset

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):  # TO-CHECK: maybe arguments should be *args
        """

        :param x:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tng_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError

    @abstractmethod
    def tng_step(self, batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """

        :param batch: data from a batch of the dataloader
        :param batch_idx: index of the the batch
        :param optimizer_idx:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """

        :return: [optimizer, ...], [scheduler, ...]
        """

    @abstractmethod
    def val_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError

    @abstractmethod
    def val_step(self, batch, batch_idx: int, optimizer_idx: int, global_step: int) -> dict:
        """

        :param batch:
        :param batch_idx:
        :param optimizer_idx:
        :param global_step:
        :return: dict, example: {
            'gt': y,
            'logits': y_hat,
            'loss': loss_val,
            'acc': val_acc,
        }
        """
        raise NotImplementedError

    @abstractmethod
    def val_agg_outputs(self, outputs: List[dict], agg_fn: AggFn) -> None:
        """
        This is where you have the opportunity to aggregate the outputs
        in order to log any metrics you wish

        :param outputs:
        :param agg_fn:
        :return:
        """
        raise NotImplementedError

    def tst_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        return NotImplemented

    def tst_step(self, batch, batch_idx: int, optimizer_idx: int) -> dict:
        return NotImplemented

    def tst_agg_outputs(self, outputs: List[dict], agg_fn: AggFn) -> None:
        return NotImplemented

    def has_tst_data(self) -> bool:
        return self.tst_data is not None

    @pl.data_loader
    def tng_dataloader(self):
        return self.tng_data_loader()

    def training_step(self, *args):
        args = self.__args_step(args)
        loss = self.tng_step(*args)
        return {
            'loss': loss
        }

    @pl.data_loader
    def val_dataloader(self):
        return self.val_data_loader()

    def validation_step(self, *args):
        args = self.__args_step(args)
        batch_idx = args[1]
        val_step = batch_idx + self.current_epoch * len(self.val_dataloader)
        return self.val_step(*args, global_step=val_step)

    def validation_end(self, outputs):
        self.val_agg_outputs(outputs)
        return {}

    @pl.data_loader
    def test_dataloader(self):
        return self.tst_data_loader()

    def test_step(self, *args):
        args = self.__args_step(args)
        return self.tst_step(*args)

    def test_end(self, outputs):
        self.tst_agg_outputs(outputs, AggFn(outputs))
        return {}

    def add_graph(self, exp):
        try:
            # using writer from tensorboardX because tensorboard graph are not working apparently
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(logdir=exp.log_dir)
            data_loader_iter = iter(self.tng_dataloader)
            x, _ = next(data_loader_iter)
            writer.add_graph(self, x)
        except Exception as e:
            raise Exception("Failed to save model graph: {}".format(e))
        finally:
            writer.close()

    @staticmethod
    def __args_step(args):
        """
        Handling the case with multiple optimizer (by forcing optimizer_idx to be defined)

        :param args:
        :return:
        """
        if len(args) == 2:
            args += (0,)
        assert len(args) == 3
        return args
