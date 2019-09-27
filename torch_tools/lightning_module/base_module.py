from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch

from ..datasets import BaseDataset


class BaseModule(pl.LightningModule, ABC):

    def __init__(self, dataset: BaseDataset, val_percentage: float):
        super().__init__()
        self.dataset = dataset
        self.tng_data, self.val_data, self.tst_data = dataset.split(val_percentage)

    @abstractmethod
    def _name(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch):  # TO-CHECK: maybe arguments should be different
        """

        :param batch:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tng_data_loader(self):
        raise NotImplementedError

    @abstractmethod
    def tng_step(self, batch, batch_idx, optimizer_idx):
        """ this method must return the loss that will be optimized (minimized) in a dict:
        return {
            'loss': loss, # required
            'prog': {'tng_loss': loss, 'batch_nb': batch_nb} # optional
        }

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
    def val_data_loader(self):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, batch, batch_idx, optimizer_idx, global_step):
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
    def val_agg_outputs(self, outputs):
        """
        This is where you have the opportunity to aggregate the outputs
        in order to log any metrics you wish

        :param outputs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tst_data_loader(self):
        raise NotImplementedError

    @abstractmethod
    def tst_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError

    @abstractmethod
    def tst_agg_outputs(self, outputs):
        raise NotImplementedError

    @property
    def name(self):
        return self._name()

    def infos(self):
        return {
            'dataset': self.dataset.name,
            'tng_batch_size': len(self.tng_data),
            'val_batch_size': len(self.val_data),
            'tst_batch_size': len(self.tst_data),
        }

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
        self.tst_agg_outputs(outputs)
        return {}

    def add_graph(self, exp):
        try:
            from tensorboardX import SummaryWriter
            # using writer from tensorboardX because tensorboard graph are not working apparently
            writer = SummaryWriter(logdir=exp.log_dir)
            data_loader_iter = iter(self.tng_dataloader)
            x, _ = next(data_loader_iter)
            writer.add_graph(self, x)
        except Exception as e:
            raise Exception("Failed to save model graph: {}".format(e))
        finally:
            writer.close()

    # not static for inheritance convenience
    # @staticmethod
    def stack_outputs(self, outputs):
        return lambda metric_name: torch.stack([x[metric_name] for x in outputs])

    @staticmethod
    def __args_step(args):
        if len(args) == 2:
            args += (0,)
        assert len(args) == 3
        return args
