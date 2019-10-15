from abc import abstractmethod
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
from torch import nn

from .. import executor as E
from .. import strategy as S
from ..util import AggFn, int_to_flags


class LightningExecutor(E.Executor):
    def __init__(self, exp_name, log_dir, model_dir, gpus, **kwargs):
        super().__init__(exp_name, log_dir, model_dir, gpus)
        self._trainers = _TrainerList()

    def train(self, strategy: S.Strategy, epochs, version=None):
        logger = TestTubeLogger(
            save_dir=self.log_dir,
            name=self.exp_name,
            version=version,
        )
        # todo: do this somewhere else
        # logger.experiment.argparse(argparser=self.argz)
        strategy.logger = logger.experiment
        strategy.add_graph()

        module = _LightningModule(strategy)
        for k, v in strategy.__dict__.items():
            if isinstance(v, nn.Module):
                setattr(module, k, v)

        ckpt_path = Path(self.model_dir) / self.exp_name / f'version_{logger.experiment.version}'
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_path,
            prefix=f'weights',
            verbose=True,
            # save_best_only=False,
            # monitor='val_loss',
            # mode='min',
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='auto'
        )

        trainer = pl.Trainer(
            logger=logger,
            max_nb_epochs=epochs,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            gpus=int_to_flags(self.gpus),
            nb_sanity_val_steps=0,
        )
        self._trainers[logger.experiment.version] = trainer
        trainer.fit(module)

    def test(self, strategy, version=None):
        module = _LightningModule(strategy)
        self._trainers[version].test(module)

    # @staticmethod
    # def add_argz(parser: ArgumentParser):
    #    super().add_argz(parser)

    # default_log_interval = 100
    # parser.add_argument('--log_interval', type=int, default=default_log_interval,
    #                    help=f'number of batches aggregated before logging the loss (default: {default_log_interval})')


class _LightningModule(pl.LightningModule):
    def __init__(self, strategy: S.Strategy):
        super().__init__()
        self.strat = strategy

    @abstractmethod
    def forward(self, x):
        # return self.net(x)
        raise NotImplementedError('No need of this? Right?')

    def configure_optimizers(self):
        return self.strat.optimizers()

    @pl.data_loader
    def train_dataloader(self):
        return self.strat.tng_data_loader()

    @pl.data_loader
    def val_dataloader(self):
        if self.strat.has_val():
            return self.strat.val_data_loader()
        return None

    @pl.data_loader
    def test_dataloader(self):
        if self.strat.has_tst():
            return self.strat.tst_data_loader()
        return None

    def training_step(self, *args):
        args = _args_step(args)
        return self.strat.tng_step(*args, epoch_idx=self.current_epoch)

    def validation_step(self, *args):
        args = _args_step(args)
        return self.strat.val_step(*args, epoch_idx=self.current_epoch)

    def validation_end(self, outputs):
        print(outputs)
        self.strat.val_agg_outputs(outputs, AggFn(outputs), self.current_epoch)
        return {}

    def test_step(self, *args):
        args = _args_step(args)
        return self.tst_step(*args)

    def test_end(self, outputs):
        self.tst_agg_outputs(outputs, AggFn(outputs))
        return {}


class _TrainerList:
    def __init__(self):
        self.trainers = dict()

    def __getitem__(self, version):
        version = self._check_version(version)
        return self.trainers[version]

    def __setitem__(self, version, trainer):
        version = self._check_version(version)
        self.trainers[version] = trainer

    def _check_version(self, version):
        if version is None:
            version = max(self.trainers.keys())
        else:
            version = int(version)
        assert version >= 0
        return version


def _args_step(args):
    """
    Handling the case with multiple optimizer (by forcing optimizer_idx to be defined)

    :param args:
    :return:
    """
    if len(args) == 2:
        args += (0,)
    assert len(args) == 3
    return args
