from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from torch import nn

from .util import int_to_flags
from .. import Strategy, Executor
from ..util import AggFn


class LightningExecutor(Executor):
    """
    An Executor using pytorch-lightning library for executing strategies.

    It handle multiple gpus, checkpointing, early stopping,..

    Args:
        exp_name (str): name of the experience
        model_dir (str): path to model weights directory
        gpus (list): list of cuda gpus, empty list for cpu.
    """

    def __init__(self, exp_name, model_dir, gpus, **kwargs):
        super().__init__(exp_name, model_dir, int_to_flags(gpus))
        self._trainers = _TrainerList()

    def train(self, strategy: Strategy, epochs, version=None, early_stop_callback=None, **kwargs):
        logger = TestTubeLogger(
            save_dir=strategy.log_dir,
            name=self.exp_name,
            version=version,
        )
        version = logger.experiment.version
        # todo: do this somewhere else
        # logger.experiment.argparse(argparser=self.argz)
        strategy.logger = logger
        strategy.add_graph()

        module = load_module(strategy)
        for k, v in strategy.__dict__.items():
            if isinstance(v, nn.Module):
                setattr(module, k, v)

        ckpt_path = Path(self.model_dir) / self.exp_name / f'version_{version}'
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_path,
            prefix=f'weights',
            verbose=True,
            # save_best_only=False,
            # monitor='val_loss',
            # mode='min',
        )
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        gpus = None if len(self.gpus) == 0 else self.gpus
        use_amp = False  # gpus is not None
        d_backend = None if len(self.gpus) <= 1 else 'dp'
        trainer = pl.Trainer(
            logger=logger,
            max_nb_epochs=epochs,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            gpus=gpus,
            use_amp=use_amp,
            amp_level='O1',
            nb_sanity_val_steps=0,
            distributed_backend=d_backend,
        )
        self._trainers[version] = trainer
        trainer.fit(module)

    def test(self, strategy, version=None):
        module = load_module(strategy)
        self._trainers[version].test(module)

    # @staticmethod
    # def add_argz(parser: ArgumentParser):
    #    super().add_argz(parser)

    # default_log_interval = 100
    # parser.add_argument('--log_interval', type=int, default=default_log_interval,
    #                    help=f'number of batches aggregated before logging the loss (default: {default_log_interval})')


def load_module(strategy: Strategy) -> pl.LightningModule:
    module = _LightningModule(strategy)
    if strategy.val_data_loader() is None:
        delattr(module.__class__, 'validation_step')
        delattr(module.__class__, 'validation_end')
    if strategy.tst_data_loader() is None:
        delattr(module.__class__, 'test_step')
        delattr(module.__class__, 'test_end')
    return module


class _LightningModule(pl.LightningModule):
    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strat = strategy

    def configure_optimizers(self):
        return self.strat.optim_schedulers()

    @pl.data_loader
    def train_dataloader(self):
        return self.strat.tng_data_loader()

    @pl.data_loader
    def val_dataloader(self):
        return self.strat.val_data_loader()

    @pl.data_loader
    def test_dataloader(self):
        return self.strat.tst_data_loader()

    def training_step(self, *args):
        args = _args_step(args)
        outputs = self.strat.tng_step(*args, epoch_idx=self.current_epoch)
        assert 'loss' in outputs, 'training step should return a dict containing the loss'
        progress_bar_logs = {k: v for k, v in outputs.items() if k != 'loss'}
        return {
            'loss': outputs['loss'],
            'progress_bar': progress_bar_logs,
            'log': {
                # 'training/loss': outputs['loss']
            },
        }

    def validation_step(self, *args):
        args = _args_step(args)
        return self.strat.val_step(*args, epoch_idx=self.current_epoch)

    def validation_end(self, outputs):
        outputs = self.strat.val_agg_outputs(outputs, AggFn(outputs), self.current_epoch)
        return {
            'progress_bar': outputs,
            'log': {},
        }

    def test_step(self, *args):
        args = _args_step(args)
        return self.strat.tst_step(*args)

    def test_end(self, outputs):
        outputs = self.strat.tst_agg_outputs(outputs, AggFn(outputs))
        return {
            'progress_bar': outputs,
            'log': {},
        }


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
