import os
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
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
        self._last_trainer: pl.Trainer = None

    def train(self, strategy: Strategy, epochs, version=None, early_stop_callback=None, **kwargs) \
            -> Tuple[Strategy, int]:

        module = load_module(strategy)
        logger = self._get_logger(strategy=strategy, version=version)
        version = logger.experiment.version

        trainer = self._get_trainer(
            logger=logger,
            epochs=epochs,
            early_stop=early_stop_callback,
        )
        self._last_trainer = trainer

        try:
            trainer.fit(module)
        except KeyboardInterrupt:
            print(f'\nTraining manually stopped at epoch {module.current_epoch}...')
            print(f'Restart anytime from here using version={version}')
        return strategy, version

    def load(self, strategy, version, **kwargs):
        ckpt_dir = f'{self.model_dir}/{self.exp_name}/version_{version}'
        prefix = 'weights_ckpt_epoch_'
        suffix = '.ckpt'
        last_epoch = max([int(p[len(prefix):-len(suffix)]) for p in os.listdir(ckpt_dir) if p.startswith(prefix)])
        module = load_module(strategy=strategy)
        module.load_state_dict(torch.load(f'{ckpt_dir}/{prefix}{last_epoch}{suffix}')['state_dict'])

    def test(self, strategy=None, version=None, **kwargs):
        if strategy is not None or version is not None:
            logger = self._get_logger(strategy=strategy, version=version, add_graph=False)
            trainer = self._get_trainer(
                logger=logger,
                epochs=0,
            )
            module = load_module(strategy=strategy)
        else:
            if self._last_trainer is not None:
                trainer = self._last_trainer
                module = None
            else:
                raise ValueError('Need to call train or load before or to specify the version to load!')

        try:
            trainer.test(module)
        except KeyboardInterrupt:
            print(f'\nTesting manually stopped...')

    def _get_logger(self, strategy: Strategy, version: int = None, add_graph=True):
        logger = TestTubeLogger(
            save_dir=strategy.log_dir,
            name=self.exp_name,
            version=version,
        )
        # todo: do this somewhere else
        # logger.experiment.argparse(argparser=self.argz)
        strategy.logger = logger
        if add_graph:
            strategy.add_graph()
        return logger

    def _get_trainer(self, logger, epochs, early_stop=None):
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        gpus = None if len(self.gpus) == 0 else self.gpus
        use_amp = gpus is not None
        d_backend = None if len(self.gpus) <= 1 else 'dp'

        ckpt_path = Path(self.model_dir) / self.exp_name / f'version_{logger.experiment.version}'
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_path,
            prefix=f'weights',
            verbose=True,
            # save_best_only=False,
            # monitor='val_loss',
            # mode='min',
        )

        return pl.Trainer(
            logger=logger,
            max_nb_epochs=epochs,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop,
            gpus=gpus,
            use_amp=use_amp,
            amp_level='O1',
            nb_sanity_val_steps=0,
            distributed_backend=d_backend,
        )


def load_module(strategy: Strategy) -> pl.LightningModule:
    module = _LightningModule(strategy)
    if strategy.val_data_loader() is None:
        delattr(module.__class__, 'validation_step')
        delattr(module.__class__, 'validation_end')
    if strategy.tst_data_loader() is None:
        delattr(module.__class__, 'test_step')
        delattr(module.__class__, 'test_end')
    for k, v in strategy.__dict__.items():
        if isinstance(v, nn.Module):
            setattr(module, k, v)
    return module


class _LightningModule(pl.LightningModule):
    def __init__(self, strategy: Strategy):
        super().__init__()
        self.strat = strategy

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Do not need forward in a module')

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
        other_logs = {k: v for k, v in outputs.items() if k != 'loss'}
        return {
            'loss': outputs['loss'],
            'progress_bar': other_logs,
            'log': {},
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
