import copy
from pathlib import Path
from typing import Tuple, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from .util import int_to_flags
from .. import Strategy, Executor
from ..util import AggFn


class LightningExecutor(Executor):
    """
    An Executor using pytorch-lightning library for executing strategies.

    It handles multiple gpus, checkpointing, early stopping,..
    """

    def __init__(self, tng_dataloader, exp_name, model_dir, gpus: int, ckpt_period: int, val_dataloader=None,
                 tst_dataloader=None, **kwargs):
        super().__init__(
            tng_dataloader=tng_dataloader,
            exp_name=exp_name,
            model_dir=model_dir,
            gpus=int_to_flags(gpus),
            ckpt_period=ckpt_period,
            val_dataloader=val_dataloader,
            tst_dataloader=tst_dataloader,
        )
        self._last_trainer: pl.Trainer = None
        self._kwargs = kwargs

    def train(self, strategy: Strategy, epochs, version=None, early_stop_callback=None, **kwargs) \
            -> Tuple[Strategy, int]:

        module = _LightningModule.load(strategy=strategy, executor=self)
        logger = self._get_logger(strategy=strategy, version=version)
        hparams = _HParams() + self._kwargs + kwargs
        logger.log_hyperparams(hparams)
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

    def test(self, strategy=None, version=None, **kwargs):
        if strategy is not None or version is not None:
            logger = self._get_logger(strategy=strategy, version=version, add_graph=False)
            trainer = self._get_trainer(
                logger=logger,
                epochs=0,
            )
            module = _LightningModule.load(strategy=strategy, executor=self)
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
            save_dir=str(strategy.log_dir).replace(self.exp_name, ''),  # the name is already added to the save_dir
            name=self.exp_name,
            version=version,
        )
        # todo: do this somewhere else
        # logger.experiment.argparse(argparser=self.argz)
        strategy.logger = logger
        self._logger = logger
        if add_graph:
            strategy.add_graph()
        return logger

    def _get_trainer(self, logger, epochs, early_stop=None):
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        gpus = None if len(self.gpus) == 0 else self.gpus
        use_amp = None  # gpus is not None
        d_backend = None if len(self.gpus) <= 1 else 'dp'

        ckpt_path = Path(self.model_dir) / self.exp_name / f'version_{logger.experiment.version}'
        checkpoint_callback = None
        if self.ckpt_period > 0:
            checkpoint_callback = ModelCheckpoint(
                filepath=ckpt_path,
                prefix=f'weights',
                verbose=True,
                period=self.ckpt_period,
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


class _HParams:
    __dict__ = dict()

    def __add__(self, kwargs):
        self.__dict__.update(kwargs)
        return self


class _LightningModule(pl.LightningModule):
    def __init__(self, strategy: Strategy, executor: Executor):
        super().__init__()
        self.strat = strategy
        self.executor = executor

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Do not need to forward in a module')

    def configure_optimizers(self):
        return self.strat.optim_schedulers()

    @pl.data_loader
    def train_dataloader(self):
        return self.executor.tng_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self.executor.val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self.executor.tst_dataloader

    def training_step(self, *args):
        args = _args_step(args)
        outputs = self.strat.tng_step(*args, epoch_idx=self.current_epoch,
                                      num_batches=len(self.executor.tng_dataloader))
        assert 'loss' in outputs, 'training step should return a dict containing the loss'
        other_logs = {k: v for k, v in outputs.items() if k != 'loss'}
        return {
            'loss': outputs['loss'],
            'progress_bar': other_logs,
            'log': {},
        }

    def validation_step(self, *args):
        args = _args_step(args)
        return self.strat.val_step(*args, epoch_idx=self.current_epoch, num_batches=len(self.executor.val_dataloader))

    def validation_end(self, outputs):
        outputs = self.strat.val_agg_outputs(outputs, AggFn(outputs), self.current_epoch)
        return {
            'progress_bar': outputs,
            'log': {},
        }

    def test_step(self, *args):
        args = _args_step(args)
        return self.strat.tst_step(*args, num_batches=len(self.executor.tst_dataloader))

    def test_end(self, outputs):
        outputs = self.strat.tst_agg_outputs(outputs, AggFn(outputs))
        return {
            'progress_bar': outputs,
            'log': {},
        }

    # fix: https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inputs, **kwargs) -> Any:
        return super().__call__(*inputs, **kwargs)

    @staticmethod
    def load(strategy: Strategy, executor: Executor) -> '_LightningModule':
        executor = copy.copy(executor)  # not sure if necessary
        module = _LightningModule(strategy, executor)
        # todo: do better than 'did_delete'!
        if not getattr(module.__class__, 'did_delete', False):
            if executor.val_dataloader is None:
                delattr(module.__class__, 'validation_step')
                delattr(module.__class__, 'validation_end')
            if executor.tst_dataloader is None:
                delattr(module.__class__, 'test_step')
                delattr(module.__class__, 'test_end')
            setattr(module.__class__, 'did_delete', True)

        for i, m in enumerate(strategy.modules):
            setattr(module, f'module_{i:04d}', m)

        return module


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
