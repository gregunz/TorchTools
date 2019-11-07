import warnings
from collections import Sequence

import torch
from tqdm.auto import tqdm

from torch_tools.training.executors.util import int_to_flags
from torch_tools.training.util import AggFn
from .. import Strategy, Executor


class SimpleExecutor(Executor):
    def __init__(self, tng_dataloader, exp_name, model_dir, gpus: int, ckpt_period: int, val_dataloader=None,
                 tst_dataloader=None, **kwargs):
        super().__init__(
            tng_dataloader=tng_dataloader,
            val_dataloader=val_dataloader,
            tst_dataloader=tst_dataloader,
            exp_name=exp_name,
            model_dir=model_dir,
            gpus=int_to_flags(gpus) if isinstance(gpus, int) else gpus,
            ckpt_period=ckpt_period,
        )
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        self.device = torch.device('cpu' if len(self.gpus) == 0 else f'cuda:{self.gpus[0]}')
        self.mode = None

    def train(self, strategy: Strategy, epochs: int, version=None, **kwargs):
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._train(strategy=strategy, epochs=epochs, version=version)

    def test(self, strategy: Strategy, version=None, **kwargs):
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._test(strategy=strategy, version=version)

    def train_test(self, strategy: Strategy, epochs: int, version=None, **kwargs):
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._train(strategy=strategy, epochs=epochs, version=version)
        self._test(strategy=strategy, version=version)

    def _init(self, strategy: Strategy, version: int, hparams):
        strategy.set_default_logger(exp_name=self.exp_name, version=version)
        strategy.log_hyperparams(hparams)
        self._strat_to_device(strategy)

    def _train(self, strategy: Strategy, epochs: int, version: int):
        try:
            do_validation = self.val_dataloader is not None
            optimizers, schedulers = strategy.opt_sched_unpack(strategy.optim_schedulers())

            for epoch_idx in tqdm(list(range(epochs)), desc=f'Train Epochs'):

                # TRAINING #
                self._set_train_mode(strategy)  # set model.train()
                for batch_idx, batch in self._batch_iter(self.tng_dataloader):
                    for optimizer_idx, optimizer in enumerate(optimizers):
                        optimizer.zero_grad()
                        output = strategy.tng_step(
                            batch=batch,
                            batch_idx=batch_idx,
                            optimizer_idx=optimizer_idx,
                            epoch_idx=epoch_idx,
                            num_batches=len(self.tng_dataloader),
                        )
                        loss = output['loss']
                        loss.backward()
                        optimizer.step()

                # VALIDATING #
                if do_validation:
                    with torch.no_grad():
                        self._set_eval_mode(strategy)  # set model.eval()
                        outputs = []
                        for batch_idx, batch in self._batch_iter(self.val_dataloader):
                            for optimizer_idx, optimizer in enumerate(optimizers):
                                output = strategy.tng_step(
                                    batch=batch,
                                    batch_idx=batch_idx,
                                    optimizer_idx=optimizer_idx,
                                    epoch_idx=epoch_idx,
                                    num_batches=len(self.val_dataloader),
                                )
                                outputs.append(output)
                        strategy.val_agg_outputs(outputs, AggFn(outputs), epoch_idx)

                # SCHEDULERS #
                for sched in schedulers:
                    sched.step()

                # LOGGING #
                strategy.logger.flush()

                # CKPT #
                if self.ckpt_period > 0 and (epoch_idx + 1) % self.ckpt_period == 0:
                    # todo: do checkpointing here
                    # self.model_dir
                    warnings.warn('Checkpointing not yet implemented')
        except KeyboardInterrupt:
            print('Manual stop of training')

    def _test(self, strategy: Strategy, version=None):
        try:
            with torch.no_grad():
                optimizers, _ = strategy.opt_sched_unpack(strategy.optim_schedulers())
                outputs = []
                self._set_eval_mode(strategy)  # set model.eval()

                for batch_idx, batch in self._batch_iter(tqdm(self.tst_dataloader, desc='Test Batches')):
                    for optimizer_idx, optimizer in enumerate(optimizers):
                        output = strategy.tst_step(
                            batch=batch,
                            batch_idx=batch_idx,
                            optimizer_idx=optimizer_idx,
                            num_batches=len(self.tst_dataloader),
                        )
                        outputs.append(output)
                strategy.tst_agg_outputs(outputs, AggFn(outputs))
        except KeyboardInterrupt:
            print('Manual stop of testing')

    def _set_train_mode(self, strategy: Strategy):
        mode_str = 'train'
        if self.mode != mode_str:
            self.mode = mode_str
            for m in strategy.modules:
                m.train()

    def _set_eval_mode(self, strategy: Strategy):
        mode_str = 'eval'
        if self.mode != mode_str:
            self.mode = mode_str
            for m in strategy.modules:
                m.eval()

    def _strat_to_device(self, strategy: Strategy):
        for m in strategy.modules:
            m.to(self.device)

    def _batch_iter(self, dataloader):
        for batch_idx, batch in enumerate(dataloader):
            yield batch_idx, self._batch_to_device(batch)

    def _batch_to_device(self, batch):
        if isinstance(batch, Sequence):
            return tuple(t.to(self.device, non_blocking=True) if hasattr(t, 'to') else t for t in batch)
        elif hasattr(batch, 'to'):
            batch = batch.to(self.device, non_blocking=True)
        return batch
