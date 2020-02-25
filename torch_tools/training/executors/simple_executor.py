from collections import Sequence
from typing import List

import torch
from tqdm.auto import tqdm

from torch_tools.training import Callback
from torch_tools.training.callbacks import CheckpointCallback
from torch_tools.training.executors.util import int_to_flags
from torch_tools.training.util import AggFn
from .. import Strategy, Executor


class SimpleExecutor(Executor):
    def __init__(self, tng_dataloader, exp_name, gpus: int, val_dataloader=None,
                 tst_dataloader=None, callbacks: List[Callback] = None, **kwargs):
        if callbacks is None:
            callbacks = []
        n_best_or_period = kwargs.get('n_best_or_period')
        if n_best_or_period is not None and n_best_or_period > 0:
            ckpt = CheckpointCallback(
                save_dir=kwargs.get('model_dir'),
                n_best_or_period=n_best_or_period,
                metric_name=kwargs.get('metric_name'),
                metric_cmp='max'
            )
            callbacks += [ckpt]
        super().__init__(
            tng_dataloader=tng_dataloader,
            val_dataloader=val_dataloader,
            tst_dataloader=tst_dataloader,
            exp_name=exp_name,
            gpus=int_to_flags(gpus) if isinstance(gpus, int) else gpus,
            callbacks=callbacks,
        )
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        self.device = torch.device('cpu' if len(self.gpus) == 0 else f'cuda:{self.gpus[0]}')
        self.mode = None
        self.latest_strat = None

    def train(self, strategy: Strategy, epochs: int, version=None, **kwargs):
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._train(strategy=strategy, epochs=epochs, version=version)
        self.latest_strat = strategy

    def test(self, strategy: Strategy = None, version=None, **kwargs):
        if strategy is None:
            strategy = self.latest_strat
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._test(strategy=strategy, version=version)

    def train_test(self, strategy: Strategy, epochs: int, version=None, **kwargs):
        self._init(strategy=strategy, version=version, hparams=kwargs)
        self._train(strategy=strategy, epochs=epochs, version=version)
        self._test(strategy=strategy, version=version)

    def _init(self, strategy: Strategy, version: int, hparams):
        self.version = strategy.set_default_logger(exp_name=self.exp_name, version=version)
        strategy.log_hyperparams(hparams)
        self._strat_to_device(strategy)
        for cb in self.callbacks:
            cb.set(strategy, self)

    def _train(self, strategy: Strategy, epochs: int, version: int):
        try:
            do_validation = self.val_dataloader is not None

            for epoch_idx in tqdm(list(range(epochs)), desc=f'Train Epochs'):

                # CALLBACKS epoch_begin #
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch_idx)

                # TRAINING #
                self._set_train_mode(strategy)  # set model.train()
                for batch_idx, batch in self._batch_iter(self.tng_dataloader):
                    for optimizer_idx, optimizer in enumerate(strategy.optimizers):
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
                val_outputs = None
                if do_validation:
                    with torch.no_grad():
                        self._set_eval_mode(strategy)  # set model.eval()
                        val_outputs = []
                        for batch_idx, batch in self._batch_iter(self.val_dataloader):
                            val_output = strategy.val_step(
                                batch=batch,
                                batch_idx=batch_idx,
                                epoch_idx=epoch_idx,
                                num_batches=len(self.val_dataloader),
                            )
                            val_outputs.append(val_output)
                        val_outputs = strategy.val_agg_outputs(val_outputs, AggFn(val_outputs), epoch_idx)

                # SCHEDULERS #
                for sched in strategy.schedulers:
                    sched.step()

                # LOGGING #
                strategy.logger.flush()  # todo: one should check if it slows down the training

                # CALLBACKS epoch_end #
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch_idx, val_outputs)

        except KeyboardInterrupt:
            strategy.logger.flush()
            print('Training stopped manually.')

    def _test(self, strategy: Strategy, version=None):
        try:
            with torch.no_grad():
                outputs = []
                self._set_eval_mode(strategy)  # set model.eval()

                for batch_idx, batch in self._batch_iter(tqdm(self.tst_dataloader, desc='Test Batches')):
                    output = strategy.tst_step(
                        batch=batch,
                        batch_idx=batch_idx,
                        num_batches=len(self.tst_dataloader),
                    )
                    outputs.append(output)
                strategy.tst_agg_outputs(outputs, AggFn(outputs))
        except KeyboardInterrupt:
            print('Testing stopped manually.')

    def _set_train_mode(self, strategy: Strategy):
        mode_str = 'train'
        if self.mode != mode_str:
            self.mode = mode_str
            for _, m in strategy.modules:
                m.train()

    def _set_eval_mode(self, strategy: Strategy):
        mode_str = 'eval'
        if self.mode != mode_str:
            self.mode = mode_str
            for _, m in strategy.modules:
                m.eval()

    def _strat_to_device(self, strategy: Strategy):
        for _, m in strategy.modules:
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
