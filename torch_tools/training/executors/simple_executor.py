import torch
from tqdm.auto import tqdm

from torch_tools.training.executors.util import int_to_flags
from torch_tools.training.util import AggFn
from .. import Strategy, Executor


class SimpleExecutor(Executor):
    def __init__(self, exp_name, model_dir, gpus, ckpt_period, **kwargs):
        super().__init__(exp_name, model_dir, int_to_flags(gpus), ckpt_period)
        assert len(self.gpus) <= 1, 'not handling multiple GPUs yet'
        self.device = torch.device('cpu' if len(self.gpus) == 0 else f'cuda:{self.gpus[0]}')
        self.mode = None

    def train(self, strategy: Strategy, epochs: int, version=None, **kwargs):
        do_validation = strategy.val_data_loader() is not None
        optimizers, schedulers = strategy.opt_sched_unpack(strategy.optim_schedulers())

        for epoch_idx in tqdm(list(range(epochs)), desc=f'Epochs'):
            # TRAINING #
            self._set_train_mode(strategy)  # set model.train()
            for batch_idx, batch in enumerate(strategy.tng_data_loader()):
                for t in batch:
                    if isinstance(t, torch.Tensor):
                        t.to(self.device)
                for optimizer_idx, optimizer in enumerate(optimizers):
                    optimizer.zero_grad()
                    output = strategy.tng_step(
                        batch=batch,
                        batch_idx=batch_idx,
                        optimizer_idx=optimizer_idx,
                        epoch_idx=epoch_idx,
                    )
                    loss = output['loss']
                    loss.backward()
                    optimizer.step()

            # VALIDATING #
            if do_validation:
                self._set_eval_mode(strategy)  # set model.eval()
                outputs = []
                for batch_idx, batch in enumerate(strategy.val_data_loader()):
                    for optimizer_idx, optimizer in enumerate(optimizers):
                        output = strategy.tng_step(
                            batch=batch,
                            batch_idx=batch_idx,
                            optimizer_idx=optimizer_idx,
                            epoch_idx=epoch_idx,
                        )
                        outputs.append(output)
                strategy.val_agg_outputs(outputs, AggFn(outputs), epoch_idx)

            # SCHEDULERS #
            for sched in schedulers:
                sched.step()

    def test(self, strategy: Strategy, version=None, **kwargs):
        optimizers, _ = strategy.opt_sched_unpack(strategy.optim_schedulers())
        outputs = []
        self._set_eval_mode(strategy)  # set model.eval()
        for batch_idx, batch in enumerate(tqdm(strategy.tst_data_loader())):
            for t in batch:
                if isinstance(t, torch.Tensor):
                    t.to(self.device)
            for optimizer_idx, optimizer in enumerate(optimizers):
                output = strategy.tst_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    optimizer_idx=optimizer_idx,
                )
                outputs.append(output)
        strategy.tst_agg_outputs(outputs, AggFn(outputs))

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
