import heapq
import warnings
from pathlib import Path
from typing import Union

import torch

from torch_tools.training import Callback


class CheckpointCallback(Callback):
    def __init__(self, save_dir: Union[Path, str], n_best_or_period: int = 1, metric_name: str = None,
                 metric_cmp='max'):
        if save_dir is None:
            raise ValueError('Cannot do checkpointing when the save_dir is None')
        if metric_cmp not in ['min', 'max']:
            raise ValueError(f'Unknown metric comparison (got {metric_cmp})')
        if n_best_or_period < 1:
            raise ValueError(f'n_best_or_period must be greater or equal than 1 (got {n_best_or_period})')

        super().__init__()
        self.save_dir = save_dir
        self.n_best_or_period = n_best_or_period
        self.metric_name = metric_name
        self.metric_cmp = metric_cmp
        self.best_metrics = []

    def _weight_path(self, epoch_idx):
        return Path(self.save_dir) / \
               self.executor.exp_name / \
               f'version_{self.executor.version}' / \
               f'weights_ckpt_epoch_{epoch_idx + 1}.ckpt'

    def _save(self, epoch_idx):
        path = self._weight_path(epoch_idx)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        self.strategy.save(path)

    def _save_period(self, epoch_idx):
        if (epoch_idx + 1) % self.n_best_or_period == 0:
            self._save(epoch_idx)

    def _save_best_metrics(self, epoch_idx, metric):
        print(metric)
        if self.metric_cmp == 'min':
            metric *= -1
        item = (metric, epoch_idx)

        if len(self.best_metrics) < self.n_best_or_period:
            heapq.heappush(self.best_metrics, item)
            self._save(epoch_idx)

        elif metric > heapq.nlargest(1, self.best_metrics)[0][0]:
            _, epoch_idx_to_del = heapq.heappushpop(self.best_metrics, item)
            self._weight_path(epoch_idx_to_del).unlink()
            self._save(epoch_idx)

    def on_epoch_end(self, epoch_idx, val_outputs=None):
        if self.metric_name is None:
            self._save_period(epoch_idx)
        else:
            metric = val_outputs.get(self.metric_name)
            if metric is None:  # treating it as period
                warnings.warn(f'Metric {self.metric_name} not found in validation outputs, saving as period')
                self._save_period(epoch_idx)
            else:
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()
                self._save_best_metrics(epoch_idx, metric)
