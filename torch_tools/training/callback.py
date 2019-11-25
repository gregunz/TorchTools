from abc import ABCMeta

from . import strategy as S


class Callback(metaclass=ABCMeta):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.strategy: S.Strategy = None
        self.executor = None

    def set(self, strategy: S.Strategy, executor):
        self.strategy = strategy
        self.executor = executor

    def on_epoch_begin(self, epoch_idx):
        pass

    def on_epoch_end(self, epoch_idx, outputs=None):
        pass
