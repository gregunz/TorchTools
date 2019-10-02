from abc import ABC, abstractmethod
from argparse import ArgumentParser


class AddArgs(ABC):
    @staticmethod
    @abstractmethod
    def add_args(parser: ArgumentParser):
        raise NotImplementedError