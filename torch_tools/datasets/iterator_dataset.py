import copy
from collections import Iterator
from typing import List, Union


class IteratorDataset:
    def __init__(self, iterator: Union[List, Iterator], length: int = None):
        if length is None:  # iterator might have a length (e.g. is a list)
            try:
                length = len(iterator)
            except TypeError:
                pass
        self.length = length
        self.__iterator = iterator
        self.iterator = self.__iterator_with_size_limit()

    def __getitem__(self, idx):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = self.__iterator_with_size_limit()
            return self.__getitem__(idx)

    def __len__(self):
        if self.length is not None:
            return self.length
        raise ValueError('This iterator has unknown length. Define it at instantiation.')

    def __iterator_with_size_limit(self):
        i = 0
        while self.length is None or i < self.length:
            yield from iter(copy.copy(self.__iterator))
            i += 1
