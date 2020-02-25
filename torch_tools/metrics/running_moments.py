import torch


class RunningMoments:
    def __init__(self):
        self.n = 0
        self._mean_old = self._mean_new = self._s_old = self._s_new = None

    def push(self, mean):
        self.n += 1

        if self.n == 1:
            self._mean_old = self._mean_new = mean
            self._s_old = torch.zeros_like(mean)
        else:
            self._mean_new = self._mean_old + (mean - self._mean_old) / self.n
            self._s_new = self._s_old + (mean - self._mean_old) * (mean - self._mean_new)

            # for next push
            self._mean_old = self._mean_new
            self._s_old = self._s_old

    @property
    def mean(self):
        self._require_stats()
        return self._mean_new

    @property
    def var(self):
        self._require_stats()
        return self._s_new / (self.n - 1)

    @property
    def std(self):
        self._require_stats()
        return self.var().sqrt()

    def _require_stats(self):
        if self.n == 0:
            raise ValueError('cannot get stats before pushing any values')
