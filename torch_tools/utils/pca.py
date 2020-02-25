import torch


class PCA:
    def __init__(self, n_components=None, X=None, high_var=True, **kwargs):
        self._n_components = n_components
        self.high_var = high_var
        self._V = self.S = self.N = self.mu = None

        if X is not None:
            self.fit(X)

    @property
    def is_init(self):
        return self._V is not None

    @property
    def n_components(self):
        k = self._n_components

        if k is not None:
            if self.is_init and 0 < k < 1:
                if self.high_var:
                    cum_sum_var = torch.cumsum(self.explained_variance_, dim=0)
                else:
                    cum_sum_var = torch.cumsum(self.explained_variance_.flip([0]), dim=0)

                cum_sum_var_ratio = cum_sum_var / cum_sum_var[-1]
                return (cum_sum_var_ratio < k).sum().item() + 1
            else:
                return k

        if self.is_init:
            return self._V.size(1)

        return None

    @n_components.setter
    def n_components(self, k):
        self._n_components = k

    @property
    def explained_variance_(self):
        self.__assert_is_init()
        return self.S.pow(2) / (self.N - 1)

    @property
    def V(self):
        if self.is_init:
            return self._V[:, self._slice]
        return None

    def __call__(self, X):
        return self.transform(X)

    def fit(self, X):
        X = X.double()
        self.mu = X.mean(dim=0)
        X = X - self.mu
        _, self.S, self._V = torch.svd(X)
        self.N = X.size(0)
        return self

    def transform(self, X):
        self.__assert_is_init()
        return ((X.double() - self.mu) @ self.V).type(X.dtype)

    def revert(self, X):
        self.__assert_is_init()
        return (X.double() @ self.V.T + self.mu).type(X.dtype)

    def to(self, device):
        if self.is_init:
            self._V = self._V.to(device)
            self.S = self.S.to(device)

    @property
    def _slice(self):
        k = self.n_components
        if self.high_var:
            return slice(None, k)
        else:
            return slice(-k, None)

    def __assert_is_init(self):
        if not self.is_init:
            raise ValueError('need to call fit beforehand')
