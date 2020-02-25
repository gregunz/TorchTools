import torch

from torch_tools.utils import PCA


# class MahalanobisDistance:
#     def __init__(self, X=None, impl='pca', n_components=None, high_var=True, min_std=1e-10, inv_cov=None):
#         if impl == 'pca':
#             self = MahalanobisDistancePCA(X=X, n_components=n_components, high_var=high_var, min_std=1e-10)
#         elif impl == 'cov':
#             self = MahalanobisDistanceCov(X=X, inv_cov=inv_cov)
#         else:
#             raise NotImplementedError(f'unknown implementation: {impl}')


class MahalanobisDistanceCov:
    def __init__(self, X=None, inv_cov=None):
        if X is not None and inv_cov is not None:
            raise ValueError('Cannot provide both X (the matrix of vectors used to compute the inverse covariance)'
                             ' and the inverse covariance matrix')
        self._cov = None
        self.inv_cov = None
        self.mu = None

        if inv_cov is not None:
            self.inv_cov = torch.tensor(inv_cov)

        if X is not None:
            self.inv_cov, self._cov, self.mu = self.compute_inv_cov(X, return_values=True)

    @property
    def cov(self):
        if self._cov is None and self.inv_cov is not None:
            self._cov = self.inv_cov.inverse()
        return self._cov

    @staticmethod
    def compute_dist_cov(x1, x2=None, inv_cov=None, return_inv_cov=False, center=True):
        if not isinstance(x1, torch.Tensor):
            x1 = torch.tensor(x1, device=inv_cov.device if inv_cov is not None else None)
        if inv_cov is None:
            if x1.dim() == 1:
                raise ValueError('Covariance matrix cannot be inferred if x1 is a single vector')
            # if x2 is not None:
            #    raise ValueError('Requires one single matrix (x1 only) or two vectors (x1 and x2)')
            inv_cov, _, mu = MahalanobisDistanceCov.compute_inv_cov(x1, return_values=True)
            if center:
                if x2 is not None:
                    raise ValueError('Cannot center when x2 is already provided and interpreted as the center')
                x2 = mu

        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)

        X = x1
        if x2 is not None:
            if not isinstance(x2, torch.Tensor):
                x2 = torch.tensor(x2, device=x1.device)
            X = X - x2

        left = X.unsqueeze(1)
        mid = inv_cov.expand(X.size(0), -1, -1)
        right = X.unsqueeze(2)

        dist = left.bmm(mid).bmm(right).sqrt().squeeze()

        if return_inv_cov:
            return dist, inv_cov
        return dist

    @staticmethod
    def compute_inv_cov(X, return_values=False):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        mu = X.mean(0)
        cov = (X - mu).T @ (X - mu) / (X.size(0) - 1)
        inv_cov = cov.inverse()

        if return_values:
            return inv_cov, cov, mu
        return inv_cov

    def dist(self, x1, x2=None, inv_cov=None, center=True):
        if inv_cov is None:
            inv_cov = self.inv_cov
            if x2 is None and center:
                x2 = self.mu
        elif not isinstance(inv_cov, torch.Tensor):
            inv_cov = torch.tensor(inv_cov)
        dist, self.inv_cov = self.compute_dist_cov(x1, x2=x2, inv_cov=inv_cov, return_inv_cov=True)
        return dist

    def to(self, device):
        if self.inv_cov is not None:
            self.inv_cov = self.inv_cov.to(device)
        return self


class MahalanobisDistancePCA:
    def __init__(self, X=None, n_components=None, high_var=True, min_std=1e-8):
        self._pca = None
        self._n_components = n_components
        self._high_var = high_var
        self._min_std = min_std

        if X is not None:
            self._init_pca(X)

    def _init_pca(self, X):
        self._pca = PCA(n_components=self._n_components, high_var=self._high_var).fit(X)
        # self._pca = decomposition.PCA(self._n_components).fit(X.cpu().numpy())
        # X_pca = self._transform(X, norm=False)
        # self._sig = X_pca.std(dim=0)
        self._sig = self._pca.explained_variance_.sqrt()
        if self._high_var:
            self._sig = self._sig[:self._pca.n_components]
        else:
            self._sig = self._sig[-self._pca.n_components:]
        # print((self._sig <= self._min_std).sum())

    def _transform(self, X, norm=True):
        if self._pca is None:
            self._init_pca(X)

        X = self._pca.transform(X)
        # X = torch.from_numpy(self._pca.transform(X.cpu().numpy()))
        if norm:
            mask = self._sig > self._min_std
            # min_non_zero_sig = self._sig[mask].min()
            # X /= self._sig.clamp_min(min_non_zero_sig)
            # X[:, mask] /= self._sig[mask]
            X = X[:, mask] / self._sig[mask]
        # filtering zero variance components.
        # X = X[:, self._sig > 1e-10]
        return X

    def dist(self, x1, x2=None):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)

        X = self._transform(x1)

        if x2 is not None:
            if x2.dim() == 1:
                x2 = x2.unsqueeze(0)
            X = X - self._transform(x2)

        return X.pow(2).sum(dim=1).sqrt()
        # return (x1 - self._pca.revert(X)).pow(2).sum(dim=1).sqrt()

    def to(self, device):
        if self._pca is not None:
            self._pca.to(device)
        return self

    @property
    def n_components(self):
        return self._pca.n_components


def maha_scores(tng_space, tst_space, use_pca=True, **kwargs):
    if use_pca:
        maha = MahalanobisDistancePCA(X=tng_space, **kwargs)
    else:
        maha = MahalanobisDistanceCov(X=tng_space, **kwargs)
    return maha.dist(tst_space)
