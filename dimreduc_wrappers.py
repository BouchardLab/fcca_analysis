import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dca.cov_util import form_lag_matrix

class PCA_wrapper():
    def __init__(self, d, lag=1, marginal_only=False, normalize=False):
        self.pcaobj = PCA()
        self.dim = d
        assert(lag > 0 and isinstance(lag, int))
        self.lag = lag
        self.marginal_only = marginal_only
        self.normalize = normalize

    def fit(self, X):
        X = np.array(X)
        if self.lag > 1:
            X = form_lag_matrix(X, self.lag)
        if np.ndim(X) == 3:
            X = np.reshape(X, (-1, X.shape[-1]))
        if np.ndim(X) == 1:
            X = np.vstack([x for x in X])

        if self.marginal_only:            
            var = np.var(X, axis=0)
            self.var = var
            var_ordering = np.argsort(var)[::-1]
            self.coef_ = np.zeros((X.shape[-1], self.dim))
            for i in range(self.dim):
                self.coef_[var_ordering[i], i] = 1
        else:
            if self.normalize:
                X = StandardScaler().fit_transform(X)
            self.pcaobj.fit(X)
            self.coef_ = self.pcaobj.components_.T[:, 0:self.dim]

    def score(self):
        if self.marginal_only:
            var_ordered = np.sort(self.var)[::-1]
            return sum(var_ordered[0:self.dim]) / sum(self.var)
        else:
            return sum(self.pcaobj.explained_variance_ratio_[0:self.dim])

class NoDimreduc():
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X):
        if isinstance(X, list):
            self.coef_ = np.eye(X[0].shape[-1])
        else:
            if np.ndim(X) == 1:
                self.coef_ = np.eye(X[0].shape[-1])
            else:
                self.coef_ = np.eye(X.shape[-1])
    
    def score(self):
        return np.nan

class RandomDimreduc():
    def __init__(self, d, seed, n_samples, **kwargs):
        self.d = d
        new_seed = int(0.5 * (d + seed) * (d + seed + 1) + seed)
        self.seed = new_seed
        self.n_samples = n_samples

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        ortho = scipy.stats.ortho_group
        ortho.random_state = rng        
        if isinstance(X, list):
            self.coef_ = ortho.rvs(X[0].shape[-1], self.n_samples)[..., 0:self.d]
        else:
            if np.ndim(X) == 1:
                self.coef_ = ortho.rvs(X[0].shape[-1], self.n_samples)[..., 0:self.d]
            else:
                self.coef_ = ortho.rvs(X.shape[-1], self.n_samples)[..., 0:self.d]
    
    def score(self):
        return np.nan