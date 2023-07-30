import warnings

import numpy as np
import scipy.optimize as spop
import scipy.signal as spsg
import scipy.stats as spst

from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)

from kdquantile import KDQuantileTransformer


class KDDiscretizer(TransformerMixin, BaseEstimator):
    """
    Bin continuous data into intervals using KDE local minima.
 
    Parameters
    ----------
    beta: float > 0, 'scott', 'silverman', or None
        bandwidth parameter for differentiated KDE (used to predict)

    precision: float
        Absolute precision of finite differences, used to compute the 
            cluster centroids and boundaries.

    enable_predict_proba: bool
        Whether to enable predict_proba by computing KDE of each bin.

    random_state: int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    bin_edges_ : ndarray of ndarray of shape (n_features,)
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    centroids_ : ndarray of ndarray of shape (n_features,)
        The centers of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    n_bins_ : ndarray of shape (n_features,), dtype=np.int_
        Number of bins per feature. 

    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """
    def __init__(
        self,
        beta="scott",
        precision=1e-3,
        enable_predict_proba=False,
        random_state=None,
    ):
        self.beta = beta
        self.precision = precision
        self.enable_predict_proba = enable_predict_proba

        self.n_bins_ = None  # ndarray of shape (n_features,)
        self.centroids_ = None # np.array of size (n_clusters_,)
        self.boundaries_ = None # np.array of size (n_clusters_ - 1,)

        self.kdes_ = None

    def fit(self, X):
        prec = self.precision
        alpha = self.alpha
        beta = self.beta
        n_bins_min = self.n_bins_min
        n_bins_max = self.n_bins_max
        if self.enable_predict_proba and X.shape[1] > 1:
            raise ValueError("enable_predict_proba requires only 1 feature")

        self.n_features_in_ = X.shape[1]
        self.kdes_ = []
        
        # Finds modality, centroids, and boundaries of transformed data
        n_evals = int(np.round(1.0 / prec)) + 4
        evals = np.linspace(np.min(X, axis=0) - prec, np.max(X, axis=0) + prec, n_evals)
        for cix in range(X.shape[1]):
            colX = X[:, cix]
            kder = spst.gaussian_kde(colX, beta)
            pdf = kder.pdf(evals)
            centroid_ixs = np.sort(spsg.argrelmax(pdf)[0])
            self.centroids_[cix] = evals[centroid_ixs, cix]
            boundary_ixs, _ = spsg.find_peaks(-1 * pdf)
            boundary_ixs = np.sort(boundary_ixs)
            self.boundaries_[cix] = evals[boundary_ixs, cix]
            n_centroids = self.centroids_[cix].shape[0]
            n_boundaries = self.boundaries_[cix].shape[0]
            if not n_centroids == n_boundaries + 1:
                warnings.warn("{}: {} centroids, {} boundaries".format(
                    cix, n_centroids, n_boundaries))
            self.n_bins_[cix] = n_centroids
            
            # Fits KDEs for each bin
            if not self.enable_predict_proba:
                continue
            kders = []
            colX_ =  colX.reshape(-1, 1)  # (N, 1)
            boundaries_ = self.boundaries_[cix].reshape(1, -1)  # (1, n_bins_[cix])
            preds = np.sum(colX_ >= boundaries_, axis=1)  # (N,)
            n_unique_preds = np.unique(preds).shape[0]
            if n_unique_preds != self.n_bins_[cix]:
                warnings.warn("{}: n_unique_preds={}, n_clusters={}".format(
                    cix, n_unique_preds, self.n_bins_[cix]))
                print(self.get_centroids())
                print(self.get_boundaries())
                print(self.get_intervals())
            for i in range(self.n_bins_[cix]):
                curX = colX[preds == i]
                if curX.shape[0] == 0:
                    warnings.warn("{}: ZERO samples for k=%i".format(cix, i))
                    kde_i = spst.gaussian_kde(X) # XXX
                elif curX.shape[0] == 1:
                    warnings.warn("{}: only one sample for k=%i".format(cix, i))
                    xv = np.asscalar(curX)
                    Xv = np.array([xv, xv]) + np.random.normal(scale=1e-5, size=2)
                    kde_i = spst.gaussian_kde(Xv)
                elif np.std(curX) < 1e-5:
                    warnings.warn("{}: zero variance for k=%i".format(cix, i))
                    kde_i = spst.gaussian_kde(
                        X + np.random.normal(loc=0, scale=1e-5, size=X.shape))
                else:
                    kde_i = spst.gaussian_kde(X[preds == i])
                kders.append(kde_i)
            self.kdes_.append(kders)

        return self

    def get_intervals(self, idx):
        """
        Returns intervals for the feature at index idx.

        Returns
        -------
        intervals: np.array of shape (n_boundaries+1, 2)
            Returns the sorted intervals, one interval per row.
            The left endpoints of the intervals would be [-inf] + boundaries.
            The right endpoints of the intervals would be boundaries + [+inf].

        being a tuple (left_i, right_i). left_0 is always -inf, and 
        """
        intervals = np.c_[
            np.r_[np.NINF, self.boundaries_[idx]], 
            np.r_[self.boundaries_[idx], np.PINF]
        ]
        return intervals

    def transform(self, X):
        """
        Parameters
        ----------
        X: ndarray of shape (N, D)

        Returns
        -------
        Y: ndarray of shape (N, D)
            The discretization of X, taking values from [0, ..., n_clusters].
        """
        N, D = X.shape
        Y = np.empty((N, D), dtype=int)
        for cix, colX in enumerate(X.T):
            colX_ = np.reshape(-1, 1)  # (N, 1)
            boundaries_ = self.boundaries_[cix].reshape(1, -1) # (1, n_bins_[cix])
            Y[:, cix] = np.sum(colX_ > boundaries_, axis=1)
        return Y

    def predict(self, X):
        proba = self.predict_proba(X)
        pred = np.zeros_like(proba)
        pred[np.arange(pred.shape[0]), np.argmax(proba, axis=1)] = 1.
        return pred

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X: np.array of shape (N, 1)

        Returns
        -------
        Y: np.array of shape (N, self.n_bins_[0])
            The KDE-estimated probability that each item in X belongs
            to each of the classes.
        """
        if not self.enable_predict_proba:
            raise ValueError("predict_proba not enabled")

        if not self.n_features_in_ == 1:
            raise ValueError("1d inputs are required for predict_proba")

        N = X.shape[0]
        unnorm_logprobas = np.zeros((N, self.n_bins_[0]))
        for k in range(self.n_bins_[0]):
            unnorm_logprobas[:, k] = self.kdes_[0][k].logpdf(X)
        pis = np.max(unnorm_logprobas, axis=1, keepdims=True)
        lnorm = pis + np.log(np.sum(np.exp(unnorm_logprobas-pis), axis=1, keepdims=True))
        pred_logprobas = unnorm_logprobas - lnorm
        pred_probas = np.exp(pred_logprobas)
        return pred_probas  


class KDQuantileDiscretizer(TransformerMixin, BaseEstimator):
    """
    Bin continuous data into intervals using KDQuantileTransformer.
    Outputs are ordinal-encoded, with the exception of predict_proba.

    Parameters
    ----------
    alpha: float > 0, 'scott', 'silverman', or None
        bandwidth parameter for integrated KDE (used to transform)

    beta: float > 0, 'scott', 'silverman', or None
        bandwidth parameter for differentiated KDE (used to predict)

    precision: float
        Absolute precision of finite differences, used to compute the 
            cluster centroids and boundaries.

    Attributes
    ----------
    bin_edges_ : ndarray of ndarray of shape (n_features,)
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    centroids_ : ndarray of ndarray of shape (n_features,)
        The centers of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.


    n_bins_ : ndarray of shape (n_features,), dtype=np.int_
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """

    def __init__(
        self,
        alpha=1.,
        n_quantiles=1000,
        subsample=10000,
        beta="scott",
        precision=1e-3,
        enable_predict_proba=False,
        random_state=None,
    ):
        self.alpha = alpha
        self.beta = beta

        self.n_bins_ = None  # ndarray of shape (n_features,)
        self.centroids_ = None # np.array of size (n_clusters_,)
        self.boundaries_ = None # np.array of size (n_clusters_ - 1,)
        self.kdqt_ = KDQuantileTransformer(
            alpha=alpha,
            n_quantiles=n_quantiles,
            subsample=subsample,
            random_state=random_state,
        )
        self.kdd_ = KDDiscretizer(
            beta=beta,
            precision=precision,
            enable_predict_proba=enable_predict_proba,
            random_state=random_state,
        )

    def fit(self, X):
        alpha = self.alpha
        beta = self.beta

        self.n_features_in_ = X.shape[1]
        
        # Finds non-parametric transformation onto [0,1] interval
        self.kdqt_.fit(X)
        T = self.kdqt_.transform(X)

        self.kdd_.fit(T)

        return self

    def get_centroids(self):
        """
        Returns peaks for each of the classes
        """
        return self.centroids_.copy()

    def get_boundaries(self):
        """
        Returns boundaries, a (possibly empty) list 
        of the boundaries between each of the discretized categories.
        The returned list will be of length (n_clusters - 1).
        """
        return self.boundaries_.copy()

    def get_intervals(self):
        """
        Returns
        -------
        intervals: np.array of shape (n_boundaries+1, 2)
            Returns the sorted intervals, one interval per row.
            The left endpoints of the intervals would be [-inf] + boundaries.
            The right endpoints of the intervals would be boundaries + [+inf].

        being a tuple (left_i, right_i). left_0 is always -inf, and 
        """
        intervals = np.c_[
            np.r_[np.NINF, self.boundaries_], 
            np.r_[self.boundaries_, np.PINF]
        ]
        return intervals

    def transform(self, X):
        Y = self.kdqt_.transform(X)
        T = self.kdd_.transform(Y)
        return T

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.array of shape (n,)

        Returns
        -------
        Y: np.array of shape (n,)
            The discretization of X, taking values from [0, ..., n_clusters].
        """
        Y = self.kdqt_.transform(X)
        pred = self.kdd_.predict(Y)
        return pred

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X: np.array of shape (N, 1)

        Returns
        -------
        Y: np.array of shape (N, self.n_bins_[0])
            The KDE-estimated probability that each item in X belongs
            to each of the classes.
        """
        Y = self.kdqt_.transform(X)
        proba = self.kdd_.predict_proba(Y)
        return proba
