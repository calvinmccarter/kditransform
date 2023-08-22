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
from sklearn.utils.validation import check_random_state

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
    boundaries_ : ndarray of ndarray of shape (n_features,)
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
        self.random_state = random_state

        self.n_bins_ = None  # ndarray of shape (n_columns,)
        self.centroids_ = None # list, each elt of size (n_bins_[cix],)
        self.boundaries_ = None # list, each elt of size (n_bins_[cix] - 1,)

        self.kdes_ = None  # list of length n_columns, each with n_bins_[cix] elts

    def fit(self, X):
        n_samples, n_features = X.shape

        if isinstance(self.beta, list):
            assert len(self.beta) == n_features
            betas = self.beta
        else:
            betas = [self.beta] * n_features

        rng = check_random_state(self.random_state)
        prec = self.precision
        if self.enable_predict_proba and X.shape[1] > 1:
            raise ValueError("enable_predict_proba requires only 1 feature")

        self.n_features_in_ = n_features
        self.n_bins_ = np.empty((n_features,), dtype=int)
        self.centroids_ = [None] * n_features
        self.boundaries_ = [None] * n_features
        
        self.kdes_ = []  # Only appended to when enable_predict_proba

        # Finds modality, centroids, and boundaries of transformed data
        n_evals = int(np.round(1.0 / prec)) + 4
        evals = np.linspace(np.min(X, axis=0) - prec, np.max(X, axis=0) + prec, n_evals)
        for cix in range(n_features):
            colX = X[:, cix]
            kder = spst.gaussian_kde(colX, betas[cix])
            pdf = kder.pdf(evals[:, cix])
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
            colX_ =  colX.reshape(-1, 1)  # (N, 1)
            boundaries_ = self.boundaries_[cix].reshape(1, -1)  # (1, n_bins_[cix])
            preds = np.sum(colX_ >= boundaries_, axis=1)  # (N, )
            n_unique_preds = np.unique(preds).shape[0]
            if n_unique_preds != self.n_bins_[cix]:
                warnings.warn("{}: n_unique_preds={}, n_clusters={}".format(
                    cix, n_unique_preds, self.n_bins_[cix]))
                print(self.get_centroids())
                print(self.get_boundaries())
                print(self.get_intervals())
            kders = []
            for i in range(self.n_bins_[cix]):
                curX = colX[preds == i]
                if curX.shape[0] == 0:
                    warnings.warn("{}: ZERO samples for k=%i".format(cix, i))
                    kde_i = spst.gaussian_kde(colX) # XXX
                elif curX.shape[0] == 1:
                    warnings.warn("{}: only one sample for k=%i".format(cix, i))
                    xv = curX.item()
                    Xv = np.array([xv, xv]) + rng.normal(scale=1e-5, size=2)
                    kde_i = spst.gaussian_kde(Xv)
                elif np.std(curX) < 1e-5:
                    warnings.warn("{}: zero variance for k=%i".format(cix, i))
                    kde_i = spst.gaussian_kde(
                        curX + rng.normal(loc=0, scale=1e-5, size=curX.shape))
                else:
                    kde_i = spst.gaussian_kde(curX)
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
            colX_ = np.reshape(colX, (-1, 1))  # (N, 1)
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
            unnorm_logprobas[:, k] = self.kdes_[0][k].logpdf(X[:, 0])
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
        Bandwidth parameter for KDQuantileTransformer (default is 1).

    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    subsample : int, default=10_000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    beta: float > 0, 'scott', 'silverman', or None
        Bandwidth parameter for KDDiscretizer (default is 'scott').

    precision: float
        Absolute precision of finite differences, used to compute the
            cluster centroids and boundaries.

    enable_predict_proba: bool
        Whether to also estimate KDEs for each discrete state, allowing
            probabilistic cluster membership predictions (default is False).

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see `subsample` for more details.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    kdqt_: fitted KDQuantileTransformer object

    kdd_: fitted KDDiscretizer object

    n_features_in_: int
        Number of features seen during `fit`.
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
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.beta = beta
        self.precision = precision
        self.enable_predict_proba = enable_predict_proba
        self.random_state = random_state

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
        Returns peaks for each of the classes.
        Returns a list, each elt corresponding to feature with cix index,
        having size (n_bins_[cix],).
        """
        centroids = [
            self.kdqt_.inverse_transform(cent.reshape(-1, 1)).ravel()
            for cent in self.kdd_.centroids_
        ]
        return centroids

    def get_boundaries(self):
        """
        Returns boundaries, a (possibly empty) list
        of the boundaries between each of the discretized categories.
        The returned list will be of length (n_clusters - 1).
        """
        boundaries = [
            self.kdqt_.inverse_transform(bound.reshape(-1, 1)).ravel()
            for bound in self.kdd_.boundaries_
        ]
        return boundaries

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


class KBinsKDQuantileDiscretizer(TransformerMixin, BaseEstimator):
    """
    Bin continuous data into intervals using KDQuantileTransformer.
    Outputs are ordinal-encoded, with the exception of predict_proba.

    Parameters
    ----------
    n_bins: int
        Number of desired bins (default: 2).

    left_alpha: float
        Minimum alpha for bisection search.

    right_alpha: float
        Maximum alpha for bisection search.

    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    subsample : int, default=10_000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    precision: float
        Absolute precision of finite differences, used to compute the
            cluster centroids and boundaries.

    enable_predict_proba: bool
        Whether to also estimate KDEs for each discrete state, allowing
            probabilistic cluster membership predictions (default is False).

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see `subsample` for more details.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    kdqt_: fitted KDQuantileTransformer object

    kdd_: fitted KDDiscretizer object

    n_features_in_: int
        Number of features seen during `fit`.
    """

    def __init__(
        self,
        n_bins=2,
        left_alpha=0.1,
        right_alpha=10.,
        beta_backtrack=1.3,
        n_quantiles=1000,
        subsample=10000,
        precision=1e-3,
        enable_predict_proba=False,
        random_state=None,
    ):
        self.n_bins = n_bins
        self.left_alpha = left_alpha
        self.right_alpha = right_alpha
        self.beta_backtrack = beta_backtrack
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.precision = precision
        self.enable_predict_proba = enable_predict_proba
        self.random_state = random_state

        self.kdqt_ = None
        self.kdd_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        scott = np.power(n_samples*(1+2.0)/4.0, -1./(1+4))

        left_kdqt = KDQuantileTransformer(
            alpha=self.left_alpha,
            n_quantiles=self.n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        left_kdd = KDDiscretizer(
            beta=[scott] * n_features,
            precision=self.precision,
            enable_predict_proba=self.enable_predict_proba,
            random_state=self.random_state,
        )

        left_kdqt.fit(X)
        T = left_kdqt.transform(X)
        left_kdd.fit(T)
        while not (left_kdd.n_bins_ <= self.n_bins).all():
            for fix in range(n_features):
                if left_kdd.n_bins_[fix] > self.n_bins:
                    left_kdd.beta[fix] *= self.beta_backtrack
            left_kdd.fit(T)

        right_kdqt = KDQuantileTransformer(
            alpha=self.right_alpha,
            n_quantiles=self.n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        right_kdd = KDDiscretizer(
            beta=left_kdd.beta,
            precision=self.precision,
            enable_predict_proba=self.enable_predict_proba,
            random_state=self.random_state,
        )
        right_kdqt.fit(X)
        T = right_kdqt.transform(X)
        right_kdd.fit(T)

        assert (left_kdd.n_bins_ <= right_kdd.n_bins_).all()
        assert (left_kdd.n_bins_ <= self.n_bins).all()
        assert (right_kdd.n_bins_ >= self.n_bins).all()

        left_n_bins = left_kdd.n_bins_.copy()
        right_n_bins = right_kdd.n_bins_.copy()
        left_alpha = [self.left_alpha] * n_features
        right_alpha = [self.right_alpha] * n_features

        geomean = list(np.sqrt(np.array(left_alpha) * np.array(right_alpha)))
        mid_kdqt = KDQuantileTransformer(
            alpha=geomean,
            n_quantiles=self.n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        mid_kdd = KDDiscretizer(
            beta=left_kdd.beta,
            precision=self.precision,
            enable_predict_proba=self.enable_predict_proba,
            random_state=self.random_state,
        )

        while not (left_n_bins == right_n_bins).all():
            geomean = list(np.sqrt(np.array(left_alpha) * np.array(right_alpha)))
            mid_kdqt.alpha = geomean

            mid_kdqt.fit(X)
            T = mid_kdqt.transform(X)
            mid_kdd.fit(T)

            for fix in range(n_features):
                # mid_n_bins is an increasing function of alpha
                if mid_kdd.n_bins_[fix] == self.n_bins:
                    left_n_bins[fix] = mid_kdd.n_bins_[fix]
                    right_n_bins[fix] = mid_kdd.n_bins_[fix]
                    left_alpha[fix] = mid_kdqt.alpha[fix]
                    right_alpha[fix] = mid_kdqt.alpha[fix]
                elif mid_kdd.n_bins_[fix] < self.n_bins:
                    left_n_bins[fix] = mid_kdd.n_bins_[fix]
                    left_alpha[fix] = mid_kdqt.alpha[fix]
                elif self.n_bins < mid_kdd.n_bins_[fix]:
                    right_n_bins[fix] = mid_kdd.n_bins_[fix]
                    right_alpha[fix] = mid_kdqt.alpha[fix]
                else:
                    assert False

            assert (left_n_bins <= self.n_bins).all()
            assert (self.n_bins <= right_n_bins).all()

        mid_kdqt.fit(X)
        T = mid_kdqt.transform(X)
        mid_kdd.fit(T)
        assert (mid_kdd.n_bins_ == self.n_bins).all()
        self.kdqt_ = mid_kdqt
        self.kdd_ = mid_kdd

        return self

    def get_centroids(self):
        """
        Returns peaks for each of the classes.
        Returns a list, each elt corresponding to feature with cix index,
        having size (n_bins_[cix],).
        """
        centroids = [
            self.kdqt_.inverse_transform(cent.reshape(-1, 1)).ravel()
            for cent in self.kdd_.centroids_
        ]
        return centroids

    def get_boundaries(self):
        """
        Returns boundaries, a (possibly empty) list
        of the boundaries between each of the discretized categories.
        The returned list will be of length (n_clusters - 1).
        """
        boundaries = [
            self.kdqt_.inverse_transform(bound.reshape(-1, 1)).ravel()
            for bound in self.kdd_.boundaries_
        ]
        return boundaries

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
