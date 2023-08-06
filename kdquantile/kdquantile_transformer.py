import warnings

import numpy as np
import scipy.interpolate as spip
import scipy.stats as spst

from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted,    
    check_random_state,
)

BOUNDS_THRESHOLD = 1e-7


class KDQuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using kernel density quantiles information.

    This method transforms the features to follow a uniform distribution,
    or transforms them by scaling and translating them into a [0, 1] range,
    or does a (hyperparameter-tunable) interpolation of the two.

    Parameters
    ----------
    alpha: float > 0, 'scott', 'silverman', or None
        Bandwidth factor parameter for kernel density estimator.

    Attributes
    ----------
    quantiles_: ndarray
        Quantiles of kernel density estimator.
    """
    def __init__(
        self,
        alpha=1.,
        n_quantiles=1000,
        output_distribution="uniform",
        subsample=10000,
        random_state=None,
        copy=True,
    ):
        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

        self.n_quantiles_ = None
        self.references_ = None
        self.quantiles_ = None

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        n_samples, n_features = X.shape

        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            if np.var(col) == 0:
                # gaussian_kde -> _compute_covariance -> linalg.cholesky error
                # We instead duplicate QuantileTransformer's behavior here.
                quantiles = np.nanpercentile(col, self.references_ * 100)
            else:
                kder = spst.gaussian_kde(col, bw_method=self.alpha)
                N = col.shape[0]
                T = np.zeros(N)
                xmin = np.min(col)
                xmax = np.max(col)
                for n in range(N):
                    T[n] = kder.integrate_box_1d(xmin, col[n])
                intcx1 = kder.integrate_box_1d(xmin, xmin)
                intcxN = kder.integrate_box_1d(xmin, xmax)
                m = 1.0 / (intcxN - intcx1)
                b = -m * intcx1 # intc0 / (intc0 - intcxN)
                # T is the result of nonlinear mapping of X onto [0,1]
                T = m*T + b
                inverse_func = spip.interp1d(
                    T, col, bounds_error=False, fill_value=(xmin,xmax))
                quantiles = inverse_func(self.references_)
            self.quantiles_.append(quantiles)
        self.quantiles_ = np.transpose(self.quantiles_)

        # Make sure that quantiles are monotonically increasing
        self.quantiles_ = np.maximum.accumulate(self.quantiles_, axis=0)


    def fit(self, X, y=None):
        """Compute the kernel-smoothed quantiles used for transforming.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        if self.n_quantiles > self.subsample:
            raise ValueError(
                "The number of quantiles cannot be greater than"
                " the number of samples used. Got {} quantiles"
                " and {} samples.".format(self.n_quantiles, self.subsample)
            )

        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        self._dense_fit(X, rng)

        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse=False,
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
        )

        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._transform_col(
                X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
            )

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. 

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis. 

        Returns
        -------
        Xt : ndarray of (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(
            X, in_fit=False, copy=self.copy
        )

        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {"allow_nan": False}
