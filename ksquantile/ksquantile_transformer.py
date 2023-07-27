import numpy as np
import scipy.interpolate as spip
import scipy.stats as spst

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class KSQuantileTransformer(BaseEstimator, TransformerMixin):
    """Transform features using kernel-smoothed quantiles information.

    This method transforms the features to follow a uniform distribution,
    or transforms them by scaling and translating them into a [0, 1] range,
    or does a (hyperparameter-tunable) mixture of the two.

    Parameters
    ----------
    alpha: float > 0, 'scott', 'silverman', or None
        Bandwidth parameter for kernel-smoothing.

    Attributes
    ----------
    alpha_: float
        Bandwidth parameter for kernel-smoothing.
        
    Examples
    --------
    """
    def __init__(
        self,
        alpha=1,
        algorithm="kde",
        output_min=None,
        output_max=None,
    ):
        self.alpha_ = alpha
        self.algorithm_ = algorithm

    def fit(
        self,
        X,
    ):
        """Compute the smoothed quantiles used for transforming.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        X = check_array(X)
        if self.algorithm_ == "kde":
            self.kders_ = [
                spst.gaussian_kde(X[:, i], bw_method=self.alpha_)
                for i in range(0, X.shape[1])
            ]
            # this will be slow because integrate_box_1d only takes scalar
        elif self.algorithm_ == "interp":
            self.kders_ = [
                spst.gaussian_kde(X[:, i], bw_method=self.alpha_)
                for i in range(0, X.shape[1])
            ]
            N = X.shape[0]
            intkquantiles = np.zeros(N)
            xmin = np.min(X)
            xmax = np.max(X)
            for n in range(N):
                intkquantiles[n] = kderX.integrate_box_1d(xmin, X[n])
            intcx1 = kderX.integrate_box_1d(xmin, xmin)
            intcxN = kderX.integrate_box_1d(xmin, xmax)
            m = 1.0 / (intcxN - intcx1)
            b = -m * intcx1 # intc0 / (intc0 - intcxN)
            # T is the result of nonlinear mapping of X onto [0,1]
            T = m*intkquantiles + b
            self.transform_func_ = spip.interp1d(
                X, T, bounds_error=False, fill_value=(0,1))
     

 
        # see if integrate_box_1d or interp1d is faster
        # see if m and b should be applied at fit-time or transform-time
        # ignore m and b for now
        return self
