import numpy as np

from sklearn.mixture import GaussianMixture

class GaussianMixtureSelect:

    def __init__(
        self,
        max_components=8,
        criteria="BIC", 
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        if criteria not in ["AIC", "BIC"]:
            raise ValueError("invalid criteria")
        self.max_components_ = max_components
        self.criteria_ = criteria

        self.covariance_type_ = covariance_type
        self.tol_ = tol
        self.reg_covar_ = reg_covar
        self.max_iter_ = max_iter
        self.n_init_ = n_init
        self.init_params_ = init_params
        self.weights_init_ = weights_init
        self.means_init_ = means_init
        self.precisions_init_ = precisions_init
        self.random_state_ = random_state
        self.warm_start_ = warm_start
        self.verbose_ = verbose
        self.verbose_interval_ = verbose_interval

        self.crits_ = np.Inf * np.ones(max_components)
        self.gmm_ = GaussianMixture()

    def fit(self, X, y=None):
        for k in range(1, self.max_components_ + 1):
            cur_gmm = GaussianMixture(
                n_components=k, 
                covariance_type=self.covariance_type_,
                tol=self.tol_,
                reg_covar=self.reg_covar_,
                max_iter=self.max_iter_,
                n_init=self.n_init_,
                init_params=self.init_params_,
                weights_init=self.weights_init_,
                means_init=self.means_init_,
                precisions_init=self.precisions_init_,
                random_state=self.random_state_,
                warm_start=self.warm_start_,
                verbose=self.verbose_,
                verbose_interval=self.verbose_interval_)
            cur_gmm.fit(X, y)
            if self.criteria_ == "BIC":
                cur_crit = cur_gmm.bic(X)
            elif self.criteria_ == "AIC":
                cur_crit = cur_gmm.aic(X)
            self.crits_[k-1] = cur_crit
            if cur_crit <= np.min(self.crits_):
                self.gmm_ = cur_gmm
        return self

    def aic(self, X):
        return self.gmm_.aic(X)

    def bic(self, X):
        return self.gmm_.bic(X)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def get_params(self, deep=True):
        p = self.gmm_.get_params()
        p["max_components"] = self.max_components_
        p["criteria"] = self.criteria_
        return p

    def predict(self, X):
        return self.gmm_.predict(X)

    def predict_proba(self, X):
        return self.gmm_.predict_proba(X)

    def sample(self, n_samples=1):
        return self.gmm_.samples(n_samples=n_samples)

    def score(self, X, y=None):
        return self.gmm_.score(X=X, y=y)

    def score_samples(self, X):
        return self.gmm_.score_samples(X)

    def set_params(self, **params):
        self.gmm_.set_params(params)
