import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansSelect:
    def __init__(
        self,
        max_clusters=8,
        init="k-means++",
        n_init=10, 
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):
        self.max_clusters_ = max_clusters
        self.init_ = init
        self.n_init_ = n_init
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.verbose_ = verbose
        self.random_state_ = random_state
        self.copy_x_ = copy_x
        self.algorithm_ = algorithm

        self.sils_ = []
        self.kmeans_ = None

    def fit(self, X, y=None, sample_weight=None):
        kmeans_list = []
        for k in range(2, self.max_clusters_ + 1):
            cur_kmeans = KMeans(
                n_clusters=k,
                init=self.init_,
                n_init=self.n_init_,
                max_iter=self.max_iter_,
                tol=self.tol_,
                verbose=self.verbose_,
                random_state=self.random_state_,
                copy_x=self.copy_x_,
                algorithm=self.algorithm_)
            cur_kmeans.fit(X, y, sample_weight)
            cur_sil = silhouette_score(X, cur_kmeans.labels_, metric = "euclidean")
            kmeans_list.append((cur_kmeans, cur_sil))
        self.kmeans_ = max(kmeans_list, key=lambda x: x[1])[0]
        return self

    def predict(self, X, sample_weight=None):
        return self.kmeans_.predict(X, sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X, sample_weight)

    def transform(self, X):
        return self.kmeans_.transform(X)

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.kmeans_.transform(X)

    def get_params(self, deep=True):
        p = self.kmeans_.get_params()
        p["max_clusters"] = self.max_clusters_
        return p

    def set_params(self, **params):
        self.kmeans_.set_params(params)

    def score(self, X, y=None):
        return self.kmeans_.score(X, y)
