# kdquantile

The kernel-density quantile transformation, like [min-max scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) and [quantile transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html), maps continuous features to the range `[0, 1]`. 
It achieves a happy balance between these two transforms, preserving the shape of the input distribution like min-max scaling, while nonlinearly attenuating the effect of outliers like quantile transformation.
It can also be used to discretize features, offering a data-driven alternative to univariate clustering or [K-bins discretization](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-discretization).

## Installation 

```
pip install -r requirements.txt
pip install -e .
pytest
```

## Usage

`kdquantile.KDQuantileTransformer` is a drop-in replacement for [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html). When `alpha` (defaults to 1.0) is small, our method behaves like the QuantileTransformer; when `alpha` is large, it behaves like MinMaxScaler.

```
from kdquantile import KDQuantileTransformer
X = np.random.uniform(size=(500, 1))
kdqt = KDQuantileTransformer(alpha=1.)
Y = kdqt.fit_transform(X)
```

`kdquantile.KDQuantileDiscretizer` offers an API based on [sklearn.preprocessing.KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html). It encodes each feature ordinally, similarly to `KBinsDiscretizer(encode='ordinal')`. 

```
from kdquantile import KDQuantileDiscretizer
rng = np.random.default_rng(1)
x1 = rng.normal(1, 0.75, size=int(0.55*N))
x2 = rng.normal(4, 1, size=int(0.3*N))
x3 = rng.uniform(0, 20, size=int(0.15*N))
X = np.sort(np.r_[x1, x2, x3]).reshape(-1, 1)
kdqd = KDQuantileDiscretizer()
T = kdqd.fit_transform(X)
```

Initialized as `KDQuantileDiscretizer(enable_predict_proba=True)`, we can also output one-hot encodings and probabilistic one-hot encodings of single-feature input data.

```
kdqd = KDQuantileDiscretizer(enable_predict_proba=True).fit(X)
P = kdqd.predict(X)  # one-hot encoding
P = kdqd.predict_proba(X)  # probabilistic one-hot encoding
```
