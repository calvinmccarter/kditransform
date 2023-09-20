import numpy as np
import pytest

import kditransform

@pytest.mark.parametrize(
    "N", [500, 1000, 2000]
)
def test_transform(N):
    rng = np.random.default_rng(1)
    x1 = rng.normal(1, 0.75, size=int(0.55*N))
    x2 = rng.normal(4, 1, size=int(0.3*N))
    x3 = rng.uniform(0, 20, size=int(0.15*N))
    x = np.sort(np.r_[x1, x2, x3])
    x_labels = np.array([0]*int(0.55*N) + [1]*int(0.3*N) + [2]*int(0.15*N))

    kdider = kditransform.KDIDiscretizer()
    t = kdider.fit_transform(x.reshape(-1, 1))
    assert t.shape == (N, 1)
    assert list(np.unique(t)) == [0, 1, 2]
    assert kdider.n_features_in_ == 1
    centroids = kdider.get_centroids()
    boundaries = kdider.get_boundaries()
    assert len(centroids) == 1  # since 1 feature
    assert len(boundaries) == 1  # since 1 feature
    assert len(centroids[0]) == 3
    assert len(boundaries[0]) == 2
    expc = np.array([1., 4., 14.])
    expb = np.array([3., 7.7])
    np.testing.assert_allclose(centroids[0], expc, rtol=0.1, atol=0.1)
    np.testing.assert_allclose(boundaries[0], expb, rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "N", [500, 1000, 2000]
)
def test_predict_proba(N):
    rng = np.random.default_rng(1)
    x1 = rng.normal(1, 0.75, size=int(0.55*N))
    x2 = rng.normal(4, 1, size=int(0.3*N))
    x3 = rng.uniform(0, 20, size=int(0.15*N))
    x = np.sort(np.r_[x1, x2, x3])
    x_labels = np.array([0]*int(0.55*N) + [1]*int(0.3*N) + [2]*int(0.15*N))

    kdider = kditransform.KDIDiscretizer(enable_predict_proba=True)
    t = kdider.fit_transform(x.reshape(-1, 1))
    assert t.shape == (N, 1)
    assert list(np.unique(t)) == [0, 1, 2]
    assert kdider.n_features_in_ == 1
    p = kdider.predict_proba(x.reshape(-1, 1))
    assert p.shape == (N, 3)
    np.testing.assert_allclose(np.sum(p, axis=1), np.ones(N))
