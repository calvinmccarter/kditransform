import numpy as np
import pytest
import time

import kdquantile

from sklearn.preprocessing import QuantileTransformer

def test_correlation():
    rng = np.random.default_rng(12345)

    X = rng.uniform(size=1000)
    Y = np.log(X/(1-X))
    Y = np.sign(Y) * np.abs(Y)**1.4

    cp = np.cov(X,Y)
    cp = cp[0,1] / np.sqrt(cp[0,0] * cp[1,1])

    XR = np.argsort(np.argsort(X))
    YR = np.argsort(np.argsort(Y))
    cr = np.cov(XR,YR)
    cr = cr[0,1] / np.sqrt(cr[0,0] * cr[1,1])

    Xkdq = kdquantile.PolyExpKDQuantileTransformer(alpha=1.).fit_transform(X.reshape(-1, 1))
    Ykdq = kdquantile.PolyExpKDQuantileTransformer(alpha=1.).fit_transform(Y.reshape(-1, 1))
    ckdq = np.cov(Xkdq.ravel(), Ykdq.ravel())
    ckdq = ckdq[0,1] / np.sqrt(ckdq[0,0] * ckdq[1,1])

    assert Xkdq.shape == (1000, 1)
    assert Ykdq.shape == (1000, 1)
    assert cp <= ckdq <= cr
    np.testing.assert_allclose(ckdq, 0.9581695121244862, rtol=1e-6, atol=1e-6)


def test_multicolumns():
    rng = np.random.default_rng(12345)
    X = rng.uniform(size=(500, 100))
    Xkdq = kdquantile.PolyExpKDQuantileTransformer(alpha=1.).fit_transform(X)
    assert Xkdq.shape == X.shape
    np.testing.assert_allclose(X, Xkdq, rtol=1e-2, atol=1e-1)


def test_constant():
    X = np.ones((100, 1))
    Ykdq = kdquantile.KDQuantileTransformer(alpha=1.).fit_transform(X)
    Yq = QuantileTransformer().fit_transform(X)
    assert Ykdq.shape == Yq.shape
    np.testing.assert_equal(Yq, Ykdq)


def test_precision():
    rng = np.random.default_rng(12345)
    X = rng.lognormal(0.5, 1, size=(12000, 1))
    ultra = kdquantile.KDQuantileTransformer(alpha=1., n_quantiles=None, subsample=None, random_state=12345)
    order4 = kdquantile.PolyExpKDQuantileTransformer(alpha=1., order=4, n_quantiles=1000)
    order1 = kdquantile.PolyExpKDQuantileTransformer(alpha=1., order=1, n_quantiles=1000)
    ultra_time = time.time()
    Y_ultra = ultra.fit_transform(X)
    ultra_time = time.time() - ultra_time
    order4_time = time.time()
    Y_order4 = order4.fit_transform(X)
    order4_time = time.time() - order4_time
    order1_time = time.time()
    Y_order1 = order1.fit_transform(X)
    order1_time = time.time() - order1_time
    print(f"ultra_time: {ultra_time:.3f} order4_time: {order4_time:.3f} order1_time: {order1_time:.3f}")
    np.testing.assert_allclose(Y_ultra, Y_order4, rtol=0.0, atol=0.0066)
    np.testing.assert_allclose(Y_ultra, Y_order1, rtol=0.0, atol=0.021)
