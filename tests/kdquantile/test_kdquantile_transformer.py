import numpy as np
import pytest

import kdquantile


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

    Xkdq = kdquantile.KDQuantileTransformer(alpha=1.).fit_transform(X.reshape(-1, 1))
    Ykdq = kdquantile.KDQuantileTransformer(alpha=1.).fit_transform(Y.reshape(-1, 1))
    ckdq = np.cov(Xkdq.ravel(), Ykdq.ravel())
    ckdq = ckdq[0,1] / np.sqrt(ckdq[0,0] * ckdq[1,1])

    assert Xkdq.shape == (1000, 1)
    assert Ykdq.shape == (1000, 1)
    assert cp <= ckdq <= cr
    np.testing.assert_allclose(ckdq, 0.9581695121244862, rtol=1e-6, atol=1e-6)


def test_multicolumns():
    rng = np.random.default_rng(12345)
    X = rng.uniform(size=(500, 100))
    Xkdq = kdquantile.KDQuantileTransformer(alpha=1.).fit_transform(X)
    assert Xkdq.shape == X.shape
    np.testing.assert_allclose(X, Xkdq, rtol=1e-2, atol=1e-1)
