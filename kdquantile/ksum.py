from numba import njit
import numpy as np

from scipy.special import factorial


def norm_const_K(betas):
    factorial_terms = np.array([
        betas[k - 1] * factorial(k-1) for k in range(1, len(betas) + 1)
    ])
    normalization_constant = 2 * np.sum(factorial_terms)
    return normalization_constant


def betas_for_order(order):
    unnorm_betas = 1 / factorial(np.arange(order + 1))
    betas = unnorm_betas / norm_const_K(unnorm_betas)
    return betas


def roughness_K(betas):
    beta_use = betas / norm_const_K(betas)
    betakj = np.outer(beta_use, beta_use)
    n = len(betas)
    kpj = np.log2(np.outer(2 ** np.arange(n), 2 ** np.arange(n))).astype(np.int64)
    result = np.sum(betakj / (2 ** kpj) * factorial(kpj))
    return result

def var_K(betas):
    beta_use = betas / norm_const_K(betas)
    factorial_terms = np.array([
        beta_use[k - 1] * factorial(k+1) for k in range(1, len(betas) + 1)
    ])
    return 2 * np.sum(factorial_terms)


@njit
def ksum_numba(x, y, x_eval, h, betas, output, counts, coefs, Ly, Ry):
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    order = betas.shape[0] - 1

    for i in range(order + 1):
        Ly[i, 0] = np.power(-x[0], i) * y[0]
    for i in range(1, n):
        for j in range(order + 1):
            Ly[j, i] = np.power(-x[i], j) * y[i] + np.exp((x[i-1] - x[i]) / h) * Ly[j, i-1]
            Ry[j, n - i - 1] = (
                np.exp((x[n - i - 1] - x[n - i]) / h) 
                * (np.power(x[n - i], j) * y[n-i] + Ry[j, n - i])
            )
 
    count = 0
    for i in range(n_eval):
        if x_eval[i] >= x[n - 1]:
            counts[i] = n
        else:
            while count < n and x[count] <= x_eval[i]:
                count += 1
            counts[i] = count

    for orddo in range(0, order + 1):
        coefs[0] = 1
        coefs[orddo] = 1
        if orddo > 1:
            num = 1.
            for j in range(2, orddo + 1):
                num *= j
            denom1 = 1.
            denom2 = num / orddo
            for i in range(2, orddo + 1):
                coefs[i - 1] = num / (denom1 * denom2)
                denom1 *= i
                denom2 /= orddo - i + 1
        denom = np.power(h, orddo)
        
        ix = 0
        for i in range(n_eval):
            ix = np.round(counts[i])
            if ix == 0:
                exp_mult = np.exp((x_eval[i] - x[0]) / h)
                output[i] += (
                    betas[orddo] * np.power(x[0] - x_eval[i], orddo) 
                    / denom * exp_mult * y[0]
                )
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo] * coefs[j] * np.power(-x_eval[i], orddo - j)
                        * Ry[j, 0] / denom * exp_mult
                    )
            else:
                exp_mult = np.exp((x[ix - 1] - x_eval[i]) / h)
                for j in range(orddo + 1):
                    output[i] += betas[orddo] * coefs[j] * (
                        np.power(x_eval[i], orddo - j) * Ly[j, ix - 1] * exp_mult
                        + np.power(-x_eval[i], orddo - j) * Ry[j, ix - 1] / exp_mult
                    ) / denom

"""

def ksum(x, y, x_eval, h=None, betas=None):
    # Assumes x and x_eval are sorted
    n = x.shape[0]
    n_eval = x_eval.shape[0]

    if betas is None:
        # Smooth first-order kernel
        betas = np.array([0.25, 0.25])
    else:
        betas = betas / norm_const_K(betas)
    print(betas)
    order = betas.shape[0] - 1

    if h is None:
        # Silverman's rule
        h = (
            8 * np.sqrt(np.pi) / 3 * roughness_K(betas) / (var_K(betas) ** 2) / n
        ) ** 0.2 * np.std(x)

    output = np.zeros_like(x_eval)
    counts = np.zeros_like(x_eval).astype(np.int64)
    coefs = np.zeros_like(betas)


    Ly = np.zeros((order + 1, n), order="C")
    Ry = np.zeros((order + 1, n), order="C")

    ksum_numba(x, y, x_eval, h, betas, output, counts, coefs, Ly, Ry)
    output = output / n / h

    return output


if __name__ == "__main__":
    x = np.sort(np.array([0., 0.1, 0.2, 1., 5.]).astype(np.float64))
    y = np.ones_like(x)
    midpts = x[:-1] + 0.5*np.diff(x)
    x_eval = np.sort(np.concatenate([x, midpts]))
    h = None # 0.1 # np.std(x)
    betas = np.array([1., 1., 1 / 2., 1 / 6., 1 / 24.])
    #betas = None
    #betas = np.array([1., 1.])
    output = ksum(x, y, x_eval, h, betas)
    print(output)
"""
