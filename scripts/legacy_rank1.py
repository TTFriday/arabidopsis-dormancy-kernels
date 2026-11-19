"""Frozen legacy rank-one formula for forensic reproduction, not the corrected estimator."""
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

def single_kernel_cv_r(Z, y, lam=1e-8, n_repeats=10, n_folds=5):
    """CV r for every column of Z treated as its own rank-1 kernel.

    A single standardised feature with a linear kernel and an explicit
    intercept is exactly ridge regression on that feature, so the leave-fold-out
    prediction is computed in closed form column by column: with intercept
    b = mean(y_train), alpha_j = <z_j, y - b> / (||z_j||^2 + lam) and
    z_pred_i,j = b + z_ij * alpha_j (elementwise across columns).
    """
    n = len(y)
    acc = np.zeros(Z.shape[1])
    for rep in range(n_repeats):
        pred = np.zeros((n, Z.shape[1]))
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=rep).split(Z):
            b = y[tr].mean()
            num = Z[tr].T @ (y[tr] - b)
            den = (Z[tr] ** 2).sum(axis=0) + lam
            alpha = num / den
            pred[te] = b + Z[te] * alpha[None, :]
        rs = np.array([stats.pearsonr(pred[:, j], y)[0] for j in range(Z.shape[1])])
        acc += rs
    return acc / n_repeats
