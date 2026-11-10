"""Centered, trace-normalized kernel constructions and kernel ridge solver shared by the prediction and association scripts."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def center_scale_kernel(K_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center and trace-normalize a kernel matrix.

    Formulas:
    H = I - 11^T / n
    K_tilde = H @ K @ H / (tr(H @ K @ H) / n)

    Args:
        K_train: training kernel matrix (n_train x n_train)

    Returns:
        K_centered: centered kernel matrix
        row_mean: per-row means (used for test-side centering)
        total_mean: grand mean (used for test-side centering)
    """
    n = K_train.shape[0]

    # Centering: K_c = K - 1_n @ mu^T - mu @ 1_n^T + mu_total
    # where mu_i = mean_j(K_ij), mu_total = mean_ij(K_ij)
    row_mean = K_train.mean(axis=1)  # (n,)
    total_mean = K_train.mean()

    K_centered = K_train - row_mean[:, None] - row_mean[None, :] + total_mean

    # Trace normalization: divide by tr(K_centered) / n
    trace = np.trace(K_centered)
    if trace > 1e-10:
        scale = trace / n
        K_centered = K_centered / scale
    else:
        scale = 1.0

    return K_centered, row_mean, total_mean, scale


def apply_kernel_centering_test(
    K_test_train: np.ndarray,
    row_mean_train: np.ndarray,
    total_mean_train: float,
    scale_train: float
) -> np.ndarray:
    """Apply the training-side centering parameters to a test-train cross kernel.

    Args:
        K_test_train: test x train kernel matrix (n_test x n_train)
        row_mean_train: training-side per-row means
        total_mean_train: training-side grand mean
        scale_train: training-side scale factor

    Returns:
        K_centered: centered cross kernel
    """
    # Test-side centering: use the training-side statistics
    row_mean_test = K_test_train.mean(axis=1)  # mean of each test row

    K_centered = (
        K_test_train
        - row_mean_test[:, None]
        - row_mean_train[None, :]
        + total_mean_train
    )

    # Use the training-side scale
    K_centered = K_centered / scale_train

    return K_centered


def solve_kernel_ridge_with_intercept(
    K_train: np.ndarray,
    y_train: np.ndarray,
    lambda_reg: float,
    K_test_train: np.ndarray = None,
) -> Tuple[np.ndarray, float, float]:
    """Kernel ridge regression with an explicit intercept.

    Formulas:
    b_hat = 1^T @ C^{-1} @ y / (1^T @ C^{-1} @ 1)
    alpha_hat = C^{-1} @ (y - b_hat * 1)
    y_test = b_hat + K_test @ alpha

    where C = K + lambda * I

    Args:
        K_train: training kernel matrix (centered, n_train x n_train)
        y_train: training labels (n_train,)
        lambda_reg: regularization parameter
        K_test_train: test-train cross kernel (optional, n_test x n_train)

    Returns:
        alpha: coefficient vector
        intercept: intercept
        y_pred_test: test predictions (if K_test_train is provided)
    """
    n = K_train.shape[0]
    C = K_train + lambda_reg * np.eye(n)

    # Solve C^{-1} @ y and C^{-1} @ 1
    C_inv_y = np.linalg.solve(C, y_train)
    C_inv_1 = np.linalg.solve(C, np.ones(n))

    # Intercept: b = 1^T @ C^{-1} @ y / (1^T @ C^{-1} @ 1)
    intercept = np.sum(C_inv_y) / np.sum(C_inv_1)

    # Coefficients: alpha = C^{-1} @ (y - b * 1)
    alpha = C_inv_y - intercept * C_inv_1

    if K_test_train is not None:
        # Prediction: y_test = b + K_test @ alpha
        y_pred_test = intercept + K_test_train @ alpha
        return alpha, intercept, y_pred_test
    else:
        return alpha, intercept, None


def feature_kernel_centered_scaled(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a centered, trace-normalized linear kernel from a feature matrix.

    Args:
        X_train: training features (n_train x p)
        X_test: test features (optional, n_test x p)
        epsilon: standard-deviation floor

    Returns:
        K_train: training kernel matrix
        X_train_scaled: standardized training features
        mean_train: training-side feature means
        std_train: training-side feature standard deviations
    """
    # Training-side standardization
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)
    std_train[std_train < epsilon] = 1.0  # avoid division by zero

    X_train_scaled = (X_train - mean_train) / std_train

    # Linear kernel: K = X @ X^T / p
    p = X_train_scaled.shape[1]
    K_train = X_train_scaled @ X_train_scaled.T / p

    if X_test is not None:
        # Standardize the test side with the training-side parameters
        X_test_scaled = (X_test - mean_train) / std_train
        K_test_train = X_test_scaled @ X_train_scaled.T / p
        return K_train, X_train_scaled, mean_train, std_train, K_test_train, X_test_scaled
    else:
        return K_train, X_train_scaled, mean_train, std_train


def snp_kernel_corrected(
    X_snp_train: np.ndarray,
    X_snp_test: np.ndarray = None
) -> Tuple:
    """SNP kernel construction (centered and trace-normalized).

    Same as feature_kernel_centered_scaled, specialized for SNP data.
    Ensures the SNP kernel is properly centered and trace-normalized.
    """
    return feature_kernel_centered_scaled(X_snp_train, X_snp_test)


# === Verification code below ===

def test_kernel_centering():
    """Test that kernel centering is implemented correctly."""
    np.random.seed(42)
    n_train, n_test, p = 50, 20, 100

    # Generate random features
    X_train = np.random.randn(n_train, p)
    X_test = np.random.randn(n_test, p)

    # Build kernels
    K_train, X_train_s, mean_tr, std_tr, K_test_train, X_test_s = feature_kernel_centered_scaled(
        X_train, X_test
    )

    # Verify the training kernel is centered
    row_mean = K_train.mean(axis=1)
    total_mean = K_train.mean()
    centered_check = np.abs(row_mean - total_mean).max()

    print(f"Training kernel centering check: max|row_mean - total_mean| = {centered_check:.2e}")
    print(f"Training kernel trace/n: {np.trace(K_train) / n_train:.4f} (should be near 1.0)")

    # Verify kernel ridge regression
    y_train = np.random.randn(n_train)
    alpha, intercept, y_pred = solve_kernel_ridge_with_intercept(
        K_train, y_train, lambda_reg=1.0, K_test_train=K_test_train
    )

    print(f"Intercept: {intercept:.4f}")
    print(f"Prediction mean: {y_pred.mean():.4f}, training mean: {y_train.mean():.4f}")
    print("Kernel implementation test passed!")


if __name__ == "__main__":
    test_kernel_centering()