"""Historical kernel helpers, retained as forensic source; audited code uses the centered eigensolver."""
from __future__ import annotations
import numpy as np
from typing import Tuple

def center_scale_kernel(K_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """对核矩阵进行中心化和尺度归一化。
    
    按照优化方案公式：
    H = I - 11^T / n
    K_tilde = H @ K @ H / (tr(H @ K @ H) / n)
    
    参数:
        K_train: 训练集核矩阵 (n_train x n_train)
    
    返回:
        K_centered: 中心化后的核矩阵
        row_mean: 每行均值 (用于测试端中心化)
        total_mean: 总均值 (用于测试端中心化)
    """
    n = K_train.shape[0]
    
    # 中心化: K_c = K - 1_n @ mu^T - mu @ 1_n^T + mu_total
    # 其中 mu_i = mean_j(K_ij), mu_total = mean_ij(K_ij)
    row_mean = K_train.mean(axis=1)  # (n,)
    total_mean = K_train.mean()
    
    K_centered = K_train - row_mean[:, None] - row_mean[None, :] + total_mean
    
    # 尺度归一: 除以 tr(K_centered) / n
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
    """将训练端的中心化参数应用到测试-训练交叉核。
    
    参数:
        K_test_train: 测试×训练核矩阵 (n_test x n_train)
        row_mean_train: 训练端每行均值
        total_mean_train: 训练端总均值
        scale_train: 训练端尺度因子
    
    返回:
        K_centered: 中心化后的交叉核
    """
    # 测试端中心化: 使用训练端的统计量
    row_mean_test = K_test_train.mean(axis=1)  # 测试端行的均值
    
    K_centered = (
        K_test_train 
        - row_mean_test[:, None] 
        - row_mean_train[None, :] 
        + total_mean_train
    )
    
    # 使用训练端的尺度
    K_centered = K_centered / scale_train
    
    return K_centered

def solve_kernel_ridge_with_intercept(
    K_train: np.ndarray,
    y_train: np.ndarray,
    lambda_reg: float,
    K_test_train: np.ndarray = None,
) -> Tuple[np.ndarray, float, float]:
    """带正确截距处理的核岭回归。
    
    按照优化方案公式：
    b_hat = 1^T @ C^{-1} @ y / (1^T @ C^{-1} @ 1)
    alpha_hat = C^{-1} @ (y - b_hat * 1)
    y_test = b_hat + K_test @ alpha
    
    其中 C = K + lambda * I
    
    参数:
        K_train: 训练核矩阵 (已中心化，n_train x n_train)
        y_train: 训练标签 (n_train,)
        lambda_reg: 正则化参数
        K_test_train: 测试-训练交叉核 (可选，n_test x n_train)
    
    返回:
        alpha: 系数向量
        intercept: 截距
        y_pred_test: 测试预测 (如果提供K_test_train)
    """
    n = K_train.shape[0]
    C = K_train + lambda_reg * np.eye(n)
    
    # 求解 C^{-1} @ y 和 C^{-1} @ 1
    C_inv_y = np.linalg.solve(C, y_train)
    C_inv_1 = np.linalg.solve(C, np.ones(n))
    
    # 计算截距: b = 1^T @ C^{-1} @ y / (1^T @ C^{-1} @ 1)
    intercept = np.sum(C_inv_y) / np.sum(C_inv_1)
    
    # 计算系数: alpha = C^{-1} @ (y - b * 1)
    alpha = C_inv_y - intercept * C_inv_1
    
    if K_test_train is not None:
        # 预测: y_test = b + K_test @ alpha
        y_pred_test = intercept + K_test_train @ alpha
        return alpha, intercept, y_pred_test
    else:
        return alpha, intercept, None

def feature_kernel_centered_scaled(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从特征矩阵构建中心化且尺度归一的线性核。
    
    参数:
        X_train: 训练特征 (n_train x p)
        X_test: 测试特征 (可选，n_test x p)
        epsilon: 标准差阈值
    
    返回:
        K_train: 训练核矩阵
        X_train_scaled: 标准化后的训练特征
        mean_train: 训练端特征均值
        std_train: 训练端特征标准差
    """
    # 训练端标准化
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)
    std_train[std_train < epsilon] = 1.0  # 避免除零
    
    X_train_scaled = (X_train - mean_train) / std_train
    
    # 线性核: K = X @ X^T / p
    p = X_train_scaled.shape[1]
    K_train = X_train_scaled @ X_train_scaled.T / p
    
    if X_test is not None:
        # 测试端使用训练端参数标准化
        X_test_scaled = (X_test - mean_train) / std_train
        K_test_train = X_test_scaled @ X_train_scaled.T / p
        return K_train, X_train_scaled, mean_train, std_train, K_test_train, X_test_scaled
    else:
        return K_train, X_train_scaled, mean_train, std_train

def snp_kernel_corrected(
    X_snp_train: np.ndarray,
    X_snp_test: np.ndarray = None
) -> Tuple:
    """修正后的SNP核构建。
    
    与feature_kernel_centered_scaled相同，但专门用于SNP数据。
    确保SNP核经过正确的中心化和尺度归一。
    """
    return feature_kernel_centered_scaled(X_snp_train, X_snp_test)
