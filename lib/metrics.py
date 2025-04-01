"""评估指标相关函数"""

import numpy as np
import torch
from typing import Tuple


def metric(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return float(mae), float(rmse), float(mape)


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """计算带掩码的MAE损失
    
    Args:
        preds: 预测值
        labels: 真实值
        null_val: 空值标记
        
    Returns:
        损失值
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def _compute_loss(y_true: torch.Tensor, y_predicted: torch.Tensor) -> torch.Tensor:
    """计算损失函数
    
    Args:
        y_true: 真实值
        y_predicted: 预测值
        
    Returns:
        损失值
    """
    return masked_mae(y_predicted, y_true, 0.0) 