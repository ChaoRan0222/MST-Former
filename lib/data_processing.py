"""数据处理转换模块"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


def seq2instance(data: np.ndarray, P: int, Q: int) -> Tuple[np.ndarray, np.ndarray]:

    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1

    x = np.lib.stride_tricks.as_strided(
        data,
        shape=(num_sample, P, nodes, dims),
        strides=(data.strides[0], data.strides[0], data.strides[1], data.strides[2])
    ).copy()

    y = np.lib.stride_tricks.as_strided(
        data[Q:],
        shape=(num_sample, P, nodes, dims),
        strides=(data.strides[0], data.strides[0], data.strides[1], data.strides[2])
    ).copy()

    return x, y


def read_meta(path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """读取元数据文件
    
    Args:
        path: 元数据文件路径
        
    Returns:
        locations: 位置数据
        meta_df: 元数据DataFrame
    """
    meta_df = pd.read_csv(path)
    lat = meta_df['Lat'].values
    lng = meta_df['Lng'].values
    locations = np.stack([lat, lng], axis=0)
    return locations, meta_df


def construct_adj(data: np.ndarray, num_node: int) -> np.ndarray:
    """构建邻接矩阵
    
    Args:
        data: 输入数据
        num_node: 节点数量
        
    Returns:
        邻接矩阵
    """
    from .constants import Constants
    chunk_size = Constants.HOURS_PER_DAY * Constants.SAMPLES_PER_HOUR
    data_mean = np.mean([data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix 