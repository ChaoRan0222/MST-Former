"""数据集加载和预处理模块"""

import os
import numpy as np
from typing import Tuple
from .data_processing import seq2instance, read_meta, construct_adj
from .road_network import roadNetworkPartition, reorderData
from datetime import datetime


def loadData(filepath: str, metapath: str, P: int, Q: int, train_ratio: float,
            test_ratio: float, adjpath: str, recurtimes: int, tod: int,
            dow: int, sps: int, log) -> Tuple:

    # 参数验证
    if not os.path.exists(filepath) or not os.path.exists(metapath):
        raise FileNotFoundError("数据文件不存在")
    if not 0 <= train_ratio + test_ratio <= 1:
        raise ValueError("无效的训练/测试比例")

    # 加载数据
    Traffic = np.load(filepath)['data'][..., :1]
    locations, meta_df = read_meta(metapath)
    num_step = Traffic.shape[0]

    # 生成时间编码
    TE = np.zeros([num_step, 2])
    TE[:, 0] = np.array([i % tod for i in range(num_step)])
    TE[:, 1] = np.array([(i // tod) % dow for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)

    log_string(log, f'数据形状: {Traffic.shape}')
    log_string(log, f'位置形状: {locations.shape}')

    # 分割数据集
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    trainData = Traffic[: train_steps]
    valData = Traffic[train_steps: train_steps + val_steps]
    testData = Traffic[-test_steps:]

    trainTE = TE_tile[: train_steps]
    valTE = TE_tile[train_steps: train_steps + val_steps]
    testTE = TE_tile[-test_steps:]

    # 生成邻接矩阵
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData, locations.shape[1])
        np.save(adjpath, adj)

    # 路网分割
    partition_recur_depth = recurtimes  # 使用新的参数命名
    spatial_patch_size = sps  # 使用新的参数命名
    
    parts_idx, mxlen = roadNetworkPartition(locations, partition_recur_depth, meta_df)
    print(f"分区递归深度(partition_recur_depth): {partition_recur_depth}")
    print(f"分区数量: {2**partition_recur_depth}")
    print(f"每个分区的节点数量: {[len(part) for part in parts_idx]}")
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, spatial_patch_size)

    # 生成数据实例
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)

    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)

    # 计算统计量
    mean, std = np.mean(trainX), np.std(trainX)

    # 记录数据集信息
    log_string(log, f'训练数据形状: {trainY.shape}')
    log_string(log, f'验证数据形状: {valY.shape}')
    log_string(log, f'测试集形状: {testY.shape}')
    log_string(log, f'均值: {mean} & 标准差: {std}')

    return (trainX, trainY, trainXTE, trainYTE, valX, valY, valXTE, valYTE,
            testX, testY, testXTE, testYTE, mean, std,
            ori_parts_idx, reo_parts_idx, reo_all_idx)


def log_string(log, string: str) -> None:
    """记录日志到文件和控制台，并添加时间戳
    
    Args:
        log: 日志对象
        string: 要记录的字符串
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_string = f'[{current_time}] {string}'
    log.write(formatted_string + '\n')
    log.flush()
    print(formatted_string) 