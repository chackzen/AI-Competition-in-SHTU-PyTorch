#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch版本的预测函数
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from data_preprocess import (
    IMAGE_SIZE, BATCH_SIZE, DATA_PATH
)

class TestImageDataset(Dataset):
    """测试图像数据集"""
    def __init__(self, filenames):
        self.filenames = filenames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # 构建完整路径
        image_path = os.path.join(DATA_PATH, 'test_images', self.filenames[idx])
        
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        
        # 调整大小
        image = image.resize(IMAGE_SIZE)
        
        # 转换为numpy数组并归一化
        image = np.array(image, dtype=np.float32) / 255.0
        
        # 转换为tensor并调整维度 (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image

def get_test_dataset(filenames, ordered=False, augmented=False):
    """
    获取测试数据集
    
    Args:
        filenames: 文件名列表
        ordered: 是否保持顺序
        augmented: 是否使用数据增强
    
    Returns:
        DataLoader: PyTorch数据加载器
    """
    # 创建数据集
    dataset = TestImageDataset(filenames)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=not ordered,  # 如果不要求有序，则打乱数据
        num_workers=4,  # 多进程加载数据
        pin_memory=True  # 将数据加载到GPU内存
    )
    
    return dataloader

def predict(dataset, model):
    """
    使用模型进行预测
    
    Args:
        dataset: 测试数据集
        model: 训练好的模型
    
    Returns:
        numpy.ndarray: 预测结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print('Calculating predictions...')
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Predicting"):
            batch = batch.to(device)
            
            # 前向传播
            outputs = model(batch)
            
            # 获取预测结果
            preds = torch.softmax(outputs, dim=1)
            predictions.append(preds.cpu().numpy())
    
    # 合并所有预测结果
    predictions = np.concatenate(predictions, axis=0)
    
    return predictions

# 使用示例
if __name__ == '__main__':
    # 示例用法
    test_filenames = pd.read_csv(os.path.join(DATA_PATH, 'data', 'test.csv'))['filename'].tolist()
    
    # 获取测试数据集
    test_loader = get_test_dataset(test_filenames)
    
    # 假设已经训练好的模型
    # model = load_trained_model()
    
    # 进行预测
    # predictions = predict(test_loader, model)
    # print(f"Predictions shape: {predictions.shape}") 