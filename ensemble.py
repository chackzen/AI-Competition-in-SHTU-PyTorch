#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch版本的模型集成和提交文件生成
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from data_preprocess import (
    load_csv, DATA_PATH, NUM_CLASSES, IMAGE_SIZE
)
from train import get_model
from predict import get_test_dataset, predict

def generate_submission(alpha=0.46):
    """
    生成提交文件
    
    Args:
        alpha: EfficientNet和DenseNet的权重比例
    """
    SUBMISSION_FILE = os.path.join(DATA_PATH, 'submission.csv')
    print('Generating submission file...')
    
    # 加载数据
    train_df, test_df, label_to_idx, idx_to_label = load_csv()
    
    possibility = []
    
    # 对每个fold进行集成
    for i in range(5):
        print(f"Processing fold {i+1}/5...")
        
        # 加载EfficientNet模型
        efficientnet_model = get_model('efficientnet')
        checkpoint_path = os.path.join(DATA_PATH, f'checkpoints/efficientnet_fold_{i}_best.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded EfficientNet fold {i} with F1: {checkpoint['best_f1']:.4f}")
        else:
            print(f"Warning: EfficientNet checkpoint for fold {i} not found")
            continue
        
        # 获取测试数据集
        test_loader = get_test_dataset(test_df['filename'], ordered=True, augmented=False)
        
        # EfficientNet预测
        preds_m1 = predict(test_loader, efficientnet_model)
        
        # 加载DenseNet模型
        densenet_model = get_model('densenet')
        checkpoint_path = os.path.join(DATA_PATH, f'checkpoints/densenet_fold_{i}_best.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            densenet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded DenseNet fold {i} with F1: {checkpoint['best_f1']:.4f}")
        else:
            print(f"Warning: DenseNet checkpoint for fold {i} not found")
            continue
        
        # DenseNet预测
        preds_m2 = predict(test_loader, densenet_model)
        
        # 加权集成
        preds_mix = alpha * preds_m1 + (1 - alpha) * preds_m2
        possibility.append(preds_mix)
    
    # 对所有fold的结果取平均
    if possibility:
        preds = np.mean(possibility, axis=0)
        preds = np.argmax(preds, axis=1)
        pred_labels = [idx_to_label[pred] for pred in preds]
        
        # 生成提交文件
        predictions = []
        for filename, label in zip(test_df['filename'], pred_labels):
            predictions.append((filename, label))
        
        submission_df = pd.DataFrame(predictions, columns=['filename', 'label'])
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        
        print(f"Submission file generated: {SUBMISSION_FILE}")
        print(f"Total predictions: {len(predictions)}")
    else:
        print("Error: No valid predictions generated")

# 使用示例
if __name__ == '__main__':
    # 生成提交文件
    generate_submission(alpha=0.46) 