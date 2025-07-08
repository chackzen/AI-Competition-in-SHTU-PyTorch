#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch版本的模型训练函数 - 优化版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os
import numpy as np
from tqdm import tqdm
import sys

# 添加src目录到路径
sys.path.append(os.path.dirname(__file__))

from data_preprocess import (
    load_csv, get_training_dataset, get_validation_dataset, 
    IMAGE_SIZE, NUM_CLASSES, SEED, BATCH_SIZE, STEPS_PER_EPOCH
)

# 模型配置
MODEL_CONFIGS = {
    'efficientnet': {
        'model_name': 'efficientnet_b7',
        'feature_dim': 2560
    },
    'densenet': {
        'model_name': 'densenet201',
        'feature_dim': 1920
    }
}

class CustomModel(nn.Module):
    """通用的自定义模型类"""
    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super().__init__()
        
        # 动态导入模型
        model_config = MODEL_CONFIGS[model_name]
        
        if model_name == 'efficientnet':
            # EfficientNet B7
            from torchvision.models import efficientnet_b7
            self.backbone = efficientnet_b7(pretrained=False)
            self.backbone.classifier = nn.Linear(model_config['feature_dim'], num_classes)
        elif model_name == 'densenet':
            # DenseNet 201
            from torchvision.models import densenet201
            self.backbone = densenet201(pretrained=False)
            self.backbone.classifier = nn.Linear(model_config['feature_dim'], num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
    def forward(self, x):
        return self.backbone(x)

def get_model(model_name):
    """获取模型"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = CustomModel(model_name)
    
    # 加载预训练权重（如果存在）
    weight_path = os.path.join(os.getcwd(), 'models', f'{model_name}.pth')
    if os.path.exists(weight_path):
        print(f"加载权重: {weight_path}")
        model.load_state_dict(torch.load(weight_path))
    
    return model

class F1Score:
    """F1分数计算器"""
    def __init__(self, num_classes=NUM_CLASSES, average='macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        # 转换为类别索引
        if y_true.dim() > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        
        self.y_true.extend(y_true.cpu().numpy())
        self.y_pred.extend(y_pred.cpu().numpy())
    
    def compute(self):
        from sklearn.metrics import f1_score as sklearn_f1_score
        return sklearn_f1_score(self.y_true, self.y_pred, average=self.average)

def train_epoch(model, train_loader, criterion, optimizer, device, f1_calculator):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    f1_calculator.reset()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        f1_calculator.update(labels, outputs)
    
    avg_loss = total_loss / len(train_loader)
    f1_score = f1_calculator.compute()
    
    return avg_loss, f1_score

def validate_epoch(model, val_loader, criterion, device, f1_calculator):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    f1_calculator.reset()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            f1_calculator.update(labels, outputs)
    
    avg_loss = total_loss / len(val_loader)
    f1_score = f1_calculator.compute()
    
    return avg_loss, f1_score

def setup_training_components(model, model_name, print_summary=False):
    """
    设置训练组件（优化器、损失函数、评估指标等）
    """
    # 打印模型结构
    if print_summary:
        print(f"\nModel ({model_name}) architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # 设置优化器、损失函数和评估指标
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    f1_calculator = F1Score(NUM_CLASSES, average='macro')
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    return optimizer, criterion, f1_calculator, scheduler

def train_and_save_model(model, model_name, fold, train_loader, val_loader, device, epochs, patience):
    """
    训练模型并保存最佳检查点
    """
    optimizer, criterion, f1_calculator, scheduler = setup_training_components(model, model_name)
    
    # 训练状态跟踪
    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = f'checkpoints/{model_name}_fold_{fold}_best.pth'
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练阶段
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, f1_calculator
        )
        
        # 验证阶段
        val_loss, val_f1 = validate_epoch(
            model, val_loader, criterion, device, f1_calculator
        )
        
        # 学习率调度
        scheduler.step(val_f1)
        
        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 模型保存和早停
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"✓ New best {model_name} model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"Early stopping triggered for {model_name} at epoch {epoch + 1}")
            break
    
    return checkpoint_path

def load_and_fit_models2(model_name1, model_name2, epochs=50, patience=8, print_summary=False):
    """
    同时训练两个不同的模型并进行K折交叉验证
    
    Args:
        model_name1: 第一个模型名称 ('efficientnet' 或 'densenet')
        model_name2: 第二个模型名称 ('efficientnet' 或 'densenet')
        epochs: 训练轮数
        patience: 早停耐心值
        print_summary: 是否打印模型结构
    
    Returns:
        tuple: (models1, models2) 两个训练好的模型列表
    """
    # 加载数据
    train_df, test_df, label_to_idx, idx_to_label = load_csv()
    models1 = []
    models2 = []
    
    # K折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):
        print(f"\n{'='*30}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*30}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 创建第一个模型
        if model_name1 not in MODEL_CONFIGS:
            raise ValueError(f"不支持的模型1: {model_name1}")
        
        model1 = get_model(model_name1)
        model1 = model1.to(device)
        
        # 创建第二个模型
        if model_name2 not in MODEL_CONFIGS:
            raise ValueError(f"不支持的模型2: {model_name2}")
        
        model2 = get_model(model_name2)
        model2 = model2.to(device)
        
        # 打印模型结构
        if print_summary:
            setup_training_components(model1, model_name1, print_summary=True)
            setup_training_components(model2, model_name2, print_summary=True)
        
        # 创建数据加载器
        train_loader = get_training_dataset(train_df.iloc[train_indices], do_onehot=True)
        val_loader = get_validation_dataset(train_df.iloc[val_indices], do_onehot=True)
        
        # 创建检查点目录
        os.makedirs('checkpoints', exist_ok=True)
        
        # 训练第一个模型
        print(f"\nTraining {model_name1}...")
        checkpoint_path1 = train_and_save_model(
            model1, model_name1, fold, train_loader, val_loader, device, epochs, patience
        )
        
        # 训练第二个模型
        print(f"\nTraining {model_name2}...")
        checkpoint_path2 = train_and_save_model(
            model2, model_name2, fold, train_loader, val_loader, device, epochs, patience
        )
        
        # 加载最佳模型
        checkpoint1 = torch.load(checkpoint_path1)
        checkpoint2 = torch.load(checkpoint_path2)
        
        model1.load_state_dict(checkpoint1['model_state_dict'])
        model2.load_state_dict(checkpoint2['model_state_dict'])
        
        print(f"Loaded best {model_name1} from epoch {checkpoint1['epoch'] + 1} with F1: {checkpoint1['best_f1']:.4f}")
        print(f"Loaded best {model_name2} from epoch {checkpoint2['epoch'] + 1} with F1: {checkpoint2['best_f1']:.4f}")
        
        models1.append(model1)
        models2.append(model2)
    
    print(f"\nTraining completed! Trained {len(models1)} {model_name1} models and {len(models2)} {model_name2} models.")
    return models1, models2

# 使用示例
if __name__ == '__main__':
    # 训练两个模型用于集成
    print("Training both models for ensemble...")
    efficientnet_models, densenet_models = load_and_fit_models2(
        'efficientnet', 'densenet', epochs=30, patience=5
    ) 