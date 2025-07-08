# [第一届“慧炼丹心杯”AI挑战赛] - 蝴蝶图像的多分类任务
  
> 这是上海科技大学首届“慧炼丹心杯”AI挑战赛中，拔得头筹的 弼马温认证炼丹团队 提出的方案的PyTorch版本。本次比赛的主题是利用机器学习技术完成蝴蝶图像的多分类任务。
> 原版请参考https://github.com/SantinorDeng/AI-Competition-in-SHTU

## ✨ 功能特性
- 支持多种主流模型（EfficientNet, DenseNet等）的ensemble
- 包括旋转、剪切、缩放、平移和随机擦除等多种数据增强手段
- 模型评估指标计算、

## 📦 环境安装
```bash
# 克隆项目
git clone https://github.com/your_username/project_name.git

# 创建虚拟环境
conda create -n AI-Competition-in-SHTU-PyTorch python=3.8
conda activate AI-Competition-in-SHTU-PyTorch    # Windows

# 安装依赖
cd AI-Competition-in-SHTU-PyTorch
pip install -r requirements.txt
```

- 下载预训练权重  
EfficientNet: https://objects.githubusercontent.com/github-production-release-asset-2e65be/189350661/b8864d80-5b42-11ea-9927-ed5ee216172a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250708%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250708T090709Z&X-Amz-Expires=1800&X-Amz-Signature=0b87e0b7589886e381f0255e4412027c5ad0ca9d7f0ce6ada5952366965fcc92&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Defficientnet-b7-dcc49843.pth&response-content-type=application%2Foctet-stream

DenseNet: https://download.pytorch.org/models/densenet201-c1103571.pth
