import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import torch.nn.functional as F
import random

DATA_PATH = os.getcwd()
TRAINING_FILENAMES = glob.glob(os.path.join(DATA_PATH, 'data', 'train_images', '*.jpg'))
TEST_FILENAMES = glob.glob(os.path.join(DATA_PATH, 'data', 'test_images', '*.jpg'))
TRAIN_CSV = os.path.join(DATA_PATH, 'data', 'train.csv')
TEST_CSV = os.path.join(DATA_PATH, 'data', 'test.csv')
SEED = 77
IMAGE_SIZE = [224, 224]

# PyTorch分布式训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE = 32 * max(1, num_gpus)  # 如果有GPU则乘以GPU数量，否则为32

def load_csv():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    labels = train_df['label'].unique().tolist()
    label_to_idx = {label: idx for idx, label in enumerate(labels)} # 用序号表示标签
    idx_to_label = {idx: label for label, idx in label_to_idx.items()} # 用标签表示序号
    return train_df, test_df, label_to_idx, idx_to_label

train_df, test_df, label_to_idx, idx_to_label = load_csv()

# 划分训练集和验证集
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df['label'])
NUM_TRAINING_IMAGES = train_df.shape[0]
NUM_VALIDATION_IMAGES = val_df.shape[0]
NUM_TEST_IMAGES = test_df.shape[0]
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
NUM_CLASSES = len(label_to_idx)
print(f"Number of training images: {NUM_TRAINING_IMAGES}")
print(f"Number of validation images: {NUM_VALIDATION_IMAGES}")
print(f"Number of test images: {NUM_TEST_IMAGES}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Device: {device}")
print(f"Number of GPUs: {num_gpus}")
print(f"Batch size: {BATCH_SIZE}")

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = torch.cos(torch.tensor(rotation, dtype=torch.float32))
    s1 = torch.sin(torch.tensor(rotation, dtype=torch.float32))
    one = torch.tensor([1.0], dtype=torch.float32)
    zero = torch.tensor([0.0], dtype=torch.float32)
    rotation_matrix = torch.tensor([[c1, s1, zero], 
                                   [-s1, c1, zero], 
                                   [zero, zero, one]], dtype=torch.float32)
        
    # SHEAR MATRIX
    c2 = torch.cos(torch.tensor(shear, dtype=torch.float32))
    s2 = torch.sin(torch.tensor(shear, dtype=torch.float32))
    shear_matrix = torch.tensor([[one, s2, zero], 
                                [zero, c2, zero], 
                                [zero, zero, one]], dtype=torch.float32)    
    
    # ZOOM MATRIX
    zoom_matrix = torch.tensor([[1.0/height_zoom, zero, zero], 
                               [zero, 1.0/width_zoom, zero], 
                               [zero, zero, one]], dtype=torch.float32)
    
    # SHIFT MATRIX
    shift_matrix = torch.tensor([[one, zero, height_shift], 
                                [zero, one, width_shift], 
                                [zero, zero, one]], dtype=torch.float32)
    
    # 矩阵乘法：rotation_matrix * shear_matrix * zoom_matrix * shift_matrix
    temp = torch.mm(rotation_matrix, shear_matrix)
    temp2 = torch.mm(zoom_matrix, shift_matrix)
    return torch.mm(temp, temp2)

def transform(image, label):
    # input image - is one image of size [C, H, W] not a batch of [B, C, H, W]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331
    
    # 生成随机变换参数
    rot = 15. * torch.randn(1, dtype=torch.float32)
    shr = 5. * torch.randn(1, dtype=torch.float32)
    h_zoom = 1.0 + torch.randn(1, dtype=torch.float32) / 10.
    w_zoom = 1.0 + torch.randn(1, dtype=torch.float32) / 10.
    h_shift = 16. * torch.randn(1, dtype=torch.float32)
    w_shift = 16. * torch.randn(1, dtype=torch.float32)
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot.item(), shr.item(), h_zoom.item(), w_zoom.item(), h_shift.item(), w_shift.item())

    # LIST DESTINATION PIXEL INDICES
    x = torch.repeat_interleave(torch.arange(DIM//2, -DIM//2, -1), DIM) # [112,112,112,..., 111,111,111,..., -112,-112,-112]
    y = torch.tile(torch.arange(-DIM//2, DIM//2), (DIM,)) # [-112,-111,...,112, -112,-111,...,112, ..., -112,-111,...,112]
    z = torch.ones(DIM*DIM, dtype=torch.int32) # [1, 1, 1, ..., 1]（112*112个1）
    idx = torch.stack([x, y, z])
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = torch.mm(m, idx.float())
    idx2 = idx2.int()
    idx2 = torch.clamp(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = torch.stack([DIM//2 - idx2[0, :], DIM//2 - 1 + idx2[1, :]])
    
    # 确保图像是PyTorch张量格式 [C, H, W]
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    # 使用grid_sample进行采样
    # 创建归一化坐标网格
    grid_x = (idx2[0, :].float() / (DIM//2)).reshape(DIM, DIM)
    grid_y = (idx2[1, :].float() / (DIM//2)).reshape(DIM, DIM)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    
    # 图像已经是 [C, H, W] 格式，添加batch维度 [1, C, H, W]
    image_batch = image.unsqueeze(0)
    
    # 使用grid_sample进行采样
    d = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)
    d = d.squeeze(0)  # 移除batch维度，保持 [C, H, W] 格式
    
    # 随机决定是否应用变换
    if torch.rand(1) > 0.8:
        return image, label
    else:
        return d, label

class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transform=None, augment=False, do_onehot=False):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.do_onehot = do_onehot
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = os.path.join(DATA_PATH, 'data', 'train_images', self.filenames[idx])
        image = Image.open(image_path).convert('RGB')
        
        # 调整图像大小
        image = image.resize(IMAGE_SIZE)
        
        # 转换为张量并归一化
        image = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        
        # 转换为PyTorch格式 (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            image, label = self.transform(image, label)
        
        # 应用数据增强
        if self.augment:
            image, label = data_augment(image, label)
        
        # 转换为one-hot编码
        if self.do_onehot:
            label = F.one_hot(torch.tensor(label, dtype=torch.long), NUM_CLASSES).float()
        
        return image, label

def random_erasing(image, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
    """随机擦除数据增强"""
    if random.random() > p:
        return image
    
    # 确保图像是PyTorch张量
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    # 获取图像尺寸 (C, H, W)
    c, h, w = image.shape
    
    # 随机选择擦除区域
    area = h * w
    target_area = random.uniform(scale[0], scale[1]) * area
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    
    # 计算擦除区域的宽高
    erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
    erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
    
    # 确保擦除区域在图像范围内
    if erase_h >= h or erase_w >= w:
        return image
    
    # 随机选择擦除区域的左上角坐标
    i = random.randint(0, h - erase_h)
    j = random.randint(0, w - erase_w)
    
    # 执行擦除 (C, H, W) 格式
    image[:, i:i+erase_h, j:j+erase_w] = value
    
    return image

def data_augment(image, label):
    # 随机水平翻转
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])  # 水平翻转
    
    # 随机擦除
    image = random_erasing(image)
    
    return image, label

def data_augment_alpha(image, label):
    # 随机水平翻转
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])  # 水平翻转
    
    # 随机擦除（概率0.7）
    image = random_erasing(image, p=0.7)
    
    return image, label

def get_training_dataset(data, do_onehot=False):
    # 准备文件名和标签
    filenames = data['filename'].tolist()
    labels = [label_to_idx[label] for label in data['label']]
    
    # 创建数据集
    dataset = ImageDataset(filenames, labels, transform=transform, augment=True, do_onehot=do_onehot)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows上设置为0避免多进程问题
        pin_memory=True
    )
    
    return dataloader

def get_validation_dataset(data, ordered=False, do_onehot=False):
    # 准备文件名和标签
    filenames = data['filename'].tolist()
    labels = [label_to_idx[label] for label in data['label']]
    
    # 创建数据集
    dataset = ImageDataset(filenames, labels, transform=None, augment=False, do_onehot=do_onehot)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=not ordered,
        num_workers=0,  # Windows上设置为0避免多进程问题
        pin_memory=True
    )
    
    return dataloader

def get_validation_dataset_alpha(data, ordered=False, do_onehot=False):
    # 准备文件名和标签
    filenames = data['filename'].tolist()
    labels = [label_to_idx[label] for label in data['label']]
    
    # 创建数据集
    dataset = ImageDataset(filenames, labels, transform=None, augment=True, do_onehot=do_onehot)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=not ordered,
        num_workers=0,  # Windows上设置为0避免多进程问题
        pin_memory=True
    )
    
    return dataloader

if __name__ == '__main__':
    print("Training data shapes:")
    train_loader = get_training_dataset(train_df)
    for i, (image, label) in enumerate(train_loader):
        if i >= 3:  # 只取前3个batch
            break
        print(image.shape, label.shape)
    print("Training data label examples:", label)

    print("Validation data shapes:")
    val_loader = get_validation_dataset(val_df)
    for i, (image, label) in enumerate(val_loader):
        if i >= 3:  # 只取前3个batch
            break
        print(image.shape, label.shape)
    print("Validation data label examples:", label)

