import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import tifffile as tiff
from collections import OrderedDict
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from datasets.CustomDataset import TyphoonDataset


def get_dataloaders(csv_file, root_dir, batch_size=8, crop_size=300, image_size=400):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = TyphoonDataset(csv_file, root_dir=root_dir, transform=transform, crop_size=crop_size, image_size=image_size)

    # 학습 및 검증에 사용할 연도 정의
    train_years = [2019, 2020, 2021, 2022]
    val_years = [2023, 2024]


    # 학습 데이터와 검증 데이터를 연도별로 필터링
    train_indices = dataset.data[dataset.data['year'].isin(train_years)].index.tolist()
    val_indices = dataset.data[dataset.data['year'].isin(val_years)].index.tolist()

    # 학습용 데이터셋 (어그멘테이션 적용)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_dataset.dataset.augment = True  # 학습 데이터에 어그멘테이션 적용

    # 검증용 데이터셋 (어그멘테이션 미적용)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_dataset.dataset.augment = False  # 검증 데이터에 어그멘테이션 미적용

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def get_predicted_centers(heatmap):
    """
    히트맵에서 가장 높은 확률을 가지는 점의 좌표를 반환합니다.

    Args:
        heatmap (torch.Tensor): (batch_size, 1, H, W) 형태의 히트맵 텐서

    Returns:
        torch.Tensor: (batch_size, 2) 형태의 (x, y) 좌표 텐서
    """
    batch_size, _, H, W = heatmap.size()
    heatmap = heatmap.view(batch_size, -1)
    _, max_indices = torch.max(heatmap, dim=1)

    y = (max_indices // W).float()
    x = (max_indices % W).float()

    return torch.stack([x, y], dim=1)

def gaussian_heatmap(center, size=21, sigma=5):
    x = torch.arange(0, size, 1).float()
    y = x.unsqueeze(1)
    x0 = y0 = size // 2
    gauss = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gauss

def center_distance_loss(pred_heatmap, true_heatmap):
    # 기존의 MSE 손실
    mse_loss = nn.MSELoss()(pred_heatmap, true_heatmap)
    
    # 추가적인 중심 손실 (선택적)
    # 중심 좌표를 직접 계산하지 않고 히트맵 자체의 특성을 활용
    return mse_loss

def train_model(csv_file, root_dir, num_epochs=300, batch_size=16, learning_rate=1e-2, device='cuda', crop_size=300, image_size=400):
    # 디바이스 확인
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("gpu")
    else:
        device = torch.device('cpu')
        print("cpu")
    
    # 데이터 로더 준비
    train_loader, val_loader = get_dataloaders(csv_file, root_dir=root_dir, batch_size=batch_size, crop_size=crop_size, image_size=image_size)

    # 모델, 손실 함수, 옵티마이저
    model = UNet(in_channels=6, out_channels=1).to(device)
    print(f"model device: {next(model.parameters()).device}")
    
    # 손실 함수를 MSELoss로 변경
    criterion = nn.MSELoss()
    print(f"loss function: {criterion}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 코사인 어닐링 스케줄러 추가
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # outputs: (batch_size, 1, H, W)
            total_loss = center_distance_loss(outputs, labels)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
        
        # 스케줄러 스텝 업데이트
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} avgloss: {avg_loss:.4f}, lr: {current_lr:.6f}")
        
        # 검증 단계
        model.eval()
        val_loss = 0
        total_distance = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # outputs: (batch_size, 1, H, W)
                
                # 손실 계산
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 예측된 중심 좌표 추출
                pred_centers = get_predicted_centers(outputs)  # 수정된 함수 사용
                
                # 실제 중심 좌표 추출
                true_centers = get_predicted_centers(labels)
                
                # 유클리드 거리 계산
                distances = torch.sqrt(torch.sum((pred_centers - true_centers) ** 2, dim=1))  # (batch_size,)
                total_distance += distances.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_distance = total_distance / len(val_loader.dataset)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation MAE: {avg_distance:.4f} pixels")
    
    # 모델 저장
    torch.save(model.state_dict(), "unet_typhoon.pth")
    print("saved: unet_typhoon.pth")
if __name__ == "__main__":
    csv_path = "datasets/data/sampled_data/metadata.csv"
    root_directory = "datasets/data/sampled_data"  # 실제 이미지가 저장된 디렉토리 경로로 변경
    dataset = TyphoonDataset(csv_file=csv_path, root_dir=root_directory)
    stats = dataset.analyze_dataset()


