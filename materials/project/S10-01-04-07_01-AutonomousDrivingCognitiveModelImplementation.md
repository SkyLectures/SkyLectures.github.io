---
layout: page
title:  "자율주행 인지 모델 구현"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-07_01-AutonomousDrivingCognitiveModelImplementation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/project/notebooks/S10-01-04-07_01-AutonomousDrivingCognitiveModelImplementation.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>


> - **딥러닝 기반 자율주행 인지 모델**
>   - 자율주행 자동차의 인지 시스템은 **차량이 주변 환경을 이해하고 상황을 파악**하는 핵심 요소
{: .common-quote}

## 1. 자율주행 인지 모델 구현 실습 개요

- **실습 내용**
    - 카메라 영상을 입력으로 받아 세 가지 핵심 인지 기능을 동시에 수행하는 통합 인지 모델 구현
        1. **차선 인식**: 도로 위 차선을 감지하여 주행 가능한 영역을 파악
        2. **객체 탐지**: 차량, 보행자, 자전거 등 도로 위의 다양한 객체를 감지
        3. **도로 상태 이해**: 도로 표면, 주행 가능 영역 등을 세그멘테이션하여 파악
    - 세 가지 기능을 하나의 통합 모델로 구현하는 멀티태스크 학습 접근법 사용

- **데이터셋**
    - 자율주행 인지 모델 학습을 위한 대표적인 데이터셋: BDD100K, Cityscapes, KITTI 등
        - 실습에서는 Berkeley DeepDrive Dataset(BDD100K)을 기준으로 진행
            - [BDD100K 데이터셋 다운로드](https://bdd-data.berkeley.edu/){: target="_blank"}

> - 국내 환경에서 사용하기 위해서는 국내용으로 개발된 데이터셋을 찾아서(또는 직접 구축) 적용해야 함
{: .expert-quote}
        

## 2. PyTorch 기반 구현

### 2.1 패키지 가져오기

```python
import os
import random
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
```

### 2.2 기반 환경 설정

- 데이터셋 및 사전학습모델 경로 설정

```python
BDD10K_DATA_ROOT_PATH = "/content/bdd10k"
PRETRAINED_MODEL_PATH = "/content/pretrained/upernet_r50-d8_769x769_40k_sem_seg_bdd100k.pth"

CLASS_LABELS = {
    "background": 0,
    "drivable": 1,
    "lane": 2,
    # "road_line": 2,
    # "other_line": 3
}

NUM_CLASSES = len(CLASS_LABELS)
INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH = 769, 769

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2.3 구성요소 구현

#### 2.3.1 BDD10K 데이터셋 로더

```python
def get_bdd10k_segmentation_paths(split='val'):
    if split == 'train':
        image_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'train')
        drivable_mask_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'labels', 'drivable_maps', '10k', 'train')
        lane_mask_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'labels', 'lane_masks', '10k', 'train')

    elif split == 'val':
        image_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'val')
        drivable_mask_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'labels', 'drivable_maps', '10k', 'val')
        lane_mask_dir = os.path.join(BDD10K_DATA_ROOT_PATH, 'labels', 'lane_masks', '10k', 'val')

    else:
        raise ValueError(f"지원하지 않는 split: {split}. 'train', 'val' 중 하나여야 합니다.")

    samples = []

    # 이미지 파일들을 기준으로 마스크를 찾음
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            base_name = os.path.splitext(image_name)[0]

            image_path = os.path.join(image_dir, image_name)
            drivable_mask_path = os.path.join(drivable_mask_dir, base_name + '.png')
            lane_mask_path = os.path.join(lane_mask_dir, base_name + '.png')

            # 모든 파일이 존재하는지 확인 (필수)
            if os.path.exists(image_path) and os.path.exists(drivable_mask_path) and os.path.exists(lane_mask_path):
                samples.append({
                    'image_path': image_path,
                    'drivable_mask_path': drivable_mask_path,
                    'lane_mask_path': lane_mask_path,
                    'image_name': image_name
                })

            # else:
            #     print(f"누락된 파일: {image_name} 관련 마스크 파일이 없습니다.")

    # 안정적인 학습/추론을 위해 리스트를 정렬
    samples = sorted(samples, key=lambda x: x['image_name'])
    return samples

def get_image_paths_from_dir(target_dir):
    """
    지정된 디렉토리에서 모든 JPG 이미지 파일의 경로를 가져옵니다.
    """
    if not os.path.exists(target_dir):
        print(f"CRITICAL ERROR: Directory does not exist: {target_dir}")
        return []

    image_paths = []
    for fname in os.listdir(target_dir):
        if fname.lower().endswith(('.jpg', '.jpeg')):
            image_paths.append(os.path.join(target_dir, fname))

    if not image_paths:
        print(f"WARNING: No JPG/JPEG images found in {target_dir}. Check directory content or file extensions.")
        return []

    # 랜덤 선택을 위해 이미지 이름을 기준으로 정렬은 불필요하지만 일관성을 위해 유지
    return sorted(image_paths)    

# BDD10KSegmentationDataset 클래스는 학습시에만 사용되므로 주석 처리하거나 제거 가능
class BDD10KSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, split='val', transform=None, target_transform=None):
        self.samples = get_bdd10k_segmentation_paths(split)
        self.transform = transform
        self.target_transform = target_transform # 마스크 변환용

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # 이미지 로드 (RGB)
        image = Image.open(sample_info['image_path']).convert('RGB')

        # 주행 가능 영역 마스크 로드 (Grayscale)
        drivable_mask = Image.open(sample_info['drivable_mask_path']).convert('L')
        # 차선 마스크 로드 (Grayscale)
        lane_mask = Image.open(sample_info['lane_mask_path']).convert('L')

        # 이미지 크기와 동일한 최종 마스크 생성 (all zeros initially for background)
        final_mask_np = np.zeros(image.size[::-1], dtype=np.uint8) # (H, W)

        drivable_mask_np = np.array(drivable_mask)
        lane_mask_np = np.array(lane_mask)

        # 1. 주행 가능 영역 (ID 1) 설정
        # drivable_mask_np의 픽셀 값이 1 (direct) 또는 2 (alternative)인 부분을 ID 1로 설정
        final_mask_np[np.where(drivable_mask_np > 0)] = CLASS_LABELS["drivable"] # drivable_mask_np > 0

        # 2. 차선 (ID 2) 설정 - 차선은 주행 가능 영역 위에 덮어씌움 (더 중요한 요소)
        # lane_mask_np의 픽셀 값이 1 (road line) 또는 2 (other lane line)인 부분을 ID 2로 설정
        final_mask_np[np.where(lane_mask_np > 0)] = CLASS_LABELS["lane"] # lane_mask_np > 0

        target_mask = Image.fromarray(final_mask_np)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target_mask = self.target_transform(target_mask)
        else: # target_transform이 없으면 기본적으로 텐서로 변환
            target_mask = torch.from_numpy(np.array(target_mask, dtype=np.int64))

        return image, target_mask, sample_info['image_name'] # image_name도 반환하여 추론 결과에 사용    
```

#### 2.3.2 Minimal UPerNet (ResNet50 백본) 구현

```python
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class UPerNet(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained_backbone=True):
        super(UPerNet, self).__init__()

        # 백본 (Feature Extractor) - ResNet50 사용
        if backbone_name == 'resnet50':
            # weights=ResNet50_Weights.IMAGENET1K_V1: ImageNet으로 사전 학습된 가중치 로드
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None)

            # ResNet의 각 Stage에서 특징맵을 추출
            # C1 = conv1, bn1, relu, maxpool
            # C2 = layer1
            # C3 = layer2
            # C4 = layer3
            # C5 = layer4

            # 실제 FPN은 C2, C3, C4, C5를 사용함
            # `self.resnet_features`는 FPN에 직접 전달될 특징 맵들을 저장함
            self.backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, # Initial layers (before C2)
                resnet.layer1, # C2 output: 256
                resnet.layer2, # C3 output: 512
                resnet.layer3, # C4 output: 1024
                resnet.layer4  # C5 output: 2048
            )

            # ResNet Stage별 출력 채널
            # ResNet50: C2=256, C3=512, C4=1024, C5=2048
            self.in_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported yet.")

        # --- FPN & PPM 관련 채널 설정 재확인 ---
        # UPerNet은 PPM의 출력을 가장 높은 피라미드 레벨의 FPN에 통합합니다.
        # 즉, C5 특징맵을 PPM에 넣고, 그 결과를 FPN의 시작점으로 사용합니다.

        self.ppm_out_channels = 512 # PPM의 최종 출력 채널
        self.ppm = PPM(self.in_channels[-1], self.ppm_out_channels // 4, bins=(1, 2, 3, 6))
        self.ppm_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + self.ppm_out_channels, self.ppm_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ppm_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # FPN 입력 채널들: C2, C3, C4
        # self.in_channels[:-1]은 [256, 512, 1024]
        self.fpn_in_channels = self.in_channels[:-1] # ResNet C2, C3, C4
        self.fpn_out_channels = 256 # FPN 각 단계의 출력 채널 (일반적으로 256)

        self.fpn_convs = nn.ModuleList() # 1x1 conv for FPN lateral connections
        self.fpn_post_convs = nn.ModuleList() # 3x3 conv for FPN output features

        for in_dim in reversed(self.fpn_in_channels): # C4(1024) -> C3(512) -> C2(256) 순서로 처리
            self.fpn_convs.append(nn.Conv2d(in_dim, self.fpn_out_channels, kernel_size=1, bias=False))
            self.fpn_post_convs.append(nn.Sequential(
                nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_out_channels),
                nn.ReLU(inplace=True)
            ))

        # 최종 분류기 (Classifier)
        # PPM 출력 채널 + 모든 FPN 출력 채널을 합친 후 num_classes 채널로
        self.final_head = nn.Sequential(
            nn.Conv2d(self.ppm_out_channels + len(self.fpn_in_channels) * self.fpn_out_channels,
                      num_classes, kernel_size=1)
        )

    def _forward_backbone(self, x):
        # ResNet의 각 Stage에서 특징맵을 추출
        # C1 = conv1, bn1, relu, maxpool
        # C2 = layer1
        # C3 = layer2
        # C4 = layer3
        # C5 = layer4

        # torchvision resnet의 경우, layer1, layer2, layer3, layer4가 각각 C2, C3, C4, C5에 해당
        x = self.backbone[0](x) # conv1, bn1, relu, maxpool
        c2 = self.backbone[1](x) # layer1
        c3 = self.backbone[2](c2) # layer2
        c4 = self.backbone[3](c3) # layer3
        c5 = self.backbone[4](c4) # layer4

        return [c2, c3, c4, c5] # List of feature maps from C2 to C5

    def forward(self, x):
        input_size = x.size()[2:] # (H, W)

        # 백본을 통해 특징맵 추출
        c_features = self._forward_backbone(x) # [c2, c3, c4, c5]

        # C5 특징맵에 PPM 적용
        ppm_out = self.ppm(c_features[-1]) # c_features[-1]은 c5
        ppm_out = self.ppm_conv(ppm_out) # (B, ppm_out_channels, H_c5, W_c5)

        # FPN (Feature Pyramid Network)
        # top-down path
        fpn_out_list = [ppm_out] # PPM 출력이 FPN의 가장 높은 레벨 출력으로 시작

        # ResNet 특징 맵은 [C2, C3, C4, C5] 순서
        # FPN은 C4 -> C3 -> C2 역순으로 합쳐나감.
        # c_features[:-1]은 [c2, c3, c4]
        # reversed(self.fpn_in_channels)는 [1024, 512, 256]

        current_fpn_feature = ppm_out # P5 (C5)에서 시작하는 FPN 특징

        # Zip fpn_convs with reversed(c_features[:-1])
        # self.fpn_convs는 (C4->256), (C3->256), (C2->256)
        # c_features[:-1]은 [C2, C3, C4]

        # 루프를 돌면서 C4, C3, C2에 해당하는 특징맵을 사용해야 합니다.
        # c_features[-2]는 C4, c_features[-3]은 C3, c_features[-4]는 C2

        for i, lateral_conv in enumerate(self.fpn_convs):
            # 이전 FPN 레벨의 특징맵을 현재 스케일로 upsample
            # 현재 current_fpn_feature의 스케일 (H, W)
            target_size = c_features[-(i+2)].size()[2:] # 예를 들어, i=0일 때 c_features[-2]는 C4
            upsampled_current_fpn_feature = F.interpolate(current_fpn_feature, size=target_size, mode='bilinear', align_corners=True)

            # 현재 ResNet 특징맵 (C4, C3, C2)을 1x1 컨볼루션으로 채널 맞춤 (lateral connection)
            # 여기였던 self.fpn_in_convs[i](c[i+1])가 문제였는데, c_features[-(i+2)]로 직접 참조합니다.
            # self.fpn_convs[i]는 lateral_conv에 해당
            lateral_feature = lateral_conv(c_features[-(i+2)]) # c_features[-2]는 C4, c_features[-3]는 C3, c_features[-4]는 C2

            # FPN Add
            current_fpn_feature = lateral_feature + upsampled_current_fpn_feature

            # 3x3 conv (post-fusion)
            current_fpn_feature = self.fpn_post_convs[i](current_fpn_feature)

            fpn_out_list.append(current_fpn_feature)

        # UPerNet은 모든 FPN 레벨의 출력을 원래 입력 크기로 upsample한 후 concatenate
        # fpn_out_list에는 [PPM_out(P5), P4, P3, P2] 순서로 담겨있습니다.

        all_upsampled_fpn_features = []
        for feature in fpn_out_list:
            all_upsampled_fpn_features.append(F.interpolate(feature, size=input_size, mode='bilinear', align_corners=True))

        # 모든 업샘플링된 특징들을 채널 방향으로 합칩니다.
        concat_features = torch.cat(all_upsampled_fpn_features, dim=1) # (B, total_channels, H, W)

        # 최종 분류기 헤드
        output = self.final_head(concat_features)

        return output        
```

#### 2.3.3 데이터 전처리 및 후처리 변환

```python
# ImageNet으로 사전 학습된 ResNet50 백본을 위한 정규화 값 사용
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
    transforms.ToTensor(), # (H, W, C) -> (C, H, W), 0-255 -> 0-1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 마스크는 Tensor로 변환하고, 픽셀 값 그대로 사용 (클래스 ID)
TRANSFORM_MASK = transforms.Compose([
    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST), # 마스크는 Nearest Neighbor 보간
    transforms.ToTensor(), # 0-255 -> 0-1.0
    # 마스크는 클래스 ID이므로 정규화하지 않음. ToTensor() 이후 (1, H, W) 형태로 float32
    # 나중에 .long()로 int64로 변환하여 Loss 함수에 전달해야 함.
])
```

#### 2.3.4 결과 시각화 유틸리티 함수

```python
COLOR_MAP = {
    0: (0, 0, 0),       # Background (Black)
    1: (0, 255, 0),     # Drivable Area (Green)
    2: (255, 0, 0),     # Lane Lines (Red)
    # 3: (0, 0, 255)    # Other lines (Blue) (사용 안 함)
}

def decode_segmentation_map(mask):
    """
    예측된 클래스 ID 마스크 (numpy array)를 컬러 이미지로 변환
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        color_mask[mask == class_id] = color
    return color_mask
```

### 2.4 학습 부분

```python
def train_upernet_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    model.train() # 모델을 훈련 모드로 전환
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=CLASS_LABELS["background"]) # 배경 픽셀은 Loss 계산에서 제외

    print("\n--- UPerNet 모델 학습 시작 (실제로 실행되지 않음) ---")
    print("BDD10K 데이터셋으로 UPerNet을 학습시키려면 상당한 시간과 GPU 자원이 필요합니다.")
    print("이 코드는 개념 이해를 위한 것이며, 실제 학습은 다음처럼 진행될 것입니다:\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, masks, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device) # (B, 1, H, W) -> (B, H, W) for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}")

        # 검증 루프 (생략)
        # model.eval()
        # with torch.no_grad(): ...

        print(f"Epoch [{epoch+1}/{num_epochs}] 평균 Loss: {running_loss / len(train_loader):.4f}")
        # 모델 저장 (예: torch.save(model.state_dict(), f"upernet_epoch_{epoch+1}.pth"))

    print("\n--- UPerNet 모델 학습 완료 (가정) ---")
```

### 2.5 사전 학습 모델 로드 및 추론

```python
# BDD100K 데이터셋을 대상으로 Validation까지 수행할 때 사용
def run_segmentation_inference(model, val_dataset, num_samples_to_show=5):
    model.eval() # 모델을 평가 모드로 전환

    print(f"\n--- 사전 학습 모델 로드 및 추론 시작 ---")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 사전 학습 모델 파일이 없습니다: {PRETRAINED_MODEL_PATH}")
        print("파일 경로를 확인하거나, 해당 파일을 다운로드하여 스크립트와 같은 위치에 배치하세요.")
        return

    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device), strict=False) # strict=False는 일부 레이어가 없을 때 유연하게 처리
        print(f"'{PRETRAINED_MODEL_PATH}' 모델 가중치를 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        print("모델 아키텍처와 .pth 파일의 가중치가 호환되는지 확인하세요.")
        return

    plt.figure(figsize=(20, num_samples_to_show * 5))

    # DataLoader를 사용하여 배치 단위로 이미지 가져오기
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    sample_count = 0
    with torch.no_grad(): # 추론 시에는 그라디언트 계산 비활성화
        for i, (images, masks_gt, image_name) in enumerate(val_loader):
            if sample_count >= num_samples_to_show:
                break

            images = images.to(device)
            # masks_gt = masks_gt.squeeze(1).long().to(device) # Ground Truth 마스크 (옵션)

            # 추론 실행
            outputs = model(images) # (B, num_classes, H, W)

            # 예측된 마스크 (가장 높은 확률을 가진 클래스 ID)
            predicted_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy() # (H, W)

            # 원본 이미지 (PyTorch Tensor -> NumPy RGB)
            original_image_np = images.squeeze(0).cpu().numpy()
            original_image_np = np.transpose(original_image_np, (1, 2, 0)) # (C, H, W) -> (H, W, C)
            original_image_np = (original_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            original_image_np = original_image_np.astype(np.uint8)

            # 예측된 마스크를 컬러로 디코딩
            predicted_color_mask = decode_segmentation_map(predicted_mask)

            # 원본 이미지와 예측된 마스크를 합성하여 시각화
            # 알파 블렌딩 (원본 이미지는 RGB, 마스크는 컬러, 두 개를 투명하게 합성)
            blended_image = cv2.addWeighted(original_image_np, 0.7, predicted_color_mask, 0.3, 0)

            plt.subplot(num_samples_to_show, 1, sample_count + 1)
            plt.imshow(blended_image)
            plt.title(f"Segmentation Result for {image_name[0]}")
            plt.axis('off')

            sample_count += 1

    plt.tight_layout()
    plt.show()
    print("--- 추론 및 시각화 완료 ---")

# BDD10K 등을 대상으로 실제 예측에만 사용하기위하여 샘플링한 데이터에만 추론 예측을 적용함
def run_segmentation_inference_on_random_samples(model, image_paths_list, preprocess_transform, num_samples_to_show=5):
    model.eval() # 모델을 평가 모드로 전환

    print(f"\n--- 사전 학습 모델 로드 및 랜덤 {num_samples_to_show}개 샘플 추론 시작 ---")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 사전 학습 모델 파일이 없습니다: {PRETRAINED_MODEL_PATH}")
        print("파일 경로를 확인하거나, 해당 파일을 다운로드하여 스크립트와 같은 위치에 배치하세요.")
        return

    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device), strict=False)
        print(f"'{PRETRAINED_MODEL_PATH}' 모델 가중치를 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        print("모델 아키텍처와 .pth 파일의 가중치가 호환되는지 확인하세요.")
        return

    if len(image_paths_list) < num_samples_to_show:
        print(f"경고: 사용 가능한 이미지({len(image_paths_list)}개)가 요청한 수({num_samples_to_show}개)보다 적습니다. 사용 가능한 모든 이미지를 추론합니다.")
        samples_to_infer_paths = image_paths_list
    else:
        samples_to_infer_paths = random.sample(image_paths_list, num_samples_to_show) # 랜덤으로 샘플 선택

    plt.figure(figsize=(20, num_samples_to_show * 5))

    with torch.no_grad():
        for i, image_path in enumerate(samples_to_infer_paths):
            image_name = os.path.basename(image_path)

            # 이미지 로드 (RGB)
            original_image = Image.open(image_path).convert('RGB')

            # 전처리
            input_tensor = preprocess_transform(original_image).unsqueeze(0).to(device) # (1, C, H, W)

            # 추론 실행
            start_time = time.time()
            outputs = model(input_tensor) # (B, num_classes, H, W)
            inference_time = time.time() - start_time

            # 예측된 마스크 (가장 높은 확률을 가진 클래스 ID)
            predicted_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy() # (H, W)

            # 원본 이미지 (PIL Image -> NumPy RGB, 시각화를 위해 원본 크기 유지)
            original_image_np = np.array(original_image)

            # 예측된 마스크를 원본 이미지 크기로 리사이즈 후 컬러 디코딩
            # 모델 출력(predicted_mask)은 769x769, 원본 이미지는 1280x720
            # 마스크를 원본 크기로 리사이즈해야 정확히 오버레이 가능
            predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8),
                                                (original_image_np.shape[1], original_image_np.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)

            predicted_color_mask = decode_segmentation_map(predicted_mask_resized)

            # 원본 이미지와 예측된 마스크를 합성하여 시각화
            blended_image = cv2.addWeighted(original_image_np, 0.7, predicted_color_mask, 0.3, 0) # 0.3은 마스크 투명도

            plt.subplot(num_samples_to_show, 1, i + 1)
            plt.imshow(blended_image)
            plt.title(f"[{image_name}] Inference Time: {inference_time:.3f}s")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("--- 추론 및 시각화 완료 ---")    
```

### 2.6 메인 실행 블록

```python
if __name__ == '__main__':
    # 'val' 디렉토리의 절대 경로 설정
    val_image_directory = os.path.join(BDD10K_DATA_ROOT_PATH, "val")

    # 'val' 디렉토리에서 JPG 이미지 파일 경로 목록만 가져옴
    image_paths_for_inference = get_image_paths_from_dir(val_image_directory)
    print(f"BDD10K Validation 이미지 디렉토리에서 {len(image_paths_for_inference)}개 이미지 경로 로드 완료.")

    if not image_paths_for_inference:
        print("로드된 BDD10K 이미지 경로가 없습니다. 'val' 디렉토리 또는 경로 설정을 확인해주세요.")

    else:
        # UPerNet 모델 생성
        model = UPerNet(num_classes=NUM_CLASSES, backbone_name='resnet50', pretrained_backbone=False).to(device)
        print("UPerNet 모델 생성 완료 (ResNet50 백본).")

        run_segmentation_inference_on_random_samples(model, image_paths_for_inference, TRANSFORM_IMG, num_samples_to_show=5)
```

## 3. 라즈베리파이에서의 최적화 방법

- 자율주행 인지 모델은 매우 복잡하고 연산 부하가 큰 모델이므로,
- 라즈베리파이와 같은 임베디드 디바이스에서 실행하기 위해서는 다양한 최적화 기법이 필요함

### 3.1 모델 경량화

- **모델 양자화 (Quantization)**

```python
# TensorFlow Lite 양자화 예제
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('perception_model_tf.h5')

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 양자화 적용 (8비트 정수형으로 변환)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_dataset_gen  # 보정 데이터셋 함수 필요

# TFLite 모델로 변환
tflite_model = converter.convert()

# 모델 저장
with open('perception_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

- **모델 가지치기 (Pruning)**

```python
# PyTorch 모델 가지치기 예제
import torch.nn.utils.prune as prune

# 가중치의 L1 노름 기준 하위 20%를 0으로 설정
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
```

- **지식 증류 (Knowledge Distillation)**

```python
# 작은 학생 모델이 큰 교사 모델의 지식을 학습
def knowledge_distillation_loss(student_outputs, teacher_outputs, true_labels, temperature=5.0, alpha=0.5):
    """
    지식 증류 손실 함수
    - student_outputs: 학생 모델의 출력
    - teacher_outputs: 교사 모델의 출력
    - true_labels: 실제 레이블
    - temperature: 소프트맥스 온도 파라미터
    - alpha: 증류 손실과 일반 손실의 가중치 조절
    """
    # 일반적인 분류 손실
    hard_loss = F.cross_entropy(student_outputs, true_labels)
    
    # 증류 손실 (소프트 타겟에 대한 KL 발산)
    soft_student = F.log_softmax(student_outputs / temperature, dim=1)
    soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # 두 손실을 결합
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

### 3.2 추론 최적화

- **이미지 크기 및 프레임 레이트 조정**

```python
# 더 작은 해상도로 처리
image_size = (192, 320)  # 기존의 절반 크기
frame_skip = 2  # 매 2프레임마다 1프레임만 처리
```

- **작업 분산 처리**

```python
# 스레드 기반 병렬 처리
import threading
import queue

# 프레임 큐와 결과 큐 생성
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# 추론 스레드 함수
def inference_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:  # 종료 신호
            break
        
        # 전처리 및 추론
        processed_frame = preprocess_frame(frame)
        lane_pred, seg_pred, obj_pred = model(processed_frame)
        
        # 후처리 및 결과 큐에 저장
        result = postprocess_results(lane_pred, seg_pred, obj_pred)
        result_queue.put(result)
        
        frame_queue.task_done()

# 추론 스레드 시작
inference_thread = threading.Thread(target=inference_worker)
inference_thread.daemon = True
inference_thread.start()
```

## 4. 자율주행 키트에 적용하기

### 4.1 인지 결과를 제어 시스템에 연결

```python
class AutonomousDrivingSystem:
    def __init__(self, perception_model_path, control_pins=None):
        """
        자율주행 시스템 초기화
        - perception_model_path: 인지 모델 경로
        - control_pins: 모터 제어 핀 설정 (딕셔너리)
        """
        # 라즈베리파이 GPIO 설정
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.GPIO.setmode(GPIO.BCM)
        
        # 기본 제어 핀 설정 (변경 가능)
        self.control_pins = control_pins or {
            'left_forward': 17,
            'left_backward': 18,
            'right_forward': 22,
            'right_backward': 23,
            'pwm_left': 19,
            'pwm_right': 26
        }
        
        # GPIO 핀 설정
        for pin in self.control_pins.values():
            self.GPIO.setup(pin, GPIO.OUT)
        
        # PWM 설정
        self.pwm_left = GPIO.PWM(self.control_pins['pwm_left'], 100)  # 100Hz
        self.pwm_right = GPIO.PWM(self.control_pins['pwm_right'], 100)
        self.pwm_left.start(0)  # 0% 듀티 사이클로 시작
        self.pwm_right.start(0)
        
        # 인지 모델 로드
        import tensorflow as tf
        self.model = tf.keras.models.load_model(perception_model_path)
        
        # 현재 주행 상태
        self.current_state = {
            'speed': 0,  # 0-100 (%)
            'steering': 0,  # -1(좌회전) ~ 0(중앙) ~ 1(우회전)
            'is_running': False
        }
    
    def process_frame(self, frame):
        """
        단일 프레임에 대한 인지 처리 및 주행 결정
        - frame: 카메라에서 캡처한 이미지
        - 반환: 처리된 결과 이미지 및 주행 결정 정보
        """
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 384))
        frame_normalized = frame_resized / 255.0
        input_tensor = np.expand_dims(frame_normalized, axis=0)
        
        # 인지 모델 예측
        lane_pred, seg_pred, obj_pred = self.model.predict(input_tensor, verbose=0)
        
        # 차선 인식 결과 처리
        lane_mask = (lane_pred[0, :, :, 0] > 0.5).astype(np.uint8)
        
        # 차선 기반 조향 계산
        steering = self._calculate_steering_from_lane(lane_mask)
        
        # 세그멘테이션 결과 처리 (도로 영역 식별)
        seg_mask = np.argmax(seg_pred[0], axis=-1).astype(np.uint8)
        road_mask = (seg_mask == 1).astype(np.uint8)  # 클래스 1이 '도로'라고 가정
        
        # 도로 영역 비율에 따른 속도 조정
        road_ratio = np.sum(road_mask) / (road_mask.shape[0] * road_mask.shape[1])
        speed = self._calculate_speed_from_road(road_ratio)
        
        # 객체 감지 결과 처리 (위험 객체 확인)
        obj_heatmap = obj_pred[0]
        danger_detected = self._check_danger_from_objects(obj_heatmap)
        
        # 위험 감지 시 속도 조절 또는 정지
        if danger_detected:
            speed = min(speed, 30)  # 위험 감지 시 속도 제한
            
        # 주행 명령 업데이트
        self._update_driving_command(steering, speed)
        
        # 결과 시각화
        result_image = self._visualize_results(frame_resized, lane_mask, seg_mask, obj_heatmap)
        
        return result_image, {
            'steering': steering,
            'speed': speed,
            'danger': danger_detected
        }
    
    def _calculate_steering_from_lane(self, lane_mask):
        """차선 마스크에서 조향각 계산"""
        # 차선 마스크의 하단 부분만 사용 (주행 방향 결정에 더 중요)
        h, w = lane_mask.shape
        bottom_half = lane_mask[h//2:, :]
        
        # 차선이 없는 경우 이전 조향각 유지
        if np.sum(bottom_half) == 0:
            return self.current_state['steering']
        
        # 차선의 무게 중심 계산
        M = cv2.moments(bottom_half)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            # 중심에서의 편차를 -1 ~ 1 사이 값으로 정규화
            steering = (cx - w/2) / (w/2)
            # 갑작스러운 조향 변화 방지 (이전 값과 평균)
            steering = 0.7 * steering + 0.3 * self.current_state['steering']
            return max(-1, min(1, steering))  # -1 ~ 1 사이로 제한
        
        return self.current_state['steering']
    
    def _calculate_speed_from_road(self, road_ratio):
        """도로 영역 비율에 따른 속도 계산"""
        if road_ratio < 0.2:  # 도로 영역이 너무 작으면 감속
            return max(0, self.current_state['speed'] - 10)
        elif road_ratio > 0.4:  # 도로 영역이 충분하면 가속
            return min(70, self.current_state['speed'] + 5)
        else:  # 현재 속도 유지
            return self.current_state['speed']
    
    def _check_danger_from_objects(self, obj_heatmap):
        """객체 탐지 결과에서 위험 상황 확인"""
        # 위험 객체 클래스 (예: 사람, 자전거, 다른 차량)
        danger_classes = [12, 14, 18]  # person, car, motorcycle 클래스 인덱스
        
        # 위험 객체 클래스의 스코어맵에서 최대값 확인
        danger_score = 0
        for cls in danger_classes:
            if cls < obj_heatmap.shape[2]:
                cls_score = np.max(obj_heatmap[:, :, cls])
                danger_score = max(danger_score, cls_score)
        
        # 위험 점수가 임계값을 초과하면 위험 상황으로 판단
        return danger_score > 0.7  # 임계값 0.7 (조정 가능)
    
    def _update_driving_command(self, steering, speed):
        """조향각과 속도에 따라 모터 제어 명령 업데이트"""
        # 현재 상태 업데이트
        self.current_state['steering'] = steering
        self.current_state['speed'] = speed
        
        # 정지 상태인 경우
        if speed == 0:
            self.pwm_left.ChangeDutyCycle(0)
            self.pwm_right.ChangeDutyCycle(0)
            self.GPIO.output(self.control_pins['left_forward'], GPIO.LOW)
            self.GPIO.output(self.control_pins['left_backward'], GPIO.LOW)
            self.GPIO.output(self.control_pins['right_forward'], GPIO.LOW)
            self.GPIO.output(self.control_pins['right_backward'], GPIO.LOW)
            self.current_state['is_running'] = False
            return
        
        # 주행 상태 설정
        self.current_state['is_running'] = True
        
        # 전진 모드 설정
        self.GPIO.output(self.control_pins['left_forward'], GPIO.HIGH)
        self.GPIO.output(self.control_pins['left_backward'], GPIO.LOW)
        self.GPIO.output(self.control_pins['right_forward'], GPIO.HIGH)
        self.GPIO.output(self.control_pins['right_backward'], GPIO.LOW)
        
        # 조향각에 따른 좌우 모터 속도 차등 적용
        if steering < 0:  # 좌회전
            left_speed = max(0, speed * (1 + steering))  # 좌측 모터 속도 감소
            right_speed = speed  # 우측 모터 속도 유지
        elif steering > 0:  # 우회전
            left_speed = speed  # 좌측 모터 속도 유지
            right_speed = max(0, speed * (1 - steering))  # 우측 모터 속도 감소
        else:  # 직진
            left_speed = speed
            right_speed = speed
        
        # PWM 듀티 사이클 설정
        self.pwm_left.ChangeDutyCycle(left_speed)
        self.pwm_right.ChangeDutyCycle(right_speed)
    
    def _visualize_results(self, frame, lane_mask, seg_mask, obj_heatmap):
        """인지 결과 시각화"""
        # 원본 이미지 복사
        result_image = frame.copy()
        
        # 차선 마스크 오버레이 (녹색)
        lane_overlay = np.zeros_like(result_image)
        lane_overlay[:, :, 1] = lane_mask * 255  # 녹색 채널
        result_image = cv2.addWeighted(result_image, 0.7, lane_overlay, 0.3, 0)
        
        # 세그멘테이션 결과 오버레이
        # 여기서는 도로 영역만 표시 (파란색)
        road_mask = (seg_mask == 1).astype(np.uint8)  # 클래스 1이 '도로'라고 가정
        road_overlay = np.zeros_like(result_image)
        road_overlay[:, :, 0] = road_mask * 150  # 파란색 채널 (약하게)
        result_image = cv2.addWeighted(result_image, 0.9, road_overlay, 0.3, 0)
        
        # 현재 주행 상태 표시
        cv2.putText(result_image, f"Speed: {self.current_state['speed']}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_image, f"Steering: {self.current_state['steering']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image
    
    def start_autonomous_driving(self, camera_index=0):
        """자율주행 시작"""
        print("자율주행 시스템을 시작합니다...")
        
        # 카메라 열기
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("카메라에서 프레임을 읽을 수 없습니다.")
                    break
                
                # 프레임 처리 및 주행 결정
                result_image, driving_info = self.process_frame(frame)
                
                # 결과 표시
                cv2.imshow('Autonomous Driving System', result_image)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("사용자에 의해 자율주행이 중단되었습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
        finally:
            # 자원 해제
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """자원 정리"""
        print("자율주행 시스템을 종료합니다...")
        # 모터 정지
        self.pwm_left.stop()
        self.pwm_right.stop()
        # GPIO 정리
        self.GPIO.cleanup()


# 메인 실행 코드
if __name__ == "__main__":
    # 모델 경로 설정
    model_path = 'perception_model_tf_dummy.h5'  # 또는 'perception_model_pt_dummy.pth'
    
    # 자율주행 시스템 초기화 및 실행
    try:
        # GPIO 사용을 위해 라즈베리파이에서 실행 필요
        # 라즈베리파이가 아닌 환경에서는 테스트용 더미 드라이버 사용 가능
        import platform
        if 'raspberrypi' in platform.uname().node.lower():
            # 실제 라즈베리파이 환경
            print("라즈베리파이 환경에서 자율주행 시스템을 초기화합니다...")
            driving_system = AutonomousDrivingSystem(model_path)
        else:
            # 테스트 환경 (GPIO 없는 PC 등)
            print("테스트 환경에서 시뮬레이션 모드로 실행합니다...")
            # 더미 GPIO 라이브러리 사용 (테스트용)
            import sys
            sys.modules['RPi.GPIO'] = type('DummyGPIO', (), {
                'setmode': lambda x: None,
                'setup': lambda x, y: None,
                'output': lambda x, y: None,
                'PWM': type('DummyPWM', (), {
                    '__init__': lambda self, pin, freq: None,
                    'start': lambda self, dc: None,
                    'ChangeDutyCycle': lambda self, dc: None,
                    'stop': lambda self: None
                }),
                'BCM': 0,
                'OUT': 0,
                'HIGH': 1,
                'LOW': 0,
                'cleanup': lambda: None
            })
            driving_system = AutonomousDrivingSystem(model_path)
        
        # 자율주행 시작
        driving_system.start_autonomous_driving(camera_index=0)
    
    except KeyboardInterrupt:
        print("사용자에 의해 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
```

## 5. 프로젝트 응용 및 확장 아이디어

### 5.1 모델 개선 및 확장

- **경량화된 실시간 객체 탐지 통합**
   - YOLOv5-nano나 MobileNet-SSD와 같은 경량 객체 탐지 모델을 통합하여 더 정확한 객체 인식 구현
   - 객체의 위치와 크기를 정확히 파악하여 충돌 회피 기능 개선

- **시간적 일관성(Temporal Consistency) 추가**
   - 연속된 프레임 간의 정보를 활용하여 인지 결과의 안정성 향상
   - 칼만 필터와 같은 추적 알고리즘을 통합하여 객체 추적 및 예측 기능 추가

- **깊이 추정(Depth Estimation) 통합**
   - 단안 카메라에서도 깊이 정보를 추정하여 3D 공간 인식 기능 추가
   - 모노큘러 깊이 추정 모델(예: MiDaS)을 활용하여 거리 정보 파악

### 5.2 하드웨어 확장

- **다중 센서 통합**
   - 초음파 센서를 추가하여 근거리 장애물 감지 보완
   - IMU(관성 측정 장치)를 추가하여 차량의 움직임 및 자세 정보 활용

- **통신 모듈 추가**
   - Wi-Fi나 블루투스 모듈을 통해 원격 모니터링 및 제어 기능 구현
   - 여러 자율주행 차량 간의 통신을 통한 협력 주행 시뮬레이션

- **GPS 모듈 통합**
   - 실외 주행 시 GPS 정보를 활용한 위치 인식 및 경로 계획 기능 추가

### 5.3 교육적 활용 방안

- **단계별 학습 모듈화**
   - 차선 인식 → 객체 탐지 → 세그멘테이션 → 통합 시스템 순으로 단계적 학습
   - 각 모듈별 성능 측정 및 비교 실험을 통한 이해도 향상

- **시나리오 기반 학습**
   - 다양한 주행 시나리오(도심, 고속도로, 교차로 등)에 맞는 인지 모델 적용
   - 악천후, 야간 등 도전적인 환경에서의 인지 성능 테스트

- **게임화(Gamification) 요소 추가**
   - 장애물 코스 완주, 목표 지점 도달 등의 미션 수행을 통한 학습 동기 부여
   - 팀 대항전 형식으로 알고리즘 성능 경쟁을 통한 협력 학습

## 6. 결론 및 학습 포인트

- **멀티태스크 학습의 이해**
   - 하나의 모델이 여러 작업(차선 인식, 세그멘테이션, 객체 탐지)을 동시에 수행하는 방식
   - 공유 특징 추출기를 통한 효율적인 학습 및 추론 과정

- **인지-판단-제어 파이프라인**
   - 자율주행의 핵심 구조인 인지-판단-제어 파이프라인의 구현 및 이해
   - 각 단계가 어떻게 유기적으로 연결되어 작동하는지 체험

- **임베디드 시스템에서의 최적화**
   - 제한된 컴퓨팅 자원을 가진 라즈베리파이에서 딥러닝 모델을 효율적으로 실행하는 방법
   - 모델 경량화, 양자화, 병렬 처리 등 최적화 기법의 실제 적용

- **실시간 처리의 중요성**
   - 자율주행에서 실시간 처리가 왜 중요한지, 그리고 이를 위한 다양한 기법
   - 지연 시간(latency)과 처리 속도(throughput) 사이의 균형 맞추기

- **인공지능 기술의 실생활 응용**
   - 딥러닝 모델이 어떻게 실제 하드웨어와 결합하여 물리적 세계에 영향을 미치는지 체험
   - 인지 모델의 예측 결과가 어떻게 실제 차량 제어로 이어지는지 이해
