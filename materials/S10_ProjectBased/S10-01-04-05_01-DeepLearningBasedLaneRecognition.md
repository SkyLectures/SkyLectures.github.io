---
layout: page
title:  "딥러닝 기반 차선 인식"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-05_01-DeepLearningBasedLaneRecognition
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/project/notebooks/S10-01-04-05_01-DeepLearningBasedLaneRecognition.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

> - **딥러닝 기반 차선 인식 시스템**
>   - 차선 인식은 자율주행 시스템의 핵심 기능 중 하나
>   - 차량이 **도로에서 자신의 위치를 파악**하고 안전하게 주행하기 위한 필수적인 요소
>   - 전통적인 컴퓨터 비전 방식: Canny 엣지 검출, 허프 변환 등
>   - 딥러닝 기반 방식: 다양한 도로 환경과 조명 조건에서도 더 강인한 성능을 보여줌
{: .common-quote}

## 1. 차선 인식 실습 개요

- **실습 내용**
    - U-Net 아키텍처를 기반으로 한 세그멘테이션(Segmentation) 모델을 사용한 차선 인식

- **데이터셋**
    - 차선 인식을 위한 대표적인 공개 데이터셋: TuSimple, CULane, BDD100K 등
    - 실습에서는 TuSimple 데이터셋을 기준으로 진행
        - [TuSimple 데이터셋 다운로드](https://github.com/TuSimple/tusimple-benchmark){: target="_blank"}
        
## 2. PyTorch 기반 구현

### 2.1 필요 라이브러리 설치

```bash
#// file: "Terminal"
pip install torch torchvision opencv-python numpy matplotlib Pillow tqdm
```

### 2.2 필요한 라이브러리 가져오기

```python
#//file: "pytorch_lane_detection_realtime.py"

import os
import random
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
```

### 2.3 데이터셋 디렉토리 정의

```python
train_image_dir = './train'
train_mask_dir = './train_label'
val_image_dir = './val'
val_mask_dir = './val_label'
```

### 2.4 데이터셋 확인

- 이미지 속성 분석

```python
def analyze_image_properties(image_dir):
    image_files = os.listdir(image_dir)
    resolutions = []
    channels = []
    missing_files = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            with Image.open(img_path) as img:
                # 해상도(너비, 높이) 수집
                resolutions.append(img.size)  # (너비, 높이)
                
                # 채널 수 수집
                channels.append(len(img.getbands()))  # RGB = 3, Grayscale = 1
                
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            missing_files.append(img_file)

    return resolutions, channels, missing_files

# 학습용 이미지와 마스크의 속성 분석
train_img_resolutions, train_img_channels, train_img_missing_files = analyze_image_properties(train_image_dir)
train_label_resolutions, train_label_channels, train_label_missing_files = analyze_image_properties(train_mask_dir)

# 검증 이미지와 마스크의 속성 분석
val_img_resolutions, val_img_channels, val_img_missing_files = analyze_image_properties(val_image_dir)
val_label_resolutions, val_label_channels, val_label_missing_files = analyze_image_properties(val_mask_dir)

# 결과 표시
print(f"Analysis of Train image dataset")
print(f"List of resolutions: {set(train_img_resolutions)}")
print(f"List of channels: {set(train_img_channels)}")
print(f"Missing files: {train_img_missing_files}")

print(f"\nAnalysis of Train mask dataset")
print(f"List of resolutions: {set(train_label_resolutions)}")
print(f"List of channels: {set(train_label_channels)}")
print(f"Missing files: {train_label_missing_files}")

print(f"\nAnalysis of Validation image dataset")
print(f"List of resolutions: {set(val_img_resolutions)}")
print(f"List of channels: {set(val_img_channels)}")
print(f"Missing files: {val_img_missing_files}")

print(f"\nAnalysis of Validation mask dataset")
print(f"List of resolutions: {set(val_label_resolutions)}")
print(f"List of channels: {set(val_label_channels)}")
print(f"Missing files: {val_label_missing_files}")
```

- 마스크 파일의 고유 클래스 분석

```python
def get_unique_classes_from_dir(mask_dir):
    mask_files = os.listdir(mask_dir)
    all_unique_classes = set()

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask)
                
                # 고유 클래스 확인
                unique_classes = np.unique(mask_array)
                
                # 모든 고유 클래스 업데이트
                all_unique_classes.update(unique_classes)
        except Exception as e:
            print(f"Error loading mask {mask_file}: {e}")
    
    return all_unique_classes

train_unique_classes = get_unique_classes_from_dir(train_mask_dir)
val_unique_classes = get_unique_classes_from_dir(val_mask_dir)

# 고유 클래스 출력
print(f"Unique classes in Train masks: {train_unique_classes}")
print(f"Number of classes in Train masks: {len(train_unique_classes)}")

print(f"Unique classes in Val masks: {val_unique_classes}")
print(f"Number of classes in Val masks: {len(val_unique_classes)}")
```

### 2.5 영역분할용 데이터셋 생성

- 영역분할 작업을 위한 사용자 정의 데이터셋 클래스 정의

```python
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_name = self.images[idx].replace('.png', '_label.png')
        label_path = os.path.join(self.label_dir, label_name)
        
        # 이미지와 마스크 읽기
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(label_path).convert('L')  # 그레이스케일 마스크
        
        # 이미지와 마스크 모두에 변환 적용
        if self.transform:
            image = self.transform(image)
            mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)
        
        mask = torch.from_numpy(np.array(mask)).long()  # 마스크를 텐서로 변환
        
        return image, mask
```

- 변환 정의

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
```

- 데이터 로더 생성

```python
train_dataset = SegmentationDataset(image_dir=train_image_dir, label_dir=train_mask_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = SegmentationDataset(image_dir=val_image_dir, label_dir=val_mask_dir, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
```

### 2.6 U-Net 아키텍처 정의

- U-Net 아키텍처 정의

```python
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        # Contracting path
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        
        # Bottleneck
        self.bottleneck = self.conv_block(1024, 2048)
        
        # Expanding path
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder5 = self.conv_block(2048, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Final output layer for multi-class segmentation
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block
    
    def forward(self, x):
        # Contracting path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.encoder5(F.max_pool2d(enc4, kernel_size=2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc5, kernel_size=2))
        
        # Expanding path
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((enc5, dec5), dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output layer (logits)
        return self.conv_last(dec1)
```

- 3개 클래스에 대한 U-Net 모델 초기화

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=3).to(device)
#model.load_state_dict(torch.load('best_model.pth'))
```

### 2.7 손실함수 및 최적화 함수 정의

```python
# 손실함수
criterion = nn.CrossEntropyLoss()

# 최적화 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 2.8 매트릭 정의: IoU 및 Dice 계수

- IoU 계산 함수

```python
def calculate_iou(preds, masks, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)  # 각 픽셀에 대한 예측 클래스 가져오기
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        union = torch.sum(pred_cls) + torch.sum(mask_cls) - intersection
        
        if union == 0:
            ious.append(1.0)  # 합집합이 없다면 IoU는 완전한 것으로 간주
        else:
            ious.append((intersection / union).item())
    
    return sum(ious) / len(ious)  # 모든 클래스의 평균 IoU 반환
```

- Dice 계수 계산 함수

```python
def calculate_dice(preds, masks, num_classes):
    dices = []
    preds = torch.argmax(preds, dim=1)  # 각 픽셀에 대한 예측 클래스 가져오기
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        dice = (2 * intersection) / (torch.sum(pred_cls) + torch.sum(mask_cls))
        
        if torch.sum(pred_cls) + torch.sum(mask_cls) == 0:
            dices.append(1.0)
        else:
            dices.append(dice.item())
    
    return sum(dices) / len(dices)  # 모든 클래스의 평균 Dice 계수를 반환
```

### 2.9 모델 훈련 및 검증

```python
num_epochs = 50
train_loss_list = []
val_loss_list = []
iou_list = []
dice_list = []

num_classes = 3  # 분할 클래스의 개수

# 학습률 스케줄러(검증 손실이 정점에 도달하면 LR을 줄임)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
save_path = 'best_model.pth'

# 조기 종료 설정
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # 훈련(학습) 루프
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
        images = images.to(device)
        masks = masks.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # 검증 루프
    model.eval()
    val_loss = 0.0
    iou_total = 0.0
    dice_total = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            iou = calculate_iou(outputs, masks, num_classes=num_classes)
            dice = calculate_dice(outputs, masks, num_classes=num_classes)
            
            iou_total += iou
            dice_total += dice
    
    # 평균값 계산
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_iou = iou_total / len(val_loader)
    avg_dice = dice_total / len(val_loader)
    
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    iou_list.append(avg_iou)
    dice_list.append(avg_dice)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"MIoU: {avg_iou:.4f}, Dice Coefficient: {avg_dice:.4f}")

    # 학습률 스케줄러 업데이트
    scheduler.step(avg_val_loss)

    # 조기 종료 로직
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        early_stopping_counter = 0
        print("Validation loss improved, resetting early stopping counter.")
    else:
        early_stopping_counter += 1
        print(f"Validation loss did not improve for {early_stopping_counter} epochs.")
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
```

### 2.10 결과 표시

```python
# 훈련 및 검증 손실, MIoU 및 Dice 메트릭을 플로팅하는 함수
def plot_metrics(train_loss_list, val_loss_list, iou_list, dice_list):
    epochs = range(1, len(train_loss_list) + 1)

    # 손실 값 플로팅
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # MIoU 와 Dice 계수 플로팅
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, iou_list, label='Mean IoU', color='blue', alpha=0.5)
    plt.plot(epochs, dice_list, label='Dice Coefficient', color='green', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('MIoU and Dice Coefficient Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 2.11 예측 시각화

```python
def visualize_random_predictions(model, dataloader, device, num_images=5):
    model.eval()  # 모델을 평가 모드로 설정
    
    dataset_size = len(dataloader.dataset)
    random_indices = random.sample(range(dataset_size), num_images)
    
    images_so_far = 0
    fig, ax = plt.subplots(num_images, 3, figsize=(10, num_images * 5))
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            batch_size = images.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            valid_indices = [i for i in random_indices if start_idx <= i < end_idx]
            
            if len(valid_indices) == 0:
                continue
            
            images = images.to(device)
            masks = masks.to(device)
            
            # 예측
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in valid_indices:
                local_idx = i - start_idx
                
                pred_mask = preds[local_idx].cpu().numpy()
                true_mask = masks[local_idx].cpu().numpy()
                
                ax[images_so_far, 0].imshow(images[local_idx].cpu().permute(1, 2, 0))
                ax[images_so_far, 0].set_title('Input Image')
                
                ax[images_so_far, 1].imshow(true_mask, cmap='gray')
                ax[images_so_far, 1].set_title('True Mask')
                
                ax[images_so_far, 2].imshow(pred_mask, cmap='gray')
                ax[images_so_far, 2].set_title('Predicted Mask')
                
                images_so_far += 1
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return

    plt.tight_layout()
    plt.show()

visualize_random_predictions(model, val_loader, device, num_images=5)
```

### 2.12 추론 속도 및 추정 FPS 측정

- FPS(Frames Per Second, 초당 프레임 수)
    - 비디오 스트리밍, 자율주행 또는 기타 시간에 민감한 작업 등 실시간 애플리케이션에 중요한 지표
    - 모델이 동적이고 빠르게 변화하는 환경에서 얼마나 효율적으로 예측을 제공할 수 있는지를 보여줌

- 추론 속도와 추정 FPS 를 측정해야 하는 이유
    - 이 단계에서는 훈련된 모델이 데이터 세트의 단일 이미지에 대해 추론을 수행하는 데 걸리는 평균 시간을 평가함
    - 이를 통해 배포 과정에서 모델의 성능을 명확하게 파악할 수 있음
    - 추론 시간을 기반으로 모델이 1초에 처리할 수 있는 이미지 수를 정량화하는 초당 프레임 수(FPS)를 계산

```python
# 이미지당 추론 시간 측정
def measure_inference_speed(model, dataloader, device, num_images=10):
    model.eval()  # 모델을 평가 모드로 설정
    total_time = 0.0
    images_processed = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            images_processed += len(images)
            
            if images_processed >= num_images:
                break
    
    avg_inference_time = total_time / images_processed
    print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
    
    return avg_inference_time

# 검증 세트에서 추론 속도 측정
avg_inference_time = measure_inference_speed(model, val_loader, device, num_images=10)

fps = 1 / avg_inference_time  # FPS(초당 프레임 수) 계산
print(f"Estimated FPS (Frames Per Second): {fps:.2f} FPS")
```

## 3. 라즈베리파이에서의 최적화 방법

- 라즈베리파이와 같은 임베디드 디바이스에서 딥러닝 모델을 실행할 때는 계산 자원의 한계를 고려해야 함
- 다음과 같은 최적화 방법을 적용하면 성능을 향상시킬 수 있음

### 3.1 모델 경량화 기법

- **모델 양자화 (Quantization)**

```python
# TensorFlow 모델 양자화 예제
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('lane_detection_unet_tf.h5')

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 양자화 적용
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # 16비트 부동소수점으로 양자화

# TFLite 모델로 변환
tflite_model = converter.convert()

# 모델 저장
with open('lane_detection_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

- **모델 가지치기 (Pruning)**

```python
# TensorFlow 모델 가지치기 예제
import tensorflow_model_optimization as tfmot

# 가지치기 스케줄 정의
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,  # 50%의 가중치 제거
    begin_step=0,
    end_step=1000
)

# 모델 가지치기 적용
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule
)

# 가지치기된 모델 컴파일
model_for_pruning.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 가지치기 적용하여 추가 학습
model_for_pruning.fit(train_dataset, epochs=5, validation_data=val_dataset)

# 최종 모델 저장 (가중치 0인 부분 제거)
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
final_model.save('lane_detection_pruned.h5')
```

### 3.2 추론 속도 향상 기법

- **이미지 크기 축소**

```python
# 입력 이미지 해상도 낮추기
img_size = (128, 256)  # 원래 (256, 512)의 절반 크기
```

- **프레임 건너뛰기**

```python
# 실시간 처리에서 프레임 건너뛰기
frame_count = 0
process_every_n_frames = 2  # 2프레임마다 1번 처리

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # 프레임 건너뛰기
        continue
        
    # 이미지 처리 코드...
```

- **모델 추론 스레드 분리**

```python
import threading
import queue

# 프레임 큐 생성
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)

# 추론 스레드 함수
def inference_thread(model, frame_queue, result_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # 전처리 및 추론
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            
            # 결과 큐에 저장
            result_queue.put(prediction)

# 스레드 시작
inference_thread = threading.Thread(target=inference_thread, 
                                   args=(model, frame_queue, result_queue))
inference_thread.daemon = True
inference_thread.start()

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 큐에 추가
    if not frame_queue.full():
        frame_queue.put(frame)
    
    # 결과 큐에서 가져오기
    if not result_queue.empty():
        prediction = result_queue.get()
        # 결과 시각화 및 표시
```

## 4. 실제 자율주행 키트에 적용하기

### 4.1 차선 인식 결과를 모터 제어에 연결

```python
# 모터 제어 클래스 (예시)
class MotorController:
    def __init__(self, left_pin=17, right_pin=18):
        # GPIO 설정 코드
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.left_pin = left_pin
        self.right_pin = right_pin
        
        # GPIO 초기화
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.left_pin, GPIO.OUT)
        GPIO.setup(self.right_pin, GPIO.OUT)
        
        # PWM 설정
        self.left_pwm = GPIO.PWM(self.left_pin, 100)
        self.right_pwm = GPIO.PWM(self.right_pin, 100)
        self.left_pwm.start(0)
        self.right_pwm.start(0)
    
    def move_forward(self, speed=50):
        self.left_pwm.ChangeDutyCycle(speed)
        self.right_pwm.ChangeDutyCycle(speed)
    
    def turn_left(self, intensity=30):
        self.left_pwm.ChangeDutyCycle(30)
        self.right_pwm.ChangeDutyCycle(60)
    
    def turn_right(self, intensity=30):
        self.left_pwm.ChangeDutyCycle(60)
        self.right_pwm.ChangeDutyCycle(30)
    
    def stop(self):
        self.left_pwm.ChangeDutyCycle(0)
        self.right_pwm.ChangeDutyCycle(0)
    
    def cleanup(self):
        self.stop()
        self.GPIO.cleanup()

# 차선 인식 결과를 모터 제어에 연결
def lane_detection_with_control(model_path, camera_index=0, img_size=(128, 256)):
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 카메라 및 모터 컨트롤러 초기화
    cap = cv2.VideoCapture(camera_index)
    motor_controller = MotorController()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 전처리 (TensorFlow 버전 가정)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (img_size[1], img_size[0]))
            normalized_frame = resized_frame / 255.0
            input_tensor = np.expand_dims(normalized_frame, axis=0)
            
            # 예측
            prediction = model.predict(input_tensor)
            lane_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # --- 차선 인식 결과 분석 및 주행 판단 ---
            # 여기서 lane_mask를 분석하여 차량이 어느 방향으로 움직여야 할지 결정
            # 예시: 마스크의 중심을 계산하여 차량의 치우침 정도 판단
            
            # 차선 마스크가 비어있지 않은 경우에만 처리
            if np.sum(lane_mask) > 0:
                # 차선 마스크의 무게 중심 (Centroid) 계산
                # moment = cv2.moments(lane_mask)
                # if moment["m00"] != 0:
                #     cx = int(moment["m10"] / moment["m00"])
                #     cy = int(moment["m01"] / moment["m00"])
                # else:
                #     cx, cy = img_size[1] // 2, img_size[0] // 2 # 차선 없을 시 중앙
                
                # 또는 단순히 하단 ROI의 차선 픽셀 분포로 판단 (더 간단)
                bottom_roi = lane_mask[img_size[0] - 20:img_size[0], :] # 하단 20픽셀만 확인
                if np.sum(bottom_roi) > 0:
                    bottom_center_x = np.mean(np.where(bottom_roi > 0)[1])
                else:
                    bottom_center_x = img_size[1] // 2 # 차선 없을 시 중앙 가정
                    
                
                # 이미지 중앙과의 오차 계산
                center_offset = bottom_center_x - (img_size[1] // 2)
                
                # --- 모터 제어 ---
                if abs(center_offset) < 20:  # 오차가 작으면 직진 (임계값 조정 필요)
                    motor_controller.move_forward(speed=40)
                    control_text = "직진"
                elif center_offset < -20:  # 왼쪽으로 치우침 -> 우회전
                    motor_controller.turn_right(intensity=abs(center_offset) / 5) # 오차 비례 강도 조절
                    control_text = "우회전"
                else:  # 오른쪽으로 치우침 -> 좌회전
                    motor_controller.turn_left(intensity=abs(center_offset) / 5)
                    control_text = "좌회전"
            else:
                # 차선을 찾지 못할 경우 정지 또는 다른 전략 (예: 마지막 경로 유지)
                motor_controller.stop()
                control_text = "차선 없음 (정지)"
            
            # 예측 결과 시각화 (원본 프레임에 차선 오버레이 및 제어 정보 추가)
            lane_mask_resized = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
            lane_overlay = np.zeros_like(frame)
            lane_overlay[:, :, 1] = lane_mask_resized * 255 # 초록색으로 차선 표시
            
            result_frame = cv2.addWeighted(frame, 1, lane_overlay, 0.5, 0)
            
            # 제어 정보 텍스트 추가
            cv2.putText(result_frame, f"Control: {control_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_frame, f"Offset: {center_offset:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Lane Detection and Control', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        motor_controller.cleanup()
        cv2.destroyAllWindows()

# 실제 실행 (학습된 TensorFlow 모델 필요)
if __name__ == "__main__":
    # 모델 경로와 카메라 인덱스 설정
    model_path = 'lane_detection_unet_tf.h5' # 학습된 모델 경로
    camera_idx = 0 # 라즈베리파이 카메라 인덱스
    
    # 모델 학습 코드는 생략 (위에 제시된 학습 함수를 통해 모델을 먼저 학습시키거나, 미리 학습된 모델 파일을 준비해야 함)
    # try:
    #     tf.keras.models.load_model(model_path)
    #     print("저장된 모델을 불러왔습니다.")
    # except:
    #     print("모델 파일을 찾을 수 없습니다. 모델을 먼저 학습시켜 주세요.")
    #     exit()
        
    lane_detection_with_control(model_path, camera_idx)
```

### 4.2 모터 제어 보정 및 전략 수립

- **비례 제어 (Proportional Control)**
    - `center_offset` 값이 클수록 모터의 조향 강도를 더 강하게 주는 방식으로 정밀도를 높일 수 있음
        - (예시 코드에 간단하게 반영되어 있음)

- **PID 제어**
    - 더 안정적인 제어를 위해 PID(비례-적분-미분) 제어를 적용하여 오차, 오차 누적, 오차 변화율을 복합적으로 고려하여 제어할 수 있음

- **차선 이탈 감지**
    - 차선 마스크가 일정 시간 동안 화면에서 사라지거나, 차량이 마스크의 경계를 넘어설 경우
    - '차선 이탈'로 판단하고 경고 또는 비상 정지 등의 전략 구현

- **회전 구간 처리**
    - 교차로나 급커브 구간에서는 차선 인식이 어려울 수 있으므로,
    - 미리 학습된 맵 정보나 다른 센서(예: IMU) 정보를 활용하여 보조적인 주행 전략을 세워야 함



> - 라즈베리파이 환경에서는 프레임 속도와 모델 추론 속도 간의 균형을 맞추는 것이 중요함
> - 최적화 기법에 대한 고려도 함께 진행하면 좋음
{: .expert-quote}