---
layout: page
title:  "딥러닝 기반 차선 인식"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-05_01-DeepLearningBasedLaneRecognition
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


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
        
## 2. TensorFlow 기반 구현

### 2.1 필요 라이브러리 설치

```bash
#// file: "Terminal"
pip install tensorflow==2.15.0 opencv-python numpy matplotlib scikit-learn
```

### 2.2 U-Net 모델 구현

```python
#//file: "tensorflow_lane_detection_realtime.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU 메모리 설정
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 이미지 전처리 함수
def preprocess_image(img_path, mask_path=None, img_size=(256, 512)):
    # 이미지 로드 및 리사이징
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # 정규화
    
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # 정규화
        return img, mask
    return img

# 데이터 로딩 함수 (예시)
def load_data(data_dir, img_size=(256, 512)):
    images = []
    masks = []
    
    # 실제 구현에서는 TuSimple 데이터셋 구조에 맞게 수정 필요
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                img, mask = preprocess_image(img_path, mask_path, img_size)
                images.append(img)
                masks.append(mask)
    
    return np.array(images), np.array(masks)[..., np.newaxis]

# U-Net 모델 구현
def build_unet_model(input_size=(256, 512, 3)):
    inputs = Input(input_size)
    
    # 인코더 (다운샘플링)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # 병목 부분
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # 디코더 (업샘플링)
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # 출력층 - 이진 세그멘테이션(차선 vs 비차선)
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 모델 학습 함수
def train_lane_detection_model(data_dir, img_size=(256, 512), batch_size=8, epochs=50):
    # 데이터 로드
    X, y = load_data(data_dir, img_size)
    
    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 구축
    model = build_unet_model(input_size=(img_size[0], img_size[1], 3))
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    
    # 모델 저장 콜백
    model_checkpoint = ModelCheckpoint(
        'lane_detection_unet_tf.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    
    # 조기 종료 콜백
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping]
    )
    
    return model, history

# 예측 및 시각화 함수
def predict_and_visualize(model, image_path, img_size=(256, 512)):
    # 이미지 로드 및 전처리
    img = preprocess_image(image_path, img_size=img_size)
    img_batch = np.expand_dims(img, axis=0)
    
    # 예측
    pred_mask = model.predict(img_batch)[0]
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('원본 이미지')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask[:, :, 0], cmap='gray')
    plt.title('예측된 차선 마스크')
    
    plt.subplot(1, 3, 3)
    # 원본 이미지에 예측 마스크 오버레이
    overlay = img.copy()
    mask = (pred_mask[:, :, 0] > 0.5).astype(np.uint8)
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[:, :, 1] = mask * 255  # 초록색으로 마스크 표시
    blended = cv2.addWeighted(overlay, 0.7, overlay_mask, 0.3, 0)
    plt.imshow(blended)
    plt.title('차선 인식 결과')
    
    plt.tight_layout()
    plt.show()
```

### 2.3 실시간 차선 인식 구현

```python
#//file: "tensorflow_lane_detection_realtime.py
def lane_detection_realtime(model_path, camera_index=0, img_size=(256, 512)):
    # 저장된 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (img_size[1], img_size[0]))
        normalized_frame = resized_frame / 255.0
        
        # 예측
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        prediction = model.predict(input_tensor)
        
        # 예측 결과 처리
        lane_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        lane_mask_resized = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
        
        # 차선 영역 표시
        lane_overlay = np.zeros_like(frame)
        lane_overlay[:, :, 1] = lane_mask_resized  # 초록색으로 차선 표시
        result = cv2.addWeighted(frame, 1, lane_overlay, 0.5, 0)
        
        # 결과 표시
        cv2.imshow('Lane Detection (TensorFlow)', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# 메인 실행 코드
if __name__ == "__main__":
    # 학습 데이터 경로
    data_dir = "tusimple_dataset_path"
    
    # 모델 학습 (또는 저장된 모델 사용)
    try:
        model = tf.keras.models.load_model('lane_detection_unet_tf.h5')
        print("저장된 모델을 불러왔습니다.")
    except:
        print("모델을 학습합니다...")
        model, history = train_lane_detection_model(data_dir)
    
    # 실시간 차선 인식 실행
    lane_detection_realtime('lane_detection_unet_tf.h5')
```

## 3. PyTorch 기반 구현

### 3.1 필요 라이브러리 설치

```bash
#// file: "Terminal"
pip install torch torchvision opencv-python numpy matplotlib scikit-learn
```

### 3.2 U-Net 모델 구현

```python
#//file: "pytorch_lane_detection_realtime.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사용자 정의 데이터셋 클래스
class LaneDataset(Dataset):
    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.masks is not None:
            mask = self.masks[idx]
            if self.transform:
                mask = self.transform(mask)
            return image, mask
        return image

# U-Net 모델 구현
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        # 인코더 (다운샘플링)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # 디코더 (업샘플링)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # 출력층
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 인코더 경로
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 디코더 경로 + 스킵 연결
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)
        
        x = self.outc(x)
        x = self.sigmoid(x)
        
        return x

# 이미지 전처리 함수
def preprocess_image_torch(img_path, mask_path=None, img_size=(256, 512)):
    # 이미지 로드 및 리사이징
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.transpose(2, 0, 1) / 255.0  # PyTorch 형식 (C, H, W)로 변환 및 정규화
    
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=0) / 255.0  # (1, H, W) 형태로 변환
        return torch.FloatTensor(img), torch.FloatTensor(mask)
    
    return torch.FloatTensor(img)

# 데이터 로딩 함수 (예시)
def load_data_torch(data_dir, img_size=(256, 512)):
    images = []
    masks = []
    
    # 실제 구현에서는 TuSimple 데이터셋 구조에 맞게 수정 필요
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                img, mask = preprocess_image_torch(img_path, mask_path, img_size)
                images.append(img)
                masks.append(mask)
    
    return images, masks

# 모델 학습 함수
def train_lane_detection_model_torch(data_dir, img_size=(256, 512), batch_size=8, epochs=50):
    # 데이터 로드
    images, masks = load_data_torch(data_dir, img_size)
    
    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = LaneDataset(X_train, y_train)
    val_dataset = LaneDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델 생성
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 학습 과정
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            
            # 손실 계산
            loss = criterion(outputs, masks)
            
            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # 에폭별 평균 손실 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lane_detection_unet_torch.pth')
            print(f'Epoch {epoch+1}: 개선된 모델을 저장했습니다.')
    
    return model

# 예측 및 시각화 함수
def predict_and_visualize_torch(model_path, image_path, img_size=(256, 512)):
    # 모델 로드
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    
    # PyTorch 텐서로 변환
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        pred_mask = model(img_tensor)
        pred_mask = pred_mask.cpu().squeeze().numpy()
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title('원본 이미지')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('예측된 차선 마스크')
    
    plt.subplot(1, 3, 3)
    # 원본 이미지에 예측 마스크 오버레이
    overlay = img_resized.copy()
    mask = (pred_mask > 0.5).astype(np.uint8)
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[:, :, 1] = mask * 255  # 초록색으로 마스크 표시
    blended = cv2.addWeighted(overlay, 0.7, overlay_mask, 0.3, 0)
    plt.imshow(blended)
    plt.title('차선 인식 결과')
    
    plt.tight_layout()
    plt.show()
```

### 3.3 실시간 차선 인식 구현

```python
#//file: "pytorch_lane_detection_realtime.py
def lane_detection_realtime_torch(model_path, camera_index=0, img_size=(256, 512)):
    # 저장된 모델 로드
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (img_size[1], img_size[0]))
        
        # PyTorch 텐서로 변환
        input_tensor = torch.from_numpy(resized_frame.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.cpu().squeeze().numpy()
        
        # 예측 결과 처리
        lane_mask = (prediction > 0.5).astype(np.uint8) * 255
        lane_mask_resized = cv2.resize(lane_mask, (frame.shape[1], frame.shape[0]))
        
        # 차선 영역 표시
        lane_overlay = np.zeros_like(frame)
        lane_overlay[:, :, 1] = lane_mask_resized  # 초록색으로 차선 표시
        result = cv2.addWeighted(frame, 1, lane_overlay, 0.5, 0)
        
        # 결과 표시
        cv2.imshow('Lane Detection (PyTorch)', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# 메인 실행 코드
if __name__ == "__main__":
    # 학습 데이터 경로
    data_dir = "tusimple_dataset_path"
    
    # 모델 학습 (또는 저장된 모델 사용)
    try:
        model_state_dict = torch.load('lane_detection_unet_torch.pth')
        model = UNet(n_channels=3, n_classes=1).to(device)
        model.load_state_dict(model_state_dict)
        print("저장된 모델을 불러왔습니다.")
    except:
        print("모델을 학습합니다...")
        model = train_lane_detection_model_torch(data_dir)
    
    # 실시간 차선 인식 실행
    lane_detection_realtime_torch('lane_detection_unet_torch.pth')
```


## 4. 라즈베리파이에서의 최적화 방법

- 라즈베리파이와 같은 임베디드 디바이스에서 딥러닝 모델을 실행할 때는 계산 자원의 한계를 고려해야 함
- 다음과 같은 최적화 방법을 적용하면 성능을 향상시킬 수 있음

### 4.1 모델 경량화 기법

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

### 4.2 추론 속도 향상 기법

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

## 5. 실제 자율주행 키트에 적용하기

### 5.1 차선 인식 결과를 모터 제어에 연결

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

### 5.2 모터 제어 보정 및 전략 수립

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