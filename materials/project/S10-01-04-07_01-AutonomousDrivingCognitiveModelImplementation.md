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
        

## 2. PyTorch 기반 Segmentation 구현

### 2.1 패키지 가져오기

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
```

### 2.2 기반 환경 설정

- 데이터셋 및 사전학습모델 경로 설정

```python
# 데이터셋 루트 폴더 (Cityscapes Mini Dataset의 root 디렉토리로 변경)
# 예: DATA_ROOT = "/home/skyy/datasets/cityspaces" (여기서는 'cityspaces' 폴더 자체의 경로)
DATA_ROOT = "./cityspaces_mini"

# U-Net 모델의 입력 이미지 크기
INPUT_IMAGE_SIZE = 256 # Cityscapes 이미지 크기 (보통 1024x2048)를 256으로 리사이즈

# Cityscapes Dataset의 클래스 및 ID 정의 (19개 클래스)
NUM_CLASSES = 19 # (0~18) - 255는 ignore_index로 처리

# Cityscapes ID to Train ID 매핑 (Cityscapes 공식 스크립트 기반)
cityscapes_id_to_trainid = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, -1: 255 # -1은 공식적으로 맵핑되지 않는 ID (예: license plate)
}

# Cityscapes Color Palette (Visualization)
CITYSCAPES_PALETTE = {
    0: [128, 64, 128],    # road
    1: [244, 35, 232],    # sidewalk
    2: [70, 70, 70],      # building
    3: [102, 102, 156],   # wall
    4: [190, 153, 153],   # fence
    5: [153, 153, 153],   # pole
    6: [250, 170, 30],    # traffic light
    7: [220, 220, 0],     # traffic sign
    8: [107, 142, 35],    # vegetation
    9: [152, 251, 152],   # terrain
    10: [70, 130, 180],   # sky
    11: [220, 20, 60],    # person
    12: [255, 0, 0],      # rider
    13: [0, 0, 142],      # car
    14: [0, 0, 70],       # truck
    15: [0, 60, 100],     # bus
    16: [0, 80, 100],     # train
    17: [0, 0, 230],      # motorcycle
    18: [119, 11, 32],    # bicycle
    255: [0, 0, 0]        # ignore (black)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2.3 U-Net 모델 정의

- Convolutional Block

```python
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    ) 
```

- Encoder Block

```python
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2) # 2x2 Max Pooling

    def forward(self, x):
        conv_out = self.conv(x) # conv_block을 먼저 적용
        pooled_out = self.pool(conv_out) # conv_block의 출력에 Max Pooling 적용
        return conv_out, pooled_out # Conv 출력과 Pool 출력을 모두 반환
```

- Decoder Block

```python
class Decoder(nn.Module):
    def __init__(self, in_channels_dec_up, in_channels_skip, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 컨볼루션 블록의 입력 채널은 업샘플링된 특징과 스킵 커넥션 특징을 합친 것
        self.conv = conv_block(in_channels_dec_up + in_channels_skip, out_channels)

    def forward(self, x, skip_connection_feature):
        x = self.up(x) # 업샘플링

        # 스킵 커넥션 특징과 채널 차원으로 합치기
        # 이 때 x와 skip_connection_feature의 공간 해상도가 정확히 일치해야 합니다.
        if x.shape[2:] != skip_connection_feature.shape[2:]:
            skip_connection_feature = F.interpolate(skip_connection_feature, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip_connection_feature], dim=1)
        x = self.conv(x) # <--- 이 부분의 self.conv(x)
        return x
```

- U-Net Architecture

```python
class UNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Encoder Path
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder Path
        # in_channels_dec_up, in_channels_skip, out_channels 순서
        self.dec4 = Decoder(1024, 512, 512) # (bottleneck 출력 1024 -> upsample -> 1024채널) + (e4_conv 512채널) -> 512 출력
        self.dec3 = Decoder(512, 256, 256) # (d4 출력 512 -> upsample -> 512채널) + (e3_conv 256채널) -> 256 출력
        self.dec2 = Decoder(256, 128, 128) # (d3 출력 256 -> upsample -> 256채널) + (e2_conv 128채널) -> 128 출력
        self.dec1 = Decoder(128, 64, 64)   # (d2 출력 128 -> upsample -> 128채널) + (e1_conv 64채널) -> 64 출력

        # Output Layer
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1_conv, e1_pool = self.enc1(x)  # e1_conv는 skip connection용
        e2_conv, e2_pool = self.enc2(e1_pool)
        e3_conv, e3_pool = self.enc3(e2_pool)
        e4_conv, e4_pool = self.enc4(e3_pool)

        # Bottleneck
        bottleneck = self.bottleneck(e4_pool)

        # Decoder (eX_conv와 concat)
        d4 = self.dec4(bottleneck, e4_conv)
        d3 = self.dec3(d4, e3_conv)
        d2 = self.dec2(d3, e2_conv)
        d1 = self.dec1(d2, e1_conv)

        # Output
        return self.output_conv(d1)
```

#### 2.4 데이터셋 로드 및 전처리

- Cityscapes Mini Dataset 로더

```python
class CityscapesMiniDataset(Dataset):
    def __init__(self, root_dir, split='train', transform_img=None, transform_mask=None, id_to_trainid_map=None):
        self.root_dir = root_dir
        self.split = split
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.id_to_trainid_map = id_to_trainid_map if id_to_trainid_map is not None else {}

        self.image_paths = []
        self.mask_paths = []

        # 스카이님의 새로운 디렉토리 구조에 맞춤
        img_dir = os.path.join(self.root_dir, self.split, 'img')
        label_dir = os.path.join(self.root_dir, self.split, 'label')

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        # 이미지 파일 목록을 가져와서 마스크 파일과 매칭
        for img_filename in os.listdir(img_dir):
            if img_filename.endswith('.png') or img_filename.endswith('.jpg'): # jpg도 처리 가능하도록
                img_path = os.path.join(img_dir, img_filename)

                # 마스크 파일 이름이 이미지 파일 이름과 동일하고 .png 확장자를 가진다고 가정
                mask_filename = img_filename # .png 확장자일 것이므로 그대로 사용
                mask_path = os.path.join(label_dir, mask_filename)

                if os.path.exists(mask_path): # 마스크 파일이 있는지 확인
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                # else:
                #     print(f"Warning: Corresponding mask not found for {img_filename} at {mask_path}")

        print(f"Loaded {len(self.image_paths)} images for {self.split} split from Cityscapes Mini Dataset with custom structure.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Label ID 마스크는 단일 채널 (Grayscale)

        mask_np = np.array(mask, dtype=np.uint8)

        # Cityscapes trainId 매핑 (클래스 ID에 따라 정의된 매핑 사용)
        mapped_mask = np.full(mask_np.shape, 255, dtype=np.uint8) # 255로 초기화 (ignore)

        for original_id, train_id in self.id_to_trainid_map.items():
            mapped_mask[mask_np == original_id] = train_id

        mask = Image.fromarray(mapped_mask)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask
```

- 이미지 전처리: 리사이즈, 텐서 변환, 정규화

```python
transform_img = transforms.Compose([
    transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
    transforms.ToTensor(), # HWC to CWH, [0, 255] to [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 통계
])
```

- 마스크 전처리: 리사이즈, 텐서 변환 (클래스 ID이므로 정규화 없음)

```python
class ConvertMaskToTensor:
    def __call__(self, mask_pil):
        # PIL Image를 NumPy 배열로 변환하고 int64 타입으로 지정
        mask_np = np.array(mask_pil, dtype=np.int64)
        # NumPy 배열을 torch.LongTensor로 변환 (스케일링 없음)
        return torch.from_numpy(mask_np)

transform_mask = transforms.Compose([
    transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=Image.NEAREST), # 마스크는 Nearest Neighbor 보간 필수
    ConvertMaskToTensor() # 사용자 정의 변환 적용
])
```

- DataLoader 생성

```python
train_dataset = CityscapesMiniDataset(root_dir=DATA_ROOT, split='train',
                                      transform_img=transform_img, transform_mask=transform_mask,
                                      id_to_trainid_map=cityscapes_id_to_trainid)
val_dataset = CityscapesMiniDataset(root_dir=DATA_ROOT, split='val',
                                    transform_img=transform_img, transform_mask=transform_mask,
                                    id_to_trainid_map=cityscapes_id_to_trainid)

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"검증 데이터셋 크기: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
```

#### 2.5 학습 함수

```python
def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print("\n--- U-Net 모델 학습 시작 ---")
    print(f"({num_epochs} 에포크 진행, 학습 데이터 {len(train_loader)} 배치)")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_val, masks_val in val_loader:
                inputs_val = inputs_val.to(device)
                masks_val = masks_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, masks_val)
                val_loss += loss_val.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Train Loss: {running_loss / len(train_loader):.4f}, Avg Val Loss: {val_loss / len(val_loader):.4f}")
        model.train()

    print("\n--- U-Net 모델 학습 완료 ---")
    return model
```

### 2.6 추론 및 시각화 함수

```python
def visualize_segmentation(model, dataset, num_samples=5):
    model.eval()
    print(f"\n--- U-Net 모델 추론 및 시각화 시작 ({num_samples}개 샘플) ---")

    if len(dataset) < num_samples:
        print(f"경고: 데이터셋 크기({len(dataset)}개)가 요청한 샘플 수({num_samples}개)보다 작습니다. 모든 샘플을 시각화합니다.")
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(20, num_samples * 5))

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            img_tensor, true_mask_tensor = dataset[idx]

            input_batch = img_tensor.unsqueeze(0).to(device)

            output = model(input_batch)

            predicted_mask = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            true_mask_np = true_mask_tensor.cpu().numpy()
            true_color_mask = np.zeros((*true_mask_np.shape, 3), dtype=np.uint8)
            for class_id, color in CITYSCAPES_PALETTE.items():
                if class_id == 255:
                    continue
                true_color_mask[true_mask_np == class_id] = color

            predicted_color_mask = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
            for class_id, color in CITYSCAPES_PALETTE.items():
                if class_id == 255:
                    continue
                predicted_color_mask[predicted_mask == class_id] = color

            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(img_np)
            plt.title(f"Original Image")
            plt.axis('off')

            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(true_color_mask)
            plt.title("Ground Truth Mask")
            plt.axis('off')

            plt.subplot(num_samples, 3, i*3 + 3)
            blended_image = cv2.addWeighted(img_np, 0.7, predicted_color_mask, 0.3, 0)
            plt.imshow(blended_image)
            plt.title("Predicted Overlay")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("--- 추론 및 시각화 완료 ---")
```

### 2.7 메인 실행 블록

```python
# 1. 모델 생성 및 디바이스 이동
model = UNet(num_classes=NUM_CLASSES).to(device)

# 2. 모델 학습
# (예시이므로 짧게 5 에포크만 진행. 실제는 더 오래 학습해야 합니다.)
trained_model = train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001)

visualize_segmentation(trained_model, val_dataset, num_samples=5)
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
