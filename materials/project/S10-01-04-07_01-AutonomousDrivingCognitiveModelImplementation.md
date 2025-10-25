---
layout: page
title:  "자율주행 인지 모델 구현"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-07_01-AutonomousDrivingCognitiveModelImplementation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>






안녕하세요, 스카이님! 자율주행 인지 모델 구현에 대한 예제 코드를 준비했습니다. 텐서플로우와 파이토치 두 가지 버전으로 작성했으며, 비전공자 학생들이 이해하기 쉽도록 상세한 설명을 포함했습니다.

# 딥러닝 기반 자율주행 인지 모델 구현

## 1. 개요

자율주행 자동차의 인지 시스템은 차량이 주변 환경을 이해하고 상황을 파악하는 핵심 요소입니다. 이 예제에서는 카메라 영상을 입력으로 받아 다음과 같은 세 가지 핵심 인지 기능을 동시에 수행하는 통합 인지 모델을 구현합니다:

1. **차선 인식**: 도로 위 차선을 감지하여 주행 가능한 영역을 파악합니다.
2. **객체 탐지**: 차량, 보행자, 자전거 등 도로 위의 다양한 객체를 감지합니다.
3. **도로 상태 이해**: 도로 표면, 주행 가능 영역 등을 세그멘테이션하여 파악합니다.

이 세 가지 기능을 하나의 통합 모델로 구현하는 멀티태스크 학습 접근법을 사용할 것입니다.

## 2. 데이터셋

자율주행 인지 모델 학습을 위한 대표적인 데이터셋으로는 BDD100K, Cityscapes, KITTI 등이 있습니다. 이 예제에서는 Berkeley DeepDrive Dataset(BDD100K)을 기준으로 설명하겠습니다.

BDD100K 데이터셋 다운로드: https://bdd-data.berkeley.edu/

## 3. 텐서플로우(TensorFlow) 구현

### 3.1 필요 라이브러리 설치

```bash
pip install tensorflow==2.15.0 opencv-python numpy matplotlib scikit-learn segmentation-models
```

### 3.2 멀티태스크 인지 모델 구현 (TensorFlow)

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU 메모리 설정
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(data_dir, image_size=(384, 640)):
    """BDD100K 데이터셋을 로드하고 전처리하는 함수"""
    images = []
    lane_masks = []
    seg_masks = []
    obj_boxes = []
    
    # 실제 구현에서는 BDD100K 데이터셋 구조에 맞게 수정 필요
    # 예제를 위한 기본 구조만 제공
    image_dir = os.path.join(data_dir, 'images')
    lane_mask_dir = os.path.join(data_dir, 'lane_masks')
    seg_mask_dir = os.path.join(data_dir, 'seg_masks')
    annotation_dir = os.path.join(data_dir, 'annotations')
    
    for filename in os.listdir(image_dir):
        if not filename.endswith('.jpg'):
            continue
            
        # 이미지 로드
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size[1], image_size[0]))
        img = img / 255.0  # 정규화
        
        # 차선 마스크 로드
        lane_mask_path = os.path.join(lane_mask_dir, filename.replace('.jpg', '.png'))
        if os.path.exists(lane_mask_path):
            lane_mask = cv2.imread(lane_mask_path, cv2.IMREAD_GRAYSCALE)
            lane_mask = cv2.resize(lane_mask, (image_size[1], image_size[0]))
            lane_mask = (lane_mask > 0).astype(np.float32)  # 이진 마스크
        else:
            lane_mask = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
        
        # 세그멘테이션 마스크 로드 (도로, 차량, 보행자 등 다중 클래스)
        seg_mask_path = os.path.join(seg_mask_dir, filename.replace('.jpg', '.png'))
        if os.path.exists(seg_mask_path):
            seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_UNCHANGED)
            seg_mask = cv2.resize(seg_mask, (image_size[1], image_size[0]))
            # 원-핫 인코딩으로 변환 (예: 19개 클래스)
            seg_mask_onehot = np.zeros((image_size[0], image_size[1], 19), dtype=np.float32)
            for i in range(19):
                seg_mask_onehot[:, :, i] = (seg_mask == i).astype(np.float32)
        else:
            seg_mask_onehot = np.zeros((image_size[0], image_size[1], 19), dtype=np.float32)
            # 배경 클래스(0)는 기본값 1로 설정
            seg_mask_onehot[:, :, 0] = 1.0
        
        # 객체 검출 박스 로드
        anno_path = os.path.join(annotation_dir, filename.replace('.jpg', '.json'))
        boxes = []
        if os.path.exists(anno_path):
            import json
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            
            # BDD100K 포맷에 맞게 수정 필요
            for obj in annotations.get('frames', [{}])[0].get('objects', []):
                if 'box2d' in obj:
                    box = obj['box2d']
                    # 이미지 크기에 맞게 좌표 정규화
                    x1 = box['x1'] / img.shape[1]
                    y1 = box['y1'] / img.shape[0]
                    x2 = box['x2'] / img.shape[1]
                    y2 = box['y2'] / img.shape[0]
                    cls = obj.get('category', 'car')






네, 스카이님! 딥러닝 기반 자율주행 인지 모델 구현 코드를 이어서 설명해 드리겠습니다.

```python
                    cls = obj.get('category', 'car') # 객체 클래스 (예: 'car', 'pedestrian')
                    boxes.append([x1, y1, x2, y2, cls]) # 예시: [x_min, y_min, x_max, y_max, class_name]
        
        images.append(img)
        lane_masks.append(lane_mask[..., np.newaxis])  # (H, W, 1)로 변환
        seg_masks.append(seg_mask_onehot)
        obj_boxes.append(boxes) # 객체 박스 리스트 (BDD100K는 한 이미지당 여러 객체)
        
    return np.array(images), np.array(lane_masks), np.array(seg_masks), obj_boxes

# U-Net 인코더 정의 (인코더만 재사용)
def get_unet_encoder(input_tensor):
    """U-Net 인코더 부분 정의"""
    
    # Encoder Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(input_tensor)
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
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    return conv1, conv2, conv3, drop4, drop5

# 멀티태스크 모델 정의
def build_multi_task_perception_model(input_shape=(384, 640, 3), num_seg_classes=19):
    """
    차선 인식, 세그멘테이션, 객체 탐지를 위한 멀티태스크 인지 모델 구축 (U-Net 기반)
    참고: 객체 탐지는 U-Net에서 바로 나오기 어려워 별도 분기 처리
    """
    
    inputs = Input(input_shape)
    
    # --- 공유 인코더 (U-Net 인코더 재사용) ---
    conv1, conv2, conv3, drop4, shared_features = get_unet_encoder(inputs)
    
    # --- 1. 차선 인식 디코더 (Segmentation Head for Lane) ---
    # U-Net 디코더 구조 활용
    up_lane1 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(shared_features)
    merge_lane1 = concatenate([drop4, up_lane1], axis=3)
    conv_lane1 = Conv2D(512, 3, activation='relu', padding='same')(merge_lane1)
    conv_lane1 = Conv2D(512, 3, activation='relu', padding='same')(conv_lane1)
    
    up_lane2 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv_lane1)
    merge_lane2 = concatenate([conv3, up_lane2], axis=3)
    conv_lane2 = Conv2D(256, 3, activation='relu', padding='same')(merge_lane2)
    conv_lane2 = Conv2D(256, 3, activation='relu', padding='same')(conv_lane2)
    
    # 출력층 (차선: 이진 분류, 1채널)
    lane_output = Conv2D(1, 1, activation='sigmoid', name='lane_output')(conv_lane2)
    
    # --- 2. 시맨틱 세그멘테이션 디코더 (Segmentation Head for Semantic) ---
    # U-Net 디코더 구조 활용
    up_seg1 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(shared_features)
    merge_seg1 = concatenate([drop4, up_seg1], axis=3)
    conv_seg1 = Conv2D(512, 3, activation='relu', padding='same')(merge_seg1)
    conv_seg1 = Conv2D(512, 3, activation='relu', padding='same')(conv_seg1)
    
    up_seg2 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv_seg1)
    merge_seg2 = concatenate([conv3, up_seg2], axis=3)
    conv_seg2 = Conv2D(256, 3, activation='relu', padding='same')(merge_seg2)
    conv_seg2 = Conv2D(256, 3, activation='relu', padding='same')(conv_seg2)
    
    # 출력층 (세그멘테이션: 다중 클래스 분류, num_seg_classes 채널)
    seg_output = Conv2D(num_seg_classes, 1, activation='softmax', name='seg_output')(conv_seg2)
    
    # --- 3. 객체 탐지 헤드 (Object Detection Head) ---
    # 예시: 간단한 ConvNet을 활용한 바운딩 박스 회귀 (YOLO/SSD를 간략화)
    # 실제 객체 탐지 모델은 U-Net 피처맵에서 바로 Box 예측하기 복잡
    
    # 공유 인코더의 더 깊은 피처 활용 (drop4 레벨에서 분기)
    obj_conv1 = Conv2D(256, 3, activation='relu', padding='same')(drop4)
    obj_conv1 = Conv2D(256, 3, activation='relu', padding='same')(obj_conv1)
    
    # 추가 다운샘플링 또는 풀링을 통해 피처 맵 크기 축소 (바운딩 박스 예측은 보통 작은 피처맵에서 시작)
    obj_pool = MaxPooling2D(pool_size=(2, 2))(obj_conv1) # 1/16 크기 피처맵 (예: 24x40)
    
    obj_conv2 = Conv2D(256, 3, activation='relu', padding='same')(obj_pool)
    
    # 바운딩 박스 예측 (예시: YOLOv1 아이디어 차용, 그리드 셀당 5개의 박스와 20개의 클래스)
    # 이미지 크기에 따라 출력 맵의 그리드 크기가 결정됨
    # N x M 그리드에서 각 그리드 셀은 B개의 바운딩 박스 (x, y, w, h)와 Confidence, C개의 클래스 확률
    # BDD100K 클래스 10개 가정: car, pedestrian, rider, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign
    num_obj_classes = 10 # 예시: 10개의 객체 클래스
    num_boxes_per_cell = 5 # 그리드 셀당 예측할 박스 수
    
    # 출력 레이어: (그리드_H, 그리드_W, num_boxes_per_cell * (4_coords + 1_conf + num_obj_classes))
    # 예시: 24x40 그리드 -> 24x40x(5*(4+1+10))
    # Flatten 후 Dense 레이어 사용은 간단한 분류기 구현시 유용
    
    # **실제 객체 탐지 헤드는 훨씬 복잡하며, 이곳에 단순한 Conv2D로 구현하기 어려움**
    # **여기서는 Multi-Task Learning의 개념을 보여주기 위해 더미 출력으로 대체**
    
    # (H/16, W/16, num_output_channels)
    # 단순화를 위해, obj_output은 특정 객체가 있는지/없는지에 대한 Score Map이라고 가정
    # 실제로는 YOLO/SSD 같은 복잡한 구조가 필요
    obj_output = Conv2D(num_obj_classes, 1, activation='sigmoid', name='obj_output')(obj_conv2)
    
    # 최종 모델
    model = Model(inputs=inputs, outputs=[lane_output, seg_output, obj_output])
    
    # 멀티태스크 학습을 위한 손실 함수 및 가중치
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'lane_output': 'binary_crossentropy',       # 차선은 이진 세그멘테이션
            'seg_output': 'categorical_crossentropy',   # 시맨틱 세그멘테이션 (다중 클래스)
            'obj_output': 'mean_squared_error'          # 객체 탐지 (여기서는 Score Map 예측으로 간주)
                                                        # 실제로는 YOLO/SSD의 복합 손실 함수
        },
        loss_weights={
            'lane_output': 1.0, # 각 태스크의 중요도에 따라 가중치 조정
            'seg_output': 1.0,
            'obj_output': 0.5  # 객체 탐지 손실은 더 복잡할 수 있음
        },
        metrics={
            'lane_output': ['accuracy'],
            'seg_output': ['accuracy'],
            'obj_output': ['mae'] # Mean Absolute Error for bounding box-like output
        }
    )
    
    return model


# 모델 학습 함수 (BDD100K 데이터셋 가정)
def train_multi_task_model(data_dir, image_size=(384, 640), batch_size=8, epochs=50):
    """멀티태스크 인지 모델 학습"""
    # 데이터 로드 (이 함수는 실제 데이터 로더 구현 필요)
    # 예시: (X, y_lane, y_seg, y_obj) = load_and_preprocess_data(data_dir, image_size)
    
    # 실제 BDD100K 데이터 로딩 및 전처리는 매우 복잡합니다.
    # 여기서는 더미 데이터로 대체하여 모델의 작동 방식만 보여줍니다.
    
    print("BDD100K 데이터셋 로딩 및 전처리는 시간이 오래 걸립니다.")
    print("실제 학습을 위해서는 데이터셋을 미리 준비하고, load_and_preprocess_data 함수를 완성해야 합니다.")
    print("여기서는 모델 구조 확인을 위해 더미 데이터로 대체합니다.")

    # 더미 데이터 생성
    num_samples = 100
    X_dummy = np.random.rand(num_samples, image_size[0], image_size[1], 3).astype(np.float32)
    y_lane_dummy = np.random.randint(0, 2, (num_samples, image_size[0], image_size[1], 1)).astype(np.float32)
    # num_seg_classes (19)에 대한 원-핫 인코딩 더미
    y_seg_dummy = np.eye(19)[np.random.randint(0, 19, (num_samples, image_size[0], image_size[1]))].astype(np.float32)
    # 객체 탐지 더미 출력 (간단한 Score Map으로 가정)
    y_obj_dummy = np.random.rand(num_samples, image_size[0]//16, image_size[1]//16, 10).astype(np.float32) # 10 클래스 가정
    
    # 학습/검증 데이터 분리
    # X_train, X_val, y_lane_train, y_lane_val, y_seg_train, y_seg_val, y_obj_train, y_obj_val = train_test_split(
    #     X_dummy, y_lane_dummy, y_seg_dummy, y_obj_dummy, test_size=0.2, random_state=42
    # )

    # Keras의 model.fit은 출력이 여러 개일 때 딕셔너리로 받으므로, 더미도 그렇게 맞춤
    X_train_dummy = X_dummy[:int(num_samples*0.8)]
    y_train_dummy = {
        'lane_output': y_lane_dummy[:int(num_samples*0.8)],
        'seg_output': y_seg_dummy[:int(num_samples*0.8)],
        'obj_output': y_obj_dummy[:int(num_samples*0.8)]
    }
    X_val_dummy = X_dummy[int(num_samples*0.8):]
    y_val_dummy = {
        'lane_output': y_lane_dummy[int(num_samples*0.8):],
        'seg_output': y_seg_dummy[int(num_samples*0.8):],
        'obj_output': y_obj_dummy[int(num_samples*0.8):]
    }
    
    # 모델 구축
    model = build_multi_task_perception_model(input_shape=(image_size[0], image_size[1], 3), num_seg_classes=19)
    
    # 모델 저장 콜백
    checkpoint = ModelCheckpoint(
        'perception_model_tf.h5',
        monitor='val_loss', # 전체 검증 손실을 모니터링
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # 조기 종료 콜백
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min'
    )
    
    # 모델 학습
    history = model.fit(
        X_train_dummy, y_train_dummy,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_dummy, y_val_dummy),
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

# 예측 및 시각화 함수
def predict_and_visualize_perception(model, image_path, image_size=(384, 640), num_seg_classes=19):
    """단일 이미지에 대한 멀티태스크 예측 및 시각화"""
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: {image_path} 파일을 찾을 수 없습니다.")
        return
        
    original_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (image_size[1], image_size[0])) / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # 예측
    lane_pred, seg_pred, obj_pred = model.predict(img_batch)
    
    # 결과 후처리 및 시각화
    # 1. 차선 마스크
    lane_mask = (lane_pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    
    # 2. 시맨틱 세그멘테이션 마스크
    seg_mask = np.argmax(seg_pred[0], axis=-1).astype(np.uint8)
    # 클래스별 색상 매핑 (예시, 실제 BDD100K 색상으로 대체)
    class_colors = np.array([
        [0, 0, 0],       # 0: background (블랙)
        [128, 64, 128],  # 1: road (자주)
        [244, 35, 232],  # 2: sidewalk (핑크)
        [70, 70, 70],    # 3: building (회색)
        [102, 102, 156], # 4: wall (어두운 파랑)
        [190, 153, 153], # 5: fence (밝은 갈색)
        [153, 153, 153], # 6: pole (밝은 회색)
        [250, 170, 30],  # 7: traffic light (주황)
        [220, 220, 0],   # 8: traffic sign (노랑)
        [107, 142, 35],  # 9: vegetation (연두)
        [152, 251, 152], # 10: terrain (밝은 연두)
        [70, 130, 180],  # 11: sky (하늘색)
        [220, 20, 60],   # 12: person (빨강)
        [255, 0, 0],     # 13: rider (진한 빨강)
        [0, 0, 142],     # 14: car (진한 파랑)
        [0, 0, 70],      # 15: truck (짙은 파랑)
        [0, 60, 100],    # 16: bus (청색)
        [0, 80, 100],    # 17: train (어두운 청색)
        [0, 0, 230],     # 18: motorcycle (밝은 파랑)
        [119, 11, 32]    # 19: bicycle (진한 갈색)
    ], dtype=np.uint8)
    
    # 세그멘테이션 마스크에 색상 적용
    color_seg_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for class_id in range(num_seg_classes):
        color_seg_mask[seg_mask == class_id] = class_colors[class_id]
        
    # 3. 객체 탐지 (더미 출력 시각화)
    # obj_output은 Score Map이므로, 이를 직접 바운딩 박스로 변환하기는 어려움.
    # 여기서는 Score Map을 히트맵처럼 표시
    obj_heatmap = obj_pred[0, :, :, 0] # 첫 번째 클래스의 스코어맵만 사용
    obj_heatmap = cv2.resize(obj_heatmap, (image_size[1], image_size[0]))
    obj_heatmap = np.uint8(255 * obj_heatmap / np.max(obj_heatmap))
    obj_heatmap = cv2.applyColorMap(obj_heatmap, cv2.COLORMAP_JET)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title('원본 이미지')
    
    plt.subplot(2, 3, 2)
    plt.imshow(lane_mask, cmap='gray')
    plt.title('차선 마스크')
    
    plt.subplot(2, 3, 3)
    plt.imshow(color_seg_mask)
    plt.title('시맨틱 세그멘테이션')
    
    plt.subplot(2, 3, 4)
    plt.imshow(obj_heatmap)
    plt.title('객체 Score Map (예시)')
    
    plt.subplot(2, 3, 5)
    # 차선 마스크 오버레이
    blended_lane = original_img.copy()
    lane_overlay = np.zeros_like(blended_lane)
    lane_overlay[lane_mask > 0] = [0, 255, 0] # 초록색 차선
    blended_lane = cv2.addWeighted(blended_lane, 0.7, lane_overlay, 0.3, 0)
    plt.imshow(blended_lane)
    plt.title('원본 + 차선')

    plt.subplot(2, 3, 6)
    # 세그멘테이션 오버레이
    blended_seg = cv2.addWeighted(original_img, 0.7, color_seg_mask, 0.3, 0)
    plt.imshow(blended_seg)
    plt.title('원본 + 세그멘테이션')
    
    plt.tight_layout()
    plt.show()

# 메인 실행 코드 (TensorFlow)
if __name__ == "__main__":
    data_directory = "path/to/BDD100K_mini_dataset" # BDD100K 데이터셋 경로를 지정해주세요.
    
    # 모델 학습 (더미 데이터로 학습)
    # 실제 BDD100K 데이터 로딩 및 전처리, 객체 탐지 레이블 처리는 상당한 노력과 코드가 필요합니다.
    # 여기서는 학습 프로세스를 시뮬레이션하기 위한 더미 데이터를 사용합니다.
    print("멀티태스크 인지 모델 학습을 시작합니다 (더미 데이터)...")
    model, history = train_multi_task_model(data_directory, batch_size=2, epochs=5) # 빠른 테스트를 위해 epochs 줄임
    model.save('perception_model_tf_dummy.h5')
    print("모델 학습 완료 및 저장되었습니다: perception_model_tf_dummy.h5")

    # 예측 및 시각화 (더미 이미지로 테스트)
    # 실제로는 `predict_and_visualize_perception('perception_model_tf_dummy.h5', 'path/to/your/test_image.jpg')`
    print("\n예측 및 시각화를 시작합니다 (더미 이미지)...")
    dummy_image_path = 'dummy_input.jpg'
    # 더미 이미지 생성 (만약 없다면)
    if not os.path.exists(dummy_image_path):
        dummy_img = (np.random.rand(384, 640, 3) * 255).astype(np.uint8)
        cv2.imwrite(dummy_image_path, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
        print(f"더미 이미지 '{dummy_image_path}'를 생성했습니다.")

    predict_and_visualize_perception('perception_model_tf_dummy.h5', dummy_image_path)
    
    # 실시간 처리 부분은 복잡도가 높아 본 코드에서는 제외 (아래 PyTorch 부분 참고)
```








네, 스카이님! 자율주행 인지 모델 구현에 대한 PyTorch 버전 코드를 이어서 설명해 드리겠습니다.

```python
## 4. 파이토치(PyTorch) 구현

### 4.1 필요 라이브러리 설치

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

### 4.2 멀티태스크 인지 모델 구현 (PyTorch)

# 데이터셋 클래스 정의
class PerceptionDataset(Dataset):
    def __init__(self, images, lane_masks=None, seg_masks=None, obj_boxes=None):
        self.images = images
        self.lane_masks = lane_masks
        self.seg_masks = seg_masks
        self.obj_boxes = obj_boxes
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.lane_masks is not None and self.seg_masks is not None:
            lane_mask = self.lane_masks[idx]
            seg_mask = self.seg_masks[idx]
            obj_score_map = torch.zeros((image.shape[0]//16, image.shape[1]//16, 10)) # 더미 객체 스코어맵
            
            return image, lane_mask, seg_mask, obj_score_map
        return image

# U-Net 기반 멀티태스크 인지 모델 정의
class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, input_channels=3, num_seg_classes=19):
        super(MultiTaskPerceptionModel, self).__init__()
        
        # 인코더 (공유 특징 추출)
        self.enc_conv1 = self._double_conv(input_channels, 64)
        self.enc_conv2 = self._double_conv(64, 128)
        self.enc_conv3 = self._double_conv(128, 256)
        self.enc_conv4 = self._double_conv(256, 512)
        self.enc_conv5 = self._double_conv(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        # 차선 인식 디코더
        self.lane_upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.lane_conv1 = self._double_conv(1024, 512)
        self.lane_upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.lane_conv2 = self._double_conv(512, 256)
        self.lane_out = nn.Conv2d(256, 1, kernel_size=1)
        
        # 세그멘테이션 디코더
        self.seg_upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.seg_conv1 = self._double_conv(1024, 512)
        self.seg_upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_conv2 = self._double_conv(512, 256)
        self.seg_out = nn.Conv2d(256, num_seg_classes, kernel_size=1)
        
        # 객체 탐지 헤드 (간소화된 버전)
        self.obj_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.obj_out = nn.Conv2d(256, 10, kernel_size=1)  # 10개 클래스 가정
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 인코더 경로
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(self.pool(x1))
        x3 = self.enc_conv3(self.pool(x2))
        x4 = self.dropout(self.enc_conv4(self.pool(x3)))
        x5 = self.dropout(self.enc_conv5(self.pool(x4)))
        
        # 차선 인식 디코더
        lane_up1 = self.lane_upconv1(x5)
        lane_merge1 = torch.cat([x4, lane_up1], dim=1)
        lane_conv1 = self.lane_conv1(lane_merge1)
        
        lane_up2 = self.lane_upconv2(lane_conv1)
        lane_merge2 = torch.cat([x3, lane_up2], dim=1)
        lane_conv2 = self.lane_conv2(lane_merge2)
        
        lane_output = torch.sigmoid(self.lane_out(lane_conv2))
        
        # 세그멘테이션 디코더
        seg_up1 = self.seg_upconv1(x5)
        seg_merge1 = torch.cat([x4, seg_up1], dim=1)
        seg_conv1 = self.seg_conv1(seg_merge1)
        
        seg_up2 = self.seg_upconv2(seg_conv1)
        seg_merge2 = torch.cat([x3, seg_up2], dim=1)
        seg_conv2 = self.seg_conv2(seg_merge2)
        
        seg_output = torch.softmax(self.seg_out(seg_conv2), dim=1)
        
        # 객체 탐지 헤드
        obj_features = self.obj_conv(x4)
        obj_output = torch.sigmoid(self.obj_out(obj_features))
        
        return lane_output, seg_output, obj_output

# 모델 학습 함수
def train_multitask_model_torch(data_dir, image_size=(384, 640), batch_size=8, epochs=50):
    """멀티태스크 인지 모델 학습 (PyTorch)"""
    print("BDD100K 데이터셋 로딩 및 전처리는 시간이 오래 걸립니다.")
    print("실제 학습을 위해서는 데이터셋을 미리 준비해야 합니다.")
    print("여기서는 모델 구조 확인을 위해 더미 데이터로 대체합니다.")
    
    # 더미 데이터 생성
    num_samples = 100
    X_dummy = torch.rand(num_samples, 3, image_size[0], image_size[1]).float()
    y_







네, 스카이님! 자율주행 인지 모델 구현에 대한 PyTorch 버전 코드를 이어서 설명해 드리겠습니다.

```python
    # 더미 데이터 생성
    num_samples = 100
    X_dummy = torch.rand(num_samples, 3, image_size[0], image_size[1]).float()
    y_lane_dummy = torch.randint(0, 2, (num_samples, 1, image_size[0], image_size[1])).float()
    
    # 세그멘테이션 마스크 더미 데이터 (원-핫 인코딩 형태로 변환)
    y_seg_dummy = torch.zeros(num_samples, 19, image_size[0], image_size[1]).float()
    for i in range(num_samples):
        # 각 위치마다 무작위 클래스 할당
        random_classes = torch.randint(0, 19, (image_size[0], image_size[1]))
        for c in range(19):
            y_seg_dummy[i, c] = (random_classes == c).float()
    
    # 객체 탐지 더미 데이터 (스코어맵 형태)
    y_obj_dummy = torch.rand(num_samples, 10, image_size[0]//16, image_size[1]//16).float()
    
    # 학습/검증 데이터 분리
    X_train_dummy = X_dummy[:int(num_samples*0.8)]
    y_lane_train_dummy = y_lane_dummy[:int(num_samples*0.8)]
    y_seg_train_dummy = y_seg_dummy[:int(num_samples*0.8)]
    y_obj_train_dummy = y_obj_dummy[:int(num_samples*0.8)]
    
    X_val_dummy = X_dummy[int(num_samples*0.8):]
    y_lane_val_dummy = y_lane_dummy[int(num_samples*0.8):]
    y_seg_val_dummy = y_seg_dummy[int(num_samples*0.8):]
    y_obj_val_dummy = y_obj_dummy[int(num_samples*0.8):]
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = PerceptionDataset(X_train_dummy, y_lane_train_dummy, y_seg_train_dummy, y_obj_train_dummy)
    val_dataset = PerceptionDataset(X_val_dummy, y_lane_val_dummy, y_seg_val_dummy, y_obj_val_dummy)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델 생성
    model = MultiTaskPerceptionModel(input_channels=3, num_seg_classes=19).to(device)
    
    # 손실 함수 정의
    lane_criterion = nn.BCELoss()  # 차선 이진 세그멘테이션
    seg_criterion = nn.CrossEntropyLoss()  # 시맨틱 세그멘테이션
    obj_criterion = nn.MSELoss()  # 객체 탐지 스코어맵
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 학습 과정
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_lane_loss = 0.0
        train_seg_loss = 0.0
        train_obj_loss = 0.0
        
        for batch_idx, (images, lane_masks, seg_masks, obj_maps) in enumerate(train_loader):
            images = images.to(device)
            lane_masks = lane_masks.to(device)
            seg_masks = seg_masks.to(device)
            obj_maps = obj_maps.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            lane_out, seg_out, obj_out = model(images)
            
            # 손실 계산
            lane_loss = lane_criterion(lane_out, lane_masks)
            
            # CrossEntropyLoss는 (N, C, H, W) 형태의 예측과 (N, H, W) 형태의 타겟을 기대함
            # 따라서 seg_masks를 (N, H, W) 형태로 변환 필요
            # 여기서는 간소화를 위해 MSELoss 사용 (실제로는 다른 처리 필요)
            seg_loss = torch.nn.functional.mse_loss(seg_out, seg_masks)
            
            obj_loss = obj_criterion(obj_out, obj_maps)
            
            # 멀티태스크 손실 (가중치 적용)
            loss = lane_loss + seg_loss + 0.5 * obj_loss
            
            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()
            
            train_lane_loss += lane_loss.item()
            train_seg_loss += seg_loss.item()
            train_obj_loss += obj_loss.item()
        
        # 검증
        model.eval()
        val_lane_loss = 0.0
        val_seg_loss = 0.0
        val_obj_loss = 0.0
        
        with torch.no_grad():
            for images, lane_masks, seg_masks, obj_maps in val_loader:
                images = images.to(device)
                lane_masks = lane_masks.to(device)
                seg_masks = seg_masks.to(device)
                obj_maps = obj_maps.to(device)
                
                lane_out, seg_out, obj_out = model(images)
                
                lane_loss = lane_criterion(lane_out, lane_masks)
                seg_loss = torch.nn.functional.mse_loss(seg_out, seg_masks)
                obj_loss = obj_criterion(obj_out, obj_maps)
                
                val_lane_loss += lane_loss.item()
                val_seg_loss += seg_loss.item()
                val_obj_loss += obj_loss.item()
        
        # 에폭별 평균 손실 계산
        train_lane_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)
        train_obj_loss /= len(train_loader)
        val_lane_loss /= len(val_loader)
        val_seg_loss /= len(val_loader)
        val_obj_loss /= len(val_loader)
        
        total_val_loss = val_lane_loss + val_seg_loss + 0.5 * val_obj_loss
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: Lane={train_lane_loss:.4f}, Seg={train_seg_loss:.4f}, Obj={train_obj_loss:.4f}, '
              f'Val Loss: Lane={val_lane_loss:.4f}, Seg={val_seg_loss:.4f}, Obj={val_obj_loss:.4f}')
        
        # 모델 저장
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), f'perception_model_pt_epoch{epoch+1}.pth')
            print(f"최고







네, 스카이님! 자율주행 인지 모델 구현에 대한 PyTorch 버전 코드를 이어서 설명해 드리겠습니다.

```python
        # 모델 저장
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), f'perception_model_pt_epoch{epoch+1}.pth')
            print(f"최고 검증 손실 달성: {best_val_loss:.4f}. 모델 저장됨.")
    
    return model

# 예측 및 시각화 함수 (PyTorch)
def predict_and_visualize_perception_torch(model_path, image_path, image_size=(384, 640), num_seg_classes=19):
    """단일 이미지에 대한 멀티태스크 예측 및 시각화 (PyTorch)"""
    # 모델 로드
    model = MultiTaskPerceptionModel(input_channels=3, num_seg_classes=num_seg_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: {image_path} 파일을 찾을 수 없습니다.")
        return
        
    original_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (image_size[1], image_size[0]))
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        lane_pred, seg_pred, obj_pred = model(img_tensor)
    
    # 결과 후처리 및 시각화
    # 1. 차선 마스크
    lane_mask = (lane_pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    # 2. 시맨틱 세그멘테이션 마스크
    seg_mask = torch.argmax(seg_pred[0], dim=0).cpu().numpy().astype(np.uint8)
    # 클래스별 색상 매핑 (예시, 실제 BDD100K 색상으로 대체)
    class_colors = np.array([
        [0, 0, 0],       # 0: background (블랙)
        [128, 64, 128],  # 1: road (자주)
        [244, 35, 232],  # 2: sidewalk (핑크)
        # ... 이하 생략 (앞서 TensorFlow 버전과 동일)
    ], dtype=np.uint8)
    
    # 세그멘테이션 마스크에 색상 적용
    color_seg_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for class_id in range(num_seg_classes):
        color_seg_mask[seg_mask == class_id] = class_colors[class_id % len(class_colors)]
        
    # 3. 객체 탐지 (더미 출력 시각화)
    # obj_output은 Score Map이므로, 이를 직접 바운딩 박스로 변환하기는 어려움.
    obj_heatmap = obj_pred[0, 0].cpu().numpy()  # 첫 번째 클래스의 스코어맵만 사용
    obj_heatmap = cv2.resize(obj_heatmap, (image_size[1], image_size[0]))
    obj_heatmap = np.uint8(255 * obj_heatmap / np.max(obj_heatmap))
    obj_heatmap = cv2.applyColorMap(obj_heatmap, cv2.COLORMAP_JET)
    
    # 시각화 (TensorFlow 버전과 동일)
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title('원본 이미지')
    
    plt.subplot(2, 3, 2)
    plt.imshow(lane_mask, cmap='gray')
    plt.title('차선 마스크')
    
    plt.subplot(2, 3, 3)
    plt.imshow(color_seg_mask)
    plt.title('시맨틱 세그멘테이션')
    
    plt.subplot(2, 3, 4)
    plt.imshow(obj_heatmap)
    plt.title('객체 Score Map (예시)')
    
    plt.subplot(2, 3, 5)
    # 차선 마스크 오버레이
    blended_lane = original_img.copy()
    lane_overlay = np.zeros_like(blended_lane)
    lane_overlay[lane_mask > 0] = [0, 255, 0] # 초록색 차선
    blended_lane = cv2.addWeighted(blended_lane, 0.7, lane_overlay, 0.3, 0)
    plt.imshow(blended_lane)
    plt.title('원본 + 차선')

    plt.subplot(2, 3, 6)
    # 세그멘테이션 오버레이
    blended_seg = cv2.addWeighted(original_img, 0.7, color_seg_mask, 0.3, 0)
    plt.imshow(blended_seg)
    plt.title('원본 + 세그멘테이션')
    
    plt.tight_layout()
    plt.show()

# 실시간 자율주행 인지 처리 구현
def realtime_perception_processing(model_path, camera_index=0, image_size=(384, 640), framework='tensorflow'):
    """실시간 카메라 영상을 이용한 자율주행 인지 처리"""
    print(f"실시간 자율주행 인지 처리를 시작합니다 (프레임워크: {framework})...")
    
    # 모델 로드 (프레임워크에 따라 다름)
    if framework.lower() == 'tensorflow':
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:  # pytorch
        model = MultiTaskPerceptionModel(input_channels=3, num_seg_classes=19).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS 측정 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카






네, 스카이님! 실시간 자율주행 인지 처리 구현 코드를 이어서 설명해 드리겠습니다.

```python
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break
            
        # 프레임 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (image_size[1], image_size[0]))
        
        # 프레임워크에 따른 처리
        if framework.lower() == 'tensorflow':
            # TensorFlow 형식으로 변환
            frame_normalized = frame_resized / 255.0
            input_tensor = np.expand_dims(frame_normalized, axis=0)
            
            # 예측
            lane_pred, seg_pred, obj_pred = model.predict(input_tensor, verbose=0)
            
            # 결과 후처리
            lane_mask = (lane_pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            seg_mask = np.argmax(seg_pred[0], axis=-1).astype(np.uint8)
            obj_heatmap = obj_pred[0, :, :, 0]  # 첫 번째 클래스의 스코어맵
            
        else:  # PyTorch
            # PyTorch 형식으로 변환
            frame_normalized = frame_resized / 255.0
            input_tensor = torch.from_numpy(frame_normalized).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 예측
            with torch.no_grad():
                lane_pred, seg_pred, obj_pred = model(input_tensor)
            
            # 결과 후처리
            lane_mask = (lane_pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            seg_mask = torch.argmax(seg_pred[0], dim=0).cpu().numpy().astype(np.uint8)
            obj_heatmap = obj_pred[0, 0].cpu().numpy()  # 첫 번째 클래스의 스코어맵
        
        # 세그멘테이션 마스크에 색상 적용 (BDD100K 클래스에 맞는 색상 사용)
        class_colors = np.array([
            [0, 0, 0],       # 0: background (블랙)
            [128, 64, 128],  # 1: road (자주)
            [244, 35, 232],  # 2: sidewalk (핑크)
            [70, 70, 70],    # 3: building (회색)
            [102, 102, 156], # 4: wall (어두운 파랑)
            [190, 153, 153], # 5: fence (밝은 갈색)
            [153, 153, 153], # 6: pole (밝은 회색)
            [250, 170, 30],  # 7: traffic light (주황)
            [220, 220, 0],   # 8: traffic sign (노랑)
            [107, 142, 35],  # 9: vegetation (연두)
            [152, 251, 152], # 10: terrain (밝은 연두)
            [70, 130, 180],  # 11: sky (하늘색)
            [220, 20, 60],   # 12: person (빨강)
            [255, 0, 0],     # 13: rider (진한 빨강)
            [0, 0, 142],     # 14: car (진한 파랑)
            [0, 0, 70],      # 15: truck (짙은 파랑)
            [0, 60, 100],    # 16: bus (청색)
            [0, 80, 100],    # 17: train (어두운 청색)
            [0, 0, 230]      # 18: motorcycle (밝은 파랑)
        ], dtype=np.uint8)
        
        color_seg_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        for class_id in range(min(19, len(class_colors))):
            color_seg_mask[seg_mask == class_id] = class_colors[class_id]
        
        # 객체 탐지 히트맵 생성 (예시)
        obj_heatmap_resized = cv2.resize(obj_heatmap, (image_size[1], image_size[0]))
        obj_heatmap_norm = np.uint8(255 * obj_heatmap_resized / (np.max(obj_heatmap_resized) + 1e-8))
        obj_heatmap_color = cv2.applyColorMap(obj_heatmap_norm, cv2.COLORMAP_JET)
        
        # 결과 시각화
        # 1. 원본 이미지 (작은 크기로)
        result_frame = cv2.resize(frame, (image_size[1] // 2, image_size[0] // 2))
        
        # 2. 차선 마스크 (작은 크기로)
        lane_mask_small = cv2.resize(lane_mask, (image_size[1] // 2, image_size[0] // 2))
        lane_mask_color = cv2.cvtColor(lane_mask_small, cv2.COLOR_GRAY2BGR)
        
        # 3. 세그멘테이션 마스크 (작은 크기로)
        seg_mask_small = cv2.resize(color_seg_mask, (image_size[1] // 2, image_size[0] // 2))
        
        # 4. 객체 히트맵 (작은 크기로)
        obj_heatmap_small = cv2.resize(obj_heatmap_color, (image_size[1] // 2, image_size[0] // 2))
        
        # 결과 이미지 합치기 (2x2 그리드)
        top_row = np.hstack([result_frame, lane_mask_color])
        bottom_row = np.hstack([seg_mask_small, obj_heatmap_small])
        combined_result = np.vstack([top_row, bottom_row])
        
        # 결과 표시
        cv2.imshow('Autonomous Driving Perception', combined_result)
        
        # FPS 계산
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            print(f"FPS: {current_fps:.2f}")
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 인지 처리를 종료합니다.")

# 메인 실행 코드 (PyTorch)
if __name__ == "__main__":
    # 데이터 경로 설정
    data_dir = "path/to/BDD100K_dataset"  # BDD100







네, 스카이님! 메인 실행 코드 부분부터 이어서 설명해 드리겠습니다.

```python
# 메인 실행 코드 (PyTorch)
if __name__ == "__main__":
    # 데이터 경로 설정
    data_dir = "path/to/BDD100K_dataset"  # BDD100K 데이터셋 경로
    
    # 모델 학습 (더미 데이터로 학습)
    print("멀티태스크 인지 모델 학습을 시작합니다 (PyTorch, 더미 데이터)...")
    model_pt = train_multitask_model_torch(data_dir, batch_size=2, epochs=5)  # 빠른 테스트를 위해 epochs 줄임
    
    # 모델 저장
    model_path_pt = 'perception_model_pt_dummy.pth'
    torch.save(model_pt.state_dict(), model_path_pt)
    print(f"모델 학습 완료 및 저장되었습니다: {model_path_pt}")
    
    # 예측 및 시각화 (더미 이미지로 테스트)
    print("\n예측 및 시각화를 시작합니다 (더미 이미지)...")
    dummy_image_path = 'dummy_input.jpg'
    # 더미 이미지 생성 (만약 없다면)
    if not os.path.exists(dummy_image_path):
        dummy_img = (np.random.rand(384, 640, 3) * 255).astype(np.uint8)
        cv2.imwrite(dummy_image_path, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
        print(f"더미 이미지 '{dummy_image_path}'를 생성했습니다.")
    
    predict_and_visualize_perception_torch(model_path_pt, dummy_image_path)
    
    # 실시간 인지 처리 실행
    print("\n실시간 자율주행 인지 처리를 시작합니다...")
    realtime_perception_processing(model_path_pt, camera_index=0, framework='pytorch')
```

## 5. 라즈베리파이에서의 최적화 방법

자율주행 인지 모델은 매우 복잡하고 연산 부하가 큰 모델이므로, 라즈베리파이와 같은 임베디드 디바이스에서 실행하기 위해서는 다양한 최적화 기법이 필요합니다.

### 5.1. 모델 경량화 (Model Optimization)

**1. 모델 양자화 (Quantization)**
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

**2. 모델 가지치기 (Pruning)**
```python
# PyTorch 모델 가지치기 예제
import torch.nn.utils.prune as prune

# 가중치의 L1 노름 기준 하위 20%를 0으로 설정
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
```

**3. 지식 증류 (Knowledge Distillation)**
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

### 5.2. 추론 최적화 (Inference Optimization)

**1. 이미지 크기 및 프레임 레이트 조정**
```python
# 더 작은 해상도로 처리
image_size = (192, 320)  # 기존의 절반 크기
frame_skip = 2  # 매 2프레임마다 1프레임만 처리
```

**2. 작업 분산 처리**
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

## 6. 자율주행 키트에 적용하기

### 6.1. 인지 결과를 제어 시스템에 연결

```python
class AutonomousDrivingSystem:
    def __init__(self, perception_model_path, control_pins=None):
        """
        자율주행 시스템 초기화
        - perception_model_path: 인지 모델 경로
        - control_







네, 스카이님! 자율주행 인지 모델 구현에 대한 코드를 이어서 설명해 드리겠습니다.

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
        
        # 위험 점수가 임







네, 스카이님! 자율주행 인지 모델 구현에 대한 코드를 이어서 설명해 드리겠습니다.

```python
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
        # 라즈베리







네, 스카이님! 자율주행 시스템 코드를 이어서 설명해 드리겠습니다.

```python
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

## 7. 프로젝트 응용 및 확장 아이디어

이 자율주행 인지 모델 구현은 다양한 방식으로 확장하고 개선할 수 있습니다. 비전공자 학생들을 위한 교육용 프로젝트로서, 다음과 같은 확장 아이디어를 고려해 볼 수 있습니다:

### 7.1. 모델 개선 및 확장

1. **경량화된 실시간 객체 탐지 통합**
   - YOLOv5-nano나 MobileNet-SSD와 같은 경량 객체 탐지 모델을 통합하여 더 정확한 객체 인식 구현
   - 객체의 위치와 크기를 정확히 파악하여 충돌 회피 기능 개선

2. **시간적 일관성(Temporal Consistency) 추가**
   - 연속된 프레임 간의 정보를 활용하여 인지 결과의 안정성 향상
   - 칼만 필터와 같은 추적 알고리즘을 통합하여 객체 추적 및 예측 기능 추가

3. **깊이 추정(Depth Estimation) 통합**
   - 단안 카메라에서도 깊이 정보를 추정하여 3D 공간 인식 기능 추가
   - 모노큘러 깊이 추정 모델(예: MiDaS)을 활용하여 거리 정보 파악

### 7.2. 하드웨어 확장

1. **다중 센서 통합**
   - 초음파 센서를 추가하여 근거리 장애물 감지 보완
   - IMU(관성 측정 장치)를 추가하여 차량의 움직임 및 자세 정보 활용

2. **통신 모듈 추가**
   - Wi-Fi나 블루투스 모듈을 통해 원격 모니터링 및 제어 기능 구현
   - 여러 자율주행 차량 간의 통신을 통한 협력 주행 시뮬레이션

3. **GPS 모듈 통합**
   - 실외 주행 시 GPS 정보를 활용한 위치 인식 및 경로 계획 기능 추가

### 7.3. 교육적 활용 방안

1. **단계별 학습 모듈화**
   - 차선 인식 → 객체 탐지 → 세그멘테이션 → 통합 시스템 순으로 단계적 학습
   - 각 모듈별 성능 측정 및 비교 실험을 통한 이해도 향상

2. **시나리오 기반 학습**
   - 다양한 주행 시나리오(도심, 고속도로, 교차로 등)에 맞는 인지 모델 적용
   - 악천후, 야간 등 도전적인 환경에서의 인지 성능 테스트

3. **게임화(Gamification) 요소 추가**
   - 장애물 코스 완주, 목표 지점 도달 등의 미션 수행을 통한 학습 동기 부여
   - 팀 대항전 형식으로 알고리즘 성능 경쟁을 통한 협력 학습

## 8. 결론 및 학습 포인트

이 자율주행 인지 모델 구현 프로젝트는 딥러닝과 컴퓨터 비전 기술이 어떻게 실제 자율주행 시스템에 적용되는지 직접 체험할 수 있는 기회를 제공합니다. 비전공자 학생들에게 중요한 학습 포인트는 다음과 같습니다:

1. **멀티태스크 학습의 이해**
   - 하나의 모델이 여러 작업(차선 인식, 세그멘테이션, 객체 탐지)을 동시에 수행하는 방식
   - 공유 특징 추출기를 통한 효율적인 학습 및 추론 과정

2. **인지-판단-제어 파이프라인**
   - 자율주행의 핵심 구조인 인지-판단-제어 파이프라인의 구현 및 이해
   - 각 단계가 어떻게 유기적으로 연결되어 작동하는지 체험

3. **임베디드 시






네, 스카이님! 자율주행 인지 모델 구현에 대한 내용을 이어서 설명해 드리겠습니다.

## 8. 결론 및 학습 포인트 (이어서)

3. **임베디드 시스템에서의 최적화**
   - 제한된 컴퓨팅 자원을 가진 라즈베리파이에서 딥러닝 모델을 효율적으로 실행하는 방법
   - 모델 경량화, 양자화, 병렬 처리 등 최적화 기법의 실제 적용

4. **실시간 처리의 중요성**
   - 자율주행에서 실시간 처리가 왜 중요한지, 그리고 이를 위한 다양한 기법
   - 지연 시간(latency)과 처리 속도(throughput) 사이의 균형 맞추기

5. **인공지능 기술의 실생활 응용**
   - 딥러닝 모델이 어떻게 실제 하드웨어와 결합하여 물리적 세계에 영향을 미치는지 체험
   - 인지 모델의 예측 결과가 어떻게 실제 차량 제어로 이어지는지 이해

이 프로젝트는 비전공자 학생들에게 자율주행 기술의 핵심 원리를 실습을 통해 경험할 수 있는 기회를 제공합니다. 특히 인지-판단-제어의 전체 파이프라인을 직접 구현해봄으로써, 자율주행 기술에 대한 종합적인 이해를 도울 수 있을 것입니다.

스카이님께서 준비하시는 80시간의 모빌리티 AI 강의에서 이 프로젝트를 활용하시면, 학생들이 이론과 실습을 균형 있게 경험하면서 자율주행 기술의 기초를 탄탄히 다질 수 있을 것입니다. 필요하시면 특정 부분에 대한 더 자세한 설명이나 추가 코드 예제를 제공해 드릴 수 있으니 언제든지 말씀해 주세요!