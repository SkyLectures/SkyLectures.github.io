---
layout: page
title:  "도로 표지판 및 신호등 인식"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-06_01-RoadSignTrafficLightRecognition
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - **딥러닝 기반 도로 표지판 및 신호등 인식 시스템**
>   - 도로 표지판과 신호등 인식은 자율주행 시스템의 핵심 기능 중 하나
>   - 차량이 **도로 규칙을 준수**하고 안전하게 주행하기 위한 필수적인 요소
>   - 기능
>       - 카메라로 촬영된 영상에서 표지판과 신호등 감지
>       - 표지판과 신호등의 종류와 의미 해석
>       - 자율주행 차량이 적절한 주행 결정을 내릴 수 있도록 함
{: .common-quote}

## 1. 도로 표지판 및 신호등 인식 실습 개요

- **실습 내용**
    - 도로 표지판 분류 (교통 표지판의 종류 식별)
    - 신호등 상태 인식 (빨간불, 노란불, 초록불 감지)

- **데이터셋**
    - **도로 표지판 데이터셋**
        - 표지판 인식을 위한 대표적인 공개 데이터셋: German Traffic Sign Recognition Benchmark(GTSRB)
        - 실습에서는 GTSRB 데이터셋을 기준으로 진행
            - [GTSRB 데이터셋 다운로드](https://benchmark.ini.rub.de/gtsrb_dataset.html){: target="_blank"}
    - **신호등 데이터셋**
    - 신호등 인식을 위한 대표적인 공개 데이터셋: LISA Traffic Light Dataset, Bosch Small Traffic Lights Dataset 등
    - 실습에서는 LISA 데이터셋을 기준으로 진행
        - [LISA 데이터셋 다운로드](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset){: target="_blank"}

> - 국내 환경에서 사용하기 위해서는 국내용으로 개발된 데이터셋을 찾아서(또는 직접 구축) 적용해야 함
{: .expert-quote}


## 2. TensorFlow 기반 구현

### 2.1 필요 라이브러리 설치

```bash
#// file: "Terminal"
pip install tensorflow==2.15.0 opencv-python numpy matplotlib scikit-learn pandas
```

### 2.2 도로 표지판 분류

```python
#// file: "Tensorflow_RoadSign_TrafficLight_Recognition.py"
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU 메모리 설정
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 이미지 전처리 함수
def preprocess_image(img, target_size=(32, 32)):
    """이미지를 전처리하는 함수"""
    # 이미지 크기 조정
    img_resized = cv2.resize(img, target_size)
    # 정규화 (0-1 범위로)
    img_normalized = img_resized / 255.0
    return img_normalized

# GTSRB 데이터 로드 함수
def load_gtsrb_data(data_dir, target_size=(32, 32)):
    """GTSRB 데이터셋을 로드하는 함수"""
    images = []
    labels = []
    
    # 각 클래스 폴더 순회
    for class_id in range(43):  # GTSRB에는 43개의 클래스가 있음
        class_dir = os.path.join(data_dir, str(class_id))
        if not os.path.isdir(class_dir):
            continue
            
        # 클래스 내의 모든 이미지 파일 로드
        for img_file in os.listdir(class_dir):
            if not img_file.endswith('.ppm'):  # GTSRB는 .ppm 형식 사용
                continue
                
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # BGR을 RGB로 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 이미지 전처리
            img_processed = preprocess_image(img, target_size)
            
            images.append(img_processed)
            labels.append(class_id)
    
    return np.array(images), np.array(labels)

# CNN 모델 구축 (도로 표지판 분류용)
def build_traffic_sign_model(input_shape=(32, 32, 3), num_classes=43):
    """교통 표지판 분류를 위한 CNN 모델 구축"""
    model = Sequential([
        # 첫 번째 컨볼루션 블록
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # 두 번째 컨볼루션 블록
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # 세 번째 컨볼루션 블록
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # 분류기
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 모델 학습 함수
def train_traffic_sign_model(data_dir, batch_size=32, epochs=30):
    """교통 표지판 분류 모델을 학습하는 함수"""
    # 데이터 로드
    X, y = load_gtsrb_data(data_dir)
    
    # 클래스 레이블을 원-핫 인코딩으로 변환
    y_one_hot = to_categorical(y, num_classes=43)
    
    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    # 모델 구축
    model = build_traffic_sign_model(input_shape=(32, 32, 3), num_classes=43)
    
    # 모델 저장 콜백
    checkpoint = ModelCheckpoint(
        'traffic_sign_model_tf.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # 조기 종료 콜백
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1,
        mode='max'
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

# 표지판 클래스 이름 매핑 (GTSRB 기준)
def get_traffic_sign_class_names():
    """GTSRB 데이터셋의 클래스 이름 반환"""
    return {
        0: '속도 제한 (20km/h)',
        1: '속도 제한 (30km/h)',
        2: '속도 제한 (50km/h)',
        3: '속도 제한 (60km/h)',
        4: '속도 제한 (70km/h)',
        5: '속도 제한 (80km/h)',
        6: '속도 제한 종료 (80km/h)',
        7: '속도 제한 (100km/h)',
        8: '속도 제한 (120km/h)',
        9: '추월 금지',
        10: '3.5톤 이상 차량 추월 금지',
        11: '교차로 우선권',
        12: '우선 도로',
        13: '양보',
        14: '정지',
        15: '차량 통행 금지',
        16: '3.5톤 이상 차량 통행 금지',
        17: '진입 금지',
        18: '일반 주의',
        19: '위험한 좌회전',
        20: '위험한 우회전',
        21: '연속 커브',
        22: '도로 상태 불량',
        23: '미끄러운 도로',
        24: '우측 도로 폭 감소',
        25: '도로 공사',
        26: '교통 신호',
        27: '보행자 주의',
        28: '어린이 보호 구역',
        29: '자전거 통행',
        30: '눈 또는 빙판 주의',
        31: '야생 동물 주의',
        32: '속도 제한 및 추월 제한 종료',
        33: '우회전 필수',
        34: '좌회전 필수',
        35: '직진 필수',
        36: '직진 또는 우회전',
        37: '직진 또는 좌회전',
        38: '우측 통행',
        39: '좌측 통행',
        40: '회전 교차로',
        41: '추월 제한 종료',
        42: '3.5톤 이상 차량 추월 금지 종료'
    }
```

### 2.3 신호등 상태 인식 모델

```python
#// file: "Tensorflow_RoadSign_TrafficLight_Recognition.py"
# 신호등 상태 인식 모델 구축 (빨간불, 노란불, 초록불 구분)
def build_traffic_light_model(input_shape=(64, 64, 3), num_classes=3):
    """신호등 상태 인식을 위한 CNN 모델 구축"""
    inputs = Input(shape=input_shape)
    
    # 첫 번째 컨볼루션 블록
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 두 번째 컨볼루션 블록
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 분류기
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# LISA 데이터셋 로드 함수 (예시)
def load_lisa_traffic_light_data(data_dir, target_size=(64, 64)):
    """LISA 교통 신호등 데이터셋 로드 함수"""
    images = []
    labels = []
    
    # LISA 데이터셋 구조에 맞게 수정 필요
    classes = {'red': 0, 'yellow': 1, 'green': 2}
    
    for class_name, class_id in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, target_size)
            img_normalized = img_resized / 255.0
            
            images.append(img_normalized)
            labels.append(class_id)
    
    return np.array(images), to_categorical(np.array(labels), num_classes=3)

# 신호등 상태 인식 모델 학습 함수
def train_traffic_light_model(data_dir, batch_size=32, epochs=30):
    """신호등 상태 인식 모델을 학습하는 함수"""
    # 데이터 로드
    X, y = load_lisa_traffic_light_data(data_dir)
    
    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 구축
    model = build_traffic_light_model(input_shape=(64, 64, 3), num_classes=3)
    
    # 모델 저장 콜백
    checkpoint = ModelCheckpoint(
        'traffic_light_model_tf.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # 조기 종료 콜백
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        mode='max'
    )
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

# 신호등 클래스 이름 매핑
def get_traffic_light_class_names():
    """신호등 상태 클래스 이름 반환"""
    return {
        0: '빨간불 (정지)',
        1: '노란불 (주의)',
        2: '초록불 (진행)'
    }
```

### 2.4 실시간 표지판 및 신호등 인식 구현

```python
#// file: "Tensorflow_RoadSign_TrafficLight_Recognition.py"
import tensorflow as tf
import cv2
import numpy as np
import time

def detect_and_classify_traffic_signs_and_lights(
    sign_model_path, 
    light_model_path, 
    camera_index=0, 
    sign_size=(32, 32), 
    light_size=(64, 64)
):
    """실시간 도로 표지판 및 신호등 인식"""
    # 모델 로드
    sign_model = tf.keras.models.load_model(sign_model_path)
    light_model = tf.keras.models.load_model(light_model_path)
    
    # 클래스 이름 가져오기
    sign_class_names = get_traffic_sign_class_names()
    light_class_names = get_traffic_light_class_names()
    
    # 표지판 검출을 위한 캐스케이드 분류기 (예시)
    # 실제로는 더 정교한 객체 검출 모델(예: YOLO, SSD)을 사용하는 것이 좋습니다.
    sign_cascade = cv2.CascadeClassifier('haarcascade_traffic_sign.xml')
    light_cascade = cv2.CascadeClassifier('haarcascade_traffic_light.xml')
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 원본 프레임 복사
        result_frame = frame.copy()
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 표지판 검출
        signs = sign_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in signs:
            # 검출된 표지판 영역 추출
            sign_roi = frame[y:y+h, x:x+w]
            
            # 표지판 분류를 위한 전처리
            sign_roi_resized = cv2.resize(sign_roi, sign_size)
            sign_roi_normalized = sign_roi_resized / 255.0
            sign_roi_batch = np.expand_dims(sign_roi_normalized, axis=0)
            
            # 표지판 분류
            sign_predictions = sign_model.predict(sign_roi_batch)
            sign_class_id = np.argmax(sign_predictions[0])
            sign_confidence = np.max(sign_predictions[0])
            
            # 신뢰도가 일정 임계값 이상일 경우에만 결과 표시
            if sign_confidence > 0.7:
                sign_label = sign_class_names.get(sign_class_id, "알 수 없음")
                
                # 결과 표시
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_frame, f"{sign_label} ({sign_confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 신호등 검출
        lights = light_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in lights:
            # 검출된 신호등 영역 추출
            light_roi = frame[y:y+h, x:x+w]
            
            # 신호등 상태 분류를 위한 전처리
            light_roi_resized = cv2.resize(light_roi, light_size)
            light_roi_normalized = light_roi_resized / 255.0
            light_roi_batch = np.expand_dims(light_roi_normalized, axis=0)
            
            # 신호등 상태 분류
            light_predictions = light_model.predict(light_roi_batch)
            light_class_id = np.argmax(light_predictions[0])
            light_confidence = np.max(light_predictions[0])
            
            # 신뢰도가 일정 임계값 이상일 경우에만 결과 표시
            if light_confidence > 0.7:
                light_label = light_class_names.get(light_class_id, "알 수 없음")
                
                # 색상 설정 (빨간불: 빨강, 노란불: 노랑, 초록불: 초록)
                if light_class_id == 0:  # 빨간불
                    color = (0, 0, 255)
                elif light_class_id == 1:  # 노란불
                    color = (0, 255, 255)
                else:  # 초록불
                    color = (0, 255, 0)
                
                # 결과 표시
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_frame, f"{light_label} ({light_confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS 계산 및 표시
        fps_start_time = time.time() if not hasattr(detect_and_classify_traffic_signs_and_lights, 'fps_start_time') else detect_and_classify_traffic_signs_and_lights.fps_start_time
        fps_frame_count = 0 if not hasattr(detect_and_classify_traffic_signs_and_lights, 'fps_frame_count') else detect_and_classify_traffic_signs_and_lights.fps_frame_count
        
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            # print(f"FPS: {current_fps:.2f}")
            
            # 다음 프레임부터 사용할 값 저장 (정적 변수처럼 활용)
            detect_and_classify_traffic_signs_and_lights.fps_start_time = fps_start_time
            detect_and_classify_traffic_signs_and_lights.fps_frame_count = fps_frame_count
        
        cv2.putText(result_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # 노란색 텍스트
        
        cv2.imshow('Traffic Sign/Light Detection (TensorFlow)', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 인식 종료.")


# 메인 실행 코드 (TensorFlow)
if __name__ == "__main__":
    # ----------------------------------------------------
    # 데이터 경로 설정 (GTSRB와 LISA 데이터셋 경로를 여기에 지정)
    # 실제 데이터셋이 없다면 이 부분은 학습 없이 모델 로드만 시도
    gtsrb_data_dir = "path/to/GTSRB/Training" # GTSRB 학습 데이터셋 경로
    lisa_data_dir = "path/to/LISA_Traffic_Light_Dataset" # LISA 학습 데이터셋 경로

    # 경고: `haarcascade_traffic_sign.xml` 및 `haarcascade_traffic_light.xml` 파일이 필요합니다.
    # OpenCV 설치 경로에서 찾아오거나 직접 웹에서 다운로드해야 합니다.
    # 예를 들어, OpenCV 설치 폴더/data/haarcascades 에 있을 수 있습니다.
    # 이 예제는 이 캐스케이드 파일이 현재 작업 디렉토리에 있다고 가정합니다.
    # ----------------------------------------------------

    # 1. 교통 표지판 분류 모델 학습 또는 로드
    traffic_sign_model_path = 'traffic_sign_model_tf.h5'
    try:
        sign_model = tf.keras.models.load_model(traffic_sign_model_path)
        print("TensorFlow 교통 표지판 모델을 불러왔습니다.")
    except:
        print("TensorFlow 교통 표지판 모델을 학습합니다...")
        sign_model, sign_history = train_traffic_sign_model(gtsrb_data_dir)
        sign_model.save(traffic_sign_model_path) # 학습 후 저장

    # 2. 신호등 상태 분류 모델 학습 또는 로드
    traffic_light_model_path = 'traffic_light_model_tf.h5'
    try:
        light_model = tf.keras.models.load_model(traffic_light_model_path)
        print("TensorFlow 신호등 모델을 불러왔습니다.")
    except:
        print("TensorFlow 신호등 모델을 학습합니다...")
        light_model, light_history = train_traffic_light_model(lisa_data_dir)
        light_model.save(traffic_light_model_path) # 학습 후 저장
    
    # 3. 실시간 인식 실행
    print("\n--- 실시간 도로 표지판 및 신호등 인식을 시작합니다 (TensorFlow) ---")
    # 카메라 인덱스 0번, 표지판 크기 (32,32), 신호등 크기 (64,64)
    detect_and_classify_traffic_signs_and_lights(traffic_sign_model_path, traffic_light_model_path, 
                                                camera_index=0, sign_size=(32, 32), light_size=(64, 64))

```

## 3. PyTorch 기반 구현

### 3.1 필요 라이브러리 설치

```bash
#// file: "Terminal"
pip install torch torchvision opencv-python numpy matplotlib scikit-learn pandas
```

### 3.2 도로 표지판 및 신호등 모델

```python
#// file: "PyTorch_RoadSign_TrafficLight_Recognition.py"
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 이미지 전처리 및 데이터 로드 헬퍼 함수
def preprocess_image_torch(img, target_size=(32, 32)):
    """PyTorch용 이미지 전처리 함수"""
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    # PyTorch는 (C, H, W) 형식이므로 차원 변경
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    return img_tensor

def load_gtsrb_data_torch(data_dir, target_size=(32, 32)):
    """GTSRB 데이터셋을 PyTorch 텐서로 로드하는 함수"""
    images = []
    labels = []
    for class_id in range(43):
        class_dir = os.path.join(data_dir, str(class_id))
        if not os.path.isdir(class_dir): continue
        for img_file in os.listdir(class_dir):
            if not img_file.endswith('.ppm'): continue
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(preprocess_image_torch(img, target_size))
            labels.append(class_id)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def load_lisa_traffic_light_data_torch(data_dir, target_size=(64, 64)):
    """LISA 교통 신호등 데이터셋을 PyTorch 텐서로 로드하는 함수"""
    images = []
    labels = []
    classes = {'red': 0, 'yellow': 1, 'green': 2}
    for class_name, class_id in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(preprocess_image_torch(img, target_size))
            labels.append(class_id)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

# 사용자 정의 데이터셋 클래스
class TrafficDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# CNN 모델 구축 (교통 표지판 및 신호등 공용)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), # 입력 크기 (예: 32x32 -> 4x4, 64x64 -> 8x8)에 따라 조절 필요
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# 모델 학습 함수 (PyTorch)
def train_classification_model_torch(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    """PyTorch 분류 모델을 학습하는 함수"""
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_accuracy = 100 * correct_train / total_train
        
        # 검증
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'best_traffic_model_epoch{epoch+1}.pth')
            print(f"최고 검증 정확도 달성: {best_val_accuracy:.2f}%. 모델 저장됨.")
            
    return model


# 표지판 클래스 이름 매핑 (GTSRB 기준)
def get_traffic_sign_class_names():
    """GTSRB 데이터셋의 클래스 이름 반환"""
    return {
        0: '속도 제한 (20km/h)', 1: '속도 제한 (30km/h)', 2: '속도 제한 (50km/h)', 3: '속도 제한 (60km/h)',
        4: '속도 제한 (70km/h)', 5: '속도 제한 (80km/h)', 6: '속도 제한 종료 (80km/h)', 7: '속도 제한 (100km/h)',
        8: '속도 제한 (120km/h)', 9: '추월 금지', 10: '3.5톤 이상 차량 추월 금지', 11: '교차로 우선권',
        12: '우선 도로', 13: '양보', 14: '정지', 15: '차량 통행 금지', 16: '3.5톤 이상 차량 통행 금지',
        17: '진입 금지', 18: '일반 주의', 19: '위험한 좌회전', 20: '위험한 우회전', 21: '연속 커브',
        22: '도로 상태 불량', 23: '미끄러운 도로', 24: '우측 도로 폭 감소', 25: '도로 공사',
        26: '교통 신호', 27: '보행자 주의', 28: '어린이 보호 구역', 29: '자전거 통행',
        30: '눈 또는 빙판 주의', 31: '야생 동물 주의', 32: '속도 제한 및 추월 제한 종료',
        33: '우회전 필수', 34: '좌회전 필수', 35: '직진 필수', 36: '직진 또는 우회전',
        37: '직진 또는 좌회전', 38: '우측 통행', 39: '좌측 통행', 40: '회전 교차로',
        41: '추월 제한 종료', 42: '3.5톤 이상 차량 추월 금지 종료'
    }

# 신호등 클래스 이름 매핑
def get_traffic_light_class_names():
    """신호등 상태 클래스 이름 반환"""
    return {
        0: '빨간불 (정지)',
        1: '노란불 (주의)',
        2: '초록불 (진행)'
    }
```

### 3.3 실시간 표지판 및 신호등 인식 구현

```python
#// file: "PyTorch_RoadSign_TrafficLight_Recognition.py"
def detect_and_classify_traffic_signs_and_lights_torch(
    sign_model_state_dict_path, 
    light_model_state_dict_path, 
    camera_index=0, 
    sign_size=(32, 32), 
    light_size=(64, 64)
):
    """실시간 도로 표지판 및 신호등 인식 (PyTorch)"""
    # 모델 로드
    sign_model = SimpleCNN(num_classes=43).to(device)
    sign_model.load_state_dict(torch.load(sign_model_state_dict_path, map_location=device))
    sign_model.eval()

    light_model = SimpleCNN(num_classes=3).to(device)
    light_model.load_state_dict(torch.load(light_model_state_dict_path, map_location=device))
    light_model.eval()
    
    # 클래스 이름 가져오기
    sign_class_names = get_traffic_sign_class_names()
    light_class_names = get_traffic_light_class_names()
    
    # 표지판 및 신호등 검출을 위한 캐스케이드 분류기 (예시)
    # 실제로는 더 정교한 객체 검출 모델(예: YOLO, SSD)을 사용하는 것이 좋습니다.
    sign_cascade = cv2.CascadeClassifier('haarcascade_traffic_sign.xml')
    light_cascade = cv2.CascadeClassifier('haarcascade_traffic_light.xml')
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    # FPS 측정 변수 초기화
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ------------------ 표지판 검출 및 분류 ------------------
        signs = sign_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in signs:
            sign_roi = frame[y:y+h, x:x+w]
            
            # PyTorch 모델을 위한 전처리
            sign_tensor = preprocess_image_torch(sign_roi, sign_size).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sign_outputs = sign_model(sign_tensor)
                sign_probabilities = F.softmax(sign_outputs, dim=1) # 소프트맥스 적용
            
            sign_confidence, sign_class_id = torch.max(sign_probabilities, 1)
            sign_confidence = sign_confidence.item()
            sign_class_id = sign_class_id.item()

            if sign_confidence > 0.7:
                sign_label = sign_class_names.get(sign_class_id, "알 수 없음")
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_frame, f"{sign_label} ({sign_confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # ------------------ 신호등 검출 및 분류 ------------------
        lights = light_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in lights:
            light_roi = frame[y:y+h, x:x+w]
            
            # PyTorch 모델을 위한 전처리
            light_tensor = preprocess_image_torch(light_roi, light_size).unsqueeze(0).to(device)
            
            with torch.no_grad():
                light_outputs = light_model(light_tensor)
                light_probabilities = F.softmax(light_outputs, dim=1) # 소프트맥스 적용

            light_confidence, light_class_id = torch.max(light_probabilities, 1)
            light_confidence = light_confidence.item()
            light_class_id = light_class_id.item()
            
            if light_confidence > 0.7:
                light_label = light_class_names.get(light_class_id, "알 수 없음")
                color = (0, 0, 255) if light_class_id == 0 else \
                        (0, 255, 255) if light_class_id == 1 else (0, 255, 0)
                
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_frame, f"{light_label} ({light_confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS 계산 및 표시
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            
        cv2.putText(result_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # 노란색 텍스트
        
        cv2.imshow('Traffic Sign/Light Detection (PyTorch)', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 인식 종료.")


# 메인 실행 코드 (PyTorch)
if __name__ == "__main__":
    # ----------------------------------------------------
    # 데이터 경로 설정 (GTSRB와 LISA 데이터셋 경로를 여기에 지정)
    gtsrb_data_dir = "path/to/GTSRB/Training" # GTSRB 학습 데이터셋 경로
    lisa_data_dir = "path/to/LISA_Traffic_Light_Dataset" # LISA 학습 데이터셋 경로

    # 경고: `haarcascade_traffic_sign.xml` 및 `haarcascade_traffic_light.xml` 파일이 필요합니다.
    # 이 예제는 이 캐스케이드 파일이 현재 작업 디렉토리에 있다고 가정합니다.
    # ----------------------------------------------------

    # 1. 교통 표지판 분류 모델 학습 또는 로드
    traffic_sign_model_path_pt = 'traffic_sign_model_pt.pth'
    sign_model_pt = SimpleCNN(num_classes=43).to(device)
    try:
        sign_model_pt.load_state_dict(torch.load(traffic_sign_model_path_pt, map_location=device))
        print("PyTorch 교통 표지판 모델을 불러왔습니다.")
    except:
        print("PyTorch 교통 표지판 모델을 학습합니다...")
        X_gtsrb, y_gtsrb = load_gtsrb_data_torch(gtsrb_data_dir)
        X_train_gtsrb, X_val_gtsrb, y_train_gtsrb, y_val_gtsrb = train_test_split(X_gtsrb, y_gtsrb, test_size=0.2, random_state=42)
        train_dataset_gtsrb = TrafficDataset(X_train_gtsrb, y_train_gtsrb)
        val_dataset_gtsrb = TrafficDataset(X_val_gtsrb, y_val_gtsrb)
        train_loader_gtsrb = DataLoader(train_dataset_gtsrb, batch_size=32, shuffle=True)
        val_loader_gtsrb = DataLoader(val_dataset_gtsrb, batch_size=32)
        
        optimizer_gtsrb = optim.Adam(sign_model_pt.parameters(), lr=0.001)
        criterion_gtsrb = nn.CrossEntropyLoss()
        
        sign_model_pt = train_classification_model_torch(sign_model_pt, train_loader_gtsrb, val_loader_gtsrb, criterion_gtsrb, optimizer_gtsrb, epochs=30)
        torch.save(sign_model_pt.state_dict(), traffic_sign_model_path_pt)
        print("PyTorch 교통 표지판 모델 학습 및 저장 완료.")

    # 2. 신호등 상태 분류 모델 학습 또는 로드
    traffic_light_model_path_pt = 'traffic_light_model_pt.pth'
    light_model_pt = SimpleCNN(num_classes=3).to(device)
    try:
        light_model_pt.load_state_dict(torch.load(traffic_light_model_path_pt, map_location=device))
        print("PyTorch 신호등 모델을 불러왔습니다.")
    except:
        print("PyTorch 신호등 모델을 학습합니다...")
        X_lisa, y_lisa = load_lisa_traffic_light_data_torch(lisa_data_dir)
        X_train_lisa, X_val_lisa, y_train_lisa, y_val_lisa = train_test_split(X_lisa, y_lisa, test_size=0.2, random_state=42)
        train_dataset_lisa = TrafficDataset(X_train_lisa, y_train_lisa)
        val_dataset_lisa = TrafficDataset(X_val_lisa, y_val_lisa)
        train_loader_lisa = DataLoader(train_dataset_lisa, batch_size=32, shuffle=True)
        val_loader_lisa = DataLoader(val_dataset_lisa, batch_size=32)
        
        optimizer_lisa = optim.Adam(light_model_pt.parameters(), lr=0.001)
        criterion_lisa = nn.CrossEntropyLoss()
        
        light_model_pt = train_classification_model_torch(light_model_pt, train_loader_lisa, val_loader_lisa, criterion_lisa, optimizer_lisa, epochs=30)
        torch.save(light_model_pt.state_dict(), traffic_light_model_path_pt)
        print("PyTorch 신호등 모델 학습 및 저장 완료.")
    
    # 3. 실시간 인식 실행
    print("\n--- 실시간 도로 표지판 및 신호등 인식을 시작합니다 (PyTorch) ---")
    detect_and_classify_traffic_signs_and_lights_torch(traffic_sign_model_path_pt, traffic_light_model_path_pt, 
                                                camera_index=0, sign_size=(32, 32), light_size=(64, 64))

```

## 4. 라즈베리파이에서의 최적화 방법

- 라즈베리파이와 같은 임베디드 디바이스에서는 모델 경량화(양자화, 가지치기) 및 추론 속도 향상(이미지 크기 축소, 프레임 건너뛰기, 스레드 분리) 기법이 중요함

- 본 실습에서는 도로 표지판 및 신호등을 '검출'하는 부분에 OpenCV의 `CascadeClassifier`를 사용
    - 이는 비교적 가볍게 객체 후보 영역을 찾는 데 유용하지만,
    - 최신 딥러닝 기반 객체 탐지 모델(예: YOLO-Nano, MobileNet-SSD)만큼 정확하지 않음

- 라즈베리파이에서도 TensorRT나 OpenVINO와 같은 최적화 툴을 사용하면 경량화된 딥러닝 객체 탐지 모델을 효율적으로 실행할 수 있음

- **최적화된 추론 예시 (TensorFlow Lite 이용)**
    - 텐서플로우 모델을 TFLite로 변환하여 라즈베리파이에서 효율적으로 실행할 수 있음

```python
# TFLite 모델 로드 및 추론 함수 예시
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def run_tflite_inference(interpreter, input_details, output_details, image_data):
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# 실제 사용 예시
# sign_interpreter, sign_input_details, sign_output_details = load_tflite_model('traffic_sign_model_tf_quantized.tflite')
# light_interpreter, light_input_details, light_output_details = load_tflite_model('traffic_light_model_tf_quantized.tflite')

# 이미지 전처리 후:
# sign_input_data = (sign_roi_normalized * 255).astype(np.uint8) # 양자화 모델은 uint8 입력 필요
# sign_input_data = np.expand_dims(sign_input_data, axis=0)
# sign_predictions = run_tflite_inference(sign_interpreter, sign_input_details, sign_output_details, sign_input_data)
# sign_class_id = np.argmax(sign_predictions[0])
# ...
```

## 5. 실제 자율주행 키트에 적용하기

- **제어 로직**
    - 인식된 표지판이나 신호등 정보에 따라 차량의 주행 속도, 방향, 정지 여부 등을 제어하는 로직 추가
        - 예: '정지' 표지판 또는 '빨간불' 인식 시 `motor_controller.stop()`.
        - 예: '속도 제한 (50km/h)' 표지판 인식 시 `motor_controller.set_max_speed(50)`.

- **다중 센서 융합**
    - 차선 인식 결과와 표지판/신호등 인식 결과를 종합하여 더 안전하고 정확한 주행 판단을 내리도록 시스템 구축
        - 예: 차선 인식이 불안정할 때 표지판 정보가 주행 방향 결정에 중요한 보조 역할을 할 수 있음
