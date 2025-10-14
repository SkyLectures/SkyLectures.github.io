---
layout: page
title:  "AI 모델 학습 및 실행 확인"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-10-02-02_01-AiModelTrainingAndVerification
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - AI 모델은 학습이 성공적으로 완료되고 실제 환경에서 의도대로 잘 작동하는지 주기적으로 확인해야 함
> - 모델의 성능을 최적화하고, 잠재적인 문제를 조기에 발견하여 해결하는 데 필수적

## 1. AI 모델 학습 확인 방법

- 모델이 '잘 배우고 있는지' 그리고 '학습이 성공적으로 끝났는지'를 확인하는 과정

### 1.1 학습 과정 모니터링

- 학습이 진행되는 동안 실시간으로 모델의 상태를 파악하는 것이 중요함

- **손실(Loss) 값 변화 확인**
    - 손실 값: 모델의 예측과 실제 값 사이의 오차를 나타냄
    - 학습이 진행되면서 이 값이 **점진적으로 감소하는지** 확인해야 함
    - 만약 손실 값이 감소하지 않거나 오히려 증가한다면, 학습률(learning rate) 설정이나 모델 구조 등 문제가 있을 수 있음
    - 훈련(training) 손실과 검증(validation) 손실을 함께 보며 **과소적합(Underfitting)** 또는 **과대적합(Overfitting)** 징후 파악
        - **과소적합**: 훈련 손실과 검증 손실 모두 높게 유지되는 경우 (모델이 충분히 학습되지 않음)
        - **과대적합**: 훈련 손실은 낮아지지만 검증 손실은 증가하는 경우 (모델이 훈련 데이터에 너무 맞춰져 새로운 데이터에 취약함)
            - 과대적합 또는 과적합으로 부름

- **평가 지표(Metrics) 변화 확인**
    - 모델의 목적에 맞는 평가 지표가 **점진적으로 개선되는지** 확인
        - 평가지표
            - 분류 모델: **정확도(Accuracy)**, **정밀도(Precision)**, **재현율(Recall)**, **F1-Score** 등
            - 회귀 모델: **RMSE**, **MAE** 등            
    - 손실 값과 마찬가지로 훈련 세트와 검증 세트에서의 지표를 함께 비교하며 일반화 성능을 평가

- **학습 곡선 시각화**
    - TensorBoard, Matplotlib 등의 도구 사용
    - 손실 값과 평가 지표의 변화를 그래프로 시각화하면 학습 과정을 한눈에 파악하기 용이함
    - 시각화를 통해 최적의 학습 중단 시점(Early Stopping)을 결정하거나, 모델 개선 방향을 수립할 수 있음

### 1.2 학습 결과 분석

- 학습이 완료된 후 모델의 최종 상태를 확인하는 과정

- **테스트 데이터셋으로 최종 평가**
    - 학습 및 검증 과정에서 사용되지 않은, 모델이 전혀 보지 못했던 **테스트 데이터셋**을 사용하여 최종 성능을 평가
    - 모델의 실제 일반화 능력을 가장 객관적으로 보여줌
    - 여기서 얻은 정확도, 정밀도 등의 지표가 목표치를 충족하는지 확인

- **오류 분석 (Error Analysis)**
    - 테스트 데이터셋에서 모델이 잘못 예측한 샘플들을 분석
    - 어떤 유형의 데이터에서 오류가 발생하는지 파악
    - 모델의 약점을 보완하거나 추가 학습 데이터 확보에 활용 가능

- **모델 저장 및 버전 관리**
    - 성공적으로 학습된 모델은 추후 다시 사용할 수 있도록 가중치와 구조를 '.h5', '.pth', '.tf' 등의 형식으로 저장
        - 최근 버전의 Tensorflow는 '.keras'를 사용할 것을 권장함
        - '.pth'는 주로 PyTorch에서 사용함
    - MLflow, Weights & Biases와 같은 실험 관리 도구 사용
    - 모델 버전, 학습 파라미터, 성능 지표 등을 체계적으로 기록하고 관리하면 지속적인 작업 및 개선에 도움이 됨

## 2. AI 모델 실행(추론) 확인 방법

- 학습된 모델이 실제 서비스 환경에서 '의도대로 잘 작동하는지'를 확인하는 과정

### 2.1 실시간 예측 확인

- 실제 운영 환경에서 새로운 입력에 대한 모델의 동작 점검

- **샘플 데이터 추론 결과 검토**
    - 새로운 실제 데이터를 모델에 입력하고, 모델이 생성하는 출력(예측 값)을 육안 또는 스크립트로 주기적으로 확인
    - 예측 결과가 논리적으로 맞는지, 기대하는 형식으로 나오는지 등을 검토

- **임계값(Threshold) 설정 및 테스트**
    - 분류 모델의 경우 예측 확률에 대한 임계값 설정이 필요함
    - 다양한 임계값에 대해 모델의 행동이 어떻게 변하는지 확인하고 최적의 임계값을 결정

### 2.2 성능(Performance) 모니터링

- 모델이 효율적이고 안정적으로 운영되는지 확인

- **추론 시간(Latency) 측정**
    - 하나의 입력 데이터를 처리하여 예측을 생성하는 데 걸리는 시간을 측정
    - 사용자 경험과 직접적으로 연결되므로 서비스 SLA(Service Level Agreement)를 충족하는지 확인해야 함

- **자원 사용량(Resource Utilization) 모니터링**
    - 모델 실행 시 CPU, GPU, 메모리 등의 자원 사용량 모니터링
    - 과도한 자원 사용은 비용 증가나 시스템 불안정으로 이어질 수 있음
    - 로컬 환경에서 모델을 구동하는 경우, 제한된 자원에서 얼마나 효율적으로 작동하는지 꾸준히 확인하고 최적화해야 함

- **처리량(Throughput) 측정**
    - 주어진 시간 동안 모델이 처리할 수 있는 입력 데이터의 양 측정
    - 서비스의 부하를 감당할 수 있는지 평가하는 중요 지표

### 2.3 모델 / 데이터 드리프트 모니터링

- 시간이 지남에 따라 모델의 성능이 저하될 수 있는 현상을 감지

- **모델 드리프트(Model Drift)**
    - 시간이 지남에 따라 모델의 예측 성능이 점진적으로 저하되는 현상
    - 데이터 드리프트의 결과일 수도 있고, 외부 환경 변화 때문일 수도 있음
    - 지속적인 성능 지표 모니터링과 주기적인 재검증을 통해 모델 드리프트를 감지
    - 필요시 모델을 재학습하거나 업데이트해야 함

- **데이터 드리프트(Data Drift)**
    - 실제 서비스에 입력되는 데이터의 분포가 모델 학습 시 사용했던 데이터의 분포와 달라지는 현상
        - 예: 신조어 등장, 새로운 트렌드의 콘텐츠 등
    - 모델 성능 저하의 주요 원인이 됨
    - 주기적으로 입력 데이터의 특성을 분석하여 변화를 감지해야 함

### 2.4 오류 및 로그 관리

- 문제가 발생했을 때 신속하게 원인을 파악하기 위함

- **로깅 시스템 구축**
    - 모델의 모든 예측 결과, 오류 발생 시점, 관련 입력 데이터 등을 상세하게 로그로 기록하는 시스템을 구축
    - ELK Stack(Elasticsearch, Logstash, Kibana) 같은 도구를 활용하여 로그를 효과적으로 수집, 저장, 분석할 수 있음

- **알림 시스템 (Alert System)**
    - 모델의 성능 지표가 특정 임계값 이하로 떨어지거나,
    - 자원 사용량이 비정상적으로 증가하는 경우,
    - 관리자에게 자동으로 알림을 보내는 시스템 구축

## 3. AI 모델 학습 및 실행 실습

### 3.1 MNIST 데이터셋 학습

#### 3.1.1 Tensorflow 기반 DNN 모델

- 모델 학습 및 평가

```python
import tensorflow as tf

tf.random.set_seed(6)

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# 모델 정의
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dense(10)
])

# 모델 컴파일
model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# 모델 훈련
model.fit(
  x_train, y_train,
  epochs=5,
  batch_size=128,
  verbose=1
)

# 모델 평가
_, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

# 모델 저장
model.save('model_mnist_dnn_tensorflow.keras')
print('Training finished')


# 학습된 모델을 이용한 실제 예측 확인(시각화)

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

num_samples = 5
random_indices = np.random.randint(0, len(x_test), num_samples)

model = tf.keras.models.load_model('model_mnist_dnn_tensorflow.keras')

plt.figure(figsize=(25, 3))

for i, idx in enumerate(random_indices):
    # 이미지와 실제 레이블
    image = x_test[idx]
    true_label = y_test[idx]

    # 예측
    pred = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label = np.argmax(pred)
    confidence = np.max(pred) * 100

    # 이미지 시각화
    plt.subplot(1, num_samples, i+1)
    plt.imshow(image.reshape(28, 28), cmap='gray')

    # 예측이 맞았는지 색상으로 표시 (초록: 맞음, 빨강: 틀림)
    title_color = 'green' if predicted_label == true_label else 'red'
    plt.title(f'Pred: {predicted_label}\nReal: {true_label}', color=title_color)
    plt.axis('off')
```

#### 3.1.2 PyTorch 기반 DNN 모델

- 모델 학습 및 평가

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# 시드 설정
torch.manual_seed(6)

# 하이퍼파라미터 설정
batch_size = 128
learning_rate = 0.001
num_epochs = 5

# 데이터셋 전처리 및 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균과 표준편차
])

# 학습 및 테스트 데이터 로드
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# DNN 모델 정의
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 1000)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 디바이스 설정 (GPU 사용 가능 시)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# 모델 초기화
model = DNN().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 순전파, 역전파, 최적화
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# 모델 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'테스트 정확도: {accuracy:.2f}%')

# 모델 저장
torch.save(model.state_dict(), 'model_mnist_dnn_pytorch.pth')
print('모델이 저장되었습니다!')


# 학습된 모델을 이용한 실제 예측 확인(시각화)

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# 테스트셋에서 무작위로 샘플 선택
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 샘플 이미지에 대한 예측
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 첫 15개 이미지와 예측 결과 시각화
plt.figure(figsize=(15, 5))

for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')

    # 예측이 맞았는지 색상으로 표시
    title_color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f'Pred: {predicted[i]}\nReal: {labels[i]}', color=title_color)
    plt.axis('off')
```

#### 3.1.3 Tensorflow 기반 CNN 모델

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# 시드 설정
tf.random.set_seed(6)

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 데이터 전처리
# CNN 모델을 위해 이미지 형식으로 reshape (샘플 수, 높이, 너비, 채널)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 레이블을 원-핫 인코딩 (선택사항)
# 원-핫 인코딩을 수행했다면 모델 컴파일 시 손실 함수로 CategoricalCrossentropy를 사용할 것
# 원-핫 인코딩을 수행하지 않았다면 모델 컴파일 시 손실 함수로 SparseCategoricalCrossentropy를 사용할 것

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 구축
model = models.Sequential([
    # 첫 번째 합성곱 레이어
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 두 번째 합성곱 레이어
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 세 번째 합성곱 레이어
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 완전 연결 레이어를 위한 Flatten
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)  # 10개 클래스에 대한 출력
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'테스트 정확도: {test_acc:.4f}')

plt.figure(figsize=(12, 4))

# 정확도(Accuracy) 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 손실(Loss) 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 모델 저장
model.save('model_mnist_cnn_tensorflow.keras')
print('모델이 저장되었습니다!')


# 학습된 모델을 이용한 실제 예측 확인(시각화)

num_samples = 5

predictions = model.predict(x_test[:num_samples])
predictions = np.argmax(predictions, axis=1)

plt.figure(figsize=(25, 3))

for i in range(num_samples):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {predictions[i]}\nReal: {y_test[i]}')
    plt.axis('off')
```

#### 3.1.4 PyTorch 기반 CNN 모델

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# 하이퍼파라미터 설정
batch_size = 128
learning_rate = 0.001
num_epochs = 5

# 데이터셋 전처리 및 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터셋의 평균과 표준편차
])

# 학습 데이터 로드
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 테스트 데이터 로드
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 데이터 로더 설정
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 합성곱 레이어
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 출력 크기: 14x14
        )

        # 두 번째 합성곱 레이어
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 출력 크기: 7x7
        )

        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 합성곱 레이어 통과
        x = self.conv1(x)
        x = self.conv2(x)

        # 텐서 평탄화
        x = x.view(x.size(0), -1)

        # 완전 연결 레이어 통과
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# 모델 초기화
model = CNN().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()  # 학습 모드 설정
    for i, (images, labels) in enumerate(train_loader):
        # GPU로 데이터 이동
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# 모델 평가
model.eval()  # 평가 모드 설정
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'테스트 정확도: {100 * correct / total:.2f}%')

# 모델 저장
torch.save(model.state_dict(), 'model_mnist_cnn_pytorch.pth')
print('모델이 저장되었습니다!')


# 학습된 모델을 이용한 실제 예측 확인(시각화)

# 테스트셋에서 무작위로 샘플 선택
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 샘플 이미지에 대한 예측
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 첫 15개 이미지와 예측 결과 시각화
plt.figure(figsize=(15, 5))

for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')

    # 예측이 맞았는지 색상으로 표시
    title_color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f'Pred: {predicted[i]}\nReal: {labels[i]}', color=title_color)
    plt.axis('off')
```

### 3.2 HouseSales 데이터셋 학습

#### 3.2.1 Tensorflow 기반 DNN 모델

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_house_data(file_path='https://raw.githubusercontent.com/SkyLectures/LectureMaterials/refs/heads/main/datasets/S03-10-02-02_01-kc_house_data.csv'):
    # 데이터 읽기
    df = pd.read_csv(file_path)

    # 필요없는 컬럼 제거
    df = df.drop(['id', 'date'], axis=1)

    # 결측치 처리
    df = df.dropna()

    # 이상치 확인 및 제거 (선택적)
    # 예: 가격이 너무 높거나 낮은 경우 제거
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[(df['price'] > q_low) & (df['price'] < q_high)]

    # 특성(X)과 타겟(y) 분리
    X = df.drop('price', axis=1)
    y = df['price']

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

    # 특성 스케일링 (표준화)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # 타겟 변수도 스케일링 (회귀 문제에서는 타겟도 스케일링하는 것이 중요)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # NaN 값이 있는지 확인
    print("X_train NaN 개수:", np.isnan(X_train).sum())
    print("y_train NaN 개수:", np.isnan(y_train).sum())

    return X_train, X_test, y_train, y_test, scaler_y

def training(save_path='model_house_sales_dnn_tensorflow.keras'):
    # 데이터 로드
    X_train, X_test, y_train, y_test, scaler_y = load_house_data()

    # 입력 특성 수 확인
    input_dim = X_train.shape[1]

    # 모델 정의 (더 안정적인 구조)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,),
                             kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.Dense(1)  # 회귀 문제이므로 출력층은 하나의 노드
    ])

    # 모델 컴파일 (옵티마이저 변경 및 학습률 감소)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  # 그래디언트 클리핑 추가
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )

    # 콜백 추가 (학습이 불안정할 때 조기 종료)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        epochs=50,  # 더 많은 에폭 설정
        batch_size=32,  # 배치 사이즈 감소
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    # 모델 평가
    loss, mae = model.evaluate(X_test, y_test)
    print(f'테스트 손실(스케일링된 데이터): {loss:.2f}')
    print(f'테스트 MAE(스케일링된 데이터): {mae:.2f}')

    # 원래 스케일로 예측 결과 변환하여 평가
    y_pred = model.predict(X_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 원래 스케일에서의 MAE 계산
    mae_original = np.mean(np.abs(y_pred_original - y_test_original))
    print(f'테스트 MAE(원래 스케일): ${mae_original:.2f}')

    # 모델 저장
    model.save(save_path)
    print('Training finished')

    return model, scaler_y

# 직접 실행
model, scaler_y = training()

# 예측 예시
X_train, X_test, y_train, y_test, _ = load_house_data()

# 테스트 데이터 중 일부 샘플에 대해 예측
sample_count = 5
sample_indices = np.random.randint(0, len(X_test), sample_count)

sample_X = X_test[sample_indices]
sample_y = y_test[sample_indices]

predictions = model.predict(sample_X)

# 원래 스케일로 변환
sample_y_original = scaler_y.inverse_transform(sample_y.reshape(-1, 1))
predictions_original = scaler_y.inverse_transform(predictions)

# 결과 출력
print("\n예측 결과 비교:")
print("실제 가격\t\t예측 가격\t\t차이")
print("-" * 60)
for i in range(sample_count):
    actual = sample_y_original[i][0]
    predicted = predictions_original[i][0]
    diff = abs(actual - predicted)
    print(f"${actual:.2f}\t\t${predicted:.2f}\t\t${diff:.2f}")
```

#### 3.2.2 PyTorch 기반 DNN 모델

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 시드 설정
torch.manual_seed(6)

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
num_epochs = 5

# 데이터셋 클래스 정의
class HouseSalesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 데이터 로드 및 전처리 함수
def load_house_data(file_path='https://raw.githubusercontent.com/SkyLectures/LectureMaterials/refs/heads/main/datasets/S03-10-02-02_01-kc_house_data.csv'):
    # 데이터 읽기
    df = pd.read_csv(file_path)

    # 필요없는 컬럼 제거
    df = df.drop(['id', 'date'], axis=1)

    # 결측치 처리
    df = df.dropna()

    # 이상치 제거 (가격 기준 상하위 1% 제거)
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[(df['price'] > q_low) & (df['price'] < q_high)]

    # 특성(X)과 타겟(y) 분리
    X = df.drop('price', axis=1)
    y = df['price']

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

    # 특성 스케일링 (표준화)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # 타겟 변수도 스케일링 (회귀 문제에서는 타겟도 스케일링하는 것이 중요)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, scaler_y

# 신경망 모델 정의
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.relu(self.batch_norm2(self.layer2(x)))
        x = self.relu(self.batch_norm3(self.layer3(x)))
        x = self.layer4(x)
        return x

def training():
    # 데이터 로드
    X_train, X_test, y_train, y_test, scaler_y = load_house_data()

    # 디바이스 설정 (GPU 사용 가능 시)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 중인 디바이스: {device}')

    # 데이터셋 및 데이터로더 생성
    train_dataset = HouseSalesDataset(X_train, y_train)
    test_dataset = HouseSalesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 모델 초기화
    input_dim = X_train.shape[1]
    model = HousePriceModel(input_dim).to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 기록 저장용
    train_losses = []

    # 모델 학습
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

    # 모델 평가
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        mae = 0.0

        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)

            # MSE 계산
            test_loss += criterion(outputs, targets).item()

            # MAE 계산
            mae += torch.mean(torch.abs(outputs - targets)).item()

        avg_test_loss = test_loss / len(test_loader)
        avg_mae = mae / len(test_loader)

        print(f'테스트 손실(스케일링된 데이터): {avg_test_loss:.4f}')
        print(f'테스트 MAE(스케일링된 데이터): {avg_mae:.4f}')

    # 원래 스케일로 예측 결과 변환하여 평가
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)

            outputs = model(features)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    # 예측 결과와 실제 값을 numpy 배열로 변환
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    # 원래 스케일로 변환
    all_targets_original = scaler_y.inverse_transform(all_targets)
    all_predictions_original = scaler_y.inverse_transform(all_predictions)

    # 원래 스케일에서의 MAE 계산
    mae_original = np.mean(np.abs(all_predictions_original - all_targets_original))
    print(f'테스트 MAE(원래 스케일): ${mae_original:.2f}')

    # 학습 곡선 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.title('Learning Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets_original, all_predictions_original, alpha=0.5)
    plt.plot([all_targets_original.min(), all_targets_original.max()],
             [all_targets_original.min(), all_targets_original.max()], 'r--')
    plt.xlabel('Real Price')
    plt.ylabel('Predicted Price')
    plt.title('Real Price vs Predicted Price')
    plt.grid(True)
    plt.show()

    # 모델 저장
    torch.save(model.state_dict(), 'model_house_sales_dnn_pytorch.pth')
    print('모델이 저장되었습니다!')

    return model, scaler_y

model, scaler_y = training()

# 예측 예시 (몇 가지 샘플에 대한 예측)
def predict_sample(model, scaler_y, num_samples=5):
  # 데이터 로드
  X_train, X_test, y_train, y_test, _ = load_house_data()

  # 샘플 선택
  indices = np.random.randint(0, len(X_test), num_samples)
  sample_X = X_test[indices]
  sample_y = y_test[indices]

  # 텐서로 변환
  sample_X_tensor = torch.tensor(sample_X, dtype=torch.float32).to(device)

  # 예측
  model.eval()
  with torch.no_grad():
      predictions = model(sample_X_tensor)

  # 원래 스케일로 변환
  sample_y_original = scaler_y.inverse_transform(sample_y.reshape(-1, 1))
  predictions_original = scaler_y.inverse_transform(predictions.cpu().numpy())

  # 결과 출력
  print("\n예측 결과 비교:")
  print("실제 가격\t\t예측 가격\t\t차이")
  print("-" * 60)
  for i in range(num_samples):
    actual = sample_y_original[i][0]
    predicted = predictions_original[i][0]
    diff = abs(actual - predicted)
    print(f"${actual:.2f}\t\t${predicted:.2f}\t\t${diff:.2f}")

# 샘플 예측 실행
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predict_sample(model, scaler_y)
```
