---
layout: page
title:  "도로 표지판 및 신호등 인식"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-06_01-RoadSignTrafficLightRecognition
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/project/notebooks/S10-01-04-06_01-RoadSignTrafficLightRecognition.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

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


## 2. YOLOv8을 이용한 교통 표지판 인식

### 2.1 필수 라이브러리 설치

- ultralytics 설치
    - ultralytics: YOLO 패키지 지원 패키지

```bash
#// file: "Terminal"
pip install ultralytics
```

- 기본 패키지
    - pathlib
        - 파일이나 디렉터리(폴더) 경로를 다루는 데 사용하는 모듈
        - os.path 모듈과 달리 경로를 객체 형태로 조작할 수 있게 해주는 것이 가장 큰 특징
    - glob
        - 파이썬 코드 안에서 파일들을 검색하고 목록으로 가져올 수 있게 해주는 모듈
        - 특정 패턴을 가진 파일 경로를 찾을 때 흔히 사용되는 편리한 도구

```python
import os
import pathlib
import glob
import random
```

- 데이터 처리용 패키지

```python
import numpy as np
import pandas as pd
```

- 시각화 처리용 패키지
    - matplotlib 패키지 가져오기
    - seaborn 패키지 가져오기 및 외형 설정
        - seaborn: 시각화 디자인 업그레이드를 위한 패키지

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'axes.facecolor': '#eae8fa'}, style='darkgrid')
```

- 개발환경 보완용 패키지
    - IPython.display
        - Video
            - Jupyter Notebook 환경에서 코드 셀 안에 동영상 파일을 직접 임베드하여 재생할 수 있게 해줌
    - tqdm.notebook
        - tqdm
            - 반복문(loop)의 진행 상황을 진행률 바(progress bar) 형태로 시각적으로 보여줌
        - trange
            - tqdm(range(N))의 단축 표현
            - 주로 특정 횟수만큼 반복되는 반복문의 진행 상황을 표시할 때 range() 함수와 함께 편리하게 사용

```python
from IPython.display import Video
from tqdm.notebook import trange, tqdm
```

- 영상 처리용 패키지

```python
import cv2
from PIL import Image
from ultralytics import YOLO
```

- 기타 처리
    - 불필요한 경고 메시지 제거

```python
import warnings
warnings.filterwarnings('ignore')
```

### 2.2 데이터셋

- 감지 전 원본 이미지 보여주기
    - 훈련 데이터셋으로부터 예시 이미지 확인

```python
Image_dir = './car/train/images'

num_samples = 9
image_files = os.listdir(Image_dir)

# Randomly select num_samples images
rand_images = random.sample(image_files, num_samples)

fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i in range(num_samples):
    image = rand_images[i]
    ax = axes[i // 3, i % 3]
    ax.imshow(plt.imread(os.path.join(Image_dir, image)))
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

- 훈련 단계에서 사용할 이미지 모양 가져오기

```python
# Get the size of the image
image = cv2.imread("./car/train/images/00000_00000_00012_png.rf.23f94508dba03ef2f8bd187da2ec9c26.jpg")
h, w, c = image.shape
print(f"The image has dimensions {w}x{h} and {c} channels.")
```

### 2.3 사전학습된 YOLOv8 모델

```python
# Use a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Use the model to detect object
image = "./car/train/images/FisheyeCamera_1_00228_png.rf.e7c43ee9b922f7b2327b8a00ccf46a4c.jpg"
result_predict = model.predict(source = image, imgsz=(640))

# show results
plot = result_predict[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot))
```

### 2.4 YOLOv8 기반 교통 표지판 감지 모델

- 기본 설정

```bash
pip install --upgrade ultralytics ray
```

- 추가 학습
    - Google T4 GPU 사용 시 약 30분 소요됨

```python
# Build from YAML and transfer weights
Final_model = YOLO('yolov8n.pt')

# Training The Final Model
Result_Final_model = Final_model.train(data="./car/data.yaml",epochs = 30, batch = -1, optimizer = 'auto')
```

- 검증 단계

```python
import os
import cv2
import matplotlib.pyplot as plt

def display_images(post_training_files_path, image_files):

    for image_file in image_files:
        image_path = os.path.join(post_training_files_path, image_file)
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10), dpi=120)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# List of image files to display
image_files = [
    'confusion_matrix_normalized.png',
    'BoxF1_curve.png',
    'BoxP_curve.png',
    'BoxR_curve.png',
    'BoxPR_curve.png',
    'results.png'
]

# Path to the directory containing the images
post_training_files_path = './runs/detect/train'

# Display the images
display_images(post_training_files_path, image_files)

Result_Final_model = pd.read_csv('./runs/detect/train/results.csv')
Result_Final_model.tail(10)

# Read the results.csv file as a pandas dataframe
Result_Final_model.columns = Result_Final_model.columns.str.strip()

# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=Result_Final_model, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=Result_Final_model, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=Result_Final_model, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=Result_Final_model, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=Result_Final_model, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=Result_Final_model, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=Result_Final_model, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=Result_Final_model, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=Result_Final_model, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=Result_Final_model, ax=axs[4,1])

# Set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

plt.suptitle('Training Metrics and Loss', fontsize=24)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()
```

- 테스트셋 기반의 검증 모델

```python
# Loading the best performing model
Valid_model = YOLO('./runs/detect/train/weights/best.pt')

# Evaluating the model on the validset
metrics = Valid_model.val(split = 'val')

# final results
print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])
print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])
```

- 테스트 이미지로 예측하기

```python
# Normalization function
def normalize_image(image):
    return image / 255.0

# Image resizing function
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Path to validation images
# dataset_path = '/kaggle/input/cardetection/car'  # Place your dataset path here
dataset_path = './car'  # Place your dataset path here
valid_images_path = os.path.join(dataset_path, 'test', 'images')

# List of all jpg images in the directory
image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]

# Check if there are images in the directory
if len(image_files) > 0:
    # Select 9 images at equal intervals
    num_images = len(image_files)
    step_size = max(1, num_images // 9)  # Ensure the interval is at least 1
    selected_images = [image_files[i] for i in range(0, num_images, step_size)]

    # Prepare subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('Validation Set Inferences', fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])

            # Load image
            image = cv2.imread(image_path)

            # Check if the image is loaded correctly
            if image is not None:
                # Resize image
                resized_image = resize_image(image, size=(640, 640))
                # Normalize image
                normalized_image = normalize_image(resized_image)

                # Convert the normalized image to uint8 data type
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

                # Predict with the model
                results = Valid_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)

                # Plot image with labels
                annotated_image = results[0].plot(line_width=1)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ax.imshow(annotated_image_rgb)
            else:
                print(f"Failed to load image {image_path}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

### 2.5 사전학습된 YOLOv8로 동영상 교통 표지판 감지

- 영상 인식 후 출력

```bash
!ffmpeg -y -loglevel panic -i ./video.mp4 output.mp4
```

```python
# Display the video
Video("./output.mp4", width=960, embed=True)

# Use the model to detect signs
Valid_model.predict(source="./video.mp4", show=True,save = True, stream=True)
```

```bash
!ffmpeg -y -loglevel panic -i ./video.avi result_out.mp4
```

```python
# Display the video
Video("./result_out.mp4", width=960, embed=True)
```

- 모델 저장

```python
# Export the model
Valid_model.export(format='onnx')
```


## 3. 라즈베리파이에서의 최적화 방법

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

## 4. 실제 자율주행 키트에 적용하기

- **제어 로직**
    - 인식된 표지판이나 신호등 정보에 따라 차량의 주행 속도, 방향, 정지 여부 등을 제어하는 로직 추가
        - 예: '정지' 표지판 또는 '빨간불' 인식 시 `motor_controller.stop()`.
        - 예: '속도 제한 (50km/h)' 표지판 인식 시 `motor_controller.set_max_speed(50)`.

- **다중 센서 융합**
    - 차선 인식 결과와 표지판/신호등 인식 결과를 종합하여 더 안전하고 정확한 주행 판단을 내리도록 시스템 구축
        - 예: 차선 인식이 불안정할 때 표지판 정보가 주행 방향 결정에 중요한 보조 역할을 할 수 있음
