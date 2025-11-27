---
layout: page
title:  "딥러닝 기반 객체 탐지"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-04-04_01-DeepLearningBasedObjectDetection
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/ai/notebooks/S10-01-04-04_01-DeepLearningBasedObjectDetection.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>


## 1. 객체 탐지 개요

- **인간의 시각을 모방하는 AI**
    - 우리가 주변 환경을 인식할 때, 단순히 '무엇인가 있다'고만 생각하지 않음
    - '앞에 **자동차**가 한 대 있고, 길가에는 **보행자** 두 명이 걸어가고 있다'와 같이 **무엇이(객체 종류)**, **어디에(위치)** 있는지를 동시에 파악
    - 객체 탐지(Object Detection)는 이처럼
        - 이미지나 영상 내에서 특정 객체의 **클래스(Class)**를 식별하고,
        - 해당 객체의 **위치(Localization)**를
        - 사각형 형태의 **바운딩 박스(Bounding Box)**로 나타내는
        - 컴퓨터 비전 기술

- **자율주행(Mobility AI)에서의 중요성**
    - 자율주행차에게 '보는' 것은 곧 '이해하는' 것
    - 객체 탐지는 자율주행 시스템의 인지(Perception) 단계에서 가장 핵심적인 역할 수행
        - **안전 운전**
            - 도로 위의 다른 차량, 보행자, 자전거, 오토바이 등 움직이는 모든 객체를 정확히 탐지하여
            - 충돌을 피하고 안전한 주행 경로 확보

        - **교통 상황 판단**
            - 신호등, 도로 표지판 등 고정된 객체를 인식하여 **→** 주행 규칙 준수
            - 주변 차량의 밀집도를 파악하여 **→** 교통 흐름 이해

        - **환경 이해**
            - 낙석, 구덩이, 건물, 교차로 등 다양한 환경 요소를 탐지하여
            - 차량의 위치를 정확하게 파악하고
            - 주행 전략을 수립

        - **판단 및 제어의 입력**
            - 탐지된 객체의 종류, 위치, 크기, 그리고 시계열 데이터를 통한 움직임 예측 정보는
            - 판단(Decision-making) 및 제어(Control) 모듈의 주요 입력으로 활용

## **2. 객체 탐지의 주요 개념 및 평가 지표**

- **바운딩 박스 (Bounding Box)**
    - 탐지된 객체의 위치를 나타내는 직사각형
    - 주로 좌상단 좌표 $$(x_1, y_1)$$와 우하단 좌표 $$(x_2, y_2)$$ 또는 중심 좌표 $$(x, y)$$, 너비 $$(w)$$, 높이 $$(h)$$로 표현
    - 모델은 객체의 클래스를 예측함과 동시에 이 바운딩 박스의 좌표도 예측함

- **클래스 분류 (Classification) & 신뢰도 점수 (Confidence Score)**
    - 모델은 각 바운딩 박스 내부에 어떤 종류의 객체가 있는지를 예측(예: `car`, `person`, `traffic light`)하고,
    - 그 예측이 얼마나 정확하다고 확신하는지 **신뢰도 점수(0~1)**를 함께 출력
    - 보통 이 신뢰도 점수가 특정 임계값(예: 0.5) 이상인 탐지만 유효하다고 간주함

- **IoU (Intersection over Union)**
    - 예측된 바운딩 박스(Predicted Box)와 실제 객체의 바운딩 박스(Ground Truth Box)가 얼마나 잘 겹치는지 측정하는 지표
    - **계산식**<br>
        &nbsp;&nbsp;&nbsp;&nbsp; $$IoU = (두\ 박스의\ 교집합\ 영역) / (두\ 박스의\ 합집합\ 영역)$$
    - **의미**
        - IoU 값이 0.5 이상이면 일반적으로 '올바르게 탐지되었다(True Positive)'고 판단
        - IoU가 클수록 예측이 정확함

- **NMS (Non-Maximum Suppression)**
    - 객체 탐지 모델은 하나의 객체에 대해 여러 개의 바운딩 박스를 중복하여 예측하는 경우가 많음
    - NMS는 이러한 중복된 바운딩 박스들을 제거하고 가장 적절한 하나의 박스만을 남기는 후처리 과정

    - **작동 방식**:
        1. 모든 예측 박스 중에서 신뢰도 점수가 가장 높은 박스를 선택
        2. 선택된 박스와 IoU가 특정 임계값(예: 0.5) 이상인 다른 박스들을 제거
        3. 남은 박스들 중에서 다시 신뢰도 점수가 가장 높은 박스를 선택하고
        4. 위 과정을 반복

- **mAP (mean Average Precision)**
    - 객체 탐지 모델의 전반적인 성능을 평가하는 가장 중요한 지표
    - 여러 클래스에 대한 Average Precision(AP) 값을 구하고, 이들의 평균을 계산한 값

    - **AP**
        - 특정 IoU 임계값에서 Precision-Recall 곡선 아래의 면적을 나타냄
            - Precision: '탐지된 것 중 진짜' 비율, Recall: '실제 객체 중 탐지된' 비율
    - **mAP**
        - 모델이 여러 종류의 객체를 얼마나 정확하고 효율적으로 탐지하는지를 종합적으로 보여줌

## 3. 딥러닝 기반 객체 탐지 모델의 종류

### 3.1. 2단계 탐지기

- 객체 탐지 과정을 `객체 후보 영역 제안(Region Proposal)`과 `클래스 분류 및 바운딩 박스 미세 조정`의 두 단계로 나누어 수행
- (Two-Stage Detectors)

- **특징**
    - **높은 정확도**
        - 각 단계가 독립적으로 최적화될 수 있으므로 일반적으로 탐지 정확도가 매우 높음

    - **느린 속도**
        - 두 단계를 순차적으로 거치므로 상대적으로 처리 속도가 느림

- **주요 모델**
    - **R-CNN (Region-based Convolutional Neural Network)**
        - Selective Search로 후보 영역을 제안하고
        - CNN으로 특징 추출
        - SVM으로 분류

    - **Fast R-CNN**
        - R-CNN의 속도 개선 모델
        - 특징 추출을 한 번만 하고
        - RoI Pooling으로 각 후보 영역의 특징을 얻음

    - **Faster R-CNN**
        - Fast R-CNN의 속도 개선 모델
        - Selective Search 대신 RPN(Region Proposal Network)이라는 딥러닝 기반 네트워크로 후보 영역을 제안
        - 전체 과정이 딥러닝 파이프라인 안에서 수행됨

### 3.2. 1단계 탐지기

- 객체 후보 영역 제안과 분류 및 바운딩 박스 예측을 하나의 네트워크에서 동시에 수행함
- (One-Stage Detectors)

- **특징**
    - **빠른 속도**
        - 단일 네트워크에서 모든 연산이 이루어져 처리 속도가 매우 빠름
        - 자율주행과 같은 실시간 응용 분야에 적합

    - **상대적으로 낮은 정확도 (초기)**
        - 초기에는 2단계 탐지기보다 정확도가 떨어졌으나,
        - 지속적인 발전으로 성능 차이가 많이 줄어듦 **→** 현재는 사실상 거의 차이가 없다고 해도 좋은 수준

- **주요 모델**
    - **YOLO (You Only Look Once)**
        - 이미지를 그리드로 나누고,
        - 각 그리드 셀이 바운딩 박스와 클래스를 직접 예측
        - 탐지 속도가 매우 빠름
        - YOLOv1 ~ YOLOv8 등 계속 발전 중

    - **SSD (Single Shot Detector)**
        - 다양한 스케일의 특징 맵에서 바운딩 박스와 클래스를 예측하는 기법 사용
        - YOLO와 더불어 대표적인 1단계 탐지기

    - **RetinaNet**
        - Focal Loss를 도입하여 클래스 불균형 문제를 해결
            - Focal Loss: 주로 객체 탐지(Object Detection) 모델, 특히 밀집 객체 탐지(Dense Object Detection)에서 발생하는 클래스 불균형(Class Imbalance) 문제를 해결하기 위해 고안된 손실 함수
        - 1단계 탐지기의 성능을 크게 끌어올린 모델

> - **자율주행을 위한 모델 선택**
>   - 자율주행과 같은 실시간 응용 분야에서는 **속도(FPS, 초당 프레임 수)**가 매우 중요함
>   - 일반적으로 **YOLO나 SSD 계열의 1단계 탐지기**가 선호됨
>   - 최신 YOLO 모델들은 정확도와 속도 모두에서 훌륭한 성능을 보여줌
{: .expert-quote}

## 4. 모빌리티 AI에 특화된 고려사항 및 데이터셋

자율주행 환경의 특수성으로 인해 객체 탐지 모델을 개발할 때 몇 가지 추가적인 고려사항이 있습니다.

### 4.1. 주요 탐지 객체
- **차량 (Cars, Trucks, Buses)**: 크기, 속도, 차종 다양성
- **보행자 (Pedestrians)**: 다양한 자세, 군집, 부분 가려짐
- **이륜차 (Bicycles, Motorcycles)**: 작고 민첩함, 예측 어려운 움직임
- **교통 신호/표지판 (Traffic Lights, Traffic Signs)**: 작고 색상, 형태 중요
- **차선 (Lane Lines)**: 일반적으로 세그멘테이션으로 처리되지만, 일부 시스템에서는 객체로 간주하기도 함

### 4.2. 주요 도전 과제
- **다양한 조명 조건**: 낮/밤, 터널, 역광 등 조명 변화에 강건해야 함
- **기상 조건**: 비, 눈, 안개 등 악천후에도 안정적인 탐지
- **부분 가려짐 (Occlusion)**: 다른 차량이나 장애물에 의해 객체의 일부가 가려진 경우
- **작은 객체 탐지 (Small Object Detection)**: 멀리 있는 차량이나 작은 표지판을 정확히 탐지하는 것은 고난도
- **실시간 성능**: 높은 정확도를 유지하면서도 초당 최소 30프레임 이상의 처리 속도 필요

### 4.3. 대표적인 모빌리티 AI 데이터셋
- **BDD100K**
    - Berkeley DeepDrive 100K
    - 10만 장 이상의 도로 주행 이미지/영상
    - 10가지 클래스에 대한 객체 탐지, 세그멘테이션, 차선 마스크 등 다양한 어노테이션을 포함
    - 날씨, 시간, 지역 등이 다양하게 포함되어 자율주행 연구에 널리 사용됨

- **Waymo Open Dataset**
    - Waymo가 공개한 대규모 데이터셋
    - HD 카메라, 라이다 센서 데이터 포함

- **nuScenes**
    - 자율주행 차량용 센서(카메라, 라이다, 레이더) 데이터를 포함하는 데이터셋

- **COCO (Common Objects in Context)**
    - 일반적인 객체 탐지 벤치마크 데이터셋
    - `car`, `person`, `bicycle`, `traffic light` 등 자율주행과 관련된 클래스도 다수 포함하고 있어 전이 학습에 유용함

## 5. 실습 예제

- **BDD100K 샘플 이미지에 대한 객체 탐지 추론**
    - 실제 자율주행 모델을 처음부터 학습시키는 것은 매우 많은 시간과 컴퓨팅 자원을 요구함
    - 여기서는 **사전 학습된(Pre-trained) 객체 탐지 모델**을 사용하여 실습 진행
        - BDD100K 데이터셋의 샘플 이미지에 대한 추론(Inference)을 수행하고 결과를 시각화
    - 라즈베리파이와 같은 임베디드 환경에서도 효율적인 실시간 탐지를 위해 자주 사용됨

**준비물**
1.  **샘플 이미지**
    - BDD100K 데이터셋에서 '차량', '보행자' 등이 포함된 몇 가지 샘플 이미지를 준비하거나,
    - 임의의 도로 이미지를 사용
    - 예: `sample_image.jpg`, `sample_image2.jpg` 등

2.  **딥러닝 프레임워크**
    - TensorFlow 또는 PyTorch 설치

### **5.1. TensorFlow 버전**

- **사용 모델**
    - TensorFlow Hub 또는 Keras Applications에서 제공하는 사전 학습된 MobileNetV2-SSD (Single Shot Detector) 모델
    - 속도와 정확도 사이의 균형이 좋아 임베디드 장치에 적합함

```python
#//file: "mobile_net_v2_ssd.py"
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 # 이미지 처리 및 드로잉에 여전히 사용
import matplotlib.pyplot as plt # Colab 시각화를 위해 필요
import os

# --------------------------------------------------------------------------
# TensorFlow Hub 모델 로드
# --------------------------------------------------------------------------
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

print(f"TensorFlow Hub 모델을 로드 중입니다: {model_url}")
detector = hub.load(model_url)
print("TensorFlow Hub 모델 로드 완료.")

# 모델이 학습된 COCO 데이터셋의 클래스 이름 (실제 모델 출력을 기반으로)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# 클래스 ID와 이름을 매핑하는 딕셔너리 생성 (1-based index)
category_index = {i + 1: {'id': i + 1, 'name': name} for i, name in enumerate(COCO_CLASSES[1:])}


# --------------------------------------------------------------------------
# 이미지 로드 및 전처리 함수
# --------------------------------------------------------------------------
def load_and_preprocess_image_tf(image_path, target_size=(320, 320)):
    # tf.io와 tf.image를 사용하여 이미지 로드 및 전처리
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # JPEG 디코딩, TensorFlow는 기본적으로 RGB 순서

    # 원본 이미지를 numpy 배열로 복사 (시각화를 위해)
    # tf.Tensor는 기본적으로 RGB 채널 순서로 읽어오므로, original_image_rgb_tf도 RGB
    original_image_rgb_tf = np.copy(img.numpy())

    # 모델 입력 형식에 맞게 이미지 크기 조정
    # 모델은 uint8 입력을 기대하므로, float32 변환을 제거합니다.
    img = tf.image.resize(img, target_size) # (height, width)
    img = tf.cast(img, tf.uint8) # 모델이 uint8을 기대하므로 명시적으로 캐스팅

    # 모델은 [batch, height, width, channels] 형태의 텐서를 기대하므로 배치 차원 추가
    input_tensor = img[tf.newaxis, ...]

    return input_tensor, original_image_rgb_tf

# --------------------------------------------------------------------------
# 검출 결과 시각화 함수 (OpenCV 그리기 -> Matplotlib으로 최종 출력)
# --------------------------------------------------------------------------
def visualize_detections_tf(image_np_rgb, detections, category_index, threshold=0.5):
    # image_np_rgb는 원본 이미지 (RGB numpy array)
    image_with_detections = image_np_rgb.copy() # 원본 이미지 복사 (RGB 유지)
    h, w, _ = image_with_detections.shape

    # detections 딕셔너리에서 필요한 정보 추출
    num_detections = int(detections['num_detections'][0].numpy())
    
    detection_boxes = detections['detection_boxes'][0].numpy() # (ymin, xmin, ymax, xmax) (0.0~1.0)
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    for i in range(num_detections):
        score = detection_scores[i]
        if score > threshold:
            box = detection_boxes[i] # [ymin, xmin, ymax, xmax]
            class_id = detection_classes[i]

            # 바운딩 박스 좌표를 원본 이미지 크기에 맞게 스케일링
            # Tensorflow 모델은 일반적으로 (ymin, xmin, ymax, xmax) 순서 사용
            ymin, xmin, ymax, xmax = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)

            # 클래스 이름 매핑
            class_name = category_index[class_id]['name'] if class_id in category_index else f'Unknown({class_id})'
            label_text = f'{class_name}: {score:.2f}'
            
            # 바운딩 박스 및 텍스트 색상 (image_with_detections가 RGB이므로, 색상도 RGB로 지정)
            color_rgb = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))

            # 바운딩 박스 그리기 (OpenCV 함수는 BGR을 기대하지만, RGB 이미지에 RGB 색상을 넘겨도 내부적으로 처리함)
            # image_with_detections가 RGB이므로 color_rgb를 그대로 사용합니다.
            cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), color_rgb, 2)
            
            # 라벨 텍스트 배경 그리기
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_detections, (xmin, ymin - text_height - baseline - 5), (xmin + text_width, ymin), color_rgb, -1)
            
            # 라벨 텍스트 쓰기 (검은색 글씨는 (0,0,0) RGB이므로 문제 없음)
            cv2.putText(image_with_detections, label_text, (xmin, ymin - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
    return image_with_detections # RGB numpy 배열 반환

# --------------------------------------------------------------------------
# 메인 실행 블록
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Colab에서 이미지를 업로드하거나 GDrive를 마운트하여 사용할 수 있습니다.
    # 여기서는 예시로 로컬 경로를 사용하지만, Colab에서는 직접 업로드해야 할 수 있습니다.
    sample_image_path_tf = './images/road_image.jpg' # 실제 이미지 경로로 변경 필요
    
    # Colab에서 이미지를 업로드할 경우 '/content/' 경로에 저장될 수 있습니다.
    # if not os.path.exists(sample_image_path_tf):
    #    # Colab 환경에서 이미지 파일이 없다면, 사용자에게 업로드 지시
    #    from google.colab import files
    #    uploaded = files.upload()
    #    for fn in uploaded.keys():
    #        sample_image_path_tf = '/content/' + fn # 업로드된 파일 경로 사용
    #        print(f"'{fn}'이(가) '{sample_image_path_tf}' 경로에 업로드되었습니다.")
    #        break # 첫 번째 업로드된 파일만 사용

    # 더미 이미지 생성 (Colab에서 파일이 없을 경우)
    if not os.path.exists(sample_image_path_tf):
        print(f"이미지 파일 '{sample_image_path_tf}'이(가) 없습니다. 더미 이미지를 생성합니다.")
        # Colab 환경은 matplotlib이 주로 RGB를 기대하므로, 더미 생성 시 RGB 컬러를 사용
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8) # RGB 형태의 검은색 이미지
        cv2.putText(dummy_img, "Dummy Image for Object Detection (TensorFlow)", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dummy_img, "Please replace with a real image!", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(dummy_img, (300, 50), (350, 150), (255, 0, 0), 2) # 파란색 사각형 (BGR) -> 여기서는 (R,G,B)=(0,0,255)인 파란색으로 그려짐
        cv2.imwrite(sample_image_path_tf, cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR)) # 저장 시에는 BGR로 변환
        print(f"더미 이미지 '{sample_image_path_tf}'를 생성했습니다.")

    print("\n--- TensorFlow 기반 객체 탐지 시작 ---")
    
    try:
        # load_and_preprocess_image_tf 함수는 original_image_rgb_tf를 RGB로 반환합니다.
        input_tensor_tf, original_image_rgb_tf = load_and_preprocess_image_tf(sample_image_path_tf, target_size=(320, 320))
        
        # 실제 모델 추론 (TensorFlow Hub 모델)
        detections_tf = detector(input_tensor_tf)

        # visualize_detections_tf 함수에 RGB 이미지를 전달하고, RGB 이미지를 반환받습니다.
        result_image_tf_rgb = visualize_detections_tf(original_image_rgb_tf, detections_tf, category_index, threshold=0.5)

        # Matplotlib 시각화
        plt.figure(figsize=(10, 8))
        # 이미 RGB이므로 cv2.cvtColor(..., cv2.COLOR_BGR2RGB) 변환이 필요 없습니다.
        plt.imshow(result_image_tf_rgb)
        plt.title('TensorFlow Object Detection Results')
        plt.axis('off') # 축(axis) 표시 제거
        plt.show() # 이미지 출력

        print("--- TensorFlow 기반 객체 탐지 완료 ---")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"오류 발생: {e}")
```

### **5.2. PyTorch 버전**

- **사용모델**
    - PyTorch `torchvision.models.detection`에서 제공하는 사전 학습된 ResNet50-FPN 기반의 Faster R-CNN (Region-based Convolutional Network) 모델
    - 뛰어난 정확도를 가지며, ResNet50-FPN (Feature Pyramid Network) 백본을 사용하여 다양한 크기의 객체 탐지에 강함

```python
#//file: "object_detection_torch.py"
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
from PIL import Image

# --------------------------------------------------------------------------
# 1. GPU (CUDA) 사용을 위한 device 정의 및 확인
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA (GPU)를 사용합니다: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA (GPU)를 사용할 수 없어 CPU를 사용합니다.")
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 2. 모델 로드 및 device 이동, eval 모드 설정
# --------------------------------------------------------------------------
# 가중치 객체 생성
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# 모델 로드 (pretrained=True는 weights=weights와 동일하며, progress=True로 다운로드 진행 상황 표시)
model_pt = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)

# !!! 중요: 모델을 정의된 device로 이동해야 함
# 모델 정의 후 eval() 모드 설정 전에 이동하는 것이 일반적
model_pt.to(device)

# 모델을 추론 모드로 설정
model_pt.eval()
# --------------------------------------------------------------------------


# COCO 클래스 이름 로드
coco_class_names = weights.meta["categories"]
print(f"COCO 클래스 수: {len(coco_class_names)}")
print(f"일부 COCO 클래스: {coco_class_names[0]}, {coco_class_names[1]}, {coco_class_names[2]}, {coco_class_names[3]}, {coco_class_names[5]}, {coco_class_names[7]}")

# 이미지 전처리를 위한 변환 파이프라인 로드
preprocess = weights.transforms()

# --------------------------------------------------------------------------
# 3. 이미지 로드 및 전처리 함수
# --------------------------------------------------------------------------
def load_and_preprocess_image_pt(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    # OpenCV는 BGR, torchvision은 RGB를 기대하므로 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 원본 이미지는 시각화를 위해 저장 (np.copy()를 사용하여 원본 보존)
    original_image_np = np.copy(img_rgb)

    # numpy.ndarray를 PIL.Image 객체로 변환 (transforms가 PIL Image를 기대)
    img_pil = Image.fromarray(img_rgb)

    # PyTorch 모델 입력 형식으로 변환 (PIL Image -> Tensor)
    input_tensor_pt = preprocess(img_pil)
    return input_tensor_pt, original_image_np
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 4. 검출 결과 시각화 함수
# --------------------------------------------------------------------------
def visualize_detections_pt(image_np, detections_output, class_names, threshold=0.7):
    # image_np는 RGB 포맷의 numpy 배열이어야 합니다.
    image_with_detections = image_np.copy()

    # 모델 출력에서 결과 추출
    # detections_output은 list[dict] 형태이며, 우리는 batch_size=1이므로 첫 번째 dict를 사용함
    boxes = detections_output['boxes'].cpu().numpy()
    labels = detections_output['labels'].cpu().numpy()
    scores = detections_output['scores'].cpu().numpy()

    # OpenCV의 바운딩 박스 그리기
    for i in range(len(boxes)):
        score = scores[i]
        if score > threshold:
            box = boxes[i].astype(int) # 바운딩 박스 좌표 [xmin, ymin, xmax, ymax]
            class_id = labels[i]

            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            # 클래스 이름 확인 (안전한 인덱싱)
            class_name = class_names[class_id] if class_id < len(class_names) else f'Unknown({class_id})'
            label_text = f'{class_name}: {score:.2f}'

            # 랜덤 색상 (매번 동일한 색상을 사용하려면 np.random.seed()를 설정하거나 미리 정의된 색상을 사용)
            color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))

            # 바운딩 박스 그리기
            cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), color, 2)

            # 라벨 텍스트 배경 그리기
            # 텍스트 크기를 미리 계산하여 배경 사각형 크기를 맞춤
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_detections, (xmin, ymin - text_height - baseline - 5), (xmin + text_width, ymin), color, -1)
            
            # 라벨 텍스트 쓰기 (배경색과 반대되는 색상이 좋음)
            cv2.putText(image_with_detections, label_text, (xmin, ymin - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image_with_detections
# --------------------------------------------------------------------------


if __name__ == '__main__':
    # BDD100K 샘플 이미지 또는 기타 도로 이미지 경로
    sample_image_path_pt = './images/road_image.jpg' # 이미지 이름을 변경했습니다 (bdd10k_sample.jpg -> road_image.jpg)

    # 더미 이미지 생성 (파일이 없을 경우)
    if not os.path.exists(sample_image_path_pt):
        print(f"이미지 파일 '{sample_image_path_pt}'이(가) 없습니다. 더미 이미지를 생성합니다.")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Dummy Image for Object Detection", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dummy_img, "A traffic light here!", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(dummy_img, (300, 50), (350, 150), (255, 0, 0), 2) # 파란색 상자 (신호등 역할)
        cv2.imwrite(sample_image_path_pt, dummy_img)
        print(f"더미 이미지 '{sample_image_path_pt}'를 생성했습니다.")
    
    print("\n--- PyTorch 기반 객체 탐지 시작 ---")
    
    try:
        input_tensor_pt, original_image_np_pt = load_and_preprocess_image_pt(sample_image_path_pt)

        # 추론 실행 (입력 텐서를 배치 차원과 함께 디바이스로 이동)
        with torch.no_grad():
            # input_tensor_pt는 [C, H, W] 형태
            # 모델은 배치 차원을 [N, C, H, W] 형태로 기대하므로 unsqueeze(0)를 통해 추가해줌
            detections_pt = model_pt([input_tensor_pt.to(device)]) # 수정: input_tensor_pt를 [input_tensor_pt.to(device)]로 감쌈

        # 결과 시각화
        # detections_pt는 모델의 출력으로, list[dict] 형태 (배치 크기만큼).
        # 우리 배치는 1이므로, detections_pt[0]이 실제 이미지의 결과 딕셔너리임
        result_image_pt = visualize_detections_pt(original_image_np_pt, detections_pt[0], coco_class_names, threshold=0.7)

        plt.figure(figsize=(10, 8))
        # 이미 RGB로 변환되어 있으므로 Matplotlib에서 바로 표시 가능
        plt.imshow(result_image_pt)
        plt.title('PyTorch Object Detection Results')
        plt.axis('off')
        plt.show()
        print("--- PyTorch 기반 객체 탐지 완료 ---")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"오류 발생: {e}")
```


> - 객체 탐지는 단순히 이미지를 '이해'하는 것을 넘어, 컴퓨터가 현실 세계와 상호작용할 수 있게 하는 핵심 기술임
> - 자율주행에서는 매 순간 정확하고 빠르게 주변 객체를 파악하는 것이 안전과 직결되기 때문에, 객체 탐지 모델의 성능 향상은 곧 자율주행 기술의 발전을 의미함
> - 객체 탐지의 기본적인 원리부터 딥러닝 기반 모델의 종류, 그리고 실제 코드에서의 적용 방법을 이해할 것
{: .expert-quote}