---
layout: page
title:  "영상 전처리 및 필터링"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-02-03-03_01-ImagePreprocessingFiltering
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/ai/notebooks/S03-02-03-03_01-ImagePreprocessingFiltering.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

## 1. 영상 전처리 개요

### 1.1. 영상 전처리란?
- **정의**: 본격적인 영상 분석이나 객체 인식 전에 영상의 품질을 개선하거나 특정 특징을 강조하는 과정
- **목적**: 노이즈 제거, 선명도 향상, 밝기 조정 등을 통해 후속 처리 단계의 정확도와 효율성 향상

### 1.2. 자율주행에서 영상 전처리의 중요성
- **안정적인 특징 추출**: 다양한 환경(비, 눈, 안개, 밤)에서의 일관된 특징 추출 지원
- **계산 효율성**: 관심 영역(ROI: Region of Interest) 설정 및 해상도 조정을 통해 처리 시간 단축
- **인식 정확도 향상**: 노이즈 제거 및 중요 특징 강조를 통해 객체 인식, 차선 검출 등의 정확도 향상


## 2. 기본적인 영상 전처리 기법

### 2.1. 색상 공간 변환

- **내용**
    - 이미지를 다른 색상 공간으로 변환
        - 예: BGR ➜ 그레이스케일, BGR ➜ HSV
    - 그레이스케일 변환
        - RGB 컬러 영상을 흑백 영상으로 변환(Color to Grayscale)
        - 일반적으로 R, G, B 채널에 각각 다른 가중치를 적용함
        - 그레이스케일 = 0.299×R + 0.587×G + 0.114×B

- **사용함수**: `cv2.cvtColor()`
    - cv2.cvtColor()는 이미지의 색상 공간(Color Space)을 변환하는 데 사용되는 OpenCV의 핵심 함수
    - 주로 BGR (OpenCV가 이미지를 읽는 기본 형식) 이미지와 RGB (Matplotlib이나 웹 등에서 일반적으로 사용하는 형식)의 변환에 사용
    - 변환코드
        - cv2.COLOR_BGR2RGB: BGR ➜ RGB
        - cv2.COLOR_RGB2BGR: RGB ➜ BGR
        - cv2.COLOR_BGR2GRAY: BGR ➜ 그레이스케일
        - cv2.COLOR_GRAY2BGR: 그레이스케일 ➜ BGR
        - cv2.COLOR_BGR2HSV: BGR ➜ HSV (Hue, Saturation, Value)
        - cv2.COLOR_HSV2BGR: HSV ➜ BGR 등        

- **용도**
    - 자율주행에서 특정 색상을 강조하거나, 색상 정보가 불필요한 연산(에지 검출, 형태 인식 등)에서 속도 향상을 위해 사용됨

```python
#// file: "image_preprocess.py"
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('./images/traffic_light.jpg')

if img_color is not None:
    # 원본 이미지 표시
    cv2.imshow('Original Image', img_color)
    cv2.waitKey(0)

    # BGR을 그레이스케일로 변환
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', img_gray)
    print(f"그레이스케일 이미지 형태: {img_gray.shape}") # 예: (480, 640)
    cv2.waitKey(0)

    # BGR을 HSV로 변환
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', img_hsv)
    print(f"HSV 이미지 형태: {img_hsv.shape}") # 예: (480, 640, 3)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
```


### 2.2. 이미지 크기 조정

- **내용**
    - 이미지의 가로와 세로 크기를 변경하는 작업 (Resizing)

- **사용함수**: `cv2.resize()`
    - `interpolation` (보간법) 파라미터: 이미지를 늘리거나 줄일 때 픽셀 값을 어떻게 채울지 결정
        - 크기 변경 시 새로운 픽셀 값을 계산하는 방법
        - 종류
            - **최근접 이웃(Nearest Neighbor)**: 가장 가까운 픽셀 값 사용 (빠르지만 품질 낮음)
            - **선형(Linear)**: 주변 픽셀 값의 가중 평균 사용 (중간 속도와 품질)
            - **큐빅(Cubic)**: 더 넓은 범위의 픽셀 값을 사용 (느리지만 품질 좋음)

- **용도**
    - 처리 속도 향상, 메모리 사용량 감소, 딥러닝 모델 입력 크기 통일 등에 활용

```python
#// file: "image_preprocess.py"
if img_color is not None:
    #이미지 크기 조정 (너비 320px, 높이 240px)
    resized = cv2.resize(img_color, (320, 240))
    cv2.imshow('320x240 Resized Image', resized)
    cv2.waitKey(0) # 키 입력이 있을 때까지 무한정 대기

    # 이미지 절반 크기로 줄이기 (가로/세로 0.5배)
    img_resized_half = cv2.resize(img_color, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Half', img_resized_half)
    print(f"절반 크기 이미지 형태: {img_resized_half.shape}")
    cv2.waitKey(0)

    # 이미지 두 배 크기로 늘리기 (가로/세로 2배)
    img_resized_double = cv2.resize(img_color, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Resized Double', img_resized_double)
    print(f"두 배 크기 이미지 형태: {img_resized_double.shape}")
    cv2.waitKey(0)

    # 다양한 보간법 적용
    resized_nearest = cv2.resize(img_color, (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Nearest-neighbor Interpolation', resized_nearest)   # 최근접 이웃 보간법/근접 보간겁
    cv2.waitKey(0) # 키 입력이 있을 때까지 무한정 대기

    resized_linear = cv2.resize(img_color, (320, 240), interpolation=cv2.INTER_LINEAR)  # 기본값
    cv2.imshow('Linear Interpolation', resized_linear)   # 선형 보간법
    cv2.waitKey(0) # 키 입력이 있을 때까지 무한정 대기

    resized_cubic = cv2.resize(img_color, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Cubic Interpolation', resized_cubic)    # 큐빅 보간법/삼차 보간법
    cv2.waitKey(0) # 키 입력이 있을 때까지 무한정 대기
    
    cv2.destroyAllWindows() # 모든 창 닫기
```


### 2.3. 관심영역 설정/ 추출

- **내용**
    - 이미지의 특정 관심 영역(Region of Interest, ROI)만 선택하여 처리
    - 불필요한 영역을 제외하고 처리함으로써 연산 효율 향상
    - 자율주행 시에는 주로 도로 영역만 선택하여 차선이나 장애물 인식을 수행하는 데 사용

- **용도**
    - 처리 효율성 향상, 불필요한 영역 제외, 특정 영역(예: 도로, 차선)에 집중할 때 사용

```python
#// file: "image_preprocess.py"
if img_color is not None:
    # 이미지 하단 절반을 ROI로 설정 (도로 영역이라고 가정)
    height, width = img_color.shape[:2]
    roi_start_y = int(height * 0.5) # 이미지 높이의 절반부터 시작
    roi_end_y = height
    roi_start_x = 0
    roi_end_x = width

    roi = img_color[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    
    cv2.imshow('Original Image', img_color)
    cv2.imshow('Region of Interest (ROI)', roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```


### 2.4 픽셀 접근 및 수정

- **내용**
    - NumPy 배열의 인덱싱을 사용하여 특정 픽셀의 값에 접근하거나 수정할 수 있음
    - 컬러 이미지의 경우 `[y좌표, x좌표, 채널]` 형태로 접근 (채널 순서: BGR)

```python
#// file: "image_preprocess.py"
if img_color is not None:
    # (높이-1, 너비-1) 픽셀의 BGR 값 확인 (우측 하단)
    # y좌표: img_color.shape[0] - 1, x좌표: img_color.shape[1] - 1
    # Note: NumPy 배열 인덱싱은 [행, 열] 또는 [y, x] 순서
    bottom_right_pixel = img_color[img_color.shape[0] - 1, img_color.shape[1] - 1]
    print(f"우측 하단 픽셀 (BGR): {bottom_right_pixel}") # 예: [120, 100, 80]

    # 특정 픽셀 (예: 이미지 중앙)의 색상을 빨간색으로 변경
    center_y, center_x = img_color.shape[0] // 2, img_color.shape[1] // 2
    # B=0, G=0, R=255 (OpenCV는 BGR 순서이므로 R 값을 마지막에 넣음)
    img_color[center_y, center_x] = [0, 0, 255] # B=0, G=0, R=255 (순수한 빨간색)

    # 수정된 이미지를 확인
    cv2.imshow('Modified Image', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```


## 3. OpenCV를 이용한 기본 영상 처리

### 3.1 이미지 이진화

- 이미지의 픽셀 값을 특정 기준(임계값)에 따라 흑 또는 백으로 만드는 과정
- 배경과 객체를 분리하거나, 특정 특징을 강조할 때 사용
    - 예: 어두운 도로에서 밝은 차선 추출

- **사용함수**: `cv2.threshold()`

```python
#// file: "image_process.py"
if img_color is not None:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 전역 이진화: 127을 임계값으로 사용
    # 127보다 크면 255(흰색), 작거나 같으면 0(검은색)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', img_binary)

    # 이진화 결과도 종종 유용한 정보를 포함함
    print(f"이진화 반환 값 (ret): {ret}") 

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 3.2 이미지 블러링

- 이미지를 부드럽게 만들어 노이즈를 줄이는 필터링 작업
- 미세한 노이즈로 인한 오작동 방지
- 에지 검출 등의 전처리 단계로 자주 사용

- **사용함수**: `cv2.GaussianBlur()`

```python
#// file: "image_process.py"
if img_color is not None:
    # 가우시안 블러 (5x5 커널 크기)
    img_blur = cv2.GaussianBlur(img_color, (5, 5), 0)
    cv2.imshow('Original Image', img_color)
    cv2.imshow('Blurred Image', img_blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 3.3 에지(Edge) 검출

- 이미지에서 밝기 변화가 급격한 부분을 찾아 선(경계선)으로 표시
- 객체의 윤곽선을 추출하여 자율주행 시 차량, 도로 경계 등을 파악하는 데 활용
- Canny 에지 검출은 노이즈 억제, 에지 방향 찾기, 이력 임계값 적용 등 여러 단계로 구성되어 있어 매우 강력함

- **사용함수**: `cv2.Canny()`

```python
#// file: "image_process.py"
if img_color is not None:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) # 노이즈 제거 후 에지 검출

    # Canny 에지 검출 (하위 임계값=50, 상위 임계값=150)
    edges = cv2.Canny(img_blur, 50, 150)
    cv2.imshow('Original Image', img_color)
    cv2.imshow('Edges Image', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```


## 4. 이미지 품질 향상 기법

### 4.1. 히스토그램 평활화 

- **내용**
    - 이미지의 픽셀 값 분포(히스토그램)를 균등하게 재분배하여 명암 대비를 향상시키는 기법(Histogram Equalization)
    - 어둡거나 밝기에 치우친 이미지의 가시성을 높이는 데 효과적
- **용도**
    - 저조도 환경에서 촬영된 이미지의 가시성 개선, 의료 영상 등 대비가 중요한 분야에서 활용

```python
#// file: "image_preprocess.py"
# 그레이스케일 이미지에 적용
img_equalized = cv2.equalizeHist(img_gray)
combined_image = np.hstack((img_gray, img_equalized))
cv2.imshow('Grayscale Image and Equalized Image', combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
```


### 4.2 노이즈 제거 필터링

- **노이즈(Noise)란?**
    - **정의**
        - 이미지에 불필요하게 섞여 들어가 화질을 저하시키는 임의의 신호나 왜곡

    - **원인**
        - 센서 결함, 조명 부족, 전송 오류 등 다양함

    - **자율주행에서 문제점**
        - 노이즈는 객체 인식이나 차선 검출의 정확도를 떨어뜨려 오작동의 원인이 될 수 있음

- **블러링(Blurring) 필터: 이미지 부드럽게 만들기**
    - **원리**
        - 픽셀 주변의 픽셀 값들을 평균하거나 가중 평균하여 해당 픽셀의 값을 업데이트
        - 이미지의 날카로운 부분을 부드럽게 만들고 노이즈를 효과적으로 줄여줌

    - **커널(Kernel)**
        - 필터링 시 주변 픽셀을 얼마나 고려할지 결정하는 작은 행렬
        - 커널 크기가 커질수록 더 많이 블러링됨

    - **종류**
        - **평균 블러 (Averaging Blur, `cv2.blur()`)**
            - 커널 영역 내 모든 픽셀 값의 산술 평균으로 중심 픽셀 값을 대체

            ```python
            #// file: "image_preprocess.py"
            # 5x5 커널을 사용한 평균 블러
            img_blur_avg = cv2.blur(img_color, (5, 5))
            combined_image = np.hstack((img_color, img_blur_avg))
            cv2.imshow('Original Image and Averaging Blur Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ```

        - **가우시안 블러 (Gaussian Blur, `cv2.GaussianBlur()`)**
            - **원리**
                - 커널 내 픽셀에 가우시안 분포(Gaussian Distribution)에 따라 가중치를 부여하여 평균 계산
                - 중심 픽셀에 가까울수록 높은 가중치 부여
                - 엣지 보존이 평균 블러보다 우수함

            - **용도**
                - 가장 널리 사용되는 블러 필터 중 하나
                - 노이즈 제거와 함께 자연스러운 블러 효과 제공

            ```python
            #// file: "image_preprocess.py"
            # 5x5 커널, 시그마 X/Y = 0을 사용한 가우시안 블러
            img_blur_gaussian = cv2.GaussianBlur(img_color, (5, 5), 0)
            combined_image = np.hstack((img_color, img_blur_gaussian))
            cv2.imshow('Original Image and Gaussian Blur Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ```

        - **미디언 블러 (Median Blur, `cv2.medianBlur()`)**
            - **원리**
                - 커널 영역 내 픽셀 값들을 정렬하여 중앙값(Median)으로 중심 픽셀 값을 대체

            - **용도**
                - 특히 소금-후추(Salt-and-Pepper) 노이즈와 같이 특정 픽셀이 강하게 튀는 노이즈 제거에 매우 효과적
                - 이미지의 엣지 보존에 강점

            ```python
            #// file: "image_preprocess.py"
            # 소금-후추 노이즈를 가진 이미지 생성 (테스트용)
            noisy_img = img_color.copy()
            num_salt = int(img_color.size * 0.005) # 0.5% 소금 노이즈
            coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
            noisy_img[coords[0], coords[1], :] = 255
            num_pepper = int(img_color.size * 0.005) # 0.5% 후추 노이즈
            coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_img.shape]
            noisy_img[coords[0], coords[1], :] = 0

            # 5x5 커널을 사용한 미디언 블러
            img_blur_median = cv2.medianBlur(noisy_img, 5) # 커널 크기는 홀수여야 함
            combined_image = np.hstack((noisy_img, img_blur_median))
            cv2.imshow('Noisy Image and Median Blur Image', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ```


### 4.3 이미지 선명화

- **선명화(Sharpening) 필터의 필요성**
    - **원리**
        - 블러링 필터와 반대로 이미지의 엣지나 세부적인 특징을 강조하여 이미지를 더 선명하게 민듦

    - **용도**
        - 노이즈 제거 후 흐려진 엣지를 다시 강조하거나,
        - 중요한 디테일을 부각시켜 인식률을 높일 때 사용

- **고주파 필터 (High-pass Filter) 및 언샤프 마스킹**
    - **원리**
        - 고주파 성분(엣지, 세부 정보)은 통과시키고 저주파 성분(부드러운 영역)은 제거하는 필터

    - **언샤프 마스킹(Unsharp Masking)**
        - 원본 이미지에서 흐려진 이미지를 빼서 '엣지 정보'만 추출한 후,
        - 이 엣지 정보를 원본 이미지에 다시 더해 이미지를 더 선명하게 만드는 기법

    ```python
    #// file: "image_preprocess.py"
    # 5x5 커널, 시그마 X/Y = 0을 사용한 가우시안 블러 (흐린 이미지 생성) 대상
    # 원본 - 흐린 이미지 = 엣지 강조 이미지 (마스크)
    # 이미지 데이터 타입이 uint8이므로 연산 시 주의 (np.float32로 변환 후 연산)
    img_unsharp_mask = cv2.addWeighted(img_color, 1.5, img_blur_gaussian, -0.5, 0) # 원본에 1.5배, 블러된 이미지에 -0.5배 가중치를 주고 0을 더함
    combined_image = np.hstack((img_blur_gaussian, img_unsharp_mask))
    cv2.imshow('Gaussian Blur Image and Unsharp Mask Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    ```


## 5. 자율주행에서의 적용 사례

- **노이즈 제거**
    - 악천후, 저조도 환경에서 획득된 센서 데이터(카메라, 라이다)의 노이즈를
    - 미디언 블러 등으로 제거하여 오탐지 감소

- **차선 인식 전처리**
    - 카메라 영상에서 그레이스케일 변환, 가우시안 블러, ROI 설정 등을 통해 차선 영역을 명확히 하고
    - 후속 알고리즘(예: Canny 에지 검출, Hough 변환)의 성능을 향상시킴

- **교통 표지판/신호등 인식 전처리**
    - 히스토그램 평활화로 이미지 대비를 높이거나,
    - 컬러 임계값 처리를 통해
    - 특정 색상(빨강, 노랑, 초록)의 신호등을 분리하여 인식률 향상
