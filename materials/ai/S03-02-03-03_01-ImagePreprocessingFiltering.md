---
layout: page
title:  "영상 전처리 및 필터링"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-02-03-03_01-ImagePreprocessingFiltering
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## **1. 영상 전처리 개요**

### 1.1. 영상 전처리란?
- **정의**: 본격적인 영상 분석이나 객체 인식 전에 영상의 품질을 개선하거나 특정 특징을 강조하는 과정
- **목적**: 노이즈 제거, 선명도 향상, 밝기 조정 등을 통해 후속 처리 단계의 정확도와 효율성 향상

### 1.2. 자율주행에서 영상 전처리의 중요성
- **안정적인 특징 추출**: 다양한 환경(비, 눈, 안개, 밤)에서의 일관된 특징 추출 지원
- **계산 효율성**: 관심 영역(ROI: Region of Interest) 설정 및 해상도 조정을 통해 처리 시간 단축
- **인식 정확도 향상**: 노이즈 제거 및 중요 특징 강조를 통해 객체 인식, 차선 검출 등의 정확도 향상


## **2. 기본적인 영상 전처리 기법**

### 2.1. 그레이스케일 변환

- **내용**
    - RGB 컬러 영상을 흑백 영상으로 변환(Color to Grayscale)
    - 일반적으로 R, G, B 채널에 각각 다른 가중치를 적용함
    - 그레이스케일 = 0.299×R + 0.587×G + 0.114×B

- **용도**
    - 처리 속도 향상, 색상 정보가 불필요한 경우(에지 검출, 형태 인식 등)에 사용

```python
#// file: "image_preprocess.py"
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('road_image.jpg')

# BGR을 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 결과 표시
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
plt.subplot(122), plt.imshow(gray, cmap='gray'), plt.title('그레이스케일 이미지')
plt.show()
```


### 2.2. 이미지 크기 조정

- **내용**
    - 이미지의 가로와 세로 크기를 변경하는 작업 (Resizing)

- **보간법(Interpolation)**
    - 크기 변경 시 새로운 픽셀 값을 계산하는 방법
    - 종류
        - **최근접 이웃(Nearest Neighbor)**: 가장 가까운 픽셀 값 사용 (빠르지만 품질 낮음)
        - **선형(Linear)**: 주변 픽셀 값의 가중 평균 사용 (중간 속도와 품질)
        - **큐빅(Cubic)**: 더 넓은 범위의 픽셀 값을 사용 (느리지만 품질 좋음)

- **용도**
    - 처리 속도 향상, 메모리 사용량 감소, 딥러닝 모델 입력 크기 통일 등에 활용

```python
#// file: "image_preprocess.py"
# 이미지 크기 조정 (너비 320px, 높이 240px)
resized = cv2.resize(img, (320, 240))

# 다양한 보간법 적용
resized_nearest = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
resized_linear = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)  # 기본값
resized_cubic = cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC)

# 결과 표시
plt.figure(figsize=(15, 10))
plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
plt.subplot(222), plt.imshow(cv2.cvtColor(resized_nearest, cv2.COLOR_BGR2RGB)), plt.title('최근접 이웃 보간법')
plt.subplot(223), plt.imshow(cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB)), plt.title('선형 보간법')
plt.subplot(224), plt.imshow(cv2.cvtColor(resized_cubic, cv2.COLOR_BGR2RGB)), plt.title('큐빅 보간법')
plt.show()
```


### 2.3. 관심 영역 설정

- **내용**
    - 전체 이미지에서 분석이 필요한 특정 영역만 선택하여 처리

- **용도**
    - 처리 효율성 향상, 불필요한 영역 제외, 특정 영역(예: 도로, 차선)에 집중할 때 사용

```python
#// file: "image_preprocess.py"
# 이미지 하단 절반을 ROI로 설정 (도로 영역이라고 가정)
height, width = img.shape[:2]
roi_start_y = int(height * 0.5)  # 이미지 높이의 절반부터 시작
roi = img[roi_start_y:height, 0:width]

# 결과 표시
plt.figure(figsize=(10, 8))
plt.subplot(211), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
plt.subplot(212), plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), plt.title('관심 영역 (ROI)')
plt.show()
```


## **3. 이미지 품질 향상 기법**

### 3.1. 히스토그램 평활화 

- **내용**
    - 이미지의 픽셀 값 분포(히스토그램)를 균등하게 재분배하여 명암 대비를 향상시키는 기법(Histogram Equalization)
    - 어둡거나 밝기에 치우친 이미지의 가시성을 높이는 데 효과적
- **용도**
    - 저조도 환경에서 촬영된 이미지의 가시성 개선, 의료 영상 등 대비가 중요한 분야에서 활용

```python
#// file: "image_preprocess.py"
# 그레이스케일 이미지에 적용
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray_img)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(gray_img, cmap='gray'), plt.title('원본 그레이스케일')
plt.subplot(122), plt.imshow(equalized_img, cmap='gray'), plt.title('히스토그램 평활화')
plt.show()
```


### 3.2 노이즈 제거 필터링

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
            blur_avg = cv2.blur(img, (5, 5))

            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
            plt.subplot(122), plt.imshow(cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB)), plt.title('평균 블러 (5x5)')
            plt.show()
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
            blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
            plt.subplot(122), plt.imshow(cv2.cvtColor(blur_gaussian, cv2.COLOR_BGR2RGB)), plt.title('가우시안 블러 (5x5)')
            plt.show()
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
            noisy_img = img.copy()
            num_salt = int(img.size * 0.005) # 0.5% 소금 노이즈
            coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
            noisy_img[coords[0], coords[1], :] = 255
            num_pepper = int(img.size * 0.005) # 0.5% 후추 노이즈
            coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_img.shape]
            noisy_img[coords[0], coords[1], :] = 0

            # 5x5 커널을 사용한 미디언 블러
            blur_median = cv2.medianBlur(noisy_img, 5) # 커널 크기는 홀수여야 함

            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
            plt.subplot(132), plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)), plt.title('노이즈 이미지')
            plt.subplot(133), plt.imshow(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB)), plt.title('미디언 블러 (5)')
            plt.show()
            ```


### 3.3 이미지 선명화

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
    # 5x5 커널, 시그마 X/Y = 0을 사용한 가우시안 블러 (흐린 이미지 생성)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 원본 - 흐린 이미지 = 엣지 강조 이미지 (마스크)
    # 이미지 데이터 타입이 uint8이므로 연산 시 주의 (np.float32로 변환 후 연산)
    unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, 0) # 원본에 1.5배, 블러된 이미지에 -0.5배 가중치를 주고 0을 더함

    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
    plt.subplot(122), plt.imshow(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)), plt.title('선명화 (Unsharp Masking)')
    plt.show()
    ```


## **4. 자율주행에서의 적용 사례**

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
