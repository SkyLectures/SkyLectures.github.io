---
layout: page
title:  "엣지 검출 및 기하학적 변환"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-02-03-04_01-EdgeDetectionTransform
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/ai/notebooks/S03-02-03-04_01-EdgeDetectionTransform.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

## 1. 엣지 검출의 기본 개념

- **엣지(Edge)란?**
    - **정의**: 이미지에서 픽셀 값(밝기, 색상)이 급격하게 변하는 부분
    - **특징**: 물체의 경계, 그림자, 질감의 변화, 표면 방향의 변화 등을 나타냄
    - **중요성**: 물체 인식, 형태 분석, 특징 추출의 기본이 되는 정보

- **자율주행에서의 엣지 검출(Edge Detection) 활용**
    - **차선 인식**: 도로 위 차선의 경계선을 감지하여 차량의 주행 경로 결정
    - **교통 표지판 인식**: 표지판의 윤곽을 추출하여 형태 기반 인식의 첫 단계로 활용
    - **장애물 감지**: 도로 위 물체의 경계를 검출하여 충돌 회피 시스템의 입력으로 사용
    - **도로 경계 인식**: 도로와 비도로 영역을 구분하는 경계 탐지


## 2. 주요 엣지 검출 알고리즘

### 2.1. 기본 원리: 미분과 그래디언트

- **1차 미분**
    - 이미지의 밝기 변화율을 계산하여 급격한 변화가 있는 부분(엣지) 검출

- **그래디언트(Gradient)**
    - 각 픽셀에서 x방향과 y방향으로의 밝기 변화량을 벡터로 표현
        - 그래디언트 크기(Magnitude): $$\sqrt{(Gx^2 + Gy^2)} - 엣지의 강도$$ 
        - 그래디언트 방향(Direction): $$tan^{-1}{(Gy/Gx)} - 엣지의 방향$$


### 2.2. Sobel 엣지 검출기
- **원리**
    - x방향과 y방향으로의 그래디언트를 각각 계산하기 위한 3x3 커널 사용

- **특징**
    - 노이즈에 비교적 강인
    - 수평/수직 엣지 검출에 효과적

- **커널 구조**

  ```
  Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
        [-2, 0, 2],            [0,  0,  0],
        [-1, 0, 1]]            [1,  2,  1]]
  ```

- **실습 코드 (Sobel 엣지 검출)**

```python
#// file: "edge_detection.py"
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img_gray = cv2.imread('./images/road_image.jpg', cv2.IMREAD_GRAYSCALE)

if img_gray is not None:
    # Sobel 엣지 검출 (x방향, y방향)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # x방향 미분
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # y방향 미분

    # 절대값 변환 및 8비트 이미지로 변환
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    sobely = np.uint8(255 * sobely / np.max(sobely))

    # x방향과 y방향 그래디언트 결합
    sobel_combined = cv2.bitwise_or(sobelx, sobely)

    cv2.putText(sobelx, "SobelX Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(sobely, "SobelY Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(sobel_combined, "Sobel Combined Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    row1 = np.hstack((img_gray, sobelx))
    row2 = np.hstack((sobely, sobel_combined))
    final_display_image = np.vstack((row1, row2))

    cv2.imshow('Sobel Edge Detection Results', final_display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 2.3. Canny 엣지 검출기

- **원리**
    - 노이즈 제거 ➜ 그래디언트 계산 ➜ 비최대 억제 ➜ 이중 임계값 ➜ 엣지 추적 단계를 거치는 다단계 알고리즘

- **특징**
    - 노이즈에 강인
    - 실제 엣지를 정확하게 검출하는 가장 널리 사용되는 엣지 검출 알고리즘
    - 단일 픽셀 너비의 좁은 엣지를 검출하는 것이 특징

- **파라미터**
    - 두 개의 임계값(하위, 상위)을 사용하여 엣지의 연결성 판단
        - **minVal**: 이 값보다 낮은 그래디언트는 엣지가 아니
        - **maxVal**: 이 값보다 높은 그래디언트는 확실한 엣지임
        - `minVal`과 `maxVal` 사이에 있는 그래디언트 픽셀은 확실한 엣지와 연결되어 있을 때만 엣지로 간주함

- **실습 코드 (Canny 엣지 검출)**

```python
#// file: "edge_detection.py"
if img_gray is not None:
    # 가우시안 블러로 노이즈 제거 후 Canny 엣지 검출 (전처리 과정 포함)
    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_img, 50, 150)  # 하위 임계값 50, 상위 임계값 150

    cv2.putText(blurred_img, "Gaussian Blur Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canny_edges, "Canny Edge Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    final_display_image = np.hstack((img_gray, blurred_img, canny_edges))

    cv2.imshow('Canny Edge Detection Results', final_display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```


## 3. 기하학적 변환

- Geometric Transformation
- 이미지의 픽셀 위치를 변경하여 형태를 바꾸는 작업
- 자율주행에서는 주로 이미지 왜곡 보정, 시점 변환 등에 활용

### 3.1. 투시 변환

- Perspective Transform
- **원리**
    - 3D 공간의 객체가 2D 이미지 평면에 투영될 때 발생하는 원근 왜곡을 제거하거나,
    - 반대로 특정 시점에서 이미지를 보는 효과 부여

- **필요성**
    - 카메라로 촬영된 비스듬한 도로 영상을 마치 위에서 내려다보는 듯한 '조감도(Bird's Eye View)'로 변환하는 데 사용
    - 이를 통해 차선 간격, 차량 간 거리 등을 더 정확하게 측정 가능

- **구현**
    - 원본 이미지의 4개 점(사각형)과 변환될 이미지의 4개 점(사각형)을 지정하여 변환 행렬(Homography Matrix)을 계산한 후,
    - 이를 이미지에 적용

- **실습 코드 (투시 변환)**

```python
#// file: "edge_detection.py"
img_color = cv2.imread('./images/road_image.jpg') # 컬러 이미지로 불러오기

# 1. 원본 이미지에서 4개의 점 선택 (도로의 사다리꼴 영역)
# 일반적으로 화면 하단 넓게, 상단 좁게 설정하여 도로 영역을 추출
# (이 값들은 사용하는 이미지에 따라 조정해야 합니다!)
# (예: 아래 4점은 640x480 이미지에 대한 가정값)
src_points = np.float32([
    [100, 470],  # 좌하단
    [540, 470],  # 우하단
    [380, 290],  # 우상단
    [260, 290]   # 좌상단
])

# 2. 변환될 이미지에서 4개의 점 선택 (직사각형 영역, 위에서 본 모습)
# (변환 후 이미지의 크기를 고려하여 목적지 점 설정)
dst_points = np.float32([
    [100, 470],  # 좌하단
    [540, 470],  # 우하단
    [540, 0],   # 우상단
    [100, 0]    # 좌상단
])

# 3. 투시 변환 행렬 계산
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 4. 이미지에 변환 적용
# 출력 이미지 크기를 원본과 동일하게 설정하거나, 원하는 크기로 지정
output_size = (img_color.shape[1], img_color.shape[0]) 
warped_img = cv2.warpPerspective(img_color, M, output_size, flags=cv2.INTER_LINEAR)

# 결과 시각화

# 원본 이미지에 선택된 점 표시 (실습을 위해)
for point in src_points:
    cv2.circle(img_color, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1) # 녹색 원

# 추가: 변환된 이미지에서 Canny 엣지 검출 적용
# Canny 함수를 적용한 결과 이미지는 3채널로 변환하여 원본과 나란히 표시
gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
edges_warped = cv2.Canny(gray_warped, 50, 150)
edges_warped = cv2.cvtColor(edges_warped, cv2.COLOR_GRAY2BGR)

cv2.putText(warped_img, "Bird\'s Eye View", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(edges_warped, "Bird\'s Eye View Canny Edges", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

final_display_image = np.hstack((img_color, warped_img, edges_warped))

cv2.imshow('Canny Edge Detection Results', final_display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


### 3.2. 어파인 변환

- Affine Transform
- **원리**
    - 평행 이동, 회전, 크기 조절, 기울기(전단) 변환을 포함하는 변환
    - 평행선이 유지된다는 특징을 가짐

- **필요성**
    - 이미지의 일부분을 회전시키거나 이동시키는 등 간단한 기하학적 조작에 사용
    - 왜곡이 없는 평면 객체의 변환에 적합함

- **구현**
    - 원본 이미지의 3개 점과 변환될 이미지의 3개 점을 지정하여 변환 행렬을 계산한 후 적용

- **실습 코드 (어파인 변환 - 회전 예시)**

```python
#// file: "edge_detection.py"
img_affine = img_color.copy()
height, width = img_affine.shape[:2]

# 이미지 중앙을 기준으로 45도 회전, 크기 변화 없음, 스케일 1.0
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)

# 이미지에 변환 적용
rotated_img = cv2.warpAffine(img_affine, rotation_matrix, (width, height))

# 결과 시각화
cv2.putText(rotated_img, "45 Degree Rotated Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

display_image = np.hstack((img_color, rotated_img))

cv2.imshow('Canny Edge Detection Results', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## 4. 자율주행에서의 적용 사례

- **차선 인식**
    - 투시 변환을 통해 카메라 시점의 도로를 조감도로 변환하여 차선을 직선으로 만들고,
    - 이 상태에서 Canny 엣지 검출을 수행하여 차선을 더 정확하게 인식하고 추적함

- **객체 자세 추정**
    - 이미지에 찍힌 객체의 윤곽선과 실제 3D 모델을 비교하여
    - 객체의 3차원 위치 및 자세를 추정하는 데 엣지 정보 활용

- **왜곡 보정**
    - 광각 렌즈에서 발생하는 이미지 왜곡(배럴 왜곡, 핀쿠션 왜곡)을 기하학적 변환을 통해 보정하여
    - 실제 세계와 더 유사한 이미지 획득
    - 이는 거리 측정의 정확도를 높임

- **가상 뷰 생성**
    - 운전자나 탑승자에게 실제 카메라 영상 외에 다양한 시점(예: 위에서 보는 전체 차량 주변 상황)의 영상을 기하학적 변환을 통해 생성하여 제공할 수 있음

