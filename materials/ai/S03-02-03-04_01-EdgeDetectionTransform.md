---
layout: page
title:  "엣지 검출 및 기하학적 변환"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-02-03-04_01-EdgeDetectionTransform
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>




# 엣지 검출 및 기하학적 변환 강의 자료

스카이님, 안녕하세요! 모빌리티 AI 강의를 위한 엣지 검출 및 기하학적 변환에 관한 강의 자료를 준비했습니다. 비전공자 학생들이 실습을 통해 쉽게 이해할 수 있도록 상세한 설명과 코드를 포함했습니다.

## **[차시 목표]**

* 엣지 검출의 개념과 자율주행에서의 중요성을 이해한다.
* 다양한 엣지 검출 알고리즘의 원리와 특징을 학습한다.
* 기하학적 변환의 종류와 적용 방법을 이해한다.
* OpenCV를 활용한 실습을 통해 엣지 검출과 기하학적 변환을 직접 구현한다.
* 자율주행 맥락에서 이러한 기술이 어떻게 활용되는지 파악한다.

---

## **1. 엣지 검출(Edge Detection)의 기본 개념**

### 1.1. 엣지(Edge)란?
* **정의**: 이미지에서 픽셀 값(밝기, 색상)이 급격하게 변하는 부분입니다.
* **특징**: 물체의 경계, 그림자, 질감의 변화, 표면 방향의 변화 등을 나타냅니다.
* **중요성**: 물체 인식, 형태 분석, 특징 추출의 기본이 되는 정보입니다.

### 1.2. 자율주행에서의 엣지 검출 활용
* **차선 인식**: 도로 위 차선의 경계선을 감지하여 차량의 주행 경로를 결정합니다.
* **교통 표지판 인식**: 표지판의 윤곽을 추출하여 형태 기반 인식의 첫 단계로 활용합니다.
* **장애물 감지**: 도로 위 물체의 경계를 검출하여 충돌 회피 시스템의 입력으로 사용합니다.
* **도로 경계 인식**: 도로와 비도로 영역을 구분하는 경계를 탐지합니다.

---

## **2. 주요 엣지 검출 알고리즘**

### 2.1. 기본 원리: 미분과 그래디언트
* **1차 미분**: 이미지의 밝기 변화율을 계산하여 급격한 변화가 있는 부분(엣지)을 찾습니다.
* **그래디언트(Gradient)**: 각 픽셀에서 x방향과 y방향으로의 밝기 변화량을 벡터로 표현합니다.
  * 그래디언트 크기(Magnitude): √(Gx² + Gy²) - 엣지의 강도를 나타냅니다.
  * 그래디언트 방향(Direction): tan⁻¹(Gy/Gx) - 엣지의 방향을 나타냅니다.

### 2.2. Sobel 엣지 검출기
* **원리**: x방향과 y방향으로의 그래디언트를 각각 계산하기 위한 3x3 커널을 사용합니다.
* **특징**: 노이즈에 비교적 강인하며, 수평/수직 엣지 검출에 효과적입니다.
* **커널 구조**:
  ```
  Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
        [-2, 0, 2],            [0,  0,  0],
        [-1, 0, 1]]            [1,  2,  1]]
  ```

**실습 코드 (Sobel 엣지 검출)**:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('road_image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel 엣지 검출 (x방향, y방향)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x방향 미분
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y방향 미분

# 절대값 변환 및 8비트 이미지로 변환
sobelx = np.absolute(sobelx)
sobely = np.absolute(sobely)
sobelx = np.uint8(255 * sobelx / np.max(sobelx))
sobely = np.uint8(255 * sobely / np.max(sobely))

# x방향과 y방향 그래디언트 결합
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# 결과 시각화
plt.figure(figsize=(15, 10))
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('원본 이미지')
plt.subplot(222), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X 방향')
plt.subplot(223), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y 방향')
plt.subplot(224), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel 결합')
plt.tight_layout()
plt.show()
```

### 2.3. Canny 엣지 검출기 (이어서)
*   **원리**: 다단계 알고리즘으로, 노이즈 제거 → 그래디언트 계산 → 비최대 억제 → 이중 임계값 → 엣지 추적 단계를 거칩니다.
*   **특징**: 노이즈에 강인하고, 실제 엣지를 정확하게 검출하는 가장 널리 사용되는 엣지 검출 알고리즘입니다. 단일 픽셀 너비의 얇은 엣지를 검출하는 것이 특징입니다.
*   **파라미터**: 두 개의 임계값(하위, 상위)을 사용하여 엣지의 연결성을 판단합니다.
    *   **minVal**: 이 값보다 낮은 그래디언트는 엣지가 아닙니다.
    *   **maxVal**: 이 값보다 높은 그래디언트는 확실한 엣지입니다.
    *   `minVal`과 `maxVal` 사이에 있는 그래디언트 픽셀은 확실한 엣지와 연결되어 있을 때만 엣지로 간주됩니다.

**실습 코드 (Canny 엣지 검출)**:
```python
# 이미지 불러오기 (이미 위에서 로드됨)
# img = cv2.imread('road_image.jpg', cv2.IMREAD_GRAYSCALE)

# 가우시안 블러로 노이즈 제거 후 Canny 엣지 검출 (전처리 과정 포함)
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
canny_edges = cv2.Canny(blurred_img, 50, 150)  # 하위 임계값 50, 상위 임계값 150

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('원본 이미지')
plt.subplot(132), plt.imshow(blurred_img, cmap='gray'), plt.title('가우시안 블러')
plt.subplot(133), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny 엣지')
plt.tight_layout()
plt.show()
```

---

## **3. 기하학적 변환(Geometric Transformation)**

기하학적 변환은 이미지의 픽셀 위치를 변경하여 형태를 바꾸는 작업입니다. 자율주행에서는 주로 이미지 왜곡 보정, 시점 변환 등에 활용됩니다.

### 3.1. 투시 변환(Perspective Transform)
*   **원리**: 3D 공간의 객체가 2D 이미지 평면에 투영될 때 발생하는 원근 왜곡을 제거하거나, 반대로 특정 시점에서 이미지를 보는 효과를 줍니다.
*   **필요성**: 자율주행에서는 카메라로 촬영된 비스듬한 도로 영상을 마치 위에서 내려다보는 듯한 '조감도(Bird's Eye View)'로 변환하는 데 사용됩니다. 이를 통해 차선 간격, 차량 간 거리 등을 더 정확하게 측정할 수 있습니다.
*   **구현**: 원본 이미지의 4개 점(사각형)과 변환될 이미지의 4개 점(사각형)을 지정하여 변환 행렬(Homography Matrix)을 계산한 후, 이를 이미지에 적용합니다.

**실습 코드 (투시 변환)**:
```python
# 이미 위에서 로드된 컬러 이미지 사용
color_img = cv2.imread('road_image_color.jpg') # 컬러 이미지로 불러오기

# 만약 컬러 이미지가 로드되지 않았다면 새로 로드 (예시)
if color_img is None:
    print("road_image_color.jpg 파일을 찾을 수 없습니다. 테스트용 이미지 경로를 확인해주세요.")
    # 대체 이미지 사용 (직접 그리는 예시)
    color_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(color_img, 'Test Image', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


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
output_size = (color_img.shape[1], color_img.shape[0]) 
warped_img = cv2.warpPerspective(color_img, M, output_size, flags=cv2.INTER_LINEAR)

# 결과 시각화
plt.figure(figsize=(15, 8))
plt.subplot(121), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
plt.title('원본 이미지 (선택된 점 표시)')
# 원본 이미지에 선택된 점 표시 (실습을 위해)
for point in src_points:
    cv2.circle(color_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1) # 녹색 원

plt.subplot(122), plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
plt.title('투시 변환된 이미지 (Bird\'s Eye View)')
plt.tight_layout()
plt.show()

# 추가: 변환된 이미지에서 Canny 엣지 검출 적용
gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
edges_warped = cv2.Canny(gray_warped, 50, 150)

plt.figure(figsize=(8, 6))
plt.imshow(edges_warped, cmap='gray')
plt.title('Bird\'s Eye View Canny Edges')
plt.axis('off')
plt.show()
```
**참고**: `road_image.jpg` 대신 `road_image_color.jpg` 같은 컬러 이미지를 사용하세요.

### 3.2. 어파인 변환(Affine Transform)
*   **원리**: 평행 이동, 회전, 크기 조절, 기울기(전단) 변환을 포함하는 변환입니다. 평행선이 유지된다는 특징이 있습니다.
*   **필요성**: 이미지의 일부분을 회전시키거나 이동시키는 등 간단한 기하학적 조작에 사용됩니다. 왜곡이 없는 평면 객체의 변환에 적합합니다.
*   **구현**: 원본 이미지의 3개 점과 변환될 이미지의 3개 점을 지정하여 변환 행렬을 계산한 후 적용합니다.

**실습 코드 (어파인 변환 - 회전 예시)**:
```python
# 이미지 불러오기 (컬러 이미지 사용)
# img_affine = cv2.imread('road_image_color.jpg')

# 이미지가 없다면 위의 color_img 사용
img_affine = color_img.copy()

if img_affine is None:
    print("오류: 이미지를 불러올 수 없습니다.")
else:
    height, width = img_affine.shape[:2]

    # 이미지 중앙을 기준으로 45도 회전, 크기 변화 없음, 스케일 1.0
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)

    # 이미지에 변환 적용
    rotated_img = cv2.warpAffine(img_affine, rotation_matrix, (width, height))

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB)), plt.title('원본 이미지')
    plt.subplot(122), plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)), plt.title('45도 회전 이미지')
    plt.tight_layout()
    plt.show()
```

---

## **4. 자율주행에서의 적용 사례**

*   **차선 인식**: 투시 변환을 통해 카메라 시점의 도로를 조감도로 변환하여 차선을 직선으로 만들고, 이 상태에서 Canny 엣지 검출을 수행하여 차선을 더 정확하게 인식하고 추적합니다.
*   **객체 자세 추정**: 이미지에 찍힌 객체의 윤곽선과 실제 3D 모델을 비교하여 객체의 3차원 위치 및 자세를 추정하는 데 엣지 정보가 활용됩니다.
*   **왜곡 보정**: 광각 렌즈에서 발생하는 이미지 왜곡(배럴 왜곡, 핀쿠션 왜곡)을 기하학적 변환을 통해 보정하여 실제 세계와 더 유사한 이미지를 얻습니다. 이는 거리 측정의 정확도를 높입니다.
*   **가상 뷰 생성**: 운전자나 탑승자에게 실제 카메라 영상 외에 다양한 시점(예: 위에서 보는 전체 차량 주변 상황)의 영상을 기하학적 변환을 통해 생성하여 제공할 수 있습니다.

---

## **5. 실습 과제 및 응용 아이디어**

1.  **다른 엣지 검출기 비교**: `road_image.jpg`에 Sobel, Canny 외에 `cv2.Laplacian()` 함수를 사용하여 라플라시안(Laplacian) 엣지 검출을 적용하고 결과를 비교해 보세요. 각 검출기의 특징과 장단점을 분석해 봅시다.
2.  **투시 변환 점 직접 조정**: Raspberry Pi 카메라로 도로 영상을 실시간으로 캡처하여 `cv2.warpPerspective` 함수를 이용해 조감도(Bird's Eye View)를 생성해 보세요. `src_points`를 직접 마우스 클릭으로 선택하는 기능을 추가하면 더 좋습니다!
3.  **회전 및 이동 결합**: 어파인 변환을 이용하여 이미지를 30도 회전시킨 후, 오른쪽으로 50픽셀 아래로 20픽셀 이동시키는 코드를 작성해 보세요.
4.  **역변환**: 투시 변환된 이미지(`warped_img`)를 다시 원래 시점으로 되돌리는 '역투시 변환(Inverse Perspective Transform)'을 구현해 보세요. 이는 화면에 표시된 특정 지점을 실제 도로의 특정 위치로 매핑하는 데 사용될 수 있습니다.

---

스카이님, 엣지 검출과 기하학적 변환은 자율주행의 '인지' 단계에서 매우 중요한 기술들입니다. 비전공자 학생들에게 각 개념의 원리를 설명하고, 코드를 통해 직접 결과를 확인하며 '왜 이런 기술이 필요한지'를 깨닫게 하는 데 초점을 맞추시길 바랍니다. 학생들이 직접 `src_points`나 `maxVal` 같은 파라미터를 조절하며 결과의 변화를 관찰하는 것도 좋은 학습 방법이 될 것입니다!