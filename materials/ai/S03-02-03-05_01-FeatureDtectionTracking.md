---
layout: page
title:  "특징점 검출 및 추적 기초"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-02-03-05_01-FeatureDtectionTracking
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>



# 특징점 검출 및 추적 기초 강의 자료

## **[차시 목표]**

* 특징점(Feature Point)의 개념과 자율주행에서의 중요성을 이해한다.
* 다양한 특징점 검출 알고리즘(SIFT, SURF, ORB 등)의 원리와 특징을 학습한다.
* 특징점 매칭 및 추적 기법을 이해하고 활용할 수 있다.
* OpenCV를 활용한 실습을 통해 특징점 검출 및 추적을 직접 구현한다.
* 자율주행 맥락에서 특징점 검출과 추적이 어떻게 활용되는지 파악한다.

---

## **1. 특징점(Feature Point)의 개념과 중요성**

### 1.1. 특징점이란?
* **정의**: 이미지에서 주변 픽셀과 구별되는 특징적인 점으로, 코너(corner), 블롭(blob), 엣지(edge) 등이 될 수 있습니다.
* **특징**: 이상적인 특징점은 다음 조건을 만족해야 합니다.
  * **반복성(Repeatability)**: 동일한 객체가 다른 이미지에 나타나도 같은 특징점이 검출되어야 함
  * **구별성(Distinctiveness)**: 주변 특징점과 구별될 수 있는 고유한 특성을 가져야 함
  * **지역성(Locality)**: 작은 영역에서 정의되어 폐색(occlusion)과 기하학적 변형에 강인해야 함
  * **정확성(Accuracy)**: 이미지 내에서 정확한 위치를 가져야 함
  * **효율성(Efficiency)**: 실시간 응용을 위해 빠르게 계산될 수 있어야 함

### 1.2. 자율주행에서의 특징점 활용
* **위치 추정(Localization)**: 특징점을 이용해 차량의 현재 위치를 추정하는 Visual SLAM(Simultaneous Localization And Mapping) 구현
* **객체 인식 및 추적**: 도로 위 다른 차량, 보행자, 표지판 등을 인식하고 추적
* **장면 인식**: 이전에 방문한 장소를 인식하여 루프 클로저(Loop Closure) 탐지
* **움직임 추정**: 연속된 프레임 간의 특징점 대응을 통해 카메라/차량의 움직임 추정

---

## **2. 주요 특징점 검출 알고리즘**

### 2.1. 해리스 코너 검출기(Harris Corner Detector)
* **원리**: 이미지의 작은 윈도우를 모든 방향으로 이동시켰을 때 픽셀 값의 변화가 큰 지점을 코너로 판단합니다.
* **특징**: 회전에 불변하지만 크기 변화에는 민감합니다.
* **장점**: 계산이 간단하고 빠르며, 코너 검출에 효과적입니다.

**실습 코드 (해리스 코너 검출)**:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('road_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 해리스 코너 검출
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 코너 강화 (dilate)
dst = cv2.dilate(dst, None)

# 임계값 이상인 부분을 빨간색으로 표시 (코너)
threshold = 0.01 * dst.max()
img[dst > threshold] = [0, 0, 255]  # 빨간색으로 표시

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()

# 코너 개수 출력
corner_count = np.sum(dst > threshold)
print(f"검출된 코너 개수: {corner_count}")
```

### 2.2. SIFT(Scale-Invariant Feature Transform)
* **원리**: 다양한 스케일에서 DoG(Difference of Gaussian) 피라미드를 구성하고, 극값을 찾아 특징점으로 선택합니다. 각 특징점에 대해 방향과 크기 정보를 포함한 디스크립터(descriptor)를 생성합니다.
* **특징**: 크기, 회전, 조명 변화에 강인하며, 부분적 가림에도 견고합니다.
* **단점**: 계산 비용이 높고, 특허 문제가 있었습니다(현재는 해결됨).

**실습 코드 (SIFT 특징점 검출)**:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('road_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 특징점 검출 및 디스크립터 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 특징점 그리기
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Features (검출된 특징점 개수: {len(keypoints)})')
plt.axis('off')
plt.show()

# 디스크립터 정보 출력
if descriptors is not None:
    print(f"특징점 개수: {len(keypoints)}")
    print(f"디스크립터 형태: {descriptors.shape}")
    print(f"각 디스크립터의 차원: {descriptors.shape[1]}")
```

### 2.3. SURF(Speeded-Up Robust Features)
* **원리**: SIFT와 유사하지만, 가우시안 2차 미분 근사를 위해 Box 필터를 사용하여 속도를 개선했습니다.
* **특징**: SIFT보다 빠르면서도 비슷한 수준의 강인함을 제공합니다.
* **단점**: 여전히 계산 비용이 높고, 특허 문제가 있었습니다(현재는 SIFT와 마찬가지로 해결됨). OpenCV에서 SURF를 사용하려면 `xfeatures2d` 모듈을 설치해야 합니다.

**실습 코드 (SURF 특징점 검출)**:
(참고: SURF는 `opencv-contrib-python` 패키지에 포함되어 있으므로, 설치가 필요할 수 있습니다: `pip install opencv-contrib-python`)

```python
# SIFT 코드와 동일하게 img, gray 변수 사용 가정

try:
    # SURF 객체 생성 (xfeatures2d 모듈 필요)
    surf = cv2.xfeatures2d.SURF_create()

    # 특징점 검출 및 디스크립터 계산
    keypoints_surf, descriptors_surf = surf.detectAndCompute(gray, None)

    # 특징점 그리기
    img_surf = cv2.drawKeypoints(img, keypoints_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 결과 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB))
    plt.title(f'SURF Features (검출된 특징점 개수: {len(keypoints_surf)})')
    plt.axis('off')
    plt.show()

    if descriptors_surf is not None:
        print(f"SURF 특징점 개수: {len(keypoints_surf)}")
        print(f"SURF 디스크립터 형태: {descriptors_surf.shape}")

except AttributeError:
    print("SURF를 사용하려면 `opencv-contrib-python`을 설치하고, `cv2.xfeatures2d` 모듈을 불러와야 합니다.")
    print("pip install opencv-contrib-python")
except Exception as e:
    print(f"SURF 실행 중 오류 발생: {e}")

```

### 2.4. ORB(Oriented FAST and Rotated BRIEF)
*   **원리**: FAST 특징점 검출기와 BRIEF 디스크립터의 아이디어를 결합하여, 회전 불변성과 스케일 불변성을 추가하고 속도를 극대화했습니다.
*   **특징**: SIFT/SURF보다 훨씬 빠르면서도 괜찮은 성능을 보여 특허 문제가 없고 실시간 응용에 적합합니다.
*   **장점**: 실시간 처리 속도와 오픈소스 라이선스로 인해 임베디드 시스템이나 모바일 환경에 적합합니다.
*   **단점**: SIFT/SURF보다 크기 및 시점 변화에 대한 강인함이 다소 떨어질 수 있습니다.

**실습 코드 (ORB 특징점 검출)**:
```python
# 이미지 불러오기 (img, gray 변수 사용 가정)

# ORB 객체 생성 (최대 특징점 개수, 스케일 팩터 등 설정 가능)
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

# 특징점 검출 및 디스크립터 계산
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

# 특징점 그리기
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title(f'ORB Features (검출된 특징점 개수: {len(keypoints_orb)})')
plt.axis('off')
plt.show()

if descriptors_orb is not None:
    print(f"ORB 특징점 개수: {len(keypoints_orb)}")
    print(f"ORB 디스크립터 형태: {descriptors_orb.shape}")
```

---

## **3. 특징점 매칭(Feature Matching) 및 추적(Tracking)**

### 3.1. 특징점 매칭의 원리
*   **목표**: 두 이미지(혹은 연속된 프레임)에서 동일한 객체에 해당하는 특징점을 서로 찾아 연결하는 과정입니다.
*   **방법**: 한 이미지의 특징점 디스크립터와 다른 이미지의 특징점 디스크립터 간의 유사도를 측정합니다. 유클리드 거리(Euclidean Distance)나 해밍 거리(Hamming Distance) 등이 사용됩니다.
*   **매칭 종류**:
    *   **Brute-Force Matcher (BFMatcher)**: 한 특징점의 디스크립터를 다른 이미지의 모든 특징점 디스크립터와 비교하여 가장 유사한 것을 찾습니다.
    *   **FLANN Matcher**: 대규모 데이터셋에 효율적인 근접 이웃 검색 알고리즘을 사용합니다.

**실습 코드 (두 이미지 간 ORB 특징점 매칭)**:
```python
# 매칭을 위한 두 번째 이미지 로드 (첫 번째 이미지와 유사하지만 약간 다른 시점/객체)
img1 = cv2.imread('road_image.jpg', cv2.IMREAD_GRAYSCALE) # 원본 이미지 (gray1)
img2 = cv2.imread('road_image_shifted.jpg', cv2.IMREAD_GRAYSCALE) # 약간 이동된 이미지 (gray2)

if img1 is None or img2 is None:
    print("두 번째 이미지를 찾을 수 없습니다. 'road_image_shifted.jpg' 파일을 확인하거나 생성하세요.")
    exit()

# ORB 특징점 검출기 생성
orb = cv2.ORB_create(nfeatures=500)

# 두 이미지에서 특징점 및 디스크립터 검출
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BFMatcher 생성 (ORB는 바이너리 디스크립터이므로 NORM_HAMMING 사용)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # crossCheck=True는 양방향 매칭으로 오매칭 줄임

# 특징점 매칭
matches = bf.match(des1, des2)

# 매칭 결과 거리에 따라 정렬 (거리가 짧을수록 유사함)
matches = sorted(matches, key=lambda x: x.distance)

# 상위 N개의 매칭 결과만 시각화
num_matches_to_draw = min(50, len(matches))
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_matches_to_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 시각화
plt.figure(figsize=(15, 10))
plt.imshow(img_matches), plt.title(f'ORB Feature Matching ({num_matches_to_draw} 매치)')
plt.axis('off')
plt.show()

print(f"총 매치된 특징점 개수: {len(matches)}")
print(f"상위 {num_matches_to_draw}개 매치 그리기.")
```
**참고**: `road_image_shifted.jpg` 파일은 `road_image.jpg`를 약간 이동시키거나 회전시킨 이미지여야 매칭 결과를 볼 수 있습니다.

### 3.2. 특징점 추적(Feature Tracking)
*   **목표**: 비디오 시퀀스(연속된 이미지 프레임)에서 동일한 특징점이 프레임 간 어떻게 이동하는지 추적합니다.
*   **방법**:
    *   **옵티컬 플로우(Optical Flow)**: 이미지 시퀀스에서 객체나 패턴의 움직임을 추정하는 방법입니다. Lucas-Kanade 옵티컬 플로우가 널리 사용됩니다.
    *   **디스크립터 매칭 기반 추적**: 각 프레임에서 특징점을 검출하고 디스크립터를 매칭하는 과정을 반복합니다.

**실습 코드 (Lucas-Kanade 옵티컬 플로우를 이용한 특징점 추적)**:
(동영상 파일이 필요합니다. 예를 들어 `video.mp4` 파일을 사용하거나 실시간 카메라를 사용할 수 있습니다.)

```python
# 추적을 위한 이전 프레임과 현재 프레임에서 Good Features to Track 사용
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# 옵티컬 플로우를 위한 파라미터
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 임의의 색상 (추적 선 그리기 용도)
color = np.random.randint(0,255,(100,3))

# 비디오 캡처 객체 생성 (0번 카메라 또는 비디오 파일)
cap = cv2.VideoCapture('video.mp4') # 'video.mp4' 파일 경로를 사용하거나 0으로 실시간 카메라 사용

if not cap.isOpened():
    print("오류: 비디오 파일 또는 카메라를 열 수 없습니다.")
    exit()

# 첫 번째 프레임 읽기
ret, old_frame = cap.read()
if not ret:
    print("첫 프레임을 읽을 수 없습니다.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) # 추적할 초기 특징점 찾기

# 추적된 점들을 그릴 마스크 이미지 생성
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 모두 읽었습니다. 또는 더 이상 프레임이 없습니다.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 옵티컬 플로우 계산 (이전 프레임의 점 p0를 현재 프레임에서 추적하여 p1 얻기)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 제대로 추적된 점들만 선택
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

        # 추적 선과 점 그리기
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a),int(b)),5, color[i].tolist(),-1)
        img_track = cv2.add(frame,mask)

        cv2.imshow('Feature Tracking', img_track)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 다음 프레임을 위해 현재 프레임과 점들을 업데이트
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    else:
        cv2.imshow('Feature Tracking', frame) # 추적된 점이 없어도 원본 프레임 표시
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
print("특징점 추적 종료.")
```

---

## **4. 자율주행에서의 적용 사례**

*   **Visual SLAM (Simultaneous Localization And Mapping)**: 차량의 카메라 영상에서 특징점을 추출하고 추적하여 차량의 정확한 위치를 파정하고 동시에 주변 환경의 3D 지도를 구축합니다. 이는 GPS 신호가 불안정한 터널이나 실내 환경에서 특히 중요합니다.
*   **객체 추적**: 도로 위의 다른 차량이나 보행자의 특징점을 추적하여 이들의 움직임을 예측하고 충돌 위험을 회피하는 데 사용됩니다.
*   **움직임 보상**: 차량의 움직임으로 인해 발생하는 영상의 흔들림을 특징점 추적을 통해 보상하여, 안정적인 영상 데이터를 얻고 다음 처리 단계의 정확도를 높입니다.
*   **모노-SLAM (Mono-SLAM)**: 단안 카메라(Mono Camera)만을 사용하여 특징점 검출 및 추적으로 3D 정보를 얻고 차량의 위치를 추정하는 기술입니다.
*   **루프 클로저 (Loop Closure)**: 차량이 이전에 방문했던 장소를 다시 지날 때, 과거와 현재의 이미지에서 공통된 특징점을 매칭하여 이를 감지하고 SLAM 지도의 오차를 수정합니다.

---

## **5. 실습 과제 및 응용 아이디어**

1.  **다른 특징점 검출기 비교**: `road_image.jpg`에 ORB 외에 Shi-Tomasi 코너 검출기 (`cv2.goodFeaturesToTrack()`)를 적용하여 코너를 검출하고 ORB 결과와 비교해 보세요.
2.  **이동/회전된 이미지에서 특징점 매칭**: `road_image.jpg`와 `road_image_rotated.jpg` (약간 회전된 이미지) 두 장을 사용하여 SIFT 특징점 매칭을 수행하고 결과를 시각화해 보세요. 회전에 강인한 SIFT의 특성을 확인합니다.
3.  **실시간 특징점 추적 개선**: `Lucas-Kanade 옵티컬 플로우` 예제에서, 추적에 실패한 특징점(`st==0`)이 많아질 경우, 새로운 특징점을 다시 검출하여 추적을 이어나가도록 코드를 개선해 보세요. (예: `p0`가 일정 개수 이하로 줄어들면 `cv2.goodFeaturesToTrack` 다시 호출)
4.  **객체 추적 시뮬레이션**: `video.mp4`에서 특정 차량의 윈도우 부분을 ROI로 설정하고, 그 ROI 안에서만 특징점을 추적하여 해당 차량이 영상 내에서 어떻게 움직이는지 시각화해 보세요.

---

스카이님, 특징점 검출 및 추적은 자율주행 차량이 주변 환경을 '인지'하고 자신의 '위치'를 파악하며 '움직임'을 추정하는 데 매우 근본적인 기술입니다. 이 강의 자료가 비전공자 학생들이 이 기술의 중요성을 이해하고 실제 코드에 적용해보는 데 도움이 되기를 바랍니다. 특히 옵티컬 플로우 실습은 라이브 영상에서 학생들이 AI의 '추적' 능력을 실감할 수 있는 좋은 기회가 될 것입니다!