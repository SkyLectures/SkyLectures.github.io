---
layout: page
title:  "라즈베리파이-카메라 실시간 영상 처리"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-03-06_01-RealtimeImageProcessing
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 기본 프로세스

- Raspberry Pi와 카메라 모듈을 연결하여 실시간으로 영상을 캡처하고 처리하기

1. **비디오 캡처 객체 생성 (`cv2.VideoCapture()`)**
    - 비디오 스트림 열기
        - 카메라 인덱스 (0, 1 등)나 비디오 파일 경로를 인자로 전달
        - Raspberry Pi 카메라 모듈의 경우 `0` 또는 `-1`을 사용하거나, `gst-launch` 같은 파이프라인을 사용할 수 있음

2. **프레임 읽기 (`read()`)**
    - 비디오 스트림에서 한 프레임(이미지)씩 읽어옴

3. **무한 루프와 종료 조건**
    - 비디오 처리는 일반적으로 무한 루프 내에서 각 프레임을 처리
    - 특정 키(예: 'q' 키) 입력 시 루프를 종료하도록 구현함

```python
# 카메라 객체 생성 (0번 카메라, 라즈베리파이 카메라 모듈이 연결되어 있다고 가정)
cap = cv2.VideoCapture(0)

# 카메라가 제대로 열렸는지 확인
if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
else:
    print("카메라 연결 성공!")
    while True:
        ret, frame = cap.read() # 프레임 읽기 (ret: 성공 여부, frame: 이미지 데이터)

        if not ret: # 프레임을 제대로 읽지 못하면 종료
            print("프레임을 받지 못했습니다. 종료합니다.")
            break

        # 예시: 캡처된 프레임을 그레이스케일로 변환하여 표시
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Live Camera Feed (Original)', frame)
        cv2.imshow('Live Camera Feed (Grayscale)', gray_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용이 끝난 카메라 객체 해제
    cap.release()
    cv2.destroyAllWindows()
    print("비디오 스트리밍 종료.")
```



## 2. 미니프로젝트

> - **프로젝트 개요**
> 
>   - **목표**
>       - 라즈베리파이와 카메라 모듈을 활용하여 실시간 영상 수집
>       - OpenCV를 통해 다양한 영상 처리 기법 적용
>
>   - **학습 효과**
>       - 라즈베리파이 카메라 모듈의 기본 작동 원리 이해
>       - OpenCV를 활용한 실시간 영상 처리 기법 습득
>       - 자율주행 시스템의 '인지' 단계에 해당하는 기술 체험
>       - 실제 영상 데이터를 활용한 프로그래밍 경험
>
>   - **필요 장비**
>       - 라즈베리파이 B+ 또는 이상 모델
>       - 라즈베리파이 카메라 모듈 (또는 USB 웹캠)
>       - SD 카드 (16GB 이상 권장)
>       - 디스플레이 (HDMI 연결)
>       - 키보드, 마우스
{: .common-quote}

### 2.1 환경 설정

- **라즈베리파이 OS 설치 및 기본 설정**

```bash
#// file: "Terminal: bash"
# 시스템 업데이트
sudo apt update
sudo apt upgrade -y

# 필요 패키지 설치
sudo apt install -y python3-pip python3-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev 
sudo apt install -y libqt5gui5 libqt5webkit5 libqt5test5

# 파이썬 패키지 설치
pip3 install numpy opencv-python picamera
```

- **카메라 모듈 활성화**

```bash
#// file: "Terminal: bash"
# 라즈베리파이 설정 도구 실행
sudo raspi-config
```
- '인터페이스 옵션(Interface Options)' 선택
- '카메라(Camera)' 선택
- '예(Yes)' 선택하여 카메라 활성화
- 라즈베리파이 재부팅

### 2.2 기본 카메라 테스트 코드

- 다음 코드를 `camera_test.py` 파일로 저장하고 실행하여 카메라가 정상적으로 작동하는지 확인

```python
#// file: "camera_test.py"
import cv2
import time

def test_camera():
    # 카메라 객체 생성 (0은 기본 카메라)
    cap = cv2.VideoCapture(0)
    
    # 카메라가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return
    
    print("카메라 테스트를 시작합니다. 종료하려면 'q'를 누르세요.")
    
    # 프레임 카운터와 시작 시간
    frame_count = 0
    start_time = time.time()
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 받아오지 못했습니다.")
            break
            
        # 프레임 카운터 증가
        frame_count += 1
        
        # 현재 FPS 계산 (1초마다 업데이트)
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # 화면에 표시
        cv2.imshow('Camera Test', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("카메라 테스트를 종료합니다.")

if __name__ == "__main__":
    test_camera()
```

- 실행 방법

```bash
#// file: "Terminal: bash"
python3 camera_test.py
```


### 2.3 기본 프레임워크

```python
#// file: "main.py"
import cv2
import numpy as np
import time
import argparse

class RealTimeVideoProcessor:
    def __init__(self, camera_index=0, resolution=(640, 480), display_scale=1.0):
        """
        실시간 영상 처리 클래스 초기화
        
        Args:
            camera_index (int): 카메라 인덱스 (기본값: 0)
            resolution (tuple): 카메라 해상도 (기본값: 640x480)
            display_scale (float): 화면 표시 크기 배율 (기본값: 1.0)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.display_scale = display_scale
        self.cap = None
        self.processing_mode = "original"  # 기본 모드
        self.available_modes = {
            "original": self.process_original,
            "grayscale": self.process_grayscale,
            "blur": self.process_blur,
            "canny": self.process_canny,
            "threshold": self.process_threshold,
            "hsv_filter": self.process_hsv_filter
            # 여기에 새로운 모드 함수를 추가할 수 있습니다.
        }
        
        # HSV 필터링을 위한 기본 값 (초록색 범위)
        self.hsv_lower = np.array([35, 100, 100])
        self.hsv_upper = np.array([85, 255, 255])
        
    def open_camera(self):
        """카메라 열기"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError("카메라를 열 수 없습니다.")
        
        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # 일부 카메라에서 FPS 설정 필요할 수 있음
        # self.cap.set(cv2.CAP_PROP_FPS, 30) 
        
        return self.cap.isOpened()
    
    def close_camera(self):
        """카메라 자원 해제"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def set_mode(self, mode_name):
        """처리 모드 설정"""
        if mode_name in self.available_modes:
            self.processing_mode = mode_name
            print(f"처리 모드를 '{mode_name}'(으)로 변경했습니다.")
        else:
            print(f"오류: 지원하지 않는 모드 '{mode_name}' 입니다. 지원 모드: {list(self.available_modes.keys())}")
            
    def process_original(self, frame):
        """원본 영상 반환"""
        return frame
    
    def process_grayscale(self, frame):
        """그레이스케일 변환"""
        # 그레이스케일로 변환하면 1채널이 되므로, 표시를 위해 다시 3채널로 변환
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    def process_blur(self, frame):
        """가우시안 블러 적용"""
        return cv2.GaussianBlur(frame, (9, 9), 0)
    
    def process_canny(self, frame):
        """Canny 엣지 검출 적용"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # 엣지 이미지는 1채널이므로, 컬러 이미지처럼 표시하기 위해 3채널로 변환
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def process_threshold(self, frame):
        """이진화 (Thresholding) 적용"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 127을 기준으로 이진화
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
    def process_hsv_filter(self, frame):
        """HSV 색상 필터링 적용"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result

    def run(self):
        """메인 루프 실행"""
        try:
            if not self.open_camera():
                return
            
            print("--------------------------------------------------")
            print("실시간 영상 처리 시작! (종료: 'q' 또는 Esc 키)")
            print(f"현재 처리 모드: '{self.processing_mode}'")
            print("모드 변경 키: 'o'(original), 'g'(grayscale), 'b'(blur), 'c'(canny), 't'(threshold), 'h'(hsv_filter)")
            print("--------------------------------------------------")
            
            fps_start_time = time.time()
            fps_frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 받아오지 못했습니다. 종료합니다.")
                    break
                
                # 프레임 처리
                processed_frame = self.available_modes[self.processing_mode](frame.copy())
                
                # 화면 표시를 위해 크기 조정
                if self.display_scale != 1.0:
                    display_width = int(self.resolution[0] * self.display_scale)
                    display_height = int(self.resolution[1] * self.display_scale)
                    processed_frame = cv2.resize(processed_frame, (display_width, display_height))
                
                # 현재 처리 모드 텍스트 추가
                cv2.putText(processed_frame, f"Mode: {self.processing_mode}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # FPS 계산 및 표시
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    print(f"FPS: {current_fps:.2f}, Mode: {self.processing_mode}")
                
                cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Real-time Video Processing', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 또는 Esc 키
                    break
                elif key == ord('o'):
                    self.set_mode("original")
                elif key == ord('g'):
                    self.set_mode("grayscale")
                elif key == ord('b'):
                    self.set_mode("blur")
                elif key == ord('c'):
                    self.set_mode("canny")
                elif key == ord('t'):
                    self.set_mode("threshold")
                elif key == ord('h'):
                    self.set_mode("hsv_filter")

        except ValueError as e:
            print(f"오류 발생: {e}")
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
        finally:
            self.close_camera()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="라즈베리파이 실시간 영상 처리 미니프로젝트")
    parser.add_argument("--camera", type=int, default=0, help="사용할 카메라 인덱스 (기본값: 0)")
    parser.add_argument("--width", type=int, default=640, help="카메라 해상도 너비 (기본값: 640)")
    parser.add_argument("--height", type=int, default=480, help="카메라 해상도 높이 (기본값: 480)")
    parser.add_argument("--scale", type=float, default=1.0, help="화면 표시 스케일 (기본값: 1.0)")
    args = parser.parse_args()

    processor = RealTimeVideoProcessor(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        display_scale=args.scale
    )
    processor.run()
```

### 2.4 실행 방법

1.  위 코드를 `main.py` 파일로 저장합니다.
2.  라즈베리파이 터미널에서 다음 명령어를 실행합니다.

```bash
#// file: "Terminal: bash"
python3 main.py
```

3.  선택적으로 해상도나 카메라 인덱스 등을 지정할 수 있습니다.

```bash
#// file: "Terminal: bash"
# USB 웹캠을 사용하는 경우 (인덱스가 1번일 수 있음)
python3 main.py --camera 1

# 해상도를 320x240으로 낮춰서 실행
python3 main.py --width 320 --height 240 --scale 1.5
```

### 2.5 프로젝트 제어 (실행 중)

- **`o`**: 원본 영상 보기
- **`g`**: 그레이스케일 영상 보기
- **`b`**: 가우시안 블러 적용 영상 보기
- **`c`**: Canny 엣지 검출 적용 영상 보기
- **`t`**: 이진화 (Thresholding) 적용 영상 보기
- **`h`**: HSV 색상 필터링 적용 영상 보기 (기본값: 초록색 범위)
- **`q` 또는 `Esc`**: 프로그램 종료


## 3. 확장 아이디어

### 3.1 ROI (관심 영역) 필터링 추가

- `RealTimeVideoProcessor` 클래스에 `process_roi_filter` 함수 추가
    - 이 함수에서는
        - 프레임에서 도로 영역(아래쪽 삼각형)을 ROI로 설정하고,
        - 해당 ROI에만 다른 필터(예: Canny)를 적용하여 표시
    - 예시 코드를 참고하여 자신의 키트 환경에 맞는 ROI 좌표를 찾아보기

```python
#// file: "main.py"

# main.py에 process_roi_filter 함수 추가 예시
# class RealTimeVideoProcessor 안에
# self.available_modes 에 "roi_canny": self.process_roi_canny 추가

def process_roi_canny(self, frame):
    """ROI 설정 후 Canny 엣지 검출 적용"""
    height, width = frame.shape[:2]
    
    # 예시 ROI (이미지 하단 사다리꼴 영역)
    # 이미지에 따라 이 좌표를 조절해야 함
    roi_vertices = np.array([
        [(0, height), (width / 2 - 50, height / 2 + 50), 
        (width / 2 + 50, height / 2 + 50), (width, height)]
    ], dtype=np.int32)
    
    # ROI 마스크 생성
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    
    # 원본 이미지에 마스크 적용
    masked_frame = cv2.bitwise_and(frame, mask)
    
    # 마스크된 영역에 Canny 엣지 검출 적용
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 엣지 이미지는 1채널이므로, 컬러 이미지처럼 표시하기 위해 3채널로 변환
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
```

- `run` 함수에 `'r'` 키 입력 시 `self.set_mode("roi_canny")` 호출 추가


### 3.2 사용자 정의 HSV 범위 조절 기능 추가

- `hsv_filter` 모드에서 키보드 입력을 통해 `self.hsv_lower`와 `self.hsv_upper` 값을 실시간으로 조절
    - 예: `+`, `-` 키로 각 채널 값 조절
- 이 기능을 통해 원하는 색상을 정확히 필터링하는 과정을 확인 가능

### 3.3 동영상 저장 기능 추가

- `'s'` 키를 누르면 **→** 현재 처리되고 있는 영상을 `.mp4` 또는 `.avi` 파일로 저장하는 기능 추가
- `cv2.VideoWriter` 객체를 사용하여 구현 가능

### 3.4 실시간 객체 탐지 (경량화 모델)

- 라즈베리파이에서 동작 가능한 경량 딥러닝 모델(예: MobileNet SSD, YOLO-tiny)을 사용하여 간단한 객체(예: 사람, 자동차)를 탐지하고 경계 상자를 그리는 기능 추가
- 이 부분은 설치 및 구현 난이도가 높을 수 있음
