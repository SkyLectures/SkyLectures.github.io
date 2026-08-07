---
layout: page
title:  "카메라를 활용한 자율주행 자동차 구현(OpenCV)"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-02-03_03-AutoDrivingUsingCameraOpenCv
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 카메라 관련 패키지 설치

- **rpicam-apps**
    - 라즈베리파이 5를 포함한 최신 라즈베리파이 모델에서 카메라를 쉽게 사용하고 테스트하기 위해 제공되는 명령줄 애플리케이션들의 묶음(패키지)
    - 터미널에서 입력하여 카메라를 동작시킬 수 있는 여러 유용한 도구들이 포함되어 있음

    - 주요 특징 및 역할
        - libcamera 프레임워크 기반
            - 라즈베리파이의 최신 카메라 시스템인 libcamera 프레임워크 위에 구축됨
            - libcamera: 카메라 하드웨어를 직접 제어하는 저수준 기술
            - rpicam-apps:  libcamera를 쉽게 사용할 수 있도록 해주는 '사용자 친화적인 인터페이스' 역할
        - 명령줄 카메라 도구 제공
            - 직접 파이썬 코드를 작성하거나 복잡한 API를 건드리지 않고도 카메라의 기본적인 기능들을 명령줄로 바로 실행할 수 있게 해줌
    - 주요 포함 명령어
        - rpicam-hello
            - 라즈베리파이 카메라가 제대로 작동하는지 테스트하기 위한 명령어
            - 약 5초간 카메라 프리뷰 화면을 띄워주는 카메라 테스트 앱
            -  카메라 모듈의 연결 상태와 libcamera 스택의 동작 여부를 간편하게 확인할 수 있음
        - rpicam-still: 사진을 촬영하고 파일로 저장
        - rpicam-vid: 동영상을 녹화하고 파일로 저장
        - rpicam-raw: RAW 이미지 데이터를 캡처하는 등 보다 전문적인 기능 제공
    - 진단 및 테스트
        - 카메라 모듈 연결이 올바른지, libcamera 드라이버가 정상적으로 로드되었는지 등을 가장 빠르고 확실하게 테스트하는 데 사용
    - 초기 설정 필요 없음
        - 이전 라즈베리파이처럼 raspi-config에서 별도로 카메라를 '활성화'할 필요 없이,
        - rpicam-apps만 설치하면 libcamera 시스템과 함께 바로 카메라 기능을 활용할 수 있음

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y rpicam-apps

sudo reboot

rpicam-hello
```

## 2. 카메라로 영상 확인하기

### 2.1 OpenCV 설치

- OpenCV를 사용하기 위해 필요한 시스템 라이브러리 설치

```bash
#// file: "Terminal"
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
    build-essential cmake pkg-config python3-dev \
    libatlas3-base libhdf5-dev libcap-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libgtk-3-0 libcanberra-gtk3-module \
    libcamera-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libexif-dev libdrm-dev libgpiod-dev
```

- 실습을 위한 가상환경 구성
    - 기존 예제는 라즈베리파이5가 기본적으로 제공하는 기능 및 패키지를 사용하고 있으므로 비교적 안전함
    - OpenCV 등 시스템에 대한 영향력이 큰 파이썬 패키지는 기본 설정을 망가뜨릴 수 있으므로 가상환경의 구성을 권장함
        - 가상환경 없이 설치하면 라즈베리파이5 시스템에서 오류를 발생시키므로 권장보다는 거의 강제의 수준임
    - 가상환경 생성 시 **<span style="color: red;">python -m venv --system-site-packages [가상환경명]</span>**으로 설치해야 함
        - 시스템에 설치된 libcamera, OpenCV 및 관련 라이브러리, 가상환경 안에 설치하는 picamera2, numpy, opencv-python 등이 버전 충돌, 경로 충돌 및 다양한 인식 오류로 인해 어떻게 수정하더라도 작동하지 않음
        - 시스템에 기본 설치된 환경과 libcamera, OpenCV 및 관련 라이브러리의 내용을 가상환경에서도 그대로 끌어와서 사용해야만 제대로 작동함

```bash
#// file: "Terminal"
python -m venv --system-site-packages aicar
cd aicar
source ./bin/activate

pip install opencv-python
```

### 2.2 Picamera2

- 라즈베리파이의 최신 카메라 시스템인 libcamera 프레임워크를 위한 공식 Python 라이브러리
- 파이썬을 통해 라즈베리파이 카메라의 다양한 기능을 제어하고 활용할 수 있도록 설계됨
- 64비트 Raspberry Pi OS Bookworm과 함께 Picamera2를 사용하기에 가장 적합함
    - 최적의 지원대상: 라즈베리파이 4, 라즈베리파이 5, 라즈베리파이 Zero 2 W 등
    - 호환성 지원대상: 라즈베리파이 3 모델 B/B+, 라즈베리파이 2 모델 B
        - 가까스로 사용은 할 수 있지만 성능이 떨어짐
- 라즈베리파이 5는 설계 단계부터 libcamera를 염두에 두고 만들어졌기 때문에 최적의 호환성과 성능을 제공함

```python
#// file: "mycamera.py"
from picamera2 import Picamera2
import numpy as np

class MyPiCamera():

    def __init__(self,width,height):
        self.cap = Picamera2()

        self.width = width;
        self.height = height
        self.is_open = True

        try:
            self.config = self.cap.create_video_configuration(main={"format":"RGB888","size":(width,height)})
            self.cap.align_configuration(self.config)
            self.cap.configure(self.config)

            self.cap.start()
        except:
            self.is_open = False
        return
    
    def read(self,dst=None):
        if dst is  None:
            dst = np.empty((self.height, self.width, 3), dtype=np.uint8)

        if self.is_open:
            dst = self.cap.capture_array()
    
        return self.is_open,dst

    def isOpened(self):
        return self.is_open

    def release(self): 
        if self.is_open is True:
             self.cap.close()
        self.is_open = False
        return

# After the test is finished, the main block is no longer needed (can be deleted)
if __name__ == "__main__":
    import cv2
    camera = MyPiCamera(640,480)

    while camera.isOpened():
        _, image = camera.read()
        cv2.imshow("mycamera", image)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
```

- mycamera.py 파일에서 메인 블록은 제거하고 import 용으로 사용

```python
#// file: "camera.py"
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
        
    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('camera test',image)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## 3. OpenCV 라인트레이서 자동차 만들기

### 3.1 카메라 뷰 확인

- **프로세스**
    - 160x120 픽셀 크기로 설정
        - 너무 크면 처리해야 할 데이터가 커지므로 가능한 작은 크기로 설정

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3.2 관심영역(ROI) 설정

- **프로세스**
    - 프레임의 세로(높이) 크기를 60~120픽셀까지 자르고
    - 프레임의 가로(폭) 크기는 전체(0~160)를 대상으로 표시
    - 자른 데이터는 crop_img 변수에 저장

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        print(height,width)
        crop_img = image[height //2 :, :]
        cv2.imshow('crop',crop_img)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3.3 그레이스케일 + 블러 처리

- **프로세스**
    - 이미지를 그레이스케일로 변경
    - 가우시안 블러링: 중심에 있는 픽셀에 높은 가중치 부여 ➜ 설정된 크기(5x5)에 맞추어 시그마를 계산하여 블러링
    - 자르고(Crop) 변조하고(GrayScale) 블러링(Blur)한 이미지를 보여줌

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        crop_img = image[height //2 :, :]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),0)
        
        cv2.imshow('crop+gray+blur',blur)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3.4 임계값 설정으로 2진화(윤곽선 추출)

- **프로세스**
    - 임계값이 130보다 큰 값 ➜ 255로 변환
    - 임계값이 130보다 작은 값 ➜ 0으로 변환
    - THRESH_BINARY_INV: 흑백으로 표현하기 위한 값

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        crop_img = image[height //2 :, :]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),0)
        
        ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
        
        cv2.imshow('thresh1',thresh1)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

### 3.5 윤곽선 기준 인식범위 설정

- **프로세스**
    - 이미지 압축+팽창으로 노이즈 제거
    - 이미지 윤곽선 검출
    - 검출된 윤곽선이 있다면 ➜ 윤곽선의 최대값 반환
    - 윤곽선에서 모멘트 계산 ➜ x축(가로)과 y축(세로)의 무게중심 확인
    - x축의 무게중심 출력
        - 선이 중심보다 왼쪽: 95~125 ➜ 자동차가 오른쪽으로 이동 중이므로 왼쪽으로 변경
        - 선이 중심보다 오른쪽: 39~65 ➜ 자동차가 왼쪽으로 이동 중이므로 오른쪽으로 변경

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        crop_img = image[height //2 :, :]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),0)
        
        ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
        
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask',mask)
    
        contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        
            #cv2.imshow('crop',crop_img)
            print(cx)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3.6 좌우 방향 결정

- **프로세스**
    - x축의 무게중심이 기준
        - 선이 중심보다 왼쪽: 30~130 ➜ 자동차가 오른쪽으로 이동 중이므로 왼쪽으로 변경
        - 선이 중심보다 오른쪽: 150~250 ➜ 자동차가 왼쪽으로 이동 중이므로 오른쪽으로 변경
    - 각 장비마다 값이 다를 수 있으므로 적절하게 조정할 것

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        crop_img = image[height //2 :, :]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),0)
        
        ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
        
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask',mask)
    
        contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            if cx >= 150 and cx <= 250:
                print(f"{cx}: Turn Right")
            elif cx >= 30 and cx <= 130:
                print(f"{cx}: Turn Left!")
            elif cx > 130 and cx < 150:
                print(f"{cx}: go")
            else:
                print(f"{cx}: stop")
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3.7 모터 제어

- **프로세스**
    - 선택된 명령에 따라 실제 모터 제어 함수 호출

```python
#// file: "line_tracer.py"
import mycamera
import cv2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)
    
def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

def motor_back(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed
    
def motor_left(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

def main():
    camera = mycamera.MyPiCamera(320,240)

    while( camera.isOpened() ):
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('normal',image)
        
        height, width, _ = image.shape
        crop_img = image[height //2 :, :]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),0)
        
        ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
        
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask',mask)
    
        contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            if cx >= 150 and cx <= 250:
                print(f"{cx}: Turn Right")
                motor_right(0.3)
            elif cx >= 30 and cx <= 130:
                print(f"{cx}: Turn Left!")
                motor_left(0.3)
            elif cx > 130 and cx < 150:
                print(f"{cx}: go")
                motor_go(0.3)
            else:
                print(f"{cx}: stop")
                motor_stop()
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    PWMA.value = 0.0
    PWMB.value = 0.0
```
