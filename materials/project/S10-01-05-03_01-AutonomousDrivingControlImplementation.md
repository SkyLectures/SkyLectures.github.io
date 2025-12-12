---
layout: page
title:  "라즈베리파이 기반 자율주행자동차 구현(1)"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-05-03_01-AutonomousDrivingControlImplementation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 데이터 획득

### 1.1 OpenCV를 활용한 키보드 입력

- 키보드 입력값 확인하기

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):        
        keyValue = cv2.waitKey(10)
        print(str(keyValue))
        
        if keyValue == ord('q') :
            break
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- 전후좌우 키보드 입력 확인

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):        
        keyValue = cv2.waitKey(10)
        #print(str(keyValue))
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("up")
        elif keyValue == 84:
            print("down")
        elif keyValue == 81:
            print("left")
        elif keyValue == 83:
            print("right")
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

### 1.2 모델 학습을 위한 이미지 보정

- 이미지 보정의 이유
    - 과한 조명 등을 보정하여 학습이 용이하게 만듦
    - 사진의 크기를 줄여 속도를 빠르게 하기 위함

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):        
        keyValue = cv2.waitKey(10)
        #print(str(keyValue))
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("up")
        elif keyValue == 84:
            print("down")
        elif keyValue == 81:
            print("left")
        elif keyValue == 83:
            print("right")
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        cv2.imshow('Save', save_image)
        
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- YUV 형식으로 변환
    - 인공지능이 학습하기 좋은 형식
        - RGB 대신 YUV를 사용하는 것은 데이터 압축 효율성과 인지적 특성을 활용한 처리 방식에서 오는 이점 때문
    - YUV 색상 공간
        - Y (Luminance): 밝기(휘도) 정보를 나타냄. 이미지의 흑백 부분을 담당
        - U (Chrominance-Blue): 색상 차이(색차) 정보를 나타냄. 파란색 계열의 색상 차이를 표현
        - V (Chrominance-Red): 색상 차이(색차) 정보를 나타냄. 빨간색 계열의 색상 차이를 표현
    - AI 학습에서 YUV가 유리한 이유
        - 데이터 압축 및 효율성 (가장 큰 이유)
            - 밝기/색상 분리
                - 사람의 눈은 색상(U, V)보다 밝기(Y)에 더 민감함
                - 이 특성을 활용하여 이미지 압축 시 색상 정보(U, V)의 해상도를 밝기 정보(Y)보다 낮게 샘플링하여 저장해도 (예: 4:2:2, 4:2:0 서브샘플링) 사람이 인지하기에 큰 화질 저하를 느끼지 못함
            - 학습 데이터 크기 감소
                - U, V 채널의 해상도를 낮추면 전체 데이터 크기가 줄어듦
                - 딥러닝 모델이 학습해야 할 파라미터 수가 감소하고, 학습 속도가 빨라질 수 있음
                - 이는 제한된 하드웨어 자원이나 대규모 데이터셋 학습 시 유용함
            - 특징 학습 효율화
                - 밝기 정보(Y)는 객체의 윤곽, 질감 등 공간적 특징을 파악하는 데 중요함
                - 색상 정보(U, V)는 색상 자체를 구별하는 데 중요함
                - 이들이 분리되어 있으면, 모델이 각각의 특징을 더 명확하게 학습할 수 있도록 초기 단계에서 신호 분리 효과를 얻을 수도 있음
        - 노이즈 감소 및 강건성(Robustness)
            - RGB 색상 공간에서는 노이즈가 발생했을 때 R, G, B 채널 전반에 걸쳐 영향을 미침
            - YUV에서는 밝기(Y) 채널과 색상(U, V) 채널이 분리되어 있기 때문에, 색상 노이즈가 밝기 정보에 직접적인 영향을 덜 미침
            - 이는 모델이 노이즈에 대해 좀 더 강건하게(robust) 특징을 학습하는 데 도움을 줄 수 있음
        - 특정 작업에 유리
            - 엣지 검출/형태 인식
                - 객체의 형태나 윤곽선은 주로 밝기(Y) 채널에 강하게 나타남
                - Y 채널만을 사용하면 색상에 덜 영향을 받는 형태 기반 특징을 더 쉽게 추출할 수 있음
                - 이는 자율주행에서의 차선, 건물 윤곽, 사람 형태 인식 등 많은 비전 작업에서 중요할 수 있음
            - 흑백-컬러화 모델(Colorization)
                - 흑백 이미지를 컬러로 바꾸는 ColorNet과 같은 모델은 흑백 이미지(밝기 정보)를 입력으로 받아 색상 정보(U, V)를 예측하도록 학습
                - YUV는 밝기와 색상을 분리하는 특성 덕분에, 흑백 이미지 처리나 영상 처리에서 특정 채널에 집중하는 모델 설계에 유리함
    - 그러나 반드시 유리한 것은 아님

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):        
        keyValue = cv2.waitKey(10)
        #print(str(keyValue))
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("up")
        elif keyValue == 84:
            print("down")
        elif keyValue == 81:
            print("left")
        elif keyValue == 83:
            print("right")
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        cv2.imshow('Save', save_image)
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- 블러링 처리 + 리사이징
    - nvidia 배포 모델의 입력사진 크기인 200 x 66 픽셀로 변경

```python
import mycamera
import cv2

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):
        keyValue = cv2.waitKey(10)
        #print(str(keyValue))
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("up")
        elif keyValue == 84:
            print("down")
        elif keyValue == 81:
            print("left")
        elif keyValue == 83:
            print("right")
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (3,3), 0)
        save_image = cv2.resize(save_image, (200,66))
        cv2.imshow('Save', save_image)
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

### 1.3 OpenCV 이미지 저장하기

- 현재 폴더의 아래에 video 폴더를 생성하여 OpenCV 이미지 저장

```python
import mycamera
import cv2
import time

def main():
    camera = mycamera.MyPiCamera(640,480)
  
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        cv2.imwrite("./video/test.png" , image)
        
        time.sleep(1.0)
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- 매 이미지를 새로운 이름으로 저장

```python
import mycamera
import cv2
import time

def main():
    camera = mycamera.MyPiCamera(640,480)
    filepath = "./video/test"
    i = 0
    
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        cv2.imwrite("%s_%05d.png" % (filepath, i), image)
        i = i + 1
        
        time.sleep(1.0)
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

### 1.4 조종기능을 통해 실제 데이터 학습하기

- 키보드의 입력에 따라 조건문 실행

```python
import mycamera
import cv2
import time

def main():
    camera = mycamera.MyPiCamera(640,480)
    filepath = "./video/train"
    i = 0
    carState = "stop"
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
        elif keyValue == 84:
            print("stop")
            carState = "stop"
        elif keyValue == 81:
            print("left")
            carState = "left"
        elif keyValue == 83:
            print("right")
            carState = "right"
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (3,3), 0)
        save_image = cv2.resize(save_image, (200,66))
        cv2.imshow('Save', save_image)
        
        if carState == "left":
            print("L")
        elif carState == "right":
            print("R")
        elif carState == "go":
            print("G")
        
            
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- 조건문에 사진을 저장하는 코드를 삽입하여 데이터 저장

```python
import mycamera
import cv2
import time

def main():
    camera = mycamera.MyPiCamera(640,480)
    filepath = "./video/train"
    i = 0
    carState = "stop"
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
        elif keyValue == 84:
            print("stop")
            carState = "stop"
        elif keyValue == 81:
            print("left")
            carState = "left"
        elif keyValue == 83:
            print("right")
            carState = "right"
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (3,3), 0)
        save_image = cv2.resize(save_image, (200,66))
        cv2.imshow('Save', save_image)
        
        if carState == "left":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 45), save_image)
            i += 1
        elif carState == "right":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 135), save_image)
            i += 1
        elif carState == "go":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 90), save_image)
            i += 1
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```

- 자동차를 움직이는 기능을 추가하여 이동 중의 이미지 저장

```python
import mycamera
import cv2
import time
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

speedSet = 0.5

def main():
    camera = mycamera.MyPiCamera(640,480)
    filepath = "./video/train"
    i = 0
    carState = "stop"
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
            motor_go(speedSet)
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            motor_stop()
        elif keyValue == 81:
            print("left")
            carState = "left"
            motor_left(speedSet)
        elif keyValue == 83:
            print("right")
            carState = "right"
            motor_right(speedSet)
        
        _, image = camera.read()
        #image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        height, _, _ = image.shape
        save_image = image[int(height/2):,:,:]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (3,3), 0)
        save_image = cv2.resize(save_image, (200,66))
        cv2.imshow('Save', save_image)
        
        if carState == "left":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 45), save_image)
            i += 1
        elif carState == "right":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 135), save_image)
            i += 1
        elif carState == "go":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 90), save_image)
            i += 1
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    PWMA.value = 0.0
    PWMB.value = 0.0
```

- 이미지 저장이 완료되면 저장된 이미지를 압축하여 PC로 전송한 후, 학습을 수행함



## 2. 모델 생성 (Tensorflow)

- Colab을 이용하여 Tensorflow로 모델 학습 수행
    - nvidia에서 배포한 모델을 기반으로 학습 진행


<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/project/notebooks/S10-01-05-03_01-AutonomousDrivingControlImplementation1.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>



- GPU 확인 및 데이터 업로드/압축해제

```python
import tensorflow as tf

# GPU 사용 가능 여부 확인
if tf.test.gpu_device_name():
    print("GPU 사용 가능")
    # 현재 GPU 디바이스 이름 출력
    print("GPU 디바이스 이름:", tf.test.gpu_device_name())
else:
    print("GPU 사용 불가능")

import os
os.system("unzip ./video.zip")

```

- 이미지 처리 시 사용하는 imgaug 패키지가 numpy 1.26.4를 요구하므로 해당 버전으로 재설치
- imgaug 패키지가 numpy 1.26.4를 삭제하고 최신 버전을 설치하려고 시도하므로 의존성 파일에 손대지 못하도록 설정

```bash
!pip uninstall numpy -y
!pip install numpy==1.26.4
!pip install imgaug --no-deps
```

- 패키지 가져오기

```python
import os
import random
import fnmatch
import datetime
import pickle

# data processing
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})

import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)

# tensorflow
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

print( f'tf.__version__: {tf.__version__}' )

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# imaging
import cv2
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from PIL import Image
```

- 데이터 불러오기

```python
data_dir = './'
file_list = os.listdir(data_dir)
image_paths = []
steering_angles = []
pattern = "*.png"
for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir,filename))
        angle = int(filename[-7:-4])
        steering_angles.append(angle)

image_index = 20
plt.imshow(Image.open(image_paths[image_index]))
print("image_path: %s" % image_paths[image_index] )
print("steering_Angle: %d" % steering_angles[image_index] )
df = pd.DataFrame()
df['ImagePath'] = image_paths
df['Angle'] = steering_angles
```

- 조향각의 분포 확인

```python
num_of_bins = 25
hist, bins = np.histogram(df['Angle'], num_of_bins)

fig, axes = plt.subplots(1,1, figsize=(12,4))
axes.hist(df['Angle'], bins=num_of_bins, width=1, color='blue')
```

- 학습데이터와 검증데이터를 분리

```python
X_train, X_valid, y_train, y_valid = train_test_split( image_paths, steering_angles, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].hist(y_train, bins=num_of_bins, width=1, color='blue')
axes[0].set_title('Training Data')
axes[1].hist(y_valid, bins=num_of_bins, width=1, color='red')
axes[1].set_title('Validation Data')
```

- 이미지 읽어오기 및 정규화함수

```python
def my_imread(image_path):
    image = cv2.imread(image_path)
    return image

def img_preprocess(image):
    image = image / 255
    return image

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
image_orig = my_imread(image_paths[image_index])
image_processed = img_preprocess(image_orig)
axes[0].imshow(image_orig)
axes[0].set_title("orig")
axes[1].imshow(image_processed)
axes[1].set_title("processed")
```

- nvidia 모델구성

```python
def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)

    return model

model = nvidia_model()
print(model.summary())
```

- 학습데이터 생성

```python
def image_data_generator(image_paths, steering_angles, batch_size):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]

            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield( np.asarray(batch_images), np.asarray(batch_steering_angles))
```

```python
ncol = 2
nrow = 2

X_train_batch, y_train_batch = next(image_data_generator(X_train, y_train, nrow))
X_valid_batch, y_valid_batch = next(image_data_generator(X_valid, y_valid, nrow))

fig, axes = plt.subplots(nrow, ncol, figsize=(15, 6))
fig.tight_layout()

for i in range(nrow):
    axes[i][0].imshow(X_train_batch[i])
    axes[i][0].set_title("training, angle=%s" % y_train_batch[i])
    axes[i][1].imshow(X_valid_batch[i])
    axes[i][1].set_title("validation, angle=%s" % y_valid_batch[i])
```

- 모델 학습(5~10분가량 소요)

```python
model_output_dir = "./model/"

checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir,'lane_navigation_check.keras'), verbose=1, save_best_only=True)

history = model.fit(image_data_generator( X_train, y_train, batch_size=100),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data = image_data_generator( X_valid, y_valid, batch_size=100),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1,
                              callbacks=[checkpoint_callback])

model.save(os.path.join(model_output_dir,'lane_navigation_final.keras'))

history_path = os.path.join(model_output_dir,'history.pickle')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
```

- 결과확인

```python
history.history

history_path = os.path.join(model_output_dir,'history.pickle')
with open(history_path, 'rb') as f:
    history = pickle.load(f)

history
plt.plot(history['loss'],color='blue')
plt.plot(history['val_loss'],color='red')
plt.legend(["training loss", "validation loss"])
```

- 검증

```python
from sklearn.metrics import mean_squared_error, r2_score

def summarize_prediction(Y_true, Y_pred):

    mse = mean_squared_error(Y_true, Y_pred)
    r_squared = r2_score(Y_true, Y_pred)

    print(f'mse       = {mse:.2}')
    print(f'r_squared = {r_squared:.2%}')
    print()

def predict_and_summarize(X, Y):
    model = load_model(f'{model_output_dir}/lane_navigation_check.keras')
    Y_pred = model.predict(X)
    summarize_prediction(Y, Y_pred)
    return Y_pred

n_tests = 100
X_test, y_test = next(image_data_generator(X_valid, y_valid, 100))

y_pred = predict_and_summarize(X_test, y_test)

n_tests_show = 2
fig, axes = plt.subplots(n_tests_show, 1, figsize=(10, 4 * n_tests_show))
for i in range(n_tests_show):
    axes[i].imshow(X_test[i])
    axes[i].set_title(f"actual angle={y_test[i]}, predicted angle={int(y_pred[i])}, diff = {int(y_pred[i])-y_test[i]}")
```

- 결과 파일 목록
    - history.pickle
    - lane_navigation_check.keras
    - lane_navigation_final.keras (학습완료된 모델)


## 3. 모델 생성 (Pytorch)

- 라즈베리파이 5에서의 각종 버전 총돌 문제로 인하여 파이토치 버전으로 재작성

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/project/notebooks/S10-01-05-03_01-AutonomousDrivingControlImplementation2.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

- 필요한 패키지 가져오기

```python
import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
import cv2
import fnmatch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

- 데이터 가져오기 및 분할

```python
data_dir = './video/'
file_list = os.listdir(data_dir)
image_paths = []
steering_angles = []
pattern = "*.png"

for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir,filename))
        angle = int(filename[-7:-4])
        steering_angles.append(angle)

X_train, X_valid, y_train, y_valid = train_test_split( image_paths, steering_angles, test_size=0.2, random_state=42)
print(f"Training data: {len(X_train)}\nValidation data: {len(X_valid)}")
```

- 전처리 함수 정의

```python
def img_preprocess_pytorch(image):
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    # Convert numpy array to PyTorch tensor
    image = torch.from_numpy(image).float()
    # Rearrange dimensions from (H, W, C) to (C, H, W)
    image = image.permute(2, 0, 1)
    return image
```

- GPU 사용 설정

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- 데이터셋 정의

```python
class DrivingDataset(Dataset):
    def __init__(self, image_paths, steering_angles):
        self.image_paths = image_paths
        self.steering_angles = steering_angles

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path) # Use cv2.imread from previous definition
        # Ensure image is not None for cases where imread might fail
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = img_preprocess_pytorch(image)
        steering_angle = torch.tensor(self.steering_angles[idx], dtype=torch.float32)
        return image, steering_angle
```

- 데이터로더 생성

```python
batch_size = 100

train_dataset = DrivingDataset(X_train, y_train)
valid_dataset = DrivingDataset(X_valid, y_valid)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of validation batches: {len(valid_dataloader)}")

for images, angles in train_dataloader:
    print(f"Sample batch image shape: {images.shape}")
    print(f"Sample batch steering angles shape: {angles.shape}")
    print(f"Sample image data type: {images.dtype}")
    print(f"Sample angle data type: {angles.dtype}")
    images = images.to(device)
    angles = angles.to(device)
    print(f"Images moved to device: {images.device}")
    print(f"Angles moved to device: {angles.device}")
    break
```

- NVIDIA 모델 정의

```python
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
```

- 모델 생성 및 설정

```python
model = NvidiaModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")
print("Model Architecture:")
print(model)

dummy_input = torch.randn(1, 3, 66, 200).to(device) # Batch_size, Channels, Height, Width
output = model(dummy_input)
print(f"Output shape with dummy input: {output.shape}")

loss_fn = nn.MSELoss()
learning_rate = 1e-3 # Consistent with the original Keras model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- 학습 함수 정의

```python
def train_epoch(model, dataloader, loss_fn, optimizer, device, steps_per_epoch):
    model.train() # Set model to training mode
    running_loss = 0.0
    data_iter = iter(dataloader)

    for batch_idx in range(steps_per_epoch):
        try:
            images, angles = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, angles = next(data_iter)

        images = images.to(device)
        angles = angles.to(device).float().unsqueeze(1) # Ensure angles are float and have correct shape

        outputs = model(images)
        loss = loss_fn(outputs, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / steps_per_epoch
```

- 검증 함수 정의

```python
def validate_epoch(model, dataloader, loss_fn, device, validation_steps):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    data_iter = iter(dataloader)

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx in range(validation_steps):
            try:
                images, angles = next(data_iter)
            except StopIteration:
                # If the iterator is exhausted, re-initialize it
                data_iter = iter(dataloader)
                images, angles = next(data_iter)

            images = images.to(device)
            angles = angles.to(device).float().unsqueeze(1) # Ensure angles are float and have correct shape

            outputs = model(images)
            loss = loss_fn(outputs, angles)

            running_loss += loss.item()

    return running_loss / validation_steps
```

- 하이퍼 파라미터 및 기타 설정

```python
model_output_dir = "./model/"
os.makedirs(model_output_dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

epochs = 10
steps_per_epoch = 300
validation_steps = 200

history = {'loss': [], 'val_loss': []}
best_val_loss = float('inf')
```

- 학습 수행

```python
print("Starting model training...")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, steps_per_epoch)
    val_loss = validate_epoch(model, valid_dataloader, loss_fn, device, validation_steps)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(model_output_dir, 'lane_navigation_check.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1}: Validation loss improved to {best_val_loss:.4f}, saving model checkpoint to {checkpoint_path}")

final_model_path = os.path.join(model_output_dir, 'lane_navigation_final.pt')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

history_path = os.path.join(model_output_dir, 'history.pickle')
with open(history_path, 'wb') as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
print(f"Training history saved to {history_path}")

print("Model training complete.")
```

- 예측 작업

```python
def predict_and_summarize_pytorch(model_class, model_path, dataloader, device, n_tests_show=2):
    model = model_class()
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode

    true_angles = []
    predicted_angles = []

    print(f"Evaluating model from: {model_path}")

    with torch.no_grad(): # Disable gradient calculations
        for images, angles in dataloader:
            images = images.to(device)
            outputs = model(images)

            true_angles.extend(angles.cpu().numpy())
            predicted_angles.extend(outputs.cpu().numpy().flatten())

    true_angles = np.array(true_angles)
    predicted_angles = np.array(predicted_angles)

    mse = mean_squared_error(true_angles, predicted_angles)
    r_squared = r2_score(true_angles, predicted_angles)

    print(f'mse       = {mse:.2f}')
    print(f'r_squared = {r_squared:.2%}')
    print()

    # Visualize a small subset of predictions
    fig, axes = plt.subplots(n_tests_show, 1, figsize=(10, 4 * n_tests_show))
    fig.tight_layout()

    for i in range(n_tests_show):
        image, actual_angle = valid_dataset[i] # Get raw preprocessed image and angle
        input_image = image.unsqueeze(0).to(device)
        predicted_angle = model(input_image).item()
        display_image = image.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(display_image)
        axes[i].set_title(f"actual angle={actual_angle.item():.0f}, predicted angle={predicted_angle:.0f}, diff = {predicted_angle - actual_angle.item():.0f}")
        axes[i].axis('off') # Hide axes for cleaner image display

    plt.show()

    return predicted_angles
```

- 학습된 모델 읽어오기

```python
model_output_dir = './model/'
checkpoint_path = os.path.join(model_output_dir, 'lane_navigation_check.pt')
history_path = os.path.join(model_output_dir, 'history.pickle')

if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    print("Training history loaded successfully.")
else:
    print(f"Error: History file not found at {history_path}")
    history = {'loss': [], 'val_loss': []}
```

- 학습 결과 출력

```python
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], color='blue', label='training loss')
plt.plot(history['val_loss'], color='red', label='validation loss')

plt.title('Model Training History (Loss over Epochs)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')

plt.legend()
plt.grid(True)
plt.show()
```


## 4. 모델 적용 후 자율 주행

```python
import mycamera
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
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

speedSet = 0.4

class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
    
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) 
    image = image / 255.0
    return image

def main():

    camera = mycamera.MyPiCamera(640,480)

    model_path = "./model/lane_navigation_final.pt"
    model = NvidiaModel()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"PyTorch model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        print("Please ensure YourPyTorchModelClass is correctly defined and the .pt file is valid.")
        return

    carState = "stop"
    
    while camera.isOpened():
        keValue = cv2.waitKey(1)
        
        if keValue == ord('q') :
            break

        elif keValue == 82 :
            print("go")
            carState = "go"

        elif keValue == 84 :
            print("stop")
            carState = "stop"
        

        _, image = camera.read()
        cv2.imshow('Original', image)
        
        preprocessed = img_preprocess(image)
        cv2.imshow('pre', preprocessed)
        
        X = torch.from_numpy(preprocessed).float()
        X = X.permute(2, 0, 1).unsqueeze(0) 

        with torch.no_grad():
            output = model(X)
        
        steering_angle = int(output.cpu().numpy()[0])
        print("predict angle:",steering_angle)

        
        #if carState == "go":
        if steering_angle >= 85 and steering_angle <= 95:
            print("go")
            motor_go(speedSet)
        elif steering_angle > 96:
            print("right")
            motor_right(speedSet)
        elif steering_angle < 84:
            print("left")
            motor_left(speedSet)
        #elif carState == "stop":
        #    motor_stop()
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    PWMA.value = 0.0
    PWMB.value = 0.0
```

- 각 시스템마다 하드웨어, 센서, 카메라 각도 등의 조정이 필요함
- 학습 상태에 따라 주행 성능이 달라짐