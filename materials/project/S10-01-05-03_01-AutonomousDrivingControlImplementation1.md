---
layout: page
title:  "라즈베리파이 기반 자율주행자동차 구현(1)"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-05-03_01-AutonomousDrivingControlImplementation1
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



## 2. 모델 생성

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

## 3. 모델 적용 후 자율 주행

- 학습완료된 모델을 라즈베리파이에 전송
    - lane_navigation_final.keras

- 파이썬 버전 변경
    - 현재 제공되는 자율주행 키트용 소스코드가 사용하는 Tensorflow 기능은 의존성 문제로 인하여 3.11.x 버전 이하의 파이썬 환경에서만 작동함

    - 빌드도구 설치

    ```bash
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
                        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

    - pyenv 설치

    ```bash
    curl https://pyenv.run | bash
    ```

    - 셸(Shell) 환경 설정

    ```bash
    nano ~/.bashrc
    ```

    - 파일의 제일 아래쪽에 추가
    
    ```bash
    #//file: "~/.bashrc"
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```

    ```bash
    source ~/.bashrc

    pyenv install 3.11.8
    pyenv rehash
    pyenv versions
    ```

    - 가상환경 생성 및 활성화

    ```bash
    pyenv virtualenv 3.11.8 ai_car_py311
    pyenv activate ai_car_py311
    ```

    - 가상환경이 활성화되었다면 작업 디렉토리는 원하는 장소를 사용하면 됨



- 자동차 조향각 예측 코드

```python
import mycamera
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) 
    image = image / 255
    return image

def main():
    camera = mycamera.MyPiCamera(640,480)
    model_path = '/home/pi/AI_CAR/model/lane_navigation_final.keras'
    model = load_model(model_path)
    
    carState = "stop"
    
    while( camera.isOpened()):
        
        keValue = cv2.waitKey(1)
        
        if keValue == ord('q') :
            break
        
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('Original', image)
        
        preprocessed = img_preprocess(image)
        cv2.imshow('pre', preprocessed)
        
        X = np.asarray([preprocessed])
        steering_angle = int(model.predict(X)[0])
        print("predict angle:",steering_angle)
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
```


> - 하드웨어 제어 시에는 **전원 연결**과 **핀 연결 오류**에 항상 주의해야 하며,
> - 특히 모터는 라즈베리파이와 별도의 고전류 전원을 사용해야 함
> - 과전류로 인한 라즈베리파이 손상을 방지하는 것이 매우 중요
{: .expert-quote}