---
layout: page
title:  "라즈베리파이 제어 기초"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S05-03-02-01_01-RaspberryPiControl
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>



## 1. 오픈소스 하드웨어(OSHW)

- **OSHW (Open Source HardWare)**
    - 해당 제품과 똑같은 모양 및 기능을 가진 제품을 만드는 데 필요한 모든 것(회로도, 자재 명세서, 인쇄 회로기판 도면 등)을 대중에게 공개한 전자제품
    - 하드웨어 기술 언어가 대중에게 공개된 프로그래머블 논리 소자
        - 하드웨어 기술 언어: 전자회로를 정밀하게 기술하는 데 사용하는 컴퓨터 언어 (Hardware Description Language, HDL)

- **대표적인 OSHW**
    - 아두이노(2005~), 라즈베리파이(2012~), 젯슨나노(2019~) 및 아류작

<div style="text-align:center; width: 990px;"><h4>OSHW 종류별 비교</h4></div>
<div class="info-table" style="width: 990px;">
    <table>
        <caption>(자료출처: 테크월드(http://www.epnc.co.kr/news/articleView.html?idxno=97412)에서 참조 후 편집)</caption>
        <thead>
            <th>종류</th>
            <th>외형</th>
            <th>등장시기</th>
            <th>운영체제</th>
            <th>가격대</th>
            <th>접근성</th>
            <th>특징</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">아두이노</td>
                <td><img style="width: 150px;" src="/materials/sbc/images/S05-03-02-01_01-001.png"></td>
                <td>2005 ~</td>
                <td>X (없음)</td>
                <td>저가</td>
                <td>높음</td>
                <td>하드웨어 센서, 모듈 제어</td>
            </tr>
            <tr>
                <td class="td-rowheader">라즈베리파이</td>
                <td><img style="width: 150px;" src="/materials/sbc/images/S05-03-02-01_01-002.png"></td>
                <td>2012 ~</td>
                <td>Linux 계열<br>(기본: 라즈비안)</td>
                <td>저가</td>
                <td>높음</td>
                <td>범용성, 애플리케이션 개발</td>
            </tr>
            <tr>
                <td class="td-rowheader">라떼판다</td>
                <td><img style="width: 150px;" src="/materials/sbc/images/S05-03-02-01_01-003.png"></td>
                <td>2015 ~</td>
                <td>Windows 10</td>
                <td>고가</td>
                <td>보통</td>
                <td>
                    윈도우 기반 IoT 시스템 개발<br>
                    아두이노를 기반으로 고성능화
                </td>
            </tr>
            <tr>
                <td class="td-rowheader">젯슨나노</td>
                <td><img style="width: 150px;" src="/materials/sbc/images/S05-03-02-01_01-004.png"></td>
                <td>2019 ~</td>
                <td>
                    Linux계열<br>(기본: Ubuntu 18.04 LTS 기반<br>NVIDIA 자체 커스터마이징)
                </td>
                <td>중가<br>(최근 고가)</td>
                <td>보통</td>
                <td>AI 기반 애플리케이션 개발</td>
            </tr>
        </tbody>
    </table>
</div>

## 2. 라즈베리파이 개요

- **대표적인 Single Board Computer(SBC)**
    - OS를 설치한 SD/Micro SD 카드로 부팅해서 PC로 사용 가능

    - GPIO를 이용하여 전자회로 등을 직접 제어할 수 있음
    - 리눅스 계열의 OS 사용
        - 권장하는 OS는 Raspberry Pi OS(구, 라즈비안(Raspbian))
        - 리눅스 기반 OS를 사용하므로 파이썬 기본 설치
        - 파이썬을 이용하여 GUI, 대규모 프로그램을 개발 가능
        - 전자회로 제어를 기반으로 한 다양한 응용 시스템 개발 가능

- **구조**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 800px;" src="/materials/sbc/images/S05-03-02-01_01-005.jpg">
    </div>

    - 제품 사양 참조: [https://www.raspberrypi.com/products/raspberry-pi-5/](https://www.raspberrypi.com/products/raspberry-pi-5/){: target="blank"}


- **구성 요소**
    - **GPIO 확장 커넥터**
        - 프로세서(SoC) 포트에 직접 연결하는 핀 타입 커넥터임
        - LED나 스위치 등 전자 부품을 연결할 수 있음

    - **2x4-레인 MIPI DSI/CSI 커넥터**
        - 카메라 모듈을 연결하는 전용 커넥터
        - MIPI: Mobile Industry Processor Interface
        - CSI: Camera Serial Interface
        - DSI: Display Serial Interface

    - **micro-HDMI 커넥터**
        - TV나 모니터를 연결하는 커넥터
        - 음성 신호도 출력할 수 있으므로 TV나 스피커가 내장된 모니터에 연결하면 됨

    - **PCI Express 인터페이스(PCIe/PCI-e)**
        - PCI, PCI-X, AGP 표준을 대체하도록 설계된 고속 데이터 전송에 사용되는 연결 단자 표준
        - 장치와 마더 보드 및 기타 하드웨어 간의 고 대역폭 통신을 허용

    - **RTC 배터리 커넥터**
        - 라즈베리파이 5 전용 RTC(Real Time Clock) 포트와 호환 가능한 배터리 커넥터

    - **UART 커넥터**
        - UART: Universal Asynchronous Receiver-Transmitter (범용 비동기 송수신기)
        - 두 장치 간의 직렬 통신을 위한 간단한 2선식 프로토콜
        - 병렬 데이터를 직렬 데이터로 변환하여 한 번에 하나씩 보내고, 수신 측에서는 다시 병렬 데이터로 재구성
        - 데이터 전송 속도를 맞추기 위한 동기 신호를 별도로 사용하지 않으므로, 미리 정해진 규칙(프로토콜)에 따라 데이터의 시작과 끝을 맞춰야 함

    - **PoE HAT 커넥터**
        - PoE: Power over Ethernet
            - 이더넷 케이블에 데이터와 함께 48 V DC 전원을 같이 보내는 기술
            - 유선 랜 케이블 하나로 인터넷 접속 및 전원 공급을 동시에 할 수 있어 라즈베리파이에 추가적인 전원 어댑터 연결을 하지 않을 수 있음
        - HAT: Hardware Attached on Top
            - 라즈베리파이의 GPIO 핀에 연결되어 추가 기능을 제공하는 확장 보드를 의미

- **사용 시 주의점**
    - 기판이 노출되어 있어서 외부 충격에 약하며 물에 닿으면 금방 고장남
    - 컴퓨터 칩(SoC)에 직접 연결된 GPIO 확장 커넥터 핀도 돌출되어 있어서 정전기 등 과도한 전압이 걸리면 SoC가 파손될 수 있음
    - 이런 사고를 막기 위해 라즈베리 파이용 케이스를 씌우기도 함

## 3. 라즈베리파이 제어 기초

### 3.1 GPIO 개요

- **GPIO(General Purpose Input/Output)**
    - 라즈베리파이 model A/B를 제외하고 모든 모델은 40핀 규격을 사용함
        - 모델 A/B는 24핀 규격

        <div class="insert-image" style="text-align: center;">
            <h4>라즈베리파이 Pin Out (40핀)</h4><br>
            <img style="width: 800px;" src="/materials/sbc/images/S05-03-02-01_01-006.png">
        </div>


### 3.2 LED로 전조등 구현하기

- 전조등 LED 제어용 GPIO 핀
    - 전방 좌측: 26
    - 전방 우측: 16
    - 후방 좌측: 21
    - 후방 우측: 20

- 

```python
#//file: "control_LED.py"
import RPi.GPIO as GPIO
import time

LED1 = 26
LED2 = 16
LED3 = 20
LED4 = 21

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED1,GPIO.OUT)
GPIO.setup(LED2,GPIO.OUT)
GPIO.setup(LED3,GPIO.OUT)
GPIO.setup(LED4,GPIO.OUT)

try:
    while True:
        GPIO.output(LED1,GPIO.HIGH)
        GPIO.output(LED2,GPIO.HIGH)
        GPIO.output(LED3,GPIO.HIGH)
        GPIO.output(LED4,GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(LED1,GPIO.LOW)
        GPIO.output(LED2,GPIO.LOW)
        GPIO.output(LED3,GPIO.LOW)
        GPIO.output(LED4,GPIO.LOW)
        time.sleep(1.0)

except KeyboardInterrupt:
    pass

GPIO.cleanup()
```

### 3.3 버튼 입력받기

```python
#//file: "input_Button.py"
import RPi.GPIO as GPIO
import time

LED_PIN = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

GPIO.output(LED_PIN, True)
print("LED turn on")

time.sleep(5.0)

GPIO.output(LED_PIN, False)
print("LED turn off")

GPIO.cleanup()
```

### 3.4 부저로 경적기능 구현하기

```python
#//file: "control_Buzzer.py"
import RPi.GPIO as GPIO
import time

LED_PIN = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

GPIO.output(LED_PIN, True)
print("LED turn on")

time.sleep(5.0)

GPIO.output(LED_PIN, False)
print("LED turn off")

GPIO.cleanup()
```

### 3.5 모터를 구동하여 자동차 움직이기

```python
#//file: "control_Motor.py"
import RPi.GPIO as GPIO
import time 

DUTY_MIN = 3   # 서보모터가 입력받을 수 있는 듀티의 최소값
DUTY_MAX = 12  # 서보모터가 입력받을 수 있는 듀티의 최대값
i=0

servo_pin = 25  # PWM 출력을 서보모터에 보내줄 pin번호

GPIO.setmode(GPIO.BCM) 
GPIO.setup(servo_pin, OUTPUT)

servo = GPIO.PWM(servo_pin, 50) # PWM을 50Hz(==20ms)로 셋업
servo.start(0) # 듀티 0에서 시작

def posContDeg(deg):
  # 입력받은 각도를 듀티로 변환
  duty = DUTY_MIN+(deg*(DUTY_MAX - DUTY_MIN)/180) 

  # 듀티에 따라 서보모터 제어
  servo.ChangeDutyCycle(duty)
  print("%d Degree Rotated" deg)

if __name__ = "__main__":
  for i in range (0,180):
    posContDeg(i)
    sleep(0.5)
    i++

  servo.stop()
  GPIO.cleanup()
```

### 3.6 스위치를 입력받아 자동차 조종해보기

```python
#//file: "control_Car.py"
import RPi.GPIO as GPIO
import time

LED_PIN = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)

GPIO.output(LED_PIN, True)
print("LED turn on")

time.sleep(5.0)

GPIO.output(LED_PIN, False)
print("LED turn off")

GPIO.cleanup()
```
