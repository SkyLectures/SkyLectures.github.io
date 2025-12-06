---
layout: page
title:  "자동차 무선 조종기능 구현"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-02-03_02-RaspberryPiRemoteControl
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 블루투스 시리얼 통신으로 데이터 주고 받기

### 1.1 HM-10 블루투스 4.0 BLE

- TI CC2540 또는 **CC2541 블루투스 SOC(시스템 온 칩)**를 기반으로 하는 소형 3.3V SMD 블루투스 4.0 BLE 모듈
- 아두이노, 라즈베리파이 등 호환

- **기본 사양**
    - CC2540 또는 CC2541 칩 기반
    - +2.5v ~ +3.3v
    - 최대 50mA 필요 
        - 활성 상태일 때: 약 9mA 사용
        - 수면모드 상태일 때: 50~200uA 사용
    - RF 전력: -23dbm, -6dbm, 0dbm, 6dbm
    - 블루투스 버전 4.0만 지원
        - HC-06 및 HC-05와 같은 Bluetooth 2/2.1 모듈에 연결할 수 없음
    - 직렬 UART 연결을 통해 전송되는 AT 명령을 통해 제어됨
    - 직렬 연결의 기본 통신 속도: 9600
    - 기본 PIN: 000000
    - 기본 이름: HMSoft

    <div class="insert-image" style="text-align: left;">
        <img style="width: 300px;" src="/materials/project/images/S10-01-02-03_02-001.png"><br>
        <div style="width: 300px; text-align: right;">(그림출처: 디바이스마트)</div>
    </div>

### 1.2 블루투스 통신 환경 설정

- Serial Port ➜ 활성화
- Serial Console ➜ 비활성화

<div class="insert-image" style="text-align: left;">
    <img style="width: 600px;" src="/materials/project/images/S10-01-02-03_02-002.png">
</div>

> - **설정 이유**
>   - **리소스(직렬 통신 포트)의 충돌 방지 및 전용 사용**이 목적
>   - 라즈베리파이에서 블루투스를 사용하려면
>       - 블루투스 모듈이 사용할 직렬 통신 포트(Serial Port)를 활성화하여 통신 채널을 열어주고
>       - 이 포트가 다른 용도(Serial Console)로 사용되지 않도록 비활성화하여
>       - 충돌을 막고 블루투스 전용으로 만들어 주어야 함
>
>   - UART (Universal Asynchronous Receiver/Transmitter, 범용 비동기 송수신기)
>       - 블루투스 모듈은 흔히 UART를 통해 라즈베리파이의 메인 CPU와 통신을 수행함
>           - SPI(Serial Peripheral Interface), USB(Universal Serial Bus), SDIO(Secure Digital Input/Output), I2C(Inter-Integrated Circuit) 등 다른 통신 수단도 있지만 UART가 가장 기본적인 시리얼 통신의 하나이므로 저속 데이터의 전송 및 제어에 적합함
>       - 한 번에 하나의 기능만 제대로 수행할 수 있음
>
>   - **Serial Port를 활성화하는 이유**
>       - 블루투스 모듈과의 통신 채널 확보
>           - 블루투스 모듈이 라즈베리파이 OS와 데이터를 주고받으려면, 이 통신을 위한 하드웨어적인 통로(직렬 포트, 즉 UART)가 열려 있어야 함
>           - 블루투스 기능(예: SPP 프로필)이 직렬 통신을 기반으로 작동하기 때문
>           - 이 포트를 활성화해야 블루투스 통신이 가능해짐
>
>   - **Serial Console을 비활성화하는 이유**
>       - 직렬 포트 점유 방지
>           - Serial Console은 라즈베리파이의 UART를 사용하여 부팅 메시지나 터미널(CLI) 접속 등을 제공하는 기능
>           - 라즈베리파이 부팅 과정이나 운영체제 동작 중 발생하는 로그나 명령 프롬프트를 직렬 케이블을 통해 외부 컴퓨터에서 볼 수 있게 해주는 `디버깅/관리용 콘솔`
>           - 굳이 직렬 포트를 점유하며 수행해야 할 필요는 없음
>       - 블루투스와의 충돌 방지
>           - Serial Console이 활성화되어 있으면, 블루투스 모듈과 Serial Console이 동일한 UART 자원을 서로 사용하려고 경쟁하게 됨
>           - 이 경우 둘 중 어느 것도 제대로 작동하지 않게 되어 블루투스 통신에 오류가 발생하거나, 콘솔 접속이 불안정해질 수 있음
>       - 블루투스 기능의 안정성 확보
>           - Serial Console을 비활성화함으로써 해당 UART 자원을 블루투스 모듈이 전용으로 사용하게 하여, 블루투스 통신의 안정성과 신뢰성을 확보할 수 있음
{: .expert-quote}

### 1.3 라즈베리파이 적용

- 라즈베리파이의 GPIO 14(RXD), 15(TXD)번 핀이 시리얼 통신용으로 할당되어 있음
- **`/dev/serial0`** 의 이름으로 호출됨

```bash
#// file: "라즈베리파이 터미널"
ls -l /dev/serial0
```

- `ttyAMA10`이라는 이름으로 할당됨 (시스템에 따라 다를 수 있음) ➜ 파이썬 코드 작성 시 사용할 serial0 접속명
<div class="insert-image" style="text-align: left;">
    <img style="width: 990px;" src="/materials/project/images/S10-01-02-03_02-003.png">
</div>

- 시리얼 통신 테스트: 전송되는 데이터가 없으므로 빈 데이터만 표시됨

```python
#// file: "bluetooth_test.py"
import serial

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

try:
    while True:
        data = bleSerial.read()
        print(data)
        
except KeyboardInterrupt:
    pass

bleSerial.close()
```

<div class="insert-image" style="text-align: left;">
    <img style="width: 990px;" src="/materials/project/images/S10-01-02-03_02-004.png">
</div>

### 1.4 무선 조종을 위한 스마트폰 앱 설치

- 안드로이드 기종
    - 플레이스토어에서 `Serial Bluetooth Terminal` 검색하여 설치

    <div class="insert-image">
        <img style="width: 300px; border: 1px solid gray;" src="/materials/project/images/S10-01-02-03_02-005.jpg">
        <span style="font-size:2em;">&nbsp;&nbsp;&nbsp;➜&nbsp;&nbsp;&nbsp;</span>
        <img style="width: 300px;" src="/materials/project/images/S10-01-02-03_02-006.jpg">
    </div>

    - `Serial Bluetooth Terminal`에서 라즈베리파이의 블루투스 탐색 및 연결
        - 주로 MLT-BT05, HM-10, BT05 등의 이름으로 검색됨

    <div class="insert-image" style="text-align: right;">
        <img style="width: 960px;" src="/materials/project/images/S10-01-02-03_02-007.png"><br><br>
        <img style="width: 990px;" src="/materials/project/images/S10-01-02-03_02-008.png"><br><br>
        <img style="width: 330px;" src="/materials/project/images/S10-01-02-03_02-009.png">
        <img style="width: 655px;" src="/materials/project/images/S10-01-02-03_02-010.png">
    </div>

- 아이폰 기종의 경우
    - App Store에서 `ble automation` 앱을 검색 후 설치할 것
        - 무료 터미널 앱이지만 가끔 광고가 나타남
        - 사용법은 동일하지만 라즈베리파이에서 전송한 값을 확인할 수 없음
            - 아이폰용 앱(ble automation)에서 전송 값을 표시하는 기능을 지원하지 않음

    <div class="insert-image" style="text-align: right;">
        <img style="width: 950px;" src="/materials/project/images/S10-01-02-03_02-011.png">
    </div>

### 1.5 라즈베리파이 ➜ 스마트폰 데이터 전송 확인

```python
#// file: "bluetooth.py"
import serial
import time

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

try:
    while True:
        sendData = "I am raspberry \r\n"
        bleSerial.write( sendData.encode() )
        time.sleep(1.0)
        
except KeyboardInterrupt:
    pass

bleSerial.close()
```

<div class="insert-image">
    <img style="width: 400px;" src="/materials/project/images/S10-01-02-03_02-012.jpg">
</div>


## 2. 시리얼 데이터를 분석하여 명령어 해석하기

```python
#// file: "serial_command.py"
import serial

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

try:
    while True:
        data = bleSerial.readline()
        data = data.decode()
        if data.find("go") >= 0:
            print("ok go")
        elif data.find("back") >= 0:
            print("ok back")
        elif data.find("left") >= 0:
            print("ok left")
        elif data.find("right") >= 0:
            print("ok right")
        elif data.find("stop") >= 0:
            print("ok stop")
        
        
except KeyboardInterrupt:
    pass

bleSerial.close()
```

## 3. 쓰레드를 활용하여 통신기능 분리하기

```python
#// file: "thread_comm.py"
import threading
import serial
import time

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

gData = ""

def serial_thread():
    global gData
    while True:
        data = bleSerial.readline()
        data = data.decode()
        gData = data

def main():
    global gData
    try:
        while True:
            print("serial data:",gData)
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target = serial_thread)
    task1.start()
    main()
    bleSerial.close()
```

## 4. 블루투스 시리얼 통신으로 조종하는 자동차 만들기

```python
#// file: "bluetooth_control.py"
import threading
import serial
import time
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

gData = ""

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
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = 0.0

def serial_thread():
    global gData
    while True:
        data = bleSerial.readline()
        data = data.decode()
        gData = data

def main():
    global gData
    try:
        while True:
            if gData.find("go") >= 0:
                gData = ""
                print("ok go")
                motor_go(0.5)
            elif gData.find("back") >= 0:
                gData = ""
                print("ok back")
                motor_back(0.5)
            elif gData.find("left") >= 0:
                gData = ""
                print("ok left")
                motor_left(0.5)
            elif gData.find("right") >= 0:
                gData = ""
                print("ok right")
                motor_right(0.5)
            elif gData.find("stop") >= 0:
                gData = ""
                print("ok stop")
                motor_stop()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target = serial_thread)
    task1.start()
    main()
    bleSerial.close()
    PWMA.value = 0.0
    PWMB.value = 0.0
```

## 5. 스위치를 이용하여 비상 정지기능 만들기

```python
#// file: "emergency_stop.py"
import threading
import serial
import time
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from gpiozero import Button

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

gData = ""

SW1 = Button(5, pull_up=False )
SW2 = Button(6, pull_up=False )
SW3 = Button(13, pull_up=False )
SW4 = Button(19, pull_up=False )

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
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = 0.0

def serial_thread():
    global gData
    while True:
        data = bleSerial.readline()
        data = data.decode()
        gData = data

def main():
    global gData
    try:
        while True:
            if gData.find("go") >= 0:
                gData = ""
                print("ok go")
                motor_go(0.5)
            elif gData.find("back") >= 0:
                gData = ""
                print("ok back")
                motor_back(0.5)
            elif gData.find("left") >= 0:
                gData = ""
                print("ok left")
                motor_left(0.5)
            elif gData.find("right") >= 0:
                gData = ""
                print("ok right")
                motor_right(0.5)
            elif gData.find("stop") >= 0:
                gData = ""
                print("ok stop")
                motor_stop()
                
            if SW1.is_pressed == True or SW2.is_pressed == True or SW3.is_pressed == True or SW4.is_pressed == True :
                motor_stop()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target = serial_thread)
    task1.start()
    main()
    bleSerial.close()
    PWMA.value = 0.0
    PWMB.value = 0.0
```

## 6. LED로 이동방향 표시하기

```python
#// file: "move_direction.py"
import threading
import serial
import time
from gpiozero import Button
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from gpiozero import LED

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

gData = ""

SW1 = Button(5, pull_up=False )
SW2 = Button(6, pull_up=False )
SW3 = Button(13, pull_up=False )
SW4 = Button(19, pull_up=False )

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

LED1 = LED(26)
LED2 = LED(16)
LED3 = LED(20)
LED4 = LED(21)

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
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = 0.0

def serial_thread():
    global gData
    while True:
        data = bleSerial.readline()
        data = data.decode()
        gData = data

def main():
    global gData
    try:
        while True:
            if gData.find("go") >= 0:
                gData = ""
                print("ok go")
                motor_go(0.5)
                LED1.on()
                LED2.on()
                LED3.off()
                LED4.off()
            elif gData.find("back") >= 0:
                gData = ""
                print("ok back")
                motor_back(0.5)
                LED1.off()
                LED2.off()
                LED3.on()
                LED4.on()
            elif gData.find("left") >= 0:
                gData = ""
                print("ok left")
                motor_left(0.5)
                LED1.on()
                LED2.off()
                LED3.on()
                LED4.off()
            elif gData.find("right") >= 0:
                gData = ""
                print("ok right")
                motor_right(0.5)
                LED1.off()
                LED2.on()
                LED3.off()
                LED4.on()
            elif gData.find("stop") >= 0:
                gData = ""
                print("ok stop")
                motor_stop()
                LED1.off()
                LED2.off()
                LED3.off()
                LED4.off()
                
            if SW1.is_pressed == True or SW2.is_pressed == True or SW3.is_pressed == True or SW4.is_pressed == True :
                motor_stop()
                LED1.off()
                LED2.off()
                LED3.off()
                LED4.off()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target = serial_thread)
    task1.start()
    main()
    bleSerial.close()
    PWMA.value = 0.0
    PWMB.value = 0.0
    LED1.off()
    LED2.off()
    LED3.off()
    LED4.off()
```

## 7. 부저를 이용하여 경적기능 추가하기

```python
#// file: "buzzer.py"
import threading
import serial
import time
from gpiozero import Button
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from gpiozero import LED
from gpiozero import TonalBuzzer

bleSerial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1.0)

gData = ""

SW1 = Button(5, pull_up=False )
SW2 = Button(6, pull_up=False )
SW3 = Button(13, pull_up=False )
SW4 = Button(19, pull_up=False )

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

LED1 = LED(26)
LED2 = LED(16)
LED3 = LED(20)
LED4 = LED(21)

BUZZER = TonalBuzzer(12)

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
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = 0.0

def serial_thread():
    global gData
    while True:
        data = bleSerial.readline()
        data = data.decode()
        gData = data

def main():
    global gData
    try:
        while True:
            if gData.find("go") >= 0:
                gData = ""
                print("ok go")
                motor_go(0.5)
                LED1.on()
                LED2.on()
                LED3.off()
                LED4.off()
            elif gData.find("back") >= 0:
                gData = ""
                print("ok back")
                motor_back(0.5)
                LED1.off()
                LED2.off()
                LED3.on()
                LED4.on()
            elif gData.find("left") >= 0:
                gData = ""
                print("ok left")
                motor_left(0.5)
                LED1.on()
                LED2.off()
                LED3.on()
                LED4.off()
            elif gData.find("right") >= 0:
                gData = ""
                print("ok right")
                motor_right(0.5)
                LED1.off()
                LED2.on()
                LED3.off()
                LED4.on()
            elif gData.find("stop") >= 0:
                gData = ""
                print("ok stop")
                motor_stop()
                LED1.off()
                LED2.off()
                LED3.off()
                LED4.off()
            elif gData.find("bz_on") >= 0:
                gData = ""
                print("ok buzzer on")
                BUZZER.play(391)
            elif gData.find("bz_off") >= 0:
                gData = ""
                print("ok buzzer off")
                BUZZER.stop()
            
            if SW1.is_pressed == True or SW2.is_pressed == True or SW3.is_pressed == True or SW4.is_pressed == True :
                motor_stop()
                LED1.off()
                LED2.off()
                LED3.off()
                LED4.off()
                BUZZER.stop()
                
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target = serial_thread)
    task1.start()
    main()
    bleSerial.close()
    PWMA.value = 0.0
    PWMB.value = 0.0
    LED1.off()
    LED2.off()
    LED3.off()
    LED4.off()
    BUZZER.stop()
```

## 8. 라즈베리파이 부팅 시 자동으로 코드 실행하기

```python
#// file: ".py"

```