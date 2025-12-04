---
layout: page
title:  "자동차 무선 조종기능 구현"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S05-03-03-01_01-RaspberryPiRemoteControl
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 블루투스 시리얼 통신으로 데이터 주고 받기

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