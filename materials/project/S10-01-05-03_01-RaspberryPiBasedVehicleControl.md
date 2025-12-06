---
layout: page
title:  "라즈베리파이 기반 차량 제어"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-05-03_01-RaspberryPiBasedVehicleControl
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - **수업 목표**
>   - 라즈베리파이의 GPIO 핀과 PWM 기능을 이해한다.
>   - DC 모터 드라이버(L298N 또는 유사)의 원리와 라즈베리파이와의 연결 방법을 학습한다.
>   - 파이썬 **RPi.GPIO** 라이브러리를 사용하여 모터의 방향과 속도를 제어하는 코드를 작성한다.
>   - 인지 모델의 예측 결과(예: 차선 중앙 편차, 목표 속도)를 PID 제어기에 연결하여 차량을 자율적으로 제어한다.
>   - TensorFlow 및 PyTorch 환경에서 딥러닝 모델의 출력이 라즈베리파이 차량 제어에 어떻게 통합되는지 실습을 통해 경험한다.
>   - 하드웨어 제어 시 발생할 수 있는 문제점과 안전 고려사항을 이해한다.
{: .summary-quote}


## 1. 라즈베리파이 기반 차량 제어의 개요

### 1.1. 자율주행 파이프라인에서 제어의 위치

- **인지(Perception)**
    - 주변 환경(차선, 객체, 신호등)을 인식함

- **판단(Decision-making)**
    - 인지 결과를 바탕으로 현재 상황에 가장 적합한 행동(직진, 회피, 정지 등)과 목표 궤적을 결정함

- **제어(Control)**
    - 판단된 목표를 달성하기 위해 차량의 스티어링, 가속, 브레이크를 직접 조작함
    - AI가 물리적 세계에 직접 영향을 미치는 부분이며, 코드가 실제 행동으로 이어지는 부분

### 1.2. 왜 라즈베리파이인가?

- **컴팩트하고 저렴함**
    - 자율주행 미니카 제작에 적합한 가격과 크기를 가짐

- **강력한 확장성**
    - GPIO 핀을 통해 다양한 센서(초음파, IMU) 및 액추에이터(모터, 서보)를 연결할 수 있음

- **Python 친화적**
    - 파이썬 개발 환경이 잘 갖춰져 있어 딥러닝 모델 연동 및 제어 로직 구현이 용이함

- **카메라 모듈**
    - 라즈베리파이 전용 카메라 모듈 또는 USB 웹캠을 통해 영상 데이터를 쉽게 얻을 수 있음

### 1.3. 시스템 구성 요소
- **라즈베리파이**
    - 두뇌 역할 (AI 모델 추론, 제어 로직 실행)

- **카메라 모듈**
    - 인지 모듈의 입력 (시각 정보 수집)

- **DC 모터**
    - 차량 구동 (바퀴 회전)

- **모터 드라이버 (예: L298N)**
    - 라즈베리파이의 낮은 전압/전류로 모터를 제어하기 위한 인터페이스

- **전원**
    - 라즈베리파이 및 모터 드라이버/모터 구동용 (별도의 고전류 전원 필요)

- **샤시 및 바퀴**
    - 차량 구조


## 2. 라즈베리파이 GPIO 기초 및 모터 드라이버 연동

### 2.1. GPIO

- **정의**
    - 범용 입출력 핀(General Purpose Input/Output, GPIO)
    - 디지털 신호(HIGH/LOW)를 주고받을 수 있음

- **모드**
    - **BCM 모드**
        - 핀 번호를 Broadcom SOC(System On Chip) 채널 번호로 사용 (권장)
    - **BOARD 모드**
    - 핀 번호를 라즈베리파이 기판의 물리적 핀 번호로 사용

- **설정**
    - `RPi.GPIO` 라이브러리를 사용하여 핀을 입력 또는 출력으로 설정함

### 2.2. PWM

- **정의**
    - PWM (Pulse Width Modulation)
    - 디지털 신호를 사용하여 아날로그와 유사한 제어를 수행하는 기법

- **원리**
    - 주기적인 HIGH/LOW 신호에서 HIGH 상태의 폭(듀티 사이클)을 조절하여 평균 전압을 변화시킴

- **활용**
    - DC 모터의 속도 조절, 서보 모터의 각도 조절 등에 사용됨

    - **듀티 사이클 (Duty Cycle)**
        - 한 주기에서 HIGH 신호가 차지하는 비율(0~100%)
        - 높을수록 모터 속도가 빨라짐

    - **주파수 (Frequency)**
        - 한 주기의 길이
        - 모터 제어에는 보통 수십~수백 Hz를 사용함

### 2.3. L298N 모터 드라이버 연동

- **역할**
    - 라즈베리파이의 3.3V/5V 신호를 받아 모터가 요구하는 더 높은 전압/전류(예: 7V~12V)를 공급함
    - 모터의 정방향/역방향 회전을 제어함

- **연결 예시 (DC 모터 2개 제어)**
    - **라즈베리파이 GPIO 핀 (출력)**
        - IN1, IN2: 모터1의 방향 제어 (HIGH/LOW 조합)
        - IN3, IN4: 모터2의 방향 제어
        - ENA: 모터1의 PWM 속도 제어
        - ENB: 모터2의 PWM 속도 제어

    - **L298N 핀**
        - OUT1, OUT2: 모터1 연결
        - OUT3, OUT4: 모터2 연결
        - 12V (VIN): 외부 전원 연결 (모터용)
        - GND: 라즈베리파이 GND와 연결
        - 5V (VSS): 라즈베리파이 5V와 연결 (내장 5V 레귤레이터 사용 시)


## 3. 차량 제어 메커니즘

### 3.1. 모터 제어 기본 동작

- **앞으로 이동**: 좌/우 모터 모두 정방향 회전
- **뒤로 이동**: 좌/우 모터 모두 역방향 회전
- **좌회전 (회전 반경)**: 좌측 모터 감속 또는 정지, 우측 모터 전진
- **우회전 (회전 반경)**: 우측 모터 감속 또는 정지, 좌측 모터 전진
- **제자리 좌회전 (Pivot Turn)**: 좌측 모터 역방향, 우측 모터 정방향 회전
- **제자리 우회전 (Pivot Turn)**: 우측 모터 역방향, 좌측 모터 정방향 회전

### 3.2. PID 제어 통합

- **입력**
    - 인지 모듈(카메라 영상) -> 딥러닝 모델 (차선/객체 인식) ➜ 판단 모듈 (목표 조향각, 목표 속도)

- **제어**
    - **오차**
        - **횡방향 제어**: 차선 중앙과의 편차 (Cross-Track Error) 또는 목표 조향각과 현재 조향각의 차이
        - **종방향 제어**: 목표 속도와 현재 차량 속도의 차이
    - **PID 컨트롤러**: 이 오차를 최소화하는 제어 신호(Duty Cycle)를 생성
    - **액추에이터 구동**: 생성된 듀티 사이클을 모터 드라이버의 PWM 핀으로 전달하여 모터 속도 조절


## 4. 라즈베리파이 기반 차량 제어 구현 실습

- **실습 내용**
    - 가상의 딥러닝 모델로부터 '조향 오차(Steering Error)'와 '목표 속도'를 입력받아 차량을 제어하는 시뮬레이션 구현
    - 실제 라즈베리파이에서 실행하려면 
        - **RPi.GPIO** 라이브러리가 필요함
        - **L298N 모터 드라이버**가 GPIO 핀에 연결되어 있어야 함

### 4.1. RPiMotorController 클래스

- 라즈베리파이의 GPIO 핀을 통해 모터를 직접 제어하는 역할(GPIO 및 PWM 제어)

```python
import RPi.GPIO as GPIO
import time
import numpy as np

# 이전 강의에서 정의한 PIDController 클래스를 재활용합니다.
# 이 파일을 PIDController.py 등으로 저장하고 import 하거나, 아래에 직접 붙여넣으세요.
# class PIDController: ... (이전 강의 코드 복사)
# 편의상 현재 코드에 포함하겠습니다.

class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point, output_limits=(-100.0, 100.0), integral_limits=None, dt=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.output_limits = output_limits
        self.integral_limits = integral_limits if integral_limits else (self.output_limits[0] * 10, self.output_limits[1] * 10)
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

    def calculate_control_signal(self, current_value):
        error = self.set_point - current_value

        p_term = self.Kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, self.integral_limits[0], self.integral_limits[1])
        i_term = self.Ki * self.integral

        derivative = (error - self.prev_error) / self.dt
        d_term = self.Kd * derivative

        control_signal = p_term + i_term + d_term
        control_signal = np.clip(control_signal, self.output_limits[0], self.output_limits[1])
        
        self.prev_error = error
        return control_signal
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

class RPiMotorController:
    def __init__(self, in1, in2, enA, in3, in4, enB):
        """
        Raspberry Pi GPIO를 사용하여 L298N 모터 드라이버 제어
        Args:
            in1, in2 (int): 모터1 (좌측) 방향 제어 핀
            enA (int): 모터1 (좌측) PWM 속도 제어 핀
            in3, in4 (int): 모터2 (우측) 방향 제어 핀
            enB (int): 모터2 (우측) PWM 속도 제어 핀
        """
        self.in1 = in1
        self.in2 = in2
        self.enA = enA
        self.in3 = in3
        self.in4 = in4
        self.enB = enB

        GPIO.setmode(GPIO.BCM)  # BCM 핀 번호 사용
        GPIO.setwarnings(False) # GPIO 경고 비활성화

        # 핀 설정
        for pin in [self.in1, self.in2, self.enA, self.in3, self.in4, self.enB]:
            GPIO.setup(pin, GPIO.OUT)

        # PWM 객체 생성 (주파수 100Hz)
        self.pwmA = GPIO.PWM(self.enA, 100)
        self.pwmB = GPIO.PWM(self.enB, 100)
        self.pwmA.start(0)  # 듀티 사이클 0으로 시작
        self.pwmB.start(0)

        print("모터 컨트롤러 초기화 완료. GPIO 핀 설정됨.")

    def set_motor_direction(self, motor_num, direction):
        """
        특정 모터의 방향 설정
        Args:
            motor_num (int): 1 (좌측 모터) 또는 2 (우측 모터)
            direction (str): 'forward', 'backward', 'stop'
        """
        if motor_num == 1: # 좌측 모터
            if direction == 'forward':
                GPIO.output(self.in1, GPIO.HIGH)
                GPIO.output(self.in2, GPIO.LOW)
            elif direction == 'backward':
                GPIO.output(self.in1, GPIO.LOW)
                GPIO.output(self.in2, GPIO.HIGH)
            elif direction == 'stop':
                GPIO.output(self.in1, GPIO.LOW)
                GPIO.output(self.in2, GPIO.LOW)
        elif motor_num == 2: # 우측 모터
            if direction == 'forward':
                GPIO.output(self.in3, GPIO.HIGH)
                GPIO.output(self.in4, GPIO.LOW)
            elif direction == 'backward':
                GPIO.output(self.in3, GPIO.LOW)
                GPIO.output(self.in4, GPIO.HIGH)
            elif direction == 'stop':
                GPIO.output(self.in3, GPIO.LOW)
                GPIO.output(self.in4, GPIO.LOW)
    
    def set_motor_speed(self, motor_num, speed):
        """
        특정 모터의 속도 설정 (PWM 듀티 사이클 0-100)
        Args:
            motor_num (int): 1 (좌측 모터) 또는 2 (우측 모터)
            speed (float): 0.0 (정지) ~ 100.0 (최대 속도)
        """
        speed = np.clip(speed, 0, 100) # 0~100 범위로 제한
        if motor_num == 1: # 좌측 모터
            self.pwmA.ChangeDutyCycle(speed)
        elif motor_num == 2: # 우측 모터
            self.pwmB.ChangeDutyCycle(speed)

    def move_forward(self, speed):
        self.set_motor_direction(1, 'forward')
        self.set_motor_direction(2, 'forward')
        self.set_motor_speed(1, speed)
        self.set_motor_speed(2, speed)

    def move_backward(self, speed):
        self.set_motor_direction(1, 'backward')
        self.set_motor_direction(2, 'backward')
        self.set_motor_speed(1, speed)
        self.set_motor_speed(2, speed)

    def stop(self):
        self.set_motor_direction(1, 'stop')
        self.set_motor_direction(2, 'stop')
        self.set_motor_speed(1, 0)
        self.set_motor_speed(2, 0)
        
    def turn(self, steering_input, base_speed):
        """
        조향 입력에 따라 차량의 방향을 제어
        Args:
            steering_input (float): -1.0 (최대 좌회전) ~ 1.0 (최대 우회전)
            base_speed (float): 기본 주행 속도 (0-100)
        """
        left_speed = base_speed
        right_speed = base_speed
        
        if steering_input < 0: # 좌회전 (좌측 모터 감속/역회전)
            # 회전 강도에 따라 좌측 모터 속도 조절
            left_speed = max(0, base_speed * (1 + steering_input)) # steering_input은 음수
            self.set_motor_direction(1, 'forward')
            self.set_motor_direction(2, 'forward')
        elif steering_input > 0: # 우회전 (우측 모터 감속/역회전)
            right_speed = max(0, base_speed * (1 - steering_input)) # steering_input은 양수
            self.set_motor_direction(1, 'forward')
            self.set_motor_direction(2, 'forward')
        else: # 직진
            self.set_motor_direction(1, 'forward')
            self.set_motor_direction(2, 'forward')
        
        self.set_motor_speed(1, left_speed)
        self.set_motor_speed(2, right_speed)

    def cleanup(self):
        """GPIO 자원 해제"""
        self.stop()
        self.pwmA.stop()
        self.pwmB.stop()
        GPIO.cleanup()
        print("모터 컨트롤러 자원 해제 완료.")

# --- 라즈베리파이가 아닌 PC 환경에서 테스트하기 위한 Dummy GPIO 클래스 ---
# 실제 라즈베리파이가 없어도 코드를 실행해볼 수 있습니다.
if not 'RPi.GPIO' in globals(): # RPi.GPIO가 import 되어 있지 않다면 Dummy 클래스 사용
    print("WARNING: RPi.GPIO 라이브러리를 찾을 수 없습니다. Dummy GPIO 모드로 작동합니다.")
    print("이 코드는 실제 라즈베리파이 하드웨어를 제어하지 않습니다.")

    class DummyGPIO:
        BCM = 0
        OUT = 0
        HIGH = 1
        LOW = 0
        
        def setmode(self, mode):
            print(f"[DummyGPIO] setmode: {mode}")
        def setwarnings(self, flag):
            pass
        def setup(self, pin, mode):
            print(f"[DummyGPIO] setup pin {pin} as {mode}")
        def output(self, pin, value):
            print(f"[DummyGPIO] pin {pin} output: {value}")
        def cleanup(self):
            print("[DummyGPIO] cleanup")
        
        class PWM:
            def __init__(self, pin, frequency):
                self.pin = pin
                self.frequency = frequency
                print(f"[DummyPWM] PWM initialized on pin {pin} with {frequency}Hz")
            def start(self, duty_cycle):
                print(f"[DummyPWM] PWM {self.pin} started with {duty_cycle}%")
            def ChangeDutyCycle(self, duty_cycle):
                print(f"[DummyPWM] PWM {self.pin} changed duty cycle to {duty_cycle}%")
            def stop(self):
                print(f"[DummyPWM] PWM {self.pin} stopped")

    GPIO = DummyGPIO()
```

### 4.2. RealtimeVehicleControl 클래스

- 실시간 제어 클래스
- 카메라 영상을 입력으로 받아 딥러닝 모델로 추론
- PID 제어기를 통해 모터를 제어하는 메인 로직을 포함


#### 4.2.1. TensorFlow 버전

```python
import cv2
import time
# import tensorflow as tf

class RealtimeVehicleControl:
    def __init__(self, model_path, motor_pins, dt=0.1, framework='tensorflow'):
        self.motor_controller = RPiMotorController(**motor_pins)
        
        # 종방향 (속도) PID 제어기
        self.speed_pid = PIDController(
            Kp=5.0, Ki=0.1, Kd=0.5, set_point=0.0,
            output_limits=(-100.0, 100.0), dt=dt
        )
        # 횡방향 (조향) PID 제어기 (목표 조향 오차 0.0을 유지)
        self.steering_pid = PIDController(
            Kp=30.0, Ki=0.05, Kd=1.0, set_point=0.0,
            output_limits=(-1.0, 1.0), # -1.0 (최대 좌회전) ~ 1.0 (최대 우회전)
            dt=dt
        )
        
        self.dt = dt
        self.framework = framework
        self.model = self._load_model(model_path)
        
        # 모델의 입력 이미지 크기 (이전 강의의 인지 모델 가정)
        self.image_size = (384, 640) 
        
        # 초기 목표 속도 및 현재 속도 (가상)
        self.target_speed_mps = 5.0 # m/s (예: 약 18km/h)
        self.current_speed_mps = 0.0 # m/s (가상)

    def _load_model(self, model_path):
        """딥러닝 모델 로드 (프레임워크에 따라 다름)"""
        if self.framework == 'tensorflow':
            import tensorflow as tf
            # 실제 인지 모델은 무거우므로 TFLite 최적화 모델을 사용하는 것이 좋음
            # model = tf.lite.Interpreter(model_path=model_path)
            # model.allocate_tensors()
            # return model
            
            # 현재는 더미 모델로 대체 (입력받아 조향/속도 명령 가상 생성)
            print(f"[TF Dummy Model] 모델 로드: {model_path}")
            return self._dummy_tf_model
        elif self.framework == 'pytorch':
            import torch
            # from your_pytorch_model_definition import MultiTaskPerceptionModel # PyTorch 모델 클래스 정의 필요
            # model = MultiTaskPerceptionModel(...) # 모델 초기화
            # model.load_state_dict(torch.load(model_path))
            # model.eval()
            # return model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 현재는 더미 모델로 대체
            print(f"[PyTorch Dummy Model] 모델 로드: {model_path}")
            return self._dummy_pytorch_model
        else:
            raise ValueError("Unsupported framework. Use 'tensorflow' or 'pytorch'.")

    def _dummy_tf_model(self, input_frame):
        """TensorFlow Dummy Model: 인지 결과를 가상으로 생성"""
        # 실제 인지 모델은 차선, 객체, 세그멘테이션 마스크를 반환
        # 여기서는 그 결과를 바탕으로 조향 오차와 속도 목표를 가상으로 생성
        # 이 값을 PID 제어기의 set_point 또는 current_value에 공급
        
        # 현재는 카메라 프레임으로부터 직접 가상의 오차값 생성
        # (실제 딥러닝 모델은 차선 마스크에서 조향 오차 계산, 도로 영역 비율에서 속도 판단)
        # 예시: 프레임의 중앙에서 무작위로 좌우 편차를 시뮬레이션
        lane_deviation_error = (np.random.rand() - 0.5) * 0.4 # -0.2 ~ 0.2 범위
        
        # 앞차와의 거리에 따른 속도 조정 (가상)
        # current_speed_measurement = self.current_speed_mps # PID 컨트롤러가 현재 속도를 받음
        
        return lane_deviation_error
    
    def _dummy_pytorch_model(self, input_frame):
        """PyTorch Dummy Model: 인지 결과를 가상으로 생성"""
        # Tensorlfow 더미와 동일하게 가상 오차값 생성
        lane_deviation_error = (np.random.rand() - 0.5) * 0.4 # -0.2 ~ 0.2 범위
        return lane_deviation_error


    def _get_dl_inference_output(self, frame):
        """
        딥러닝 모델 추론 및 결과 반환
        (여기서는 더미 모델을 사용)
        Args:
            frame (numpy.array): 카메라 프레임 (BGR)
        Returns:
            float: 조향 오차 (예: -1.0 ~ 1.0)
            float: (가상) 현재 속도 (m/s)
        """
        # 프레임 전처리 (모델 입력 크기에 맞게)
        processed_frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        
        if self.framework == 'tensorflow':
            # TensorFlow 모델 추론 (더미)
            lane_deviation_error = self._dummy_tf_model(processed_frame)
            current_speed_measurement = self.current_speed_mps # 가상 측정
        else: # PyTorch
            # PyTorch 모델 추론 (더미)
            lane_deviation_error = self._dummy_pytorch_model(processed_frame)
            current_speed_measurement = self.current_speed_mps # 가상 측정
            
        return lane_deviation_error, current_speed_measurement


    def run_control_loop(self, camera_index=0, target_speed_kph=20):
        """
        카메라 영상을 기반으로 실시간 차량 제어 루프 실행
        Args:
            camera_index (int): 사용할 카메라 인덱스
            target_speed_kph (float): 목표 속도 (km/h)
        """
        self.target_speed_mps = target_speed_kph / 3.6 # km/h를 m/s로 변환
        self.speed_pid.set_point = self.target_speed_mps # 속도 PID의 목표값 설정

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다. 카메라 인덱스를 확인하거나 USB 카메라를 연결하세요.")
            self.motor_controller.cleanup()
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start_time = time.time()
        frame_count = 0

        try:
            print(f"\n--- 라즈베리파이 차량 제어 루프 시작 (목표 속도: {target_speed_kph:.1f} km/h) ---")
            print("종료하려면 'q' 키를 누르세요.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 받아오지 못했습니다.")
                    break
                
                # 1. 딥러닝 모델 추론 (인지 결과 얻기)
                #   lane_deviation_error: 차선 중앙으로부터의 편차 (-1.0:최대좌, 0:중앙, 1.0:최대우)
                #   current_speed_measurement: 현재 차량의 (가상) 속도 m/s
                lane_deviation_error, current_speed_measurement = self._get_dl_inference_output(frame)
                
                # 가상 현재 속도 업데이트 (여기서는 제어 신호에 의해 속도가 변하는 것을 시뮬레이션)
                # 이 부분은 실제 차량에서는 인코더 등에서 측정됩니다.
                self.current_speed_mps = current_speed_measurement 
                
                # 2. 횡방향 제어 (조향) - 목표 조향 오차 0.0을 유지
                # current_value = lane_deviation_error
                # set_point = 0.0 (차선 중앙)
                # output = steering_control_signal (-1.0 ~ 1.0)
                steering_control_signal = self.steering_pid.calculate_control_signal(lane_deviation_error)
                
                # 3. 종방향 제어 (속도) - 목표 속도 self.target_speed_mps 유지
                # current_value = self.current_speed_mps
                # set_point = self.target_speed_mps
                # output = speed_control_signal (-100.0 ~ 100.0) -> 가속/감속 힘
                speed_control_signal = self.speed_pid.calculate_control_signal(self.current_speed_mps)

                # 제어 신호를 모터 제어 명령으로 변환 (base_speed와 steering_input으로)
                # speed_control_signal (m/s 가속)을 모터 듀티 사이클 (0-100%)로 변환 필요
                # 이 변환은 시스템에 따라 다릅니다. 여기서는 간략하게 처리합니다.
                
                # 양의 speed_control_signal이면 가속, 음의 speed_control_signal이면 감속으로 해석
                # 현재 시뮬레이션은 차량 속도를 PID로 직접 제어하므로, 제어 신호에 맞춰 현재 속도를 업데이트
                
                # (가상의) Base Speed 결정 (예: 듀티 사이클 0~100으로 변환)
                # 이 값은 control_signal이 100이면 100%, -100이면 0% 등으로 매핑되어야 함
                base_speed_duty_cycle = np.clip(speed_control_signal * (100 / self.speed_pid.output_limits[1]), 0, 100)
                
                # 4. 모터 제어 액추에이터 구동
                self.motor_controller.turn(steering_control_signal, base_speed_duty_cycle)
                
                # (가상) 속도 업데이트 (차량 모델의 가속도 반영을 위해)
                # 이 시뮬레이션에서는 PID 출력이 다음 current_speed_mps에 영향을 줌
                # (실제 차량에서는 모터 제어를 통해 가속되고 센서로 속도 측정)
                # self.current_speed_mps += speed_control_signal * self.dt * 0.1 # 간단한 가속도 모델

                # 5. 시각화 및 디버깅
                current_time = time.time()
                elapsed_time = current_time - start_time
                frame_count += 1
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # 시각화를 위한 텍스트 추가
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Target Speed: {self.target_speed_mps:.2f} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Current Speed (sim): {self.current_speed_mps:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Steering Error: {lane_deviation_error:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Steering Output: {steering_control_signal:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Speed Output (PWM %): {base_speed_duty_cycle:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 중앙선 표시
                cv2.line(display_frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (0, 0, 255), 1)
                # 예측된 오차를 시각적으로 표현 (빨간 점)
                predicted_center_x = int(frame.shape[1] / 2 + lane_deviation_error * (frame.shape[1] / 2))
                cv2.circle(display_frame, (predicted_center_x, frame.shape[0]-50), 10, (0, 0, 255), -1)

                cv2.imshow('Vehicle Control Interface', display_frame)
                
                if cv2.waitKey(int(self.dt * 1000)) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            self.motor_controller.cleanup()
            print("\n--- 차량 제어 루프 종료 ---")

# --- 메인 실행 ---
if __name__ == '__main__':
    # 라즈베리파이 GPIO 핀 번호 설정 (BCM 모드 기준)
    # 실제 연결에 따라 수정 필요!
    motor_pin_config = {
        'in1': 27, # Left Motor Direction Pin 1
        'in2': 22, # Left Motor Direction Pin 2
        'enA': 17, # Left Motor PWM Speed Pin
        'in3': 24, # Right Motor Direction Pin 1
        'in4': 23, # Right Motor Direction Pin 2
        'enB': 18, # Right Motor PWM Speed Pin
    }
    
    # -----------------------------------------------------------
    # TensorFlow 기반 모델 연동 예시
    print("=== TensorFlow 기반 차량 제어 시뮬레이션 ===")
    tf_model_dummy_path = "dummy_tf_model.h5" # 실제 모델 경로로 대체
    tf_control_system = RealtimeVehicleControl(
        model_path=tf_model_dummy_path, 
        motor_pins=motor_pin_config, 
        framework='tensorflow'
    )
    # tf_control_system.run_control_loop(camera_index=0, target_speed_kph=20)
    
    # -----------------------------------------------------------
    # PyTorch 기반 모델 연동 예시
    print("\n=== PyTorch 기반 차량 제어 시뮬레이션 ===")
    pt_model_dummy_path = "dummy_pt_model.pth" # 실제 모델 경로로 대체
    pt_control_system = RealtimeVehicleControl(
        model_path=pt_model_dummy_path, 
        motor_pins=motor_pin_config, 
        framework='pytorch'
    )
    # pt_control_system.run_control_loop(camera_index=0, target_speed_kph=20)
    
    # --- 한 가지만 선택하여 실행 ---
    # 실제 라즈베리파이에서는 이 주석을 해제하여 실행하세요.
    # PC에서는 dummy GPIO와 camera_index=0이 웹캠을 의미할 수 있습니다.
    
    # 예시: TensorFlow 버전을 20km/h로 실행
    tf_control_system.run_control_loop(camera_index=0, target_speed_kph=20)
```

## **5. 실습 과제 및 응용 아이디어**

- **PID 게인 재조정**
    - `speed_pid`와 `steering_pid`의 $$K_p, K_i, K_d$$ 값을 수정하여
    - 차량의 주행 안정성(진동 여부)과 반응성(목표 추종 속도)을 조절해 보기
        - 실제 미니카에서 테스트하면 최적의 게인 값을 찾는 것이 중요함

- **경량 딥러닝 모델 통합**
    - 이전 강의에서 학습한 TFLite 또는 PyTorch ONNX 모델을
    - `_get_dl_inference_output` 함수에 직접 통합하여
    - 차선 인식 또는 객체 탐지 결과를 실제 입력으로 사용해 보기
        - 이를 위해서는 인지 모델의 출력에서 `lane_deviation_error`와 `current_speed_measurement`를 계산하는 로직을 추가해야 함

- **안전 기능 추가**
    - **초음파 센서 연동**
        - 초음파 센서로 전방 장애물과의 거리를 측정하여
        - 너무 가까워지면 속도를 줄이거나 정지하는 코드를 추가

    - **비상 정지 버튼**
        - GPIO 핀에 버튼을 연결하여 누르면 차량이 즉시 정지하는 기능을 구현

- **웹 인터페이스 제어**
    - 라즈베리파이에 웹 서버(Flask 등)를 구축하여
    - 웹 브라우저를 통해 차량의 속도나 방향을 원격으로 제어하는 기능을 추가하고,
    - 인지된 영상과 제어 상태를 실시간으로 모니터링해 보기

- **차량 동역학 시뮬레이션**
    - `RealtimeVehicleControl` 클래스 내에
    - `self.current_speed_mps`를 `SimulatedVehicle` 클래스(이전 강의 참고)와 연동시켜
    - PID 제어 신호가 실제 차량 속도에 어떻게 반영되는지 좀 더 정교하게 시뮬레이션 해 볼 것


> - 하드웨어 제어 시에는 **전원 연결**과 **핀 연결 오류**에 항상 주의해야 하며,
> - 특히 모터는 라즈베리파이와 별도의 고전류 전원을 사용해야 함
> - 과전류로 인한 라즈베리파이 손상을 방지하는 것이 매우 중요
{: .expert-quote}