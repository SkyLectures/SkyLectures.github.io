---
layout: page
title:  "자율주행 제어의 기초 및 PID 제어"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-05-02_01-AutonomousDrivingControlPidControl
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>






스카이님, 자율주행의 핵심 단계 중 하나인 **"제어(Control)"** 단계, 특히 그 기초와 가장 널리 사용되는 **PID 제어**에 대한 강의 자료를 상세하게 준비했습니다. 인지 및 판단 모듈에서 결정된 '무엇을 할지'를 바탕으로 차량을 '어떻게 움직일지'를 결정하는 단계입니다.

이번 강의에서는 PID 제어의 원리를 설명하고, 이 제어기가 텐서플로우와 파이토치 환경에서 어떻게 시뮬레이션될 수 있는지 실습 예제와 함께 보여드리겠습니다. PID 제어 자체는 딥러닝 모델이 아니지만, 딥러닝 프레임워크를 활용하여 제어 시스템의 동작을 시뮬레이션하고 학습 환경을 구축하는 방법을 이해하는 데 도움이 될 것입니다.

---

# **자율주행 제어의 기초 및 PID 제어**

## **[차시 목표]**

*   자율주행 시스템에서 '제어(Control)' 단계의 역할과 중요성을 이해한다.
*   자율주행차의 종방향 제어(속도)와 횡방향 제어(조향)의 기본 개념을 학습한다.
*   PID 제어(Proportional-Integral-Derivative)의 각 요소(P, I, D) 원리를 이해한다.
*   OpenCV, TensorFlow, PyTorch 환경에서 간단한 PID 제어를 시뮬레이션하고 구현한다.
*   PID 제어의 자율주행 적용 사례와 한계점을 파악한다.

---

## **1. 자율주행 제어(Control) 단계의 이해: 운전자의 '손과 발'**

### 1.1. 제어 단계란?
*   **정의**: 판단(Decision-making) 모듈로부터 전달된 주행 명령(예: "목표 속도 50km/h로 차선 중앙 유지", "좌회전 궤적 추종")을 받아, 차량의 물리적 구동 장치(액추에이터)인 **스티어링 휠(조향), 가속 페달, 브레이크를 직접 조작**하여 차량을 실제 세계에서 움직이게 하는 단계입니다.
*   **운전자의 손과 발**: 인지 모듈이 '눈', 판단 모듈이 '두뇌'라면, 제어 모듈은 '손과 발' 역할을 하여 운전자의 지시에 따라 차량을 조작하는 역할을 합니다.

### 1.2. 제어의 중요성
*   **정밀성**: 명령된 궤적을 오차 없이 정확하게 추종해야 합니다.
*   **안정성**: 차량이 불안정하게 흔들리거나, 과도한 움직임을 보이지 않도록 부드럽게 제어해야 합니다.
*   **반응성**: 외부 환경 변화나 상위 모듈의 명령 변경에 즉각적으로 반응하여 대응해야 합니다.
*   **안전성**: 급가속/급정거, 과도한 조향 등으로 인한 위험 상황을 방지하고, 승차감을 해치지 않아야 합니다.

### 1.3. 제어의 종류
자율주행 제어는 크게 두 가지 방향으로 나눌 수 있습니다.
*   **종방향 제어 (Longitudinal Control)**:
    *   **목표**: 차량의 **속도** 및 **가속도**를 제어합니다.
    *   **액추에이터**: 가속 페달(Throttle), 브레이크(Brake)
    *   **적용**: 크루즈 컨트롤, 앞차와의 간격 유지(ACC)
*   **횡방향 제어 (Lateral Control)**:
    *   **목표**: 차량의 **방향** 및 **경로**를 제어합니다.
    *   **액추에이터**: 스티어링 휠(Steering Wheel)
    *   **적용**: 차선 유지(LKA), 경로 추종

### 1.4. 제어의 입력과 출력
*   **입력**: 판단 모듈이 생성한 **목표 궤적(Desired Trajectory)**, **목표 속도**, **목표 조향각** 등의 명령과 현재 차량의 **현재 상태(Current State)** 정보(속도, 위치, 조향각 등 센서 피드백)
*   **출력**: 차량의 액추에이터(모터, 브레이크 등)로 전달되는 **구동 신호** (예: 가속 페달 개도율, 브레이크 압력, 스티어링 휠 각도)

---

## **2. PID 제어의 기초 원리**

PID 제어(Proportional-Integral-Derivative Control)는 산업 제어 시스템에서 가장 널리 사용되는 피드백 제어 방식 중 하나입니다. 단순하면서도 강력한 성능을 발휘하여 다양한 시스템에 적용됩니다.

### 2.1. PID 제어의 목표
*   **목표값(Set Point, SP)**과 **현재값(Process Variable, PV)** 사이의 **오차(Error, $e(t)$)**를 최소화하여 시스템의 현재값을 목표값에 도달시키고 유지하는 것입니다.

### 2.2. PID 제어식
*   $u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$
    *   $u(t)$: 제어량 (액추에이터에 전달되는 신호)
    *   $e(t)$: 오차 ($SP - PV$)
    *   $K_p$: 비례 게인 (Proportional Gain)
    *   $K_i$: 적분 게인 (Integral Gain)
    *   $K_d$: 미분 게인 (Derivative Gain)

### 2.3. 각 요소의 역할

#### P (Proportional) 제어: 비례 제어
*   **원리**: 현재 오차($e(t)$)에 비례하여 제어량을 출력합니다. 오차가 클수록 더 강하게 제어합니다.
*   **효과**: 시스템을 목표값으로 빠르게 접근시킵니다.
*   **단점**:
    *   너무 크면 오버슈트(Over-shoot, 목표값을 넘어감)나 진동이 발생할 수 있습니다.
    *   일반적으로 정상 상태 오차(Steady-state Error)를 완전히 제거하지 못합니다. (목표값에 정확히 도달하지 못하고 일정한 오차를 남김)

#### I (Integral) 제어: 적분 제어
*   **원리**: 과거의 모든 오차($\int e(t) dt$)를 누적하여 제어량을 출력합니다. 시스템에 남아있는 미세한 정상 상태 오차를 제거하는 데 사용됩니다.
*   **효과**: 시스템이 목표값에 **정확히 도달**하도록 합니다.
*   **단점**:
    *   오차를 누적하므로 반응 속도를 늦출 수 있습니다.
    *   너무 크면 진동이 심해지거나 '와인드업(Wind-up)' 현상이 발생할 수 있습니다.

#### D (Derivative) 제어: 미분 제어
*   **원리**: 오차의 변화율($\frac{de(t)}{dt}$)에 비례하여 제어량을 출력합니다. 오차가 급격하게 변하는 것을 막아 시스템을 안정시킵니다.
*   **효과**: 오버슈트를 줄이고 시스템의 안정성과 반응 속도를 향상시킵니다. 시스템이 급변하는 상황에 미리 대응할 수 있게 합니다.
*   **단점**:
    *   노이즈에 민감하게 반응하여 시스템이 불안정해질 수 있습니다. (노이즈는 오차의 변화율을 매우 크게 만들 수 있기 때문)

### 2.4. PID 게인 튜닝 (Tuning)
*   $K_p, K_i, K_d$ 세 가지 게인 값의 조합이 PID 제어기의 성능을 좌우합니다.
*   **튜닝 방법**: 지글러-니콜스 방법, 비례-적분-미분 이득 최적화(PIM) 방법, 시행착오(Trial and Error) 방법 등
*   **일반적인 튜닝 가이드라인**:
    1.  $K_p$만 사용하여 목표값 근처로 빠르게 도달하게 조절 (약간의 진동이나 오버슈트 허용)
    2.  $K_i$를 사용하여 정상 상태 오차를 제거하고 정확히 목표값에 도달하게 조절
    3.  $K_d$를 사용하여 오버슈트를 줄이고 시스템이 안정적으로 빠르게 수렴하게 조절

---

## **3. 자율주행에서의 PID 제어 적용**

### 3.1. 종방향 제어: 속도 제어
*   **목표**: 현재 차량 속도를 목표 속도(판단 모듈에서 전달)에 맞춥니다.
*   **오차**: $e(t) = 목표 속도 - 현재 차량 속도$
*   **제어량**: $u(t)$는 가속 페달 개도율 또는 브레이크 압력으로 변환됩니다.
*   **활용**: 어댑티브 크루즈 컨트롤(ACC)에서 앞차와의 간격 유지 및 속도 조절

### 3.2. 횡방향 제어: 조향 제어
*   **목표**: 현재 차량이 목표 경로(판단 모듈에서 전달)를 벗어나지 않도록 조향합니다.
*   **오차**:
    *   **CTE (Cross-Track Error)**: 현재 차량의 중심과 목표 경로 사이의 수직 거리.
    *   **Heading Error (방향 오차)**: 현재 차량의 주행 방향과 목표 경로의 방향 사이의 각도 차이.
*   **제어량**: $u(t)$는 스티어링 휠의 회전 각도로 변환됩니다.
*   **활용**: 차선 유지 보조(LKA), 자동 주차

### 3.3. 장점 및 단점
*   **장점**:
    *   **간단한 구현**: 비교적 적은 파라미터로 쉽게 구현할 수 있습니다.
    *   **강력한 성능**: 선형 시스템에서 뛰어난 성능을 보이며, 적절히 튜닝하면 비선형 시스템에도 적용 가능합니다.
    *   **높은 안정성**: 잘 튜닝하면 시스템을 안정적으로 유지합니다.
*   **단점**:
    *   **비선형 시스템 한계**: 차량 동역학은 비선형적이므로, 모든 조건에서 최적의 성능을 보장하기 어렵습니다.
    *   **게인 튜닝의 어려움**: 환경 변화(노면, 차량 속도, 적재량)에 따라 최적의 게인 값이 달라질 수 있으며, 실시간으로 변화하는 게인 값 튜닝은 어렵습니다.
    *   **제한적인 예측 능력**: 주로 현재 오차에 기반하므로, 미래 상황에 대한 능동적인 예측 능력이 부족합니다 (이를 보완하기 위해 모델 예측 제어(MPC) 등이 사용됨).

---

## **4. PID 제어 실습 예제: 간단한 목표 속도 추종 시뮬레이션**

여기서는 간단한 차량의 속도 제어를 PID 컨트롤러로 시뮬레이션하는 예제를 제공합니다. 텐서플로우와 파이토치는 기본적으로 딥러닝 프레임워크이지만, 이러한 수치 계산 및 시뮬레이션을 위한 텐서 연산을 효율적으로 수행할 수 있습니다. 여기서는 PID 제어의 각 구성 요소(P, I, D)를 텐서 연산을 통해 구현하고, 가상의 시스템(차량)이 목표 속도를 추종하는 과정을 시각화합니다.

### 4.1. 공통 PID 컨트롤러 클래스 (Python)

먼저, TensorFlow/PyTorch 예제에서 사용할 재사용 가능한 PID 컨트롤러 클래스를 정의합니다.

```python
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point, output_limits=(-1.0, 1.0), integral_limits=None, dt=0.1):
        """
        PID 컨트롤러 초기화.
        Args:
            Kp (float): 비례 게인 (Proportional gain)
            Ki (float): 적분 게인 (Integral gain)
            Kd (float): 미분 게인 (Derivative gain)
            set_point (float): 목표값 (Set point)
            output_limits (tuple): 제어 출력의 최소/최대 제한 (예: 가속/브레이크 %)
            integral_limits (tuple): 적분 오차의 최소/최대 제한 (Integral wind-up 방지)
            dt (float): 제어 주기 (Delta time, 초 단위)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.output_limits = output_limits
        self.integral_limits = integral_limits if integral_limits else (self.output_limits[0] * 10, self.output_limits[1] * 10)
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def calculate_control_signal(self, current_value):
        """
        현재값을 기반으로 제어 신호 계산.
        Args:
            current_value (float): 현재 측정값 (Process variable)
        Returns:
            float: 계산된 제어 신호 (Control output)
        """
        error = self.set_point - current_value

        # P (Proportional) 항
        p_term = self.Kp * error

        # I (Integral) 항
        self.integral += error * self.dt
        # Integral wind-up 방지
        self.integral = np.clip(self.integral, self.integral_limits[0], self.integral_limits[1])
        i_term = self.Ki * self.integral

        # D (Derivative) 항
        derivative = (error - self.prev_error) / self.dt
        d_term = self.Kd * derivative

        # 총 제어 신호
        control_signal = p_term + i_term + d_term

        # 출력 제한
        control_signal = np.clip(control_signal, self.output_limits[0], self.output_limits[1])
        
        # 다음 스텝을 위해 값 업데이트
        self.prev_error = error
        self.prev_output = control_signal

        return control_signal
    
    def reset(self):
        """PID 컨트롤러 상태 초기화"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

```

### 4.2. 가상 시스템 모델 (Simulated System Model)

PID 컨트롤러가 제어할 간단한 가상 시스템(예: 모터로 구동되는 차량)을 정의합니다. 여기서는 목표 속도를 추종하는 시스템을 시뮬레이션합니다. 제어 신호($u(t)$)는 차량의 가속/감속에 영향을 줍니다.

```python
class SimulatedVehicle:
    def __init__(self, initial_speed=0.0, max_acceleration=5.0, friction_coefficient=0.1, dt=0.1):
        """
        가상의 차량 시스템 초기화.
        Args:
            initial_speed (float): 초기 속도 (m/s)
            max_acceleration (float): 최대 가속도 (m/s^2)
            friction_coefficient (float): 마찰 계수 (속도에 비례하는 저항)
            dt (float): 시뮬레이션 시간 간격 (초)
        """
        self.speed = initial_speed
        self.max_acceleration = max_acceleration
        self.friction_coefficient = friction_coefficient
        self.dt = dt
    
    def update(self, control_signal):
        """
        제어 신호에 따라 차량의 속도를 업데이트합니다.
        Args:
            control_signal (float): -1.0(최대 감속) ~ 1.0(최대 가속) 범위의 제어 신호.
        """
        # 제어 신호를 실제 가속도로 변환 (max_acceleration에 비례)
        target_acceleration = control_signal * self.max_acceleration
        
        # 마찰(공기 저항 등)에 의한 감속 (속도에 비례한다고 가정)
        friction_deceleration = self.friction_coefficient * self.speed
        
        # 순 가속도 계산
        net_acceleration = target_acceleration - friction_deceleration
        
        # 속도 업데이트
        self.speed += net_acceleration * self.dt
        
        # 속도는 음수가 될 수 없음
        self.speed = max(0.0, self.speed)
        
        return self.speed

```

### 4.3. TensorFlow를 활용한 PID 시뮬레이션

TensorFlow를 사용하여 `SimulatedVehicle`의 상태와 `PIDController`의 계산을 텐서 연산으로 시뮬레이션할 수 있습니다. 이는 복잡한 시스템의 제어 학습 환경을 구축할 때 유용합니다.

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. PID 컨트롤러 설정
Kp_tf, Ki_tf, Kd_tf = 0.5, 0.1, 0.05  # PID 게인
set_point_tf = 30.0                  # 목표 속도 (m/s)
dt_tf = 0.1                          # 시뮬레이션 시간 간격 (초)
output_limits_tf = (-1.0, 1.0)       # 제어 신호 범위 (가속/감속%)

# TensorFlow 버전 PID 컨트롤러 클래스
class TFPIDController(PIDController):
    def __init__(self, Kp, Ki, Kd, set_point, output_limits, integral_limits, dt):
        super().__init__(Kp, Ki, Kd, set_point, output_limits, integral_limits, dt)
        self.integral_tf = tf.Variable(0.0, dtype=tf.float32)
        self.prev_error_tf = tf.Variable(0.0, dtype=tf.float32)

    @tf.function
    def calculate_control_signal_tf(self, current_value_tf):
        """TensorFlow 연산을 사용하여 제어 신호 계산"""
        error_tf = self.set_point - current_value_tf

        # P 항
        p_term_tf = self.Kp * error_tf

        # I 항
        self.integral_tf.assign_add(error_tf * self.dt)
        self.integral_tf.assign(tf.clip_by_value(self.integral_tf, self.integral_limits[0], self.integral_limits[1]))
        i_term_tf = self.Ki * self.integral_tf

        # D 항
        derivative_tf = (error_tf - self.prev_error_tf) / self.dt
        d_term_tf = self.Kd * derivative_tf

        control_signal_tf = p_term_tf + i_term_tf + d_term_tf
        control_signal_tf = tf.clip_by_value(control_signal_tf, self.output_limits[0], self.output_limits[1])
        
        # 다음 스텝을 위해 값 업데이트
        self.prev_error_tf.assign(error_tf)

        return control_signal_tf

# 2. 가상 차량 시스템 설정
initial_speed_tf = tf.constant(0.0, dtype=tf.float32)
vehicle_tf = SimulatedVehicle(initial_speed_tf.numpy(), max_acceleration=5.0, friction_coefficient=0.1, dt=dt_tf) # Vehicle 클래스는 numpy 기반이므로 초기화 시 numpy 값 전달
current_speed_tf = tf.Variable(initial_speed_tf, dtype=tf.float32) # TensorFlow에서 차량의 속도를 추적하기 위한 변수

# 3. PID 컨트롤러 인스턴스 생성
pid_controller_tf = TFPIDController(Kp_tf, Ki_tf, Kd_tf, set_point_tf, output_limits_tf, dt=dt_tf)

# 4. 시뮬레이션 실행
time_steps_tf = 200 # 시뮬레이션 스텝 수
speeds_tf = [current_speed_tf.numpy()]
control_signals_tf = [0.0]

print("\n--- TensorFlow PID 시뮬레이션 시작 ---")
for t in range(time_steps_tf):
    # 제어 신호 계산
    control_signal = pid_controller_tf.calculate_control_signal_tf(current_speed_tf)
    control_signals_tf.append(control_signal.numpy())

    # 차량 속도 업데이트 (차량 모델은 numpy 기반으로 동작)
    current_speed_np = vehicle_tf.update(control_signal.numpy())
    current_speed_tf.assign(current_speed_np) # TensorFlow 변수에 업데이트된 속도 할당

    speeds_tf.append(current_speed_tf.numpy())

# 5. 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, time_steps_tf * dt_tf + dt_tf, dt_tf), speeds_tf, label='Vehicle Speed (m/s)')
plt.axhline(set_point_tf, color='r', linestyle='--', label='Set Point (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('TensorFlow PID Control Simulation for Vehicle Speed')
plt.grid(True)
plt.legend()
plt.show()

print("--- TensorFlow PID 시뮬레이션 완료 ---")
```

### 4.4. PyTorch를 활용한 PID 시뮬레이션

PyTorch에서도 마찬가지로 텐서 연산을 통해 PID 제어 시뮬레이션을 구현할 수 있습니다.

```python
import torch
import matplotlib.pyplot as plt

# 1. PID 컨트롤러 설정
Kp_pt, Ki_pt, Kd_pt = 0.5, 0.1, 0.05  # PID 게인
set_point_pt = 30.0                  # 목표 속도 (m/s)
dt_pt = 0.1                          # 시뮬레이션 시간 간격 (초)
output_limits_pt = (-1.0, 1.0)       # 제어 신호 범위 (가속/감속%)

# PyTorch 버전 PID 컨트롤러 클래스
class PyTorchPIDController(PIDController):
    def __init__(self, Kp, Ki, Kd, set_point, output_limits, integral_limits, dt):
        super().__init__(Kp, Ki, Kd, set_point, output_limits, integral_limits, dt)
        self.integral_pt = torch.tensor(0.0, dtype=torch.float32)
        self.prev_error_pt = torch.tensor(0.0, dtype=torch.float32)

    def calculate_control_signal_pt(self, current_value_pt):
        """PyTorch 연산을 사용하여 제어 신호 계산"""
        error_pt = self.set_point - current_value_pt

        # P 항
        p_term_pt = self.Kp * error_pt

        # I 항
        self.integral_pt = self.integral_pt + error_pt * self.dt
        self.integral_pt = torch.clamp(self.integral_pt, self.integral_limits[0], self.integral_limits[1])
        i_term_pt = self.Ki * self.integral_pt

        # D 항
        derivative_pt = (error_pt - self.prev_error_pt) / self.dt
        d_term_pt = self.Kd * derivative_pt

        control_signal_pt = p_term_pt + i_term_pt + d_term_pt
        control_signal_pt = torch.clamp(control_signal_pt, self.output_limits[0], self.output_limits[1])
        
        # 다음 스텝을 위해 값 업데이트
        self.prev_error_pt = error_pt

        return control_signal_pt

# 2. 가상 차량 시스템 설정
initial_speed_pt = torch.tensor(0.0, dtype=torch.float32)
vehicle_pt = SimulatedVehicle(initial_speed_pt.item(), max_acceleration=5.0, friction_coefficient=0.1, dt=dt_pt) # Vehicle 클래스는 numpy 기반이므로 item()으로 numpy 값 전달
current_speed_pt = torch.tensor(initial_speed_pt, dtype=torch.float32) # PyTorch에서 차량의 속도를 추적하기 위한 변수

# 3. PID 컨트롤러 인스턴스 생성
pid_controller_pt = PyTorchPIDController(Kp_pt, Ki_pt, Kd_pt, set_point_pt, output_limits_pt, dt=dt_pt)

# 4. 시뮬레이션 실행
time_steps_pt = 200 # 시뮬레이션 스텝 수
speeds_pt = [current_speed_pt.item()]
control_signals_pt = [0.0]

print("\n--- PyTorch PID 시뮬레이션 시작 ---")
for t in range(time_steps_pt):
    # 제어 신호 계산
    control_signal = pid_controller_pt.calculate_control_signal_pt(current_speed_pt)
    control_signals_pt.append(control_signal.item())

    # 차량 속도 업데이트 (차량 모델은 numpy 기반으로 동작)
    current_speed_np = vehicle_pt.update(control_signal.item())
    current_speed_pt = torch.tensor(current_speed_np, dtype=torch.float32) # PyTorch 변수에 업데이트된 속도 할당

    speeds_pt.append(current_speed_pt.item())

# 5. 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, time_steps_pt * dt_pt + dt_pt, dt_pt), speeds_pt, label='Vehicle Speed (m/s)')
plt.axhline(set_point_pt, color='r', linestyle='--', label='Set Point (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('PyTorch PID Control Simulation for Vehicle Speed')
plt.grid(True)
plt.legend()
plt.show()

print("--- PyTorch PID 시뮬레이션 완료 ---")
```

---

## **5. 실습 과제 및 응용 아이디어**

1.  **PID 게인 튜닝**: 위에 제시된 `Kp`, `Ki`, `Kd` 값을 다양하게 변경하여 시뮬레이션 결과를 비교하고, 각 게인 값의 변화가 시스템의 반응(오버슈트, 안정성, 정상 상태 오차)에 어떤 영향을 미치는지 분석해 보세요. (예: `Kp`만 높게, `Ki`만 높게, `Kd`만 높게)
2.  **가상 시스템 개선**: `SimulatedVehicle` 클래스에 다음과 같은 요소를 추가하여 좀 더 현실적인 시뮬레이션을 구현해 보세요.
    *   **최대 속도 제한**: 차량의 최대 속도($V_{max}$)를 추가하여 그 이상 가속되지 않도록 합니다.
    *   **지연(Lag)**: 제어 신호가 액추에이터에 전달되어 실제 가속/감속이 시작되기까지의 시간 지연을 추가합니다.
    *   **외란(Disturbance)**: 임의의 외부 힘(예: 바람, 노면 경사)을 추가하여 PID 제어기가 어떻게 반응하는지 관찰합니다.
3.  **횡방향 제어 시뮬레이션**: 목표 차선 중앙을 추종하는 차량의 횡방향 움직임을 시뮬레이션하는 PID 제어기를 구현해 보세요. 오차는 차선 중앙으로부터의 CTE가 됩니다. (라즈베리파이 기반의 미니카를 사용한다면 실제로 조향 모터를 제어할 수 있습니다!)
4.  **강화 학습 기반 PID 튜닝**: TensorFlow나 PyTorch의 자동 미분 기능을 활용하여, 시뮬레이션 환경에서 에이전트(강화 학습 모델)가 PID 게인($K_p, K_i, K_d$)을 자동으로 학습하고 최적화하도록 시도해 볼 수 있습니다. (이는 심화 주제이며, 보상 함수 설계가 핵심입니다.)

---

스카이님, PID 제어는 자율주행의 '제어' 단계에서 매우 기본적인 동시에 필수적인 기술입니다. 특히 비전공자 학생들에게 각 요소의 직관적인 의미와 함께 시뮬레이션 코드를 통해 시각적인 결과를 보여주는 것이 이해도를 높이는 데 매우 효과적일 것입니다.

이 강의 자료가 스카이님의 모빌리티 AI 강의 준비에 큰 도움이 되기를 바랍니다! 

참고 자료 

[1] wikidocs.net - 🧩 Chapter 21. ⚙️ PID 제어를 이용한 자율주행 제어 구현 (https://wikidocs.net/300880)
[2] tutorials.pytorch.kr - 예제로 배우는 파이토치(PyTorch) (https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html)
[3] saint-swithins-day.tistory.com - 특강 5,6,7일차 : JETBOT 딥러닝 자율주행 따라하기 (https://saint-swithins-day.tistory.com/44)
[4] blog.naver.com - PID 콘트롤 : 네이버 블로그 (https://blog.naver.com/tramper2/223319733303?viewType=pc)
[5] www.acusys.co.kr - C로 구현한 PID(비례,적분,미분)제어 알고리즘 예 (https://www.acusys.co.kr/?p=2444)