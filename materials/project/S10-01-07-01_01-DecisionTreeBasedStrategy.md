---
layout: page
title:  "Decision Tree 기반 판단 시스템"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-07-01_01-DecisionTreeBasedStrategy
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


<div class="insert-image" style="text-align: center;">
    <img style="width: 400px;" src="/assets/img/PagePreparing.png">
</div>





네, 스카이님! Decision Tree 기반 자율주행 판단 시스템 실습 코드 이어서 설명해 드리겠습니다.

```python
    # 2. 속도 명령 결정
    speed_command = []
    for idx, row in df.iterrows():
        # 빨간불 또는 매우 가까운 거리에 차량이 있으면 긴급 제동
        if (row['traffic_light'] == 'red' and row['front_vehicle_distance'] < 10) or \
           (row['front_vehicle_distance'] < 5 and row['front_vehicle_rel_speed'] < 0): # 5m 이내이고 앞차가 감속 중
            speed_command.append('emergency_brake')
        # 노란불 또는 가까운 거리에 차량이 있으면 제동
        elif (row['traffic_light'] == 'yellow' and row['current_speed'] > 5) or \
             (row['front_vehicle_distance'] < 15 and row['front_vehicle_rel_speed'] < -2): # 15m 이내이고 앞차가 꽤 감속 중
            speed_command.append('brake')
        # 앞차와의 거리가 충분하고 상대 속도가 음수이면 감속
        elif (row['front_vehicle_distance'] < 30 and row['front_vehicle_rel_speed'] < -0.5) or \
             (row['traffic_light'] == 'yellow' and row['current_speed'] > 10):
            speed_command.append('brake') # 상황에 따라 'slow_down' 추가 가능
        # 앞차와의 거리가 충분하고 신호등 초록/없음, 현재 속도가 낮으면 가속
        elif (row['front_vehicle_distance'] > 30 and (row['traffic_light'] in ['green', 'none']) and row['current_speed'] < 25):
            speed_command.append('accelerate')
        else: # 그 외의 경우 속도 유지
            speed_command.append('maintain')
            
    df['steering_command'] = steering_command
    df['speed_command'] = speed_command
    
    # 클래스 레이블 인코딩
    df['steering_command_encoded'] = df['steering_command'].astype('category').cat.codes
    df['speed_command_encoded'] = df['speed_command'].astype('category').cat.codes
    
    return df

# 데이터셋 생성
dataset_df = generate_autonomous_driving_dataset(n_samples=10000)
print("--- 생성된 데이터셋 상위 5개 ---")
print(dataset_df.head())

# --- 2. TensorFlow Decision Forests 모델 학습 (steering command) ---
print("\n--- Steering Command Decision Tree 학습 ---")
# 특성(features)과 레이블(label) 분리
X_steer = dataset_df.drop(['steering_command', 'speed_command', 'speed_command_encoded', 'steering_command_encoded'], axis=1)
y_steer = dataset_df['steering_command_encoded']

# 학습 데이터와 테스트 데이터 분리
X_train_steer, X_test_steer, y_train_steer, y_test_steer = train_test_split(X_steer, y_steer, test_size=0.2, random_state=42)

# tfdf에 맞게 데이터 변환
# 범주형 특성으로 인식하도록 pandas dataframe의 dtype을 'object'로 변경 (tfdf가 자동 인식)
for col in ['traffic_light']:
    X_train_steer[col] = X_train_steer[col].astype(str)
    X_test_steer[col] = X_test_steer[col].astype(str)

train_ds_steer = tfdf.keras.pd_dataframe_to_tf_dataset(X_train_steer, label="steering_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)
test_ds_steer = tfdf.keras.pd_dataframe_to_tf_dataset(X_test_steer, label="steering_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)

# Random Forest 모델 생성
model_steer = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)

# 모델 학습
model_steer.fit(train_ds_steer)

# 모델 평가
metrics = model_steer.evaluate(test_ds_steer, return_dict=True)
print(f"Steering Model Accuracy: {metrics['accuracy']:.4f}")

# 예측 예시
sample_input_steer = X_test_steer.iloc[0:1] # 테스트셋의 첫 번째 샘플
predictions_steer = model_steer.predict(tfdf.keras.pd_dataframe_to_tf_dataset(sample_input_steer, task=tfdf.keras.Task.CLASSIFICATION))
predicted_class_steer = np.argmax(predictions_steer, axis=1)[0]
actual_command_steer = dataset_df['steering_command'].astype('category').cat.categories[y_test_steer.iloc[0]]
predicted_command_steer = dataset_df['steering_command'].astype('category').cat.categories[predicted_class_steer]
print(f"\nSteering: 실제 행동: {actual_command_steer}, 예측 행동: {predicted_command_steer}")


# --- 3. TensorFlow Decision Forests 모델 학습 (speed command) ---
print("\n--- Speed Command Decision Tree 학습 ---")
X_speed = dataset_df.drop(['steering_command', 'speed_command', 'speed_command_encoded', 'steering_command_encoded'], axis=1)
y_speed = dataset_df['speed_command_encoded']

X_train_speed, X_test_speed, y_train_speed, y_test_speed = train_test_split(X_speed, y_speed, test_size=0.2, random_state=42)

for col in ['traffic_light']:
    X_train_speed[col] = X_train_speed[col].astype(str)
    X_test_speed[col] = X_test_speed[col].astype(str)

train_ds_speed = tfdf.keras.pd_dataframe_to_tf_dataset(X_train_speed, label="speed_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)
test_ds_speed = tfdf.keras.pd_dataframe_to_tf_dataset(X_test_speed, label="speed_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)

model_speed = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
model_speed.fit(train_ds_speed)

metrics = model_speed.evaluate(test_ds_speed, return_dict=True)
print(f"Speed Model Accuracy: {metrics['accuracy']:.4f}")

sample_input_speed = X_test_speed.iloc[0:1]
predictions_speed = model_speed.predict(tfdf.keras.pd_dataframe_to_tf_dataset(sample_input_speed, task=tfdf.keras.Task.CLASSIFICATION))
predicted_class_speed = np.argmax(predictions_speed, axis=1)[0]
actual_command_speed = dataset_df['speed_command'].astype('category').cat.categories[y_test_speed.iloc[0]]
predicted_command_speed = dataset_df['speed_command'].astype('category').cat.categories[predicted_class_speed]
print(f"Speed: 실제 행동: {actual_command_speed}, 예측 행동: {predicted_command_speed}")


# --- 4. TensorFlow 모델을 라즈베리파이 시뮬레이터에 통합하는 예시 ---
# 이전 'IntegratedAutonomousVehicle' 클래스에서 dl_model 대신 Decision Tree 모델을 사용합니다.
# 인지 모듈의 출력이 Decision Tree의 입력 특성으로 직접 변환되어야 합니다.

# 현재는 RPi.GPIO가 import되어 있지 않으면 DummyGPIO가 사용됩니다.
# from common_modules import PIDController, RPiMotorController # 기존 common_modules 사용

class DecisionTreeAutonomousVehicle:
    def __init__(self, steer_model, speed_model, motor_pin_config, dt=0.1):
        self.motor_controller = RPiMotorController(**motor_pin_config)
        self.steer_model = steer_model
        self.speed_model = speed_model
        self.dt = dt

        self.steering_pid = PIDController(Kp=30.0, Ki=0.05, Kd=1.0, set_point=0.0, output_limits=(-1.0, 1.0), dt=dt)
        self.speed_pid = PIDController(Kp=10.0, Ki=0.2, Kd=0.8, set_point=0.0, output_limits=(-100.0, 100.0), dt=dt)

        self.current_speed_mps = 0.0 # 시뮬레이션용 현재 속도
        self.prev_frame_time = time.time()
        self.fps = 0

        # 행동 명령 인코딩/디코딩 정보 저장
        self.steering_categories = dataset_df['steering_command'].astype('category').cat.categories
        self.speed_categories = dataset_df['speed_command'].astype('category').cat.categories

    def _get_simulated_perception_data(self):
        """
        가상 인지 모듈로부터의 데이터 (실제는 카메라/DL 모델로부터)
        """
        lane_deviation = np.random.uniform(-1.0, 1.0)
        front_vehicle_distance = np.random.exponential(30)
        front_vehicle_rel_speed = np.random.normal(0, 5)
        traffic_light = np.random.choice(['red', 'yellow', 'green', 'none'], p=[0.2, 0.1, 0.4, 0.3])
        lane_change_possible_left = np.random.choice([True, False])
        lane_change_possible_right = np.random.choice([True, False])
        current_speed = self.current_speed_mps # 현재 시뮬레이션 속도 사용
        
        # 실제 시스템에서는 딥러닝 모델의 출력에서 이 값들을 추출
        return pd.DataFrame({
            'lane_deviation': [lane_deviation],
            'front_vehicle_distance': [front_vehicle_distance],
            'front_vehicle_rel_speed': [front_vehicle_rel_speed],
            'traffic_light': [traffic_light],
            'lane_change_possible_left': [lane_change_possible_left],
            'lane_change_possible_right': [lane_change_possible_right],
            'current_speed': [current_speed]
        })

    def _update_simulated_speed(self, speed_control_signal_output):
        """속도 제어기의 출력을 바탕으로 가상 속도 업데이트"""
        # 속도 제어기의 출력이 PWM 듀티 사이클 %라고 가정 (0-100)
        # 0%면 감속, 100%면 최대 가속 (간단한 모델)
        acceleration_effect = speed_control_signal_output / 50.0 - 1.0 # -1.0 ~ 1.0 가속력
        
        friction = self.current_speed_mps * 0.1 # 속도 비례 마찰
        net_acceleration = acceleration_effect - friction
        
        self.current_speed_mps += net_acceleration * self.dt
        self.current_speed_mps = max(0.0, self.current_speed_mps) # 속도는 음수 불가

    def _map_steering_action_to_error(self, action):
        """결정된 조향 행동을 PID 제어기가 이해하는 오차 값으로 매핑"""
        mapping = {
            'hard_left': 1.0,    # 우측으로 1.0 오차가 나야 좌회전
            'soft_left': 0.5,    # 우측으로 0.5 오차가 나야 좌회전
            'straight': 0.0,
            'soft_right': -0.5,  # 좌측으로 0.5 오차가 나야 우회전
            'hard_right': -1.0   # 좌측으로 1.0 오차가 나야 우회전
        }
        return mapping.get(action, 0.0)

    def _map_speed_action_to_target_speed(self, action):
        """결정된 속도 행동을 PID 제어기가 이해하는 목표 속도로 매핑"""
        mapping = {
            'emergency_brake': 0.0,
            'brake': 0.0,       # 실제 브레이크는 목표 속도 0으로 하고 PID로 서서히 감속
            'maintain': self.current_speed_mps, # 현재 속도 유지
            'accelerate': 15.0  # 가속 목표 속도 (m/s) (조절 필요)
        }
        # 특별히 정지/감속이 아니면 현재 속도를 유지하다가 가속 목표 도달
        if action == 'accelerate':
            return mapping['accelerate']
        elif action in ['emergency_brake', 'brake']:
            return mapping[action]
        else: # maintain 등은 PID가 현 속도 유지하도록 set_point를 현 속도로 설정
            return self.current_speed_mps
            
    def run_control_loop(self, total_duration_seconds=30):
        """
        Decision Tree 기반 자율주행 제어 루프 실행
        """
        num_steps = int(total_duration_seconds / self.dt)
        print(f"\n--- Decision Tree 기반 자율주행 시뮬레이션 시작 ({total_duration_seconds}초) ---")

        for step in range(num_steps):
            # 1. 인지 모듈로부터 데이터 획득 (여기서는 가상 데이터)
            perception_data = self._get_simulated_perception_data()
            
            # tfdf는 범주형 특성으로 인식하도록 str로 변환
            for col in ['traffic_light']:
                perception_data[col] = perception_data[col].astype(str)

            # 2. 판단 (Decision-making) - Decision Tree 모델 사용
            # TensorFlow Dataset으로 변환
            steer_ds = tfdf.keras.pd_dataframe_to_tf_dataset(perception_data, task=tfdf.keras.Task.CLASSIFICATION)
            speed_ds = tfdf.keras.pd_dataframe_to_tf_dataset(perception_data, task=tfdf.keras.Task.CLASSIFICATION)
            
            # 예측
            steer_preds = self.steer_model.predict(steer_ds, verbose=0)
            speed_preds = self.speed_model.predict(speed_ds, verbose=0)
            
            # 예측된 행동 디코딩
            predicted_steer_class_idx = np.argmax(steer_preds, axis=1)[0]
            predicted_speed_class_idx = np.argmax(speed_preds, axis=1)[0]
            
            action_steer = self.steering_categories[predicted_steer_class_idx]
            action_speed = self.speed_categories[predicted_speed_class_idx]

            # 3. 행동 명령을 PID 컨트롤러의 목표/오차로 매핑
            # 횡방향 PID: 목표 조향 오차 계산
            target_steer_error_for_pid = self._map_steering_action_to_error(action_steer)
            self.steering_pid.set_point = target_steer_error_for_pid
            
            # 종방향 PID: 목표 속도 설정
            target_speed_for_pid = self._map_speed_action_to_target_speed(action_speed)
            self.speed_pid.set_point = target_speed_for_pid

            # 4. 제어 (Control) - PID 컨트롤러 구동
            # 횡방향 제어: 현재 차선 오차 대신 PID 목표 오차를 기준으로 조향 제어 신호 계산
            # PID의 current_value는 실제 인지된 차선 오차가 되어야 합니다.
            # 여기서는 조향 행동에서 추출된 target_steer_error_for_pid를 PID의 current_value로 넣어 PID가 이를 0으로 만들도록 합니다.
            steering_control_signal = self.steering_pid.calculate_control_signal(target_steer_error_for_pid) # 현재 상태에 따른 피드백
            
            # 종방향 제어: 현재 속도를 기준으로 속도 제어 신호 계산
            speed_control_signal = self.speed_pid.calculate_control_signal(self.current_speed_mps)

            # 5. 모터 구동 명령으로 변환 및 하드웨어 제어
            base_speed_duty_cycle = np.clip(speed_control_signal, 0, 100) # 속도 제어기의 출력이 바로 PWM 듀티 사이클이 된다고 가정
            
            # 시뮬레이션: 모터 컨트롤러에 제어 신호 전달 (실제 하드웨어 작동)
            self.motor_controller.turn(steering_control_signal, base_speed_duty_cycle)
            
            # 6. 차량의 가상 속도 업데이트 (다음 스텝의 'current_speed_mps'로 사용)
            self._update_simulated_speed(base_speed_duty_cycle)

            # 7. 시뮬레이션 및 디버깅 정보 출력
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.prev_frame_time) if (current_time - self.prev_frame_time) > 0 else 0
            self.prev_frame_time = current_time

            print(f"Step {step+1}/{num_steps} | FPS: {self.fps:.1f} | Speed: {self.current_speed_mps:.2f} m/s ({self.current_speed_mps*3.6:.1f} km/h)")
            print(f"  DL Data -> Lane: {perception_data['lane_deviation'].iloc[0]:.2f}, FrontDist: {perception_data['front_vehicle_distance'].iloc[0]:.1f}, TL: {perception_data['traffic_light'].iloc[0]}")
            print(f"  Decision -> Steer: {action_steer} (Cmd={steering_control_signal:.2f}), Speed: {action_speed} (Cmd={speed_control_signal:.1f}%)")
            print("-" * 50)
            
            # 시뮬레이션 시간 지연 (실제 제어 주기)
            time.sleep(self.dt)
            
            # 시뮬레이션 종료 조건
            if self.current_speed_mps < 0.1 and target_speed_for_pid < 0.1 and action_speed == 'emergency_brake':
                print("차량이 완전히 정지했습니다. 시뮬레이션 종료.")
                break

        self.motor_controller.cleanup()
        print("\n--- Decision Tree 기반 자율주행 시뮬레이션 종료 ---")


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
    
    # 이전에 학습된 TensorFlow Decision Forest 모델 객체 (model_steer, model_speed)를 사용
    # 실제로는 .savedmodel 형식으로 저장 후 로드하여 사용
    # steer_model = tf.saved_model.load("steer_decision_tree_model_path")
    # speed_model = tf.saved_model.load("speed_decision_tree_model_path")
    
    # 이 예제에서는 위에서 학습된 model_steer, model_speed 객체를 직접 전달합니다.
    if 'model_steer' not in locals() or 'model_speed' not in locals():
        print("Decision Tree 모델이 학습되지 않았습니다. 먼저 스크립트를 실행하여 모델을 학습시키세요.")
        dataset_df = generate_autonomous_driving_dataset(n_samples=1000) # 가볍게 다시 생성
        X_steer = dataset_df.drop(['steering_command', 'speed_command', 'speed_command_encoded', 'steering_command_encoded'], axis=1)
        y_steer = dataset_df['steering_command_encoded']
        X_train_steer, _, y_train_steer, _ = train_test_split(X_steer, y_steer, test_size=0.2, random_state=42)
        for col in ['traffic_light']: X_train_steer[col] = X_train_steer[col].astype(str)
        train_ds_steer = tfdf.keras.pd_dataframe_to_tf_dataset(X_train_steer, label="steering_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)
        model_steer = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
        model_steer.fit(train_ds_steer)

        X_speed = dataset_df.drop(['steering_command', 'speed_command', 'speed_command_encoded', 'steering_command_encoded'], axis=1)
        y_speed = dataset_df['speed_command_encoded']
        X_train_speed, _, y_train_speed, _ = train_test_split(X_speed, y_speed, test_size=0.2, random_state=42)
        for col in ['traffic_light']: X_train_speed[col] = X_train_speed[col].astype(str)
        train_ds_speed = tfdf.keras.pd_dataframe_to_tf_dataset(X_train_speed, label="speed_command_encoded", task=tfdf.keras.Task.CLASSIFICATION)
        model_speed = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
        model_speed.fit(train_ds_speed)
        print("--- 모델 학습 완료 ---")

    decision_tree_vehicle_tf = DecisionTreeAutonomousVehicle(
        steer_model=model_steer,
        speed_model=model_speed,
        motor_pin_config=motor_pin_config,
        dt=0.1
    )
    decision_tree_vehicle_tf.run_control_loop(total_duration_seconds=30)
```

## **4. 실습 코드: PyTorch 버전**

PyTorch 자체에는 Decision Tree 모델이 내장되어 있지 않으므로, `scikit-learn` 라이브러리의 Decision Tree를 사용하고, PyTorch 모델과 유사하게 클래스를 구성하여 연동하는 방법을 보여드립니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # PyTorch는 sklearn 활용
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder # 범주형 데이터 인코딩

# 시드 설정
np.random.seed(42)

# --- 1. 자율주행 판단을 위한 가상 데이터셋 생성 (TensorFlow 버전과 동일) ---
# generate_autonomous_driving_dataset 함수 재사용
# (생성 코드는 위 TensorFlow 섹션 참조)

# 데이터셋 생성 (TensorFlow 버전과 동일하게)
dataset_df_pt = generate_autonomous_driving_dataset(n_samples=10000)

print("\n--- PyTorch (Scikit-learn) 기반 Decision Tree 학습 ---")
# 특성(features)과 레이블(label) 분리
X_pt = dataset_df_pt.drop(['steering_command', 'speed_command', 'steering_command_encoded', 'speed_command_encoded'], axis=1)
y_steer_pt = dataset_df_pt['steering_command_encoded']
y_speed_pt = dataset_df_pt['speed_command_encoded']

# 범주형 특성 인코딩 (scikit-learn Decision Tree는 숫자형 입력만 받음)
# Pandas의 one-hot encoding 또는 LabelEncoder 사용
# 여기서는 편의를 위해 직접 LabelEncoder 적용 (범주 수가 적고 Tree 모델이라 가능)
le_traffic_light = LabelEncoder()
X_pt['traffic_light_encoded'] = le_traffic_light.fit_transform(X_pt['traffic_light'])
X_pt = X_pt.drop('traffic_light', axis=1) # 원본 컬럼 제거

# 학습 데이터와 테스트 데이터 분리
X_train_steer_pt, X_test_steer_pt, y_train_steer_pt, y_test_steer_pt = train_test_split(X_pt, y_steer_pt, test_size=0.2, random_state=42)
X_train_speed_pt, X_test_speed_pt, y_train_speed_pt, y_test_speed_pt = train_test_split(X_pt, y_speed_pt, test_size=0.2, random_state=42)

# Decision Tree Classifier 모델 생성 및 학습 (steering command)
model_steer_pt = DecisionTreeClassifier(random_state=42, max_depth=10) # max_depth 제한
model_steer_pt.fit(X_train_steer_pt, y_train_steer_pt)

# 모델 평가
steer_accuracy = model_steer_pt.score(X_test_steer_pt, y_test_steer_pt)
print(f"Steering Model Accuracy (PyTorch/Sklearn): {steer_accuracy:.4f}")

# 예측 예시
sample_input_steer_pt = X_test_steer_pt.iloc[0:1]
predicted_class_steer_pt = model_steer_pt.predict(sample_input_steer_pt)[0]
actual_command_steer_pt = dataset_df_pt['steering_command'].astype('category').cat.categories[y_test_steer_pt.iloc[0]]
predicted_command_steer_pt = dataset_df_pt['steering_command'].astype('category').cat.categories[predicted_class_steer_pt]
print(f"\nSteering (PyTorch/Sklearn): 실제 행동: {actual_command_steer_pt}, 예측 행동: {predicted_command_steer_pt}")


# Decision Tree Classifier 모델 생성 및 학습 (speed command)
model_speed_pt = DecisionTreeClassifier(random_state=42, max_depth=10)
model_speed_pt.fit(X_train_speed_pt, y_train_speed_pt)

# 모델 평가
speed_accuracy = model_speed_pt.score(X_test_speed_pt, y_test_speed_pt)
print(f"Speed Model Accuracy (PyTorch/Sklearn): {speed_accuracy:.4f}")

sample_input_speed_pt = X_test_speed_pt.iloc[0:1]
predicted_class_speed_pt = model_speed_pt.predict(sample_input_speed_pt)[0]
actual_command_speed_pt = dataset_df_pt['speed_command'].astype('category').cat.categories[y_test_speed_pt.iloc[0]]
predicted_command_speed_pt = dataset_df_pt['speed_command'].astype('category').cat.categories[predicted_class_speed_pt]
print(f"Speed (PyTorch/Sklearn): 실제 행동: {actual_command_speed_pt}, 예측 행동: {predicted_command_speed_pt}")


# --- 2. PyTorch (Sklearn) 모델을 라즈베리파이 시뮬레이터에 통합하는 예시 ---
# TensorFlow 버전의 DecisionTreeAutonomousVehicle 클래스를 PyTorch/Sklearn 모델에 맞게 수정
# 인지 모듈의 출력이 Decision Tree의 입력 특성으로 직접 변환되어야 합니다.

# 현재는 RPi.GPIO가 import되어 있지 않으면 DummyGPIO가 사용됩니다.
# from common_modules import PIDController, RPiMotorController # 기존 common_modules 사용

class SklearnDecisionTreeAutonomousVehicle(DecisionTreeAutonomousVehicle): # 상속하여 오버라이드
    def __init__(self, steer_model, speed_model, motor_pin_config, dt=0.1, le_traffic_light=None):
        super().__init__(steer_model, speed_model, motor_pin_config, dt)
        self.le_traffic_light = le_traffic_light
        print("Using Scikit-learn Decision Trees for Autonomous Vehicle control.")

    def run_control_loop(self, total_duration_seconds=30):
        num_steps = int(total_duration_seconds / self.dt)
        print(f"\n--- Scikit-learn Decision Tree 기반 자율주행 시뮬레이션 시작 ({total_duration_seconds}초) ---")

        for step in range(num_steps):
            # 1. 인지 모듈로부터 데이터 획득 (여기서는 가상 데이터)
            perception_data_df = self._get_simulated_perception_data()
            
            # Scikit-learn 모델에 맞게 범주형 특성 인코딩
            perception_data_df['traffic_light_encoded'] = self.le_traffic_light.transform(perception_data_df['traffic_light'])
            perception_data_processed = perception_data_df.drop('traffic_light', axis=1)

            # 2. 판단 (Decision-making) - Decision Tree 모델 사용 (Sklearn)
            action_steer_idx = self.steer_model.predict(perception_data_processed)[0]
            action_speed_idx = self.speed_model.predict(perception_data_processed)[0]
            
            action_steer = self.steering_categories[action_steer_idx]
            action_speed = self.speed_categories[action_speed_idx]

            # 3. 행동 명령을 PID 컨트롤러의 목표/오차로 매핑
            target_steer_error_for_pid = self._map_steering_action_to_error(action_steer)
            self.steering_pid.set_point = target_steer_error_for_pid
            
            target_speed_for_pid = self._map_speed_action_to_target_speed(action_speed)
            self.speed_pid.set_point = target_speed_for_pid

            # 4. 제어 (Control) - PID 컨트롤러 구동
            steering_control_signal = self.steering_pid.calculate_control_signal(target_steer_error_for_pid) 
            speed_control_signal = self.speed_pid.calculate_control_signal(self.current_speed_mps)

            # 5. 모터 구동 명령으로 변환 및 하드웨어 제어
            base_speed_duty_cycle = np.clip(speed_control_signal, 0, 100) 
            self.motor_controller.turn(steering_control_signal, base_speed_duty_cycle)
            
            # 6. 차량의 가상 속도 업데이트
            self._update_simulated_speed(base_speed_duty_cycle)

            # 7. 시뮬레이션 및 디버깅 정보 출력
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.prev_frame_time) if (current_time - self.prev_frame_time) > 0 else 0
            self.prev_frame_time = current_time

            print(f"Step {step+1}/{num_steps} | FPS: {self.fps:.1f} | Speed: {self.current_speed_mps:.2f} m/s ({self.current_speed_mps*3.6:.1f} km/h)")
            print(f"  DL Data -> Lane: {perception_data_df['lane_deviation'].iloc[0]:.2f}, FrontDist: {perception_data_df['front_vehicle_distance'].iloc[0]:.1f}, TL: {perception_data_df['traffic_light'].iloc[0]}")
            print(f"  Decision -> Steer: {action_steer} (Cmd={steering_control_signal:.2f}), Speed: {action_speed} (Cmd={speed_control_signal:.1f}%)")
            print("-" * 50)
            
            time.sleep(self.dt)
            
            if self.current_speed_mps < 0.1 and target_speed_for_pid < 0.1 and action_speed == 'emergency_brake':
                print("차량이 완전히 정지했습니다. 시뮬레이션 종료.")
                break

        self.motor_controller.cleanup()
        print("\n--- Scikit-learn Decision Tree 기반 자율주행 시뮬레이션 종료 ---")


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
    
    # 이전에 학습된 Scikit-learn Decision Tree 모델 객체를 사용
    if 'model_steer_pt' not in locals() or 'model_speed_pt' not in locals():
        print("Scikit-learn Decision Tree 모델이 학습되지 않았습니다. 먼저 스크립트를 실행하여 모델을 학습시키세요.")
        dataset_df_pt = generate_autonomous_driving_dataset(n_samples=1000) # 가볍게 다시 생성
        X_pt = dataset_df_pt.drop(['steering_command', 'speed_command', 'steering_command_encoded', 'speed_command_encoded'], axis=1)
        y_steer_pt = dataset_df_pt['steering_command_encoded']
        y_speed_pt = dataset_df_pt['speed_command_encoded']
        
        le_traffic_light = LabelEncoder()
        X_pt['traffic_light_encoded'] = le_traffic_light.fit_transform(X_pt['traffic_light'])
        X_pt = X_pt.drop('traffic_light', axis=1)

        X_train_steer_pt, _, y_train_steer_pt, _ = train_test_split(X_pt, y_steer_pt, test_size=0.2, random_state=42)
        X_train_speed_pt, _, y_train_speed_pt, _ = train_test_split(X_pt, y_speed_pt, test_size=0.2, random_state=42)

        model_steer_pt = DecisionTreeClassifier(random_state=42, max_depth=10)
        model_steer_pt.fit(X_train_steer_pt, y_train_steer_pt)
        
        model_speed_pt = DecisionTreeClassifier(random_state=42, max_depth=10)
        model_speed_pt.fit(X_train_speed_pt, y_train_speed_pt)
        print("--- 모델 학습 완료 ---")

    # Decision Tree 기반 자율주행 시스템 (PyTorch/Sklearn 버전) 실행
    dt_vehicle_pt = SklearnDecisionTreeAutonomousVehicle(
        steer_model=model_steer_pt,
        speed_model=model_speed_pt,
        motor_pin_config=motor_pin_config,
        dt=0.1,
        le_traffic_light=le_traffic_light # LabelEncoder 전달
    )
    dt_vehicle_pt.run_control_loop(total_duration_seconds=30)
```

## **5. 학습 포인트 및 응용 아이디어**

*   **규칙 기반의 자동화**: 이 예제는 초기 규칙 기반으로 생성된 데이터를 Decision Tree가 학습하여, 그 규칙을 '자동으로' 재현하는 과정을 보여줍니다. 복잡한 `if-elif-else` 구조를 Decision Tree 하나로 압축할 수 있습니다.
*   **쉬운 확장성**: 새로운 센서(예: 초음파 거리)나 새로운 판단 기준(예: 요철 감지)이 추가될 때, Decision Tree의 특성으로 추가하고 데이터를 다시 학습시켜 모델을 쉽게 업데이트할 수 있습니다.
*   **직관적인 시각화**: `sklearn.tree.export_graphviz` (scikit-learn)나 `tfdf.model_to_graphviz` (TensorFlow Decision Forests)를 사용하여 학습된 Decision Tree의 구조를 시각화하여 학생들이 판단 과정의 투명성을 직접 확인할 수 있습니다.
*   **제한점**: 데이터셋의 품질에 매우 의존적이며, 학습 데이터에 없는 복잡하고 미묘한 상황에는 대응하기 어렵습니다. 또한, Decision Tree는 작은 변화에도 구조가 크게 달라질 수 있어 안정성이 떨어질 수 있습니다 (이를 보완하기 위해 Random Forest와 같은 앙상블 모델을 사용합니다).

스카이님, 이 Decision Tree 기반 판단 시스템은 자율주행 판단 모듈의 또 다른 중요한 접근 방식을 보여주며, 학생들에게 규칙 기반 시스템과 머신러닝 모델의 연계 가능성을 탐색하게 할 수 있는 좋은 추가 자료가 될 것입니다.