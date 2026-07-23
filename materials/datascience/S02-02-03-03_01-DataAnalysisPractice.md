---
layout: page
title:  "[실습] 공정 데이터 기반 품질 영향 요인 분석 및 인사이트 도출"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S02-02-03-03_01-DataAnalysisPractice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 실습 개요

- **실습 주제:** 사출 성형 공정 데이터를 활용한 품질(불량 발생) 주요 영향 요인 규명 및 개선 인사이트 도출
- **실습 목적:**
    1. 시계열 센서 데이터와 공정 이벤트(4M/품질) 데이터를 결합(Join)하는 방식을 이해한다.
    2. 단순 통계 및 탐색적 데이터 분석(EDA)을 통해 불량과 상관관계가 높은 공정 파라미터를 발굴한다.
    3. 현장에 적용 가능한 '데이터 기반 공정 최적화 조건(Recipe)'을 도출하고 의사결정 보고서를 작성한다.



## 2. 실습 준비물 및 환경

- **실습 도구:** Python (Google Colab) 또는 Excel (선택 가능)
- **필수 라이브러리 (Python 진행 시):** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **제공 데이터 세트 (2개 CSV 파일):**
    1. `sensor_timeseries.csv` (1초 단위 설비 센서 수집 데이터)
    2. `production_logs.csv` (Lot 단위 생산/품질 및 4M 데이터)



## 3. 실습 데이터 세트 구조

- **`sensor_timeseries.csv` (Level 1~2 시계열 센서)**

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">컬럼명</th>
        <th style="width: 100px;">데이터 타입</th>
        <th style="width: 300px;">설명</th>
        <th style="width: 300px;">비고</th>
    </thead>
    <tbody>
        <tr><td class="td-rowheader">timestamp</td><td>Datetime</td><td>센서 측정 시각 (1초 간격)</td><td>YYYY-MM-DD HH:MM:SS</td></tr>
        <tr><td class="td-rowheader">machine_id</td><td>String</td><td>설비 식별자</td><td> 예: 'PRESS_01'</td></tr>
        <tr><td class="td-rowheader">temp_nozzle</td><td>Float</td><td>노즐 온도 (℃)</td><td>정상 범위: 220 ~ 240</td></tr>
        <tr><td class="td-rowheader">pressure_injection</td><td>Float</td><td>사출 압력 (bar)</td><td>정상 범위: 80 ~ 100</td></tr>
        <tr><td class="td-rowheader">cooling_time</td><td>Float</td><td>냉각 시간 (sec)</td><td>설정값</td></tr>
    </tbody>
</table>
</div>

- **`production_logs.csv` (Level 3 공정/품질 맥락 데이터)**

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">컬럼명</th>
        <th style="width: 100px;">데이터 타입</th>
        <th style="width: 300px;">설명</th>
        <th style="width: 300px;">비고</th>
    </thead>
    <tbody>
        <tr><td class="td-rowheader">lot_id</td><td>String</td><td>로트 식별자</td><td>Key 역할 (예: 'LOT_2026_01')</td></tr>
        <tr><td class="td-rowheader">start_time</td><td>Datetime</td><td>해당 Lot 작업 시작 시각</td><td>Time Window 조인용</td></tr>
        <tr><td class="td-rowheader">end_time</td><td>Datetime</td><td>해당 Lot 작업 종료 시각</td><td>Time Window 조인용</td></tr>
        <tr><td class="td-rowheader">worker_id</td><td>String</td><td>작업자 ID (Man)</td><td>A / B / C 조</td></tr>
        <tr><td class="td-rowheader">material_batch</td><td>String</td><td>원자재 롯데 번호 (Material)</td><td>Raw_MAT_X / Y</td></tr>
        <tr><td class="td-rowheader">defect_status</td><td>String/Int</td><td>품질 결과</td><td>'0': 양품, '1': 불량 (수축/치수 불량)</td></tr>
    </tbody>
</table>
</div>

- **Dataset 생성하기**

```python
#//file: "data_gen.py"
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 랜덤 시드 고정 (재현성 확보)
np.random.seed(42)

# ==========================================
# 1. production_logs.csv 생성 (100개 Lot)
# ==========================================
num_lots = 100
start_base_time = datetime(2026, 3, 1, 8, 0, 0) # 2026년 3월 1일 오전 8시 시작

lots_data = []
current_time = start_base_time

workers = ['Worker_A', 'Worker_B', 'Worker_C']
materials = ['Raw_MAT_X', 'Raw_MAT_Y'] # Raw_MAT_Y가 투입될 때 온도 변동성 증가 패턴 삽입

for i in range(1, num_lots + 1):
    lot_id = f"LOT_2026_{i:03d}"
    duration_minutes = np.random.randint(12, 18) # Lot당 약 12~17분 소요
    end_time = current_time + timedelta(minutes=duration_minutes)
    
    worker = np.random.choice(workers, p=[0.4, 0.3, 0.3])
    material = np.random.choice(materials, p=[0.6, 0.4])
    
    lots_data.append({
        'lot_id': lot_id,
        'start_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'worker_id': worker,
        'material_batch': material,
        'defect_status': 0 # 기본값, 센서 데이터 생성 후 조건에 맞게 업데이트
    })
    
    current_time = end_time + timedelta(seconds=10) # Lot 간 10초 휴지시간

df_lots = pd.DataFrame(lots_data)

# ==========================================
# 2. sensor_timeseries.csv 생성 (1초 단위)
# ==========================================
sensor_rows = []

# 전체 시간 범위 설정
total_start = datetime.strptime(df_lots['start_time'].min(), '%Y-%m-%d %H:%M:%S')
total_end = datetime.strptime(df_lots['end_time'].max(), '%Y-%m-%d %H:%M:%S')

current_sensor_time = total_start

# Lot별 특성에 따른 센서 노이즈 및 노즐 온도 하락 로직 적용
defect_lot_ids = []

for idx, row in df_lots.iterrows():
    lot_start = datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S')
    lot_end = datetime.strptime(row['end_time'], '%Y-%m-%d %H:%M:%S')
    
    # 의도적 불량 패턴 주입: Worker_B 이면서 Raw_MAT_Y 자재를 사용할 때 40% 확률로 온도 급감 현상 발생
    is_anomaly = False
    if row['worker_id'] == 'Worker_B' and row['material_batch'] == 'Raw_MAT_Y':
        if np.random.rand() < 0.5:
            is_anomaly = True
            defect_lot_ids.append(row['lot_id'])
    elif np.random.rand() < 0.05: # 그 외 일반 로트 5% 확률로 불량 발생
        is_anomaly = True
        defect_lot_ids.append(row['lot_id'])

    # 1초 간격으로 센서 데이터 생성
    t = lot_start
    while t <= lot_end:
        if is_anomaly:
            # 불량 로트: 노즐 온도가 212~217도로 크게 떨어짐
            temp_nozzle = np.round(np.random.normal(215.0, 1.8), 2)
            pressure_inj = np.round(np.random.normal(82.0, 3.5), 2)
        else:
            # 정상 로트: 노즐 온도가 228~232도 정상 유지
            temp_nozzle = np.round(np.random.normal(230.0, 1.2), 2)
            pressure_inj = np.round(np.random.normal(90.0, 2.0), 2)
            
        cooling_time = 15.0 # 고정 설정값
        
        sensor_rows.append({
            'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
            'machine_id': 'PRESS_01',
            'temp_nozzle': temp_nozzle,
            'pressure_injection': pressure_inj,
            'cooling_time': cooling_time
        })
        t += timedelta(seconds=1)

df_sensor = pd.DataFrame(sensor_rows)

# ==========================================
# 3. production_logs의 불량 여부(defect_status) 업데이트 및 저장
# ==========================================
df_lots['defect_status'] = df_lots['lot_id'].apply(lambda x: 1 if x in defect_lot_ids else 0)

# CSV 파일 저장
df_sensor.to_csv('sensor_timeseries.csv', index=False, encoding='utf-8-sig')
df_lots.to_csv('production_logs.csv', index=False, encoding='utf-8-sig')

print("✅ 성공적으로 2개의 실습 데이터 파일이 생성되었습니다!")
print(f"- sensor_timeseries.csv : 총 {len(df_sensor):,} 행 (용량: 약 1.2 MB)")
print(f"- production_logs.csv   : 총 {len(df_lots):,} 행 (불량 수: {sum(df_lots['defect_status'])}개)")
```


## 4. 단계별 실습 진행 과정

- **[1단계] 데이터 전처리 및 융합 (Data Join & Aggregation)**
    - **목표:** 
        - 서로 다른 주기를 가진 시계열 센서 데이터와 Lot 품질 데이터를 결합하여 '분석용 데이터 마트(Data Mart)' 생성

    - **실습 수행 내용:**
        1. `production_logs`의 `start_time`과 `end_time` 사이 구간(Time Window)에 해당하는 `sensor_timeseries` 데이터를 필터링
        2. 해당 구간 동안의 센서 데이터 통계량(평균값 `mean`, 최대값 `max`, 표준편차 `std`) 계산
        3. 계산된 센서 통계 수치를 `production_logs`의 해당 `lot_id` 행에 조인(Join)


    - **결과 코드 예시 (Python):**

        ```python
        # Lot별 센서 데이터 평균 집계 후 조인
        lot_summary = []
        for idx, row in production_logs.iterrows():
            # Time Window 필터링
            mask = (sensor_df['timestamp'] >= row['start_time']) & (sensor_df['timestamp'] <= row['end_time'])
            sub_sensor = sensor_df[mask]

            # 집계값 생성
            lot_summary.append({
                'lot_id': row['lot_id'],
                'temp_avg': sub_sensor['temp_nozzle'].mean(),
                'pressure_avg': sub_sensor['pressure_injection'].mean(),
                'cooling_avg': sub_sensor['cooling_time'].mean()
            })

        # 데이터 마트 완성
        data_mart = pd.merge(production_logs, pd.DataFrame(lot_summary), on='lot_id')
        ```


- **[2단계] 탐색적 데이터 분석 (EDA) 및 변수 간 상관관계 파악**
    - **목표:**
        - 양품(`0`)과 불량(`1`) 발생 집단 간에 어떤 공정 파라미터가 유의미한 차이를 보이는지 확인

    - **실습 수행 내용:**
        1. **요약 통계 비교:** 양품/불량 그룹별 `temp_avg`, `pressure_avg`의 평균 및 표준편차 비교
        2. **상관관계 분석:** 주요 변수 간 Correlation Matrix(상관계수 히트맵) 작성
        3. **시각화 (Boxplot / Distribution plot):** * `temp_avg` 분포와 `defect_status` 관계 시각화

    - `worker_id`(작업자) 및 `material_batch`(원자재)별 불량 발생 비율(Cross-tabulation) 비교


- **[3단계] 현장 문제 원인(Root-Cause) 규명 및 패턴 발견**
    - **목표:**
        - 데이터 분석 결과를 도메인 관점으로 해석하여 진짜 원인을 추론

    - **실습 가이드 포인트 (데이터 속 숨겨진 패턴 예시):**
        - **패턴 A (물리 변수):**
            - 불량이 발생한 Lot들은 공통적으로 `temp_avg`가 215℃ 이하로 떨어진 구간에서 집중 발생함 (노즐 온도 저하 ➔ 수지 용융 불량 ➔ 치수 불량)
        - **패턴 B (4M 변수):**
            - 특정 원자재 롯데(`Raw_MAT_Y`)가 투입될 때 노즐 온도가 쉽게 하락하는 경향이 관찰됨 (원자재별 점성 차이 존재)


- **[4단계] 최적 공정 레시피 및 도출 인사이트 정리**
    - **목표:**
        - 단순 현상 파악에 그치지 않고,
        - 현장에 적용 가능한 개선 가이드라인(Prescriptive Insight)을 작성


## 5. 실습 제출 결과물

- 제출 결과물: 공정 데이터 기반 인사이트 도출 보고서

<br>

- **[결과물 양식] 공정 개선 분석 및 인사이트 보고서**

1. **문제 데이터 요약**
    - 총 분석 대상 Lot 수: `N`건 (양품: `N`건, 불량: `N`건 / 불량률: `X%`)
    - 주요 불량 유형: 사출 수축 및 치수 불량

2. **주요 데이터 분석 결과**
    - **결과 1 (핵심 인자 발굴):**
        - 사출 노즐 온도가 불량에 가장 결정적인 영향을 미침 (상관계수 r = -0.72)
    - **결과 2 (임계값 발견):**
        - 노즐 온도가 218℃ 미만으로 떨어지는 순간, 불량 발생률이 1.2%에서 18.5%로 급증함
    - **결과 3 (4M 연계 분석):**
        - `Worker_B` 작업조 가동 시 및 `Raw_MAT_Y` 자재 투입 시 노즐 온도가 설정값 대비 크게 흔들림(Variability 상승)을 확인

3. **현장 적용 개선 인사이트 및 액션 플랜**
    - **[설비 제어 및 Interlock 설정]:**
        - 사출 노즐 온도 하한 관리 기준(LSL)을 기존 210℃에서 **220℃로 상향 조정**
        - 온도가 218℃ 이하로 내려갈 경우, 설비가 자동 정지(Interlock)되고 알람이 발생하도록 PLC 조건 수정

    - **[표준 작업 가이드(SOP) 개정]:**
        - `Raw_MAT_Y` 자재 투입 시 예열 시간을 기존 15분에서 25분으로 연장하는 표준 작업 지침 업데이트

    - **[기대 효과 (ROI Estimate)]:**
        - 노즐 온도 이탈 방지를 통해 전체 불량률 2.5% 🡪 0.6% 수준으로 감소 예상 (연간 약 N,000만 원 손실 절감)
