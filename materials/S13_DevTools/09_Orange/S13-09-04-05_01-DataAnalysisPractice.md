---
layout: page
title:  "Orange: 데이터 분석 실습"
permalink: /materials/S13-09-04-05_01-DataAnalysisPractice
description: "오렌지를 이용한 데이터 분석 실습을 진행합니다."
categories:
    - materials
tags:
    - AI
    - Data Analysis
    - Orange
---
* toc
{:toc}



## 1. 실습 시나리오

> - **스마트팩토리 모터 가공 라인의 '예지보전(PdM) 프로젝트' 시나리오**<br><br>
>   - **배경**
>       - 자동차 부품을 정밀 가공하는 '대일정밀' 제2공장
>       - CNC 밀링 및 고속 회전 가공기(M01, M02, M03) 3대가 24시간 가동 중
>   - **현장 문제:** 최근 공장에 심각한 문제가 발생함
>       - **불시 정지(다운타임):** 가공 중 주축 모터가 갑자기 멈추거나 공구가 부러지는 사고가 월 2~3회 발생
>       - **손실 발생:** 설비가 멈추면 라인 전체가 대기 상태가 되고, 고장 시점에 가공 중이던 고가의 부품은 전량 폐기되어 회당 수천만 원의 손실 발생
>       - **현장 정비의 한계:** 주기적으로 부품을 교체하자니 멀쩡한 공구를 버려 비용이 낭비되고, 고장 날 때까지 쓰자니 언제 터질지 몰라 불안함
>   - **프로젝트의 목표**
>       - 공장장은 설비에 부착된 센서 데이터(온도, 회전수, 토크, 진동 등)를 활용해
>       - **"설비가 완전히 고장 나기 직전에 고장 유형을 미리 감지하여 작업자에게 경고하는 AI 모델"**을 구축하라는 미션을 부여함
{: .yellow-quote}

<br>

## 2. 실습용 제조 현장 합성 데이터셋 설계

- 회전 설비(모터, 베어링, 절삭 가공기 등)에서 흔히 수집되는 센서 시계열 데이터 요약본(Tabular format, 약 1,000~5,000행 권장)

<div class="info-table">
    <table>
        <thead>
            <th style="width: 130px;">컬럼명</th>
            <th style="width: 150px;">Orange3 데이터 타입</th>
            <th style="width: 100px;">역할 (Role)</th>
            <th style="width: 450px;">비고</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">timestamp</td>
                <td>Time</td><td>Meta</td>
                <td class="td-left">설비 가동 시각 (2026-09-01 08:00:00 단위)</td>
            </tr>
            <tr>
                <td class="td-rowheader">machine_id</td>
                <td>Categorical</td>
                <td>Meta</td>
                <td class="td-left">설비 식별자 (M01, M02, M03)</td>
            </tr>
            <tr>
                <td class="td-rowheader">air_temp</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">공장 내부 공기 온도 (°C, 18~35)</td>
            </tr>
            <tr>
                <td class="td-rowheader">process_temp</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">가공 부위 공정 온도 (°C, 35~90)</td>
            </tr>
            <tr>
                <td class="td-rowheader">rotational_speed</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">회전 속도 (RPM, 1200~2800)</td>
            </tr>
            <tr>
                <td class="td-rowheader">torque</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">부하 토크 (Nm, 10~80)</td>
            </tr>
            <tr>
                <td class="td-rowheader">vibration</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">진동 가속도 RMS (mm/s, 0.5~15.0)</td>
            </tr>
            <tr>
                <td class="td-rowheader">tool_wear</td>
                <td>Numeric</td>
                <td>Feature</td>
                <td class="td-left">공구 누적 사용 시간 (min, 0~250)</td>
            </tr>
            <tr>
                <td class="td-rowheader">failure_type</td>
                <td>Categorical</td>
                <td>Target</td>
                <td class="td-left">불량/고장 유형: `Normal`, `Heat_Dissipation`, `Tool_Wear`, `Overstrain`</td>
            </tr>
        </tbody>
    </table>
    <span class="caption">(Source: Sky Lectures / AiDALab)</span>
</div>

<br>

## 3. 실습용 샘플 데이터 생성 Python 스크립트

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 2000

air_temp = np.random.normal(25, 3, n_samples)
process_temp = air_temp + 10 + np.random.normal(0, 1.5, n_samples)
rot_speed = np.random.normal(1800, 200, n_samples)
torque = np.random.normal(40, 10, n_samples)
vibration = np.random.normal(2.5, 0.8, n_samples)
tool_wear = np.random.uniform(0, 240, n_samples)

failure_type = ['Normal'] * n_samples

for i in range(n_samples):
    # 과열 결함 (방열 불량)
    if (process_temp[i] - air_temp[i] < 8.5) and (rot_speed[i] < 1500):
        failure_type[i] = 'Heat_Dissipation'
        process_temp[i] += 12
    # 공구 마모 결함
    elif tool_wear[i] > 210 and vibration[i] > 4.0:
        failure_type[i] = 'Tool_Wear'
        vibration[i] += np.random.uniform(2, 5)
    # 과부하 결함
    elif torque[i] > 60 and rot_speed[i] > 2100:
        failure_type[i] = 'Overstrain'
        torque[i] += 15

df = pd.DataFrame({
    # '10T' -> '10min'으로 변경
    'timestamp': pd.date_range('2026-09-01', periods=n_samples, freq='10min'),
    'machine_id': np.random.choice(['M01', 'M02', 'M03'], n_samples),
    'air_temp': np.round(air_temp, 2),
    'process_temp': np.round(process_temp, 2),
    'rotational_speed': np.round(rot_speed, 1),
    'torque': np.round(torque, 2),
    'vibration': np.round(vibration, 3),
    'tool_wear': np.round(tool_wear, 1),
    'failure_type': failure_type
})

df.to_csv('mfg_predictive_maintenance.csv', index=False)
print("mfg_predictive_maintenance.csv 생성 완료")
```

- ['mfg_predictive_maintenance.csv' 다운로드](/materials/datasets/mfg_predictive_maintenance.csv)

<br>

## 4. 실습 워크플로우 구성 및 단계별 가이드

- Orange3 캔버스에 위젯(Widget)을 배치하고 연결하는 순서

    <div class="insert-image">
        <img src="/materials/S13_DevTools/images/S13-09-04-05_01-001.png" style="width: 90%;">
        <span class="caption">(Source: Sky Lectures / AiDALab)</span>
    </div>


    - **1단계: 데이터 로드 및 메타데이터 정의:**
        - **File/CSV File** 위젯으로 `mfg_predictive_maintenance.csv` 파일을 불러오기
        - **File/CSV File** 위젯 🡪 **Select Columns** 위젯 연결
        - **Select Columns** 위젯
            - 컬럼 속성 지정: 하단 컬럼 목록에서 각 컬럼의 **Type**과 **Role**을 아래와 같이 설정
                - timestamp와 machine_id는 분석 피처에서 제외하기 위해 **Meta**로 설정
                    - `timestamp`: Type = `Time`, Role = `meta`
                    - `machine_id`: Type = `Categorical`, Role = `meta`
                - failure_type을 **Target**으로 지정
                    - `failure_type`: Type = `Categorical`, Role = **`target`**
                - 나머지 연속형 센서 변수 6종을 **Features**로 배치<br><br>
                    - `air_temp` ~ `tool_wear` (6개 센서): Type = `Numeric`, Role = `feature`<br><br>

    - **2단계: 데이터 구조 점검 (Data Table 등):**
        - 기초 통계량 및 클래스 불균형 확인
            - **Visualize** 탭에서 **Distributions** 위젯 🡨 **Select Columns** 위젯 연결
            - **Distributions** 위젯
                - `failure_type` 행을 클릭하여 
                - 우측 히스토그램에서 `Normal`이 대부분이고 결함 유형(`Heat_Dissipation(열 방출)`, `Tool_Wear(공구 마모)`, `Overstrain(과도한 변형)`)이 소수인지(불균형 상태) 확인
            - **Data Table** 위젯
                - 데이터 정렬 등으로 Min/Max 값 확인
            - **Box Plot** 위젯
                - 센서 변수들의 4분위수, 평균, 중앙값등 확인
            - **Data Info** 위젯
                - Missing(결측치) 여부/개수 확인<br><br>

    - **3단계: 탐색적 데이터 분석 (Scatter Plot):**
        - 센서 간 물리적 경계 시각화
        - **Visualize** 탭 🡪 **Scatter Plot** 위젯 🡨 **Select Columns** 위젯 연결
        - **Scatter Plot** 위젯
            - 좌측 축 및 시각화 옵션을 다음과 같이 지정
                - **Axis X**: `rotational_speed`
                - **Axis Y**: `torque`
                - **Color**: `failure_type`
                - **Shape**: None (또는 동일)
            - 우측 산점도
                - 우측 상단(높은 토크와 높은 RPM 영역)에 `Overstrain(과도한 변형, 과부하)` 데이터가 어떻게 군집을 이루는지 육안으로 확인

                    <div class="insert-image">
                        <img src="/materials/S13_DevTools/images/S13-09-04-05_01-002.png">
                        <span class="caption">(Source: Sky Lectures / AiDALab)</span>
                    </div>

                    - Overstrain (과부하 고장)의 뚜렷한 물리적 분리
                        - 연두색 포인트(Overstrain)들이 그래프 우측 최상단(회전수 2,000 RPM 이상, 토크 75~85 Nm 이상) 영역에 완전히 고립되어 분포
                        - 고속 회전 중에 과도한 토크(부하)가 가해질 때 모터나 구동축에 무리가 가며 발생하는 전형적인 과부하 파손을 보여줌
                        - 두 변수(rotational_speed, torque) 조합만으로도 "토크 > 75 Nm & RPM > 2,000"과 같은 단순 임계치(Threshold) 룰을 세워 100% 가깝게 사전 차단 및 탐지가 가능함 (연두색 회귀선의 $r = 0.53$도 이러한 강한 양의 상관성을 반영)

                    - Normal (정상 구동)의 거대한 밀집과 클래스 불균형
                        - 붉은색 점들이 중앙 영역(1,600~2,000 RPM, 30~60 Nm)에 거대한 정규분포 형태로 밀집해 있음
                        - 대부분의 데이터가 정상 상태에 몰려 있는 전형적인 제조 현장의 클래스 불균형(Class Imbalance)을 보여줌
                        - 붉은색 회귀선($r = 0.04$)이 거의 수평인 것은 정상 가동 범위에서는 회전수와 토크가 설계 제어 범위 내에서 독립적으로 안정 유지됨을 의미함

                    - Heat_Dissipation (방열 불량)의 저속 편향
                        - 하늘색 포인트들이 좌측 중간 영역(1,300~1,500 RPM 부근)에 모여 있음
                        - 저속 회전 구간에서는 모터의 자체 냉각 팬(Cooling Fan) 회전력이 떨어져 방열 성능이 급격히 저하되는 물리적 특성이 반영된 결과
                        - 토크 수치 자체보다는 낮은 RPM과 온도 센서(process_temp, air_temp)의 조합을 추가로 확인해야 정확한 진단이 가능함을 시사함
                        
                    - Tool_Wear (공구 마모)가 이 차트에서 분리되지 않는 이유
                        - 주황색 포인트(Tool_Wear)들이 정상(붉은색) 데이터 한가운데 뒤섞여 있음
                        - 공구 마모는 회전 속도나 단순 모터 토크 축만으로는 감지할 수 없는 고장 유형
                        - 고장 예측 모델을 만들 때 이 두 피처만 쓰면 공구 마모는 절대 잡아낼 수 없음 🡪 따라서 X축을 tool_wear(누적 마모 시간), Y축을 vibration(진동 가속도)으로 변경해야 주황색 군집이 분리되는 것을 확인할 수 있음
                    <br><br>

                - X축을 `tool_wear`, Y축을 `vibration`으로 변경하여 마모도와 진동 증가에 따른 `Tool_Wear` 고장 분포도 확인

                    <div class="insert-image">
                        <img src="/materials/S13_DevTools/images/S13-09-04-05_01-003.png">
                        <span class="caption">(Source: Sky Lectures / AiDALab)</span>
                    </div>

                    - Tool_Wear (공구 마모 결함)의 완벽한 고립 군집화
                        - 주황색 포인트(Tool_Wear)들이 그래프 우측 최상단(마모 시간 210분 이상, 진동 6.0 mm/s 이상)에 완전히 독립된 클러스터로 모여 있음
                        - 공구가 닳아 절삭날이 마모되면 가공 저항이 불규칙해지면서 설비 전체에 심한 떨림(진동)을 유발하는 전형적인 기계적 마모 현상
                        - 현장 판별 룰 도출 🡪 이 차트 하나만으로도 `tool_wear > 210 AND vibration > 6.0`이라는 명확한 차단/경고 규칙(Rule)을 세울 수 있음

                    - 정상 영역(붉은색)의 진동 베이스라인 한계치 확인
                        - 정상 가동 시 진동은 마모 시간이 0에서 200분까지 증가하더라도 대부분 1.0 ~ 4.5 mm/s 대역에 안정적으로 머물러 있음
                        - 붉은색 회귀선($r = -0.01$)이 수평을 이루는 것은 통상적인 마모 범위 내에서는 진동이 급격히 튀지 않음을 뜻함
                        - 경계선 파악 🡪 진동이 5.0 mm/s를 초과하는 순간부터는 정상 범위를 벗어난 이상 징후로 간주할 수 있는 기준선이 확인됨

                    - Heat_Dissipation 및 Overstrain의 분포 의미
                        - 하늘색(방열 불량)과 연두색(과부하) 점들이 정상 데이터 밴드(진동 2~4 사이) 한가운데 섞여 있음
                        - 방열 불량이나 순수 모터 과부하는 공구의 기계적 파손/마모가 아니므로 진동이 비정상적으로 치솟지 않음 🡪 즉, "진동 센서만 감시해서는 모터 과열이나 과부하를 절대 잡아낼 수 없다"는 한계를 보여줌
                    
                - **두 시각화 결과를 종합한 EDA 결론**
                    - 토크 vs 회전수 차트
                        - 과부하(Overstrain)를 핀포인트로 탐지
                    - 마모 시간 vs 진동 차트
                        - 공구 마모(Tool_Wear)를 핀포인트로 탐지
                    - 종합 결론
                        - 단일 2D 차트(2개 변수)로는 각 고장을 부분적으로만 잡을 수 있음
                        - 6개 센서를 종합 판단하는 다변량 AI 분류 모델(Tree, Random Forest 등)이 반드시 필요한 이유를 입증

        > - **탐색적 데이터 분석 (EDA):**
        >   - Feature Statistics / Box Plot / Scatter Plot
        >       - 제조 데이터 특유의 클래스 불균형과 센서 간 물리적 상관관계를 시각적으로 확인
        >   - Feature Statistics
        >       - 클래스(Normal 대 고장 유형) 불균형 비율과 센서값의 결측치/이상치 범위 점검
        >   - Scatter Plot
        >       - rotational_speed 대비 torque를 X/Y축에 두고
        >       - 색상(Color)을 failure_type으로 매핑하여
        >       - 물리적 경계(과부하 영역)를 확인
        >   - Box Plot
        >       - 정상 상태와 고장 상태 간 vibration 및 process_temp 분포 차이 확인
        {: .gray-quote}

        <br>

    - **4단계: 데이터 전처리 (Preprocess):**
        - 스케일 차이 보정 (표준화)
            - 온도(수십 단위), RPM(수천 단위), 진동(소수점 단위) 간 스케일 격차를 맞추기 위해 표준화를 진행함
            - Z-score 정규화(Normalize features: Standardize)를 적용
        - **Transform** 탭 🡪 **Preprocess** 위젯 🡨 **Select Columns** 위젯 연결
        - **Preprocess** 위젯
            - 머신러닝 및 신경망 모델 학습을 위한 기초 정규화 수행
            - 좌측 'Preprocessors' 목록  🡪 **Normalize Features** 🡪 더블클릭하여 가운데 작업 영역으로 넘김
                - 넘겨진 **Normalize Features** 항목을 클릭
                - 우측 설정에서 **Standardize (mean = 0, var = 1)** 옵션을 선택
            - 좌측 'Preprocessors' 목록  🡪 **Impute Missing Values**
                - 센서 통신 에러로 발생한 결측치를 평균/중앙값으로 대체<br><br>

    - **5단계: AI 모델 학습 및 성능 평가 (Test & Score):**
        - 알고리즘 연결 및 교차 검증
        - 분류 알고리즘 위젯들을 Test & Score에 동시 연결하여 성능을 벤치마킹함<br><br>
        - **Evaluate** 탭 🡪 **Test & Score** 위젯 🡨 **Preprocess** 위젯
        - **Preprocess** 위젯 - **Test & Score** 위젯 사이의 링크 클릭
            - 팝업이 뜨면 `Preprocessed Data` 🡪 `Data`로 매핑
        - **Model** 탭
            - 다음 4가지 알고리즘 위젯을 꺼내어 각각 **Test & Score** 위젯에 연결
                - **Tree** (Decision Tree)
                - **Random Forest**
                - **SVM**
                - **Neural Network** (기본 MLP)
        - **Test & Score** 위젯
            - Cross Validation (10-fold 또는 5-fold) 적용
            - 좌측 패널 설정 확인
                - **Sampling**: `Cross validation` 선택
                - **Number of folds**: `5` 또는 `10`
                - **Target class**: (Average over classes) 또는 특정 결함 클래스 선택
            - 중앙 표에서 모델별 **AUC**, **CA (Accuracy)**, **F1**, **Precision**, **Recall** 지표를 비교
        - 제조 현장 특성을 고려하여 단순 Accuracy 외에 불량을 놓치지 않는 지표인 **Recall** 및 **F1-score**를 중점 비교<br><br>

        <div class="insert-image">
            <img src="/materials/S13_DevTools/images/S13-09-04-05_01-004.png"><br><br>
            <img src="/materials/S13_DevTools/images/S13-09-04-05_01-005.png">
            <span class="caption">(Source: Sky Lectures / AiDALab)</span>
        </div>

        - 분석 내용
            - 전형적인 제조 현장 데이터의 **극심한 클래스 불균형(Class Imbalance) 환경에서 모델 평가 시 나타나는 핵심 함정과 현상**이 명확히 드러난 결과<br><br>

            - `Tree` 모델의 붕괴: "정확도(CA)의 함정"과 0.000의 MCC
                - **CA 0.979, F1 0.969**이지만 **AUC 0.377, MCC 0.000**
                - CA(정확도)가 97.9%로 매우 높아 보이지만, 이는 "단 하나의 고장도 맞추지 못하고 모든 샘플을 Normal(정상)로만 찍어버렸기 때문"
                - 전체 데이터 중 정상이 97.9%를 차지하므로, 아무것도 학습하지 않고 전부 정상이라고만 답해도 정확도는 97.9%가 나옴
                - 불균형 데이터 평가에 가장 엄격한 지표인 **MCC(Matthews Correlation Coefficient)가 0.000**이라는 점이 이를 입증함 (무작위 찍기와 동일)
                - 다중 클래스에서 AUC가 0.5(이진 기준 무작위)보다 한참 낮은 0.377로 떨어진 것 역시 소수 클래스(고장 유형들)에 대한 예측 확률을 완전히 엉뚱하게 부여했음을 보여줌

            - 고도화 모델(SVM, Neural Network, Random Forest)의 뛰어난 성능
                - **Neural Network (최우수 모델):**
                    - **AUC 0.999 / CA 0.997 / F1 0.996 / MCC 0.925**로 전 지표에서 가장 압도적인 성능을 보임
                    - 앞선 단계에서 `Preprocess`를 통해 센서값들을 표준화(Standardize)해 두었기 때문에 다층 퍼셉트론(MLP)의 가중치 수렴과 결정 경계 형성이 매우 정밀하게 이루어졌음

                - **SVM:**
                    - **AUC 0.996 / MCC 0.898**로 신경망에 준하는 매우 높은 분류 성능을 보임
                    - 비선형 RBF 커널이 복합 센서 경계를 잘 포착함

                - **Random Forest:**
                    - 단일 트리의 과적합 및 다수 클래스 쏠림 현상을 앙상블 배깅(Bagging) 기법으로 극복하여
                    - **MCC 0.758, AUC 0.985** 수준으로 준수하게 끌어올림

    <br>

    - **6단계: 결과 해석 및 설비 진단 기준 도출 🡪 혼동 행렬 및 룰 추출 (Tree Viewer):**
        - 오분류 분석 및 현장 룰 도출
            - **Visualize** 탭 🡪 **Tree Viewer** 위젯 선택
                - **Tree** 위젯과 연결
            - **Tree Viewer** 선택
                - 학습된 Decision Tree를 시각화하여 "어떤 센서 임계치(예: 진동 > 8.2 mm/s, 공구 마모 > 200분)에서 고장으로 분기되는지" 현장 작업자용 룰(Rule)을 도출
                    - 트리의 분기 조건(예: `torque > 1.8`, `tool_wear > 1.5` 등 표준화된 임계값)을 보며
                    - 현장 제어 시스템에 적용할 수 있는 직관적인 IF-THEN 규칙을 도출<br><br>

            - **Evaluate** 탭 🡪 **Confusion Matrix** 위젯 선택
            - **Confusion Matrix** 위젯
                - 실제 고장인데 정상(Normal)으로 오분류한 False Negative 비율을 확인
                - **Test & Score** 위젯과 연결
                - 위젯 상단에서 **Random Forest** 선택
                - 실제 고장(`Tool_Wear`, `Overstrain` 등)인데 `Normal`로 예측된 건수(False Negative, 미탐지율)가 몇 건인지 대각선 외 영역을 확인

                <div class="insert-image">
                    <img src="/materials/S13_DevTools/images/S13-09-04-05_01-006.png">
                    <span class="caption">(Source: Sky Lectures / AiDALab)</span>
                </div>

                - `Tree Viewer`에 노드가 단 1개만 뜨고 트리가 전혀 뻗어나가지 못했던 명확한 이유가 이 혼동 행렬(Confusion Matrix)에 고스란히 담겨 있음
                - `Tree` 모델의 분류 결과 분석: "완벽한 학습 실패"
                    - 화면의 가로축은 예측값(Predicted), 세로축은 실제값(Actual)
                    - **예측 결과 분포:**
                        - 모델이 예측한 `Normal` 열의 합계($\Sigma$)를 보면 **2,000건 전체가 `Normal` 열 하나에만** 몰려 있음
                        - `Heat_Dissipation`: 예측 0건
                        - `Overstrain`: 예측 0건
                        - `Tool_Wear`: 예측 0건
                    - **결론:**
                        - 모델이 입력 데이터의 센서값들을 전혀 보지 않고, 2,000개 데이터 전체를 무조건 "Normal(정상)"이라고만 판정한 상태
                        - 트리가 분기(Split)를 단 하나도 생성하지 않고, 시작점(Root Node)에서 다수결에 따라 "전부 Normal"이라고 결론을 내버렸기 때문에 `Tree Viewer`에서도 하위 가지 없이 **단 1개의 루트 노드**만 덩그러니 그려졌던 것

                - 제조 현장 관점에서의 치명적인 실패
                    - **미탐지율 100% (False Negative = 42건):**
                        - 실제 방열 불량 28건, 과부하 4건, 공구 마모 10건 등 총 42건의 설비 고장이 발생했으나, 모델은 단 1건도 감지하지 못하고 모두 "정상 가동 중"으로 판단함
                        - 현장에 도입했을 경우 모터가 타버리거나 공구가 부러지는 사고를 전혀 막지 못하는 전형적인 **무용지물 모델**
                    - **겉보기 정확도(97.9%)의 착시:**
                        - 고장을 42건이나 전부 놓쳤음에도, 정상 데이터 1,958건을 얻어걸려 맞췄기 때문에 앞선 단계에서 정확도(CA)가 97.9%로 높게 표기되었던 것

                - 왜 이런 현상이 발생했는가?
                    - **극심한 클래스 불균형:**
                        - 정상(1958건, 97.9%) 대비 결함 데이터의 총합이 42건(2.1%)에 불과함
                        - 특히 `Overstrain`은 4건밖에 되지 않음
                    - **지니 불순도(Gini Impurity)의 한계:**
                        - 단일 의사결정나무는 노드를 쪼갤 때 전체 불순도가 낮아지는 방향으로 기준을 잡음
                        - 정상 데이터가 98%에 달하면 굳이 노드를 복잡하게 쪼개지 않고 그냥 "전부 정상"으로 묶어버리는 것이 수학적으로 불순도 감소 폭이 가장 작아 학습을 조기 중단해 버림


                <div class="insert-image">
                    <h3 style="text-align: center">최종 작성 화면</h3>
                    <img src="/materials/S13_DevTools/images/S13-09-04-05_01-006.png">
                    <span class="caption">(Source: Sky Lectures / AiDALab)</span>
                </div>


> - **실습 완료 체크포인트**
>   - [ ] `timestamp`와 `machine_id`가 불필요하게 모델의 학습 변수(`feature`)로 들어가지 않고 `meta`로 격리되었는가?
>   - [ ] 단위 차이가 큰 진동 센서와 RPM 수치가 정규화(`Standardize`)되어 SVM 및 Neural Network에 전달되었는가?
>   - [ ] 정상 데이터 비율이 훨씬 높으므로 단순 정확도(CA) 대신 **F1-score**와 고장을 놓치지 않는 **Recall** 기준으로 최적 모델을 선정했는가?
{: .yellow-quote}

<br>

## 5. 단계별 현장 대응과 Orange3 실습의 매핑

<div class="info-table">
    <table>
        <thead>
            <th style="width: 130px;">현장 엔지니어의 액션</th>
            <th style="width: 130px;"> Orange3 실습 단계</th>
            <th style="width: 690px;">실제 다루는 작업과 현장 의미</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader" style="text-align: left;">1. 설비 로그 취합</td>
                <td>File</td>
                <td class="td-left">
                    3대 설비에서 10분 주기로 수집된 센서 CSV 로드<br>
                    시간(timestamp)과 설비 번호(machine_id)는 식별용 Meta로 빼두고, 목표값인 고장 유형(failure_type)을 Target으로 지정
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" style="text-align: left;">2. 데이터 상태 점검</td>
                <td>Feature Statistics</td>
                <td class="td-left">
                    전체 2,000건 중 고장 데이터는 불과 수십~수백 건에 불과한 전형적인 클래스 불균형(Class Imbalance) 상태임을 파악
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" style="text-align: left;">3. 고장 징후 시각적 확인</td>
                <td>Scatter Plot</td>
                <td class="td-left">
                    회전수(RPM) 대비 토크(Nm)가 비정상적으로 치솟는 지점에 과부하(`Overstrain`)가 몰려 있고,<br>
                    마모도 증가 시 진동이 급증하는 물리적 패턴을 육안으로 확인
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" style="text-align: left;">4. 센서 단위 통일</td>
                <td>Preprocess</td>
                <td class="td-left">
                    RPM(수천 단위)과 진동(소수점 단위)은 스케일 차이가 너무 커 AI가 특정 센서만 편애해 학습할 수 있으므로,<br>
                    모든 수치를 평균 0, 분산 1로 표준화(Standardize)
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" style="text-align: left;">5. 최적 모델 선발</td>
                <td>Test & Score</td>
                <td class="td-left">
                    4가지 알고리즘(Tree, RF, SVM, Neural Network)을 돌려 교차 검증<br>
                    현장에서는 고장을 정상으로 잘못 판단하면 치명적이므로 Recall(재현율)이 가장 높은 모델을 선별
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" style="text-align: left;">6. 현장 조치 룰 정의</td>
                <td>Tree Viewer &<br>Confusion Matrix</td>
                <td class="td-left">
                    "토크가 특정 기준을 넘고 마모도가 200분을 초과하면 즉시 경고등 점등" 같은 명확한 IF-THEN 규칙을 도출하여<br>
                    PLC 및 제어반 알람 로직으로 이식
                </td>
            </tr>
        </tbody>
    </table>
    <span class="caption">(Source: Sky Lectures / AiDALab)</span>
</div>


## 6. 프로젝트 결과 및 현장 적용 효과

- **사전 경보 체계 구축:**
    - 진동과 온도가 특정 임계치 패턴을 보이면,
    - AI가 모터 정지 30분 전에 작업반장 스마트폰으로 알림 발송
        - "Tool_Wear 의심: 2번 라인 공구 점검 필요"
- **비용 절감:**
    - 불시 정지가 분기당 0건으로 감소하고,
    - 소모품 수명 한계까지 안전하게 사용하여
    - 공구 교체 비용 18% 절감