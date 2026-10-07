---
layout: page
title: 데이터 기반 공정 진단 개요
subtitle: 
permalink: /materials/S06-05-02-03_01-DataDrivenProcessDiagnosis
description: 데이터 수집부터 전처리, 분석, 신사업 라인 최적화 예측까지의 전체 파이프라인 이해
categories:
    - materials
tags:
    - 인공지능
    - AI
    - 데이터 품질 관리
    - Data Quality Management
related_posts:
    - /materials/S06-05-02-01_01-AiAndManufacturingData
    - /materials/S06-05-02-02_01-EquipmentDataCollection
    - /materials/S06-05-02-04_01-DataQualityManagement1
    - /materials/S06-05-02-04_02-DataQualityManagement2
    - /materials/S06-05-02-05_01-ProcessDataAnalysis
qna: false
---
* toc
{:toc}



## 1. 왜 데이터 기반 공정 진단인가?

<div class="insert-image">
    <img style="width: 800px;" src="/materials/S06_SmartFactory/images/S06-05-02-03_01-001_DataDrivenProcessDiagnosis.png">
    <span style="width: 840px;" class="caption">(Source: Sky Lectures / AiDALab)</span>
</div>

- **설비/공정 변경에 따른 초기 변동성 극복:**
    - 사업재편 시 새로운 금형, 신소재, 미검증 가공 장비가 결합되면서 공정 파라미터 간 복잡한 비선형적 상호작용 발생
    - 숙련공의 기존 노하우만으로는 신소재의 열팽창 특성, 가공 시 변형률 등의 미세 거동을 예측하기 어려움

- **품질 검사 체계의 패러다임 전환:**
    - **과거 (사후 전수/샘플 검사):** 완성된 완제품을 파괴/비파괴 검사하여 불량을 선별(폐기 및 재작업 비용 발생)
    - **현재 (가상 계측 / Virtual Metrology):** 가공 중 수집된 압력·온도·진동 프로파일만으로 가공 종료 즉시 품질 합/불 판정 및 치수 오차를 실시간 예측

- **신속한 램프업(Ramp-up) 달성:**
    - 데이터 기반 진단 모델을 적용하면 시운전 단계에서 최적 운전 조건(Golden Batch Recipe)을 단기간에 수렴시킬 수 있음



## 2. 엔드투엔드 제조 데이터 파이프라인

- 제조 데이터 파이프라인은 단방향 흐름이 아닌,
- 예측 결과가 현장 설비 제어계로 환류(Feedback Loop)되는 유기적 순환 구조를 가짐

<div class="insert-image">
    <img style="width: 600px;" src="/materials/S06_SmartFactory/images/S06-05-02-03_01-002_EndToEndPipeline.png">
    <span style="width: 740px;" class="caption">(Source: Sky Lectures / AiDALab)</span>
</div>


## 3. 파이프라인 단계별 핵심 역할

- **Step 1. 수집 및 통합 (Acquisition & Integration)**
    - **다차원 데이터 원천:**
        - **설비 센서 계측치 (OT):** 주축 회전수(RPM), 서보 모터 부하율(%), 냉각수 온도($^\circ\text{C}$), 유압 압력($\text{bar}$) 등 초 단위 연속 시계열
        - **공정 환경 데이터:** 작업장 온·습도, 공조 압력
        - **생산 실행 및 품질 이력 (IT):** 작업지시 번호, 투입 원소재 로트(Lot), 금형 번호, 3차원 측정기(CMM) 치수 측정 결과
    - **핵심 과제:**
        - 이종 프로토콜(OPC-UA, Modbus, MQTT, REST API)의 단일 게이트웨이 통합
        - 시간 동기화(NTP 기반)를 통한 데이터 간 정렬 기준선 확립<br><br>

- **Step 2. 데이터 전처리 및 정제 (Preprocessing & Cleaning)**
    - 제조 현장 원천 데이터(Raw Data)는 결측, 노이즈, 오류가 혼재되어 있어 정제 없이 모델에 직접 투입할 수 없음

    - **데이터 정제 4대 실무 작업:**
        - **물리적 결측치(Missing Values) 처리:**
            - 센서 일시 차폐, 전송 실패로 인한 단기 결측은 직전값 대체(Forward Fill) 또는 스플라인 보간(Spline Interpolation) 적용
            - 장기 결측 구간(설비 비가동, 점검 시간)은 분석 대상 타임윈도우에서 필터링 분리
        - **센서 이상치(Outlier/Noise) 제거:**
            - 전기적 서지(Spike 노이즈)에 대해 이동 중앙값 필터(Rolling Median Filter) 또는 국소 이상치 요인(Local Outlier Factor) 기법 적용
            - 물리적 설비 스펙 범위를 벗어난 한계값 필터링 (예: 상온 공정에서 온도가 $9999^\circ\text{C}$로 계측되는 센서 단선 오류)
        - **서로 다른 샘플링 주기 정렬 (Resampling & Alignment):**
            - 진동 센서($1\text{kHz}$), 압력 센서($10\text{Hz}$), 온도 센서($1\text{Hz}$)의 샘플링 레이트를 공통 분석 단위(예: $100\text{ms}$ 그리드)로 다운샘플링/업샘플링 동기화
        - **시공간 맥락 결합 (Contextual Joining):**
            - MES의 로트 생산 시작/종료 시점 타임스탬프를 기준으로 설비 시계열 데이터를 슬라이싱 🡪 `[로트 ID - 품번 - 공정 파라미터 시계열]` 매트릭스로 병합<br><br>

- **Step 3. 특징 추출 및 데이터 구조화 (Feature Engineering)**
    - 단순한 시계열 원시 파형 전체를 머신러닝에 직접 입력하기보다는,
    - 공정 물리적 의미를 갖는 핵심 피처(Feature)로 압축 변환해야 모델의 설명력과 연산 효율이 극대화됨

        <div class="insert-image">
            <img style="width: 700px;" src="/materials/S06_SmartFactory/images/S06-05-02-03_01-003_FeatureEngineering.png">
            <span style="width: 780px;" class="caption">(Source: Sky Lectures / AiDALab)</span>
        </div>

- **Step 4. 공정 진단 및 원인 분석 (Process Diagnostics & RCA)**
    - 신사업 라인 안정화를 위해 현재 라인의 상태가 정상 범주 내에 있는지,
    - 이상이 있다면 원인은 무엇인지 분석

    - **통계적 공정 관리 (Statistical Process Control, SPC):**
        - 관리도($\bar{X}-R$, EWMA)를 활용하여 공정 능력 지수($C_p, C_{pk}$)를 산출
        - 우연 원인과 이상 원인을 구분
    - **다변량 통계적 공정 모니터링 (MSPC):**
        - 주성분 분석(PCA) 또는 부분 최소제곱법(PLS)을 사용 🡪 수십 개의 센서 변수 간 상관관계를 2~3개의 잠재 변수 공간으로 압축
        - $T^2$ 통계량(공정 변동 크기)과 $Q$ 통계량(변수 간 상관관계 깨짐)을 모니터링 🡪 평소와 다른 복합 이상 징후 포착
    - **근본 원인 분석 (Root Cause Analysis, RCA):**
        - 불량 제품 로트와 정상 로트 간 센서 피처 기여도 분석(SHAP Value, 변수 중요도 분석)을 통해
        - 불량을 유발한 핵심 공정 변수(예: 특정 노즐의 압력 저하)를 특정<br><br>

- **Step 5. 신사업 라인 최적화 및 예측 (Predictive Modeling & Optimization)**
    - 변경된 신규 제품 사양에 맞춰 최적의 설비 가동 조건을 도출하고 품질을 사전 예측

    1. **품질 예측 및 가상 계측 (Virtual Metrology):**
        - **입력:** 신규 라인의 원소재 스펙 + 실시간 가공 중 센서 피처
        - **출력:** 가공 직후 제품의 인장강도, 표면 조도, 치수 정밀도($Y$) 예측
        - **활용 모델:** Gradient Boosting(XGBoost, LightGBM), Random Forest, 1D-CNN
    2. **공정 파라미터 역최적화 (Prescriptive Optimization):**
        - 예측 모델 $Y = f(X_{제어}, X_{환경})$을 기반으로,
        - 목표 품질($Y^* $)을 달성하면서 에너지 소비나 사이클 타임을 최소화하는 최적 제어 입력값($X_{제어}^*$)을
        - 유전 알고리즘(GA) 또는 베이지안 최적화(Bayesian Optimization)로 탐색
        - 신규 설비 도입 시 수십 번의 시행착오(Trial & Error)를 디지털 시뮬레이션 기반으로 단축


## 4. 신사업 라인 구축 시 파이프라인에서 직면하는 핵심 난제와 해결책

| 파이프라인 단계 | 신사업/신규 설비 라인의 현실적 문제 | 엔지니어링 해결 전략 |
| :--- | :--- | :--- |
| **수집 단계** | 신규 설비와 구형 설비 간 인터페이스 프로토콜 불일치 | 엣지 게이트웨이 기반 OPC-UA 표준 노드화 |
| **전처리 단계** | 신규 공정이라 과거 결측·이상치 패턴 정의 부족 | 초기 룰 기반 임계치와 비지도 통계 필터 병행 적용 |
| **분석/진단 단계** | 신제품 양산 초기 불량 데이터(Defect Label) 절대 부족 | **정상 데이터 중심 이상치 탐지 (One-Class SVM, AutoEncoder)** |
| **예측/최적화 단계** | 과거 데이터가 없어 AI 지도학습 모델 학습 불가 | **유사 라인 전이 학습(Transfer Learning) 및 물리기반 모델(PINN) 결합** |

<br>

## 정리

> - **파이프라인은 단절 없는 연속 체계:**
>   - 수집이 잘못되면 전처리가 불가능하고,
>   - 전처리가 왜곡되면 AI 모델은 잘못된 최적화 값(Garbage In, Garbage Out)을 제시함
> - **도메인 지식과 데이터 사이언스의 결합:**
>   - 진동·압력 등의 원시 시계열 데이터를 물리적 의미를 지닌 '공정 피처'로 변환하는 피처 엔지니어링이 모델의 성능을 결정함
> - **신사업 라인의 성패는 '소량 데이터 대응력':**
>   - 사업재편 초기에는 불량 데이터가 극히 적으므로,
>   - 정상 데이터 기반 이상 탐지와 전이 학습을 적극 활용하여
>   - 신속하게 황금 레시피(Golden Recipe)를 구축해야 함
{: .green-quote}
