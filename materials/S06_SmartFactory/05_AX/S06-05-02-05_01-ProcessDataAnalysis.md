---
layout: page
title: (실습) 사업재편 공정 데이터 분석
subtitle: 
permalink: /materials/S06-05-02-05_01-ProcessDataAnalysis
description: 실제 제조 데이터셋을 활용해 신규 공정 전환 시 발생할 수 있는 문제점 진단 및 공유
categories:
    - materials
tags:
    - 인공지능
    - AI
    - 공정 데이터 분석
    - Process Data Analysis
related_posts:
    - /materials/S06-05-02-01_01-AiAndManufacturingData
    - /materials/S06-05-02-02_01-EquipmentDataCollection
    - /materials/S06-05-02-03_01-DataDrivenProcessDiagnosis
    - /materials/S06-05-02-04_01-DataQualityManagement1
    - /materials/S06-05-02-04_02-DataQualityManagement2
qna: false
---
* toc
{:toc}



> **[실습 주제] 제조 AI 활용: 신규 공정 전환 리스크 진단 및 데이터 정제 실습**
{: .yellow-quote}


## 1. 실습 개요

- **실습명**: 생성형 AI(대화형 데이터 분석기) 및 노코드 AI를 활용한 신규 공정 데이터 이상 진단
- **실습 목적**:
    - 파이썬 코딩 없이 생성형 AI(ChatGPT Advanced Data Analysis / Claude Artifacts / 사내 LLM 분석기)를 "AI 데이터 분석 비서"로 활용하는 실무 역량 배양
    - 사람이 일일이 찾기 힘든 다변량 센서 이상치(센서 엇박자)와 **결측치**를 AI에게 감지·정제하도록 지시하는 프롬프트 엔지니어링 체득
    - 머신러닝 기반 중요도(Feature Importance) 및 의사결정 규칙 도출을 AI에게 지시하여, 공정 불량을 유발하는 핵심 제어 임계치를 도출하고 최종 보고서 자동 작성

<br>

## 2. 실습 환경 및 도구

- **기본 도구 (택 1 또는 병행)**:
    - **옵션 A (추천: 생성형 AI 기반)**: ChatGPT Plus/Team (Data Analysis 기능) 또는 Claude (Projects/Artifacts)
    - **옵션 B (설치형 노코드 AI 도구)**: Orange Data Mining (마우스 드래그 앤 드롭으로 AI 모델을 조립하는 GUI 무료 도구)

- **제공 데이터셋**: `smart_factory_transformation_data.csv` (1,000건, 결측치 및 계측 노이즈 포함)

<br>

## 3. 실습 시나리오

> - **AI 분석 비서와 함께하는 공정 문제 진단**<br><br>
> - **현장 상황**
>   - 내연기관 실린더 블록(주철) 라인을 친환경 배터리 팩 케이스(알루미늄) 성형 라인으로 사업재편한 뒤, 3주 만에 수율이 76%로 급락
>   - 현장 담당자는 엑셀로 원인을 찾지 못해, 1,000건의 센서 데이터가 담긴 CSV 파일을 **제조 전문 AI 분석 에이전트**에 업로드하고
>   - 결측치 정제부터 불량 원인 분석, 경영진 보고서 초안까지 단 30분 만에 완성하고자 함
{: .yellow-quote}

<br>

## 4. 단계별 실습 진행 과정 (자연어 프롬프트 기반 AI 분석)

- 아래의 [공정 도메인 기반 AI 지시 프롬프트]를 순차적으로 입력
- AI가 수행한 분석 결과와 시각화 차트를 검증

    ```
    [1단계: AI 데이터 품질 감사 및 결측치/노이즈 자동 정제]
                        🡫
    [2단계: AI 기반 공정 능력(Cpk) 및 지표 산포 비교 진단]
                        🡫
    [3단계: 머신러닝 이상 탐지(Isolation Forest)로 엇박자 센서 색출]
                        🡫
    [4단계: AI 불량 기여도 분석(SHAP/트리)으로 설비 제어 임계치 도출]
    ```

- **1단계: AI 데이터 품질 감사 (결측치 및 계측 노이즈 자동 정제)**

    - AI에 CSV 파일을 드래그 앤 드롭한 뒤 아래 프롬프트 입력

    - **입력 프롬프트**:
        > 이 파일은 제조 공정 센서 데이터야. 본격적인 분석에 앞서 **데이터 품질 감사(Data Quality Audit)**를 수행해 줘.
        > 1. 결측치(빈칸)가 있는 센서와 행 번호를 찾고,
        > 2. 물리적으로 불가능한 센서 노이즈(예: 비정상 고온 900도 이상, 음수 압력 등)를 자동 감지해 줘.
        > 3. 발견된 결측치는 직전/직후 센서 추세를 반영해 선형 보간하고, 계측 노이즈는 정상 센서 범위(도메인 기준)로 클렌징한 정제 데이터셋을 메모리에 생성해 줘.
        {: .gray-quote}

    - **AI의 수행 및 출력**:
        - 결측치 15건(온도 8건, 압력 7건) 위치 표기 및 보간 완료 안내
        - 센서 튐(999.0°C, -50.0 Bar 등 5건)을 탐지하고 이상치로 플래그 처리한 내역 시각화

<br>

- **2단계: AI 기반 공정 능력(Cpk) 및 지표 산포 비교 진단**

    - **입력 프롬프트**:
        > 정제된 데이터를 바탕으로 `Process_Type`별(Legacy_Process vs New_Transformation) 비교 분석을 수행해 줘.
        > 1. 두 공정의 수율(Yield)과 불량률을 계산하고,
        > 2. 온도, 압력, 진동, 사이클 타임의 평균 및 표준편차 차이를 한눈에 볼 수 있는 비교 시각화 차트(Boxplot 또는 바이올린 플롯)를 그려줘.
        > 3. 신규 공정에서 '산포(변동폭)'가 가장 심각하게 벌어진 센서가 무엇인지 요약해 줘.
        {: .gray-quote}

    - **AI의 수행 및 출력**:
        - 수율 비교: 기존 97% 🡪 신규 76% (21%p 하락 확인)
        - 상자수염 차트 생성 및 "신규 라인에서 유압 압력의 표준편차가 3.7배, 금형 온도가 2.3배 폭증하여 공정 능력이 급격히 상실됨"을 텍스트로 브리핑

<br>

- **3단계: 머신러닝 기반 이상 탐지 (설비 스트레스·엇박자 구간 도출)**

    - 단순히 상하한선(Spec Out)을 넘은 것 외에,
    - "압력과 온도가 정상 범위 내에 있지만 서로 엇박자를 내는 다변량 이상"을 AI에게 찾도록 함

    - **입력 프롬프트**:
        > 신규 공정(`New_Transformation`) 데이터에 **비지도학습 이상치 탐지 알고리즘(Isolation Forest)**을 적용해 줘.<br>
        > 단일 센서로는 정상 범위처럼 보이지만, 여러 센서가 결합했을 때 비정상적인 스트레스를 유발하는 공정 구간을 탐지해서 산점도(Scatter Plot)에 표시해 줘.<br>
        > 이상으로 분류된 구간들의 공통적인 설비 특징(온도-압력 패턴)이 무엇인지 설명해 줘.
        {: .gray-quote}

    - **AI의 수행 및 출력**:
        - Isolation Forest 알고리즘 자동 실행
        - 압력 135 Bar 이상이면서 온도가 198°C를 넘는 설비 과부하 구간을 붉은색 이상 클러스터로 시각화

<br>

- **4단계: 머신러닝 설명 가능 AI(XAI)로 설비 제어 임계치 도출**

    - **입력 프롬프트**:
        > 머신러닝 분류 모델(Decision Tree 또는 Random Forest)을 사용해 `Defect_Flag`(불량)에 가장 큰 영향을 미치는 핵심 센서 3가지를 찾고 **특성 중요도(Feature Importance)**를 시각화해 줘.<br>
        > 그리고 현장 작업자가 설비 PLC에 즉시 입력할 수 있도록, **'불량을 회피하기 위한 최적의 안전 운전 조건(온도 상한, 압력 상한 임계치)'**을 의사결정 규칙 형태로 도출해 줘.
        {: .gray-quote}

    - **AI의 수행 및 출력**:
        - 특성 중요도 1위: `Pressure_Bar` (42%), 2위: `Temperature_C` (35%)
        - 도출된 현장 규칙:
            - **[위험]** 성형 압력 > 136.2 Bar AND 금형 온도 > 200.5°C 🡪 **불량률 88.4%**
            - **[안전]** 성형 압력 $\le$ 135.0 Bar 유지 시 🡪 **불량률 4.1% 수준으로 급감**

<br>

## 5. 산출물: 공정 진단 보고서 검증 및 보완

- 다음의 프롬프트를 입력하여 **경영진 보고용 표준 양식**으로 초안을 작성한 뒤,
- 현장 엔지니어의 관점에서 타당성을 검증하고 최종 수정본을 완성할 것

- **보고서 생성 프롬프트**:
    > 지금까지 분석한 모든 결과를 종합하여, 경영진과 생산본부장에게 보고할 **'사업재편 신규 공정 조기 안정화 AI 진단 보고서'**를 작성해 줘.<br>
    > 현황 요약, 수율 급락의 근본 원인(Root Cause), AI 모델이 도출한 안전 임계치, 그리고 설비/공정 관점의 즉각적인 액션 플랜(Action Plan)을 포함한 마크다운 문서로 출력해 줘.
    {: .gray-quote}

<br>

- **엑셀 방식 대비 기대 효과**

    | 구분                    | 기존 엑셀 방식                                | **제조 AI(프롬프트/노코드) 활용 방식**                               |
    | :---------------------- | :-------------------------------------------- | :------------------------------------------------------------------- |
    | **실습생의 역할**       | 마우스 클릭, 수식 작성, 필터링 등 단순 노가다 | **"AI에게 문제를 정의하고, 결과를 해석해 조치안을 결정하는 관리자"** |
    | **다루는 기술**         | 단순 통계(평균, 합계), 2차원 차트             | **선형 보간, Isolation Forest, Feature Importance, XAI 규칙 도출**   |
    | **교육 주제 부합도**    | 일반 OA / 사무 실습에 가까움                  | **"제조 산업에서의 AI 활용" 주제에 100% 부합**                       |
    | **난이도 및 소요 시간** | 코딩이 없어 비전문가도 100% 수행 가능         | 동일하게 비코딩이지만, 분석 결과물 완성 속도는 3배 이상 빠름         |

<br>

- **데이터셋 생성 파이썬 코드**

```python
import numpy as np
import pandas as pd

def generate_transformation_dataset(n_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    half = n_samples // 2
    
    # 1. 시계열 타임스탬프 (2분 간격)
    timestamps = pd.date_range(start="2026-03-01 08:00:00", periods=n_samples, freq="2min")
    
    # 2. 공정 구분 (전반 500개: 기존 주철 공정 / 후반 500개: 신규 알루미늄 전환 공정)
    process_type = ['Legacy_Process'] * half + ['New_Transformation'] * half
    
    # 3. 핵심 센서 데이터 기본 생성
    # - 기존 공정: 좁은 산포, 안정적 운영
    # - 신규 공정: 평균 상승, 산포 급증(제어 미숙)
    temp_legacy = np.random.normal(loc=180.0, scale=3.5, size=half)
    temp_new = np.random.normal(loc=198.5, scale=8.2, size=half)
    temp = np.concatenate([temp_legacy, temp_new])
    
    press_legacy = np.random.normal(loc=120.0, scale=2.0, size=half)
    press_new = np.random.normal(loc=135.0, scale=7.5, size=half)
    press = np.concatenate([press_legacy, press_new])
    
    vib_legacy = np.random.normal(loc=15.0, scale=1.2, size=half)
    vib_new = np.random.normal(loc=24.0, scale=4.8, size=half)
    vib = np.concatenate([vib_legacy, vib_new])
    
    cycle_legacy = np.random.normal(loc=45.0, scale=1.0, size=half)
    cycle_new = np.random.normal(loc=58.0, scale=5.0, size=half)
    cycle = np.concatenate([cycle_legacy, cycle_new])
    
    # 두께 편차 (mm)
    thick_err_legacy = np.random.normal(loc=0.02, scale=0.01, size=half)
    thick_err_new = (
        0.05 + 0.003 * (temp_new - 190) + 0.004 * (press_new - 130) 
        + np.random.normal(loc=0, scale=0.03, size=half)
    )
    thickness_error = np.concatenate([thick_err_legacy, thick_err_new])
    
    # 4. 불량 판정 (Defect_Flag: 0=정상, 1=불량)
    # 기존: 불량률 ~3%
    defect_legacy = np.random.choice([0, 1], size=half, p=[0.97, 0.03])
    
    # 신규: 압력 > 136 Bar 및 온도 > 200°C 복합 과부하 시 불량률 85% 이상 치솟음
    new_prob = 1 / (1 + np.exp(-(0.18 * (temp_new - 198) + 0.22 * (press_new - 135) + 0.15 * (vib_new - 24))))
    new_prob = np.clip(new_prob, 0.05, 0.95)
    defect_new = (np.random.rand(half) < new_prob).astype(int)
    defect = np.concatenate([defect_legacy, defect_new])
    
    # 데이터프레임 초기화
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Process_Type': process_type,
        'Temperature_C': np.round(temp, 2),
        'Pressure_Bar': np.round(press, 2),
        'Vibration_mm_s': np.round(vib, 2),
        'Cycle_Time_s': np.round(cycle, 2),
        'Thickness_Error_mm': np.round(thickness_error, 4),
        'Defect_Flag': defect
    })
    
    # -------------------------------------------------------------
    # [데이터 품질 관리 실습용 주입 요소: 결측치 및 계측 노이즈]
    # -------------------------------------------------------------
    # (1) 통신 끊김으로 인한 결측치(NaN/빈칸) 주입: 15개 행
    missing_indices = np.random.choice(range(50, 950), size=15, replace=False)
    df.loc[missing_indices[:8], 'Temperature_C'] = np.nan
    df.loc[missing_indices[8:], 'Pressure_Bar'] = np.nan
    
    # (2) 센서 노이즈/전압 튐 이상치 (물리적으로 불가능한 극단값): 5개 행
    noise_indices = [120, 310, 680, 750, 890]
    df.loc[noise_indices[0], 'Temperature_C'] = 999.0   # 센서 통신 에러 고정값
    df.loc[noise_indices[1], 'Pressure_Bar'] = -50.0     # 마이너스 압력 에러
    df.loc[noise_indices[2], 'Vibration_mm_s'] = 250.0   # 순간 정전기 노이즈
    df.loc[noise_indices[3], 'Temperature_C'] = 0.0      # 영점 탈락 에러
    df.loc[noise_indices[4], 'Pressure_Bar'] = 999.9     # 전압 스파이크
    
    return df

if __name__ == '__main__':
    df = generate_transformation_dataset()
    df.to_csv('smart_factory_transformation_data.csv', index=False)
    print("실습용 데이터셋 생성 완료: smart_factory_transformation_data.csv")
    print(f"- 전체 데이터: {len(df)}건")
    print(f"- 결측치 개수: {df.isna().sum().sum()}개")
    print(f"- 기존 공정 수율: {((1 - df[df['Process_Type']=='Legacy_Process']['Defect_Flag'].mean())*100):.1f}%")
    print(f"- 신규 공정 수율: {((1 - df[df['Process_Type']=='New_Transformation']['Defect_Flag'].mean())*100):.1f}%")
```