---
layout: page
title: (실습) 데이터 분석 Skill 개발
subtitle: 
permalink: /materials/S06-05-03-05_01-DataAnalysisSkills(Orange3)
description: 반복되는 공정 데이터 분석 작업 및 사업재편 보고서 수집을 자동화하는 Skill 제작
categories:
    - materials
tags:
    - AI 활용 및 자동화
    - 공정 개선
    - AI Utilization and Automation
    - Process Improvement
related_posts:
    - /materials/S06-05-03-01_01-DataDrivenBizOptimization
    - /materials/S06-05-03-02_01-BizRestrProcPredictionModels
    - /materials/S06-05-03-03_01-NoCodePredictionModels
    - /materials/S06-05-03-04_01-AiAgentsPoweredByCuttingEdgeAi
    - /materials/S06-05-03-06_01-BizRestrProcImprovementScenarios(Orange3)
qna: true
---
* toc
{:toc}




## 1. 실습 개요 및 실습 목표

- **실습 개요**
    - 사업재편을 추진하는 제조 현장에서는 매일 신규 라인에서 쏟아지는 수백 건의 공정 로그를 내려받아
        - **① 불량 위험 배치 선별**, **② 주요 결함 원인 분석**, **③ 경영진 일일 보고용 피벗 집계표 작성**을 수작업으로 반복하고 있음
    - 본 실습은 복잡한 프로그래밍(Python)이나 유료 클라우드(Dify 등) 없이, 오픈소스 GUI 분석 도구인 **Orange 3**만을 활용하여
        - 현업 담당자가 매일 반복하는 공정 분석 및 보고서 수집 업무를
        - 단 1회의 파일 교체만으로 완결하는 2대 핵심 No-Code Skill(위젯 파이프라인)을 직접 구축함

- **실습 목표**
    - **No-Code Skill의 개념 체득:**
        - AI 에이전트의 'Skill'이 거창한 코딩이 아니라, 목적에 맞게 입출력이 정의된 **재사용 가능한 위젯 파이프라인**임을 이해함
    - **Skill 1 (불량 고위험군 자동 선별):**
        - 학습된 머신러닝 모델(`Random Forest`)과 신규 데이터를 연결하여
        - 불량 확률 60% 이상인 위험 배치를 자동으로 걸러내는 필터링 Skill을 완성
    - **Skill 2 (사업재편 일일 보고서 집계):**
        - 작업조(Shift)별, 결함 유형(Defect Type)별 불량 건수를 자동으로 요약 집계하고
        - CSV/엑셀 파일로 자동 추출하는 리포팅 Skill을 완성
    - **완전 자동화(Zero-Cost) 구현:**
        - 비용 0원, 인터넷 연결 불필요 환경에서 실무 데이터를 즉시 처리할 수 있는 운영 능력 확보

<br>

## 2. 실습 환경 및 준비 데이터

- **실습 환경**
    - **도구:** Orange 3 (최신 안정 버전)
    - **비용:** 0원 (오픈소스 영구 무료, 가입/크레딧 불필요)
    - **환경:** 윈도우/맥/리눅스 일반 PC 어디서나 로컬 실행 가능

- **실습 데이터셋 구성**
    - **`S06-05-03-03_01-process_yield_dataset.csv` (1,200건):** 앞선 실습에서 모델 학습에 사용한 과거 이력 데이터 (학습용)
    - **`S06-05-03-05_01-today_new_batches.csv` (80건):** 사업재편 신규 라인에서 오늘 오전 생산된 신규 배치 로그 (Skill 검증용 신규 데이터)
        - [신규 데이터 구조: `S06-05-03-05_01-today_new_batches.csv`]

            | 컬럼명 | 데이터 타입 | 설명 및 단위 |
            | --- | --- | --- |
            | `batch_id` | Text | 신규 생산 배치 번호 (LOT_2610_001 ~ 080) |
            | `raw_tensile_strength` | Numeric | 원자재 인장강도 (MPa) |
            | `raw_thickness` | Numeric | 원자재 두께 (mm) |
            | `press_pressure` | Numeric | 프레스 성형 압력 (ton) |
            | `die_temp` | Numeric | 금형 가열 온도 (℃) |
            | `cool_rate` | Numeric | 급랭 냉각 속도 (℃/s) |
            | `operator_shift` | Categorical | 작업 조 (Day, Evening, Night) |
            | `defect_type` | Categorical | 현장 가검사 결함 유형 (None, Crack, UnderFill, Burr) |

    - [💾 파일 다운로드(S06-05-03-03_01-process_yield_dataset)](/materials/datasets/S06-05-03-03_01-process_yield_dataset.csv)
    - [💾 파일 다운로드(S06-05-03-05_01-today_new_batches.csv)](/materials/datasets/S06-05-03-05_01-today_new_batches.csv)

        ```python
        import numpy as np
        import pandas as pd

        # 난수 시드 고정 (재현성 확보)
        np.random.seed(2026)
        n_batches = 80

        # 1. 배치 식별자 생성
        batch_ids = [f"LOT_2610_{i+1:03d}" for i in range(n_batches)]

        # 2. 공정 입력 변수 생성 (기존 학습 데이터와 동일한 분포)
        raw_tensile_strength = np.random.normal(280, 15, n_batches).round(1)  # MPa
        raw_thickness = np.random.normal(2.0, 0.05, n_batches).round(3)       # mm
        press_pressure = np.random.uniform(130, 190, n_batches).round(1)      # ton
        die_temp = np.random.uniform(190, 250, n_batches).round(1)            # ℃
        cool_rate = np.random.uniform(12, 32, n_batches).round(1)             # ℃/s
        operator_shift = np.random.choice(["Day", "Evening", "Night"], size=n_batches, p=[0.4, 0.35, 0.25])

        # 3. 결함 유형(defect_type) 부여 (피벗 테이블 실습용)
        defect_types = []
        for t, p, c in zip(die_temp, press_pressure, cool_rate):
            if t < 210.0:
                defect_types.append("UnderFill")   # 저온 미성형
            elif t > 235.0:
                defect_types.append("Crack")       # 과열 균열
            elif p > 175.0:
                defect_types.append("Burr")        # 과가압 쇳물 누출
            elif c > 28.0:
                defect_types.append("Crack")       # 급랭 열응력 균열
            else:
                defect_types.append("None")        # 정상

        # 4. 데이터프레임 구성 및 CSV 저장
        df_today = pd.DataFrame({
            "batch_id": batch_ids,
            "raw_tensile_strength": raw_tensile_strength,
            "raw_thickness": raw_thickness,
            "press_pressure": press_pressure,
            "die_temp": die_temp,
            "cool_rate": cool_rate,
            "operator_shift": operator_shift,
            "defect_type": defect_types
        })

        # CSV 파일 저장 (한글 깨짐 방지를 위해 utf-8-sig 인코딩 사용)
        df_today.to_csv("S06-05-03-05_01-today_new_batches.csv", index=False, encoding="utf-8-sig")
        print("✅ 'S06-05-03-05_01-today_new_batches.csv' 파일 생성이 완료되었습니다. (총 80건)")
        ```

<br>

## 3. 전체 Skill 워크플로우 구성도

- Orange 3 캔버스에 구성할 완성 파이프라인의 전체 구조

    <div class="insert-image" style="width: 800px;">
        <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-010_ScreenCapture_10.png">
        <span class="caption">(Source: Sky Lectures / AiDALab)</span>
    </div>


## 4. 단계별 따라하기 실습 과정

- **[1단계] 기반 모델 준비 (Random Forest 학습)**
    1. 캔버스에 **`File`** 위젯을 놓고, 기존의 과거 학습 데이터 `S06-05-03-03_01-process_yield_dataset.csv`를 불러옴
        - `quality_status`: **Target** (Categorical)
        - `batch_id`, `yield_rate`: **Meta**
        - 나머지 공정 파라미터: **Feature**

    2. **`Model`** 탭에서 **`Random Forest`** 위젯을 꺼내 `File` 위젯과 연결
        - *(연결선: `Data` 🡪 `Random Forest`)* 🡪 불량을 89.3% 이상 잡아내는 주력 예측 엔진 준비 완료<br><br>


- **[2단계] Skill 1 구축: "불량 고위험군 자동 선별 및 조기 경보"**
    - 매일 아침 쏟아지는 수십 건의 신규 배치 중 "불량 발생 확률이 60% 이상인 위험 로트만 골라내는 도구"를 제작

    1. **신규 데이터 입력창 배치:**
        - 좌측 **`Data`** 탭에서 두 번째 **`File`** 위젯을 꺼내 캔버스에 놓기
        - 이름을 구분하기 쉽게 위젯 라벨을 마우스 우클릭하여 `File (오늘 신규 데이터)`로 변경
        - 더블클릭하여 오늘 생산 로그 파일 `S06-05-03-05_01-today_new_batches.csv`를 불러옴

    2. **예측 엔진(Predictions) 연동:**
        - 좌측 **`Evaluate`** 탭에서 **`Predictions`** 위젯을 꺼내 캔버스에 놓기
        - **[중요 - 2개 선 연결]:**
            - ① `Random Forest`의 우측 원 $\longrightarrow$ `Predictions`의 좌측 입력으로 연결 (신호: **`Model`**)
            - ② `File (오늘 신규 데이터)`의 우측 원 $\longrightarrow$ `Predictions`의 좌측 입력으로 연결 (신호: **`Data`**)

    3. **위험 배치 자동 필터링 (Select Rows 위젯):**
        - 좌측 **`Transform`** 탭에서 **`Select Rows`** 위젯을 꺼내어 `Predictions` 위젯 우측에 연결
        - `Select Rows`를 더블클릭하여 자동 필터 조건을 설정
            - **Conditions** 섹션에서 **`Add Condition`** 클릭
            - 첫 번째 드롭다운에서 **`Random Forest: Fail`** (불량 확률)을 선택
            - 조건을 `is`로 두고, 우측 값에 `Fail`을 입력

        <div class="insert-image" style="width: 800px;">
            <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-001_ScreenCapture_01.png">
            <span class="caption">(Source: Sky Lectures / AiDALab)</span>
        </div>


    4. **결과 확인 (Data Table):**
        - `Select Rows` 뒤에 **`Data Table`** 위젯을 연결하고 더블클릭
        - 전체 80건의 신규 배치 중 모델이 불량 위험 60% 이상으로 판정한 52건의 '긴급 조치 대상<br>
            🡪 로트 번호(LOT_xxxx)와 해당 온·습도 조건'만 화면에 즉시 분리되어 표시

            <div class="insert-image" style="width: 1000px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-002_ScreenCapture_02.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

        - `Prediction` 위젯에서 `Show probabilities for`의 값을 **`Classes known to the model`**로 설정하면<br>
            🡪 Fail을 설정한 확률 값을 확인할 수 있음

            <div class="insert-image" style="width: 600px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-003_ScreenCapture_03.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

            <div class="insert-image" style="width: 1000px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-004_ScreenCapture_04.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

            - 또한 Fail, Pass 외에 Fail/Pass 확률(0~1.0)값을 기준으로 결과를 도출할 수 있음

            <div class="insert-image" style="width: 600px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-007_ScreenCapture_07.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

            <div class="insert-image" style="width: 1000px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-008_ScreenCapture_08.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

    > **Skill 1의 가치:** 현장 작업자는 80개 배치를 일일이 검사할 필요 없이, 이 위젯에 뜬 로트만 즉시 선별 검사(샘플링)하여 시간과 비용을 80% 이상 절감
    {: .yellow-quote}

<br>

- **[3단계] Skill 2 구축: "사업재편 일일 공정 보고서 자동 집계 및 파일 저장"**

    - 걸러진 데이터와 오늘 생산 실적을 바탕으로
    - 경영진 보고용 "작업조(Shift)별 × 불량 유형(Defect)별 건수 매트릭스"를 자동으로 뽑아내는 도구

    1. **자동 피벗 집계 (Pivot Table 위젯):**
        - 좌측 **`Transform`** 탭에서 **`Pivot Table`** 위젯을 꺼내어 `Select Rows` (또는 `Predictions`) 뒤에 연결
        - `Pivot Table`을 더블클릭하여 보고서 서식을 구성
            - **Rows (행):** `defect_type` (결함 유형: UnderFill, Crack, Burr 등)
            - **Columns (열):** `operator_shift` (Day, Evening, Night)
            - **Values (값):** `Random Forest: Fail` (Fail에 대하여)
            - **Aggregation (집계함수):** `Count` (발생 건수), `Mean` (평균) 선택

        - 창 우측에 엑셀 피벗 테이블과 똑같은 "일일 불량 원인 분석 매트릭스 표"가 자동으로 완성됨

            > - 단순 불량 발생 개수를 보고서로 집계하려면: Values에 Random Forest를 두고 Aggregation = Count
            > - 교대조별 평균 불량 위험도(확률 %)를 보고서로 집계하려면: Values에 Random Forest: Fail을 두고 Aggregation = Mean
            {: .yellow-quote}

            <div class="insert-image" style="width: 600px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-005_ScreenCapture_05.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

    2. **보고서 파일 자동 추출 (Save Data 위젯):**
        - 좌측 **`Data`** 탭에서 **`Save Data`** 위젯을 꺼내어 `Pivot Table` 우측에 연결
        - `Save Data`를 더블클릭한 후 **[Save As...]** 버튼을 클릭
        - 파일 이름을 `사업재편_일일_공정_모니터링_보고서.csv`로 지정하고 저장함

            <div class="insert-image" style="width: 800px;">
                <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-006_ScreenCapture_06.png">
                <span class="caption">(Source: Sky Lectures / AiDALab)</span>
            </div>

<br>

- **[4단계] 자동화 운영 검증 (실무 시뮬레이션)**

    - 내일 새로운 날짜의 데이터(`tomorrow_batches.csv`)가 들어왔다고 가정함
    - **`File (오늘 신규 데이터)`** 위젯을 더블클릭하여 새 파일로 교체하기만 하면 끝
    - **결과:**
        - `Predictions` 🡪 `Select Rows` 🡪 `Pivot Table` 🡪 `Save Data`로 연결된 전체 파이프라인이 연속 갱신됨
        - 별도의 엑셀 작업이나 재계산 없이, 오늘 자 긴급 조치 대상 로트 목록과 새로운 일일 보고서 파일이 자동으로 완성됨

<br>

## 5. 실습 결과 산출물 및 실무 분석

- **산출물 1: 고위험 배치 조기 선별 리스트 (`Data Table` 출력)**
    - 전체 80건 중 불량 확률 60% 이상으로 자동 선별된 상위 배치 결과 예시:

        | batch_id         | die_temp (℃) | press_pressure (ton) | Random Forest: Fail 확률 | 조치 권고사항                               |
        | ---------------- | ------------ | -------------------- | ------------------------ | ------------------------------------------- |
        | **LOT_2610_012** | 211.2        | 164.5                | **0.88 (88%)**           | 저온 미성형 위험 🡪 히터 전력 즉시 보정      |
        | **LOT_2610_027** | 238.5        | 165.1                | **0.79 (79%)**           | 과열 크랙 위험 🡪 냉각 라인 밸브 개도율 상향 |
        | **LOT_2610_041** | 219.0        | 174.2                | **0.65 (65%)**           | 과가압 버(Burr) 위험 🡪 하사점 압력 감압     |

- **산출물 2: 사업재편 일일 공정 요약 매트릭스 (`Pivot Table` 출력)**

    | 불량 유형 (Rows)        | 주간 조 (Day) | 야간 조 (Night) | 합계 (Total) | 현장 원인 해석                            |
    | ----------------------- | ------------- | --------------- | ------------ | ----------------------------------------- |
    | **UnderFill (미성형)**  | 3건           | 8건             | **11건**     | 야간 기온 하강에 따른 초기 금형 예열 부족 |
    | **Crack (열응력 균열)** | 4건           | 1건             | **5건**      | 주간 조 가열로 과열 이탈                  |
    | **Burr (초과 가압)**    | 1건           | 1건             | **2건**      | 유압 밸브 미세 편차                       |
    | **합계**                | **8건**       | **10건**        | **18건**     | 야간 조 교대 전 예열 표준(SOP) 개정 필요  |

<br>

> - **실습의 의의**
>   - **진입 장벽 제로 (Zero-Barrier):**
>       - 파이썬 문법 에러나 유료 플랫폼(Dify) 크레딧 고갈 없이, 클릭과 선 잇기만으로 완성되므로 수강생 모두가 100% 실습을 완주 가능
>   - **에이전트 'Skill'의 직관적 이해:**
>       - 스킬이란 어려운 기술이 아니라, **새로운 입력이 들어왔을 때 정해진 지능(모델)과 룰에 따라 결과를 뽑아내는 자동화 블록**이라는 개념을 확인
>   - **현업 즉시 적용성:**
>       - 실습이 끝난 후 이 Orange 3 워크플로우 파일(`.ows`)을 그대로 저장하여 회사 공장의 실제 MES 로그에 연결하면, 당장 실무 자동화 도구로 쓸 수 있음
{: .green-quote}

<br>

> - **현장 MES 로그와 연결하여 "실무 자동화"로 쓰는 실제 운영 방식**
>   - '.ows' 파일 자체가 각 MES에 호환되는 것은 아님
>   - '.ows' 파일은 'XML' 형식으로 작성된 파일이므로 각 MES 시스템에 맞게 조정해야 함<br><br>
>
>   - **방법 1: "공유 폴더 기반의 배치(Batch) 모니터링" (비전문가/현장 추천)**
>       - MES 시스템을 건드릴 권한이 없는 일반 공정 기술팀이나 품질팀 PC에서 가장 많이 쓰는 실무 방식
>
>           <div class="insert-image" style="width: 700px;">
>               <img src="/materials/S06_SmartFactory/images/S06-05-03-05_01-009_ScreenCapture_09.png">
>               <span class="caption">(Source: Sky Lectures / AiDALab)</span>
>           </div>
>
>           1. 설정: .ows 파일 안의 File 위젯 경로를 MES가 주기적으로 덤프(Export)해주는 고정된 경로(예: Z:\MES_Export\latest_run.csv)로 지정
>           2. 운영: 현장 엔지니어는 아침에 출근해서 Orange 3의 .ows 파일을 열고, File 위젯의 [Reload] 버튼만 누르면(또는 위젯을 열고 닫으면)<br>
                     그날의 최신 MES 데이터를 읽어와서 자동으로 불량 예측과 피벗 보고서가 완성됨
>           3. 특징: 개발 지식이 전혀 없는 현장 엔지니어가 MES 개발팀의 협조 없이 즉시 자기 업무를 자동화할 때 사용<br><br>
>   - **방법 2: Python 환경으로의 전환 (IT/스마트팩토리 전산팀 연계)**
>       - Orange 3는 내부적으로 순수 파이썬 라이브러리(scikit-learn 등)를 기반으로 구동됨
>       - 현장 IT/MES 개발팀에 배포할 때는 .ows 파일 자체를 넘기는 것이 아니라:
>           1. Orange 3에서 검증된 최적 알고리즘(Random Forest)과 파라미터(트리 깊이, 주요 변수 가중치 등)를 확정
>           2. 이를 바탕으로 10~20줄 내외의 가벼운 Python 추론 스크립트(.py)나 모델 파일(.pkl, .joblib)로 변환하여 MES 백엔드 데몬에 탑재
>       - 즉, Orange 3와 .ows는 현업 실무자가 복잡한 코딩 없이 머신러닝의 로직과 공정 룰을 완벽하게 검증하고 PoC(개념 증명)를 끝내는<br>
              "프로토타입 도구"로서 결정적인 역할을 수행함
{: .green-quote}