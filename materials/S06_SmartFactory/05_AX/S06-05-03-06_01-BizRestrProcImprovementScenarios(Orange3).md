---
layout: page
title: (실습) 사업재편 공정 개선 시나리오 설계
subtitle: 
permalink: /materials/S06-05-03-06_01-BizRestrProcImprovementScenarios(Orange3)
description: 예측 결과를 기반으로 현장 알람/작업 표준(SOP) 변경 및 의사결정 자동화 시나리오 작성
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
    - /materials/S06-05-03-05_01-DataAnalysisSkills(Orange3)
qna: false
---
* toc
{:toc}





## 1. 실습 개요 및 교육 목적

- **실습 배경**
    - 사업재편(내연기관 부품 🡪 친환경 배터리 팩 알루미늄 하우징 성형) 공정에서는 설비 파라미터(온도, 압력 등)의 미세한 편차가 대형 결함으로 이어짐
    - 단순히 "불량이 발생할 것 같다"는 머신러닝 예측 수치만으로는 공장을 제어할 수 없음
    - 예측값을 바탕으로
        - **① 현장 누구에게 어떤 단계의 알람을 울릴지**
        - **② 작업 표준(SOP)을 어떻게 동적으로 변경할지**
        - **③ 언제 설비를 자동 정지(Interlock)시킬지**
    - 에 대한 명확한 **의사결정 시나리오 체계**가 수립되어야 비로소 실질적인 공정 자동화(AX)가 완성됨

- **실습 목표**
    1. **예측 모델과 의사결정의 연결:**
        - Orange 3의 `Predictions` 위젯에서 출력되는 **불량 확률(Fail Probability)** 수치를
        - 위험도 4단계(정상/주의/경고/비상) 임계치와 직접 매핑
    2. **트리 기반 동적 SOP 도출:**
        - `Tree Viewer`의 주요 분기 조건(`die_temp`, `press_pressure` 등)을 해석하여,
        - 결함 유형별 물리적 긴급 조치 파라미터를 규정
    3. **최종 산출물 완성:**
        - Orange 3의 시각적 분석 결과를 바탕으로,
        - 현업 부서에서 즉시 배포하여 사용할 수 있는 [사업재편 공정 개선 의사결정 자동화 시나리오 정의서]를 완성

<br>

## 2. Orange 3 워크플로우 구성 및 연동 체계

- 본 실습은 앞선 모델링 실습 결과(Orange 3)를 의사결정 시나리오의 기초 입력 데이터로 활용함

    <div class="insert-image" style="width: 700px;">
        <img src="/materials/S06_SmartFactory/images/S06-05-03-06_01-001_Pipeline.png">
        <span class="caption">(Source: Sky Lectures / AiDALab)</span>
    </div>


## 3. 단계별 실습 수행 절차

- **[1단계] 기반 모델 준비 (과거 데이터 로드 및 Random Forest 설정)**

    1. **`File` 위젯 배치 (File 1):**
        - 좌측 **`Data`** 탭에서 **`File`** 위젯을 꺼내 캔버스에 놓기
        - 더블클릭하여 **`process_yield_dataset.csv`** (과거 학습 데이터 1,200건)를 불러오기
        - **컬럼 속성(Role) 설정:**
            - `batch_id`: **Text / Meta**
            - `raw_tensile_strength` ~ `cool_rate`: **Numeric / Feature**
            - `operator_shift`: **Categorical / Feature**
            - **`yield_rate`**: **Numeric / Meta** (※ 치팅 방지를 위해 반드시 Meta 지정)
            - **`quality_status`**: **Categorical / Target**
        - 하단 **`Apply`** 버튼을 클릭하고 창 닫기

    2. **`Random Forest` 위젯 배치 및 연결:**
        - 좌측 **`Model`** 탭에서 **`Random Forest`** 위젯을 꺼내 캔버스에 놓기
        - `File 1`의 우측 점 🡪 **`Random Forest`** 좌측 점으로 선을 연결 (신호: **`Data`**)

    3. **`Random Forest` 내부 옵션 설정:**
        - **`Random Forest`** 위젯을 더블클릭하여 열기
        - 기본 설정을 확인
            - **`Number of trees`**: `10` (기본값)
            - **`Number of attributes considered at each split`**: 기본값 유지<br><br>


- **[2단계] 신규 데이터 로드 및 불량 위험도(확률) 예측**
    1. **`File` 위젯 배치 (File 2 - 신규 데이터용):**
        - 좌측 **`Data`** 탭에서 두 번째 **`File`** 위젯을 꺼내 캔버스에 놓기
        - 더블클릭하여 **`today_new_batches.csv`** (신규 80건)를 불러오기
        - **컬럼 속성(Role) 설정:**
            - `batch_id`: **Text / Meta**
            - `raw_tensile_strength` ~ `cool_rate`: **Numeric / Feature**
            - `operator_shift`: **Categorical / Feature**
            - `defect_type`: **Categorical / Meta** (피벗 집계용)

    2. **`Predictions` 위젯 배치 및 2개 선 연결:**
        - 좌측 **`Evaluate`** 탭에서 **`Predictions`** 위젯을 꺼내 캔버스에 놓기
        - **선 연결 1 (모델 전달):** `Random Forest` 우측 점 🡪 `Predictions` 좌측 점 (신호: **`Model`**)
        - **선 연결 2 (데이터 전달):** `File 2 (신규 데이터)` 우측 점 🡪 `Predictions` 좌측 점 (신호: **`Data`**)

    3. **`Predictions` 위젯에서 불량 확률(Probability) 활성화:**
        - **`Predictions`** 위젯을 더블클릭하여 열기
        - 좌측 패널 상단의 **`Show probabilities for`** 체크
        - 드롭다운에서 **`Classes known to the model`** 선택 확인<br>
            🡪 모델이 알고 있는 모든 클래스의 확률을 한 번에 전부 계산하여 표시함

    4. **`Data Table` 위젯으로 위험 등급 확인:**
        - 좌측 **`Data`** 탭에서 **`Data Table`** 위젯을 꺼내 `Predictions` 우측에 연결
        - 더블클릭하여 열고, `Random Forest: Fail` 컬럼 헤더를 클릭해 정렬해 보며 배치별 불량 확률을 확인
            - **정상:** $< 0.20$
            - **주의:** $0.20 \le \text{확률} < 0.40$
            - **경고:** $0.40 \le \text{확률} < 0.70$
            - **비상:** $\ge 0.70$<br><br>


- **[3단계] `Tree Viewer`를 통한 결함별 긴급 조치 룰(SOP) 추출**
    1. **`Tree` 위젯 배치 및 연결:**
        - 좌측 **`Model`** 탭에서 **`Tree`** 위젯을 꺼내 캔버스에 놓기
        - 첫 번째 **`File 1 (과거 데이터)`** 우측 점 🡪 **`Tree`** 좌측 점으로 선을 연결 (신호: **`Data`**)

    2. **`Tree Viewer` 위젯 배치 및 연결:**
        - 좌측 **`Visualize`** 탭에서 **`Tree Viewer`** 위젯을 꺼내 놓기
        - **`Tree`** 위젯 우측 점 🡪 **`Tree Viewer`** 좌측 점으로 선을 연결 (신호: **`Tree`**)

    3. **`Tree Viewer` 화면 설정:**
        - **`Tree Viewer`** 위젯을 더블클릭
        - 좌측 패널 옵션을 변경:
            - **`Depth`**: `3` 또는 `4`로 지정 (화면 축약)
            - **`Target class`**: `Fail` 선택 (불량 노드 파란색 강조)
            - **`Node labels`**: `(None)` 선택 (LOT 번호 숨김)

    4. **공정 룰(Rule) 확인 및 메모:**
        - 파란색 100% Fail 노드로 이어지는 상단 분기 조건의 수치를 읽고 기록:
            - **저온 미성형 (UnderFill):** 금형 온도(`die_temp`) $\le 205℃$ (또는 $210℃$)
            - **과열 크랙 (Crack):** 금형 온도(`die_temp`) $\ge 235℃$
            - **과가압 버 (Burr):** 프레스 압력(`press_pressure`) $\ge 175\text{ton}$<br><br>

    > - **노드 안에서 읽어야 할 핵심 정보**
    >   - **상자 바로 위 연결선에 적힌 조건문 (가장 중요):**
    >       - 예: `die_temp ≤ 205.0`, `die_temp > 235.0`, `press_pressure > 175.0`<br>🡪 **이 조건문이 바로 SOP 규정에 들어갈 물리적 임계치 수치**
    >   
    >   - **상자 색상의 진한 정도:**
    >       - `Target class`를 `Fail`로 두었기 때문에, **파란색이 진할수록 불량률이 90~100%에 달하는 고위험 구간**
    >   
    >   - **상자 내부의 비율(Distribution):**
    >       - 노드 안에 표시된 `Fail: 180 / 185 (97%)` 같은 비율을 통해 해당 온도/압력 구간에서 불량이 얼마나 집중 발생하는지 한눈에 확인할 수 있음
    {: .pink-quote}

<br>

- **[4단계] [공정 개선 의사결정 자동화 시나리오 정의서] 완성**
    - 위 2단계(확률 구간)와 3단계(트리 분기 수치)에서 확인한 값을 아래 정의서 표준 양식에 기록하여 최종 산출물 작성


```markdown
# [사업재편 공정 개선 의사결정 자동화 시나리오 정의서]
■ 대상 공정: 친환경 배터리 팩 하우징 성형 라인 (Line-B)
■ 연동 도구: Orange 3 예측 엔진 (Random Forest) ──► 현장 HMI / PLC 연동

1. 위험 수준별 현장 알람 매트릭스 (Predictions 확률 연동)
--------------------------------------------------------------------------------------------------------------------------------------------------
위험 등급   판정 조건 (Random Forest: Fail)     알람 수신자 및 매체         제어 방식           설비 및 조업 지침
--------------------------------------------------------------------------------------------------------------------------------------------------
[1] 정상    불량 확률 < 20% (0.20 미만)         현장 모니터 (녹색등)        완전 자동           현행 조업 레시피 유지
[2] 주의    20% ≤ 불량 확률 < 40%               작업자 HMI (황색 경광등)    완전 자동           샘플링 검사 주기 단축 (5배치당 1개 → 2배치당 1개)
[3] 경고    40% ≤ 불량 확률 < 70%               현장 조장/반장 (PDA 알림)   작업자 1클릭 승인   HMI 동적 SOP 팝업 확인 터치 시 파라미터 보정
[4] 비상    불량 확률 ≥ 70% (연속 2회 경고)     라인장, 품질팀장 (SMS)      시스템 선제 정지    설비 PLC 인터록 자동 비상 정지
--------------------------------------------------------------------------------------------------------------------------------------------------

2. 파라미터 이탈 유형별 동적 작업 표준(SOP) (Tree Viewer 분기 연동)
--------------------------------------------------------------------------------------------------------------------------------------------------
주요 이탈 변수      예상 결함 유형      적용 SOP 코드   현장 파라미터 동적 보정 세부 지침
--------------------------------------------------------------------------------------------------------------------------------------------------
die_temp < 205℃     UnderFill           SOP-B-TEMP-01   ① 예열 챔버 히터 출력 +8% 상향 조정
(금형 저온 이탈)    (미성형 불량)                       ② 서보 프레스 가압 유지시간 1.0초 연장
die_temp > 235℃     Crack               SOP-B-TEMP-02   ① 냉각 라인 밸브 개도율 15% 상향
(금형 과열 이탈)    (열응력 균열)                       ② 성형 직후 급랭 팬 가동 시간 2초 추가 확보
press_pressure      Burr                SOP-B-PRESS-01  ① 서보 모터 최대 가압력을 165ton으로 하향 제한
> 175ton (과가압)   (쇳물 누출/단차)                    ② 금형 파팅 라인 유격 점검
--------------------------------------------------------------------------------------------------------------------------------------------------
```

<br>

## 5. 실습 평가 및 체크포인트

> - [ ] Orange 3 `Predictions` 위젯에서 `Probabilities` 옵션을 켜고, 각 배치의 불량 확률 수치(`Random Forest: Fail`)를 확인했는가?
> - [ ] `Tree Viewer` 위젯의 분기 조건을 추적하여 불량이 발생하는 구체적인 수치 한계선(온도, 압력 등)을 파악했는가?
> - [ ] 위험 등급(정상/주의/경고/비상)에 따른 알람 대상자와 시스템 제어 방식(완전자동 vs 1클릭 승인 vs 인터록 정지)을 명확히 구분했는가?
> - [ ] 작성된 시나리오 정의서가 실제 공장 라인에 즉시 배포 가능한 수준의 실무 규칙을 담고 있는가?
{: .gray-quote}