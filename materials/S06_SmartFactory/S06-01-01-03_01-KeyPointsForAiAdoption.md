---
layout: page
title:  "현장 적용 관점에서의 AI 도입 포인트"
date:   2026-07-21 11:30:00 +0900
permalink: /materials/S06-01-01-03_01-KeyPointsForAiAdoption
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> **핵심 메시지:** 
> - AI 모델의 정확도가 99%여도 현장에 안착하지 못하면 쓰레기통으로 감
> - 스마트 제조 AI의 성공은 알고리즘이 아니라 **현장 도메인 지식, 데이터의 무결성, 그리고 사람(작업자)과의 상호작용**에 달려 있음
{: .expert-quote}


## 1. PoC의 함정과 Pilot에서 Roll-out까지의 장벽

<div class="info-table"><b>[용어]</b> PoC (Proof of Concept): 개념 검증 / Roll-out: 전사 확대</div><br>

> - **Point:** &nbsp;&nbsp; 실험실(Colab/Jupyter Notebook)에서 잘 돌아가는 AI 모델이 <span style="color: darkred;">실제 공장에 적용되면 성능이 급격히 떨어지는 현상</span> 이해시키기
{: .common-quote}

- **주요 내용:**
    - **PoC 잔혹사 (PoC Trap):**
        - 정제된 과거 데이터 10,000건으로 학습시킨 모델이 현장의 실시간 스트리밍 데이터와 만나면 터지는 이유

    - **환경 변수의 차이:**
        - 온·습도 변화, 원자재 롯데(Lot) 변경, 설비 마모, 작업자 교체 등 실험실 데이터에 없던 변수 발생

- **💡 현장 예시 (사출 성형 공정):**
    - **상황:** 
        - 여름철 데이터로 완벽하게 학습된 사출 성형 불량 예측 AI가 겨울철에 불량 폭탄을 만들어냄
    - **원인:**
        - 공장 내부 외기 온도 변화로 수지의 냉각 속도가 달라졌으나, 
        - AI 모델에 '공장 환경 온도' 데이터가 반영되지 않았음
    - **도입 포인트:**
        - **데이터의 계절성(Seasonality) 및 최악의 조건(Corner Case) 반영 필수**



## 2. ROI 산정과 문제지점의 명확화

<div class="info-table"><b>[용어]</b> ROI(Return On Investment): 투자 대비 효과</div><br>

> - **Point:** &nbsp;&nbsp; <span style="color: darkred;">AI 기술이 멋져 보여서</span> 도입하는 것이 아니라, 현장의 확실한 <span style="color: darkred;">돈이 되는 문제</span>부터 풀어야 함
{: .common-quote}

- **주요 내용:**
    - **Top-down vs Bottom-up:** 
        - 경영진의 "우리도 AI 해봐"가 아닌,
        - 현장 작업자의 "이 작업 때문에 너무 피곤하다/시간이 많이 걸린다"에서 출발

    - **ROI 계산 방식의 차이:**
        - **단순 비용 절감:**
            - 검사원 2명 대체<br><span style="color: darkred">🡪 X (현장의 반발만 부름)</span>

        - **품질 및 기회비용 개선:**
            - 미세 불량 유출로 인한 고객사 클레임 비용(연간 N억) 방지
            - 설비 돌발 정지(Downtime) 1시간 감소당 생산 손실 5,000만 원 보전<br><span style="color: darkred">🡪 O</span>

- **💡 현장 예제 (PoC 과제 선정 프레임워크):**

<div class="info-table">
    <table>
        <thead>
            <th style="width: 150px;">평가 항목</th>
            <th style="width: 400px;">높은 우선순위 (AI 도입 적합)</th>
            <th style="width: 400px;">낮은 우선순위 (AI 도입 비적합)</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">문제의 빈도</td>
                <td>매일 발생하고 데이터가 쌓이는 문제</td>
                <td>1년에 한 번 발생하여 데이터가 없는 고장</td>
            </tr>
            <tr>
                <td class="td-rowheader">원인 규명</td>
                <td>변수가 너무 많아 사람 머리로 추적 불가능</td>
                <td>규칙(Rule)이 명확하여 simple IF-THEN으로 해결 가능</td>
            </tr>
            <tr>
                <td class="td-rowheader">파급 효과</td>
                <td>발생 시 공정 전체가 멈추는 핵심 설비</td>
                <td>멈춰도 대체가 가능한 서브 설비</td>
            </tr>
        </tbody>
    </table>
</div>



## 3. 현장 작업자와의 협업 및 XAI(설명 가능한 AI)

<div class="info-table"><b>[용어]</b> XAI(Explainable AI): 설명 가능한 AI</div><br>

> - **Point:** &nbsp;&nbsp; 현장 베테랑 작업자의 노하우를 AI가 대체하는 것이 아니라 **'디지털 무기'로 쥐여주는 관점**이 필요함
{: .common-quote}

- **주요 내용:**
    - **블랙박스(Black-box) AI의 거부감:**
        - AI가 "10분 뒤 고장 납니다"라고만 알려주면, 현장 엔지니어는 "왜?"라고 물으며 알람을 꺼버림(Mute)

    - **XAI(Explainable AI)의 필요성:**
        - "10분 뒤 고장 확률 87% (원인: 3번 베어링 진동 센서값 $$3\sigma$$ 초과 및 오일 온도 급상승)"과 같이 **이유를 설명해 주는 AI** 구축


- **💡 비유 예시:**
    - AI는 운전자의 자율주행 보조 장치(ADAS)와 같음
        - 운전대(최종 결정)는 여전히 20년 경력의 베테랑 작업자가 잡고,
        - AI는 사람이 보지 못하는 사각지대를 센서 데이터로 감지해 주는 역할



## 4. OT와 IT(AI)의 융합 및 인터페이스 설계

<div class="info-table"><b>[용어]</b> OT(Operational Technology): 운용 기술/운영 기술, 현장 제어 기술</div><br>

> - **Point:** &nbsp;&nbsp; AI 분석 결과를 현장에 어떻게 피드백할 것인가(Loop-back)의 문제
{: .common-quote}

- **주요 내용:**
    - **Passive AI (추천/알람):**
        - 화면에 "가공 속도를 5% 낮추세요"라고 띄우고 작업자가 버튼을 누르게 함 (초기 안착 단계)
    - **Active AI (클로즈드 루프 제어):**
        - AI 판단 결과가 PLC로 직접 신호를 보내 모터 속도를 자동 제어함 (고도화 단계 - **안전장치/Interlock 필수**)

- **💡 현장 적용 실패 예시 (비전 검사 팝업 폭탄):**
    - **상황:**
        - 모니터링 화면에 AI의 미세한 이상 알람이 초당 5번씩 팝업으로 뜸

    - **결과:**
        - 작업자가 작업에 방해된다고 모니터 전원을 꺼버림

    - **도입 포인트:**
        - **UX/UI 단순화 및 알람 임계값(Threshold)의 현장 맞춤형 조율.**



## 5. 데이터 지속성 및 MLOps (Model Drift 대응)

<div class="info-table">
<b>[용어]</b> MLOps(Machine Learning Operations): 머신러닝 운영 체계/머신러닝 운용 자동화<br>
AI 모델의 개발부터 현장 배포, 성능 감시, 재학습까지 전 과정을 지속적으로 관리·자동화하는 시스템
</div><br>

> - **Point:** &nbsp;&nbsp; AI 모델은 만드는 것으로 끝나지 않으며, 공장 설비처럼 지속해서 '유지보수'해야 함
{: .common-quote}

- **주요 내용:**
    - **Model Drift (모델 성능 저하):**
        - 설비를 보수하거나 부품을 교체하면 현장의 데이터 특성이 바뀌어 기존 AI 모델의 정확도가 떨어짐

    - **재학습(Re-training) 체계 구축:**
        - 새로운 양품/불량 데이터를 주기적으로 수집하여 AI 모델을 업데이트하는 MLOps 체계의 필요성



## 📋 [워크시트/체크리스트 예제]

> - **우리 공장 AI 도입 전 체크리스트 (5-Point Checklist)**<br><br>
>   - [ ] &nbsp; **1. Data Availability (데이터가 존재하는가?)** 
>      - 원하는 이상을 감지할 수 있는 센서가 달려있고, 결측 없이 저장되고 있는가?
>   - [ ] &nbsp; **2. Clear Problem Definition (문제가 명확한가?)** 
>      - "공정을 최적화하고 싶다"가 아니라 "3번 사출기의 불량률을 2%에서 0.5%로 낮춘다"처럼 구체적인가?
>   - [ ] &nbsp; **3. Domain Expert Involvement (현장 전문가가 참여하는가?)** 
>      - AI 프로젝트팀에 10년 이상 경력의 현장 엔지니어가 포함되어 있는가?
>   - [ ] &nbsp; **4. Actionable Insight (결과를 얻으면 조치할 방법이 있는가?)** 
>      - AI가 불량을 예측했을 때, 실제로 공정을 멈추거나 조건(Recipe)을 바꿀 수 있는 권한과 시스템이 있는가?
>   - [ ] &nbsp; **5. Safety & Interlock (안전 장치가 마련되어 있는가?)** 
>      - AI 오작동 시 설비와 인명을 보호할 물리적 비상정지(Emergency Stop) 장치가 유효한가?
{: .summary-quote}

