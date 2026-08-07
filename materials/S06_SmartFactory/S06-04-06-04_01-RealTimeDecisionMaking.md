---
layout: page
title:  "실시간 데이터 기반 의사결정 구조"
date:   2026-07-22 22:50:00 +0900
permalink: /materials/S06-04-06-04_01-RealTimeDecisionMaking
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - 데이터를 실시간으로 모으고 예쁘게 보여주어도, 사람의 조치가 늦어지면 아무 소용이 없음
> - **이벤트 발생(Event)부터 맥락 파악, 작업자 알림, PLC 제어, 정비 지시(WO) 발행까지 단절 없이 연결되는 자동화된 의사결정 파이프라인(Data-to-Action)**을 갖춰야 함
{: .common-quote}



## 1. 실시간 의사결정 파이프라인

- 현장의 모니터링 시스템이 단순 '감시'에 그치지 않고 실제 '조치'로 이어지도록 만드는 **4단계 데이터 파이프라인** (Data-to-Action)<br>
$$\text{Data 수집} \longrightarrow \text{Event 감지 (Rule/AI)} \longrightarrow \text{Context 결합 (MES/ERP)} \longrightarrow \text{Action (알람/제어)}$$

- **[1단계] Data 수집 (Continuous Ingestion)**
    - **내용:**
        - 현장의 OT(PLC, 센서) 영역에서 초/밀리초 단위의 시계열 데이터(Vibration, Temp, Current) 및 설비 Status 신호를 실시간 수집

    - **기술 요소:**
        - Edge Computing, Industrial IoT, OPC UA, MQTT 프로토콜

- **[2단계] Event 감지 (Rule / AI Anomaly Detection)**
    - **내용:**
        - 수집된 시계열 데이터 흐름 속에서 **단순 임계치 초과(Rule)나 AI 기반의 패턴 이탈(Anomaly Score 상승)을 판단하여 '이벤트(Event)'를 생성**

    - **예시:**
        - 2번 사출기 메인 스핀들 진동 피크값 4.5 mm/s 초과 및 AI 이상 점수 88% 달성 🡪 [이벤트 ID: EVT-2026-0891] 발급

- **[3단계] Context 결합 (도메인 정보 융합 - MES/ERP 연동)**
    - **내용:**
        - 이벤트 신호만으로는 현장에서 판단을 내릴 수 없음
        - **MES/ERP의 IT 시스템과 실시간 연동하여 '현재 공정의 맥락(Context)'을 합성**

    - **합성되는 맥락 정보:**
        - 현재 어떤 품목/Lot을 작업 중인가? (고객사 납기 임박 건인가?)
        - 해당 부품의 예비 재고가 자재 창고에 있는가? (ERP 재고 상태 확인)
        - 다음 정기 점검까지 남은 시간은 얼마인가?

- **[4단계] Action (자동화된 피드백 및 조치)**
    - **내용:**
        - 결합된 Context를 바탕으로 **우선순위(Severity)를 판단하여 알람 발송, PLC 제어 명령, 정비 작업지시서(Work Order) 발행**을 즉시 자동 수행


## 2. Event-Triggered Workflow 설계 및 시나리오

- 사람이 일일이 보고 판단하여 전화나 메신저로 전달하는 구식 방식 대신,
- **시스템이 이벤트를 감지하고 트리거(Trigger)되어 연쇄적으로 작동하는 현장 워크플로우 예시**

- **[실제 공정 적용 시나리오] 자동차 부품 가공 라인 스핀들 이상 발생 시**
    1. **[T = 0초 / Event 발생]:**
        - 가공 CNC 3번기 스핀들 베어링에서 이상 진동 주파수 감지

    2. **[T = 0.5초 / Context 결합]:**
        - MES 확인 결과 🡪 현재 Lot `[PART-A]` 100개 중 85번째 가공 중 (남은 가공 시간 3분)
        - ERP 확인 결과 🡪 창고에 스핀들 예비 베어링 재고 2개 존재 확인

    3. **[T = 1초 / Action - Interlock 예약]:**
        - 즉시 설비를 멈추면 해당 Part가 폐기(Scrap)되므로,
        - **"현재 Part 가공 완료(3분 뒤) 즉시 다음 작업 투입 중단 및 설비 정지(Safety Interlock)"** 명령을 PLC로 예약 전달

    4. **[T = 2초 / Action - 정비 지시 자동 발송]:**
        - 정비 MRO 시스템에서 **작업지시서(Work Order #9021) 자동 생성**
        - 현장 보전 담당자의 **스마트워치 및 태블릿으로 팝업 알람 전송**:
            - 3번 CNC 스핀들 베어링 이상 마모! 3분 후 설비 정지 예정
            - 예비 부품 A-12 수령 후 2번 라인 출동 요망


## 3. 의사결정 주체별 역할 분담

- 모든 의사결정을 사람이 할 수는 없음
- 반대로 AI/시스템에 전적으로 맡길 수도 없음
- **조치에 필요한 '골든타임(Time Criticality)'에 따라 의사결정 주체를 3단계로 명확히 분리**해야 함 (3-Tier Decision Structure)

```text
[ Decision Hierarchy (의사결정 계층) ]
  ▲  High Level (Manager)    : 시간/일 단위 ➔ 정비 일정, 부품 발주, 라인 재배치 (ERP)
  │  Mid Level  (Operator)   : 분 단위      ➔ 가공 조건 조정, 현장 점검, 승인 (HMI)
  │  Low Level  (Machine)    : 초/밀리초    ➔ 실시간 자율 제어, 비상 정지 (PLC/Edge)
```

- **Machine Level (시스템/PLC 레벨자율 제어) - "초/밀리초 단위"**
    - **특징:** 설비 파손이나 인명 사고 위험 등 **인간이 판단할 시간적 여유가 없는 초급박 상황**
    - **주체:** Edge AI, PLC, Safety Controller
    - **역할:** 설비 비상 정지(E-Stop), 토크 자동 감발, 인터락(Interlock) 작동

- **Operator Level (현장 작업자/보전원) - "분 단위"**
    - **특징:** 공정 변수의 미세 조정이나 물리적 부품 교체가 필요한 상황
    - **주체:** 현장 작업자, 설비 정비 엔지니어 (Human-in-the-Loop)
    - **역할:** HMI 모니터의 AI 추천 처방을 보고 사출 압력 조정, 가공 툴 교체, 정비 작업 수행

- **Manager Level (공장장/운영 관리자) - "시간/일/주 단위"**
    - **특징:** 공장 전체의 생산성, 예산, 자재 수급과 연관된 구조적 결정
    - **주체:** 생산관리팀장, 설비보전 파트장, 공장장
    - **역할:** 정비 휴무일 지정, 대체 생산 라인으로의 작업 지시 변경, 예비 부품 구매 승인 (ERP 연동)

<br>

> - **아무리 뛰어난 AI가 0.01초 만에 고장을 예지해도, 정비 기사가 2시간 뒤에 알림을 받는다면 그 시스템은 실패한 시스템**
> - 데이터 관제 시스템을 설계할 때 **'Data-to-Action 파이프라인'**을 명확히 구축할 것
> - 초 단위의 급박한 위협은 **Machine(PLC)이 자율 제어**하고,
>   - 분 단위 현장 이슈는 **Operator의 스마트기기로 즉시 정비 지시(WO)를 전송하며**,
>   - 시간 단위 운영 결정은 **Manager의 ERP 시스템과 연결**될 때
>   - 비로소 사람이 일일이 전화하지 않아도 스스로 돌아가는 '스마트 운영 체계'가 완성됨
{: .expert-quote}