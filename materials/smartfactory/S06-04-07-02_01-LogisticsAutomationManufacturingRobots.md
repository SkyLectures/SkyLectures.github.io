---
layout: page
title:  "물류 자동화 및 제조 로봇(AVR, 협동로봇) 이해"
date:   2026-07-22 22:50:00 +0900
permalink: /materials/S06-04-07-02_01-LogisticsAutomationManufacturingRobots
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - 가공 속도를 1초 줄이는 것보다
> - **자재와 반제품이 라인 사이에 멍하니 대기하는 재공 재고 시간(WIP: Work In Process)을 줄이는 것이 전체 생산성에 훨씬 더 결정적**
> - 자율주행 물류 로봇(AMR)과 협동로봇(Cobot)은 공장의 혈관을 유연하게 뚫어주는 자율 물류의 핵심 무기
{: .common-quote}


## 1. AGV vs AMR 기술 및 운용 방식 비교

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">비교 항목</th>
        <th style="width: 420px;">AGV (Automated Guided Vehicle)</th>
        <th style="width: 420px;">AMR (Autonomous Mobile Robot)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">주행 경로 방식</td>
            <td style="text-align: left;">
                - 고정 경로 (Fixed Path)<br>
                - 바닥의 마그네틱 테이프, QR코드, 유선 라인을 지정하여 주행
            </td>
            <td style="text-align: left;">
                - 자율 유연 경로 (Flexible Path)<br>
                - 지도(Map) 기반 자율주행. 목적지만 주어지면 경로 스스로 생성
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">주요 센서 및 기술</td>
            <td style="text-align: left;">
                - 마그네틱 센서, QR코드 카메라, 단순 장애물 감지 감응 센서
            </td>
            <td style="text-align: left;">
                - LiDAR, 3D Depth 비전 센서, SLAM<br>
                - (Simultaneous Localization and Mapping)
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">장애물 만났을 때</td>
            <td style="text-align: left;">
                - 앞길에 장애물(사람, 상자)이 있으면 제자리에 스톱(Stop) 후<br>&nbsp;&nbsp;막힌 경로가 뚫릴 때까지 대기
            </td>
            <td style="text-align: left;">
                - 장애물을 감지하면 실시간 우회 경로(Rerouting)를<br>&nbsp;&nbsp;스스로 계산하여 돌아서 목적지 이동
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">공정 레이아웃 변경 시</td>
            <td style="text-align: left;">
                - 바닥 마그네틱 공사를 다시 해야 하므로 라인 변경 비용 및 시간 막대함
            </td>
            <td style="text-align: left;">
                - 로봇을 매핑 모드로 켜고 공장을 한 바퀴 돌려<br>&nbsp;&nbsp;소프트웨어 지도를 갱신하면 즉시 적용 완료
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">적합한 공정 환경</td>
            <td style="text-align: left;">
                - 동선이 유행을 타지 않고 단순 대량 운송이 이루어지는<br>&nbsp;&nbsp;고정형 컨베이어 대체 공정
            </td>
            <td style="text-align: left;">
                - 다품종 가변 생산, 작업자와 동선이 겹치고<br>&nbsp;&nbsp;레이아웃 변경이 잦은 유연 셀(Cell) 생산 공정
            </td>
        </tr>
    </tbody>
</table>
</div>


## 2. 협동로봇(Cobot)과 기존 산업용 로봇의 핵심 차이점

- **안전 펜스(Safety Fence) 유무와 안전 메커니즘**
    - **기존 산업용 로봇:**
        - 무겁고 고속으로 동작하여 충돌 시 인명 사고 위험 🡪 법적으로 철제 **안전 펜스 및 안전 센서(Light Curtain)** 내부에서만 동작해야 함
        - 공장 바닥 면적을 많이 차지함

    - **협동로봇 (Cobot):**
        - **안전 펜스 없이 사람 바로 옆자리**에서 함께 작업 가능
        - **충돌 감지 기술 (Force/Torque Sensor):**
            - 로봇 관절마다 충격/토크 센서가 내장되어 있어,
            - 사람이나 물체와 1 ~ 2 kgf 정도의 아주 미세한 충돌이라도 감지되는 순간
            - **밀리초(ms) 단위로 즉시 동작을 정지(Power and Force Limitation)**

- **작업 전환(Re-tasking)의 용이성**
    - **기존 산업용 로봇:**
        - 전문 로봇 핑거/티칭 펜던트를 이용해 3D 좌표 코딩(TP 언어)을 줄줄이 짜야 하므로,
        - 공정 하나 바꾸려면 로봇 엔지니어가 며칠간 상주해야 함

    - **협동로봇 (Cobot):**
        - **Direct Teaching (Hand Guiding):**
            - 작업자가 로봇 팔의 '자유 이동' 버튼을 누르고
            - **손으로 잡고 원하는 움직임을 직접 움직여주면, 로봇이 해당 궤적과 위치 좌표를 스스로 기록 및 기억**
        - 비전공자도 **10분 만에 새로운 부품 조립 궤적을 교시(Teaching)** 가능



## 3. ACS의 중요성

- 로봇 1~2대를 도입하는 것은 쉽지만, 50대 이상으로 늘어나면 차원이 다른 문제가 발생함 🡪 관제 시스템 필요

```text
       [ MES / WMS (상위 물류 지시) ]
                    │
                    ▼
     ┌──────────────────────────────────┐
     │  ACS (AGV/AMR Control System)    │ ──► 로봇 군집 실시간 제어 시스템
     └──────────────────────────────────┘
         ├── AMR #1 : 최단 경로 할당 및 자재 이송
         ├── AMR #2 : 교차로 충돌 회피 (우회 경로 제어)
         └── AMR #3 : 배터리 임계치 감지 (자율 충전 도킹)
```

- **ACS (AGV/AMR Control System)란?**
    - 공장이나 물류창고 내에서 가동되는 여러 대의 AGV/AMR 로봇들이 서로 충돌하지 않고 효율적으로 이동하도록
    - **실시간으로 통합 제어·배차하는 전용 컨트롤 시스템**

- **ACS의 핵심 제어 기능:**
    - **Traffic Control (실시간 교통 제어):**
        - 좁은 통로에서 로봇 간 교차 진입 시 우선순위를 부여하고 교착 상태(Deadlock)를 방지

    - **Dispatching & Routing (지능형 배차 및 경로 제어):**
        - 상위 MES/WMS의 물류 이송 명령을 받아 가장 적합한 로봇에 명령을 내리고 동적 최단 경로를 실시간 생성

    - **Charge Management (충전 관리):**
        - 로봇의 배터리 잔량을 실시간 추적하여 가동 공백이 생기지 않도록 충전 도킹을 자동 제어

<br>

> - **자동화 로봇 도입의 진짜 목적은 '사람을 없애는 것'이 아니라 '공정 간의 이음새(WIP 대기)를 매끄럽게 잇는 것'**
> - 고정된 컨베이어 벨트나 마그네틱 AGV는 공장 레이아웃을 고착화시켜 변화에 취약하게 만듦
> - **자율 우회가 가능한 AMR과 안전 펜스 없이 작업자 옆에서 보조하는 협동로봇, 그리고 이들을 하나의 유기체처럼 통제하는 ACS**를 결합할 때,
>   - 아침과 오후에 생산 품목이 바뀌어도 멈추지 않는 진짜 '유연 물류 체계'가 완공됨
{: .summary-quote}

<br>

- **[참고] ACS와 FMS의 구분**

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">구분</th>
        <th style="width: 440px;">ACS (AGV/AMR Control System)</th>
        <th style="width: 440px;">FMS (Fleet Management System)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">핵심 역할</td>
            <td style="text-align: left;">물류 현장 내 AMR/AGV 로봇 군집의<br>실시간 제어, 교통 통제, 경로 할당, 충돌 방지 전용 시스템</td>
            <td style="text-align: left;">로봇뿐만 아니라 수송 차량, 트럭, 컨테이너 등<br>전체 운송 자산(Fleet)의 상태, 유지보수, 관제를 총괄하는 상위 시스템</td>
        </tr>
        <tr>
            <td class="td-rowheader">제어 수준</td>
            <td style="text-align: left;">Low-Level / Real-time Control (초/밀리초 단위 로봇 직접 제어)</td>
            <td style="text-align: left;">High-Level / Asset Management (시간/일 단위 자산 운영 관제)</td>
        </tr>
    </tbody>
</table>
</div>