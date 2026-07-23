---
layout: page
title:  "데이터 통합 및 품질 관리 방법"
date:   2026-07-22 13:50:00 +0900
permalink: /materials/S02-02-02-03_01-DataIntegrationQuality
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - 데이터 통합과 품질 관리는 별개의 과제가 아니라,
> - **"흩어진 데이터를 모아서(통합), 믿고 쓸 수 있는 상태로 만드는(품질 관리)"** 하나의 유기적인 데이터 파이프라인 과정
{: .common-quote}


## 1. 데이터 통합 (Data Integration)

- 여러 출처(데이터베이스, 서비스, API, 파일 등)에 파편화되어 있는 데이터를 하나의 일관된 형태 및 저장소로 연결하고 모으는 기술과 과정

### 1.1 핵심 데이터 통합 아키텍처 패턴

- 데이터를 모으는 방식은 기술의 발전과 처리 목적(Batch vs Real-time)에 따라 크게 3가지로 나뉨

```
[ETL 패턴]      Source DBs ───(Extract)───► Staging ───(Transform)───► Data Warehouse
[ELT 패턴]      Source DBs ───(Extract)───► Data Lake ───(Transform)───► Analytics Engine
[가상화 패턴]   Source DBs ───(Virtual Federation Layer)────────────► User/BI Tool (Zero Load)
```

- **ETL (Extract, Transform, Load)**
    - **원리:**
        - 원천 시스템에서 데이터를 추출(Extract)한 후,
        - 전용 서버에서 가공·정제·변환(**Transform**)하여
        - 목적지(Data Warehouse)에 적재(Load)하는 전통적 방식
    - **특징:**
        - 정형화된 일괄(Batch) 처리에 적합하며,
        - 목적지 DB의 부하를 줄일 수 있음

- **ELT (Extract, Load, Transform)**
    - **원리:**
        - 빅데이터 환경(Hadoop, Cloud Data Lake 등)의 압도적인 저장/연산 능력을 활용해,
        - 일단 원천 데이터를 있는 그대로 적재(Load)한 뒤,
        - 필요할 때 클라우드 인프라 내에서 변환(Transform)하는 방식
    - **특징:**
        - 데이터 가공 유연성이 높고,
        - 대용량 데이터를 빠르게 수집할 수 있어
        - 현대 데이터 엔지니어링의 주류로 자리 잡음

3.- **데이터 가상화 (Data Virtualization / Federation)**
    - **원리:**
        - 데이터를 실제로 한곳에 복사·이동시키지 않고,
        - 중앙의 가상화 레이어가 각 원천 DB에 실시간으로 쿼리를 날려
        - 결과를 합쳐서 보여주는 방식
    - **특징:**
        - 물리적 저장 비용이 들지 않고 실시간성이 높으나,
        - 원천 시스템의 성능에 영향을 줄 수 있음


## 2. 데이터 품질 관리 (Data Quality Management, DQM)

- Garbage In, Garbage Out(쓰레기가 들어가면 쓰레기가 나온다)
- 아무리 잘 통합된 데이터라도 품질이 낮으면 잘못된 의사결정이나 시스템 오작동을 유발함

### 2.1 데이터 품질의 6대 핵심 지표 (Data Quality Dimensions)

<div class="info-table">
<table>
    <thead>
        <th style="width: 180px;">품질 지표</th>
        <th style="width: 400px;">의미</th>
        <th style="width: 420px;">위반 예시</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">정확성 (Accuracy)</td>
            <td>데이터가 실제 존재하거나 합의된 사실을 정확히 반영하는가?</td>
            <td>나이 필드에 -5가 입력되어 있거나, 생년월일과 일치하지 않음</b></td>
        </tr>
        <tr>
            <td class="td-rowheader">완전성 (Completeness)</td>
            <td>필수적으로 존재해야 하는 데이터 값이 누락 없이 채워져 있는가?</td>
            <td>고객 테이블의 '전화번호'나 '이메일' 값이 'NULL'로 비어 있음</td>
        </tr>
        <tr>
            <td class="td-rowheader">일관성 (Consistency)</td>
            <td>서로 다른 시스템 간에 동일한 데이터가 불일치 없이 유지되는가?</td>
            <td>A 시스템엔 주소가 '서울시', B 시스템엔 '서울특별시'로 다르게 적힘</td>
        </tr>
        <tr>
            <td class="td-rowheader">유효성 (Validity)</td>
            <td>정의된 데이터 형식, 범위, 정규식 구조를 준수하는가?</td>
            <td>이메일 주소 형식에 '@' 기호가 빠져 있음</td>
        </tr>
        <tr>
            <td class="td-rowheader">적시성 (Timeliness)</td>
            <td>필요한 시점에 데이터가 즉시 업데이트되고 제공되는가?</td>
            <td>어제 발생한 결제 데이터가 3일 뒤에 반영됨</td>
        </tr>
        <tr>
            <td class="td-rowheader">유일성 (Uniqueness)</td>
            <td>동일한 식별자를 가진 중복 데이터가 존재하지 않는가?</td>
            <td>동일한 고객 정보가 DB에 ID만 다르게 2개 이상 존재함</td>
        </tr>
    </tbody>    
</table>
</div>



## 3. 데이터 품질 관리 프로세스 (DQM Lifecycle)

- 품질 관리는 단발성 이벤트가 아니라 지속적인 관리 선순환 체계(PDCA Cycle)로 이루어져야 함

    ```text
    [1. 진단 및 profiling] ➔ [2. 규칙 정의 및 정제] ➔ [3. 모니터링 및 측정] ➔ [4. 거버넌스 및 예방]
    ```

    1. **데이터 프로파일링 (Profiling) & 진단:**
        - 데이터의 통계 수치(Null 비율, 최소/최대값, 분포, 타입)를 파악하여 현 상태의 데이터 오류율 및 수집 상태를 정밀 진단

    2. **품질 규칙(Data Quality Rules) 정의 및 데이터 정제:**
        - 도메인별 비즈니스 규칙(예: `계좌 잔액 >= 0`, `일자 형식: YYYY-MM-DD`)을 정의
        - 기존의 결측치 처리, 중복 제거, 표준화 작업 수행

    3. **실시간 모니터링 & 측정:**
        - 데이터 파이프라인상에 품질 검증 툴(예: Great Expectations, Soda 등)을 삽입
        - 적재 시점에 결함 데이터를 자동으로 감지하고 격리/알람 처리

    4. **데이터 거버넌스 (Data Governance) 체계 수립:**
        - **데이터 표준화:**
            - 단어집, 용어 사전, 도메인, 코드 정의서의 전사 표준화
        - **오너십(Ownership) 명확화:**
            - 데이터의 생성, 수정, 관리에 대한 담당자(Data Steward) 및 권한 정의

<br>

> - **요약**
>   - 통합(Integration)은 파편화된 시스템의 데이터를 **ETL/ELT 체계**를 통해 모으고 이어주는 "길을 뚫는 작업"
>   - 품질 관리(Quality Management)는 모인 데이터에 **표준 규칙과 품질 지표**를 적용하여 믿고 쓸 수 있게 가꾸는 "수질 관리 작업"
>   - 이 두 가지가 '데이터 거버넌스'라는 관리 체계 아래에서 지속적으로 순환할 때, 비로소 데이터 기반의 신뢰할 수 있는 분석과 AI 활용이 가능해짐
{: .summary-quote}