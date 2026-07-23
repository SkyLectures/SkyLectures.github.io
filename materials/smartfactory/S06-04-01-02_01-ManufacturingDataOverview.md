---
layout: page
title:  "제조 데이터의 구조 이해"
date:   2026-07-21 03:00:00 +0900
permalink: /materials/S06-04-01-02_01-ManufacturingDataOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - 제조 데이터 유형과 특징에서 제조 데이터의 독특한 물리적·시계열적 특성을 반영했다면
> - **제조 데이터의 구조 이해**에서는
>   - 단순히 테이블의 스키마를 보는 것을 넘어
>   - **현장의 물리적 설비와 신호가 어떻게 디지털 데이터의 형태로 구조화되고 정렬되는가?**에 초점을 맞출 것
{: .expert-quote}


## 1. 수집 계층에 따른 데이터 구조

- **ISA-95 프레임워크(Framework = 전체 체계)**
    - 국제자동화학회(ISA)에서 제정한 
    - 기업 IT 시스템(ERP/CRP)과 현장 제어 시스템(MES/PLC) 간의 인터페이스 및 데이터통합 전체 국제 표준 체계<br><br>
    - 포함 범위: 
        - 계층 구조(피라미드)뿐만 아니라,
        - 공정 정보 모델, 자재/장비/인력 데이터 구조, 시스템 간 데이터 교환 포맷, 용어 정의 전체를 다룸
    - 존재 이유:
        - Siemens 설비, SAP ERP, 자체 개발 MES가 서로 대화(데이터 통합)할 수 있도록 만드는 표준 규격서

- **ISA-95 피라미드 모델 (Functional Hierarchy Model = 계층 구조)**
    - ISA-95 프레임워크 안에 포함된 여러 표준 모델 중
    - 가장 유명하고 직관적인 '기능적 계층 구조'를 시각화한 5단계 피라미드

    <div class="info-table">
    <table>
        <thead>
            <th style="width: 100px;">단계</th>
            <th style="width: 430px;">의미</th>
            <th style="width: 420px;">현장 내용</th>                
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">Level 0</td>
                <td class="td-left">물리적 공정 (Physical Process)</td>
                <td class="td-left">실제 가공/조립되는 제품 및 하드웨어 장치</td>
            </tr>
            <tr>
                <td class="td-rowheader">Level 1</td>
                <td class="td-left">기본 제어 (Basic Control)</td>
                <td class="td-left">센서, 액추에이터, PLC, DCS (실시간 제어 및 신호 발생)</td>
            </tr>
            <tr>
                <td class="td-rowheader">Level 2</td>
                <td class="td-left">공정 감독 제어 (Supervisory Control)</td>
                <td class="td-left">SCADA, HMI, 현장 Edge (설비 모니터링, 감독 및 데이터 1차 수집)</td>
            </tr>
            <tr>
                <td class="td-rowheader">Level 3</td>
                <td class="td-left">제조 운영 관리 (Manifacturing Operations Management, MOM)</td>
                <td class="td-left">MES, POP, WMS (제조 실행, 스케줄링 및 현장 관리)</td>
            </tr>
            <tr>
                <td class="td-rowheader">Level 4</td>
                <td class="td-left">기업 기획 및 물류 관리 (Business Planning & Logistics)</td>
                <td class="td-left">ERP, SCM (기업 자원 관리 및 비즈니스 기획)</td>
            </tr>
        </tbody>
    </table>
    </div>

    - **도표 해설**
        - 데이터는 아래(Level 0)에서 위(Level 4)로 올라갈수록 '단순 신호'에서 '의미 있는 정보(맥락)'로 변환됨
        - 하부 계층 (Level 0~2 - Raw Data / OT 영역):
            - 특징: 고주파(ms 단위), 대용량, 단순 수치(Numeric), 시계열(Time-Series) 중심
            - 상태: 숫자만 늘어서 있어 이 값이 무언가를 뜻하는지 단독으로는 알기 어려움 (예: 45.2)
        - 상부 계층 (Level 3~4 - Contextualized Data / IT 영역):
            - 특징: 이벤트 중심(Event-driven), 관계형(Relational), 요약/집계(Summary) 중심
            - 상태: 맥락이 입혀짐 (예: 45.2라는 숫자가 사출기 1호기의 노즐 온도라는 의미를 가짐)

<br>

> - 공장 자동화 표준인 **ISA-95 피라미드 모델**을 기반으로 하는 데이터의 흐름과 계층 확인하기
>   - 제조 데이터의 구조를 이해하는 가장 클래식하면서도 강력한 방법
>   - <span style="color: darkred;">**데이터가 상위 계층으로 올라갈수록 집계(Aggregation)되고 컨텍스트가 풍부해짐**</span>
{: .common-quote}

<br>

- **제조 AI 분석의 핵심: 계층 간 데이터 융합 (Data Join)**
    - "AI 분석이 실패하는 이유"와 연결하기 좋은 포인트
    - 센서 데이터만 있는 경우 (Level 1만 활용):
        - 센서 수치가 갑자기 튀는 것은 알 수 있지만, 
        - "이게 원자재(Lot) 때문인지, 작업자 조작 실수인지, 설비 노후화 때문인지" 원인을 규명할 수 없음
    - MES 데이터만 있는 경우 (Level 3만 활용):
        - 불량이 5개 났다는 결과는 알지만, 
        - "공정 진행 중 설비의 전압이나 압력이 정확히 어떤 순간에 어떻게 변했는지" 미시적 원인을 알 수 없음
    - 💡 결론:
        - 제조 데이터 분석의 핵심은 Level 1의 시계열 센서 데이터와 Level 3의 MES 공정 이벤트(작업지시서, Lot 번호, 작업자 등)를
        - Timestamp나 Lot ID 기준으로 조인(Join)하여 데이터 마트를 구축하는 것

<br>

> - ISA-95 피라미드는 단순한 장비 배치도가 아니라 바로 '데이터가 익어가는 과정'
> - **데이터 파이프라인 설계를 위해서는**
>   - 센서가 뿜어내는 '의미 없는 숫자(Level 1)'에,
>   - MES의 '작업 맥락(Level 3)'을 입히고, ERP의 '비즈니스 가치(Level 4)'로 환산하는
>   - 전체 흐름을 이해해야 제대로 된 제조 데이터 파이프라인을 설계할 수 있음
> - **분석을 위해서는** 
>   - Level 1~2의 센서 데이터와 Level 3의 MES 이벤트 데이터(예: 이 센서 값이 튄 시점에 어떤 작업지시서와 제품이 흘러가고 있었는가?)를 결합하는
>   - **'계층 간 데이터 매핑 구조'**를 이해하는 것이 핵심
{: .expert-quote}

---

## 2. 시계열 Tag 데이터 구조

- **일반 데이터 vs 시계열 Tag 데이터**
    - **일반 RDBMS 방식 (Wide Table):** 
        - 설비 1대의 여러 센서 값을 한 행에 저장
        - **문제점:** 센서가 추가될 때마다 테이블 컬럼을 늘려야 함 (스키마 변경 부담)

        <div class="info-table">
        <table>
            <thead>
                <th style="width: 200px;">Timestamp</th>
                <th style="width: 150px;">설비ID</th>
                <th style="width: 150px;">온도 센서</th>
                <th style="width: 150px;">압력 센서</th>
                <th style="width: 150px;">진동 센서</th>
                <th style="width: 100px;">...</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">10:00:01</td>
                    <td>Unit01</td>
                    <td>230.5</td>
                    <td>5.2</td>
                    <td>0.01</td>
                    <td>...</td>
                </tr>
            </tbody>
        </table>
        </div>

    - **시계열 Tag 데이터 방식 (Narrow Table):**
        - 시간과 Tag ID를 키(Key)로 하여 수직으로 저장
        - **장점:** 센서가 1개든 1,000개든 스키마 변경 없이 데이터 적재 가능.

        <div class="info-table">
        <table>
            <thead>
                <th style="width: 200px;">Timestamp(시간)</th>
                <th style="width: 150px;">Tag ID (이름)</th>
                <th style="width: 150px;">Value (값)</th>
            </thead>
            <tbody>
                <tr><td class="td-rowheader">10:00:01</td><td>U1_TEMP</td><td>230.5</td></tr>
                <tr><td class="td-rowheader">10:00:01</td><td>U1_PRES</td><td>5.2</td></tr>
                <tr><td class="td-rowheader">10:00:01</td><td>U1_VIB</td><td>0.01</td></tr>
                <tr><td class="td-rowheader">10:00:02</td><td>U1_TEMP</td><td>230.7</td></tr>
            </tbody>
        </table>
        </div>


- **기본 3요소 + 1요소 구조:**
    - **Timestamp (언제?):**
        - 데이터가 발생한 정확한 시각 (고주파 데이터일수록 나노초/밀리초 단위 정밀도 구조)
        - 단순히 시간이 아니라, **데이터의 고유 키(Primary Key)** 역할을 수행함
        - **도입 포인트:** 
            - 고주파 센서(진동, 전류)는 $$ms$$(밀리초) 또는 $$\mu s$$(마이크로초) 단위까지 저장해야 오탐을 줄일 수 있음

    - **Tag Name/ID (무엇을?):**
        - 설비 및 센서의 고유 식별자
            - 어떤 설비의 어떤 부위인지 식별하는 고유 코드 (예: `LINE1_MOLD_TEMP_01`)
        - **도입 포인트:**
            - 현장마다 Tag 명명 규칙(Naming Convention)이 다름
            - 이를 표준화하는 작업이 분석 전처리에서 가장 중요함

    - **Value (값은?):**
        - 실제 측정값 (정수, 실수 등)과 측정된 데이터 타입
        - **도입 포인트:**
            - 데이터 타입(Integer, Float, Boolean 등)에 따라 저장 용량과 분석 알고리즘이 달라짐

    - **(+추가) Quality / Flag (믿을 수 있는가?):**
        - 실제 현장 데이터에는 센서 고장, 통신 단절 등으로 인한 노이즈가 많음
        - 데이터가 '정상'적으로 수집되었는지 표시하는 Quality 값(Good/Bad/Uncertain)이나 **Flag** 데이터가 이 구조에 함께 저장되어야 함


- **설비 메타데이터 구조 (Asset Framework):**
    - Tag ID 하나만 보면 이것이 어느 공장, 어느 라인, 어떤 설비의 부품인지 알 수 없음
    - 설비를 **트리(Tree)나 그래픽(Graph) 형태의 계층 구조**로 추상화하여 Tag와 매핑하는 데이터 구조를 다루어야 함

    - **[메타데이터 결합] Asset Framework의 실체**
        - **Tag ID의 한계:**
            - `LINE1_MOLD_TEMP_01` 🡪 분석가는 이 Tag가 속한 공장의 목표 생산량이나, 이 금형의 정비 이력을 알 수 없음
                - "이 금형이 언제 설치되었지?", "이 라인의 목표 생산량은 얼마지?"라는 비즈니스적/물리적 맥락을 알 수 없음
            - 해결책:
                - 이 Tag ID를 실제 물리적 자산(Asset)인 '사출성형기 #01'에 연결하고,
                - 그 자산에 대한 메타데이터(제조사, 설치일 등)를 트리 구조로 관리해야 함

        - **Asset Framework (가산 체계) 매핑:**
            - Tag ID(가상)와 실제 물리적 자산(Asset)을 계층형 구조로 연결하고, 각 레벨에 속성(Attribute) 데이터를 부여함

                ```yaml
                # Asset Framework 예시 (YAML 형식 트리)
                Factory: 안산공장
                Line: 가공1라인
                    Asset: 사출성형기 #01
                    Attributes:
                        Manufacturer: Sumitomo
                        Install_Date: 2020-01-01
                    Sensors (Tags):
                        - Name: 노즐 온도 ➔ Mapping to Tag ID: LINE1_MOLD_TEMP_01
                        - Name: 사출 압력 ➔ Mapping to Tag ID: LINE1_MOLD_PRES_01
                ```


- **Time-Series DB (TSDB)의 필요성**
    - **그렇다면 이 방대한 시계열 데이터는 어디에 저장하는가?**
        - 거대한 시계열 데이터를 처리하기 위해 RDBMS가 아닌 'TSDB(시계열 데이터베이스)'라는 전용 저장소가 필요함
            - "어? 그냥 3개 컬럼짜리 테이블이면 우리가 흔히 쓰는 MySQL/Oracle 같은 RDBMS나 엑셀에 저장해도 되는 것 아닌가요?"
    
    - **TSDB vs RDBMS**    
        <div class="info-table">
        <table>
            <thead>
                <th style="width: 150px;">비교 항목</th>
                <th style="width: 380px;">시계열 전용 DB (TSDB)</th>
                <th style="width: 380px;">전통적 관계형 DB (RDBMS)</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">주요 목적</td>
                    <td>고주파 시계열 데이터의 초고속 적재 및 연속 추이 분석</td>
                    <td>복잡한 관계형 트랜잭션의 정확성 및 무결성 보장</td>
                </tr>
                <tr>
                    <td class="td-rowheader">쓰기 성능</td>
                    <td>
                        Append-only(누적 저장) 위주의 초고속 쓰기<br>
                        (초당 수십만 건 수신 가능)
                    </td>
                    <td>트랜잭션 Lock 및 인덱스 갱신으로 초고속/대용량 쓰기에 한계</td>
                </tr>
                <tr>
                    <td class="td-rowheader">데이터 수정/삭제</td>
                    <td>거의 발생하지 않음 (과거 기록은 불변)</td>
                    <td>UPDATE / DELETE 작업이 빈번하게 발생</td>
                </tr>
                <tr>
                    <td class="td-rowheader">저장 효율성</td>
                    <td>동일한 Tag ID 반복 구조에 최적화된 고압축률 (용량 80~90% 절감)</td>
                    <td>데이터 가변성으로 인해 압축 효율이 낮고 용량 급증</td>
                </tr>
                <tr>
                    <td class="td-rowheader">조회 및 분석 (Query)</td>
                    <td>시간 범위 집계(평균, 이동평균, Downsampling)에 특화된 전용 함수 제공</td>
                    <td>시간 단위 그룹화/집계 시 쿼리가 매우 복잡하고 속도가 현저히 느림</td>
                </tr>
            </tbody>
        </table>
        </div>

        - **대조 항목별 세부 설명**
            - 쓰기 성능 (Insert Speed): '누적' vs '무결성'
                - RDBMS의 한계:
                    - RDBMS는 데이터의 ACID(트랜잭션 무결성)를 보장하기 위해 데이터를 쓸 때마다 Index를 갱신하고 테이블에 Lock을 적용
                    - 밀리초($ms$) 단위로 센서 수만 개가 뿜어져 나오는 상황에서는 DB에 병목(Overhead)이 생겨 데이터가 유실될 수 있음
                - TSDB의 강점:
                    - 과거 데이터를 수정하지 않는 'Append-only(단순 누적)' 방식을 채택
                    - 인덱싱 부담을 최소화하고 쏟아지는 데이터를 스폰지처럼 주입

            - 저장 공간과 압축 (Storage & Compression): '패턴 압축' vs '일반 저장'
                - RDBMS의 한계:
                    - Timestamp, Tag ID 같은 동일한 텍스트 데이터가 매초/매밀리초마다 무한 반복 저장
                    - 디스크 용량이 순식간에 소모됨
                - TSDB의 강점:
                    - "어차피 Tag ID는 계속 똑같고, Timestamp는 1초씩 일정하게 증가한다"는 특성을 이용
                    - 차이값(Delta)만 저장하는 시계열 특화 압축 알고리즘(Gorilla 등)을 사용 🡪 용량을 1/10 수준으로 감소
            - 조회 및 집계 쿼리 (Querying & Aggregation): '전용 함수' vs '복잡한 SQL'
                - RDBMS의 한계: 
                    -"최근 1시간 동안의 1분 단위 평균값"을 구하려면 GROUP BY와 시간 변환 함수를 복잡하게 조합해야 함
                    - 수억 건의 레코드를 스캔하느라 조회 시간이 수십 초 이상 소요
                - TSDB의 강점:
                    - time_bucket(), moving_average() 같은 시계열 전용 함수를 기본 제공
                    - 미리 집계된 데이터(Downsampling) 구조가 적용되어 있음
                    - 수억 건의 데이터도 0.1초 만에 그래프로 출력 가능

<br>

> - 시계열 Tag 데이터 구조를 이해한다는 것은 단순히 Timestamp, Tag, Value라는 세 단어를 아는 것이 아님<br><br>
> - 쏟아지는 센서 신호를 **유실 없이 저장하고(TSDB)**,
> - 그 숫자가 어느 공장, 어느 설비의 것인지 **맥락(Asset Framework)**을 입히고,
> - 그 데이터가 **믿을 수 있는지(Quality Flag)**까지 함께 관리하는
> - 전체 체계를 이해하는 것이 제대로 된 제조 데이터 분석의 출발점
{: .expert-quote}

---

## 3. 공정 컨텍스트 데이터 구조

- 센서 데이터가 아무리 많아도 '맥락(Context)'이 없으면 무의미한 숫자 나열에 불과함
- 제조 데이터 구조화의 핵심은 시계열 데이터에 **공정의 맥락을 입히는** <span style="color: darkred;">**데이터 모델링**</span>

- **Lot / Serial 기반 구조:**
    - 제품 한 단위(Unit) 또는 한 묶음(Lot)이 시각 A에 공정 1에 진입하여 시각 B에 나갔다는 시간적 구간(Interval) 데이터 구조

- **4M 데이터 구조 (Man, Machine, Material, Method):**
    - **Man:** 해당 시점에 작업한 작업자 정보
    - **Machine:** 사용된 설비 및 금형/툴 번호
    - **Material:** 투입된 원자재의 Lot 번호 (추적성, Traceability)
    - **Method:** 당시 설비에 세팅된 레시피(Recipe) 및 파라미터 조건

<br>

<div class="insert-image">
    <img src="/materials/smartfactory/images/S06-04-01-02_01-001.png" style="width: 90%;"><br><br>
    <caption>시계열 센서 데이터(Continuous)와 4M 컨텍스트 데이터(Discrete)를<br>특정 <b>'Time Window(시간 창)'</b>이나 <b>'Lot ID'</b>를 기준으로 어떻게 조인(Join)하고 융합 구조를 만드는지 시각적으로 보여주는 그림</caption>
</div>

---

## 4. 제조 특화 데이터 포맷과 표준 프로토콜 구조

- **통신 표준과 데이터 트리: OPC-UA (Open Platform Communications Unified Architecture)**
    - **개념:**
        - 서로 다른 제조사의 설비(Siemens, Mitsubishi 등)가 브랜드에 상관없이 동일한 언어로 대화할 수 있도록 만들어진
        - **스마트팩토리 표준 통신 아키텍처**
    - **노드(Node)와 오브젝트(Object) 데이터 구조:**
        - 설비와 센서를 단순한 숫자가 아니라, **객체(Object) 중심의 트리(Tree) 구조**로 추상화하여 관리
        - **구조 예시:**

            ```text
            [Root] (뿌리)
            └── [Objects]
                └── [Line1_CNC_Machine] (오브젝트: 1라인 CNC 설비)
                    ├── [Variables] (변수 노드: 실제 측정 데이터)
                    │    ├── Temperature: 42.5 (°C)
                    │    └── Spindle_RPM: 1200 (RPM)
                    └── [Methods] (메서드 노드: 제어 명령)
                            └── Start_Cooling() (냉각기 가동 명령)
            ```

            > - OPC-UA는 단순히 숫자를 주고받는 것이 아니라
            > - `1라인 CNC 설비`라는 상자(오브젝트) 안에 `온도`라는 변수(Node)와 `냉각 가동`이라는 명령(Method)을
            > - **트리 형태로 묶어서 주고받는 데이터 구조**를 의미함
            {: .common-quote}

<br>

- **반도체/디스플레이 특화 구조: SECS/GEM 프로토콜**
    - **개념:**
        - 반도체/디스플레이 공정처럼 매우 정밀하고 복잡한 설비에서 표준으로 사용되는
        - **장비-호스트(MES) 간 통신 규격**
    - **핵심 수집 데이터 구조 (SVID & 이벤트 리포트):**
        - **SVID (Status Variable ID):**
            - 장비의 현재 상태를 나타내는 정적/동적 변수 번호
                - 예: `SVID 1001` = Chamber Temperature, `SVID 1002` = Gas Flow Rate
        - **이벤트 기반 리포트 (Event-Driven Report):**
            - 밀리초 단위로 데이터를 무조건 출력/전달하는 것이 아니라,
            - **특정 사건(Event)이 발생했을 때 관련된 SVID 묶음을 전송**하는 효율적인 구조
        - **데이터 수집 예시 구조:**

            ```text
            [Event Trigger] ➔ "웨이퍼 가공 완료 (Event ID: 501)" 발생 시!
            └── [Event Report Data Package]
                ├── Time: 2026-03-27 10:00:00.123
                ├── SVID 1001 (Chamber Temp): 250.0 °C
                ├── SVID 1002 (Gas Flow): 50.2 sccm
                └── SVID 2005 (Wafer ID): WAF_2026_09
            ```

            > - SECS/GEM은 무작정 데이터를 쌓는 게 아니라,
            > - `웨이퍼 투입`, `가공 완료` 같은 **특정 이벤트가 발생한 순간에**
            > - **해당 이벤트와 연관된 SVID(변수 값들)를 한 묶음의 리포트로 패키징해서 전송하는 데이터 구조**를 가짐
            {: .common-quote}

<br>

- **빅데이터 저장 포맷: Columnar(컬럼 기반) 저장 구조 vs CSV**
    - **개념:**
        - 엑셀이나 CSV처럼 가로(Row) 방향으로 데이터를 저장하는 것이 아니라,
        - **세로(Column) 방향으로 데이터를 모아서 저장**하는 빅데이터 전용 파일 포맷(Parquet, Avro 등)
        - 대규모 시계열 데이터를 분석할 때, **디스크 읽기(Disk I/O) 병목을 줄이고** 분석 조회 속도를 극대화하기 위해 사용함

    - **구조적 차이점 비교:**
        <div class="info-table">
        <table>
            <thead>
                <th style="width: 150px;">구분</th>
                <th style="width: 350px;">일반 CSV / RDBMS (Row-based)</th>
                <th style="width: 400px;">Parquet / ORC (Columnar-based)</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">저장 방식</td>
                    <td>텍스트 (문자열)</td>
                    <td>바이너리 (컬럼별 인코딩 및 압축)</td>
                </tr>
                <tr>
                    <td class="td-rowheader">압축률/디스크 사용량</td>
                    <td>텍스트 형태 그대로 저장되어 용량이 큼</td>
                    <td>동일한 센서 데이터끼리 모여 있어 압축률 80~90% 달성</td>
                </tr>
                <tr>
                    <td class="td-rowheader">저장 시 CPU 소모</td>
                    <td>적음 (단순 텍스트 쓰기)</td>
                    <td>큼 (컬럼 재배치 및 압축 연산 필요)</td>
                </tr>
                <tr>
                    <td class="td-rowheader">분석 조회 속도</td>
                    <td>느림 (불필요한 컬럼까지 전체 스캔)</td>
                    <td>압도적으로 빠름 (필요한 컬럼만 디스크에서 로드)</td>
                </tr>
            </tbody>
        </table>
        </div>
