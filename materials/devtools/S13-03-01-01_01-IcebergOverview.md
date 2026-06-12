---
layout: page
title:  "Apache Iceberg 개요 및 설치, 환경설정"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-03-01-01_01-IcebergOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 데이터 레이크(Data Lake)

- **정의**
    - 정형, 반정형, 비정형 등 다양한 형태의 모든 원시 데이터(Raw Data)를 가공하지 않은 상태로 한곳에 중앙 집중식으로 저장하는 거대한 저장소
    - 비유하자면
        - 기존의 데이터 웨어하우스(DW)가 정제된 데이터만 담는 '생수 수조'라면,
        - 데이터 레이크는 강물, 빗물, 지하수 등 온갖 원천 데이터가 가공 없이 그대로 흘러 들어오는 '거대한 호수'에 비유할 수 있음

- **주요 특징 (Core Characteristics)**
    - **원시 데이터 그대로 저장 (Raw Data Storage):**
        - 텍스트, 로그, 이미지, 비디오, JSON, RDBMS 테이블 등 출처와 형식을 가리지 않고 훼손 없이 원본 그대로 보관함

    - **Schema-on-Read (읽기 시점 스키마 정의):**
        - 데이터를 저장할 때 미리 구조(Schema)를 정의하지 않음
        - 우선 저장(Write)부터 해 두고, 
        - 나중에 데이터를 분석하거나 AI 모델링을 위해 **읽어올 때(Read) 필요한 형태로 구조를 정의**하여 사용

    - **뛰어난 확장성과 저렴한 비용:**
        - HDFS나 클라우드 오브젝트 스토리지(AWS S3, Google Cloud Storage 등)를 기반으로 구축
        - 페타바이트(PB) 이상의 대용량 데이터도 저렴한 비용으로 유연하게 저장 공간을 확장할 수 있음

    - **분석 및 AI/ML 유연성:**
        - 데이터가 가공되지 않은 상태로 유지됨
        - 데이터 사이언티스트나 분석가가 목적에 맞게 가공하여 BI 리포팅, 데이터 마이닝, 머신러닝(ML) 학습 등 다각도로 활용할 수 있음

> - **요약**
>   - 데이터 레이크는
>       - 나중에 어떻게 쓸지 모르는 대규모 원시 데이터를 저비용으로 모아두고,
>       - 필요할 때마다 꺼내어 다목적(분석, AI 등)으로 활용하는 데이터 인프라
>   - 메타데이터 관리가 부실하면 쓸모없는 데이터만 가득 찬 **'데이터 늪(Data Swamp)'**이 될 수 있음
>       - 최근, 이를 보완하기 위해 *Apache Iceberg* 같은 테이블 포맷을 도입하여 레이크하우스(Lakehouse) 형태로 발전하는 추세
{: .summary-quote}

## 2. Apache Iceberg 개요

### 2.1 정의 및 핵심 철학

- **정의**
    - 거대한 규모의 데이터 레이크(Data Lake) 환경에서 신뢰성과 성능을 제공하기 위해 설계된 오픈 소스 고성능 테이블 포맷(Table Format)
    - **오브젝트 스토리지 위에서 RDBMS와 같은 ACID 트랜잭션을 구현한 추상화 계층**의 위치
    - 페타바이트(PB) 규모의 대형 테이블을 위한 개방형 테이블 규격

- **핵심 철학:**
    - 컴퓨팅 엔진(Spark, Flink, Trino 등)과 데이터 저장소(S3, Azure Blob, HDFS) 사이에서 독립적인 테이블 추상화 레이어를 제공하여 "엔진 독립성"을 유지


### 2.2 개발 역사

1. **태동기(2017년) : 넷플릭스(Netflix) 사내 프로젝트로 출발**
    - 대규모 분석 데이터셋 관리를 위해 **Apache Hive** 테이블 포맷을 사용 중, 데이터의 규모가 페타바이트(PB) 규모로 거대화됨
    - Apache Hive의 '디렉터리 기반 추적' 방식의 한계
        - 수천 개의 파티션을 디렉터리 리스팅(Listing)하는 데만 수분이 소요됨
        - 동시 쓰기 작업 시 데이터가 깨지는 고질적인 정합성 문제
    - Hive의 한계를 극복하기 위한 프로젝트로 개발 시작
        - 넷플릭스의 엔지니어 라이언 블루(Ryan Blue)와 단 위크스(Dan Weeks)
        - 디렉터리가 아닌 **'파일 단위'로 스냅샷을 추적하는 새로운 테이블 포맷 사양**을 설계하기 시작

2. **오픈소스 전환기 (2018년 ~ 2020년) : 아파치 재단 기증 및 탑레벨 프로젝트 등극**
    - **2018년 11월:**
        - 전 세계 데이터 엔지니어링 커뮤니티의 참여와 엔진 독립성을 보장하기 위해 Iceberg 프로젝트를 오픈소스로 공개
        - **아파치 소프트웨어 재단(ASF) 인큐베이터에 기증** 🡲 현재는 아파치 소프트웨어 재단에서 관리 중
    - **2020년 5월:**
        - 프로젝트의 성숙도와 커뮤니티 확장성을 인정받아 아파치 재단의 탑레벨 프로젝트(Top-Level Project, TLP)로 공식 졸업
        - 이 시기를 기점으로 에어비앤비(Airbnb), 애플(Apple), 익스피디아(Expedia), 링크드인(LinkedIn) 등 글로벌 빅테크 기업들이 프로덕션 환경에 Iceberg를 대거 채택하기 시작

3. **사양(Specification)의 진화와 생태계 확장 (2021년 ~ 2024년)**
    - Iceberg는 기능을 고도화하며 버전 스펙(Spec)을 점진적으로 발표
        - **Spec V1 (Analytic Tables):**
            - 대규모 분석용 불변 데이터 파일(Parquet, ORC, Avro)에 RDBMS 수준의 ACID 트랜잭션과 스키마 진화를 제공하는 기반 확림
        - **Spec V2 (Row-level Updates/Deletes):**
            - 2021~2022년경 확정된 스펙
            - 데이터 레이크하우스에서 가장 까다로웠던 행 단위의 즉각적인 수정/삭제(Upsert/Delete)를 지원하기 위해 Merge-on-Read(MoR) 메커니즘 도입
        - **상업적 생태계 폭발 (2024년):**
            - Snowflake, Google Cloud(BigQuery), AWS 등이 Iceberg를 자사 플랫폼의 핵심 외부 테이블 포맷으로 기본 지원 시작
            - 2024년 6월, 데이터브릭스(Databricks)가 Iceberg 메인 관리 기업인 '타불러(Tabular)'를 인수 🡲 메이저 진영 간의 포맷 표준화 경쟁이 Iceberg 중심으로 크게 기울게 됨

4. **현대 및 미래 (2025년 ~ 2026년 현재) : 차세대 Spec V3 도입과 천하통일**
    - **Spec V3 출시:**
        - 최근 정식 지원되기 시작
        - 읽기 성능 오버헤드를 극적으로 줄여주는 바이너리 삭제 벡터(Deletion Vectors) 도입
        - 반정형 JSON 데이터를 고속 파싱할 수 있는 **Variant 타입 표준** 도입
    - **현재 상태:**
        - 특정 컴퓨팅 엔진이나 벤더(Vendor Lock-in)에 종속되지 않는 '진정한 오픈 레이크하우스의 표준 레이어'로 완벽하게 자리 잡음
        - 클라우드뿐만 아니라 온프레미스 스마트팩토리 AI 인프라 등 이기종 데이터 소스를 통합해야 하는 엔터프라이즈 환경에서 필수 아키텍처로 다뤄짐

<br>

> - 아파치 아이스버그(Apache Iceberg)의 개발 역사는 빅데이터 생태계가 
> - '하둡(Hadoop) 중심의 파일/디렉터리 관리'에서 **'클라우드 오브젝트 스토리지 중심의 데이터 레이크하우스(Lakehouse)'**로
> - 패러다임이 전환되는 과정을 그대로 담고 있음
{: .summary-quote}


### 2.3 주요 아키텍처 (3 Layer 구조)

- 메타데이터를 계층화하여 관리함으로써 성능과 안정성을 확보함
- 디렉터리 경로가 아닌 **파일 중심의 트리 구조 메타데이터**를 통해 스냅샷을 관리
- 이 구조 덕분에 O(1)에 가까운 속도로 쿼리 플래닝이 가능함

<div class="insert-image">
    <img src="/materials/devtools/images/S13-03-01-01_01-001_Iceberg_Architecture.png" style="width: 60%;">
</div>

- **Catalog Layer:**
    - 최신 메타데이터 포인터(`version-hint.text`)를 관리하는 루트 계층
        - REST Catalog, AWS Glue, Hive Metastore 등
    - 현재 테이블의 최신 상태(Current Metadata File)가 어디인지 가리키는 포인터 역할

- **Metadata Layer:**
    - **Metadata File (`.json`):**
        - 테이블의 '상태 변경 이력'을 기록
        - 스키마, 파티션 명세, 그리고 각 시점의 스냅샷(Snapshot) ID를 포함
    
    - **Manifest List (`.avro`):**
        - 특정 스냅샷 시점에 유효한 Manifest 파일들의 목록 관리
        - 각 Manifest 파일이 담고 있는 데이터의 통계 정보(파티션 범위 등)를 관리
    
    - **Manifest File (`.avro`):**
        - 실제 데이터 파일(`.parquet`, `.orc` 등)의 절대 경로와 행(Row) 개수, 컬럼별 최소/최대값(Min/Max) 등의 통계 정보 보유
        - 쿼리 엔진은 이 단계를 읽고 불필요한 데이터 파일을 원천 배제(Data Skipping)함
            - 쿼리 시 불필요한 파일을 건너뛰게(Pruning) 함

- **Data Layer:**
    - 실제 데이터가 저장되는 레이어


### 2.4 핵심 동작 메커니즘 및 고급 기능

- **낙관적 동시성 제어 (Optimistic Concurrency Control, OCC) 기반 ACID**
    - 여러 쓰기 작업(Writers)이 동시에 발생할 때 서로 충돌하지 않는다고 가정하고 각자 스냅샷을 준비함
    - 여러 프로세스가 동시에 데이터를 쓰거나 읽어도 데이터의 일관성을 유지함 🡲 **ACID 트랜잭션 보장**
        - 커밋 시점에 다른 작업이 먼저 메타데이터 포인터를 업데이트했다면,
            - 내 작업을 취소하고
            - 변경된 최신 상태 위에서 자동 재시도(Atomic Swap & Retry)를 수행하여
            - 데이터 일관성을 유지함
    - 단순히 ACID 보장을 넘어, 낙관적 동시성 제어를 기반으로 **직렬 가능성(Serializability)** 또는 스냅샷 격리(Snapshot Isolation)를 제공
    - **타임 트래블 및 롤백 (Time Travel):**
        - 특정 시점의 스냅샷으로 쿼리를 실행하거나, 잘못된 업데이트 발생 시 이전 상태로 복구할 수 있음

- **숨겨진 파티셔닝 (Hidden Partitioning) & 파티션 진화**
    - **Hidden Partitioning:**
        - 사용자가 쿼리에 파티션 컬럼을 명시하지 않아도, Iceberg가 데이터의 상관관계를 파악해 필요한 파티션만 읽음
        - 유저가 `WHERE timestamp >= ...`로 조회하면,
            - Iceberg가 내부에 정의된 파티션 함수(예: `days(timestamp)`)를 식별하여
            - 알아서 해당 파티션만 읽음
        - 유저가 쿼리에 `WHERE date_col = ...`을 직접 매핑해 줄 필요가 없음

    - **Partition Evolution:**
        - 파티션 구조 변경 시에도 과거 쿼리를 수정할 필요가 없음
            - 컬럼 추가, 삭제, 이름 변경, 타입 변경 시 **데이터 파일을 다시 쓰지 않고** 메타데이터만 업데이트하여 즉각 반영
            - 데이터 규모가 커져서 일별(Daily) 파티션을 월별(Monthly)로 바꾸더라도, **과거 데이터를 재배치(Rewrite)할 필요가 없음**
        - 메타데이터에 파티션 버전(Partition Spec)을 기록하여,
            - 이전 데이터는 과거 규칙대로,
            - 신규 데이터는 새 규칙대로
            - 하나의 테이블 안에서 동시 조회

- **행 단위 변경 전략 (CoW vs MoR)**
    - Iceberg는 데이터의 특성(배치 위주 vs 스트리밍/CDC 위주)에 따라 두 가지 쓰기 모드를 지원함
        - **Copy-on-Write (CoW):**
            - 특정 행이 업데이트/삭제되면 🡲 해당 행이 속한 데이터 파일 전체를 새로 씀
            - **(읽기 속도 최적화)**

        - **Merge-on-Read (MoR):**
            - 기존 데이터 파일은 그대로 두고,
            - 변경된 행의 위치나 조건만 담은 삭제 파일(Delete File)을 따로 기록한 뒤 읽기 시점에 병합
            - **(쓰기 속도 최적화)**

- **최신 Iceberg V3 스펙의 진화**
    - **바이너리 삭제 벡터 (Deletion Vectors):**
        - V2의 MoR 방식에서 발생하던 삭제 파일 누적 문제를 해결하기 위해,
        - 비트맵 형태의 고성능 바이너리 삭제 벡터를 도입 🡲 읽기 병합 오버헤드를 극적으로 줄임

    - **Variant 타입 지원:**
        - 반정형 데이터(JSON)를 파켓(Parquet) 내부에 효율적인 바이너리 트리 구조로 저장
        - 스키마 변동이 심한 데이터도 고속으로 쿼리할 수 있게 됨


### 2.5 장단점 분석

- **장점 (Pros)**
    - **성능 최적화:**
        - 파일 단위의 통계(Min/Max)를 메타데이터에 들고 있어 쿼리 실행 시 I/O를 획기적으로 줄임

    - **엔진 범용성:**
        - Spark, Trino, Flink, Hive, Presto 등 다양한 컴퓨팅 엔진에서 동일한 테이블에 접근 가능

    - **신뢰성:**
        - 쓰기 작업 중 장애가 발생해도 Snapshot 방식 덕분에 데이터 오염이 발생하지 않음

    - **저렴한 비용:**
        - 비싼 데이터 웨어하우스(DW) 대신 S3 같은 저렴한 오브젝트 스토리지를 DW처럼 활용(Lakehouse)할 수 있음

    - **벤더 락인(Lock-in) 해제:**
        - 특정 솔루션에 종속되지 않는 REST Catalog 표준 사양을 제시
        - 하나의 데이터(S3)를 두고 AWS EMR(Spark), Trino, Snowflake가 동시에 읽고 쓸 수 있는 진정한 오픈 레이크하우스 인프라를 가능하게 함

    - **안정적인 대규모 파일 스캔:**
        - 수십억 개의 파일이 있는 초거대 테이블에서도 메타데이터 수준에서 쿼리 대상을 확 줄여줌<br>
            🡲 오브젝트 스토리지의 `LIST` API 호출 비용과 쿼리 플래닝 타임을 최소화

- **단점 (Cons)**
    - **주기적인 메타데이터 관리(Compaction) 필수:**
        - 스냅샷이 쌓일수록 메타데이터 파일이 늘어남
            - MoR 방식을 쓰거나 스트리밍 인프라(Flink 등) 연결 시 작은 파일과 삭제 파일이 폭발적으로 늘어남
        - 주기적으로 오래된 스냅샷을 삭제하고 데이터 파일을 병합(Compaction)하는 관리 작업이 필요함
            - 주기적으로 병합해주는 `optimize` 및 오래된 스냅샷을 정리하는 `expire_snapshots` 관리 파이프라인을 필수적으로 구축·운영해야 성능 저하를 막을 수 있음

    - **작은 파일 문제:**
        - 실시간 스트리밍으로 데이터를 넣을 경우 작은 파일(Small Files)이 많이 생성되어 성능이 저하될 수 있음
        - 별도의 튜닝이 필수적

    - **쓰기 동시성 충돌 (OCC Conflict):**
        - 마이크로 배치나 실시간 인프라에서 수많은 컴포넌트가 한 테이블에 동시다발적으로 쓰기를 시도하면<br>
            🡲 커밋 충돌로 인한 재시도 횟수가 늘어나고, 심할 경우 쓰기 실패가 발생할 수 있음 (이 경우 Hudi가 더 유리할 수 있음)

    - **학습 곡선:**
        - 기존 Hive 방식과는 아키텍처가 다르므로 운영 조직의 이해도가 요구됨
            - 기존의 데이터 레이크 방식
                - 데이터를 파일 시스템(HDFS, S3 등)의 '디렉터리 구조'로 관리함
                - 예: Hive 등

            - Iceberg의 데이터 레이크 방식
                - **'파일 단위'의 스냅샷**을 통해 테이블의 상태를 추적함



- **타 포맷(Delta Lake, Hudi) 대비 장단점 비교**

<div class="info-table">
<table>
    <thead>
        <th style="width: 120px;">비교 항목</th>
        <th style="width: 320px;">Apache Iceberg</th>
        <th style="width: 320px;">Delta Lake</th>
        <th style="width: 230px;">Apache Hudi</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">거버넌스</td>
            <td class="td-left">완전 개방형 (ASF)</td>
            <td class="td-left">Linux Foundation (실질적 Databricks 주도)</td>
            <td class="td-left">개방형 (ASF)</td>
        </tr>
        <tr>
            <td class="td-rowheader">엔진 독립성</td>
            <td class="td-left">최상 (Spark, Trino, Flink, Snowflake, BigQuery 등 대등하게 지원)</td>
            <td class="td-left">Spark / Databricks 최적화 (타 엔진 지연 지원 존재)</td>
            <td class="td-left">Spark / Flink 중심</td>
        </tr>
        <tr>
            <td class="td-rowheader">파티셔닝</td>
            <td class="td-left">숨겨진 파티셔닝 & 진화 지원</td>
            <td class="td-left">컬럼 매핑 기반 제한적 지원</td>
            <td class="td-left">수동 관리 필요</td>
        </tr>
        <tr>
            <td class="td-rowheader">주요 활용처</td>
            <td class="td-left">멀티 엔진 기반 오픈 레이크하우스</td>
            <td class="td-left">Databricks 중심 에코시스템</td>
            <td class="td-left">대규모 무중단 스트리밍 및 CDC</td>
        </tr>
    </tbody>    
</table>
</div>


### 2.6 활용도 (Use Cases)

- **데이터 레이크하우스 구축:**
    - AWS S3나 Azure Data Lake Storage 위에서 DW 기능을 구현할 때 핵심 기술로 사용됨

- **실시간 및 배치 통합 처리:**
    - Flink를 이용한 실시간 CDC(Change Data Capture) 데이터를 Iceberg에 적재
    - 동시에 Spark로 배치 분석을 수행하는 아키텍처에 적합함

- **머신러닝(ML) 데이터 관리:**
    - AI 연구 시 데이터 재현성(Reproducibility)이 중요함
    - 타임 트래블 기능을 통해 특정 시점의 학습 데이터를 정확히 추출할 수 있음

- **규제 준수 (GDPR/CCPA):**
    - 특정 사용자의 데이터를 삭제해야 할 때, 기존 파일 시스템 방식보다 훨씬 정교하고 안전하게 레코드 단위 삭제를 처리할 수 있음

<br>

> - Apache Iceberg는
>   - 스토리지 비용은 데이터 레이크 수준으로 낮추고, 데이터 관리는 RDBMS/DW 수준으로 끌어올리는 표준 프레임워크
>   - 특히 특정 벤더에 종속되지 않고 **Spark(배치 처리), Flink(실시간 스트리밍), Trino(Ad-hoc 대화형 쿼리)** 등 다양한 이기종 엔진을 융합하여 데이터 플랫폼을 설계할 때 가장 강력한 시너지를 발휘함
>   - 실무 도입 시에는 작은 파일 병합 전략(Compaction 튜닝)과 비즈니스 요구사항에 맞는 **CoW/MoR 전략 선택**이 성공의 핵심 요인
{: .expert-quote}


## 3. Apache Iceberg 환경 설정

- Iceberg는 컴퓨팅 엔진이 아니라 '파일을 배치하는 규격(Spec)'이자 '메타데이터 명세'임
- 기본 기능을 중심으로 할 때
    - **Python 코드 몇 줄만으로도 데이터를 읽고 쓰며 Iceberg의 메타데이터 트리 구조가 어떻게 생성되는지 완벽하게 확인**할 수 있음

- **최소 아키텍처**
    - **Client:** PyIceberg (순수 파이썬 라이브러리, 가볍고 독립적임)
    - **Catalog:** Iceberg REST Catalog (테이블의 최신 포인터만 관리)
    - **Storage:** MinIO (실제 메타데이터 `.json` 파일과 데이터 `.parquet` 파일이 저장되는 곳)


### 3.1 환경 구축 (Docker Compose & Python 환경)

1. **`docker-compose.yml`** 작성

    ```yaml
    version: '3.8'

    services:
    # 1. Iceberg의 두뇌 역할을 하는 카탈로그 서버
    rest-catalog:
        image: tabulario/iceberg-rest:0.6.0
        container_name: iceberg-rest-catalog
        ports:
        - "8181:8181"
        environment:
        - CATALOG_WAREHOUSE=s3a://warehouse/
        - CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO
        - CATALOG_S3_ENDPOINT=http://minio:9000
        - CATALOG_S3_PATH_STYLE_ACCESS=true
        - AWS_ACCESS_KEY_ID=admin
        - AWS_SECRET_ACCESS_KEY=password
        depends_on:
        - minio

    # 2. 실제 파일들이 저장될 오브젝트 스토리지
    minio:
        image: minio/minio:RELEASE.2024-01-11T06-46-16Z
        container_name: iceberg-minio
        ports:
        - "9000:9000"       # API 포트
        - "9001:9001"       # 웹 관리 콘솔 포트
        environment:
        - MINIO_ROOT_USER=admin
        - MINIO_ROOT_PASSWORD=password
        command: server /data --console-address ":9001"

    # MinIO 기동 시 'warehouse' 버킷을 자동으로 생성해주는 유틸리티
    mc:
        image: minio/mc:RELEASE.2024-01-11T06-46-16Z
        depends_on:
        - minio
        entrypoint: >
        /bin/sh -c "
        until (/usr/bin/mc alias set myminio http://minio:9000 admin password) do echo 'Waiting for MinIO...' && sleep 1; done;
        /usr/bin/mc mb myminio/warehouse;
        exit 0;
        "
    ```

    - 터미널에서 `docker-compose up -d`를 실행하여 컨테이너 구동

2. **로컬 Python 환경 설정 (PyIceberg 설치)**

- 로컬 PC 또는 개발 서버의 가상환경에서 아래 명령어로 Iceberg 조작을 위한 파이썬 패키지 설치
    - PyArrow 기반으로 작동하므로 매우 빠르고 가벼움

    ```bash
    pip install "pyiceberg[s3fs,pyarrow]"
    ```


### 3.2 Iceberg 조작 및 메타데이터 추적 (Python 실습)

- 실제 저장소(MinIO)에 어떤 파일들이 파일 단위로 꼽히는지 확인하기

- **[1 단계] 카탈로그 연결 및 네임스페이스(DB) 생성**

    ```python
    from pyiceberg.catalog import load_catalog
    import pyarrow as pa

    # 1. Docker로 띄운 REST 카탈로그와 MinIO 스토리지 정보 설정
    catalog = load_catalog(
        "default",
        **{
            "type": "rest",
            "uri": "http://localhost:8181",
            "s3.endpoint": "http://localhost:9000",
            "s3.access-key-id": "admin",
            "s3.secret-access-key": "password",
        }
    )

    # 2. 네임스페이스(RDBMS의 Database 개념) 생성
    catalog.create_namespace("factory_db")
    print("네임스페이스 생성 완료!")
    ```

- **[2 단계] 테이블 생성 (스키마 정의)**

    ```python
    # 3. PyArrow를 이용해 테이블 구조(Schema) 정의
    schema = pa.schema([
        ("sensor_id", pa.string()),
        ("reading", pa.float64()),
    ])

    # 4. Iceberg 테이블 생성
    table = catalog.create_table("factory_db.sensors", schema=schema)
    print("Iceberg 테이블 생성 완료!")
    ```

    - **🔍 저장소 확인 포인트:**
        - 이 단계를 실행한 후 `http://localhost:9001` (MinIO 콘솔)에 접속해 보면,
        - `warehouse/factory_db/sensors/metadata/` 경로에 `v1.metadata.json` 파일이 생성된 것을 볼 수 있음
        - 아직 데이터 파일은 없고 **"테이블이 만들어졌다"는 메타데이터 스펙만 기록**된 상태임


- **[3 단계] 데이터 삽입 (첫 번째 스냅샷 생성)**

    ```python
    # 5. 삽입할 데이터 준비 (Arrow Table 형태)
    data_v1 = pa.Table.from_pydict({
        "sensor_id": ["SNS-001", "SNS-002"],
        "reading": [23.5, 42.1]
    })

    # 6. 테이블에 데이터 추가 (Append) -> 내부적으로 ACID 트랜잭션 커밋 발생
    table.append(data_v1)
    print("첫 번째 데이터 업로드 완료!")
    ```

    - **🔍 저장소 확인 포인트:**
        - `data/` 폴더가 새로 생기며 그 밑에 실제 데이터가 담긴 `.parquet` 파일이 생성됨
        - `metadata/` 폴더에는 `v2.metadata.json`과 함께 스냅샷 정보를 담은 `manifest_list` 및 `manifest` 파일(`.avro`)이 추가됨
        - **이것이 바로 Iceberg가 스냅샷 단위로 파일을 추적하는 실체**


- **[4 단계] 데이터 추가 삽입 및 타임 트래블(Time Travel) 검증**

    ```python
    # 7. 두 번째 데이터 준비 및 추가
    data_v2 = pa.Table.from_pydict({
        "sensor_id": ["SNS-003"],
        "reading": [11.8]
    })
    table.append(data_v2)

    # 8. 현재 최신 상태 조회 (SNS-001, 002, 003 총 3개 행이 나옴)
    print("--- 최신 데이터 상태 ---")
    print(table.scan().to_arrow())

    # 9. 타임 트래블: 과거 스냅샷 이력 확인 후 첫 번째 시점으로 되돌려 조회
    history = table.history()
    first_snapshot_id = history[0].snapshot_id

    print(f"\n--- 첫 번째 스냅샷({first_snapshot_id}) 시점으로 타임 트래블 ---")
    print(table.scan(snapshot_id=first_snapshot_id).to_arrow())
    ```
