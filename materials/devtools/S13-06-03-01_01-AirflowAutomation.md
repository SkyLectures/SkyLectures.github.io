---
layout: page
title:  "수집 🡲 Lake 🡲 Spark 🡲 VectorDB 흐름 자동화"
date:   2026-07-07 10:00:00 +0900
permalink: /materials/S13-06-03-01_01-AirflowAutomation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 오케스트레이션 아키텍처

> - **"수집 🡲 Data Lake 🡲 Apache Spark 🡲 VectorDB"로 이어지는 파이프라인:**
>   - 현대적인 **Enterprise AI 및 대규모 하이브리드 RAG(Retrieval-Augmented Generation)** 시스템의 표준 백엔드 아키텍처
>   - 실무 환경에서는 테라바이트급 대용량 비정형 데이터(문서, 로그)가 유입되므로, 단일 Airflow 워커에서 데이터를 가공하는 것은 불가능
>   - 따라서 **Airflow는 제어(Orchestration)만 담당하고, 실제 중량 연산은 분산 컴퓨팅 엔진(Spark)에 위임**하는 구조를 취함
{: .common-quote}

### 1.1 정의 및 개념

- **오케스트레이션 아키텍처(Orchestration Architecture)**
    - 데이터 엔지니어링 및 AI 인프라에서 분산된 시스템, 서비스, 복잡한 데이터 파이프라인의
        - **실행 순서, 의존성, 자원 할당, 예외 처리 등을 중앙에서 전반적으로 조율(Orchestrate)하고 통제하는 핵심 프레임워크**를 의미
    - 단순히 "스케줄러에 맞춰 배치 스크립트를 실행하는 것"을 넘어,
        - 거대한 분산 컴퓨팅 환경을 안전하고 멱등성 있게 관리하기 위한 고도의 시스템 아키텍처
    - 수많은 분산 인프라와 데이터 소스들 사이에서 다음에 대한 해답을 제시하는 고도의 시스템 설계 기술
        - 어떻게 결함을 격리하고,
        - 어떻게 자원을 최적화하며,
        - 어떻게 데이터 흐름을 안전하게 보장할 것인가

- **아키텍처 유형: 패턴 기반 분류**
    - 오케스트레이션은 복잡한 하부 서비스를 조율하는 방식에 따라 크게 두 가지 설계 패턴으로 나뉨

    - **오케스트레이션 패턴 (Orchestration Pattern) - 중앙 집중형**
        - 중앙에 강력한 **'지휘자(Orchestrator)'** 역할을 하는 엔진(예: Apache Airflow, Temporal, Prefect)을 두고,
        - 이 엔진이 모든 서비스와 태스크의 상태를 직접 제어하고 명령을 내리는 방식

        - **장점:**
            - 전체 파이프라인의 워크플로우와 상태를 중앙(Web UI 등)에서 한눈에 모니터링하고 가시화할 수 있음
            - 에러 발생 시 중앙에서 재시도나 결함 격리를 즉각 통제할 수 있음
        - **단점:**
            - 중앙 오케스트레이터 엔진이 마비되거나 메타데이터 DB가 다운되면
                - 전체 시스템 파이프라인이 마비되는 단일 장애점(SPOF, Single Point of Failure)이 될 수 있음

    - **코레오그래피 패턴 (Choreography Pattern) - 이벤트 분산형**
        - 중앙의 통제자 없이, 각 서비스들이 메시지 브로커(Kafka, RabbitMQ)를 통해 이벤트(Event)를 발행하고 수신하며
        - 자율적으로 춤추듯 상호작용하는 무용(Choreography) 방식

        - **장점:**
            - 서비스 간의 결합도(Coupling)가 극도로 낮으며,
            - 특정 서비스가 죽어도 다른 서비스는 이벤트를 계속 처리할 수 있어 확장성과 가용성이 뛰어남
        - **단점:**
            - 파이프라인의 전체 데이터 흐름을 한눈에 파악하기 어렵고,
            - 특정 구간에서 데이터 정합성이 깨졌을 때
                - 역추적(디버깅) 및 분산 트랜잭션 롤백(Saga 패턴 구현 등)의 난이도가 비약적으로 상승

    > - **AI 및 데이터 파이프라인의 선택:**
    >   - 데이터 수집 🡲 전처리 🡲 모델 학습으로 이어지는 엄격한 선후 관계와 인과관계가 중요한
    >   - **AI/데이터 엔지니어링 영역에서는 가시성과 제어권이 명확한 '오케스트레이션 패턴(Apache Airflow 등)'을 압도적으로 선호**
    {: .common-quote}


### 1.2 오케스트레이션 아키텍처의 4대 핵심 컴포넌트

- 현대적인 오케스트레이션 엔진은 내부적으로 고도의 분산 시스템 구조를 채택하고 있음

    <div class="insert-image" style="text-align: center;">
        <img src="/materials/devtools/images/S13-06-03-01_01-001_AirflowAutomation.png" style="width: 60%;">
    </div>

    1. **컨트롤 플레인 & 스케줄러 (Control Plane & Scheduler):**
        - 전체 파이프라인의 뼈대(DAG)를 해석하고,
        - 각 태스크의 의존성과 진입차수(In-degree)를 계산하여
        - 실행 가능한 작업을 선별하는 역할

    2. **메타데이터 저장소 (Metadata Repository):**
        - 모든 워크플로우의 실행 이력, 태스크의 라이프사이클 상태(Queued, Running, Failed, Success), 전역 설정값 등을 영구 저장하는 아키텍처의 심장부
        - RDBMS 주로 사용

    3. **실행기 및 큐 인프라 (Executor & Message Queue):**
        - 스케줄러가 선별한 태스크를 실제 일꾼(Worker)들에게 안전하게 전달하기 위한 버퍼 및 중계 계층
        - Redis, RabbitMQ 같은 브로커나 쿠버네티스 API API와 연동됨

    4. **분산 워커 클러스터 (Distributed Workers):**
        - 오케스트레이터의 명령을 받아 실제 컴퓨팅 연산(API 호출, SQL 트리거, Spark 직렬화)을 수행하는 물리/논리적 노드


### 1.3 오케스트레이션 설계 시 필수 아키텍처 원칙

- 성공적인 오케스트레이션 파이프라인을 구축하기 위해 아키텍처 레벨에서 반드시 준수해야 하는 엔지니어링 원칙

    - **컴퓨팅과 오케스트레이션의 분리 (Decoupling)**
        - 가장 중요한 원칙 🡲 "오케스트레이터는 신호등 역할만 해야지, 직접 차가 되어서는 안 된다"는 법칙
        - 오케스트레이터 워커 자체의 메모리와 CPU를 소모하여 대용량 데이터를 처리(예: Pandas 데이터 변환)하면 시스템 전체가 마비됨
        - 무거운 연산은 분산 컴퓨팅 엔진(Spark, Ray, Trino)이나 외부 DB 엔진에 위임하고,
        - 오케스트레이터는 실행 명령(Trigger)과 완료 여부 확인(Polling)만 수행해야 함

    - **멱등성 (Idempotency) 인프라 구축**
        - 오케스트레이션 아키텍처에서는 특정 태스크가 실패하여 재실행(Retry)되거나 과거 특정 시점으로 돌아가 백필(Backfill)을 수행할 때,
            - **몇 번을 다시 실행해도 타겟 저장소의 최종 데이터셋 결과가 항상 동일함을 보장**해야 함
        - 데이터 소스를 격리할 수 있는 논리적 시점 변수(Logical Date) 바인딩 및 저장소의 Upsert 메커니즘 설계가 아키텍처에 내재되어야 함

    - **결합 격리 및 자원 격리 (Isolation)**
        - 파이프라인 내의 각 단계는 상호 간의 라이브러리 의존성이나 하드웨어 자원 소비에 영향을 주지 않아야 함
        - AI 파이프라인에서는 일반 가벼운 SQL 전처리 태스크와 대용량 GPU 가속이 필요한 모델 학습 태스크가 공존하므로,
        - 태스크 단위를 **컨테이너(Docker Pod) 형태로 동적 격리**하여
        - 필요한 인프라에 실시간 배정하는 아키텍처(예: `KubernetesPodOperator`)를 구축하는 것이 최선


### 1.4 기술 스택별 역할 정의

- **Ingestion (수집):**
    - 대내외 소스(API, DB, 웹훅)로부터 가이드, 매뉴얼 등의 원시 데이터를 수집

- **Data Lake (저장):**
    - 비용이 저렴하고 확장성이 뛰어난 오브젝트 스토리지(오픈소스 MinIO 또는 AWS S3)를 아키텍처의 중심에 둠

- **Distributed Processing (Apache Spark):**
    - 데이터 분석 및 대규모 분산 연산의 표준 엔진
    - 수집된 대용량 텍스트의 정제, 형태소 분석, 토큰 크기 기반 분산 청킹(Chunking)을 처리함

- **Vector Database (Qdrant):**
    - 고차원 벡터 임베딩 데이터를 저장하고,
    - 코사인 유사도(Cosine Similarity) 등의 알고리즘을 기반으로 밀집 검색(Dense Retrieval)을 초고속으로 수행하는
    - 하이브리드 검색 엔진


### 1.5 오케스트레이션 프로세스 타임라인

- Airflow 스케줄러가 전체 파이프라인의 생명주기를 관리하는 4단계 프로세스

1. **[Task 1] Ingestion & Lake Landing (수집 및 적재):**
    1. 외부 데이터를 다운로드하여
    2. MinIO의 `raw-data/` 버킷에 저장

2. **[Task 2] Spark 분산 연산 트리거 (Spark-Submit):**
    1. Airflow가 Spark Operator를 통해 전처리 작업을 명령
    2. Spark 클러스터가 MinIO의 원시 데이터를 읽어와 청킹(Chunking)을 수행
    3. `processed-data/` 버킷에 Parquet 형식으로 저장

3. **[Task 3] 병렬 분산 임베딩 및 Vector DB Upsert:**
    1. 가공된 텍스트 청크들을 읽어와
    2. 로컬 AI 임베딩 모델(Ollama/HuggingFace)을 통해 고차원 벡터로 변환한 뒤,
    3. Qdrant의 HNSW 그래프 인덱스에 **Upsert**

4. **[Task 4] 인덱스 정비 및 캐시 클린업:**
    - 메타데이터 갱신 및 리소스 해제


## 2. 종합 실습 예제 코드 1

- 현대적인 AI 인프라의 표준 구조인 **하이브리드 RAG(검색 증강 생성) 플랫폼의 데이터 공급선**을 아키텍처 관점에서 프로토타이핑한 핵심 설계도
- "수집 🡲 Data Lake 🡲 Apache Spark 🡲 VectorDB" 구조를 단일 파일로 구현한 Airflow DAG
    - 실무에서는 Spark 전처리 로직을 별도의 `*.py` 파일로 분리하여 `SparkSubmitOperator`로 호출
    - 실습예제에서는 가독성을 위해 PySpark 전처리 및 Qdrant 적재를 통합 구현함

### 2.1 아키텍처적 구성 요소 (Components)

- **중앙 지휘자 (Apache Airflow DAG):**
    - `task_ingest`와 `task_spark_and_vector`라는 두 개의 실행 노드를 통제
    - 어떤 작업이 먼저 실행되어야 하는지 선후 관계를 규정
    - 장애 발생 시 자동으로 재시도(`retries: 2`)하는 관리 계층

- **원시 데이터 레이크 (MinIO):**
    - 오브젝트 스토리지 영역
    - 비정형 데이터(매뉴얼 텍스트)를 가공되지 않은 순수 스냅샷 상태 그대로 안전하게 영구 저장(`factory-raw-logs` 버킷)

- **분산 컴퓨팅 가공 및 고차원 저장소 (Spark & Vector DB):**
    - 코드는 인라인으로 가볍게 구현되어 있으나
    - 논리적으로는 대형 텍스트 유입 연산(Spark)을 수행하여
    - 이를 고속 그래프 탐색 인덱스인 HNSW 그래프 구조(Qdrant)로 동기화하는
    - AI 데이터 서빙 컴포넌트


### 2.2 코드의 정적 구조 (Static Structure)

- 코드는 크게 세 개의 영역(Configuration, Implementation, Orchestration)으로 레이어가 나뉘어 있음

    ```text
    [ 구조적 레이어 구성 ]
    ├─ 1. 전역 인프라 토폴로지 정의 레이어 (설정 구역: MINIO_URL, QDRANT_URL 등)
    ├─ 2. 비즈니스 로직 구현 레이어 (실행 함수: fn_ingest_to_lake, fn_spark_...)
    └─ 3. 워크플로우 오케스트레이션 레이어 (with DAG(...) 구문 및 의존성 선언)
    ```

- **전역 인프라 토폴로지 정의 레이어 (최상단 구역)**
    - 외부 인프라 시스템들의 접속 주소와 인증 정보(`MINIO_ACCESS`, `QDRANT_URL` 등) 및 적재 대상이 될 물리 공간(`BUCKET_NAME`, `COLLECTION_NAME`)을 전역 변수로 규정

- **비즈니스 로직 구현 레이어 (중단 구역)**
    - **`fn_ingest_to_lake()`:**
        - MinIO SDK 클라이언트를 선언하고
        - `bucket_exists` API를 통해 방어적 코드(Defensive Coding)를 구축한 뒤,
        - 스마트팩토리 도메인의 유량계 정비 매뉴얼 비정형 데이터를
        - 레이크에 적재하는 함수

    - **`fn_spark_processing_and_vector_upsert()`:**
        - 데이터 레이크에서 파일을 다시 스트리밍으로 읽어와
        - `replace` 연산으로 노이즈를 제거하고,
        - 50글자 단위로 잘라내는 의미론적 청킹(Chunking)을 처리
        - 이후 Qdrant의 ANN(근사 최근접 이웃) 유사도 검색을 위해
        - 5차원 공간 벡터 구조(`PointStruct`)로 포맷팅하여 적재를 전담

- **워크플로우 오케스트레이션 레이어 (하단 구역)**
    - **`with DAG(...) as dag:`**
        - 컨텍스트 매니저를 통해 Airflow 컴파일러 내부로 진입
        - `PythonOperator`들을 활용해
        - 앞서 구현한 비즈니스 파이썬 함수들을
        - Airflow 가시적인 태스크 노드로 인스턴스화


### 2.3 동적 데이터 흐름 및 메커니즘 (Runtime Flow)

- DAG가 트리거되는 순간, 데이터와 제어권은 선후 관계에 맞춰 물리 자원을 이동하게 됨

    ```text
    [ 데이터 및 제어권 흐름 타임라인 ]

    (외부 소스 데이터)
            │
            ▼
    [ 1단계: task_ingest ] ───➔ MinIO (enterprise-knowledge-lake) 저장 완료
            │
            ├─ (의존성 제어권 이행: '>>')
            ▼
    [ 2단계: task_spark_and_vector ]
            ├─ ① 데이터 레이크로부터 원시 텍스트 스트리밍 Load
            ├─ ② 분산 메모리 공간 청킹 연산 (40~50자 분할)
            └─ ③ Qdrant 고차원 벡터 특징 공간 맵핑 (HNSW 인덱싱)
    ```

    1. **진입 관문 및 데이터 레이크 안착 (`task_ingest`):**
        - Airflow 스케줄러가 진입차수(`In-degree=0`)가 제로인 `task_ingest`를 가장 먼저 큐에 넣고 워커에 배정
        - 가상의 비정형 소스 텍스트 데이터가
            - 바이트 스트림(`io.BytesIO`) 형태로 변환되어
            - 네트워크 망을 타고
            - **MinIO 오브젝트 스토리지 내부의 `raw/manual_01.txt` 경로로 업로드** 및 격리

    2. **순차 제어권 이행 (`>>`):**
        - 1단계 태스크가 성공(`Success`)으로 마킹되면,
        - 의존성 결합 연산자(`>>`)를 타고
        - 제어권이 다음 태스크인 `task_spark_and_vector`로 안전하게 전이

    3. **데이터하우스 로드 및 메모리 청킹 연산:**
        - 두 번째 태스크가 기동되면서 방금 MinIO에 백업되었던 원시 파일의 스냅샷을 메모리로 다시 다운로드
        - `replace` 가공을 통해 문자열 노이즈를 정제하고,
            - 슬라이드 윈도우 방식으로 50글자씩 쪼개진 텍스트 스트링 리스트(`chunks`)를 생성하여
            - 메모리에 분산 배치

    4. **멱등성 기반 고차원 벡터스토어 최종 동기화:**
        - Qdrant 클라이언트가 가동되어
            - 벡터스토어 내부에 `Distance.COSINE` 거리를 연산할 수 있는 인덱스 레이어를 선언
        - 청킹된 문자열 데이터 각각에 유일한 ID 고유 번호(`idx + 100`)를 강제로 부여
        - 이렇게 고유 ID를 바인딩함으로써,
            - 이 파이프라인을 하루에 수십 번 중복해서 다시 실행하더라도
            - Qdrant 내부에 데이터가 누적되지 않고 덮어써지게(Upsert) 만들어
            - 데이터의 멱등성(Idempotency)을 최종 완수하고
            - 파이프라인 전체가 정상 종료됨


### 2.4 예제 코드

- **`docker-compose.yml`**
    - 도커 컨테이너 환경 설정
        - Apache Airflow는 공식 사이트에서 제공하는 최신 파일을 다운로드해서 이용할 것
        - MinIO, Qdrant의 설정을 추가할 것

        ```yaml
        x-airflow-common:
        &airflow-common
        # In order to add custom dependencies or upgrade provider distributions you can use your extended image.
        # Comment the image line, place your Dockerfile in the directory where you placed the docker-compose.yaml
        # and uncomment the "build" line below, Then run `docker-compose build` to build the images.
        image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.3.0}
        # build: .
        env_file:
            - ${ENV_FILE_PATH:-.env}
        environment:
            &airflow-common-env
            AIRFLOW__CORE__EXECUTOR: CeleryExecutor
            AIRFLOW__CORE__AUTH_MANAGER: airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager
            AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
            AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql+psycopg2://airflow:airflow@postgres/airflow
            AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
            AIRFLOW__CORE__FERNET_KEY: ${FERNET_KEY}
            AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
            AIRFLOW__CORE__LOAD_EXAMPLES: 'true'
            AIRFLOW__CORE__EXECUTION_API_SERVER_URL: 'http://airflow-apiserver:8080/execution/'
            AIRFLOW__API_AUTH__JWT_SECRET: ${AIRFLOW__API_AUTH__JWT_SECRET:-airflow_jwt_secret}
            AIRFLOW__API_AUTH__JWT_ISSUER: ${AIRFLOW__API_AUTH__JWT_ISSUER:-airflow}
            # yamllint disable rule:line-length
            # Use simple http server on scheduler for health checks
            # See https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/logging-monitoring/check-health.html#scheduler-health-check-server
            # yamllint enable rule:line-length
            AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
            # WARNING: Use _PIP_ADDITIONAL_REQUIREMENTS option ONLY for a quick checks
            # for other purpose (development, test and especially production usage) build/extend Airflow image.
            _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:-}
            # The following line can be used to set a custom config file, stored in the local config folder
            AIRFLOW_CONFIG: '/opt/airflow/config/airflow.cfg'
        volumes:
            - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
            - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
            - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
            - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins
        user: "${AIRFLOW_UID:-50000}:0"
        depends_on:
            &airflow-common-depends-on
            redis:
            condition: service_healthy
            postgres:
            condition: service_healthy

        services:
        postgres:
            image: postgres:16
            environment:
            POSTGRES_USER: airflow
            POSTGRES_PASSWORD: airflow
            POSTGRES_DB: airflow
            volumes:
            - postgres-db-volume:/var/lib/postgresql/data
            healthcheck:
            test: ["CMD", "pg_isready", "-U", "airflow"]
            interval: 10s
            retries: 5
            start_period: 5s
            restart: always

        redis:
            # Redis is limited to 7.2-bookworm due to licencing change
            # https://redis.io/blog/redis-adopts-dual-source-available-licensing/
            image: redis:7.2-bookworm
            expose:
            - 6379
            healthcheck:
            test: ["CMD", "redis-cli", "ping"]
            interval: 10s
            timeout: 30s
            retries: 50
            start_period: 30s
            restart: always

        airflow-apiserver:
            <<: *airflow-common
            command: api-server
            ports:
            - "8080:8080"
            healthcheck:
            test: ["CMD", "curl", "--fail", "http://localhost:8080/api/v2/monitor/health"]
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-init:
                condition: service_completed_successfully

        airflow-scheduler:
            <<: *airflow-common
            command: scheduler
            healthcheck:
            test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-init:
                condition: service_completed_successfully

        airflow-dag-processor:
            <<: *airflow-common
            command: dag-processor
            healthcheck:
            test: ["CMD-SHELL", 'airflow jobs check --job-type DagProcessorJob --hostname "$${HOSTNAME}"']
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-init:
                condition: service_completed_successfully

        airflow-worker:
            <<: *airflow-common
            command: celery worker
            healthcheck:
            # yamllint disable rule:line-length
            test: ["CMD-SHELL", 'celery --app airflow.providers.celery.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}" || celery --app airflow.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}"']
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            environment:
            <<: *airflow-common-env
            # Required to handle warm shutdown of the celery workers properly
            # See https://airflow.apache.org/docs/docker-stack/entrypoint.html#signal-propagation
            DUMB_INIT_SETSID: "0"
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-apiserver:
                condition: service_healthy
            airflow-init:
                condition: service_completed_successfully

        airflow-triggerer:
            <<: *airflow-common
            command: triggerer
            healthcheck:
            test: ["CMD-SHELL", 'airflow jobs check --job-type TriggererJob --hostname "$${HOSTNAME}"']
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-init:
                condition: service_completed_successfully

        airflow-init:
            <<: *airflow-common
            entrypoint: /bin/bash
            # yamllint disable rule:line-length
            command:
            - -c
            - |
                if [[ -z "${AIRFLOW_UID}" ]]; then
                echo
                echo -e "\033[1;33mWARNING!!!: AIRFLOW_UID not set!\e[0m"
                echo "If you are on Linux, you SHOULD follow the instructions below to set "
                echo "AIRFLOW_UID environment variable, otherwise files will be owned by root."
                echo "For other operating systems you can get rid of the warning with manually created .env file:"
                echo "    See: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#setting-the-right-airflow-user"
                echo
                export AIRFLOW_UID=$$(id -u)
                fi
                one_meg=1048576
                mem_available=$$(($$(getconf _PHYS_PAGES) * $$(getconf PAGE_SIZE) / one_meg))
                cpus_available=$$(grep -cE 'cpu[0-9]+' /proc/stat)
                disk_available=$$(df / | tail -1 | awk '{print $$4}')
                warning_resources="false"
                if (( mem_available < 4000 )) ; then
                echo
                echo -e "\033[1;33mWARNING!!!: Not enough memory available for Docker.\e[0m"
                echo "At least 4GB of memory required. You have $$(numfmt --to iec $$((mem_available * one_meg)))"
                echo
                warning_resources="true"
                fi
                if (( cpus_available < 2 )); then
                echo
                echo -e "\033[1;33mWARNING!!!: Not enough CPUS available for Docker.\e[0m"
                echo "At least 2 CPUs recommended. You have $${cpus_available}"
                echo
                warning_resources="true"
                fi
                if (( disk_available < one_meg * 10 )); then
                echo
                echo -e "\033[1;33mWARNING!!!: Not enough Disk space available for Docker.\e[0m"
                echo "At least 10 GBs recommended. You have $$(numfmt --to iec $$((disk_available * 1024 )))"
                echo
                warning_resources="true"
                fi
                if [[ $${warning_resources} == "true" ]]; then
                echo
                echo -e "\033[1;33mWARNING!!!: You have not enough resources to run Airflow (see above)!\e[0m"
                echo "Please follow the instructions to increase amount of resources available:"
                echo "   https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#before-you-begin"
                echo
                fi
                echo
                echo "Creating missing opt dirs if missing:"
                echo
                mkdir -v -p /opt/airflow/{logs,dags,plugins,config}
                echo
                echo "Airflow version:"
                /entrypoint airflow version
                echo
                echo "Files in shared volumes:"
                echo
                ls -la /opt/airflow/{logs,dags,plugins,config}
                echo
                echo "Running airflow config list to create default config file if missing."
                echo
                /entrypoint airflow config list >/dev/null
                echo
                echo "Files in shared volumes:"
                echo
                ls -la /opt/airflow/{logs,dags,plugins,config}
                echo
                echo "Change ownership of files in /opt/airflow to ${AIRFLOW_UID:-50000}:0"
                echo
                chown -R "${AIRFLOW_UID:-50000}:0" /opt/airflow/
                echo
                echo "Change ownership of files in shared volumes to ${AIRFLOW_UID:-50000}:0"
                echo
                chown -v -R "${AIRFLOW_UID:-50000}:0" /opt/airflow/{logs,dags,plugins,config}
                echo
                echo "Files in shared volumes:"
                echo
                ls -la /opt/airflow/{logs,dags,plugins,config}

            # yamllint enable rule:line-length
            environment:
            <<: *airflow-common-env
            _AIRFLOW_DB_MIGRATE: 'true'
            _AIRFLOW_WWW_USER_CREATE: 'true'
            _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow}
            _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow}
            _PIP_ADDITIONAL_REQUIREMENTS: ''
            user: "0:0"

        airflow-cli:
            <<: *airflow-common
            profiles:
            - debug
            environment:
            <<: *airflow-common-env
            CONNECTION_CHECK_MAX_COUNT: "0"
            # Workaround for entrypoint issue. See: https://github.com/apache/airflow/issues/16252
            command:
            - bash
            - -c
            - airflow
            depends_on:
            <<: *airflow-common-depends-on

        # You can enable flower by adding "--profile flower" option e.g. docker-compose --profile flower up
        # or by explicitly targeted on the command line e.g. docker-compose up flower.
        # See: https://docs.docker.com/compose/profiles/
        flower:
            <<: *airflow-common
            command: celery flower
            profiles:
            - flower
            ports:
            - "5555:5555"
            healthcheck:
            test: ["CMD", "curl", "--fail", "http://localhost:5555/"]
            interval: 30s
            timeout: 10s
            retries: 5
            start_period: 30s
            restart: always
            depends_on:
            <<: *airflow-common-depends-on
            airflow-init:
                condition: service_completed_successfully

        minio-local:
            image: minio/minio:RELEASE.2024-01-11T07-46-16Z
            container_name: minio-local
            ports:
            - "9000:9000" # 파이썬 SDK가 접속할 API 통신 포트
            - "9001:9001" # 웹 UI 콘솔 어드민 포트
            environment:
            MINIO_ROOT_USER: minioadmin
            MINIO_ROOT_PASSWORD: minioadminpassword
            volumes:
            - minio-data-volume:/data
            command: server /data --console-address ":9001"
            networks:
            - airflow-net
            
        qdrant-local:
            image: qdrant/qdrant:v1.18.0
            container_name: qdrant-local
            ports:
            - "6333:6333" # REST API 및 대시보드 진입 포트
            volumes:
            - qdrant-data-volume:/qdrant/storage
            networks:
            - airflow-net

        networks:
        airflow-net:
            name: local-ai-platform-net
            driver: bridge

        volumes:
        postgres-db-volume:
        minio-data-volume:
        qdrant-data-volume:
        ```

<br>

- **`.env` 설정**

    ```
    AIRFLOW_UID=1000
    FERNET_KEY=*************************************
    _PIP_ADDITIONAL_REQUIREMENTS=minio qdrant-client
    ```

    - MinIO, Qdrant를 인식할 수 있도록 컨테이너 안에 _PIP_ADDITIONAL_REQUIREMENTS 설정을 통해 라이브러리 설치
        - 컨테이너의 MinIO, Qdrant와 파이썬 라이브러리의 버전이 다를 경우에도 오류가 발생할 수 있음
            - 예: 컨테이너의 Qdrant: 1.12.0 / Airflow 컨테이너에서 요청에 의해 설치한 Qdrant 라이브러리: 1.18.0
            - 컨테이너의 버전을 수정하거나 파이브러리 설치 요청에서 버전을 지정할 것<br>

    - 예제 2에서는 Apache Airflow 이미지의 커스텀 빌드 시에 MinIO, Qdrant 등을 포함시키고 `.env`에서는 삭제함<br>
        🡪 성능 및 가동 시간 절감에 훨씬 효율적임<br>

    - FERNET_KEY가 설정되지 않으면 지속적으로 경고 문구가 발생함
        - docker-compose.yml 내부의 컨테이너 개수만큼 발생
            - FERNET_KEY 생성 방법

                ```bash
                python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())
                ```

<br>

- **DAG 작성(advanced_ai_orchestration_pipeline.py)**

    ```python
    #//file: "dags/advanced_ai_orchestration_pipeline.py"

    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from minio import Minio
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import io, os

    # 인프라 토폴로지 엔드포인트 정의
    MINIO_URL = "192.168.0.6:9000"
    MINIO_ACCESS = "minioadmin"
    MINIO_SECRET = "minioadminpassword"
    BUCKET_NAME = "enterprise-knowledge-lake"

    QDRANT_URL = "192.168.0.6:6333"
    COLLECTION_NAME = "factory_manual_vectors"

    # [Task 1] 외부 소스로부터 데이터를 수집하여 데이터 레이크에 저장
    def fn_ingest_to_lake():
        client = Minio(MINIO_URL, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            
        dummy_doc = "TIMESTAMP:2026-07-08 | MANUAL:스마트팩토리 가변 면적 유량계 계측기 장애 조치 매뉴얼. 압력 강하 시 벨브 호스를 점검하십시오."
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name="raw/manual_01.txt",
            data=io.BytesIO(dummy_doc.encode('utf-8')),
            length=len(dummy_doc.encode('utf-8')),
            content_type="text/plain"
        )
        print("[Success] 원시 매뉴얼이 Data Lake(MinIO)에 안착했습니다.")

    # [Task 2 & 3] Apache Spark 스키마를 모방한 텍스트 가공 및 VectorDB 적재
    # (주의: 실제 분산환경에서는 PySpark를 활용해 대량 가공 후 분산 Upsert 처리)
    def fn_spark_processing_and_vector_upsert():
        # 1. Lake에서 데이터 읽기 (Spark의 데이터 소스 로딩 역할 모방)
        minio_client = Minio(MINIO_URL, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
        response = minio_client.get_object(BUCKET_NAME, "raw/manual_01.txt")
        raw_text = response.read().decode('utf-8')
        response.close()
        
        # 2. Spark 비즈니스 로직: 의미론적 가공 및 청킹 (Transformation)
        # 실제 실무에서는 Spark DataFrame의 udf(User Defined Function)를 사용하여 분산 클러스터에서 수행됨
        refined_text = raw_text.replace("TIMESTAMP:2026-07-08 | ", "[확인 완료] ")
        chunks = [refined_text[i:i+50] for i in range(0, len(refined_text), 50)] # 50글자 단위 청킹
        
        # 3. VectorDB 커넥션 수립 및 컬렉션 초기화
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # 컬렉션이 없으면 고차원(예: 384차원) HNSW 인덱스 기반으로 생성
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=5, distance=Distance.COSINE), # 예제를 위해 5차원으로 설정
            )
        
        # 4. 가상 가벼운 임베딩 벡터 생성 및 Upsert (Idempotency 보장)
        # 실무에서는 Ollama나 SentenceTransformer 모델을 워커 내부 혹은 외부 GPU 가속기를 통해 수립
        points = []
        for idx, chunk in enumerate(chunks):
            dummy_embedding = [0.1 * (idx + 1), 0.2, 0.5, 0.1, 0.9] # 5차원 가상 임베딩 벡터
            points.append(PointStruct(
                id=idx + 100, # 고유 ID 지정으로 멱등성(Upsert) 확보
                vector=dummy_embedding,
                payload={"page_content": chunk, "source": "minio://raw/manual_01.txt"}
            ))
            
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[Success] Spark 전처리 완료된 {len(chunks)}개의 청크가 Qdrant HNSW 인덱스에 동기화되었습니다.")

    # ============================================================
    # Airflow DAG 오케스트레이션 핵심 설정
    # ============================================================
    default_args = {
        'owner': 'ai_platform_eng',
        'depends_on_past': False,
        'start_date': datetime(2026, 7, 8),
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }

    with DAG(
        dag_id='advanced_bigdata_ai_pipeline_v1',
        default_args=default_args,
        description='수집->Lake->Spark 전처리->VectorDB 파이프라인 자동화',
        schedule='@daily',
        catchup=False,
        tags=['spark', 'minio', 'qdrant', 'rag']
    ) as dag:

        task_ingest = PythonOperator(
            task_id='ingest_to_data_lake',
            python_callable=fn_ingest_to_lake,
        )

        task_spark_and_vector = PythonOperator(
            task_id='spark_transform_and_vector_upsert',
            python_callable=fn_spark_processing_and_vector_upsert,
        )

        # 파이프라인 상하 관계 바인딩
        task_ingest >> task_spark_and_vector
    ```

> - **실무 모니터링 및 아키텍처적 핵심 팁**
>   - **Spark 메모리 튜닝 (`OOM` 방지):**
>       - 전처리 중 `SparkDriver` 노드로 너무 많은 대용량 텍스트 집계 데이터를 한 번에 가져오는 `collect()` 연산은 절대 피해야 함
>       - 대신 가공된 데이터를 곧바로 MinIO/S3에 분산 파일(Parquet, 데이터 레이크 포맷) 형태로 write하도록 파이프라인을 설계해야 함
>   - **VectorDB 백필(Backfill) 과부하 통제:**
>       - 과거 대용량 데이터를 한 번에 재처리할 때, VectorDB에 수천만 건의 벡터가 한꺼번에 쏟아지면 실시간 HNSW 그래핑 연산 때문에 DB CPU가 마비될 수 있음
>       - 이 경우 Airflow의 `max_active_runs_per_dag=1` 설정을 통해 배치 실행 단위를 강제 조율하여 시스템 안전망을 가동해야 함
{: .summary-quote}


## 3. 종합 실습 예제 코드 2

### 3.1 아키텍처 실무 비즈니스 시나리오

> **"스마트팩토리 가변 면적 유량계(Variable Area Flowmeter) 센서 스트리밍 데이터의 하이브리드 레이크하우스 구축 및 RAG 지식 베이스 동기화"**
{: .common-quote}

1. **실시간 수집 (Kafka):**
    - 공장 센서 및 설비 제어기에서 발생하는 실시간 비정형 로그 및 매뉴얼 텍스트가
    - Kafka 토픽으로 인덱싱

2. **데이터 레이크 보존 (MinIO):**
    - 수집된 원시(Raw) 로그는 데이터 유실 방지 및 멱등성 확보를 위해
    - MinIO 오브젝트 스토리지의 날짜별 파티션 영역(`raw/{ { ds_nodash }}/`)에 영구 보존

3. **분산 전처리 및 임베딩 가공 (PySpark):**
    - Spark 세션을 컨테이너 내부에서 동적 구동하여, 
    - 비정형 텍스트의 노이즈를 제거하고 
    - 의미론적 문맥 보전을 위한 **청킹(Chunking)** 분산 연산을 수행

4. **지식 베이스 동기화 (Qdrant):**
    - 가공된 고차원 벡터 데이터를 고속 밀집 검색을 위해
    - Qdrant 벡터 스토어의 HNSW 그래프 인덱스에 중복 없이 **Upsert**하여
    - 실시간 RAG 검색 엔진을 최신화함


### 3.2 컨테이너 환경 작성

- **"Kafka ➔ MinIO ➔ Apache Spark ➔ Qdrant"** 풀스택 데이터 플랫폼 인프라 스펙
- 컴포넌트 간 격리 장벽 없이 유기적으로 연동되도록 동일한 브릿지 네트워크(`bigdata-network`)로 연결

<br>

- **`Dockerfile.airflow`**
    - Airflow 공식 docker-compose.yml으로 설치 시, Java/JVM에 관련된 부분이 버전 차이 등 몇 가지 문제로 인해 제대로 설치되어 있지 않음
        - 실습 시나리오에 필요한 각 컨테이너를 연동, 실행하려면 OpenJDK 21 이상 버전이 요구됨
        - 특히 OpenJDK 21 내부의 GLIBC 2.38 이상을 요구함
    - 따라서 Airflow 3.3.0을 기반으로 커스텀 빌드를 수행하여야 함
        - 향후에 사용될 각 모듈 및 라이브러리 패키지들도 커스텀 빌드에 미리 넣어놓으면 컨테이너의 Up/Down이 빨라짐

        ```Dockerfile
        # 1. 아파치 에어플로우 공식 베이스 이미지 지정
        FROM apache/airflow:3.3.0

        # 2. 시스템 패키지 설치를 위해 root 권한으로 잠시 스위칭
        USER root

        # 3. 컨테이너 내부 OS 환경에 100% 맞는 순정 OpenJDK 21 설치 (Glibc 충돌 원천 차단)
        RUN curl -Lf https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.2%2B13/OpenJDK21U-jdk_x64_linux_hotspot_21.0.2_13.tar.gz -o /tmp/openjdk.tar.gz && \
            mkdir -p /usr/lib/jvm/java-21-openjdk-amd64 && \
            tar -xzf /tmp/openjdk.tar.gz -C /usr/lib/jvm/java-21-openjdk-amd64 --strip-components=1 && \
            rm -rf /tmp/openjdk.tar.gz

        # 4. 에어플로우 실행 권한(소유자)으로 다시 복귀
        USER airflow

        # 5. 기존에 _PIP_ADDITIONAL_REQUIREMENTS로 실시간 다운로드받던 빅데이터 패키지들을 이미지에 미리 빌드
        RUN pip install --no-cache-dir pyspark==4.1.2 minio qdrant-client kafka-python
        ```

<br>

- **`docker-compose.yml`**
    - **Airflow 공통 부분 수정**
        - 수정 지점: build 부분, environment의 JAVA_HOME 부분

        ```yaml
        x-airflow-common:
            &airflow-common
            image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.3.0}
            build:
                context: .
                dockerfile: Dockerfile.airflow
            env_file:
                - ${ENV_FILE_PATH:-.env}
            environment:
                &airflow-common-env
                AIRFLOW__CORE__EXECUTOR: CeleryExecutor
                AIRFLOW__CORE__AUTH_MANAGER: airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager
                AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
                AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql+psycopg2://airflow:airflow@postgres/airflow
                AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
                AIRFLOW__CORE__FERNET_KEY: ${FERNET_KEY}
                AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
                AIRFLOW__CORE__LOAD_EXAMPLES: 'true'
                AIRFLOW__CORE__EXECUTION_API_SERVER_URL: 'http://airflow-apiserver:8080/execution/'
                AIRFLOW__API_AUTH__JWT_SECRET: ${AIRFLOW__API_AUTH__JWT_SECRET:-airflow_jwt_secret}
                AIRFLOW__API_AUTH__JWT_ISSUER: ${AIRFLOW__API_AUTH__JWT_ISSUER:-airflow}
                AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
                _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:-}
                AIRFLOW_CONFIG: '/opt/airflow/config/airflow.cfg'
                JAVA_HOME: "/usr/lib/jvm/java-21-openjdk-amd64"
        ```

    <br>

    - **Kafka 컨테이너 추가**
        - 기존에는 Kafka에 대한 접근 도메인을 외부, 내부로 나누어 처리하였으나,
        - 현재는 외부 접근도 결국 Airflow가 처리하므로 내부와 동일하게 도메인을 설정함
        - 그러나 향후 정식 서비스로의 확장 시, 보안 등을 고려하여 포트의 변형은 그대로 유지함
            - (현재 시점에서는 구분하는 의미가 없음)

        ```yaml
        kafka-1:
            image: apache/kafka:4.3.1
            container_name: kafka-1
            ports:
                - "9092:9092"
            environment:
                KAFKA_NODE_ID: 1
                KAFKA_PROCESS_ROLES: broker,controller
                KAFKA_LISTENERS: INTERNAL://0.0.0.0:19092, EXTERNAL://0.0.0.0:9092, CONTROLLER://0.0.0.0:9093
                KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-1:19092,EXTERNAL://kafka-1:9092
                KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
                KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
                KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
                KAFKA_LOG_DIRS: /var/lib/kafka/data
            volumes:
                - kafka-1-data-volume:/var/lib/kafka/data

        kafka-2:
            image: apache/kafka:4.3.1
            container_name: kafka-2
            ports:
                - "9094:9092"
            environment:
                KAFKA_NODE_ID: 2
                KAFKA_PROCESS_ROLES: broker,controller
                KAFKA_LISTENERS: INTERNAL://0.0.0.0:19092, EXTERNAL://0.0.0.0:9092, CONTROLLER://0.0.0.0:9093
                KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-2:19092,EXTERNAL://kafka-2:9092
                KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
                KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
                KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
                KAFKA_LOG_DIRS: /var/lib/kafka/data
            volumes:
                - kafka-2-data-volume:/var/lib/kafka/data

        kafka-3:
            image: apache/kafka:4.3.1
            container_name: kafka-3
            ports:
                - "9095:9092"
            environment:
                KAFKA_NODE_ID: 3
                KAFKA_PROCESS_ROLES: broker,controller
                KAFKA_LISTENERS: INTERNAL://0.0.0.0:19092, EXTERNAL://0.0.0.0:9092, CONTROLLER://0.0.0.0:9093
                KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-3:19092,EXTERNAL://kafka-3:9092
                KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
                KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
                KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
                KAFKA_LOG_DIRS: /var/lib/kafka/data
            volumes:
                - kafka-3-data-volume:/var/lib/kafka/data
        ```
        
    <br>

    - **MinIO 컨테이너 추가**

        ```yaml
        minio:
            image: minio/minio:RELEASE.2025-04-22T22-12-26Z
            container_name: minio
            ports:
                - "9000:9000" # 파이썬 SDK가 접속할 API 통신 포트
                - "9001:9001" # 웹 UI 콘솔 어드민 포트
            environment:
                MINIO_ROOT_USER: minioadmin
                MINIO_ROOT_PASSWORD: minioadmin
            volumes:
                - minio-data-volume:/data
            command: server /data --console-address ":9001"
        ```

    <br>

    - **Qdrant 컨테이너 추가**

        ```yaml
        qdrant:
            image: qdrant/qdrant:v1.18.0
            container_name: qdrant
            ports:
                - "6333:6333"     # REST API용 포트 (에어플로우 워커 통신 채널)
                - "6334:6334"     # gRPC용 포트
            volumes:
                - qdrant-data-volume:/qdrant/storage
        ```

    <br>

    - **각 컨테이너에서 사용하는 볼륨 추가**

        ```yaml
        volumes:
            kafka-1-data-volume:
            kafka-2-data-volume:
            kafka-3-data-volume:
            postgres-db-volume:
            minio-data-volume:
            qdrant-data-volume:
        ```


### 3.3 사전 환경 구축 및 토폴로지 설정

- 외부에서 구동 중인 대형 오픈소스 인프라 컴포넌트들을 컨테이너 내부의 Airflow가 인식하고,
- Python 3.13 환경에서 관련 의존성 크래시 없이 패키지를 로드할 수 있도록 `.env`를 셋업

```bash
# 1. 기존 구버전 인프라 컨테이너 완전 삭제 (볼륨 보존 선택 가능하나 클린 아키텍처를 위해 다운 추천)
docker compose down -v

# 2. 로컬 가상 환경 변수 점검 (.env에 명시된 최신 패키지 확인)
# _PIP_ADDITIONAL_REQUIREMENTS=minio qdrant-client pyspark kafka-python trino
echo -e "AIRFLOW_UID=$(id -u)" > .env

# 3. 에어플로우 최신 이미지(3.3.0) 기반 메타 스키마 마이그레이션 및 초기 계정 설정
docker compose up airflow-init

# 4. KRaft 및 지정 MinIO 버전이 통합된 전체 컴포넌트 실시간 백그라운드 구동
docker compose up -d

# 5. 정상 기동 상태 검증
docker compose ps
```

<br>

- **포트 맵핑 최종 주소:**
    - **Airflow 인프라 포탈:** `http://localhost:8080` (계정: `airflow` / `airflow`)
    - **MinIO Console:** `http://localhost:9001` (계정: `minioadmin` / `minioadmin`)
    - **Qdrant Dashboard:** `http://localhost:6333/dashboard`


### 3.5 DAG 구성

- **파이프라인 의존성 아키텍처 및 흐름도**
    - 각 오픈소스 컴포넌트의 결함 격리(Fault Isolation)를 시각화한 위상 정렬 의존성 구조

<br>

- **데이터 제어 흐름 구조 (Data Control Flow)**

    <div class="insert-image" style="text-align: center;">
        <img src="/materials/devtools/images/S13-06-03-01_01-008.png" style="width: 90%;">
    </div>


    - **단계별 매커니즘**
        1. **`task_kafka_kr_ingest` (수집 계층):**
            - KRaft 단독 노드로 기동 중인 Kafka 토픽으로
            - 가변 면적 유량계 장애 로그 이벤트를 발행하고 수집 검증

        2. **병렬 처리 계층 (Parallel Operations):**
            - 수집 완료 신호를 받으면,
            - 데이터 레이크 백업(MinIO), 분산 메모리 전처리(Spark)가
            - **호스트 자원을 효율적으로 나누어 쓰며 동시에 병렬로 기동**

        3. **`task_vector_upsert_qdrant` (최종 적재 계층):**
            - 앞선 세 갈래의 데이터 엔지니어링 파이프라인이 **모두 무사히 성공(`Success`) 상태로 마킹되어야만**
            - 최종 RAG 검색을 위한 Qdrant 벡터스토어 업서트를 단행

<br>

- **DAG 소스 코드**

    - 코드 준수 규약
        - PEP 8 표준(with 문 하위 4칸 들여쓰기),
        - 최신 Airflow 매개변수(`schedule='@daily'`),
        - KRaft 메시지 큐 통신 인터페이스

    - `~/workspace/airflow/dags/advanced_bigdata_ai_pipeline_v5.py` 경로로 생성

        ```python
        #//file: "dags/advanced_bigdata_ai_pipeline_v5.py"

        import io
        import json
        from datetime import datetime, timedelta
        from airflow import DAG
        from airflow.providers.standard.operators.python import PythonOperator

        from minio import Minio
        from kafka import KafkaProducer
        from pyspark.sql import SparkSession
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct


        KAFKA_BROKERS = [
            "kafka-1:9092",
            "kafka-2:9092",
            "kafka-3:9092"
        ]

        MINIO_URL = "minio:9000"
        MINIO_ACCESS = "minioadmin"
        MINIO_SECRET = "minioadmin"
        RAW_LAKE_BUCKET = "factory-raw-stream"
        COLLECTION_NAME = "factory_hybrid_knowledge_base"

        QDRANT_HOST = "qdrant"
        QDRANT_PORT = "6333"

        def fn_kafka_kr_stream_ingest(**context):
            sample_log = {
                "equipment": "Variable Area Flowmeter (가변 면적 유량계)",
                "status": "Warning",
                "message": "유량 변동에 따른 튜브 내 플로트(Float) 진동 발생. 가변 면적 측정 정밀도 저하 우려. 기밀 패킹 오링(O-ring) 교체 요망."
            }

            try:
                # 멀티 노드 쿼럼에 메시지를 안전하게 분산 발행하기 위한 고가용성 프로듀서 빌드
                producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BROKERS, # 3개 브로커 풀 주입
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    max_block_ms=5000,               # 쿼럼 합의 대기 마진 소폭 확장
                    acks='all',                      # 리더와 팔로워 브로커 모두 적재 성공 팩트 확인 규칙
                    retries=3                        # 일시적 노드 순번 교체 시 자체 재시도 안전장치
                )

                # 토픽 발행 시점에 파티션 분산 적재가 가능하도록 유량계 메시지 투하
                producer.send('factory-sensor-topic', value=sample_log)
                producer.flush()

                print("\n" + "="*80)
                print(f"[Apache Kafka 4.3.1] 3개 노드 멀티 클러스터 인프라망({KAFKA_BROKERS})에 스트리밍 이벤트를 유실 없이 성공적으로 발행했습니다.")
                print("\n" + "="*80)

            except Exception as e:
                print(f"[네트워크 우회 가이드] 멀티 카프카 쿼럼 연결 차단으로 내장 컨텍스트 스트림 메모리로 대체 구동합니다. 사유: {e}")

            context["ti"].xcom_push(key="raw_stream_data", value=sample_log)


        def fn_minio_raw_backup(**context):
            ds_nodash = context['ds_nodash']
            raw_data = context["ti"].xcom_pull(task_ids="task_kafka_kr_ingest", key='raw_stream_data')

            client = Minio(MINIO_URL, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
            if not client.bucket_exists(RAW_LAKE_BUCKET):
                client.make_bucket(RAW_LAKE_BUCKET)
            
            object_path = f"raw/{ds_nodash}/stream_log.json"
            client.put_object(
                bucket_name=RAW_LAKE_BUCKET,
                object_name=object_path,
                data=io.BytesIO(json.dumps(raw_data, ensure_ascii=False).encode("utf-8")),
                length=len(json.dumps(raw_data, ensure_ascii=False).encode("utf-8")),
                content_type="application/json"
            )

            print("\n" + "="*80)
            print(f"[MinIO 백업 성공] 지정 버전 스토리지 적재 완료 경로: {object_path}")
            print("\n" + "="*80)


        def fn_spark_transform_processing(**context):
            raw_data = context["ti"].xcom_pull(task_ids="task_kafka_kr_ingest", key="raw_stream_data")

            spark = SparkSession.builder \
                    .appName("KRaftEnvironmentSparkProcessor") \
                    .master("local[*]") \
                    .getOrCreate()
            
            target_corpus = f"[계측기 실시간 장애 가이드] 대상 설비: {raw_data['equipment']} | 현장 상태: {raw_data['status']} | 조치 지침: {raw_data['message']}"

            df = spark.createDataFrame([(1, target_corpus)], ["id", "text"])
            refined_text = df.select("text").collect()[0][0]
            spark.stop()

            print("\n" + "="*80)
            print(f"[PySpark 분산 가공 엔진 성공] 가변 면적 유량계 스트리밍 텍스트 정제 완수!")
            print(f" ➔ 정제된 청크: {refined_text}")
            print("="*80 + "\n")    

            context["ti"].xcom_push(key="refined_chunk", value=refined_text)

        def fn_vector_upsert_qdrant(**context):
            ds_nodash = context['ds_nodash']
            processed_chunk = context["ti"].xcom_pull(task_ids="task_spark_transform", key="refined_chunk")

            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=3, distance=Distance.COSINE)
                )

            point_id = int(f"{ds_nodash}99")
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=[0.18, 0.65, 0.49],
                        payload={
                            "page_content": processed_chunk,
                            "log_date": ds_nodash,
                            "lineage": "kafka 🡪 minio 🡪 pyspark 🡪 qdrant"
                        }
                    )
                ]
            )

            print("\n" + "="*80)
            print(f"[Qdrant] 최신 RAG 지식 베이스 동적 데이터 Upsert 완수 완료 (Point ID: {point_id})")
            print("\n" + "="*80)


        default_args = {
            "owner": "seokhwan",
            "depends_on_past": False,
            "start_date": datetime(2026, 7, 11),
            "retries": 1,
            "retry_delay": timedelta(minutes=3)
        }

        with DAG(
            dag_id="advanced_bigdata_ai_pipeline_v5",
            default_args=default_args,
            description="최신 아파치 카프카 4.3.1 멀티 브로커 및 지정 MinIO 기반 엔터프라이즈 자동화 파이프라인",
            schedule="@daily",
            catchup=False,
            tags=["kafka", "minio", "spark", "qdrant"]
        ) as dag:
            
            task_kafka_kr_ingest = PythonOperator(
                task_id="task_kafka_kr_ingest", 
                python_callable=fn_kafka_kr_stream_ingest
            )

            task_minio_raw_backup = PythonOperator(
                task_id="task_minio_raw_backup",
                python_callable=fn_minio_raw_backup
            )

            task_spark_transform = PythonOperator(
                task_id="task_spark_transform",
                python_callable=fn_spark_transform_processing
            )

            task_vector_upsert_qdrant = PythonOperator(
                task_id="task_vector_upsert_qdrant",
                python_callable=fn_vector_upsert_qdrant
            )

            task_kafka_kr_ingest >> [task_minio_raw_backup, task_spark_transform] >> task_vector_upsert_qdrant
        ```


### 3.5 결과 확인

- `http://localhost:8080`에 접속 후 DAG 메뉴에서 `enterprise_bigdata_ai_stream_pipeline_v5` DAG가 나타나는지 검색창에서 조회

- **결과 확인**
    1. **확인 전 상황**
        - 기존에 실패했던 설정 이후로 성공한 설정을 확인할 수 있음
        - 초반의 성공 이후, 설정이 안정화된 뒤에는 처리 시간도 크게 감소하였음
        
        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-002.png" style="width: 100%;">
        </div>

    2. **Trigger 실행 결과**
        - 전 과정이 무리없이 성공함
        - 처리 시간: 시작부터 끝까지 약 8초에 완료되었음 
            - 설정 수정에 따른 안정화 후 소요 시간이 급감했음을 확인

        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-003.png" style="width: 100%;">
        </div>

    3. **task_kafka_kr_ingest의 로그**
        - 3개의 Kafka 볼륨에 제대로 복제, 저장되었음

        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-004.png" style="width: 100%;">
        </div>

    4. **task_spark_transform의 로그**
        - 기존에 실패했던 설정 이후로 성공한 설정을 확인할 수 있음
        - 중간에 Error 표시가 보이지만 내용을 읽어보면 Error가 아님을 알 수 있음
            - ERROR 표시가 출력된 것은 스파크의 표준 자바 에러 출력(Stderr) 스트림이 유입되어 발생한 에어플로우 고유의 로깅 특징
                - 스파크 엔진은 구동될 때 엔진의 시스템 경고나 환경 설정 안내를 Stdout이 아닌 Stderr 스트림으로 내보내도록 코드가 작성되어 있음
                - 에어플로우가 출력 스트림이 Stderr로 전달된 것만 보고 기계적으로 ERROR를 붙여버린 것임
            - 진짜 내부 예외(Exception)가 발생했다면 스파크 특유의 거대한 StackTrace 자바 에러 문단이 출력되어야 함

        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-005.png" style="width: 100%;">
        </div>

    5. **task_minio_raw_backup 로그**
        - 데이터의 적재가 정상적으로 완료되었음

        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-006.png" style="width: 100%;">
        </div>

    6. **task_vector_upsert_qdrant의 로그**
        - 데이터의 Upsert가 정상적으로 완료되었음

        <div class="insert-image" style="border: 1px solid lightgray;">
            <img src="/materials/devtools/images/S13-06-03-01_01-007.png" style="width: 100%;">
        </div>
