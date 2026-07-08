---
layout: page
title:  "수집->Lake->Spark->VectorDB 흐름 자동화"
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

    ```text
    [ Web UI / API ] ── 조회/조작 ──┐
                                   ▼
    [ Scheduler ] ─── 상태 감시 ───➔ [ Metadata DB ] 
        │                               ▲
    태스크 발행                          │
        │                          상태 업데이트
        ▼                               │
    [ Executor / Queue ] ─── 할당 ────➔ [ Distributed Workers ]
    ```

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


## 2. 종합 실습 예제 코드

- "수집 🡲 Data Lake 🡲 Apache Spark 🡲 VectorDB" 구조를 단일 파일로 구현한 Airflow DAG
    - 실무에서는 Spark 전처리 로직을 별도의 `*.py` 파일로 분리하여 `SparkSubmitOperator`로 호출
    - 실습예제에서는 가독성을 위해 PySpark 전처리 및 Qdrant 적재를 통합 구현함

```python
# file: dags/advanced_ai_orchestration_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os

# 인프라 토폴로지 엔드포인트 정의
MINIO_URL = "localhost:9000"
MINIO_ACCESS = "minioadmin"
MINIO_SECRET = "minioadminpassword"
BUCKET_NAME = "enterprise-knowledge-lake"

QDRANT_URL = "http://localhost:6333"
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
    schedule_interval='@daily',
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