---
layout: page
title:  "데이터 파이프라인 구축 및 최종 RAG 챗봇 개발"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S10-99-05-01_01-DataPipelineRagChatbot
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - **실습 주제**
>   - **모던 데이터 엔지니어링 및 실시간 AI 파이프라인 기술 스택**들의 유기적 결합을 통한 **"실시간 RAG 챗봇 시스템"**
>       - 모든 기술을 억지로 엮어 스파게티 코드가 되는 것을 방지하기 위해,
>       - 현업에서 **실시간 대용량 로그 분석 및 RAG를 구축할 때 사용하는 가장 표준적인 아키텍처**로 파이프라인을 설계
{: .common-quote}


## 1. 시스템 아키텍처 및 데이터 흐름

- 이 시스템은 실시간 데이터 수집부터 정제, 로드, 벡터 임베딩, 그리고 최종 챗봇 대시보드 조회를 연동, 구현하는
- **실시간 온프레미스 AI 관제 파이프라인**을 모사함

### 1.1 데이터 파이프라인 프로세스

1. **실시간 데이터 생산 및 수집 (Kafka 🡪 Spark Streaming)**
    1. **Kafka**:
        - 공장 센서 로그나 실시간 고객 문의 데이터(JSON)를 스트리밍 수집
    2. **Spark Structured Streaming**:
        - Kafka 토픽을 실시간으로 구독(Subscribe)
        - 데이터를 가공 및 정제

2. **분석용 데이터 스토리지 적재 (DuckDB)**
    1. Spark: 가공한 구조화 데이터를 가볍고 빠른 내장형 분석용 컬럼 DB인 **DuckDB**에 적재
    2. 로컬 실습 환경의 자원 제약과 속도를 감안하여 DuckDB를 가상 레이크하우스 저장소로 활용
        - 실무에서는 데이터 레이크인 Iceberg/Trino를 사용할 수 있음

3. **벡터화 및 캐싱 (Ollama Embeddings 🡪 Qdrant 🡪 Redis)**
    1. **LangChain**을 통해 정제된 데이터를 읽어 텍스트 청킹 수행
    2. 로컬 **Ollama(Gemma 4)** 모델로 텍스트를 임베딩하여 로컬 벡터 DB인 **Qdrant**에 적재
    3. **Redis**를 시맨틱 캐시(Semantic Cache) 및 세션 저장소로 활용
        - 목적: 실시간 유저 질문에 대한 빠른 응답과 동일 질문에 대한 LLM 비용/자원 절약

4. **대화형 서비스 (Streamlit)**
    - 사용자는 **Streamlit** UI를 통해 챗봇과 대화
    - 입력된 질문은 Redis 캐시를 거쳐 Qdrant 검색(RAG) 후
    - 로컬 Gemma 4 모델의 실시간 스트리밍 답변(`st.write_stream`)으로 사용자에게 반환됨



## 🐳 2. 인프라 구축 (`docker-compose.yml`)

- 모든 외부 솔루션들을 한 번에 실행하기 위한 Docker Compose 설정
- 로컬 하드웨어(VRAM/RAM) 부하를 최소화하면서 상호 로컬 통신이 가능하도록 네트워크를 통합함

### 2.1 Ollama 환경 구축

- 로컬 LLM 서비스(Ollama)를 데이터 파이프라인 컨테이너 네트워크에 묶어두지 않고 별도 분리 구동

- **기술적 이유**
    - **하드웨어 가속기(GPU) 가속의 제약 해소:**
        - Docker 환경 내부에서 NVIDIA GPU 가속(GPU Passthrough)을 정밀하게 제어하려면
            - 호스트 시스템에 nvidia-container-toolkit 설치 필요
            - 복잡한 컨테이너 런타임 수동 수정 작업이 필수
        - Ollama를 호스트 OS 수준에 단독으로 설치해 구동하면
            - 무거운 커널 가상화 단계를 건너뛰어 VRAM 관리 효율성이 극대화
            - 하드웨어 성능을 온전히 활용할 수 있음

    - **전역 공용 API 게이트웨이 역할 지원:**
        - Ollama가 제공하는 Gemma 4 모델은 본 실습 파이프라인 대시보드뿐만 아니라
            - 개발자 개인의 VSCode IDE 플러그인(예: Continue Extension),
            - 가상 비서 웹 콘솔,
            - 터미널 보조 도구(CLI),
            - 사내 타 서비스 API 등
        - 다양한 경로에서 실시간 호출이 가능한 독립 서비스 엔드포인트여야 함
        - 이를 본 프로젝트 컨테이너들과 밀결합(Tight Coupling)해 놓으면,
            - 파이프라인을 중단하거나 스택을 리셋할 때
            - 호스트 전체의 LLM 추론 엔진까지 꺼져버리는 치명적인 인프라 종속이 발생

    - **인프라 패치 및 가중치 업데이트 주기 격리:**
        - 수 기가바이트에서 수십 기가바이트에 달하는 LLM 가중치 디렉토리(~/.ollama)의 라이프사이클과,
        - 몇 메가바이트 크기의 경량화 컨테이너(Kafka, Redis, Qdrant) 빌드 주기는
        - 완전히 달라야 관리가 용이함

- **`docker-compose.yaml`**

```yaml
services:
    ollama:
        image: ollama/ollama:latest
        container_name: ollama-container
        restart: unless-stopped
        ports:
            - "11434:11434"
        volumes:
            - /home/[USERID]/ollama/ollama_data:/root/.ollama

        deploy:
            # resources의 내용은 각자의 시스템 사양에 맞추어 조정할 것
            resources:
                limits:
                    # [CPU 세팅] 이전 답변의 밸런스형 세팅 유지 (32개 스레드 중 20개 할당)
                    # P코어 성능을 다 쓰면서도 호스트 OS의 멀티태스킹 쾌적성 보장
                    cpus: '20.0'

                    # [Memory 세팅] 전체 32GB 중 16GB ~ 20GB 할당 추천
                    # 모델이 VRAM에 올라가지만, Docker 컨테이너의 안전 버퍼용으로 16g~20g가 가장 안전함
                    memory: 20g

                reservations:
                    devices:
                        - driver: nvidia
                        count: all
                        capabilities: [gpu]

        oom_score_adj: -500
```

- **인프라 기동 후 필수 사전 작업**
    - 컨테이너를 띄운 후, 로컬 터미널에서 임베딩과 추론에 사용할 `gemma4` 모델을 다운로드해 두어야 함
    - 또한 데이터의 임베딩 수행을 위해 `nomic-embed-text` 모델을 다운로드해 두어야 함

    ```bash
    docker exec -it ollama-container ollama pull gemma4
    docker exec -it ollama-container ollama pull nomic-embed-text
    ```


### 2.2 Apache Kafka - Redis - Qdrant 조합 환경 구축

- **세 가지 오픈소스를 핵심 파이프라인으로 구성한 이유**
    - **실시간성(Real-time), 고가용성(High Availability), 그리고 대규모 데이터 처리**를 동시에 달성하기 위한
    - 가장 표준적이면서도 강력한 아키텍처 조합

- **컴포넌트별 명확한 역할 정의 (R&R)**
    - 각 기술은 본연의 전문 영역에서 최상의 성능을 낼 수 있도록 분리되어 시너지를 발휘

    <div class="insert-image" style="text-align: left;">
        <img src="/materials/project/images/S10-99-05-01_01-001_SystemArchitecture.png" style="width: 80%;">
    </div>

    - **Apache Kafka (실시간 메시징 & 데이터 파이프라인):**
        - 초당 수만 건 이상 쏟아지는 주식 호가, 체결 데이터, 외부 뉴스 피드를
            - 백프레셔(Backpressure) 없이 안정적으로 수집(Ingestion)하는 유연한 척추 역할 담당
        - 높은 유연성(Decoupling)을 제공하여,
            - 수집된 데이터를 여러 컨슈머(전처리기, 로깅, 분석기 등)가 서로 간섭 없이 각자의 속도로 가져갈 수 있게 함

    - **Redis (초고속 인메모리 상태 관리 & 공유 버퍼):**
        - 가장 최근의 현재가, 호가창, 매매 알고리즘의 실시간 상태값처럼
            - '극도의 저지연(Low latency, 1ms 미만)'이 필요한 실시간 데이터를 임시 저장하고 캐싱
        - 여러 AI 에이전트와 매매 프로세스가 동시다발적으로 현재 시장 상태를 참조할 수 있도록
            - 공유 메모리 허브 역할을 수행

    - **Qdrant (실시간 벡터 유사도 검색 DB):**
        - 과거의 주가 패턴, 재무 분석 보고서 텍스트, 실시간 뉴스 본문 등을 
            -AI가 이해할 수 있는 벡터(Embedding) 형태로 저장
        - 현재 발생한 시장 상황이나 패턴과 가장 유사한 과거 사례를
            - 밀리초(ms) 단위로 고속 검색(ANN, Approximate Nearest Neighbor)해내는
            - 실시간 지식 기반(Knowledge Base) 역할

- **왜 이 '조합'인가? (시너지 및 아키텍처적 당위성)**

    - **완전한 비동기 및 마이크로서비스 아키텍처(MSA) 구현**
        - 데이터를 수집하는 주체(Kafka)와 이를 빠르게 정제하고 캐싱하는 주체(Redis), 그리고 의미론적 분석을 수행하는 주체(Qdrant)가 완전히 분리됨
        - 특정 컴포넌트에 병목이 생기거나 장애가 발생하더라도 전체 시스템이 멈추지 않고 버퍼링(Kafka)을 통해 자연스럽게 회복(Fault Tolerance)됨

    - **데이터의 생명 주기(Lifecycle)에 따른 계층화**
        - **Kafka (흐르는 데이터 - Stream):** 순서 보장이 필요한 연속적인 이벤트 로그 저장
        - **Redis (가장 최신의 데이터 - Hot Data):** 극도로 빠른 읽기/쓰기가 필요한 메모리 영역
        - **Qdrant (누적된 컨텍스트 데이터 - Cold to Warm Data):** 의미가 함축된 벡터 데이터의 다차원 인덱싱 및 장기 기억 공간

    - **고성능 분산 처리 최적화**
        - 세 기술 모두 분산 클러스터링을 네이티브로 지원하므로,
        - 데이터 처리량이 급증하더라도 서버 스케일아웃(Scale-out)을 통해 손쉽게 성능을 확장할 수 있음

<br>

> - **요약**
>   - Kafka로 끊김 없이 데이터를 빨아들이고, Redis로 1ms 미만의 실시간 계산과 호가를 처리하며, Qdrant로 AI 에이전트가 최적의 결정을 내릴 수 있도록 지식 패턴을 매칭하는 구조
>   - 초당 수많은 이벤트를 실시간으로 처리해야 하는 AI 기반 에이전트 시스템이나 실시간 분석 백엔드 환경에서, 지연(Latency)과 데이터 유실을 동시에 완벽히 방어할 수 있는 가장 신뢰할 수 있는 조합
{:. summary-quote}

- **`docker-compose.yaml`**

```yaml
services:
  # 1. Apache Kafka 4.3.1 (KRaft 모드 단일 브로커 설정)
  kafka:
    image: apache/kafka:4.3.1
    container_name: kafka-kraft
    ports:
      - "9092:9092"
    environment:
      # KRaft 모드 활성화를 위한 기본 ID 설정
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: 'broker,controller'
      KAFKA_CONTROLLER_QUORUM_VOTERS: '1@kafka:19093'
      
      # 🚀 핵심 수정: 어떤 리스너가 어떤 프로토콜을 사용하는지 맵을 명시 (오류 방지 필수)
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT'
      
      # 🚀 핵심 수정: 포트 바인딩 구조 정돈
      # - 19092: 내부 컨테이너 통신용 (플러그인, 컨슈머용)
      # - 19093: KRaft 컨트롤러 합의용
      # - 9092: 호스트 PC(localhost) 접근용
      KAFKA_LISTENERS: 'PLAINTEXT://:19092,CONTROLLER://:19093,PLAINTEXT_HOST://:9092'
      KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:19092,PLAINTEXT_HOST://localhost:9092'
      
      KAFKA_CONTROLLER_LISTENER_NAMES: 'CONTROLLER'
      KAFKA_INTER_BROKER_LISTENER_NAME: 'PLAINTEXT'
      
      # 데이터 영속성 설정
      KAFKA_LOG_DIRS: '/var/lib/kafka/data'
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      
      # KRaft 클러스터 인증 고유 ID
      CLUSTER_ID: 'MkU3OEVGODktQ0RFNy00M'
    volumes:
      - kafka_data:/var/lib/kafka/data

  # 2. Redis 7.2-bookworm (인메모리 세션 및 캐시)
  redis:
    image: redis:7.2-bookworm
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # 3. Qdrant v1.18.0 (최신 고성능 로컬 벡터 DB)
  qdrant:
    image: qdrant/qdrant:v1.18.0
    container_name: qdrant-vector
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  kafka_data:
  redis_data:
  qdrant_data:
```


## 🐍 3. 실습 소스코드

- `producer.py`:
    - 실시간 원천 로그 데이터를 생성하여 Kafka로 밀어 넣는 가상 소스 모듈
- `pipeline.py`:
    - Spark를 이용해 Kafka 데이터를 읽어 DuckDB에 저장하고,
    - LangChain으로 벡터화하여 Qdrant에 적재하는 파이프라인 모듈
- `app.py`:
    - Streamlit을 활용한 실시간 RAG 챗봇 모니터링 모듈

- **필수 라이브러리 설치하기**
    - kafka-python을 안 쓰는 이유: 
        - 예전에 많이 쓰이던 kafka-python 라이브러리는
        - 현재 파이썬 3.11+ 버전 및 최신 Kafka 4.x (KRaft) 환경에서 내부 비동기 루프 에러를 발생시킴
        - 사실상 기업들의 표준인 confluent-kafka를 사용해야 안정적으로 기동됨    
    - langchain 전체를 안 쓰는 이유: 
        - 이번 예제에서는 무겁게 전체를 다 설치할 필요 없이, 
        - 외부 모델 연동용 플러그인 모듈만 모아놓은 langchain-community 패키지만 설치하면
        - 로컬 외부 Ollama 연결을 완벽하게 수행할 수 있음
        - 따라서 가상환경을 훨씬 가볍게 유지할 수 있음
    - pandas, numpy 등은 streamlit 설치 시 의존성에 의해 함께 설치됨

```bash
pip install confluent-kafka duckdb qdrant-client redis langchain-community langchain-qdrant openai streamlit
```


- **[모듈 1] `producer.py` (실시간 로그 제너레이터)**
    - 공장 설비 및 센서 오작동 이력 정보를 실시간 스트리밍으로 모사하여 Kafka에 발행

```python
#//file: "producer.py"

import json
import time
import random
from datetime import datetime
from confluent_kafka import Producer

# Kafka 최신 연결 구성 (포트 9092)
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ 메시지 전송 실패: {err}")
    else:
        print(f"📡 [발행 완료] Topic: {msg.topic()} | Part: [{msg.partition()}]")

# 스마트팩토리 알람 로그 가상 데이터 풀
topics_pool = [
    {"error_code": "E0041", "component": "프레스 압축 피스톤", "status": "임계 온도 초과 수치 180°C 도달. 윤활유 보충 요망"},
    {"error_code": "E0085", "component": "컨베이어 로봇 그리퍼", "status": "압력 공기압 실린더 감쇠 현상 발생. 10% 압력 강하 감지"},
    {"error_code": "E0012", "component": "조립 라인 센서 모듈", "status": "Hough 변환 정렬 오차 1.5mm 초과. 가변 면적 범위 이탈"}
]

print("🚀 KRaft 기반 Kafka 실시간 공정 로그 발행 중...")
try:
    while True:
        log_data = random.choice(topics_pool).copy()
        log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 9092 포트로 메시지 전송
        producer.produce(
            'factory-logs', 
            key=log_data["error_code"], 
            value=json.dumps(log_data).encode('utf-8'),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(3)
except KeyboardInterrupt:
    print("🛑 발행 중단")
finally:
    producer.flush()
```


- **[모듈 2] `pipeline.py` (데이터 정제 & 임베딩 파이프라인)**
    - Kafka 토픽을 실시간으로 가로채어 분석용 DuckDB에 쌓고,
    - 이 중 의미 있는 텍스트를 추출하여 로컬 임베딩 처리 후
    - Qdrant에 적재하는 실전 파이프라인 코드

```python
#//file: "pipeline.py"

# pipeline.py
import json
import duckdb
from confluent_kafka import Consumer, KafkaError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings

# =====================================================================
# [Step 1] 테이블 생성 (이 순간에만 잠깐 열고 바로 닫아서 Lock을 방지합니다)
# =====================================================================
with duckdb.connect('factory_analytics.db') as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            timestamp VARCHAR,
            error_code VARCHAR,
            component VARCHAR,
            status VARCHAR
        )
    """)

# =====================================================================
# [Step 2] Qdrant v1.18.0 컬렉션(768 차원 nomic-embed-text 규격) 빌드
# =====================================================================
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "factory_vector_db"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# 로컬 Ollama에 등록된 전용 임베딩 모델 포인터 선언
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

# =====================================================================
# [Step 3] Kafka Consumer 환경 구축 (컨테이너 포트 19092 사용)
# =====================================================================
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'pipeline-group',
    'auto.offset.reset': 'latest'
}
consumer = Consumer(conf)
consumer.subscribe(['factory-logs'])

print("⚡ KRaft 데이터 파이프라인 활성화 - 이벤트를 대기 중입니다...")

# =====================================================================
# [Step 4] 실시간 이벤트를 받아 적재하는 메인 루프
# =====================================================================
try:
    point_id_counter = 1
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"❌ Kafka 에러 감지: {msg.error()}")
                break

        # Kafka 토픽에서 메시지 수신 및 역직렬화
        log = json.loads(msg.value().decode('utf-8'))
        print(f"📥 수집 로그: {log}")

        # -----------------------------------------------------------------
        # 🚀 [핵심 보완] DuckDB 실시간 데이터 로깅 (with 블록 안으로 INSERT 진입)
        # 이 블록 안에서만 DB 파일을 열고, 작업을 끝내고 블록을 나가는 순간 즉시 닫힙니다 (Lock 즉시 해제).
        # -----------------------------------------------------------------
        with duckdb.connect('factory_analytics.db') as duck_conn:
            duck_conn.execute(
                "INSERT INTO system_logs VALUES (?, ?, ?, ?)",
                (log['timestamp'], log['error_code'], log['component'], log['status'])
            )

        # -----------------------------------------------------------------
        # [RAG 단계 1] 실시간 로그 정보를 결합해 벡터 임베딩 생성 (nomic-embed-text)
        # -----------------------------------------------------------------
        doc_content = f"[{log['timestamp']}] 에러코드 {log['error_code']}가 {log['component']}에서 발생함. 세부 상태: {log['status']}"
        vector = embeddings.embed_query(doc_content)

        # -----------------------------------------------------------------
        # [RAG 단계 2] Qdrant v1.18.0 벡터 스토어에 업서트(Upsert)
        # -----------------------------------------------------------------
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id_counter,
                    vector=vector,
                    payload={"page_content": doc_content, "metadata": {"error_code": log['error_code']}}
                )
            ]
        )
        point_id_counter += 1
        print("✅ DuckDB 및 Qdrant v1.18.0 벡터 저장 완료")

except KeyboardInterrupt:
    print("🛑 파이프라인 수동 중단")
finally:
    consumer.close()
```

- **[모듈 3] `app.py` (Streamlit RAG 및 Redis 캐시 챗봇)**
    - 실시간 분석 결과와 모니터링 테이블을 출력
    - Redis 시맨틱 캐싱 기반의 초고속 Gemma 4 대답 처리를 포함한 대시보드 웹 앱을 빌드

```python
#//file: "app.py"

# app.py (레이아웃 및 입력창 버그 완벽 패치 버전)
import os
import streamlit as st
import redis
import duckdb
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

# 페이지 구성 정의
st.set_page_config(page_title="스마트공장 관제 에이전트", layout="wide")
st.title("🏭 실시간 스마트팩토리 AI 관제 대시보드")

# =====================================================================
# [Step 1] 인프라 클라이언트 통합 초기화
# =====================================================================
@st.cache_resource
def init_connections():
    r_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    
    client = QdrantClient(url="http://localhost:6333")
    vector_db = QdrantVectorStore(
        client=client,
        collection_name="factory_vector_db",
        embedding=embeddings
    )
    
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return r_client, vector_db, ollama_client

r_client, vector_db, ollama_client = init_connections()

# 세션 상태 초기화 (대화 기록 보존용)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어떤 공정 설비의 오류를 파악해 드릴까요?"}]

# =====================================================================
# [Step 2] 화면 레이아웃 분할
# =====================================================================
col_db, col_chat = st.columns([5, 5])

# --- [좌측 패널] DuckDB 적재 이력 모니터링 ---
with col_db:
    st.subheader("📊 DuckDB 적재 이력 모니터링")
    if st.button("🔄 실시간 데이터 새로고침"):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, 'factory_analytics.db')
            
            with duckdb.connect(db_path, read_only=True) as duck_conn:
                df = duck_conn.execute("SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT 10").fetchdf()
                st.dataframe(df, width="stretch")
        except Exception as e:
            st.info("데이터베이스가 비어 있습니다. 파이프라인을 먼저 가동해 주세요.")

# --- [우측 패널] 공정 설비 RAG 관제 에이전트 챗봇 ---
with col_chat:
    st.subheader("💬 공정 설비 가이드 챗봇 (RAG + Cache)")
    
    # 🚀 핵심 개선 1: 대화 기록만 전용으로 렌더링할 독립된 컨테이너 영역 생성
    # height를 지정하여 대화가 길어져도 스크롤바가 안쪽에만 생기도록 가두어 둡니다.
    chat_container = st.container(height=500)
    
    # 🚀 핵심 개선 2: 기존 대화 기록을 전용 컨테이너 안에 순차 렌더링
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    # 🚀 핵심 개선 3: 입력창(st.chat_input)을 컨테이너 바깥 가장 하단에 영구 배치
    # 이렇게 분리하면 대화 출력 중에도 입력 폼의 DOM 포인터가 깨지지 않고 고정됩니다.
    if prompt := st.chat_input("에러 코드나 부품명을 입력하세요 (예: E0041)"):
        # 사용자의 질문을 화면과 세션에 즉시 즉시 추가
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # [Step 1] Redis 캐시 레이어 조회
        cached_response = r_client.get(prompt)
        
        with chat_container:
            with st.chat_message("assistant"):
                if cached_response:
                    st.info("⚡ [Redis Cache Hit] 이전 동일 응답을 즉시 불러왔습니다.")
                    st.markdown(cached_response)
                    full_response = cached_response
                else:
                    st.warning("🔍 [Cache Miss] 로컬 벡터스토어 및 Gemma 4 실시간 추론 구동")
                    
                    # [Step 2] QdrantVectorStore 유사도 기반 시맨틱 탐색
                    with st.status("실시간 컨텍스트 데이터 수집 중..."):
                        found_docs = vector_db.similarity_search(prompt, k=2)
                        context_text = "\n".join([doc.page_content for doc in found_docs])
                    
                    # [Step 3] 시스템 프롬프트 조립
                    sys_msg = f"""
                    당신은 스마트팩토리 수석 공정 제어 엔지니어입니다.
                    다음 [수집된 실시간 설비 컨텍스트]에 명시된 팩트만을 근거로 삼아 정중하고 격식 있는 어조의 리포트 조언을 작성하세요.
                    
                    [수집된 실시간 설비 컨텍스트]
                    {context_text}
                    """
                    
                    # [Step 4] Ollama 스트리밍 호출
                    stream = ollama_client.chat.completions.create(
                        model="gemma4",
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True
                    )
                    
                    def stream_generator():
                        for chunk in stream:
                            content_chunk = chunk.choices[0].delta.content
                            if content_chunk:
                                yield content_chunk
                                
                    full_response = st.write_stream(stream_generator())
                    r_client.setex(prompt, 600, full_response)
                    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 🚀 핵심 개선 4: 입력 완료 후 화면 싱크를 맞추기 위해 강제 1회성 Rerun 트리거
        st.rerun()
```


## 🛠️ 4. 실습 및 검증 순서

- 이 시나리오는 모든 모듈을 로컬 터미널 4개에 순서대로 실행해 봄으로써 검증할 수 있음

1. **터미널 ① (Docker 실행)**

    ```bash
    docker compose up -d
    # gemma4 모델 다운로드 상태 체크
    docker exec -it ollama-container ollama run gemma4
    ```

2. **터미널 ② (Kafka 데이터 발행)**

    ```bash
    python producer.py
    ```

3. **터미널 ③ (실시간 데이터 적재 및 파이프라인 가동)**

    ```bash
    python pipeline.py
    ```

4. **터미널 ④ (Streamlit 웹 대시보드 기동)**

    ```bash
    streamlit run app.py
    ```

    - 웹 화면 접속 🡪 `실시간 데이터 새로고침` 🡪 Kafka를 거쳐 DuckDB에 누적되고 있는 최신 설비 데이터의 변화를 볼 수 있음
    - 이후 챗봇에 `E0041` 혹은 `컨베이어 로봇 그리퍼`를 질문
        - 최초 호출 시
            - **`Cache Miss`** 경고 노출
            - Qdrant 벡터 저장소를 실시간 검색
            - 최신 상태 정보를 Gemma 4가 리포팅
        - **동일 질문을 연속 두 번 수행하는 경우**
            - 시스템 연산량을 소모하지 않는 `Redis Cache Hit`가 트리거
                - 0.01초 만에 캐시 데이터가 화면에 노출되는 초고속 자원 제어 성능을 직접 확인가능



5. **확인용 시나리오**

    - **1단계. 실시간 "Hough 변환 정렬" 관련 기하학적 질문 (가장 최신 로그 유도)**
        - **던질 질문:**
            - `조립 라인의 정렬 오차 상태는 어때?` 또는
            - `E0012 조치 방법 알려줘.`
        - **관찰 포인트 (RAG 및 센서 용어 검증):**
            - 센서 모듈에서 실시간 검출된 **"Hough 변환 정렬 오차 1.5mm 초과"** 및 **"가변 면적 범위 이탈"** 등의 원본 상태 값이 Qdrant에서 실시간으로 조회되어 Gemma 4의 컨텍스트로 주입되는지 확인
            - 단순히 "E0012가 발생했다"가 아니라, 컨텍스트에 박힌 실시간 수치(1.5mm)를 기반으로 정밀한 제어 보고서 어조의 가이드라인을 생성하는지 검증

    - **2단계. 모호한 자연어로 질문하기 (시맨틱 유사도 검색 검증)**
        - **던질 질문:**
            - `공장에 지금 열나는 장비가 하나 있는 것 같은데 확인해 봐.`
        - **관찰 포인트 (Qdrant Vector DB의 진가):**
            - 로그 원본에는 "열나는 장비"라는 말이 없고 "임계 온도 초과 수치 180°C 도달"이라는 단어만 있음
            - Qdrant의 `nomic-embed-text` 임베딩 엔진은 '열나다'와 '온도 초과 180°C'가 의미론적으로 매우 유사하다는 것을 이해함
            - 챗봇이 정확하게 `E0041(프레스 압축 피스톤)` 로그를 찾아내어 "임계 온도 180°C 초과 상태가 감지되었습니다"라고 답변하는지 확인

    - **3단계. 똑같은 질문 연달아 두 번 던지기 (Redis 캐시 엔진 속도 검증)**
        - **던질 질문 (1회차):**
            - `컨베이어 로봇 그리퍼 공기압 상태 보고서 써줘.`
        - **관찰 포인트 (Redis 인메모리 고속 처리 검증):**
            - **1회차:**
                - 질문 창 아래에 노란색 경고로 `🔍 [Cache Miss]`가 뜨며,
                - Qdrant 검색과 외부 Gemma 4가 동작하느라 타이핑 효과가 나오며 조금 지연 동작함

        - **던질 질문 (2회차 - 답변 완료 후 똑같이 또 질문):**
            - `컨베이어 로봇 그리퍼 공기압 상태 보고서 써줘.`
        - **관찰 포인트 (Redis 인메모리 고속 처리 검증):**
            - **2회차:**
                - 엔터를 치는 순간 노란색 경고가 파란색 `⚡ [Redis Cache Hit]`으로 순식간에 바뀌면서,
                - 단 0.01초 만에 방대한 보고서 답변이 화면에 그대로 복사되어 출력됨
            - 2회차 호출 시에는 로컬 Ollama 터미널에 아무런 연산 로그가 남지 않는 것(LLM 자원 낭비 원천 차단)을 확인