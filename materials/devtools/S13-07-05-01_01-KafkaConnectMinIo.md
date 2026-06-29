---
layout: page
title:  "Kafka Connect를 이용한 실시간 데이터의 MinIO 적재"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-05-01_01-KafkaConnectMinIo
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



# [6주차 심화] Kafka Connect를 이용한 실시간 데이터 MinIO 적재

* **교육 목적:** 오픈소스 분산 메시지 중추인 Kafka와 S3 호환 오브젝트 스토리지인 MinIO를 코드 작성 없이 연결하는 Kafka Connect 아키텍처를 이해하고, 실시간 데이터 레이크(Data Lake) 파이프라인을 구축하는 실무 역량 내재화.

---

## 1. Kafka Connect의 상세 개념 및 핵심 철학

- **Kafka Connect**
    - Apache Kafka의 공식 에코시스템 중 하나
    - Kafka와 외부 시스템 간에 데이터를 효율적이고 안전하게 주고받을 수 있도록 표준화된 방법을 제공하는 프레임워크
        - 외부 시스템: 데이터베이스, 키-값 저장소, 검색 엔진, 파일 시스템, 클라우드 스토리지 등
    - 코드 한 줄 작성하지 않고(No-Code/Low-Code), 선언적인 JSON 설정 파일만으로 대규모 데이터 파이프라인을 연동할 수 있음
    - 데이터가 흐르는 방향에 따라 소스 커넥터와 싱크 커넥터로 분류됨

- **Source Connector (수집):**
    - 외부 시스템(예: MySQL, Oracle, Application Log)의 데이터를 캡처하여 Kafka 토픽으로 밀어 넣는 역할
    - 대표적인 소스 커넥터: Debezium CDC

- **Sink Connector (적재):**
    - Kafka 토픽에 쌓여 있는 실시간 데이터를 구독(Consume)하여 외부 시스템(예: Elasticsearch, MinIO, AWS S3)으로 안전하게 내보내고 적재하는 역할을 합니다. **본 과정에서는 S3/MinIO 전용 Sink Connector를 사용합니다.**

### 1.3 Kafka Connect를 사용하는 핵심 이유 (개발 생산성 및 안정성)

만약 Kafka Connect를 쓰지 않고 파이썬이나 자바로 직접 적재 프로그램(Consumer Application)을 짜서 MinIO에 저장한다면 다음과 같은 복잡한 문제를 개발자가 직접 코드로 해결해야 합니다.

1. **장애 복구(Fail-over):** 데이터를 저장하던 중 Consumer 프로세스가 죽으면 어느 시점부터 다시 읽어서 저장해야 하는가?
2. **분산 처리 및 스케일 아웃:** 초당 수십만 건의 데이터가 밀려올 때, 여러 프로세스로 어떻게 부하를 분산하고 파티션을 나눠 맡을 것인가?
3. **오프셋 관리:** 데이터가 누락되거나 중복 저장되지 않도록 오프셋 커밋(Commit) 처리를 어떻게 신뢰성 있게 보장할 것인가?

Kafka Connect 프레임워크는 내부적으로 **오프셋 자동 관리, 태스크(Task) 단위의 자동 부하 분산, 장애 발생 시 유기적인 재밸런싱(Rebalancing)** 메커니즘을 내장하고 있습니다. 개발자는 인프라의 안정성을 프레임워크에 맡기고, 오직 데이터의 '출발지'와 '목적지' 설정에만 집중할 수 있습니다.

---

## 2. 실시간 데이터 MinIO 적재 아키텍처 및 관련 기술

본 실습에서 구현할 실시간 스트리밍 데이터 파이프라인의 전체 아키텍처는 다음과 같습니다.

```
+------------------+     (1) Produce      +---------------------------------+
|  Python Producer | -------------------> | Apache Kafka (3-Broker Cluster) |
+------------------+                      +---------------------------------+
                                                           |
                                                           | (2) Pull Stream
                                                           v
+------------------+     (3) Upload       +---------------------------------+
|   MinIO Bucket   | <------------------- |  Kafka Connect (S3 Sink Conn)  |
| (S3 Data Lake)   |                      +---------------------------------+
+------------------+

```

### 관련 기술 구성 요소 설명

* **Apache Kafka (3-Broker Cluster):** 데이터 고속도로 역할을 수행하며, 주키퍼 없이 KRaft 모드로 자율 제어되는 3대의 브로커 시스템입니다.
* **MinIO (민아이오):** 오픈소스 기반의 고성능 오브젝트 스토리지(Object Storage)입니다. AWS S3와 완전히 동일한 API 규격을 공유하기 때문에, 로컬 환경에서 AWS S3 데이터 레이크 인프라를 그대로 시뮬레이션할 수 있는 표준 도구입니다.
* **Camel-S3 Sink Connector:** Apache Camel 프로젝트에서 제공하는 S3 호환 전용 싱크 커넥터입니다. Kafka 토픽에 들어오는 메시지를 낚아채어 지정된 분량(시간 또는 개수 크기)만큼 묶어서 MinIO 버킷에 파일 형식으로 자동 업로드합니다.

---

## 3. 엔드 투 엔드(End-to-End) 실전 실습 예제

본 실습은 호스트 PC(Ubuntu) 환경의 동일한 디렉토리 내부에서 진행됩니다. 아파치 공식 Kafka 클러스터 위에 MinIO와 Kafka Connect 인프라를 결합하고, 파이썬을 이용해 데이터를 흘려보내는 전 과정을 수행합니다.

### [1단계] 복합 인프라 환경 구축 (`docker-compose.yml`)

기존의 3-브로커 설정 아래에 **MinIO**와 **Kafka Connect** 서비스를 깔끔하게 통합한 파일입니다. 이 내용을 `docker-compose.yml`로 저장합니다.

```yaml
services:
  kafka-1:
    image: apache/kafka:latest
    container_name: kafka-1
    ports:
      - "9092:9092"
    environment:
      - KAFKA_NODE_ID=1
      - KAFKA_PROCESS_ROLES=controller,broker
      - KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
      - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_LOG_DIRS=/var/lib/kafka/data
    volumes:
      - kafka_1_data:/var/lib/kafka/data

  kafka-2:
    image: apache/kafka:latest
    container_name: kafka-2
    ports:
      - "9094:9092"
    environment:
      - KAFKA_NODE_ID=2
      - KAFKA_PROCESS_ROLES=controller,broker
      - KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9094
      - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_LOG_DIRS=/var/lib/kafka/data
    volumes:
      - kafka_2_data:/var/lib/kafka/data

  kafka-3:
    image: apache/kafka:latest
    container_name: kafka-3
    ports:
      - "9095:9092"
    environment:
      - KAFKA_NODE_ID=3
      - KAFKA_PROCESS_ROLES=controller,broker
      - KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9095
      - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_LOG_DIRS=/var/lib/kafka/data
    volumes:
      - kafka_3_data:/var/lib/kafka/data

  # [추가] S3 대체품 오픈소스 오브젝트 스토리지 MinIO
  minio:
    image: minio/minio:latest
    container_name: minio
    ports:
      - "9000:9000"       # API 포트
      - "9001:9001"       # 웹 GUI 대시보드 포트
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  # [추가] S3 Sink 커넥터가 사전 탑재된 Kafka Connect 엔진
  kafka-connect:
    image: camel/kafka-connector:latest
    container_name: kafka-connect
    ports:
      - "8083:8083"       # 커넥터 관리를 위한 REST API 포트
    environment:
      - CONNECT_BOOTSTRAP_SERVERS=kafka-1:9093,kafka-2:9093,kafka-3:9093 # 도커망 내부 통신 포트 활용
      - CONNECT_REST_PORT=8083
      - CONNECT_REST_ADVERTISED_HOST_NAME=kafka-connect
      - CONNECT_GROUP_ID=connect-cluster
      - CONNECT_CONFIG_STORAGE_TOPIC=connect-configs
      - CONNECT_OFFSET_STORAGE_TOPIC=connect-offsets
      - CONNECT_STATUS_STORAGE_TOPIC=connect-status
      - CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR=1
      - CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR=1
      - CONNECT_STATUS_STORAGE_REPLICATION_FACTOR=1
      - CONNECT_KEY_CONVERTER=org.apache.kafka.connect.storage.StringConverter
      - CONNECT_VALUE_CONVERTER=org.apache.kafka.connect.storage.StringConverter

volumes:
  kafka_1_data:
  kafka_2_data:
  kafka_3_data:
  minio_data:

```

터미널에서 아래 명령어로 모든 인프라를 구동합니다.

```bash
docker compose up -d

```

### [2단계] MinIO 웹 버킷(Bucket) 생성 및 확인

1. 브라우저를 열고 `http://localhost:9001`에 접속합니다.
2. ID: `minioadmin` / PW: `minioadmin`을 입력하여 대시보드에 로그인합니다.
3. **[Create Bucket]** 버튼을 클릭하여 데이터가 적재될 공간인 `telemetry-data-lake`라는 이름의 버킷을 생성합니다.

### [3단계] Kafka Connect에 MinIO Sink Connector 등록하기

이제 Kafka Connect의 REST API 포트(`8083`)로 JSON 명세서를 발송하여, "카프카의 `minio-stream-topic`에 들어오는 데이터를 실시간으로 감시해서 MinIO 버킷으로 쏴라" 하는 가동 명령을 내릴 차례입니다.

호스트 PC 터미널 창에서 다음 `curl` 명령어를 그대로 실행합니다.

```bash
curl -X POST -H "Content-Type: application/json" --data '{
  "name": "minio-sink-connector",
  "config": {
    "connector.class": "org.apache.camel.kafkaconnector.aws2s3sink.CamelAws2s3sinkSinkConnector",
    "tasks.max": "1",
    "topics": "minio-stream-topic",
    "camel.sink.path.bucketNameOrArn": "telemetry-data-lake",
    "camel.component.aws2-s3.amazonS3Client": "#class:org.apache.camel.component.aws2.s3.utils.S3ClientUtils",
    "camel.component.aws2-s3.accessKey": "minioadmin",
    "camel.component.aws2-s3.secretKey": "minioadmin",
    "camel.component.aws2-s3.region": "us-east-1",
    "camel.component.aws2-s3.overrideEndpoint": "true",
    "camel.component.aws2-s3.uriEndpointOverride": "http://minio:9000",
    "camel.sink.endpoint.keyName": "${date:now:yyyyMMdd}/${exchangeId}.txt"
  }
}' http://localhost:8083/connectors

```

#### 💡 핵심 설정 파헤치기

* `"topics"`: 실시간 타겟 카프카 토픽 이름 명시 (`minio-stream-topic`)
* `"camel.sink.path.bucketNameOrArn"`: 아까 생성한 MinIO 버킷 이름과 매핑
* `"camel.component.aws2-s3.uriEndpointOverride"`: 외부 AWS가 아닌 도커 가상망 안에 있는 로컬 `http://minio:9000` 주소로 우회 강제 지정
* `"camel.sink.endpoint.keyName"`: 버킷 안에 데이터가 저장될 때 날짜별로 디렉토리를 자동 분할하여 배치(`연월일/아이디.txt`)하라는 규칙 정의

### [4단계] 파이썬 실시간 시뮬레이터 구동 (`stream_producer.py`)

이제 원천 데이터를 생성할 파이썬 시뮬레이터를 가동합니다. 가상환경이 켜진 호스트 쉘에서 `stream_producer.py` 파일을 만들어 실행합니다.

```python
import time
import json
import random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = "minio-stream-topic"
print(f"📡 [시뮬레이터 가동] '{topic_name}'으로 초당 5개씩 IoT 센서 데이터를 전송합니다.\n")

try:
    while True:
        sensor_data = {
            "device_id": f"SENSOR-{random.randint(1, 5)}",
            "temperature": round(random.uniform(20.0, 35.5), 2),
            "humidity": round(random.uniform(40.0, 60.0), 2),
            "timestamp": time.time()
        }
        
        producer.send(topic_name, value=sensor_data)
        print(f"📤 데이터 전송 -> ID: {sensor_data['device_id']} | Temp: {sensor_data['temperature']}°C")
        time.sleep(0.2) # 0.2초당 1개씩 (초당 5개 고속 발행)

except KeyboardInterrupt:
    print("\n🛑 전송을 중단합니다.")
finally:
    producer.flush()
    producer.close()

```

---

## 4. 최종 결과 검증 및 총평

파이썬 코드 스크립트를 가동하여 터미널 창에 `📤 데이터 전송` 로그가 물 흐르듯 떨어지기 시작하면, 다시 MinIO 대시보드 브라우저(`http://localhost:9001`)로 들어갑니다.

1. **`telemetry-data-lake`** 버킷 내부로 진입합니다.
2. 오늘 날짜로 명명된 디렉토리(예: `20260627/`)가 자동으로 생성되어 있는 것을 확인할 수 있습니다.
3. 디렉토리 내부를 클릭하면 코드 작성을 전혀 하지 않았음에도 불구하고, 파이를 쪼개듯 수많은 실시간 데이터 파일들(`.txt`)이 유실 없이 실시간으로 고속 적재되고 있는 장관을 볼 수 있습니다.

**💡 파이프라인 완성 총평:**
본 실습을 통해 원천 생산자로부터 뿜어져 나온 이벤트가 자율형 분산 Kafka 브로커 클러스터를 통과하고, 가용성이 확보된 Kafka Connect 엔진의 제어를 받아 최종 클라우드 스토리지인 MinIO 인프라에 안전하게 파이프라인 정착이 완료됨을 확인하였습니다. 이로써 완벽한 **실시간 인메모리 데이터 레이크 수집 아키텍처**를 독자적으로 빌드하는 역량을 완전히 확보하게 되었습니다.