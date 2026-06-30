---
layout: page
title:  "Kafka Connect를 이용한 실시간 데이터의 MinIO 적재"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-05-01_01-KafkaConnectMinIo
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. Kafka Connect 개요

- **Kafka Connect**
    - Apache Kafka의 공식 에코시스템 중 하나
    - Kafka와 외부 시스템 간에 데이터를 효율적이고 안전하게 주고받을 수 있도록 표준화된 방법을 제공하는 프레임워크
        - 외부 시스템: 데이터베이스, 키-값 저장소, 검색 엔진, 파일 시스템, 클라우드 스토리지 등
    - 코드 한 줄 작성하지 않고(No-Code/Low-Code), 선언적인 JSON 설정 파일만으로 대규모 데이터 파이프라인을 연동할 수 있음
        - 과거:
            - 외부 데이터를 카프카로 가져오거나 보낼 때
            - 직접 Producer와 Consumer 애플리케이션 코드를 자바나 파이썬으로 작성
        - Kafka Connect:
            - 이러한 반복적인 데이터 이동 작업을 템플릿화
            - JSON 설정 파일 하나만으로 완벽한 데이터 파이프라인을 구동할 수 있게 정형화

- **Kafka Connect를 사용하는 핵심 이유 (개발 생산성 및 안정성)**
    - **개발 생산성**과 **시스템 안정성**을 프레임워크 수준에서 보장받기 위함
        - 만약 Kafka Connect를 쓰지 않고 파이썬이나 자바로 직접 적재 프로그램(Consumer Application)을 짜서 MinIO에 저장한다면
        - 다음과 같은 복잡한 문제를 개발자가 직접 코드로 해결해야 함
            - **장애 복구(Fail-over):**
                - 데이터를 저장하던 중 Consumer 프로세스가 죽으면 어느 시점부터 다시 읽어서 저장해야 하는가?
            - **분산 처리 및 스케일 아웃:**
                - 초당 수십만 건의 데이터가 밀려올 때, 여러 프로세스로 어떻게 부하를 분산하고 파티션을 나눠 맡을 것인가?
            - **오프셋 관리:**
                - 데이터가 누락되거나 중복 저장되지 않도록 오프셋 커밋(Commit) 처리를 어떻게 신뢰성 있게 보장할 것인가?

    - Kafka Connect 프레임워크는 **Offset 자동 관리, Task 단위의 자동 부하 분산, 장애 발생 시 유기적인 Rebalancing** 메커니즘을 내장
        - **오프셋(Offset) 관리의 자동화:**
            - "내가 원천 DB 로그를 어디까지 읽었는지", "MinIO 스토리지에 몇 번 메시지까지 저장했는지"와 같은 상태 정보(Offset)를
            - 카프카 내부 메타데이터 토픽에 알아서 저장하고 관리해 줌
            - 시스템이 불시에 뻗어도 데이터 누락이나 중복 없이 멈춘 곳부터 정확히 복구됨
        * **REST API 기반의 동적 제어:**
            - 코드를 수정하고 서버를 재빌드할 필요가 없음
            - `8083` 포트 인터페이스를 통해 JSON 명령어를 쏘는 것만으로 실시간으로 파이프라인을 켜고, 끄고, 설정을 변경할 수 있음
        - **파편화된 연동 코드의 통합:**
            - 기업 내에 MySQL, PostgreSQL, S3, Elasticsearch 등 수십 개의 저장소가 존재할 때,
            - 카프카와 연결하는 코드를 일일이 짜면 유지보수가 불가능해짐
            - Connect는 검증된 플러그인(디비지움 등)을 꽂기만 하면 되므로 인프라가 극도로 단순해짐

    - 개발자는 인프라의 안정성을 프레임워크에 맡기고, 오직 데이터의 '출발지'와 '목적지' 설정에만 집중할 수 있음


## 2. Kafka Connect의 핵심 구조 및 아키텍처

- **커넥터 (Connector)**
    - 데이터 파이프라인의 **'방향성과 규칙'을 정의하는 뇌**의 역할
    - 어떤 토픽의 데이터를 읽을지, 외부 어떤 DB에 저장할지 등의 메타데이터 설정을 관리 🡲 실질적인 데이터 이동을 지휘
    - 데이터가 흐르는 방향에 따라 소스 커넥터와 싱크 커넥터로 분류됨
        - **Source Connector (수집):**
            - 외부 시스템(예: MySQL, Oracle, Application Log)의 데이터를 캡처하여 Kafka 토픽으로 밀어 넣는 역할 (생산자 우회)
            - 대표적인 소스 커넥터: Debezium CDC

        - **Sink Connector (적재):**
            - Kafka 토픽의 데이터를 읽어와 외부 저장소(예: MinIO, S3, Elasticsearch)로 내보내는 역할 (소비자 우회)
            - **본 과정에서는 S3/MinIO 전용 Sink Connector를 사용**

- **태스크 (Task)**
    - 커넥터가 지휘관이라면, 태스크는 실제로 데이터를 나르는 '일꾼 스레드(Thread)'
    - 카프카의 파티션 구조와 연계되어 물리적인 데이터 복사 작업을 수행
        - 예: 카프카 토픽의 파티션이 3개 🡲 태스크를 최대 3개까지 늘려 파티션 하나씩을 전담 마크하게 함 🡲 병렬 처리(Scale-Out) 달성

- **워커 (Worker)**
    - 태스크와 커넥터가 실행되는 **물리적인 서버 프로세스(컨테이너)** 환경
    - **단독 모드(Standalone):**
        - 단 한 대의 프로세스만 띄우는 방식
        - 개발 환경이나 단순 파일 백업용으로 사용
    - **분산 모드(Distributed):**
        - 여러 대의 워커를 클러스터로 묶는 방식
        - 특정 워커 서버가 죽으면(Fault), 그 안에서 돌던 태스크들을 다른 건강한 워커 서버로 자동으로 이사시키는 **고가용성(Fail-over)** 메커니즘을 내장
        - 실무 표준으로 사용됨

- **컨버터 (Converter)**
    - 데이터가 국경을 넘을 때 언어를 통역해 주는 '번역기'
        - 카프카 브로커 내부에서는 데이터를 단순 바이트(Byte) 배열로 저장하지만,
        - 외부 DB나 파이썬 애플리케이션은 이를 JSON, Avro, Protobuf 등의 구조화된 포맷으로 읽어야 함
    - 컨버터가 중간에서 이 직렬화(Serialization) 및 역직렬화 작업을 자동으로 수행

<br>

> - **Kafka Connect는**
>   - **"카프카와 외부 저장소 사이를 연결하는 No-Code 기반의 분산 데이터 셔틀 버스"**
>   - 분산 아키텍처 특유의 복잡한 장애 복구, 병렬 처리, 분산 잠금 메커니즘을 백엔드 깊숙이 숨겨두고,
>   - 엔지니어에게는 "오직 출발지와 목적지만 명시하면 데이터는 우리가 안전하게 나르겠다"는 인터페이스를 제공하는 도구
>   - 이 인프라가 갖춰져야 비로소 엔드투엔드(End-to-End) 실시간 데이터 레이크 파이프라인이 완성됨
{: .summary-quote}



## 3. 실시간 데이터 MinIO 적재 아키텍처 및 관련 기술

- 실습에서 구현할 실시간 스트리밍 데이터 파이프라인의 전체 아키텍처

<div class="insert-image">
    <img src="/materials/devtools/images/S13-07-05-01_01-001_KafkaConnectMinIo.png" style="width: 80%;">
</div>


- **관련 기술 구성 요소 설명**
    - **Apache Kafka (3-Broker Cluster):**
        - 데이터 고속도로 역할을 수행
        - 주키퍼 없이 KRaft 모드로 자율 제어되는 3대의 브로커 시스템

    - **MinIO:**
        - 오픈소스 기반의 고성능 오브젝트 스토리지(Object Storage)
        - AWS S3와 완전히 동일한 API 규격을 공유
        - 로컬 환경에서 AWS S3 데이터 레이크 인프라를 그대로 시뮬레이션할 수 있는 표준 도구

    - **Camel-S3 Sink Connector:**
        - Apache Camel 프로젝트에서 제공하는 S3 호환 전용 싱크 커넥터
        - Kafka 토픽에 들어오는 메시지를 낚아채어 지정된 분량(시간 또는 개수 크기)만큼 묶어서 MinIO 버킷에 파일 형식으로 자동 업로드함


## 4. 실습 예제

- 본 실습은 호스트 PC(Ubuntu) 환경의 동일한 디렉토리 내부에서 진행
- 아파치 공식 Kafka 클러스터 위에 MinIO와 Kafka Connect 인프라를 결합하고,
- 파이썬을 이용해 데이터를 흘려보내는 전 과정을 수행함

- **[1단계] 복합 인프라 환경 구축 (`docker-compose.yml`)**
    - 기존의 3-브로커 설정 아래에 **MinIO**와 **Kafka Connect** 서비스를 통합

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

    - 터미널에서 아래 명령어로 모든 인프라를 구동

        ```bash
        docker compose up -d
        ```


- **[2단계] MinIO 웹 버킷(Bucket) 생성 및 확인**

    1. 브라우저를 열고 `http://localhost:9001`에 접속
    2. ID: `minioadmin` / PW: `minioadmin`을 입력하여 대시보드에 로그인
    3. **[Create Bucket]** 버튼을 클릭하여 데이터가 적재될 공간인 `telemetry-data-lake`라는 이름의 버킷을 생성


- **[3단계] Kafka Connect에 MinIO Sink Connector 등록하기**

    - Kafka Connect의 REST API 포트(`8083`)로 JSON 명세서 발송
    - "카프카의 `minio-stream-topic`에 들어오는 데이터를 실시간으로 감시해서 MinIO 버킷으로 전송"하는 명령
        - 호스트 PC 터미널 창에서 다음 `curl` 명령어 실행

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

            - 핵심 설정
                - `"topics"`:
                    - 실시간 타겟 카프카 토픽 이름 명시 (`minio-stream-topic`)
                - `"camel.sink.path.bucketNameOrArn"`:
                    - 앞에서 생성한 MinIO 버킷 이름과 매핑
                - `"camel.component.aws2-s3.uriEndpointOverride"`:
                    - 외부 AWS가 아닌 도커 가상망 안에 있는 로컬 `http://minio:9000` 주소로 우회 강제 지정
                - `"camel.sink.endpoint.keyName"`:
                    - 버킷 안에 데이터가 저장될 때 날짜별로 디렉토리를 자동 분할하여 배치(`연월일/아이디.txt`)하라는 규칙 정의

- **[4단계] 파이썬 실시간 시뮬레이터 구동 (`stream_producer.py`)**

    - 원천 데이터를 생성할 파이썬 시뮬레이터 가동
        - 가상환경이 켜진 호스트 쉘에서 `stream_producer.py` 파일 실행

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
            print(f"[시뮬레이터 가동] '{topic_name}'으로 초당 5개씩 IoT 센서 데이터를 전송합니다.\n")

            try:
                while True:
                    sensor_data = {
                        "device_id": f"SENSOR-{random.randint(1, 5)}",
                        "temperature": round(random.uniform(20.0, 35.5), 2),
                        "humidity": round(random.uniform(40.0, 60.0), 2),
                        "timestamp": time.time()
                    }
                    
                    producer.send(topic_name, value=sensor_data)
                    print(f"데이터 전송 -> ID: {sensor_data['device_id']} | Temp: {sensor_data['temperature']}°C")
                    time.sleep(0.2) # 0.2초당 1개씩 (초당 5개 고속 발행)

            except KeyboardInterrupt:
                print("\n전송을 중단합니다.")
            finally:
                producer.flush()
                producer.close()
            ```

- **최종 결과 검증 및 총평**

    - 파이썬 코드 스크립트 가동 🡲 터미널 창에 `데이터 전송` 로그가 쌓이기 시작<br>
        🡲 MinIO 대시보드 브라우저(`http://localhost:9001`) 접속

        1. **`telemetry-data-lake`** 버킷 내부로 진입
        2. 오늘 날짜로 명명된 디렉토리(예: `20260627/`)가 자동으로 생성되어 있는 것을 확인
        3. 디렉토리 내부 클릭 🡲 수많은 실시간 데이터 파일들(`.txt`)이 유실 없이 실시간으로 고속 적재되고 있는 모습 확인
            - 코드 작성을 전혀 하지 않았음에도 불구하고 잘 진행됨을 확인할 것

<br>

> - **파이프라인 완성 총평:**
>   - 원천 생산자로부터 뿜어져 나온 이벤트가
>   - 자율형 분산 Kafka 브로커 클러스터를 통과하고,
>   - 가용성이 확보된 Kafka Connect 엔진의 제어를 받아
>   - 최종 클라우드 스토리지인 MinIO 인프라에 안전하게 파이프라인 정착이 완료됨을 확인<br><br>
> - **실시간 인메모리 데이터 레이크 수집 아키텍처**를 독자적으로 빌드하는 역량 확보
{: .summary-quote}