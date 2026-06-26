---
layout: page
title:  "Docker 기반 Kafka 클러스터 구축하기"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-02-01_01-KafkaClusterDocker
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - **Docker 환경의 권장**
>   - Docker 환경에서 Apache Kafka 클러스터를 구축하는 것은 이식성과 확장성 측면에서 매우 효율적
>   - 최신 Kafka 버전은 **KRaft(Kafka Raft)** 모드를 지원하여 Zookeeper 없이도 클러스터 구성 가능
{: .common-quote}


## 1. KRaft 모드 기반의 3-브로커 클러스터 구축

### 1.1 Docker Compose 설정 (KRaft 모드)

- 아파치 공식 가상화 이미지(`apache/kafka:latest`)를 사용하여, 주키퍼(Zookeeper) 없이 작동하는 **KRaft 모드 기반의 3개 노드 분산 Kafka 클러스터**를 로컬 환경에 구성하기 위한 정의서
- 아키텍처가 단순하며 관리 포인트가 적다는 장점을 가진 구조

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

volumes:
  kafka_1_data:
  kafka_2_data:
  kafka_3_data:

```

- **전체 아키텍처의 핵심 요약**
  - 설정 파일이 구성하고자 하는 물리적 시스템 구조
    - **동일 스펙의 3개 브로커:**
      - `kafka-1`, `kafka-2`, `kafka-3`라는 이름의 컨테이너 3개가 동시에 작동

    - **하이브리드 역할:**
      - 모든 노드가
        - 실제 메시지를 저장하고 서비스하는 **브로커(Broker)** 역할과,
        - 클러스터의 상태 및 메타데이터를 관리하는 고도의 투표권을 가진 **컨트롤러(Controller)** 역할을
      - 동시에(`controller,broker`) 수행하도록 설계됨

- **블록별 상세 분석 (중요 환경변수 해석)**
  - **프로세스 및 역할 식별 정의**

    ```yaml
    - KAFKA_NODE_ID=1
    - KAFKA_PROCESS_ROLES=controller,broker
    ```

    - **`KAFKA_NODE_ID`**:
      - 클러스터 내에서 이 브로커를 식별하는 고유한 등록번호(숫자)
      - `kafka-1`은 1, `kafka-2`는 2, `kafka-3`은 3으로 겹치지 않게 매겨져야 함

    - **`KAFKA_PROCESS_ROLES`**:
      - 이 서버가 무엇을 할지 정의함
      - `controller,broker`로 기재했으므로
        - 이 노드는 메시지도 사고팔고,
        - 클러스터 Leader를 뽑는 투표권도 행사하는
      - 다재다능한 노드가 됨

  - **리스너(Listener) 통신망 정의**

    ```yaml
    - KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
    - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
    - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER
    ```

    - **`KAFKA_LISTENERS`**:
      - 컨테이너가 내부적으로 귀를 열고(바인딩) 기다릴 통신 규격과 포트
        - 현재의 설정 내용은
          - 일반 데이터 송수신은 `9092` 포트에서 암호화되지 않은 평문(`PLAINTEXT`)으로 수신
          - 컨트롤러 간의 중추 통신은 `9093` 포트에서 수신
    - **`KAFKA_ADVERTISED_LISTENERS`**:
      - 외부 클라이언트(파이썬 소스코드 등)에게 **"나한테 데이터 보내려면 이 주소로 찾아와라"** 하고 광고하는 주소
      - 호스트 PC에서 도커 밖 인터페이스에서 접속할 수 있도록 `localhost` 포트로 매핑해 둔 것

  - **KRaft 핵심 합의 연동 (주키퍼 독립의 핵심)**

    ```yaml
    - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
    ```

    - **`KAFKA_CONTROLLER_QUORUM_VOTERS`**:
      - KRaft 알고리즘의 핵심 설정
      - 클러스터 메타데이터를 의결하고 리더를 뽑을 권한이 있는 '투표단 명부'

    - 구조: `노드ID@컨테이너이름:포트` 조합
      - 1번, 2번, 3번 노드가
      - 각각 컨테이너 이름을 도메인 삼아
      - 내부 `9093` 제어 포트로 엮여
      - 서로를 투표단으로 인지하게 만듦

  - **영속성(Durability) 데이터 볼륨 설정**

    ```yaml
    - KAFKA_LOG_DIRS=/var/lib/kafka/data
    volumes:
      - kafka_1_data:/var/lib/kafka/data
    ```

    - Kafka는 메시지를 메모리가 아닌 디스크 파일에 영구 저장함
      - `KAFKA_LOG_DIRS`에 데이터가 기록될 컨테이너 내부 디렉토리를 지정
      - 이를 Docker 호스트 컴퓨터가 관리하는 독립 볼륨(`kafka_1_data`)으로 설정
      - 컨테이너를 지우고 새로 띄워도 수집한 메시지가 날아가지 않는 비결이 바로 이 부분

    ```yaml
    volumes:
      kafka_1_data:
      kafka_2_data:
      kafka_3_data:
    ```

    - Docker 엔진에게 *"컴퓨터에 kafka_1_data, kafka_2_data, kafka_3_data라는 이름의 안전한 저장 공간을 각각 하나씩 총 3개 개설해줘"*라고 요청하는 단계
    - 해당 설정으로 얻을 수 있는 것
      - 컨테이너 재생성 시 데이터 보존: 
        -실수로 docker compose down으로 컨테이너를 완전히 싹 지웠다가 docker compose up -d로 다시 띄워도,
        - 카프카는 아까 만들었던 토픽(my-official-topic)과 메시지들을 고스란히 기억하고 정상 작동함
      - 격리성:
        - 1번, 2번, 3번 브로커의 데이터 저장소가 물리적으로 완전히 분리되어 있으므로,
        - 서로 데이터가 꼬이지 않고 완벽한 분산 환경을 유지할 수 있음

- **포트 포워딩(Ports)의 트릭 이해하기**
  - 세 컨테이너의 내부 포트는 모두 `9092`로 동일
  - 호스트(밖)로 뚫어놓은 외부 포트가 다름

    - `kafka-1`: `9092:9092` (밖에서 9092로 접속하면 1번 컨테이너의 9092로 이동)
    - `kafka-2`: `9094:9092` (밖에서 9094로 접속하면 2번 컨테이너의 9092로 이동)
    - `kafka-3`: `9095:9092` (밖에서 9095로 접속하면 3번 컨테이너의 9092로 이동)

  - 호스트 컴퓨터라는 단 하나의 IP 안에서 3대의 서버가 충돌 없이 동시에 살아남고, 외부 개발 환경과 연결되기 위한 포트 분산 아키텍처

- **이 파일 구성의 명확한 장점과 한계점**
  - **장점**
    - **극도의 단순함:**
      - 복잡한 주키퍼 서비스가 통째로 빠져 있음
      - 인프라 자원을 적게 먹고 배포 속도가 매우 빠름

    - **고가용성 테스트 최적화:**
      - 로컬 가상 환경 안에서 완벽하게 3대 규모의 대기업식 실전 분산 처리 및 장애 복구(Fail-over) 실습 가능

  - **실무적 한계점 (아키텍처적 주의 사항)**
    - `KAFKA_ADVERTISED_LISTENERS`가 오직 `localhost` 단일 채널로만 통일되어 있음
      - 이로 인해 **컨테이너 바깥(호스트 컴퓨터의 파이썬 코드)에서 접속할 때는 완벽**하게 작동하지만,
      - 컨테이너 내부 툴끼리 대화하거나
      - 이 도커 가상망 안으로 다른 도커 서비스(예: CDC 커넥터, 웹 UI)가 결합하여 내부망 통신을 하려고 할 때는
      - 주소 꼬임(Loopback) 현상이 발생하여 네트워크 통신이 막히게 되는 구조적 한계를 가짐
      - (추후 심화 파이프라인 확장 시 내부/외부 망 분리 튜닝이 추가로 필요하게 되는 원인)

### 1.2 Python을 이용한 동작 점검

- 호스트 PC(컨테이너 외부) 관점에서 아파치 공식 Kafka 클러스터의 내부 메타데이터 상태를 추적하고 진단하는 실무형 인프라 점검 도구

```python
#//file: "check.py"
from kafka import KafkaAdminClient

try:
    # 호스트 PC에서 아파치 공식 카프카 클러스터 3대에 접속
    admin_client = KafkaAdminClient(
        bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"],
        client_id='my-check-client'
    )
    
    # 내부 low-level 클라이언트 정보 가져오기
    client = admin_client._client
    
    print("\n==============================================")
    print("아파치 공식 Kafka 클러스터가 정상 동작 중입니다!")
    print("==============================================")
    
    # 호스트 이름을 기반으로 현재 연결 가능한 브로커들 출력
    brokers = client.cluster.brokers()
    print("▶ 활성화된 브로커 목록:")
    for b in brokers:
        print(f"   - Broker ID: {b.nodeId} ({b.host}:{b.port})")
        
    print(f"▶ 현재 컨트롤러(KRaft Leader) 노드 ID: {client.cluster.controller}")
    
    # 존재하는 토픽 목록 가져오기
    topics = admin_client.list_topics()
    print(f"▶ 생성되어 있는 토픽 목록: {topics}")
    print("==============================================\n")

except Exception as e:
    print(f"\n카프카 클러스터 연결 실패! 브로커 상태를 확인하세요.")
    print(f"에러 내용: {e}\n")
```

- **코드 설명**
  - **관리자 클라이언트 객체 생성 (`KafkaAdminClient`)**

    ```python
    admin_client = KafkaAdminClient(
        bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"],
        client_id='my-check-client'
    )
    ```

    - **역할:**
      - Kafka 브로커들에게 관리자 명령(토픽 생성/삭제, 클러스터 정보 조회 등)을 내릴 수 있는 통로 개설
    - **동작 원리:**
      - `bootstrap_servers`에 명시된 3개의 주소로 동시에 접속 시도
      - 이 주소들은 `docker-compose.yml`에서 호스트 PC로 포트 포워딩해 놓은 통로
        - 컨테이너 외부 인터페이스를 타고 각각 1번, 2번, 3번 브로커의 심장부로 직접 연결됨
        - 3대 중 단 1대만 살아있어도 메타데이터를 받아올 수 있음

  - **로우레벨 클라이언트 접근 (`admin_client._client`)**

    ```python
    client = admin_client._client
    ```

    - **역할:**
      - 고수준 관리자 기능(Admin API) 뒤에 숨겨진
      - Kafka 클라이언트 라이브러리의 원천 네트워크 엔진 객체(`_client`)에 직접 접근
    - **이유:**
      - 토픽 리스트뿐만 아니라 브로커들의 IP, 포트, 컨트롤러 쿼럼 상태 같은 상세한 물리적 클러스터 정보(Topology)를 직접 덤프하기 위해 이 하부 객체를 꺼내온 것

  - **브로커 생존 여부 및 주소 확인 (`client.cluster.brokers()`)**

    ```python
    brokers = client.cluster.brokers()
    for b in brokers:
        print(f"   - Broker ID: {b.nodeId} ({b.host}:{b.port})")
    ```

    - **역할:**
      - 현재 카프카 클러스터에 참여하여
      - 정상적으로 작동하고 있는 모든 브로커 멤버의 명단을 출력
    - **출력의 의미:**
      - `nodeId`: `KAFKA_NODE_ID`로 설정했던 고유 번호(1, 2, 3)를 의미
      - host와 port: 외부 클라이언트가 이 브로커들과 통신하기 위해 인지하고 있는 네트워크 엔드포인트 주소
      - 3개의 아이디가 다 보인다는 것: 분산 클러스터가 깨지지 않고 완전히 동기화되어 있다는 의미

  - **KRaft 제어권 확인 (`client.cluster.controller`)**

    ```python
    print(f"▶ 현재 컨트롤러(KRaft Leader) 노드 ID: {client.cluster.controller}")
    ```

    - **역할:**
      - 주키퍼가 없는 이 클러스터에서,
      - 현재 어떤 브로커가 메타데이터의 의결권을 쥐고 있는 대장(컨트롤러 리더)인지 확인

    - **출력의 의미:**
      - 결과창의 `BrokerMetadata(nodeId=3...)`: 현재 3번 브로커가 리더가 되어 전체 클러스터의 투표와 브로커 상태 관리를 통제하고 있음을 알려줌

    - **가상 데이터 채널 검증 (`admin_client.list_topics()`)**

      ```python
      topics = admin_client.list_topics()
      print(f"▶ 생성되어 있는 토픽 목록: {topics}")
      ```

      - **역할:**
        - 현재 브로커들의 디스크 내에 논리적으로 생성되어 운영 중인 메시지 저장소(Topic)의 이름들을 배열 형태로 가져옴

      - **출력의 의미:**
        - 컨테이너 안에서 생성한 `'my-official-topic'`이 증발하지 않고 3대의 분산 디스크 파티션 체계 내에 안전하게 안착해 있음을 최종 증명

> - **이 스크립트가 증명하는 것 (의의)**
>   - 터미널에서 복잡하게 `docker exec -it ... /opt/kafka/bin/kafka-topics.sh` 명령어를 치며 고생했던 네트워크 혼선 문제를,
>   - 짧은 파이썬 코드가 **컨테이너 바깥(호스트)의 올바른 통신 경로**를 통해 해결하고 다음 3가지를 검증함
>     1. **네트워크 개통 완료:** 호스트 PC와 Docker 가상 환경 간의 포트 맵핑이 올바르게 설정됨
>     2. **분산 인프라 정상화:** 3대의 브로커가 유실 없이 서로 연결되어 동기화 중
>     3. **KRaft 합의 완료:** 주키퍼 없이도 자체적으로 컨트롤러 리더를 선출하여 클러스터가 자율 제어되고 있음
{: .summary-quote}


## 2. 결국 카프카는 뭐하러 사용하는 도구인가?

- 한마디로 쉽게 정리하면, 카프카는
  - 사방에서 쏟아지는 수많은 실시간 데이터가 길을 잃거나 막히지 않도록,
  - 중앙에서 안전하게 받아다가 필요한 곳에 제때 뚫어주는
  - **초고속 데이터 고속도로(배관)**이다.

- **왜 쓸까? (핵심 목적)**
  - 과거에는 프로그램들이 서로 데이터를 주고받을 때 선을 복잡하게 연결해야 했음
  - 데이터 종류가 많아지면 시스템이 거미줄처럼 꼬여서 하나만 고장 나도 전체가 마비됨<br><br>
  - 카프카는 그 복잡한 선들을 다 끊어버리고,
  - **중앙에서 모든 데이터를 받아주는 거대한 '터미널(중앙 허브)'** 역할을 하기 위해 사용함
  - 데이터를 보내는 쪽도, 받는 쪽도 카프카 하나만 바라보면 되기 때문에 시스템이 극도로 단순하고 안전해짐

- **기존의 다른 데이터 전달 도구들과 다른 카프카의 사기적인 능력 2가지 (차별점)**
  - **서버가 죽어도 데이터가 안 날아감:**
    - 데이터를 메모리가 아니라 **디스크(파일)에 영구 보관**하기 때문에,
    - 받는 서버가 고장 나서 뻗었다가 3일 뒤에 켜져도
    - 밀린 데이터를 처음부터 다시 안전하게 다 받아 갈 수 있음

  - **기차처럼 이어 붙이기 가능:**
    - 데이터가 폭발적으로 늘어나면
    - 서버를 2대, 3대, 100대로 그냥 이어 붙여서 처리량을 무한대로 늘릴 수 있음


> - **최종 요약**
>   - 데이터가 태어나는 순간부터 목적지에 도착할 때까지, 절대로 유실되지 않고 막힘없이 흐르도록 만드는 실시간 데이터 중추 신경망<br><br>
>   - 결국 우리가 6주차 과정에서 하려는 것
>     - **"원본 DB에서 데이터가 바뀌면(CDC) 🡲 카프카 고속도로에 태워서 🡲 최종 저장소(MinIO)까지 단 1초의 끊김과 데이터 유실 없이 실시간으로 흐르게 만드는 인프라"**를 구축하는 것
{: .expert-quote}