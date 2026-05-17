---
layout: page
title:  "Docker 기반 Kafka 클러스터 구축하기"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-02-01_01-KafkaClusterDocker
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

Docker 환경에서 Apache Kafka 클러스터를 구축하는 것은 이식성과 확장성 측면에서 매우 효율적입니다. 특히 최신 Kafka 버전에서는 **KRaft(Kafka Raft)** 모드를 지원하여 과거 필수였던 Zookeeper 없이도 클러스터 구성이 가능해졌습니다.

여기서는 실무에서 가장 권장되는 **KRaft 모드 기반의 3-브로커 클러스터** 구축 방법을 중심으로 정리해 드립니다.

---

## 1. Docker Compose 설정 (KRaft 모드)

Zookeeper 없이 브로커 자체적으로 컨트롤러 역할을 수행하는 KRaft 모드 설정 파일입니다. 이 구조는 아키텍처가 단순하며 관리 포인트가 적다는 장점이 있습니다.

```yaml
services:
  kafka-1:
    image: bitnami/kafka:latest
    container_name: kafka-1
    ports:
      - "9092:9092"
    environment:
      - KAFKA_CFG_NODE_ID=1
      - KAFKA_CFG_PROCESS_ROLES=controller,broker
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
    volumes:
      - kafka_1_data:/bitnami/kafka

  kafka-2:
    image: bitnami/kafka:latest
    container_name: kafka-2
    ports:
      - "9094:9092"
    environment:
      - KAFKA_CFG_NODE_ID=2
      - KAFKA_CFG_PROCESS_ROLES=controller,broker
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9094
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
    volumes:
      - kafka_2_data:/bitnami/kafka

  kafka-3:
    image: bitnami/kafka:latest
    container_name: kafka-3
    ports:
      - "9095:9092"
    environment:
      - KAFKA_CFG_NODE_ID=3
      - KAFKA_CFG_PROCESS_ROLES=controller,broker
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9095
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
      - KAFKA_KRAFT_CLUSTER_ID=abcdefghijklmnopqrstuv
    volumes:
      - kafka_3_data:/bitnami/kafka

volumes:
  kafka_1_data:
  kafka_2_data:
  kafka_3_data:

```

---

## 2. 핵심 설정 값 설명

* **KAFKA_CFG_PROCESS_ROLES:** 해당 서버의 역할을 정의합니다. `broker`는 데이터 저장, `controller`는 클러스터 관리를 담당하며, 위 설정처럼 둘 다 부여할 수 있습니다.
* **KAFKA_CFG_CONTROLLER_QUORUM_VOTERS:** 컨트롤러 역할을 수행할 노드들의 리스트입니다. KRaft 모드에서 가장 중요한 합의 알고리즘을 위한 설정입니다.
* **KAFKA_KRAFT_CLUSTER_ID:** 클러스터를 식별하는 고유 ID입니다. 모든 브로커가 동일한 ID를 가져야 하나의 클러스터로 묶입니다.
* **ADVERTISED_LISTENERS:** 외부 클라이언트(Java, Python 앱 등)가 브로커에 접속할 때 사용하는 주소입니다. Docker 외부에서 접근할 경우 `localhost`와 매핑된 포트번호를 정확히 기재해야 합니다.

---

## 3. 구축 및 검증 절차

### 1) 실행

작성한 `docker-compose.yml` 파일이 있는 위치에서 명령어를 실행합니다.

```bash
docker-compose up -d

```

### 2) 토픽 생성 및 확인

컨테이너 내부로 접속하여 클러스터가 정상 작동하는지 토픽을 생성해 봅니다.

```bash
# kafka-1 컨테이너 접속
docker exec -it kafka-1 /bin/bash

# 'test-topic' 생성 (복제본 3개, 파티션 3개)
kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092 --replication-factor 3 --partitions 3

# 토픽 상세 정보 확인
kafka-topics.sh --describe --topic test-topic --bootstrap-server localhost:9092

```

---

## 4. 구축 시 주요 고려사항 (Best Practices)

1. **데이터 영속성 (Volumes):** Docker 컨테이너는 삭제 시 내부 데이터가 사라집니다. 반드시 호스트 시스템의 디렉토리나 Docker Volume을 매핑하여 데이터를 보존해야 합니다.
2. **리소스 할당:** Kafka는 JVM 기반이므로 메모리 사용량이 많습니다. `docker-compose` 설정에 `deploy.resources.limits`를 사용하여 메모리와 CPU 제한을 두는 것이 좋습니다.
3. **네트워크 분리:** 운영 환경에서는 Kafka 전용 Docker Network를 생성하여 다른 애플리케이션과의 간섭을 최소화하고 보안을 강화하십시오.
4. **모니터링 도구:** 구축 후에는 **Kafka UI** 또는 **CMAK(Cluster Manager for Apache Kafka)** 같은 GUI 도구를 함께 Docker로 띄우면 관리가 훨씬 수월해집니다.

이 설정을 바탕으로 개발 환경을 구성해 보시면, Kafka의 파티션 복제 및 부하 분산 메커니즘을 직접 실습하며 이해하는 데 큰 도움이 될 것입니다.

혹시 특정 프로그래밍 언어(Python, Java 등)와의 연동 예시나 성능 최적화를 위한 커널 파라미터 튜닝 정보가 필요하신가요?