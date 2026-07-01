---
layout: page
title:  "CDC(Debezium) DB 변경분 실시간 캡처"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-04-01_01-KafkaCdcRealtimeCapture
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 데이터 엔지니어링의 패러다임 변화

### 1.1 데이터베이스(DB)

- **DB: 데이터가 생성되는 가장 근원적인 공간**
    - 전통적인 IT 아키텍처에서 데이터베이스는 단순히 '데이터를 저장하는 창고'
    - 스트리밍 아키텍처 관점에서는 모든 비즈니스 사건(Fact)이 기록되는 '최초의 탄생지'

    - **가장 순수한 원천 데이터(Single Source of Truth):**
        - 애플리케이션 로그는 개발자의 가공이나 누락이 있을 수 있지만,
        - DB의 트랜잭션 로그(MySQL의 Binlog 등)는
            - 데이터의 생성(C), 수정(U), 삭제(D)라는 비즈니스의 모든 순간이
            - 단 하나의 오차도 없이 기록되는 가장 신뢰할 수 있는 근원

    * **배치(Batch)의 한계 극복:**
        - 과거에는 이 근원적인 공간에서 데이터를 가져오기 위해 밤마다 대량의 쿼리를 날려 데이터를 읽어옴(Batch)
        - 이는 DB에 엄청난 부하이며 낮 동안 일어난 변화를 즉시 알 수 없다는 한계가 있음


### 1.2 아파치 카프카(Kafka)

- **Kafka: 실시간 스트리밍의 심장**
    - 카프카가 스트리밍 아키텍처의 '심장'으로 비유되는 이유는
        - 생명체의 심장이 피를 받아 전신으로 뿜어내듯,
        - **모든 실시간 이벤트를 받아 전사 시스템으로 중단 없이 순환시키기 때문**

    - **동력원 (Throttling & Buffer):**
        - 사방에서 데이터가 폭발적으로 쏟아져 들어올 때,
        - 뒤에서 데이터를 받아 처리하는 하부 시스템(분석 엔진, 데이터 레이크)이 지치거나 쓰러지지 않도록
        - 카프카가 중앙에서 압력을 조절하고 데이터를 안전하게 보관해줌

    - **전신으로의 박동 (Decoupling):**
        - 심장에서 나간 피가 신체 각 기관으로 가듯,
        - 카프카에 안착한 데이터 스트림은 실시간 대시보드, 검색 엔진, 인공지능 모델, 정적 저장소 등 필요한 모든 곳으로 동시에 공급(Pub/Sub)됨


### 1.3 CDC(Debezium)

- **CDC(Debezium): 이 둘을 유기적으로 연결하는 핵심 과정**
    - 아무리 근원(DB)이 훌륭하고 심장(Kafka)이 튼튼해도, 이 둘을 연결하는 혈관이 막히거나 비효율적이면 실시간 아키텍처는 완성되지 않음
    - Debezium 기반의 CDC는 이 연결을 '유기적(Organic)'으로 만듦

    - **무부하, 무간섭의 연결:**
        - Debezium은 DB에 직접 무거운 `SELECT` 쿼리를 보내지 않음
        - DB가 내부적으로 백업과 복구를 위해 조용히 쓰고 있는 트랜잭션 로그 파일만 가볍게 가로채어 읽음
        - DB 입장에서는 자신이 감시당하고 있다는 사실조차 모를 정도로 부하가 없는, 시스템에 '유기적으로 녹아든' 방식

    - **실시간 이벤트화:**
        - Debezium은 DB 테이블의 단순한 스냅샷을 넘겨주는 것이 아니라,
        - "X번 유저의 데이터가 기존 [A]에서 [B]로 수정되었다"라는 완벽한 전/후 맥락을 갖춘 JSON 이벤트 스트림으로 변환하여 카프카에 던져줌

    - 이 CDC 파이프라인이 성공적으로 안착해야
        - 기업은 "어제 무슨 일이 일어났는가?"를 분석하는 수준을 넘어,
        - "지금 이 순간 무슨 일이 일어나고 있는가?"에 즉각적으로 반응하는
        - 진정한 실시간 이벤트 기반 기업(Event-Driven Enterprise)으로 진화할 수 있음



## 2. CDC의 정의 및 개요

- **실시간 데이터 캡처(CDC, Change Data Capture)**
    - 데이터 아키텍처에서 데이터의 원천지(주로 OLTP 데이터베이스)로부터 발생한 변경 사항을 실시간으로 감지하고 가로채어,<br>
    하부 시스템(데이터 레이크, 데이터 웨어하우스, 다른 마이크로서비스 등)으로 즉시 동기화하는 핵심 기술 파이프라인<br><br>
    - 데이터베이스 내의 데이터가 생성(Create), 수정(Update), 삭제(Delete)될 때 발생하는 변경 이벤트를<br>
    실시간으로 추적하여 다른 시스템에 전달하는 일련의 기술적 프로세스

- **왜 CDC인가? (기존 방식과의 비교)**
    - 기존의 전통적인 데이터 통합
        - **쿼리 기반 폴링(Query-based Polling)이나 ETL 배치(Batch) 방식**
            - 예: 매일 밤 특정 시간에 `SELECT * FROM table WHERE updated_at > ...`와 같은 쿼리를 날려 데이터를 퍼 올리는 방식

        - 치명적인 한계
            - **소스 DB 부하:**
                - 서비스 중인 대규모 OLTP 데이터베이스에 무거운 대량의 쿼리 전송 🡲 시스템 성능 저하
            - **삭제 데이터(Delete) 추적 불가:**
                - 데이터베이스에서 행(Row)이 아예 삭제된 경우 🡲 쿼리 기반으로는 무엇이 지워졌는지 알아내기가 매우 어려움
            - **실시간성 부재:**
                - 데이터가 변경된 시점과 분석 시스템에 반영되는 시점 사이에 수 시간에서 하루 이상의 시차 발생

    - **CDC의 방식**
        - DB에 쿼리를 날리지 않고,
        - 데이터베이스 내부의 엔진 수준에서 일어나는 변화를 실시간으로 가로챔


## 3. CDC 개념의 태동과 발전

- **1990년대 ~ 2000년대 초반: 트리거(Trigger)와 전용 솔루션의 등장**
    - 초기 CDC
        - 데이터베이스 내부의 **트리거(Trigger)** 기능을 활용하여 구현
        - 특정 테이블에 변경이 생기면 별도의 '변경 기록 테이블(Shadow Table)'에 데이터를 복사해 두는 방식
        - 역시 원본 DB의 트랜잭션 성능 저하의 원인이 됨
    - 이후
        - 오라클(Oracle) 등 거대 벤더사들이 자체 데이터베이스 복제를 위해 내부 Redo Log를 직접 읽는 전용 솔루션들을 개발하기 시작
            - 예: 오라클 골든게이트-Oracle GoldenGate의 모태가 되는 기술들
            - Redo Log: 
                - 오라클 RDBMS에서 데이터베이스 변경 이력을 기록하는 파일
                - 장애 발생 시 원래 상태로 복구하는 데 사용

- **2010년대 중반: 마이크로서비스(MSA)와 이벤트 기반 아키텍처(EDA)의 폭발**
    - 시스템 구조가 거대한 단일 아키텍처(Monolith)에서 수많은 마이크로서비스(MSA)로 분할<br>
    🡲 "각 서비스가 가진 DB 간의 실시간 데이터 동기화"가 엔지니어들의 가장 큰 숙제가 됨
    - 대용량 분산 메시지 브로커인 **Apache Kafka**가 스트리밍의 표준으로 자리 잡음<br>
    🡲 CDC 기술은 단순한 'DB 대 DB 복제'를 넘어 '이벤트 스트리밍 인프라의 첫 단추'로 재정의

- **2010년대 후반 ~ 현재: 오픈소스 CDC(Debezium)의 대중화와 쿼럼리스화**
    - 수억 원에 달하는 외산 상용 솔루션의 대안
        - 레드햇(RedHat)이 주도하는 오픈소스 프로젝트인 Debezium(디비지움)이 등장하면서 CDC 기술의 대중화 시작
    - 현대의 CDC
        - 인메모리 기반 데이터 레이크 구축
        - 실시간 캐시 갱신(Cache Invalidation)
        - 데이터 탐지(FDS) 등
    - 실시간성이 극대화된 비즈니스의 중추 신경망 역할을 담당


## 4. CDC의 기반 기술 및 핵심 메커니즘

### 4.1 로그 기반 CDC

- **트랜잭션 로그(Transaction Log)** 또는 WAL(Write-Ahead Log)
    - 관계형 데이터베이스(RDB)가 시스템이 불시 다운되었을 때 데이터를 복구하기 위해,<br>
    모든 쓰기 작업(C, U, D)을 실제 디스크에 쓰기 전에 **최초의 사실 확인용 로그 파일**에 먼저 기록하는 것
    - RDBMS 별 종류
        - **MySQL:** 바이너리 로그 (Binlog)
        - **PostgreSQL:** Write-Ahead Log (WAL)
        - **Oracle:** Redo Log

- 로그 기반 CDC는
    - 데이터베이스에 쿼리를 날리는 것이 아니라, **로그 파일의 꼬리(Tail)를 실시간으로 스트리밍하여 읽어내는 방식**
    - DB 엔진 내부에서 독자적으로 도는 파일을 직접 읽음 🡲 원본 DB의 SQL 실행 엔진에 부하를 주지 않음

### 4.2 복제 프로토콜 우회

- CDC 도구(예: Debezium)는
    - 데이터베이스가 지원하는 **'복제 서버(Replica Node/Slave Node)' 프로토콜을 그대로 흉내 내어 DB에 접속**
    - 데이터베이스 마스터는
        - CDC 도구를 '또 하나의 백업용 세컨더리 서버'로 인식
        - 트랜잭션 로그가 갱신될 때마다 변경분 이벤트를 CDC 도구 쪽으로 실시간 밀어내기(Push) 처리

### 4.3 복합 데이터 구조화

- 로그 기반 CDC가 가로챈 원시 바이너리 데이터는 사람이 읽을 수 없음
- CDC 엔진은 이를 분석하여 고도의 구조화된 **이벤트 페이로드(Payload)** 구조(JSON 또는 Avro 형태)로 변환

    ```json
    {
        "op": "u",                           // Operation 형식 (u: Update)
        "ts_ms": 1719468520000,              // 이벤트 발생 시각
        "before": { "id": 1, "score": 80 },  // 변경 전 데이터 상태
        "after":  { "id": 1, "score": 95 },  // 변경 후 데이터 상태
        "source": { "db": "school", "table": "students" } // 메타데이터
    }
    ```

- 이 `before`와 `after` 구조 덕분에 🡲 이벤트를 받는 하부 시스템은 데이터가 어떻게 변했는지에 대한 '맥락(Context)'을 제공받게 됨


## 5. CDC 관련 기술 생태계 (Ecosystem)

- CDC는 홀로 작동하지 않음
- 주로 다음과 같은 현대 스트리밍 생태계 기술들과 결합하여 거대한 파이프라인을 이룸

    <div class="info-table">
    <table>
        <thead>
            <th style="width: 170px;">기술 분류</th>
            <th style="width: 230px;">대표적인 오픈소스/솔루션</th>
            <th style="width: 560px;">파이프라인 내 역할 및 의미</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">CDC 캡처 엔진</td>
                <td><b>Debezium</b>, Canal,<br>Oracle GoldenGate</td>
                <td class="td-left">
                    DB의 트랜잭션 로그를 직접 바라보며 이벤트를 추출하고 포맷팅하는 심장부 역할
                </td>
            </tr>
            <tr>
                <td class="td-rowheader">스트림 이송 프레임워크</td>
                <td><b>Kafka Connect</b></td>
                <td class="td-left">
                    CDC 엔진을 플러그인 형태로 탑재하여 분산 처리, 태스크(Task) 스케일 아웃,<br>
                    오프셋 백업을 담당하는 골격
                </td>
            </tr>
            <tr>
                <td class="td-rowheader">중앙 메시지 브로커</td>
                <td><b>Apache Kafka</b>,<br>Apache Pulsar</td>
                <td class="td-left">
                    CDC가 뿜어낸 이벤트 스트림을 디스크에 순서대로 영구 저장하고,<br>
                    수많은 소비자가 멀티캐스팅으로 퍼갈 수 있게 하는 고속도로
                </td>
            </tr>
            <tr>
                <td class="td-rowheader">스트림 연산 엔진</td>
                <td>Apache Flink,<br>Spark Streaming</td>
                <td class="td-left">
                    CDC 이벤트를 실시간으로 조인(Join), 윈도우 집계(Window Aggregate)하여<br>
                    실시간 분석 지표를 도출하는 가공 창고
                </td>
            </tr>
        </tbody>    
    </table>
    </div>

<br>

> - **요약**
>   - 실시간 데이터 캡처(CDC)는
>       - 데이터를 더 이상 고여있는 정적 자산이 아닌, "발생하는 즉시 사방으로 순환해야 하는 살아있는 혈류"로 바라보는 패러다임의 산물
>       - 데이터베이스의 가장 깊숙한 **트랜잭션 로그 파싱 기술**을 기반으로 작동하며, **Kafka 에코시스템**과의 유기적인 결합을 통해
>       - 데이터 소실과 시스템 부하 없이 기업 전체의 전사 인프라를 실시간 이벤트 기반 아키텍처로 바꾸는 핵심 기술
{: .summary-quote}



## 6. 실습

### 6.1 실시간 CDC의 철학과 Debezium의 이해

- **데이터 수집 패러다임의 변화: Polling vs CDC**
    - 전통적인 데이터 수집 방식과 현대적인 로그 기반 CDC 방식은 데이터를 바라보는 관점 자체가 다름

    - **쿼리 기반 폴링(Query-based Polling)의 작동 방식과 한계**
        - **메커니즘:**
            - 스케줄러(예: 10분 주기)가 DB에 `SELECT * FROM 테이블 WHERE 수정일시 > '마지막_수집일시'` 형태의 쿼리를 지속적으로 던져 변경분을 가져옴

        - **한계 1 (DB 성능 저하):**
            - 수집 주기마다 인덱스를 타고 대량의 데이터를 조회<br> 🡲 서비스 중인 OLTP DB에 심각한 성능 병목을 유발

        - **한계 2 (Hard Delete 추적 불가):**
            - 어플리케이션이 DB에서 데이터를 아예 삭제(`DELETE FROM...`)해 버리면<br> 🡲 다음 폴링 시점에는 데이터가 존재하지 않으므로 무엇이 지워졌는지 추적할 방법이 없음

        - **한계 3 (중간 상태 손실):**
            - 수집 주기(예: 10분) 사이에 데이터가 'A 🡲 B 🡲 C'로 빠르게 두 번 수정<br> 🡲 폴링 방식은 최종 상태인 'C'만 가져오고 'B'라는 중간 비즈니스 흐름은 유실

    - **로그 기반 CDC(Log-based CDC)의 필요성**
        - DB에 직접적인 데이터 조회 쿼리를 단 한 번도 전송하지 않고,
        - 데이터베이스가 내부 복구 및 백업용으로 기록하는 **트랜잭션 로그**를 뒤에서 조용히 읽어옴
        - 삭제 데이터(`DELETE`)는 물론, 아주 짧은 찰나에 일어난 모든 변경 이벤트를
        - 실시간(밀리초 단위)으로 한 건도 놓치지 않고 완벽한 시퀀스로 수집할 수 있음

- **Debezium 기술 핸드북: 작동 원리와 엔진 구조**
    - **디비지움(Debezium):**
        - 복잡한 DB 트랜잭션 로그를 파싱하여 이벤트 스트림으로 변환해 주는 가장 대표적인 오픈소스 분산 CDC 엔진

    - **데이터베이스별 트랜잭션 로그 명칭 및 역할**
        - **MySQL (Binlog - Binary Log):**
            - 생성, 수정, 삭제, 테이블 구조 변경(DDL) 등 모든 가공 명령이 바이너리 형태로 기록되는 로그
            - CDC를 위해서는 문장 기준(`STATEMENT`)이 아닌 데이터 행 단위의 변화를 기록하는 `ROW` 포맷 설정이 필수

        - **PostgreSQL (WAL - Write-Ahead Log):**
            - 디스크의 데이터 페이지를 수정하기 전에 변경 사항을 먼저 기록하는 로그
            - Debezium은 PostgreSQL의 논리적 복제(Logical Replication) 스트리밍 프로토콜을 활용하여 WAL 로그를 이벤트로 수신

    * **Debezium의 내부 아키텍처 작동 원리**
        1. **복제 노드 위장:**
            - Debezium 커넥터는 DB 마스터에게 자신을 '백업용 복제 서버(Replica/Slave)'라고 속이고 접속 세션을 연결
        2. **로그 스트리밍:**
            - DB 마스터는 새로운 트랜잭션(C, U, D)이 발생할 때마다
            - 트랜잭션 로그의 바이너리 스트림을 Debezium에게 실시간으로 전송
        3. **이벤트 변환:**
            - Debezium은 이 바이너리 데이터를 가공하여 가독성이 높은 구조화된 JSON(또는 Avro) 메시지로 변환한 뒤,
            - Kafka 토픽에 순서대로 적재

    - **Debezium 이벤트 구조 분석 (페이로드 해부)**
        - Debezium이 생성하여 Kafka로 보내는 JSON 메시지는
        - 변경 전/후 상태와 인프라 메타데이터를 모두 포함하고 있기 때문에 매우 거대하고 정교함

        - Debezium 샘플 페이로드 구조 (Update 발생 시)

            ```json
            {
                "schema": { ... }, 
                "payload": {
                    "op": "u",
                    "ts_ms": 1719648000000,
                    "before": {
                        "id": 1004,
                        "name": "홍길동",
                        "grade": "GOLD"
                    },
                    "after": {
                        "id": 1004,
                        "name": "홍길동",
                        "grade": "VIP"
                    },
                    "source": {
                        "version": "2.4.0.Final",
                        "connector": "mysql",
                        "name": "my-db-server",
                        "ts_ms": 1719647999000,
                        "db": "inventory",
                        "table": "customers"
                    }
                }
            }
            ```

    - **핵심 메타데이터 필드 설명**
        - **`op` (Operation):**
            - 데이터가 어떤 형태로 변했는지 단 한 글자로 명시함
            - `c`: Create, `u`: Update, `d`: Delete, `r`: Read/Snapshot

        - **`before`:**
            - 이벤트가 발생하기 **직전**의 데이터 상태
            - `op`가 `c`(생성)일 때는 `null`이 됨

        - **`after`:**
            - 이벤트가 발생한 **직후**의 최종 데이터 상태
            - `op`가 `d`(삭제)일 때는 `null`이 됨

        - **`source`:**
            - 이 이벤트가 어떤 DB 서버, 어떤 스키마, 어떤 테이블에서 기인했는지,
            - 그리고 DB 내부에서 로그가 찍힌 시각(`ts_ms`)은 언제인지 등의 인프라 정보를 투명하게 제공

- **실습 및 확인 사항**
    - 로컬 개발 환경(Ubuntu 및 가상환경)에서 Docker로 임시 구동한 MySQL을 CDC가 가능한 상태로 직접 튜닝하는 실습 과정<br><br>

    - **[실습 과정 1] 환경 구성**
        - **Docker Compose 활용**
            - **`kafka-1, 2, 3`**:
                - 아파치 공식 이미지 기반의 3-브로커 클러스터 (KRaft 모드)
            - **`mysql-cdc`**:
                - Debezium cnf 설정이 주입되어 트랜잭션 로그(`ROW` 포맷)를 남기는 원천 데이터베이스

        - **통합 환경 설정 파일 (`docker-compose.yml`)**
            - 작업 디렉토리(예: `~/workspace/mykafka`) 및 docker-compose.yml 파일 생성

                ```yaml
                services:
                # -----------------------------------------------------------------
                # 1. CDC 원천 데이터베이스 (MySQL 8.0)
                # -----------------------------------------------------------------
                mysql-cdc:
                    image: mysql:8.0
                    container_name: mysql-cdc
                    ports:
                    - "3306:3306"
                    environment:
                    - MYSQL_ROOT_PASSWORD=root_pass!
                    - MYSQL_DATABASE=inventory
                    command:
                    - --server-id=223344
                    - --log-bin=mysql-bin
                    - --binlog_format=ROW
                    - --binlog-row-image=FULL
                    - --binlog_expire_logs_seconds=604800
                    volumes:
                    - mysql_data:/var/lib/mysql

                # -----------------------------------------------------------------
                # 2. 아파치 공식 이미지 기반 Kafka 3-브로커 클러스터 (KRaft)
                # -----------------------------------------------------------------
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
                mysql_data:
                kafka_1_data:
                kafka_2_data:
                kafka_3_data:
                ```

                - 의도된 구성적 트릭 (Command 주입)
                    - MySQL 공식 이미지는 원래 CDC 옵션이 꺼져 있음
                        - 이를 켜기 위해 원래는 외부 `cnf` 파일을 마운트해야 함
                        - 이번 실습에서는 Docker Compose의 `command` 옵션으로 설정을 직접 주입

                    - 파일 하나만 실행하면 **따로 파일 설정을 수정할 필요 없이 즉시 CDC 학습이 가능한 상태의 세팅**이 완료

        - **실습 가동 및 초기 환경 검증 절차**

            1. **인프라 구동**

                ```bash
                docker compose up -d
                ```

                - `docker compose ps` 명령어로 4개의 컨테이너(`mysql-cdc`, `kafka-1, 2, 3`)가 모두 `Up` 상태인지 확인

            2. **MySQL 내부 진입 및 CDC 환경 검증**
                - 호스트 터미널에서 MySQL 컨테이너 내부로 접속하여 CDC 관련 시스템 변수가 제대로 활성화되었는지 쿼리로 확인

                    ```bash
                    # 호스트 PC에서 실행하여 MySQL 컨테이너 내부 쉘 접속
                    docker exec -it mysql-cdc mysql -uroot -proot_pass!
                    ```

                
                - **확인 사항(CDC 가능 상태 환경 설정값 검증법)**
                    - 접속된 MySQL 프롬프트(`mysql>`)에서 다음을 검증
                    - 설정 및 계정 생성이 끝난 후, DB 터미널창에서 인프라가 완벽히 구성되었는지 확인

                    - **검증 명령어 1: Binlog 포맷 확인**

                        ```sql
                        SHOW VARIABLES LIKE 'binlog_format';
                        ```

                        - **합격 기준 출력 결과:**

                            ```text
                            +---------------+-------+
                            | Variable_name | Value |
                            +---------------+-------+
                            | binlog_format | ROW   |
                            +---------------+-------+
                            ```

                        - **설명:**
                            - `Value` 자리에 반드시 `ROW`가 출력되어야 함
                            - 만약 `STATEMENT`나 `MIXED`가 출력된다면 1.1절에서 배운 중간 상태 유실 및 복잡한 페이로드 구성이 불가능하므로 탈락

                    - **검증 명령어 2: Binlog 이미지 범위 확인**

                        ```sql
                        SHOW VARIABLES LIKE 'binlog_row_image';
                        ```

                        - **합격 기준 출력 결과:**

                            ```text
                            +------------------+-------+
                            | Variable_name    | Value |
                            +------------------+-------+
                            | binlog_row_image | FULL  |
                            +------------------+-------+
                            ```

                        - **설명:**
                            - `FULL`로 지정되어 있어야만 데이터가 수정되었을 때
                            - Debezium 페이로드에 `before`와 `after` 구조가 완벽하게 다 담겨 출력될 수 있음


            3. **실습용 테스트 테이블 및 CDC 계정 생성**
                - Debezium이 읽어갈 대상 데이터와 전용 권한 계정을 설정(MySQL 창에 그대로 입력)

                    ```sql
                    -- 실습용 데이터베이스 전환
                    USE inventory;

                    -- 테스트용 고객 테이블 생성
                    CREATE TABLE customers (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(50),
                        grade VARCHAR(20)
                    );

                    -- 디비지움 연동용 전용 사용자 계정 생성 및 복제 권한 부여
                    CREATE USER 'debezium_user'@'%' IDENTIFIED BY 'dbz_pass123!';
                    GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'debezium_user'@'%';
                    FLUSH PRIVILEGES;
                    ```

                    - CREATE USER 'debezium_user'@'%' IDENTIFIED BY 'dbz_pass123!';
                        - 디비지움 전용 ID(debezium_user)와 비밀번호(dbz_pass123!)로 사용자 계정 생성
                        - 외부 컨테이너(kafka-connect)에서 MySQL로 접속할 때 보안을 위해 최고 관리자(root) 계정을 주는 대신, 
                        - 제한된 접근망(%: 어디서나 접속 가능)을 가진 전용 파이프라인 계정을 하나 개설

                    - GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'debezium_user'@'%';
                        - 권한별 세부 역할:
                            - REPLICATION SLAVE (핵심):
                                - 트랜잭션 바이너리 로그 스트림을 실시간으로 수신할 수 있는 권한 (CDC의 핵심)
                                    - 디비지움이 단순 조회용 프로그램이 아니라
                                    - "백업용 복제 서버(Slave)니까 최신 트랜잭션 로그(Binlog)를 실시간으로 보내줘"라고
                                    - 마스터 DB에 요청할 수 있는 결정적인 권한
                            - SELECT:
                                - 디비지움이 처음 가동될 때,
                                - 기존에 이미 테이블에 쌓여있던 과거 데이터의 스냅샷을 읽어오기 위한 읽기 권한
                            - RELOAD & REPLICATION CLIENT:
                                - `REPLICATION CLIENT`: 현재 빈로그의 위치와 상태를 모니터링할 수 있는 권한
                                - 마스터 DB의 로그 파일 위치가 몇 번 번호인지 모니터링하고,
                                - 필요할 때 로그 상태를 리로드하여 동기화 포인트를 맞추는 권한
                                
                            - SHOW DATABASES:
                                - 디비지움 내부 엔진이 어떤 데이터베이스가 존재하는지 카탈로그 조회
                    - FLUSH PRIVILEGES;
                        - MySQL 서버를 껐다 켜지 않아도,
                        - 방금 부여한 복제 및 읽기 권한들이 시스템 엔진에 실시간으로 즉각 반영되도록 뇌를 동기화하는 명령
                            - "방금 설정한 유저 권한 메모리를 즉시 새로고침(적용)해라."


    - **[실습 과정 2] MySQL CDC 전용 설정 파일 작성**
        - MySQL 엔진이 모든 행 단위 변경 분을 누락 없이 바이너리 로그에 쓰도록 프로젝트 디렉토리에 구성 파일(`mysql_cdc.cnf`) 생성        

            ```ini
            [mysqld]
            # 1. 고유한 서버 ID 설정 (클러스터 내 중복 불가)
            server-id=223344

            # 2. 바이너리 로그 활성화 및 파일 저장 이름 지정
            log-bin=mysql-bin

            # 3. 중요: CDC 생성을 위한 필수 설정 (행 단위 기록)
            binlog_format=ROW

            # 4. 행 변경 시 이전/이후 값을 로그에 모두 기록하도록 설정
            binlog_row_image=FULL

            # 5. 빈로그 보존 기간 (초 단위 - 7일 지정)
            binlog_expire_logs_seconds=604800
            ```

        - 이번 실습에서는 docker-compose.yml 파일에 넣어두었으므로 따로 실행하지 않아도 됨


### 6.2 Docker 기반 CDC 파이프라인 인프라 빌드

- **실습 내용:**
    - 기존 Kafka 3-브로커 파일에 `mysql`과 `kafka-connect` 서비스를 추가 정의하고 구동
    - 이전 실습에서의 Kafka Connect(Debezium 플러그인 탑재 버전)를 유기적으로 결합하여, 실전 스트리밍 인프라의 뼈대 구축

- **Kafka Connect 프레임워크의 구조적 이해**
    - Debezium은 단독으로 구동되는 프로그램이 아니라, **Kafka Connect**라는 프레임워크 위에서 동작하는 '플러그인(Plugin)'

    - **Kafka Connect의 역할:**
        - 외부 시스템과의 데이터 연동을 총괄하는 표준 엔진룸
        - 내부적으로 분산 처리, 가용성 보장, 오프셋(내가 로그를 어디까지 읽었는지) 관리를 대신 처리함

    - **태스크(Task)와 워커(Worker):**
        - **Worker:** Connect 엔진이 실행되는 컨테이너 프로세스
        - **Task:** 워커 안에서 실제로 특정 테이블의 로그를 읽어오는 스레드 단위
        - 트래픽이 늘어나면 이 Task를 여러 브로커나 컨테이너로 분산시켜 병렬 처리를 달성함

- **멀티 컨테이너 네트워크 아키텍처 (통신 흐름)**
    - 이번 인프라 빌드에서 가장 중요한 것은 "도커 내부망(Internal)과 외부망(External)의 완벽한 분리"
    - 전체 컴포넌트가 결합된 물리적 아키텍처는 다음과 같이 유기적으로 연결됨

        1. **내부 통신 (Docker Network):**
            - `kafka-connect` 컨테이너와 `mysql-cdc` 컨테이너는
            - 브로커들끼리 통신하는 내부 포트인 `9093`(컨트롤러 및 내부 브로커 포트)망을 주소 삼아 서로를 찾아감
        2. **외부 통신 (Host PC 포트포워딩):**
            - 개발자(호스트)가 실행할 파이썬 스크립트나 관리 인터페이스는
            - 외부 포트인 `9092`, `9094`, `9095`를 통해 클러스터에 접속


- **엔드 투 엔드(End-to-End) 인프라 빌드 실습 과정**
    - 기존 1단계 설정에 `kafka-connect(Debezium 탑재)`를 추가하여, CDC 파이프라인을 위한 최종 인프라 정의서를 완성하고 구동하는 실습

    - **[1단계] 최종 `docker-compose.yml` 파일 작성**
        - 작업 디렉토리(`~/workspace/mykafka`)의 기존 파일 삭제
        - 아래의 **인프라 파일** 작성

        ```yaml
        services:
            # 1. CDC 원천 데이터베이스
            mysql-cdc:
                image: mysql:8.0
                container_name: mysql-cdc
                ports:
                    - "3306:3306"
                environment:
                    - MYSQL_ROOT_PASSWORD=root_pass!
                    - MYSQL_DATABASE=inventory
                command:
                    - --server-id=223344
                    - --log-bin=mysql-bin
                    - --binlog-format=ROW
                    - --binlog-row-image=FULL
                    - --binlog_expire_logs_seconds=604800
                volumes:
                    - mysql_data:/var/lib/mysql

            # 2. Kafka 3-브로커 클러스터 (내/외부 리스너 완벽 분리)
            kafka-1:
                image: apache/kafka:latest
                container_name: kafka-1
                ports:
                    - "9092:9092"
                environment:
                    - KAFKA_NODE_ID=1
                    - KAFKA_PROCESS_ROLES=broker,controller
                    - KAFKA_LISTENERS=INTERNAL://:19092,EXTERNAL://:9092,CONTROLLER://:9093
                    - KAFKA_ADVERTISED_LISTENERS=INTERNAL://kafka-1:19092,EXTERNAL://localhost:9092
                    - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                    - KAFKA_INTER_BROKER_LISTENER_NAME=INTERNAL
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
                    - KAFKA_PROCESS_ROLES=broker,controller
                    - KAFKA_LISTENERS=INTERNAL://:19092,EXTERNAL://:9094,CONTROLLER://:9093
                    - KAFKA_ADVERTISED_LISTENERS=INTERNAL://kafka-2:19092,EXTERNAL://localhost:9094
                    - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                    - KAFKA_INTER_BROKER_LISTENER_NAME=INTERNAL
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
                    - KAFKA_PROCESS_ROLES=broker,controller
                    - KAFKA_LISTENERS=INTERNAL://:19092,EXTERNAL://:9095,CONTROLLER://:9093
                    - KAFKA_ADVERTISED_LISTENERS=INTERNAL://kafka-3:19092,EXTERNAL://localhost:9095
                    - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
                    - KAFKA_INTER_BROKER_LISTENER_NAME=INTERNAL
                    - KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER
                    - KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka-1:9093,2@kafka-2:9093,3@kafka-3:9093
                    - KAFKA_LOG_DIRS=/var/lib/kafka/data
                volumes:
                    - kafka_3_data:/var/lib/kafka/data

            # 3. Kafka Connect 엔진
            kafka-connect:
                image: debezium/connect:2.4
                container_name: kafka-connect
                ports:
                    - "8083:8083"
                environment:
                    # 내부망 전용 포트인 19092 채널을 지정
                    - BOOTSTRAP_SERVERS=kafka-1:19092,kafka-2:19092,kafka-3:19092
                    - GROUP_ID=1
                    - CONFIG_STORAGE_TOPIC=my_connect_configs
                    - OFFSET_STORAGE_TOPIC=my_connect_offsets
                    - STATUS_STORAGE_TOPIC=my_connect_statuses
                    - CONFIG_STORAGE_REPLICATION_FACTOR=1
                    - OFFSET_STORAGE_REPLICATION_FACTOR=1
                    - STATUS_STORAGE_REPLICATION_FACTOR=1
                    - KEY_CONVERTER=org.apache.kafka.connect.json.JsonConverter
                    - VALUE_CONVERTER=org.apache.kafka.connect.json.JsonConverter
                depends_on:
                    - kafka-1
                    - kafka-2
                    - kafka-3

        volumes:
            mysql_data:
            kafka_1_data:
            kafka_2_data:
            kafka_3_data:
        ```

    - **[2단계] 전체 인프라 가동 명령**
        - 작성 완료 후 터미널 백그라운드 모드로 인프라 부팅

            ```bash
            docker compose up -d
            ```

    - **[3단계] 컨테이너 생존 및 구동 상태 확인**
        - 모든 컨테이너가 정상 궤도에 올라왔는지 프로세스 스냅샷 검증

            ```bash
            docker compose ps
            ```

        - **합격 기준 점검:**
            - `mysql-cdc`, `kafka-1`, `kafka-2`, `kafka-3`, `kafka-connect` 총 5개의 서비스가 모두 **`Up`** 상태여야 함


- **핵심 확인 및 검증 사항 (REST API 진단)**
    - Kafka Connect는 GUI 화면이 없는 대신, **`8083` 포트를 통한 HTTP REST API**로 모든 상태를 보고하고 제어함
    - 인프라 빌드가 완벽한지 호스트 PC(Ubuntu 쉘)에서 다음 두 가지 명령어로 최종 진단

    - **검증 1: Kafka Connect 엔진 생존 여부 확인**
        - Connect 서버가 정상 작동하고 인터페이스를 열었는지 확인하기 위하여 메인 엔드포인트 호출

            ```bash
            curl -s http://localhost:8083/
            ```

        - **합격 기준 출력:**

            ```json
            {"version":"2.4.0.Final","commit":"...","kafka_cluster_id":"..."}
            ```

        - **의미:**
            - Connect 호스트가 정상 응답을 보냈으며,
            - 내부적으로 아파치 카프카 클러스터 ID를 올바르게 인지하고 연결을 수립했다는 뜻

    - **검증 2: Debezium MySQL 플러그인 로드 여부 확인 (★가장 중요)**
        - Connect 엔진 내부 덤프 폴더에 Debezium MySQL 커넥터가 플러그인 형태로 이식되어 작동 대기 중인지 리스트를 확인

            ```bash
            curl -s http://localhost:8083/connector-plugins | grep -o '"class":"[^"]*"'
            ```

        - **합격 기준 출력에 반드시 포함되어야 할 항목:**

            ```text
            "class":"io.debezium.connector.mysql.MySqlConnector"
            ```

        - **의미:**
            - 이 클래스명이 나타나면 Connect 엔진이 Debezium 기술을 완벽하게 수용했다는 뜻
            - 다음 세션에서 이 엔진에 JSON 조작 명령문만 전송하리면 즉시 MySQL 트랜잭션 로그를 캡처할 준비가 끝난 것


### 6.3 Debezium 커넥터 등록/실시간 이벤트 스트리밍

- 이 단계에서는 전 세션에서 빌드한 `kafka-connect` 인프라에 REST API를 통해 선언적 JSON 명세서를 전달하여 커넥터를 활성화하고,
- MySQL에서 발생하는 데이터 변경(C, U, D) 이벤트가 카프카 토픽으로 실시간 발행되는 과정을 수행함


- **선언적 파이프라인(Declarative Pipeline) 설정의 이해**
    - Kafka Connect와 Debezium의 가장 큰 장점
        - 자바나 파이썬 코드를 짤 필요 없이,
        - **JSON 포맷의 설정 파일만 생성하여 HTTP POST 요청으로 Connect 엔진에 등록하면 파이프라인이 구동된다는 점**

    - 엔진은 주입된 JSON 명세서를 파싱하여 내부적으로 필요한 스레드(Task)를 할당하고 DB 트랜잭션 로그 위치(오프셋)를 초기화함

- **Debezium의 토픽 자동 생성 및 네이밍 규칙**
    - Debezium 커넥터가 성공적으로 등록되면,
    - 소스 데이터베이스의 테이블들을 감시하다가
    - 변경이 감지되는 순간 **카프카 브로커 내부에 토픽을 자동으로 개설**
    - 이때 무작위로 만드는 것이 아니라 철저한 네이밍 규칙을 따름

    - **기본 네이밍 공식:**
        - `{topic.prefix}.{database.name}.{table.name}`
    - **본 실습의 예시:**
        - JSON 설정에서
            - 접두어(Prefix)를 `cdc`로 지정
            - 데이터베이스명이 `inventory`
            - 테이블명이 `customers`라면
        - 카프카 내부에 생성되는 토픽의 이름은 `cdc.inventory.customers`가 됨
    - **의의:**
        - 하부 시스템(소비자)은 전사 시스템의 수많은 테이블 중
        - 자신이 구독해야 할 테이블의 데이터 스트림을
        - 이 토픽 이름을 통해 명확하게 찾아갈 수 있음


- **엔드 투 엔드(End-to-End) 커넥터 등록 및 스트리밍 실습 과정**
    - 호스트 PC(Ubuntu 쉘) 환경에서 순서대로 진행하는 실전 가이드

    - **[1단계] Debezium MySQL 커넥터 등록 JSON 명세서 작성**
        - 작업 디렉토리(`~/workspace/mykafka`)에 `register-mysql-cdc.json` 파일 생성

            ```json
            {
            "name": "mysql-cdc-connector",
            "config": {
                "connector.class": "io.debezium.connector.mysql.MySqlConnector",
                "tasks.max": "1",
                "database.hostname": "mysql-cdc",
                "database.port": "3306",
                "database.user": "debezium_user",
                "database.password": "dbz_pass123!",
                "database.server.id": "184054",
                "topic.prefix": "cdc",
                "database.include.list": "inventory.customers",
                "schema.history.internal.kafka.bootstrap.servers": "kafka-1:19092,kafka-2:19092,kafka-3:19092",
                "schema.history.internal.kafka.topic": "schemahistory.inventory",
                "include.schema.changes": "false"
            }
            }
            ```

            - 파라미터 설명
                - `"database.hostname"`
                    - 도커 내부 네트워크망 명칭인 `mysql-cdc`를 지정하여 컨테이너 간 직접 통신 유도
                - `"database.user" / "password"`
                    - 세션 1에서 생성하고 REPLICATION 권한을 주었던 전용 계정 정보 입력
                - `"topic.prefix"`
                    - 카프카 토픽 이름의 맨 앞에 붙을 고유 식별자
                - `"table.include.list"`
                    - 전체 DB를 다 읽어오면 비효율적
                    - 특정 데이터베이스의 특정 테이블(`inventory.customers`)만 타겟팅하여 감시하도록 제한
                    
                    > - database.include.list로 작성하면 inventory 까지 작성(데이터베이스 레벨까지)
                    > - table.include.list로 작성하면 inventory.customers 까지 작성(테이블 레벨까지)

                - `"schema.history.internal.kafka.topic"`
                    - DDL(테이블 구조 변경) 이벤트를 추적하여
                    - 내부에 보관할 Kafka 메타데이터 토픽 지정


    - **[2단계] REST API를 통한 커넥터 등록 명령 실행**
        - 작성한 JSON 파일을 Kafka Connect 엔진(`localhost:8083`)으로 전송하여 커넥터 실행

            ```bash
            curl -X POST -H "Content-Type: application/json" --data @register-mysql-cdc.json http://localhost:8083/connectors
            ```

    - **[3단계] 커넥터 구동 상태 확인 (HTTP GET)**
        - 등록된 커넥터가 에러 없이 정상적으로 `RUNNING` 상태를 유지하고 있는지 파이프라인 헬스체크 수행

            ```bash
            curl -s http://localhost:8083/connectors/mysql-cdc-connector/status | grep -o '"state":"[^"]*"'
            ```

    - **합격 기준 출력:**

        ```text
        "state":"RUNNING"
        "state":"RUNNING"
        ```

    - **의미:**
        - 첫 번째는 커넥터 총괄 상태,
        - 두 번째는 실제 로그를 파싱하는 태스크(Task)의 상태
        - 둘 다 `RUNNING`이 떠야 정상


- **실시간 이벤트 스트리밍 결과 검증**
    - 소스 데이터베이스에 임의의 조작(C, U, D)을 가했을 때 카프카 토픽에 실시간으로 데이터가 인입되는지 검증

    - **1단계: 카프카 토픽 감시(Consume) 쉘 가동**
        - 새로운 터미널 창을 열고,
        - 아파치 공식 카프카의 내장 스크립트를 이용하여
        - 자동 생성될 토픽(`cdc.inventory.customers`)을 실시간으로 모니터링
            - 처음에는 토픽이 만들어지기 전이거나 데이터가 없어 대기 상태로 멈춰 있음

            ```bash
            docker exec -it kafka-1 /opt/kafka/bin/kafka-console-consumer.sh \
            --bootstrap-server kafka-1:19092 \
            --topic cdc.inventory.customers \
            --from-beginning
            ```
    

    - **2단계: 다른 터미널에서 MySQL 데이터 조작 (Event 발생)**
        - 기존 터미널 창을 통해 MySQL 컨테이너에 접속한 뒤,
        - `INSERT` 문과 `UPDATE` 문 실행

            ```bash
            # MySQL 접속
            docker exec -it mysql-cdc mysql -uroot -proot_pass! inventory
            ```

            ```sql
            -- [이벤트 1] 데이터 삽입 (Create)
            INSERT INTO customers (name, grade) VALUES ('양석환', 'GOLD');

            -- [이벤트 2] 데이터 수정 (Update)
            UPDATE customers SET grade = 'VIP' WHERE name = '양석환';
            ```

    - **3단계: 최종 스트리밍 결과 검증**
        - 데이터를 입력 및 수정하는 즉시, 대기 중이던 **1단계의 카프카 토픽 감시 쉘 창**을 확인
        - 화면에 디비지움이 바이너리 로그를 실시간 가로채 가공한 JSON 페이로드가 출력되는 것을 확인

            ```json
            /* INSERT 실행 시 카프카에 찍히는 데이터 결과 구조 */
            {
            "payload": {
                "op": "c",
                "before": null,
                "after": { "id": 1, "name": "양석환", "grade": "GOLD" },
                "source": { "db": "inventory", "table": "customers" }
            }
            }

            /* UPDATE 실행 시 카프카에 찍히는 데이터 결과 구조 */
            {
            "payload": {
                "op": "u",
                "before": { "id": 1, "name": "양석환", "grade": "GOLD" },
                "after": { "id": 1, "name": "양석환", "grade": "VIP" },
                "source": { "db": "inventory", "table": "customers" }
            }
            }
            ```

        - `op` 값이 각각 생성(`c`), 수정(`u`)으로 실시간 매핑되는지 확인
        - 수정(`u`) 이벤트가 발생했을 때
            - `before`에는 과거 등급인 `GOLD`가,
            - `after`에는 변경된 등급인 `VIP`가 공존하여
        - 비즈니스 데이터의 역사적 맥락이 완벽히 스트리밍되고 있는지 대조



### 6.4 파이썬 기반 CDC 이벤트 파싱 및 실시간 가공

- 카프카 토픽(`cdc.inventory.customers`)에 쌓여 있는 복잡한 Debezium JSON 스트림을 파이썬 코드로 소비(Consume)하여,
- 적재된 원천 데이터의 알맹이(`payload`)를 솎아내고 연산하는 실전 스트리밍 애플리케이션을 구축

- **Debezium 페이로드 껍질 벗기기 (Data Flattening)**
    - Debezium이 발행하는 메시지는
        - 이벤트를 안전하게 전달하기 위해
        - 스키마 정보와 다양한 메타데이터 메트릭이 중첩 구조로 감싸져 있어
        - 매우 거대함
    - 실시간 스트리밍 분석 애플리케이션이 이 데이터를 매번 전체 파싱하면 오버헤드가 발생<br>
    🡲 애플리케이션 진입부에서 가볍게 실제 알맹이 데이터인 `payload` 블록만 신속하게 추출(Flattening)하는 기법이 필요

- **`op` 필드를 활용한 비즈니스 분기 제어 (Event-Driven Routing)**
    - CDC 스트림의 정수는 변경 유형을 나타내는 **`op` 메타데이터**에 있음
    - 파이썬의 조건문(`if / elif`) 구조를 활용해 각 이벤트 타입에 따른 정교한 실시간 비즈니스 파이프라인을 분기 제어해야 함

        - **`op == 'c'` (Create):**
            - 신규 데이터 유입 프로세스 가동
            - 예: 웰컴 이메일 발송, 신규 캐시 등록
        - **`op == 'u'` (Update):**
            - 데이터 변경 전/후 비교 연산
            - 예: 중요 고객 등급 업그레이드 탐지 및 푸시 알림
        - **`op == 'd'` (Delete):**
            - 데이터 삭제 맥락 복원 및 동기화
            - 예: 타 시스템의 연관 레코드 소프트 삭제 전환


- **엔드 투 엔드(End-to-End) 파이썬 실실형 앱 개발 실습 과정**
    - 호스트 PC(Ubuntu)의 활성화된 가상환경 `(mykafka)`에서 코드를 작성하고 실시간 매핑 상태를 검증하는 과정

    - **[1단계] CDC 전용 파이썬 실시간 가공 스크립트 작성**

        - 작업 디렉토리(`~/workspace/mykafka`)에 `cdc_processor.py` 파일 생성

            ```python
            import json
            from kafka import KafkaConsumer

            # 1. KafkaConsumer 인프라 정의
            # 외부 호스트 포트 스펙 반영 및 JSON 직렬화 해제 처리
            consumer = KafkaConsumer(
                "cdc.inventory.customers"
                , bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"]
                , group_id="cdc-analytics-processor-group"
                , auto_offset_reset="latest" # 켠 순간부터 흐르는 트랜잭션 실시간 추적
                , enable_auto_commit=True
                # 수정된 부분--------------------------------------------------
                # , value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                # -------------------------------------------------------------
            )

            print("========================================================")
            print("CDC 실시간 이벤트 프로세서가 가동되었습니다.")
            print("MySQL 변경 로그를 실시간 가공 분석합니다... (Ctrl+C 종료)")
            print("========================================================\n")

            try:
                # 실시간 인메모리 스트리밍 루프 구동
                for message in consumer:
                    # 수정된 부분--------------------------------------------------
                    if message.value is None:
                        print("--> [Tombstone] 데이터가 삭제되어 로그 컴팩션 마킹이 유입되었습니다. 스킵하겠습니다.")
                        continue
                    
                    event_msg = json.loads(message.value.decode('utf-8'))
                    # -------------------------------------------------------------
                    
                    # 데이터 유효성 검사 및 껍질 벗기기 (payload 추출)
                    if not event_msg or 'payload' not in event_msg:
                        continue
                        
                    payload = event_msg['payload']
                    op_type = payload.get('op') # 데이터 연산 타입 추출 ('c', 'u', 'd')
                    
                    print(f"[이벤트 포착] 파티션: {message.partition} | 오프셋: {message.offset}")
                    
                    # -----------------------------------------------------------
                    # [분기 1] 신규 데이터 생성 (Create)
                    # -----------------------------------------------------------
                    if op_type == 'c':
                        after = payload.get('after')
                        print(f"    [회원가입] 신규 고객 정보가 유입되었습니다.")
                        print(f"    고객번호: {after['id']} | 이름: {after['name']} | 등급: {after['grade']}")
                        
                    # -----------------------------------------------------------
                    # [분기 2] 기존 데이터 수정 (Update)
                    # -----------------------------------------------------------
                    elif op_type == 'u':
                        before = payload.get('before')
                        after = payload.get('after')
                        
                        print(f"    [정보수정] 기존 데이터에 변경 내역이 감지되었습니다.")
                        
                        # 비즈니스 로직 가공: 등급(grade)의 변화를 실시간으로 대조 연산
                        if before['grade'] != after['grade']:
                            print(f"    [등급 변동 특이사항 파악] 고객명 [{after['name']}]")
                            print(f"    등급 조정 내역: {before['grade']} ──> {after['grade']}")
                        else:
                            print(f"    일반 필드 변경: {before} -> {after}")
                            
                    # -----------------------------------------------------------
                    # [분기 3] 데이터 물리 삭제 (Delete)
                    # -----------------------------------------------------------
                    elif op_type == 'd':
                        before = payload.get('before')
                        print(f"    [탈퇴/삭제 위험 감지] 원천 데이터베이스에서 행이 영구 삭제되었습니다.")
                        print(f"    유실 데이터 복원 맥락 -> 기존 고유ID: {before['id']} | 대상자: {before['name']}")
                        
                    print("-" * 65)

            except KeyboardInterrupt:
                print("\n프로세서 애플리케이션을 안전하게 종료합니다.")
            finally:
                consumer.close()
            ```

    - **[2단계] 파이썬 가공 프로세서 스크립트 실행**
        - 터미널을 열고 가상환경 내부에서 방금 만든 스크립트를 업로드

            ```bash
            python cdc_processor.py
            ```

        - 최초 실행 시, 메시지가 인입되기 전까지 정상적으로 대기 상태 쉘을 유지해야 합격

- **최종 연동 테스트 및 실시간 검증 시나리오**

    - **1단계: 멀티 터미널 레이아웃 준비**
        - **터미널 1:**
            - MySQL 데이터 조작 창 (`docker exec -it mysql-cdc mysql -uroot -proot_pass! inventory`)
        - **터미널 2:**
            - 파이썬 가공 엔진 가동 창 (`python cdc_processor.py`)

    - **2단계: 소스 DB 데이터 가공 및 파이썬 콘솔 반응 대조**

        1. 신규 회원 유입 테스트 (Create)

            - **터미널 1 (MySQL) 입력:**

                ```sql
                INSERT INTO customers (name, grade) VALUES ('이순신', 'SILVER');
                ```

            - **터미널 2 (Python) 출력 결과:**

                ```text
                [이벤트 포착] 파티션: 0 | 오프셋: 2
                    [회원가입] 신규 고객 정보가 유입되었습니다.
                    고객번호: 2 | 이름: 이순신 | 등급: SILVER
                -----------------------------------------------------------------
                ```

        2. 비즈니스 규칙 변경 테스트 (Update)

            - **터미널 1 (MySQL) 입력:**

                ```sql
                UPDATE customers SET grade = 'GOLD' WHERE name = '이순신';
                ```

            - **터미널 2 (Python) 출력 결과 (등급 대조 연산 작동 확인):**
                
                ```text
                [이벤트 포착] 파티션: 0 | 오프셋: 3
                    [정보수정] 기존 데이터에 변경 내역이 감지되었습니다.
                    [등급 변동 특이사항 파악] 고객명 [이순신]
                    등급 조정 내역: SILVER ──> GOLD
                -----------------------------------------------------------------
                ```

        3. Hard Delete 데이터 추적 테스트 (Delete)

            - **터미널 1 (MySQL) 입력:**

                ```sql
                DELETE FROM customers WHERE name = '이순신';
                ```

            - **터미널 2 (Python) 출력 결과 (삭제된 과거 정보 완벽 매핑):**

                ```text
                [이벤트 포착] 파티션: 0 | 오프셋: 4
                    [탈퇴/삭제 위험 감지] 원천 데이터베이스에서 행이 영구 삭제되었습니다.
                    유실 데이터 복원 맥락 -> 기존 고유ID: 2 | 대상자: 이순신
                -----------------------------------------------------------------
                ```

            <br>
            
            > - **Delete 시, 오류 발생의 원인**
            >   - Debezium CDC는 MySQL에서 데이터가 삭제(DELETE)되면 카프카 토픽에 연속으로 2개의 메시지를 발행함
            >       - Delete 메시지: op 코드가 'd'이고, before에 삭제 전 데이터가 들어있으며 after가 null인 JSON 메시지
            >       - Tombstone 메시지: 카프카의 로그 컴팩션(Log Compaction) 기능을 위해 Key값만 남기고 Value를 완전히 null(None)로 비워서 보내는 빈 메시지<br><br>
            >   - 기존 예제 코드의 value_deserializer는 들어오는 모든 메시지의 Value에 값이 채워져 있다고 가정하고 x.decode('utf-8')를 수행
            >       - 하지만 삭제 직후 유입되는 Tombstone 메시지는 Value가 없기 때문에(x가 None)<br>
            >       🡲 변환하는 시점에 NoneType object has no attribute 'decode'를 발생시키며 즉시 크래시가 나는 것<br><br>
            > - **수정 방법**
            >   - consumer = KafkaConsumer(...) 생성 시,
            >       - value_deserializer=lambda x: json.loads(x.decode('utf-8')) 구문 삭제
            >       - message.value가 None이 아닐 때에만 수행하도록 함
            >   - 스트리밍 루프가 시작될 때,
            >       - 루프 속의 메시지를 확인하여 None이 발생했다면 🡲 Tombstone 메시지 발행으로 판단하여 스킵
            >       - 루프 속의 메시지가 None이 아니라면 🡲 디코딩 수행<br>
            >
            >           <div class="info-table"><table><tr><td style="width:700px;" class="td-left">
            >           if message.value is None:<br>
            >           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print("--> [Tombstone] 데이터가 삭제되어 로그 컴팩션 마킹이 유입되었습니다. 스킵하겠습니다.")<br>
            >           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;continue<br><br>
            >           event_msg = json.loads(message.value.decode('utf-8'))
            >           </td></tr></table></div>
            {: .common-quote}
