---
layout: page
title:  "Trino(Presto) 기반 S3 데이터 SQL 엔진 구축"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-04-02-01_01-TrinoS3SqlEngine
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - Trino-AWS S3(또는 MinIO 등) 연동 🡲 대용량 데이터 레이크하우스의 쿼리 엔진 구축 🡲 현대 데이터 아키텍처의 핵심 패턴
> - MinIO(S3 대용), Trino(쿼리 엔진), Iceberg(테이블 포맷/카탈로그), Python/Pandas(데이터 생성), DuckDB(데이터 검증용)를 조합하여, 
>   - 복잡한 하둡 생태계 도구(Hive) 없이도 깔끔하게 가상 S3 데이터 레이크하우스를 구축하는 실습 과정
{: .common-quote}


## 1. 실습 시나리오 개요

- **주제: 스마트팩토리 실시간 설비 예지보전 및 OLAP 데이터 레이크하우스 구축**
    - 공장 내 여러 생산 라인(Line A, B, C)에 설치된 IoT 센서들이 유기적으로 데이터(온도, 진동 상태 등)를 뿜어내고 있는 상황
    - **기존의 문제점:**
        - 기존 전통적인 하이브(Hive) 방식의 데이터 웨어하우스는 파일 수정이 어렵고,
        - 스키마를 변경하려면 테이블을 통째로 새로 만들어야 해서
        - 실시간 설비 모니터링 및 이력 관리가 불가능함
    - **해결 방안:**
        - 초경량 **Iceberg REST Catalog** 아키텍처 도입 🡲 오브젝트 스토리지(MinIO)를 엔터프라이즈급 데이터 레이크하우스로 진화
        - Trino를 통해 개발자나 분석가들이 익숙한 SQL 표준 문법으로 
        - 대용량 시계열 센서 데이터를 적재, 수정, 스키마 확장, 그리고 과거 이력 추적(타임 트래블)까지 수행하는
        - OLAP 엔진 환경 검증


## 2. 아키텍처 구성 (Architecture)

- **전체 구조**
    - 외부 메타스토어(HMS)와 백엔드 DB의 무거운 종속성을 완전히 제거한 
    - **현업 오픈소스 진영의 가장 모던하고 정석적인 초경량 REST 아키텍처**

- **컴포넌트별 핵심 역할**

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">컴포넌트</th>
        <th style="width: 200px;">기술 스택</th>
        <th style="width: 600px;">실무 및 강의상 핵심 역할</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">Storage Layer</td>
            <td><b>MinIO</b></td>
            <td class="td-left">
                ● S3 API 호환 오브젝트 스토리지<br>
                ● 실제 데이터 파일(`Parquet`)과 테이블 구조를 담은 메타데이터 파일들이 원시 저장되는 물리적 공간
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">Catalog Layer</td>
            <td><b>Iceberg REST Catalog</b></td>
            <td class="td-left">
                ● 하이브 메타스토어(HMS)를 대체하는 <b>"중앙 소스 오브젝트 트러스트(Source of Truth)"</b><br>
                ● 어떤 데이터 파일이 최신 스냅샷인지 루트 포인터를 관리하며,<br>
                &nbsp;&nbsp;&nbsp;도커의 파일 권한 장벽을 네트워크 레이어 수준에서 격리
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">Query Engine</td>
            <td><b>Trino (v435)</b></td>
            <td class="td-left">
                ● MPP(대규모 분산 병렬 처리) SQL 질의 엔진<br>
                ● 사용자가 날린 SQL을 해석하여 Iceberg REST Catalog에서 파일 목록을 받아온 뒤,<br>
                &nbsp;&nbsp;&nbsp;MinIO의 Parquet 파일들을 고속으로 스캔·집계
            </td>
        </tr>
    </tbody>    
</table>
</div>


## 3. 엔드투엔드 데이터 프로세스 (Data Process)

- 데이터가 생성되어 최종 OLAP 분석 리포트로 추출되기까지의 4단계 라이프사이클 흐름

- **1단계: 데이터 적재 및 트랜잭션 (Ingestion & ACID)**

    1. **Client / Python Agent** 또는 **Trino CLI**가 SQL 문 실행 (`INSERT`, `UPDATE`)
    2. **Trino**가 **Iceberg REST Catalog**에 연락 🡲 "새로운 데이터를 쓸 테니 새로운 스냅샷 격리 번호를 달라"고 요청
    3. 데이터는 MinIO 스토리지의 `.../data/` 디렉터리에 압축률과 읽기 성능이 극대화된 **Parquet** 포맷으로 안전하게 분할 저장

- **2단계: 메타데이터 원자적 갱신 (Atomic Commit)**

    1. 파일 쓰기가 완료되면 🡲 Iceberg 엔진이 새로운 메타데이터 파일(`v2.metadata.json`)을 MinIO의 `.../metadata/`에 생성
    2. **Iceberg REST Catalog**가 테이블의 최상위 포인터를 방금 생성된 새 메타데이터 파일 주소로 원자적(Atomic)으로 변경<br>
        🡲 쿼리 도중 데이터가 꼬이는 현상(Dirty Read)이 원천 차단

- **3단계: 쿼리 최적화 및 수행 (Query Parsing & Execution)**

    1. 분석가가 Trino에 `SELECT * FROM ... WHERE temperature > 80` 쿼리 수행
    2. Trino는 메타데이터 파일에 적힌 통계 정보(Min/Max 값 등)를 먼저 읽어 들임
    3. **[핵심 기술]**
        - WHERE 조건에 부합하지 않는 불필요한 파티션과 파일 조각들을
        - 물리적으로 아예 읽지 않고 스킵하는 **Predicate Pushdown** 및 **Partition Pruning**이 발동
        - 수십억 건의 데이터 중 필요한 파일 몇 개만 정확하게 골라 고속 스캔

- **4단계: 스키마 진화 및 타임 트래블 (Schema Evolution & Time Travel)**

    1. 공정에 새로운 센서(예: 진동 센서)가 추가되면 🡲 기존 데이터를 건드리지 않고 `ALTER TABLE` 명령으로 테이블 구조를 실시간 확장 (스키마 진화)
    2. 과거 특정 시점의 사고 이력을 조사해야 할 경우 🡲 테이블의 과거 스냅샷 ID를 역추적 🡲 수개월 전의 데이터 상태 그대로 롤백 조회 (타임 트래블)


## 4. 실습 환경 구성

1. **정석 디렉터리 구조 (Directory Tree)**
    - 호스트 OS(Ubuntu)에서 도커 엔진과의 소유권 마찰을 피하기 위해 반드시 아래 구조로 디렉터리와 설정을 먼저 선점해 두어야 함

        ```text
        ~/AiDALab/Workspaces/DE_Lectures/trino_iceberg/
        ├── docker-compose.yml
        └── configs/
            └── trino/
                └── catalog/
                    └── iceberg.properties  (※ 권한: chmod 755 필수)
        ```

2. **초경량 REST 아키텍처 `docker-compose.yml`**
    - 외부 메타스토어와 백엔드 RDBMS 종속성을 완벽히 걷어내고,
    - `tabulario/iceberg-rest`를 핵심 브리지로 배치하여 인프라를 안정화한 정석 코드

        ```yaml
        version: "3.9"

        services:
        # 1. 분산 SQL 쿼리 엔진 (Trino)
        trino:
            image: trinodb/trino:435
            container_name: demo-trino
            ports:
                - "8080:8080"
            volumes:
            # 호스트에 생성된 카탈로그 설정 폴더를 컨테이너 내부로 바인딩
                - ./configs/trino/catalog:/etc/trino/catalog
            depends_on:
                - iceberg-rest
                - minio

        # 2. Iceberg REST Catalog (중앙 메타데이터 관리 서버)
        iceberg-rest:
            image: tabulario/iceberg-rest:latest
            container_name: demo-iceberg-rest
            ports:
                - "8181:8181"
            environment:
            CATALOG_WAREHOUSE: s3://warehouse/
            CATALOG_IO__IMPL: org.apache.iceberg.aws.s3.S3FileIO
            CATALOG_S3_ENDPOINT: http://demo-minio:9000
            CATALOG_S3_PATH__STYLE__ACCESS: "true"
            CATALOG_S3_ACCESS__KEY__ID: minioadmin
            CATALOG_S3_SECRET__ACCESS__KEY: minioadmin
            AWS_REGION: us-east-1
            depends_on:
                - minio

        # 3. S3 호환 오브젝트 스토리지 (MinIO)
        minio:
            image: minio/minio:RELEASE.2024-01-11T07-46-16Z
            container_name: demo-minio
            command: server /data --console-address ":9001"
            ports:
                - "9000:9000"
                - "9001:9001"
            environment:
            MINIO_ROOT_USER: minioadmin
            MINIO_ROOT_PASSWORD: minioadmin
        ```

3. **카탈로그 연동 설정 (`iceberg.properties`)**
    - Trino 엔진이 `iceberg-rest` 서버를 바라보고, 동시에 실제 데이터 읽기/쓰기를 위해
    - `demo-minio` 스토리지 레이어에 다이렉트로 Native S3 통신을 맺도록 유기적으로 엮어주는 핵심 명세
    - **파일 경로:** `./configs/trino/catalog/iceberg.properties`

        ```properties
        # 커넥터 타입 선언
        connector.name=iceberg

        # [핵심] 외부 REST Catalog 서버를 메타데이터 원천(Source of Truth)으로 지정
        iceberg.catalog.type=rest
        iceberg.rest-catalog.uri=http://demo-iceberg-rest:8181
        iceberg.rest-catalog.warehouse=s3://warehouse/

        # Native S3 가속 성능 극대화 및 엔드포인트 연동 설정
        fs.native-s3.enabled=true
        s3.endpoint=http://demo-minio:9000
        s3.path-style-access=true
        s3.aws-access-key=minioadmin
        s3.aws-secret-key=minioadmin
        s3.region=us-east-1
        ```

4. **리눅스 환경 필수 보안 및 소유권 조치법**
    - 우분투/리눅스 환경에서 도커가 `root` 소유로 폴더를 강제 점령하여 생기는 먹통 현상을 방지하기 위해,
    - 인프라 기동 전에 호스트에서 반드시 수행해야 하는 **정석 3단계 명령셋**

        ```bash
        # 1단계: 도커가 root 계정으로 폴더를 선점하기 전에 계정 권한으로 하위 폴더 선행 생성
        mkdir -p ./configs/trino/catalog

        # 2단계: 설정 파일(iceberg.properties) 작성 후, 소유권이 root로 꼬여있다면 강제 회수
        sudo chown -R $USER:$USER ./configs

        # 3단계: Trino 컨테이너 내부의 자바 일반 유저(UID 1001)가 자유롭게 설정을 읽도록 권한 개방
        chmod -R 755 ./configs
        ```

5. **초기 인프라 가동 및 런타임 검증 명령어**

    ```bash
    # 1. 기존 유령 볼륨/네트워크 캐시 파괴 및 고스트 오펀 컨테이너 제거 후 기동
    docker-compose down -v
    docker-compose up -d --remove-orphans

    # 2. REST Catalog 및 Trino JVM 엔진의 부팅 안정기 확보 (15초 대기)
    sleep 15

    # 3. 인프라 무결성 가동 상태 최종 확인 (3대장이 완벽히 상주해야 함)
    docker ps

    # 4. Trino 분산 SQL 분기점 셸 진입 후 카탈로그 안착 여부 최종 검증
    docker exec -it demo-trino trino --execute "SHOW CATALOGS;"
    ```


## 5. 실습 예제 수행

- "스마트팩토리 설비 예지보전" 시나리오 + 아키텍처 프로세스(ACID, 스키마 진화, 타임 트래블)
- 위 상황에 맞춘 실습 예제 SQL
- Trino CLI(`trino>`) 창에서 그대로 순서대로 수행

1. **스키마 및 초기 설비 마트 테이블 생성 (1단계)**
    - 시나리오에 맞춰 라인별 설비 센서 데이터를 적재할 구조를 정의
    - 대용량 시계열 처리를 위해 `location` 단위로 파티셔닝을 지정

        ```sql
        -- 1. Iceberg REST Catalog 내부에 스마트팩토리 데이터베이스(Schema) 생성
        CREATE SCHEMA iceberg.factory_db;

        -- 2. 예지보전용 설비 센서 로그 테이블 생성 (Parquet 포맷 및 파티션 지정)
        CREATE TABLE iceberg.factory_db.sensor_logs (
            device_id INT,
            location VARCHAR,
            temperature DOUBLE,
            status VARCHAR,
            timestamp TIMESTAMP
        )
        WITH (
            format = 'PARQUET',
            partitioning = ARRAY['location']
        );
        ```

2. **ACID 트랜잭션 및 실시간 상태 업데이트 (2단계)**
    - 공장에서 실시간으로 뿜어져 나오는 초기 데이터 적재(`INSERT`) 프로세스
    - 특정 설비의 이상 징후 발생 시 수행하는 행 단위 수정(`UPDATE`) 프로세스

        ```sql
        -- 1. 생산 라인 A, B의 초기 IoT 센서 데이터 적재
        INSERT INTO iceberg.factory_db.sensor_logs VALUES 
        (101, 'Line_A', 72.5, 'NORMAL', TIMESTAMP '2026-06-17 17:00:00'),
        (102, 'Line_B', 88.1, 'WARN',   TIMESTAMP '2026-06-17 17:01:00'),
        (103, 'Line_A', 69.8, 'NORMAL', TIMESTAMP '2026-06-17 17:02:00');

        -- 2. 전체 데이터 적재 상태 확인
        SELECT * FROM iceberg.factory_db.sensor_logs;

        -- 3. [예지보전 트리거] 102번 설비의 온도가 급상승하여 CRITICAL 상태로 실시간 업데이트
        UPDATE iceberg.factory_db.sensor_logs 
        SET status = 'CRITICAL', temperature = 98.4 
        WHERE device_id = 102;

        -- 4. 변경된 설비 상태 즉시 반영 여부 확인
        SELECT * FROM iceberg.factory_db.sensor_logs WHERE device_id = 102;
        ```

3. **메타데이터 기반 타임 트래블(Time Travel) 조사 (3단계)**
    - 설비 장애가 발생했을 때,
    - Iceberg REST Catalog가 기록한 스냅샷을 역추적하여
    - "장애 발생 전(UPDATE 전)의 정상적인 온도 상태"를 과거 시점으로 되돌려 조회하는 프로세스

        ```sql
        -- 1. 테이블의 격리된 스냅샷 이력(History) 메타데이터 조회
        -- (언제 데이터가 누적되고 UPDATE 되었는지 타임스탬프와 Snapshot ID 확인)
        SELECT committed_at, snapshot_id, parent_id, operation 
        FROM iceberg.factory_db."sensor_logs$snapshots" 
        ORDER BY committed_at DESC;
        ```

        <br>

        > - 조회 결과에서 2번째 행(가장 과거에 생성된 `append` 오퍼레이션의 `snapshot_id`) 값을 복사하여
        > - 아래 `<과거_SNAPSHOT_ID>` 자리에 기입

        ```sql
        -- 2. 102번 설비가 폭사하기 전, 최초 'WARN' 상태(온도 88.1)였던 과거 마트 상태로 타임 트래블 질의
        SELECT * FROM iceberg.factory_db.sensor_logs 
        FOR VERSION AS OF <과거_SNAPSHOT_ID>;
        ```

4. **설비 고도화에 따른 스키마 진화 (4단계)**
    - 공정 고도화로 인해 기존 설비에 "진동(vibration) 센서"가 추가로 장착된 상황을 가정
    - 기존 테이블을 Drop 하지 않고 실시간으로 아키텍처를 확장

        ```sql
        -- 1. 진동 센서 값 적재를 위한 컬럼 동적 추가 (Schema Evolution)
        ALTER TABLE iceberg.factory_db.sensor_logs ADD COLUMN vibration DOUBLE;

        -- 2. 구조가 끊김 없이 확장되었는지 확인
        DESCRIBE iceberg.factory_db.sensor_logs;

        -- 3. 새 진동 센서가 탑재된 신규 라인(Line_C) 데이터 적재
        INSERT INTO iceberg.factory_db.sensor_logs VALUES 
        (104, 'Line_C', 65.2, 'NORMAL', TIMESTAMP '2026-06-17 17:10:00', 0.15);

        -- 4. 최종 통합 데이터 마트 OLAP 조회 
        -- (과거 101~103번 데이터의 vibration 컬럼은 유연하게 NULL 처리되어 하위 호환성 유지)
        SELECT * FROM iceberg.factory_db.sensor_logs;
        ```



## 6. 실습의 핵심 의의

- **'Hadoop-Less' 모던 레이크하우스(Lakehouse)의 완벽한 재현**
    - 과거 빅데이터 인프라의 필수 공식이었던 **Hadoop 생태계(HDFS, 고도로 복잡한 Hive Metastore 및 백엔드 RDBMS)를 단 한 줄도 쓰지 않고**, 
    - S3 호환 오브젝트 스토리지(MinIO)와 가벼운 **REST Catalog** 조합만으로 엔터프라이즈급 데이터 저장소를 구축할 수 있음을 증명
    - 이는 인프라의 복잡성을 획기적으로 낮추고, 클라우드 네이티브 환경으로의 즉각적인 마이그레이션이 가능한 아키텍처

- **스토리지와 컴퓨팅의 완전한 디커플링(Decoupling)**
    - 실제 데이터가 저장되는 물리 레이어(MinIO), 
    - 테이블 구조와 트랜잭션을 관리하는 논리 레이어(Iceberg),
    - 그리고 초고속 질의를 담당하는 컴퓨팅 레이어(Trino)가
    - **완벽하게 독립적으로 분리**되어 작동하는 현대 분산 데이터 시스템의 구성을 확인


## 7. 이 실습을 통해 얻어내야 할 것

- **ACID 트랜잭션의 가치 이해 (`INSERT`, `UPDATE` 검증)**
    - **얻어야 할 것:**
        - 기존의 전통적인 S3 데이터 레이크(Hive 방식)에서는 특정 행(Row)만 골라서 `UPDATE`하거나 `DELETE`하는 것이 불가능
        - 파일 전체를 다시 써야 했기 때문
    - **학습 포인트:**
        - Iceberg 포맷을 적용함으로써,
        - 분산 대용량 환경에서도 데이터가 꼬이거나 깨질 걱정 없이(ACID 무결성 보장)
        - 스마트팩토리 설비의 최신 상태를 **행 단위로 안전하게 실시간 업데이트**할 수 있음을 확인

- **'시간 여행'을 통한 데이터 무결성 및 감사(Audit) 역량 (Time Travel)**
    - **얻어야 할 것:**
        - `sensor_logs$snapshots` 메타데이터를 역추적하여 과거 특정 시점의 데이터로 돌아가는 실습
    - **학습 포인트:**
        - 실무에서 데이터가 오염되거나, 예지보전 시스템이 특정 장애 시점을 분석해야 할 때,
        - **과거 스냅샷 ID 지정만으로 완벽하게 과거 마트 상태를 재현**해 내는 타임 트래블 기능의 강력함과 그 원리(Copy-on-Write) 이해

- **무중단 아키텍처 확장의 유연성 체득 (Schema Evolution)**
    - **얻어야 할 것:**
        - 서비스 운영 중에 `vibration` 컬럼을 동적으로 추가(`ALTER TABLE`)하고,
        - 과거 데이터와 신규 데이터가 공존하는 상태를 확인하는 실습
    - **학습 포인트:**
        - 일반적인 데이터베이스는 스키마를 바꾸면 전체 테이블에 락(Lock)이 걸리거나 마이그레이션 지옥이 펼쳐짐
        - 하지만 Iceberg는 메타데이터 레이어만 가볍게 갱신하므로,
        - **공정 고도화나 비즈니스 요구사항 변경에 무중단으로 유연하게 대처**할 수 있음을 체감

- **리눅스/도커 인프라의 권한(Ownership)과 가상 격리의 본질 이해 (디버깅의 교훈)**
    - **얻어야 할 것:**
        - `root` 소유권 충돌과 경로 바인딩 에러는 실무 및 강의에서 가장 흔히 발생하는 **'리눅스 커널 vs 도커 데몬'의 소유권 마찰**
    - **학습 포인트:**
        - 도커 볼륨 마운트가 무조건 만능이 아님
        - 컨테이너 내부의 자바 일반 유저(`UID 1001`)가 호스트 파일 시스템을 바라볼 때 발생하는 보안 레이어를 정확히 계산해야 함
        - 복잡한 파일 바인딩 대신 **네트워크 기반의 REST Catalog 프로토콜로 소유권 장벽을 우회하는 설계의 안전성** 확인

<br>

> - 이 실습은
>   - 파일 권한의 늪을 건너, 무거운 Hadoop을 버리고,
>   - 오직 **표준 SQL(Trino)과 초경량 카탈로그(Iceberg REST)**만으로 S3 스토리지를 완벽한 엔터프라이즈 데이터 레이크하우스로 진화시키는
>   - '모던 데이터 엔지니어링의 정석'을 배우는 과정
{: .summary-quote}