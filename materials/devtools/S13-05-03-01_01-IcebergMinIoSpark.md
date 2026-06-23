---
layout: page
title:  "Spark+Iceberg+MinIO 연결 및 대용량 데이터 분석 준비"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-03-01_01-IcebergMinIoSpark
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}
 
 

> - **Spark(연산) + Apache Iceberg(테이블 포맷) + MinIO(오브젝트 스토리지) 조합**
>   - 오픈소스 기반으로 완벽하게 구현한 저비용·고성능 차세대 데이터 레이크하우스(Lakehouse)
{: .common-quote}

## 1. 시스템 연동의 필요성

### 1.1 왜 필요한가?

- **기존 아키텍처의 한계 극복**
    - 기존 데이터 웨어하우스(DW), 초기 데이터 레이크(HDFS + Hive) 등
    - 대규모 AI 연구, 스마트팩토리 로그 분석, 주식 데이터 백테스팅 등 현대적인 데이터 요구사항을 충족하기에 명확한 한계가 있음<br><br>

    - **컴퓨팅과 스토리지의 결합 한계**
        - 기존 HDFS 방식
            - 데이터가 늘어나면 연산 노드도 같이 늘려야 함
            - 비용 낭비가 심함

    - **데이터 정합성(Consistency) 문제**
        - 전통적인 데이터 레이크
            - 오브젝트 단위로 저장
            - 수많은 파일이 동시에 쓰이거나 수정될 때 데이터가 깨지거나(Dirty Read) 트랜잭션 보장이 안됨

    - **클라우드 종속성(Lock-in) 탈피**
        - AWS S3, Athena, Glue 같은 완전 관리형 서비스
            - 사용은 편하지만, 비용 통제가 어려움
            - 온프레미스(On-Premise) 환경이나 하이브리드 환경으로의 이전이 불가능<br><br>

- **세 가지 요소의 조합은**
    - 컴퓨팅(Spark)과 스토리지(MinIO)를 완전히 분리
    - 그 사이를 고성능 테이블 포맷(Iceberg)으로 연결
    - 완전히 독립적이고 제어 가능한 데이터 플랫폼을 만들기 위해 필요함


### 1.2 어디에 필요한가?

- 이 아키텍처는 다음의 상황에서 최적의 효율을 발휘함
    - 대용량 시계열 데이터
    - 실시간 스트리밍과 배치가 공존하는 환경
    - AI 모델 학습을 위한 데이터 파이프라인

- **스마트팩토리 및 IoT 센서 데이터 수집/분석**
    - 초단위로 쏟아지는 수만 개의 공정 센서 로그를 실시간으로 MinIO에 적재 🡲 Spark로 정제 🡲 Iceberg 테이블로 관리
    - 과거 특정 시점의 공정 상태를 추적할 때 유용함

- **금융/주식 데이터 분석 및 백테스팅**
    - 수십 년 치의 틱(Tick) 데이터, 호가창 데이터 저장
    - 대규모 퀀트 분석이나 AI 기반 주식 투자 모델을 학습시킬 때,
    - 필요한 기간의 데이터만 빠르게 스캔(Partition Evolution 기능 활용)하여 연산할 수 있음

- **AI/ML 연구를 위한 데이터 레이크하우스**
    - 정형 데이터뿐만 아니라 비정형 데이터까지 모두 MinIO에 모음
    - Spark를 통해 머신러닝 피처(Feature)를 생성·관리
    - 모델의 학습 데이터 버전 관리 시 🡲 Iceberg의 타임 트래블 기능이 핵심 역할을 수행


## 2. 연동함으로써 얻을 수 있는 이득

- **비용 효율성 및 인프라 통제권 확보 (MinIO)**
    - **S3 API 호환 온프레미스 구축**
        - AWS S3와 100% 호환되는 오브젝트 스토리지를 사내 서버(On-Premise)에 구축할 수 있음
        - 클라우드 아웃바운드 트래픽 비용 및 스토리지 비용을 획기적으로 감축

    - **유연한 스케일 아웃**
        - 데이터 용량이 늘어나면 저렴한 디스크를 탑재한 스토리지 노드만 증설하면 됨
        - 비용 효율적

- **완전한 ACID 트랜잭션 및 타임 트래블 (Apache Iceberg)**
    - **데이터 신뢰성 보장**
        - Spark가 대량의 데이터를 쓰다가 실패해도,
        - Iceberg의 원자성(ACID) 보장 덕분에 이전의 안전한 상태를 유지 🡲 **Upsert/Delete 완벽 지원**

    - **시간 여행(Time Travel) 및 롤백**
        - `SELECT * FROM table FOR SYSTEM_TIME AS OF ...` 구문을 통해 특정 과거 시점의 데이터 스냅샷을 조회할 수 있음
            - AI 모델 학습 시 데이터 재현성 확보
            - 버그로 인한 데이터 오염 시 복구에 결정적 역할

    - **스키마 및 파티션 에볼루션**
        - 테이블 삭제 없이 파티션 레이아웃을 실시간으로 변경할 수 있음(컬럼 추가 등) 🡲 시스템 운영 공수 극적 감소

- **고성능 분산 연산 (Apache Spark)**
    - **Iceberg와의 시너지 (Hidden Partitioning)**
        - Spark가 쿼리를 날릴 때 Iceberg가 쿼리에 불필요한 파일들을 메타데이터 단계에서 미리 걸러냄(Data Skipping)<br>
            🡲 수조 건의 데이터 중 필요한 부분만 수 초 만에 찾아냄 🡲 연산 효율 극대화

    - **다양한 언어 및 에코시스템 지원**
        - Python(PySpark), Java 등 익숙한 언어로 고성능 분산 처리 가능
        - MLlib를 통한 대규모 분산 AI 학습으로 연계 가능

> - 이 조합은 빅테크 기업(Netflix, Uber 등)이 직면했던 대규모 데이터 관리의 난제들을 오픈소스 조합만으로 풀 수 있게 해줌
>   - 인프라 비용은 오브젝트 스토리지(MinIO) 수준으로 낮추고
>   - 성능과 데이터 안정성은 최고 수준의 DW(Snowflake, BigQuery 등) 못지않게 끌어올릴 수 있음
{: .summary-quote}


## 3. Spark + Apache Iceberg + MinIO 연결 환경 설정

<div class="insert-image" style="text-align: left;">
    <img src="/materials/devtools/images/S13-05-03-01_01-001.png" style="width: 60%;">
</div>


### 3.1 전체 구조 이해

- **Iceberg는 DB가 아님**

    > - 스스로 데이터를 담는 저장소(DB)가 아니라, 오브젝트 스토리지에 흩어진 대용량 데이터 파일(Parquet 등)을 마치 하나의 관계형 DB 테이블처럼 부릴 수 있게 돕는 '내비게이션(메타데이터 관리자)'
    > - "데이터 파일들은 밖에 따로 두고, 나는 그 파일들의 족보(메타데이터)와 히스토리(스냅샷)만 완벽하게 관리할 테니, Spark 같은 연산 엔진은 나를 통해서 안전하게 데이터를 읽고 쓰라"고 중개해 주는 **고성능 테이블 포맷 아키텍처**
    {: .common-quote}

    - **전통적인 DB(MySQL, PostgreSQL)와의 차이점**
        - 전통적인 DB(MySQL, PostgreSQL 등)
            - 데이터 저장소와 메타데이터(카탈로그, 스키마, 인덱스 등)가 하나의 시스템 내부에 결합되어 있음
            - DB가 데이터의 생성, 수정, 삭제를 직접 통제
        - Apache Iceberg
            - 데이터 자체를 내부에 저장하지 않음
            - 오직 데이터가 '어디에', '어떤 형태로' 저장되어 있는지에 대한 정보(메타데이터)만 정교하게 관리하는 추상화 계층

    - **역할의 분리 (Iceberg vs 실제 데이터 파일)**
        - 실제 데이터는 파일로 저장
            - 실제 대용량 레코드들은 성능과 압축률이 뛰어난 Parquet, ORC, Avro 같은 오픈소스파일 포맷 형태로 외부 스토리지(MinIO, S3 등)에 직접 저장
        - Iceberg는 메타데이터 구조만 관리
            - Iceberg는 그 파일들 위에 얹어져서 다음과 같은 메타데이터 구조를 유지
                - **Table Schema:** 테이블의 컬럼 이름, 타입 등 구조를 정의
                - **Manifest:** 실제 데이터 파일들(Parquet 등)의 위치와 통계 정보(최소/최대값 등)를 낱개 단위로 기록
                - **Snapshot:** 특정 시점에 어떤 Manifest 파일들이 유효했는지를 기록 🡲 '시간 여행(Time Travel)'이나 트랜잭션 관리가 가능하게 함

- **최종 구조**
    - <span style="color: darkred;">**Spark 🡲 Iceberg 🡲 MinIO**</span>
    - 실제 저장은 MinIO가 담당

    <div class="insert-image">
        <img src="/materials/devtools/images/S13-05-03-01_01-002.png" style="width: 90%;"><br><br>
        (현재 업계에서 가장 많이 사용하는 Lakehouse 구조 중 하나)
    </div>

### 3.2 구성요소별 역할

- **Spark**
    - 역할: 연산 수행, ETL, DataFrame 처리, SQL 실행
    - 예

        ```python
        df.writeTo("demo.customer").create()
        ```

- **Iceberg**
    - 역할: 테이블 정의, Schema, Partition, Snapshot, Time Travel 관리
    - 예

        ```sql
        CREATE TABLE demo.customer
        ```

- **MinIO**
    - 역할: Parquet 저장, Manifest 저장, Metadata 저장 🡲 즉 실제 데이터 저장소


### 3.3 MinIO 먼저 구축

1. **docker-compose 추가**

    ```yaml
    minio:
        image: minio/minio
        container_name: minio
        command: server /data --console-address ":9001"
        environment:
            MINIO_ROOT_USER: admin
            MINIO_ROOT_PASSWORD: password123
        ports:
            - "9000:9000"
            - "9001:9001"
        volumes:
            - ./minio/data:/data
    ```

    - 기존 파일에 추가

        ```yaml
        name: spark-cluster

        services:
            spark-master:
                image: apache/spark:3.5.0
                container_name: spark-master
                command:
                    - /opt/spark/bin/spark-class
                    - org.apache.spark.deploy.master.Master
                ports:
                    - "7077:7077"
                    - "8080:8080"
                volumes:
                    - ./jars:/opt/spark/jars-extra

            spark-worker-1:
                image: apache/spark:3.5.0
                container_name: spark-worker-1
                command:
                    - /opt/spark/bin/spark-class
                    - org.apache.spark.deploy.worker.Worker
                    - spark://spark-master:7077
                depends_on:
                    - spark-master
                volumes:
                    - ./jars:/opt/spark/jars-extra

            spark-worker-2:
                image: apache/spark:3.5.0
                container_name: spark-worker-2
                command:
                    - /opt/spark/bin/spark-class
                    - org.apache.spark.deploy.worker.Worker
                    - spark://spark-master:7077
                depends_on:
                    - spark-master
                volumes:
                    - ./jars:/opt/spark/jars-extra

            spark-client:
                build:
                    context: .
                    dockerfile: Dockerfile.spark-client
                container_name: spark-client
                command: tail -f /dev/null
                volumes:
                    - ./:/workspace
                    - ./jars:/opt/spark/jars-extra
                depends_on:
                    - spark-master

            minio:
                image: minio/minio
                container_name: minio
                command: server /data --console-address ":9001"
                environment:
                    MINIO_ROOT_USER: admin
                    MINIO_ROOT_PASSWORD: password123
                ports:
                    - "9000:9000"
                    - "9001:9001"
                volumes:
                    - ./warehouse:/data
        ```    

2. **실행**

    ```bash
    docker compose up -d
    ```

3. **확인**
    - 브라우저 (http://localhost:9001)
    - 로그인 (admin / password123)


### 3.4 Bucket 생성

- Iceberg용 Bucket 생성
    - 예: warehouse

- 구조: **`warehouse/`**
    - 향후에는 아래와 같이 생성

        ```text
        warehouse/
        ├─ demo.db
        ├─ metadata
        ├─ snapshots
        └─ parquet files
        ```


### 3.5 Spark에 필요한 라이브러리

- 가장 중요한 단계
- Spark는 기본적으로 **`Iceberg 모름`, `S3 모름`, `MinIO 모름`** 상태 🡲 **JAR 필요**
    - 필요한 JAR (Spark 3.5.0 기준)
        - Iceberg Runtime [iceberg-spark-runtime-3.5_2.12-1.5.2.jar]
            - [Apache Iceberg Releases](https://iceberg.apache.org/releases/?utm_source=chatgpt.com)
            - 또는 [Maven Central Iceberg Runtime](https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.2/?utm_source=chatgpt.com)
        - AWS S3 Connector [hadoop-aws-3.3.4.jar]
            - [Maven Central Hadoop AWS](https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/?utm_source=chatgpt.com)
        - AWS SDK Bundle [aws-java-sdk-bundle-1.12.262.jar]
            - [Maven Central AWS SDK Bundle](https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/?utm_source=chatgpt.com)

    - 다운로드 후

        ```text
        jars/
        │
        ├── iceberg-spark-runtime-3.5_2.12-1.5.2.jar
        ├── hadoop-aws-3.3.4.jar
        └── aws-java-sdk-bundle-1.12.262.jar
        ```

- **Iceberg SparkSession**

    ```python
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("Iceberg Test")
        .master("spark://spark-master:7077")

        .config(
            "spark.jars",
            ",".join([
                "/opt/spark/jars-extra/iceberg-spark-runtime-3.5_2.12-1.5.2.jar",
                "/opt/spark/jars-extra/hadoop-aws-3.3.4.jar",
                "/opt/spark/jars-extra/aws-java-sdk-bundle-1.12.262.jar"
            ])
        )

        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
        )

        .config(
            "spark.sql.catalog.demo",
            "org.apache.iceberg.spark.SparkCatalog"
        )

        .config(
            "spark.sql.catalog.demo.type",
            "hadoop"
        )

        .config(
            "spark.sql.catalog.demo.warehouse",
            "s3a://warehouse/"
        )

        .config(
            "spark.sql.catalog.demo.io-impl",
            "org.apache.iceberg.aws.s3.S3FileIO"
        )

        .config(
            "spark.hadoop.fs.s3a.endpoint",
            "http://minio:9000"
        )

        .config(
            "spark.hadoop.fs.s3a.access.key",
            "admin"
        )

        .config(
            "spark.hadoop.fs.s3a.secret.key",
            "password123"
        )

        .config(
            "spark.hadoop.fs.s3a.path.style.access",
            "true"
        )

        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

        .getOrCreate()
    )
    ```

    - Apache Iceberg에서는 최근 iceberg-spark-runtime-3.5_2.12와 iceberg-aws-bundle를 사용할 것을 권장하고 있으나 현업에서는 REST Catalog, Glue Catalog, DynamoDB Lock 등과 사용할때 iceberg-aws-bundle을 사용하고, 그 외의 경우에는(특히 MinIO와 연동 시) iceberg-aws-bundle 없이 S3 기능 활용으로만 사용하는 편이라서 예제에서는 iceberg-aws-bundle을 제외함


- **Master와 Worker 모두 마운트**

    ```yaml
    volumes:
    - ./spark/jars:/opt/spark/jars
    ```


### 3.6 Spark Catalog 설정

- Iceberg 테이블을 Spark에서 사용할 수 있도록 Catalog를 등록

1. **spark-defaults.conf 설정**

    ```properties
    # Iceberg 기능 활성화
    spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions

    # Catalog 이름 정의
    spark.sql.catalog.demo=org.apache.iceberg.spark.SparkCatalog

    # Catalog 타입
    spark.sql.catalog.demo.type=hadoop

    # Iceberg 메타데이터 저장 위치
    spark.sql.catalog.demo.warehouse=s3a://warehouse/

    # MinIO 접속 정보
    spark.hadoop.fs.s3a.endpoint=http://minio:9000
    spark.hadoop.fs.s3a.access.key=minioadmin
    spark.hadoop.fs.s3a.secret.key=minioadmin
    spark.hadoop.fs.s3a.path.style.access=true
    spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
    ```

2. **설정 의미**

| 설정                                  | 설명                          |
| ------------------------------------- | ----------------------------- |
| spark.sql.extensions                  | Spark SQL에 Iceberg 기능 추가 |
| spark.sql.catalog.demo                | Iceberg Catalog 이름 등록     |
| spark.sql.catalog.demo.type           | Hadoop Catalog 사용           |
| spark.sql.catalog.demo.warehouse      | Iceberg 테이블 저장 위치      |
| spark.hadoop.fs.s3a.endpoint          | MinIO 서버 주소               |
| spark.hadoop.fs.s3a.access.key        | MinIO Access Key              |
| spark.hadoop.fs.s3a.secret.key        | MinIO Secret Key              |
| spark.hadoop.fs.s3a.path.style.access | MinIO 호환 설정               |
| spark.hadoop.fs.s3a.impl              | S3A 파일시스템 사용           |


3. **사용 예시**
    - 데이터베이스 생성

        ```sql
        CREATE DATABASE demo.factory;
        ```

    - 테이블 생성

        ```sql
        CREATE TABLE demo.factory.sensor_data (
            sensor_id INT,
            temperature DOUBLE,
            status STRING
        )
        USING iceberg;
        ```

    - 데이터 조회

        ```sql
        SELECT * FROM demo.factory.sensor_data;
        ```


- 4. **Catalog 이름 사용 규칙**
    - Catalog 이름을 `demo`로 지정했으므로 모든 Iceberg 객체는 다음 형식으로 접근

        ```sql
        demo.데이터베이스명.테이블명
        ```

        - 예시

            ```sql
            demo.factory.sensor_data
            demo.sales.orders
            demo.iot.devices
            ```


### 3.7 MinIO 연결 설정

- Spark가 MinIO를 S3처럼 사용하도록 설정 (spark-defaults.conf)

    ```properties
    # MinIO(S3) 서버 주소
    spark.hadoop.fs.s3a.endpoint=http://minio:9000

    # MinIO 계정 정보
    spark.hadoop.fs.s3a.access.key=admin
    spark.hadoop.fs.s3a.secret.key=password123

    # MinIO 호환 설정(Path Style)
    spark.hadoop.fs.s3a.path.style.access=true

    # S3A 파일시스템 사용
    spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
    ```

- **설정 설명**

| 설정              | 설명                      |
| ----------------- | ------------------------- |
| endpoint          | MinIO 서버 주소           |
| access.key        | MinIO 사용자 계정         |
| secret.key        | MinIO 비밀번호            |
| path.style.access | MinIO 사용 시 필수 설정   |
| fs.s3a.impl       | Spark가 S3A 프로토콜 사용 |

- **결과**
    - 이 설정을 완료하면 Spark는 **`Spark 🡲 Iceberg 🡲 MinIO(S3)`** 경로를 통해 Iceberg 데이터를 저장할 수 있음
    - 사용 예

        ```text
        s3a://warehouse/factory/sensor_data
        ```


### 3.8 Iceberg Catalog 선택

- **방법 1: Hadoop Catalog**

    ```properties
    spark.sql.catalog.demo.type=hadoop
    spark.sql.catalog.demo.warehouse=s3a://warehouse/
    ```

    - 장점: 설정 쉬움, 단일 Spark 실습에 적합
    - 단점: Spark 외 다른 엔진과 메타데이터 공유가 불편함

- **방법 2: REST Catalog (권장)**

    ```properties
    spark.sql.catalog.demo.type=rest
    spark.sql.catalog.demo.uri=http://iceberg-rest:8181
    spark.sql.catalog.demo.warehouse=s3a://warehouse/
    ```

    - Spark, Trino, Flink가 동일한 메타데이터 사용 가능

    - 추가 필요 사항
        - Iceberg REST Catalog 서버 실행
        - Spark에 REST Catalog 설정 추가
        - Trino/Flink도 동일한 Catalog 주소 사용

    - 장점: 데이터 레이크하우스 구성에 적합
    - 단점: 별도 REST Catalog 서버 필요

- **실습 선택**
    - 본 실습: Hadoop Catalog 사용
    - 실무 확장: REST Catalog 사용 권장

- **핵심 내용**
    1. Hadoop Catalog는 설정만 하면 됨
    2. REST Catalog는 서버를 하나 더 띄워야 함
    3. 이번 실습에서는 Hadoop Catalog만 사용


- **참고: Iceberg REST Catalog**
    - Hadoop Catalog 대신 사용할 수 있는 실무형 구성
        - 실무에서는 Spark뿐 아니라 Trino, Flink 등 여러 엔진이 동일한 Iceberg 메타데이터를 공유하기 위해 REST Catalog를 사용함
    - 여러 데이터 처리 엔진이 동일한 Iceberg 메타데이터를 공유할 때 활용<br><br>

    - 추가 컨테이너

        ```yaml
        iceberg-rest:
            image: tabulario/iceberg-rest
            ports:
                - "8181:8181"
        ```

    - 주요 설정

        ```yaml
        environment:
            CATALOG_WAREHOUSE=s3://warehouse/
            CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO
            AWS_ACCESS_KEY_ID=admin
            AWS_SECRET_ACCESS_KEY=password123
            CATALOG_S3_ENDPOINT=http://minio:9000
        ```

        - 역할: Spark / Trino / Flink 🡲 Iceberg REST Catalog 🡲 MinIO

### 3.10 Spark에서 연결 확인

- **Spark SQL 실행**

    ```bash
    docker exec -it spark-client bash
    spark-sql
    ```

- **Namespace 생성**

    ```sql
    CREATE NAMESPACE demo.db;
    ```

    - Namespace가 정상적으로 생성되면
        - Iceberg 메타데이터가 저장되고,
        - MinIO의 Warehouse 영역에 관련 디렉터리가 생성됨

        ```text
        MinIO
            warehouse/
            └─ demo.db/
        ```

- **의미**
    - Spark ↔ Iceberg 연결 확인
    - Iceberg ↔ MinIO 연결 확인
    - Catalog 설정 정상 동작 확인

### 3.11 Iceberg 테이블 생성

- **Spark SQL 실행**

    ```bash
    docker exec -it spark-client bash
    spark-sql
    ```

- **테이블 생성**

    ```sql
    CREATE TABLE demo.db.customer (
        id bigint,
        name string
    )
    USING iceberg;
    ```

- **테이블 확인**

    ```sql
    SHOW TABLES IN demo.db;
    ```

    - 결과 예시

    ```text
    +---------+---------+
    |namespace|tableName|
    +---------+---------+
    |db       |customer |
    +---------+---------+
    ```

- **의미**
    - Iceberg 테이블 생성 확인
    - Catalog 정상 동작 확인
    - MinIO에 메타데이터 저장 확인

- **PySpark에서 생성하기**

    ```python
    spark.sql("""
    CREATE TABLE demo.sensor_data (
        device_id INT,
        temperature DOUBLE,
        status STRING
    )
    USING iceberg
    """)
    ```

    - 테이블 확인

        ```python
        spark.sql("SHOW TABLES IN demo.db").show()
        ```


### 3.12 데이터 입력

```sql
INSERT INTO demo.db.customer
VALUES
(1,'Kim'),
(2,'Lee');
```

- 조회

    ```sql
    SELECT * FROM demo.db.customer;
    ```

- 결과

    ```text
    1 Kim
    2 Lee
    ```

데이터 저장

```python
df = spark.createDataFrame([
    (101,72.5,"NORMAL"),
    (102,91.2,"CRITICAL")
],["device_id","temperature","status"])

df.writeTo("demo.sensor_data").append()
```
조회

```python
spark.sql("""
SELECT * FROM demo.sensor_data
""").show()
```


### 3.13 MinIO에서 확인

- MinIO Console: **`warehouse/`**
- 확인: 다음과 유사한 구조의 생성 여부

    ```text
    warehouse/

    demo.db.db/

    metadata/
    ├─ v1.metadata.json
    ├─ snap-xxx.avro

    data/
    ├─ part-00000.parquet
    ```


### 3.14 Python(PySpark)에서 사용

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("spark://spark-master:7077")
    .appName("Iceberg Example")
    .getOrCreate()
)

df = spark.createDataFrame(
    [(1, "Kim"), (2, "Lee")],
    ["id", "name"]
)

df.writeTo("demo.db.customer").append()

spark.sql("""
SELECT *
FROM demo.db.customer
""").show()
```

- **Iceberg의 진짜 장점**
    - 일반 Parquet

        ```text
        INSERT
        UPDATE 불가
        DELETE 불가
        TIME TRAVEL 불가
        ```

    - Iceberg: 다음이 가능함

        ```sql
        SELECT * FROM sensor_data VERSION AS OF 100

        SELECT * FROM sensor_data TIMESTAMP AS OF
        '2025-06-22 10:00:00'

        DELETE FROM sensor_data
        WHERE temperature > 90

        UPDATE sensor_data
        SET status='WARN'
        WHERE device_id=101
        ```


### 3.15 최종 체크리스트

- 환경 구성이 끝났을 때 아래가 모두 통과하면 성공

- **MinIO**

    ```text
    ✓ Console 접속 가능
    ✓ warehouse Bucket 존재
    ```

- **Spark**

    ```text
    ✓ Spark Master 접속 가능
    ✓ Worker 2개 등록
    ✓ spark-submit 실행 가능
    ```

- **Iceberg**

    ```text
    ✓ CREATE NAMESPACE 성공
    ✓ CREATE TABLE 성공
    ✓ INSERT 성공
    ✓ SELECT 성공
    ```

- **MinIO 내부**

    ```text
    ✓ metadata 폴더 생성
    ✓ parquet 파일 생성
    ```
















> - 현대적인 데이터 레이크하우스(Data Lakehouse)의 표준 스택인 Spark(연산) + Apache Iceberg(테이블 포맷) + MinIO(오브젝트 스토리지)를 연결하는 실습
> - 로컬 환경에서 Docker Compose를 이용해 전체 인프라를 한 번에 구축하고, PySpark를 통해 Iceberg 테이블에 데이터를 적재·조회하는 핵심 과정
{: .common-quote}


## 1. 실습 아키텍처 이해

- **MinIO:**
    - AWS S3와 호환되는 오픈소스 오브젝트 스토리지
    - 실제 데이터 파일(Parquet)과 메타데이터가 저장되는 원격 저장소 역할

- **Apache Iceberg:**
    - MinIO 위에 얹어지는 고성능 테이블 포맷 라이브러리
    - S3 환경에서도 SQL 트랜잭션(ACID), 타임 트래블(과거 데이터 조회), 스키마 진화(Schema Evolution)를 가능하게 함

- **Apache Spark:**
    - Iceberg 및 MinIO 커넥터를 장착하고 대용량 데이터를 메모리상에서 분산 처리하는 계산 엔진



## 2. 실습 내용

- **[단계 1] Docker Compose 환경 구축**
    - 프로젝트를 진행할 빈 폴더 생성 후, `docker-compose.yml` 파일을 작성

    ```yaml
    version: '3.8'

    services:
    # 1. 분산 저장소: MinIO
    minio:
        image: minio/minio:RELEASE.2024-01-18T22-51-58Z
        ports:
        - "9000:9000"       # S3 API 포트
        - "9001:9001"       # 웹 콘솔 UI 포트
        environment:
        MINIO_ROOT_USER: admin
        MINIO_ROOT_PASSWORD: password123
        command: server /data --console-address ":9001"
        volumes:
        - minio_data:/data

    # 2. 저장소 초기화용 컨테이너 (자동으로 버킷 생성)
    mc:
        image: minio/mc:RELEASE.2024-01-11T05-49-32Z
        depends_on:
        - minio
        entrypoint: >
        /bin/sh -c "
        until (/usr/bin/mc config host add myminio http://minio:9000 admin password123) do echo 'Waiting for MinIO...' && sleep 1; done;
        /usr/bin/mc mb myminio/warehouse;
        exit 0;
        "

    # 3. 계산 엔진: Spark Master + 주피터 노트북 내장 환경
    spark-iceberg:
        image: tabulario/spark-iceberg:3.5.0_2.12-1.4.2
        ports:
        - "8888:8888"       # 주피터 노트북 웹 UI
        - "8080:8080"       # Spark Master Web UI
        environment:
        - SPARK_MODE=master
        volumes:
        - ./notebooks:/home/iceberg/notebooks
        - ./apps:/home/iceberg/apps

    volumes:
    minio_data:

    ```

    - **Tip:**
        - `tabulario/spark-iceberg` 이미지는 Spark와 Iceberg, AWS S3 커넥터(JAR 패키지)들이 이미 깔끔하게 빌드되어 있어 복잡한 jar 다운로드 설정을 수동으로 할 필요가 없음

- **클러스터 실행 명령어**
    - 터미널에서 위 파일이 있는 경로로 이동한 뒤 아래 명령어를 입력

    ```bash
    docker-compose up -d
    ```

    - `http://localhost:9001`에 접속하여 MinIO 콘솔(admin / password123)이 잘 열리는지, `warehouse` 버킷이 생성되었는지 확인
    - `http://localhost:8888`에 접속하여 주피터 노트북 환경이 잘 뜨는지 확인


- **[단계 2] PySpark 연결 설정 (Jupyter Notebook에서 진행)**
    - 주피터 노트북에서 새 노트북(`Python 3`)을 생성
    - Spark가 MinIO 스토리지와 Iceberg 카탈로그를 인식하도록 초기화 코드를 실행

    ```python
    from pyspark.sql import SparkSession

    # Iceberg 및 MinIO(S3) 연동을 위한 SparkSession 빌드
    spark = SparkSession.builder \
        .appName("Iceberg-MinIO-Data-Lakehouse") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.demo", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.demo.type", "hadoop") \
        .config("spark.sql.catalog.demo.warehouse", "s3a://warehouse/iceberg") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "admin") \
        .config("spark.hadoop.fs.s3a.secret.key", "password123") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    print("Spark가 성공적으로 연결되었습니다.")
    ```

- **핵심 설정 옵션 해설**
    - **`spark.sql.catalog.demo`:**
        - `demo`라는 이름의 논리적 카탈로그 정의
    - **`s3a://warehouse/iceberg`:**
        - MinIO의 `warehouse` 버킷 내 `iceberg` 폴더를 데이터 저장소로 설정
    - **`fs.s3a.endpoint`**:
        - 외부 클라우드가 아닌 도커 네트워크 내부의 `http://minio:9000` 주소로 연결되도록 타겟팅


- **[단계 3] 대용량 가상 데이터 가공 및 Iceberg 적재**
    - 샘플 데이터를 대량으로 생성하여 Iceberg 테이블을 만들고 저장하는 실습

    ```python
    import random
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

    # 1. 10만 건의 가상 IoT 센서 데이터 생성 함수
    def generate_sample_data(num_rows):
        data = []
        cities = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"]
        status_codes = [200, 404, 500]
        for i in range(num_rows):
            data.append((
                f"device_{random.randint(1, 100)}",
                random.choice(cities),
                random.choice(status_codes),
                random.randint(20, 40), # 온도
                1710000000 + i # 타임스탬프 고정값 가산
            ))
        return data

    schema = StructType([
        StructField("device_id", StringType(), True),
        StructField("city", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("temperature", IntegerType(), True),
        StructField("timestamp", LongType(), True)
    ])

    # 데이터프레임 변환 (10만 건 생성)
    raw_data = generate_sample_data(100000)
    df = spark.createDataFrame(raw_data, schema=schema)

    # 2. Iceberg 테이블 포맷으로 MinIO에 적재 (city 기준으로 분산 파티셔닝)
    df.write \
        .format("iceberg") \
        .partitionBy("city") \
        .mode("overwrite") \
        .save("demo.db.sensor_logs")

    print("Iceberg 테이블 저장이 완료되었습니다.")
    ```

- **[단계 4] 대용량 데이터 분석 및 고성능 기능 테스트**
    - 저장된 Iceberg 테이블을 SQL 및 DataFrame API로 다각도 분석

    - **고속 집계 쿼리 실행**

        ```python
        # Spark SQL을 이용한 데이터 조회
        spark.sql("""
            SELECT city, COUNT(*) as log_count, AVG(temperature) as avg_temp 
            FROM demo.db.sensor_logs 
            GROUP BY city
        """).show()
        ```

    - Iceberg의 강력한 기능: 메타데이터 트래킹 및 타임 트래블
        - 일반 S3 파이프라인과 달리, Iceberg는 데이터가 업데이트된 내역(Snapshots)을 추적할 수 있음

        ```python
        # 데이터 테이블의 역사(스냅샷 기록) 조회
        spark.sql("SELECT committed_at, snapshot_id, operation FROM demo.db.sensor_logs.snapshots").show()
        ```

> - **확인 과제:**
>   - 위 쿼리를 실행하면 데이터가 입력된 시점의 `snapshot_id`를 볼 수 있음
>   - 추후 데이터를 더 추가(`append`)하거나 삭제한 후, 과거 특정 스냅샷 ID를 지정하여 `spark.read.option("snapshot-id", <아이디>).load(...)` 문법을 쓰면 과거 데이터로 되돌아가는 **타임 트래블(Time Travel)**을 구현할 수 있음
{: .common-quote}


- **실습 종료 후 정리**
    - 모든 실습이 끝나면 도커 컨테이너를 내려 자원을 반납

    ```bash
    docker-compose down
    ```























<br>

- **클러스터 실행 및 웹 UI 확인**
    1. 터미널에서 해당 디렉터리로 이동 후 명령어 실행

        ```bash
        docker-compose up -d
        ```

    2. 웹 브라우저를 열고 `http://localhost:8080`에 접속
    3. **Spark Master Web UI**가 나타남
        - 하단 `Workers` 항목에 2대의 Worker 노드가 정상적으로 등록(Alive 상태)되어 있는지 확인<br><br>

        <div class="insert-image">
            <img src="/materials/devtools/images/S13-05-01-02_01-001_SparkInstall.png" style="width: 90%;">
        </div>


- **핵심 환경 설정 파일 튜닝 (`$SPARK_HOME/conf/`)**
    - 실무 및 대용량 데이터 처리를 위해 반드시 알아야 하는 주요 설정 파일들
        - Docker 환경의 경우 환경 변수나 SparkSession 생성 시 옵션으로 주입할 수 있음
    - MinIO + Iceberg + Spark + Trino 연동 시스템을 구축할 경우
        - 기술적으로는 docker-compose.yml만으로도 가능함
        - MinIO + Iceberg + Spark + Trino 정도 규모가 되면 설정 파일을 분리하는 것이 사실상 표준적인 운영 방식

    - **`spark-env.sh` (시스템 환경 설정)**
        - 메모리와 CPU 코어 자원을 제한하거나 지정할 때 사용

            ```bash
            SPARK_MASTER_HOST=localhost
            SPARK_WORKER_CORES=4          # 각 Worker가 사용할 CPU 코어 수
            SPARK_WORKER_MEMORY=4g        # 각 Worker가 사용할 총 메모리
            SPARK_DRIVER_MEMORY=2g        # 드라이버 프로그램이 사용할 메모리
            ```

    - **`spark-defaults.conf` (Spark 애플리케이션 기본값)**
        - 제출되는 모든 Spark 작업에 공통으로 적용될 옵션을 지정함
        - 향후 **Iceberg 및 MinIO 연동** 시 이 파일이나 작업 제출 시점에 커넥터 JAR 패키지 및 S3 엔드포인트를 지정해야 함

            ```properties
            # 셔플 파티션 수 기본값 조정 (대용량이 아닐 경우 200개는 너무 많으므로 줄여서 최적화)
            spark.sql.shuffle.partitions   50

            # 향후 전개될 MinIO(S3) 및 Iceberg 연동 예시 세팅
            spark.sql.extensions           org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            spark.sql.catalog.demo         org.apache.iceberg.spark.SparkCatalog
            spark.sql.catalog.demo.type    hadoop
            spark.sql.catalog.demo.warehouse s3a://my-bucket/warehouse
            ```

> - **환경 설정 확인을 위한 체크리스트**
>   - 설정이 완료된 후 아래의 요소들이 정상 작동하는지 확인하면 준비가 끝남
>       1. **포트 충돌 여부:**
>           - Master UI(`8080`) 또는 Spark 애플리케이션 UI(`4040`)가 다른 서비스와 충돌하지 않는지 확인
>       2. **자원 할당 확인:**
>           - Web UI에서 할당된 Memory와 Cores가 시스템 스펙 내에서 적절히 잡혔는지 확인
>           - 데이터가 밀릴 경우 이 자원 설정(Data Skew 및 Memory 부족 이슈)을 가장 먼저 튜닝하게 됨
{: .common-quote}





### 1.4 Trino와 연동 시 8080 포트 충돌

- **Trino(과거 Presto) 역시 기본 포트로 `8080`을 사용**하기 때문에,
- 동일한 호스트(서버)나 로컬 PC에서 Spark Master와 Trino를 동시에 띄우면 무조건 포트 충돌(Port Collision)이 발생함

- 최근의 데이터 레이크하우스 아키텍처에서는 **"저장(MinIO/S3 + Iceberg) + 계산(Spark) + 인터랙티브 쿼리/대시보드(Trino)"** 조합이 글로벌 표준(대세)
- 이 포트 충돌을 해결하는 표준적인 처리 패턴이 존재함

    - **Docker Compose 환경에서의 포트 포워딩 (가장 추천)**
        - Docker 환경을 사용 중이라면, 컨테이너 내부 포트는 `8080`으로 그대로 두더라도,
        - **호스트(사용자 PC)로 노출하는 외부 포트를 변경**하여 충돌을 간단히 피할 수 있음<br><br>

        - **Spark Master 설정 (외부 포트를 `8180`으로 변경)**

            ```yaml
            services:
            spark-master:
                image: bitnami/spark:3.5
                ports:
                - '8180:8080' # [호스트 포트 8180] : [컨테이너 내부 포트 8080]
                - '7077:7077'
            ```

        - **Trino 설정 (외부 포트를 `8080`으로 유지 또는 `8090`으로 변경)**

            ```yaml
            services:
            trino:
                image: trinodb/trino
                ports:
                - '8080:8080' # Trino는 원래대로 8080 사용
            ```

        - 이렇게 설정하면 브라우저에서 Spark UI는 `http://localhost:8180`으로, Trino UI는 `http://localhost:8080`으로 충돌 없이 깔끔하게 접근할 수 있음


    - **Spark 자체 설정으로 기본 포트 변경하기 (Standalone 환경)**
        - Docker를 쓰지 않고 로컬 서버에 직접 설치한 환경이라면,
        - Spark의 환경 설정 파일이나 실행 스크립트에서 기본 웹 UI 포트를 다른 값(예: `8282`)으로 바꿀 수 있음<br><br>

        - **방법 A: `spark-env.sh` 파일 수정**
            - `$SPARK_HOME/conf/spark-env.sh` 파일에 아래 설정을 추가합니다.

                ```bash
                export SPARK_MASTER_WEBUI_PORT=8282
                ```

        - **방법 B: 데몬 실행 시 옵션 주입**
            - 마스터 노드를 구동하는 스크립트를 실행할 때 포트를 직접 지정

                ```bash
                ./sbin/start-master.sh --webui-port 8282
                ```

    - **Spark 애플리케이션 UI(`4040`)의 자동 포트 포워딩 기능 활용**
        - 마스터 UI(`8080`) 외에, 
        - 개별 Spark 애플리케이션(작업)이 실행될 때 열리는 Spark Driver UI(`4040`)도 충돌 가능성이 있음
            - 예를 들어 여러 명의 개발자가 한 서버에서 동시에 `pyspark`를 실행하거나 여러 개의 배치 작업이 동시에 돌면 `4040` 포트가 겹치게 됨

        - Spark는 `4040` 포트에 대해 **자체적인 충돌 회피 메커니즘**을 내장하고 있음
            - `4040` 포트가 이미 사용 중이면 -> `4041` 시도 -> 또 사용 중이면 -> `4042` 시도
            - 이 과정이 성공할 때까지 자동으로 포트를 1씩 올리며 바인딩함<br><br>
            - 만약 이 자동 변경 범위를 제어하거나 명시적으로 바꾸고 싶다면
                - SparkSession을 생성할 때 아래 옵션을 부여

                    ```python
                    from pyspark.sql import SparkSession

                    spark = SparkSession.builder \
                        .appName("Iceberg-Trino-Spark-Test") \
                        .config("spark.ui.port", "4050") \ # 기본 4040 대신 4050부터 시작하도록 설정
                        .getOrCreate()
                    ```


<br>

> - **권장 아키텍처 구성**
>   - Trino와 Spark를 함께 엮어 데이터 레이크하우스를 구성할 때
>       - 아래와 같이 포트 맵핑을 깔끔하게 정리해 두고 진행하시는 것이 정신 건강(?)에 좋음<br><br>
>
>       <div class="info-table">
>       <table>
>           <thead>
>               <th style="width: 200px;">서비스 구성 요소</th>
>               <th style="width: 150px;">내부 기본 포트</th>
>               <th style="width: 200px;">추천 외부(호스트) 포트</th>
>               <th style="width: 350px;">용도</th>
>           </thead>
>           <tbody>
>               <tr>
>                   <td class="td-rowheader">Trino</td>
>                   <td>8080</td>
>                   <td>8080</td>
>                   <td>대시보드 연동 및 BI용 쿼리 UI</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">Spark Master</td>
>                   <td>8080</td>
>                   <td><b>8180 (변경)</b></td>
>                   <td>클러스터 및 Worker 상태 모니터링</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">Spark Driver App</td>
>                   <td>4040</td>
>                   <td><b>4040(자동 순차 증가)</b></td>
>                   <td>개별 Spark Job 실행 상세 트래킹</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">MinIO Web UI</td>
>                   <td>9001</td>
>                   <td>9001</td>
>                   <td>Iceberg 데이터 저장소 오브젝트 브라우징</td>
>               </tr>
>           </tbody>    
>       </table>
>       </div>
{: .common-quote}