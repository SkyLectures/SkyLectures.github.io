---
layout: page
title:  "Iceberg, MinIO 연결 및 대용량 데이터 분석 준비"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-03-01_01-IcebergMinIoSpark
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}
 
 
좋습니다. 사용자님의 현재 상황은

```text
Docker Compose
 ├─ Spark Master
 ├─ Spark Worker 1
 └─ Spark Worker 2
```

까지 구축된 상태에서

```text
Spark (연산 엔진)
      ↓
Iceberg (테이블 포맷)
      ↓
MinIO (S3 호환 스토리지)
```

를 연결하려는 것입니다.

이 과정을 단순히 설정 파일만 나열하지 않고 **왜 필요한지**, **어떤 역할을 하는지**, **어디서 오류가 나는지**까지 포함하여 설명하겠습니다.

---

# 1. 전체 구조 이해

먼저 Iceberg는 DB가 아닙니다.

많은 사람들이 처음에 착각합니다.

```text
MySQL
 ├─ 데이터
 └─ 메타데이터

PostgreSQL
 ├─ 데이터
 └─ 메타데이터
```

와 달리 Iceberg는

```text
Iceberg
 ├─ 메타데이터
 ├─ Snapshot
 ├─ Manifest
 └─ Table Schema
```

만 관리합니다.

실제 데이터는

```text
Parquet
ORC
Avro
```

파일로 저장됩니다.

---

따라서 구조는

```text
Spark
  ↓
Iceberg
  ↓
MinIO
```

입니다.

실제 저장은 MinIO가 담당합니다.

---

# 2. 구성요소별 역할

## Spark

역할

```text
연산 수행
ETL
DataFrame 처리
SQL 실행
```

예

```python
df.writeTo("demo.customer").create()
```

실행

---

## Iceberg

역할

```text
테이블 정의
Schema
Partition
Snapshot
Time Travel
```

관리

예

```sql
CREATE TABLE demo.customer
```

---

## MinIO

역할

```text
Parquet 저장
Manifest 저장
Metadata 저장
```

즉 실제 데이터 저장소

---

# 3. MinIO 먼저 구축

## docker-compose 추가

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

---

실행

```bash
docker compose up -d
```

---

확인

브라우저

```text
http://localhost:9001
```

---

로그인

```text
admin
password123
```

---

# 4. Bucket 생성

Iceberg용 Bucket 생성

예

```text
warehouse
```

---

구조

```text
warehouse/
```

---

나중에

```text
warehouse/
 ├─ demo.db
 ├─ metadata
 ├─ snapshots
 └─ parquet files
```

생성

---

# 5. Spark에 필요한 라이브러리

이 단계가 가장 중요합니다.

Spark는 기본적으로

```text
Iceberg 모름
S3 모름
MinIO 모름
```

입니다.

---

따라서 JAR 필요

```text
iceberg-spark-runtime-3.5_2.12
iceberg-aws-bundle
hadoop-aws
aws-java-sdk-bundle
```

---

예

```text
spark/
 └─ jars/
```

---

```text
spark/jars/

iceberg-spark-runtime-3.5_2.12-1.10.0.jar
iceberg-aws-bundle-1.10.0.jar
hadoop-aws-3.3.4.jar
aws-java-sdk-bundle.jar
```

---

Master와 Worker 모두 마운트

```yaml
volumes:
  - ./spark/jars:/opt/spark/jars
```

---

# 6. Spark Catalog 설정

여기서 Spark가

```text
Iceberg를 어떻게 찾을지
```

정의

---

spark-defaults.conf

```properties
spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
```

의미

```text
Spark SQL에 Iceberg 기능 추가
```

---

Catalog 정의

```properties
spark.sql.catalog.demo=org.apache.iceberg.spark.SparkCatalog
```

---

Catalog 이름

```text
demo
```

---

사용 시

```sql
SELECT * FROM demo.db.customer
```

형태

---

# 7. MinIO 연결 설정

Spark가 MinIO를 S3처럼 사용

---

```properties
spark.hadoop.fs.s3a.endpoint=http://minio:9000
```

---

설명

```text
S3 주소
```

---

접속 정보

```properties
spark.hadoop.fs.s3a.access.key=admin

spark.hadoop.fs.s3a.secret.key=password123
```

---

Path Style

```properties
spark.hadoop.fs.s3a.path.style.access=true
```

MinIO에서 거의 필수

---

# 8. Iceberg Catalog 선택

여기서 선택지가 있습니다.

---

## 방법 1

Hadoop Catalog

```text
간단
```

---

설정

```properties
spark.sql.catalog.demo.type=hadoop

spark.sql.catalog.demo.warehouse=s3a://warehouse/
```

---

장점

```text
설정 쉬움
```

---

단점

```text
Trino 연동 불편
```

---

## 방법 2

REST Catalog

권장

---

```text
Spark
Trino
Flink
```

모두 공유 가능

---

실무에서 증가 추세

---

# 9. Iceberg REST Catalog

추가 컨테이너

```yaml
iceberg-rest:

  image: tabulario/iceberg-rest

  ports:
    - "8181:8181"
```

---

설정

```yaml
environment:

  CATALOG_WAREHOUSE=s3://warehouse/

  CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO

  AWS_ACCESS_KEY_ID=admin

  AWS_SECRET_ACCESS_KEY=password123

  CATALOG_S3_ENDPOINT=http://minio:9000
```

---

역할

```text
Spark
Trino
     ↓
 Iceberg Catalog
     ↓
   MinIO
```

---

# 10. Spark에서 연결 확인

컨테이너 진입

```bash
docker exec -it spark-master bash
```

---

Spark SQL

```bash
spark-sql
```

---

Namespace 생성

```sql
CREATE NAMESPACE demo.db;
```

---

성공하면

```text
MinIO
  warehouse/
      demo.db/
```

생성

---

# 11. Iceberg 테이블 생성

```sql
CREATE TABLE demo.db.customer (
    id bigint,
    name string
)
USING iceberg;
```

---

확인

```sql
SHOW TABLES IN demo.db;
```

---

# 12. 데이터 입력

```sql
INSERT INTO demo.db.customer
VALUES
(1,'Kim'),
(2,'Lee');
```

---

조회

```sql
SELECT * FROM demo.db.customer;
```

---

결과

```text
1 Kim
2 Lee
```

---

# 13. MinIO에서 확인

MinIO Console

```text
warehouse/
```

확인

---

대략

```text
warehouse/

demo.db.db/

metadata/
 ├─ v1.metadata.json
 ├─ snap-xxx.avro

data/
 ├─ part-00000.parquet
```

생성

---

# 14. Python(PySpark)에서 사용

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

---

# 최종 체크리스트

환경 구성이 끝났을 때 아래가 모두 통과하면 성공입니다.

### MinIO

```text
✓ Console 접속 가능
✓ warehouse Bucket 존재
```

### Spark

```text
✓ Spark Master 접속 가능
✓ Worker 2개 등록
✓ spark-submit 실행 가능
```

### Iceberg

```text
✓ CREATE NAMESPACE 성공
✓ CREATE TABLE 성공
✓ INSERT 성공
✓ SELECT 성공
```

### MinIO 내부

```text
✓ metadata 폴더 생성
✓ parquet 파일 생성
```

---

사용자님처럼 교육·실습 환경을 구축하는 목적이라면, 처음에는 **Hadoop Catalog 방식(Spark ↔ Iceberg ↔ MinIO)** 으로 먼저 성공시키고, 그 다음 단계에서 **REST Catalog를 추가하여 Spark + Trino가 동일 Iceberg 테이블을 공유하는 구조**로 확장하는 것이 가장 이해하기 쉽고 문제 분석도 수월합니다. Hadoop Catalog가 성공한 뒤 REST Catalog로 넘어가면 어떤 설정이 추가되는지 명확하게 보입니다.



























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