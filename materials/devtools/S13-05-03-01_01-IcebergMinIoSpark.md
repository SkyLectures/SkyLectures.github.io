---
layout: page
title:  "Iceberg, MinIO 연결 및 대용량 데이터 분석 준비"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-03-01_01-IcebergMinIoSpark
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


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
