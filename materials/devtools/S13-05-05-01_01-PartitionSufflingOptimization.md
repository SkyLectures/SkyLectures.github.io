---
layout: page
title:  "파티셔닝과 셔플링 최적화 이해하기"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-05-01_01-PartitionSufflingOptimization
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}
 
 
## 1. 파티셔닝 (Partitioning)

- **개념 및 정의**

    - 대규모의 단일 데이터를 분산 클러스터 내부의 여러 워커 노드가 나누어 처리할 수 있도록, **물리적으로 작고 독립적인 데이터 조각(Partition)으로 쪼개는 행위**
    - 스파크의 파티셔닝은 크게 '메모리 상의 파티셔닝(RDD/DataFrame)'과 '스토리지 상의 파티셔닝(Iceberg/MinIO)'으로 분류됨

- **기술적 기반 내용**

    - **메모리 파티셔닝:**
        - 1,000만 건의 데이터를 정확히 8개로 쪼개어 분산 메모리(JVM 호스트)에 분할 적재하는 메커니즘
        - `pdp.py`에서 수행한 `spark.range(0, TOTAL_ROWS, numPartitions=8)`이 대표적

    - **스토리지 파티셔닝:**
        - Iceberg 테이블 선언 시 지정한 `PARTITIONED BY (factory_id)` 규격
        - 데이터를 디스크에 쓸 때 파일 시스템 레벨에서 `factory_id=FAC_SEOUL`과 같은 물리 디렉터리로 분기하여 저장하는 방식

- **중요성 및 필요성**

    - **병렬 처리의 단위:**
        - 스파크에서 1개의 파티션은 1개의 태스크(Task)와 매핑되며, 워커 노드의 CPU 코어 1개에 할당됨
        - 파티셔닝이 제대로 안 되어 파티션 수가 코어 수보다 적으면 놀고 있는 CPU 코어(Idle)가 발생하여 분산 인프라의 리소스가 낭비됨

    - **데이터 편향(Skewness) 방지:**
        - 데이터가 특정 파티션에만 비정상적으로 몰리면,
            - 아무리 워커 노드가 많아도 해당 파티션을 처리하는 단 하나의 코어 때문에 전체 파티프라인 속도가 느려짐
            - 이를 방지하기 위해 균등한 파티셔닝이 필수

- **활용 방법 및 최적화 방향**

    - **파티션 프루닝(Partition Pruning):**
        - `WHERE factory_id = 'FAC_SEOUL'` 쿼리를 실행할 때,
            - 스파크는 부산 공장 폴더를 아예 쳐다보지도 않고 서울 공장 폴더의 Parquet 파일만 직접 조준하여 읽어 들임
            - I/O 비용을 드라마틱하게 아끼는 기법

    - **적정 파티션 크기 유지:**
        - 스파크 메모리 내 파티션 1개당 가장 이상적인 크기는 대략 **100MB ~ 200MB**
        - 너무 잘게 쪼개면 메타데이터 관리 오버헤드가 커지고,
        - 너무 크면 가상머신의 `OutofMemory(OOM)`가 발생함

- **예제 및 설명**
    - 최초 생성된 8개의 파티션 조각들을
        - 비즈니스 마트 규격에 맞게 `factory_id`를 기준으로 딱 2개의 덩어리로 재배치(`repartition`)하는 예제
        - 이 가공을 거친 데이터가 MinIO에 써질 때 정확하게 공장별 단일 파일 스트림 계층을 형성하게 됨

    ```python
    # [예제] 1,000만 건 데이터를 공장별 균등 파티션으로 분할 가공하기
    df_large = spark.range(0, 10000000, numPartitions=8) # 메모리 상에 8개 파티션 강제 할당

    # factory_id 컬럼 기준으로 데이터프레임 내부 파티션 재정렬
    df_repartitioned = df_large.withColumn(
        "factory_id", when(col("id") % 2 == 0, "FAC_SEOUL").otherwise("FAC_BUSAN")
    ).repartition(2, "factory_id") # 공장 ID 기준 2개의 깨끗한 파티션으로 재컴팩트
    ```

    - 전체 코드

    ```python
    #//file: "part_shuffle_lab.py"
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, when, expr
    import time

    if __name__ == "__main__":
        # [Step 1] Spark 세션 기동 (내장 라이브러리 기반 안전 가동)
        spark = SparkSession.builder \
                .appName("Spark-Partition-Shuffle-Optimization-Lab") \
                .master("spark://spark-master:7077") \
                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
                .config("spark.sql.catalog.minio_lake", "org.apache.iceberg.spark.SparkCatalog") \
                .config("spark.sql.catalog.minio_lake.type", "hadoop") \
                .config("spark.sql.catalog.minio_lake.warehouse", "s3a://warehouse/iceberg") \
                .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
                .config("spark.hadoop.fs.s3a.access.key", "admin") \
                .config("spark.hadoop.fs.s3a.secret.key", "password123") \
                .config("spark.hadoop.fs.s3a.path.style.access", "true") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                .getOrCreate()

        print("\n=== 1. 파티셔닝/셔플링 최적화 실험 엔진 가동 완료 ===")

        TOTAL_ROWS = 10_000_000
        print(f"\n=== 2. 메모리 상에 {TOTAL_ROWS:,}건 물리 파티션(8개) 분할 할당 ===")
        start_time = time.time()

        # 메모리 상에 8개 분산 파티션 블록 강제 생성
        df_large = spark.range(0, TOTAL_ROWS, numPartitions=8)

        # factory_id 컬럼 기준으로 데이터프레임 내부 파티션 재정렬 연산 계획 수립
        df_repartitioned = df_large.withColumn(
            "factory_id", when(col("id") % 2 == 0, "FAC_SEOUL").otherwise("FAC_BUSAN")
        ).withColumn("temperature", (rand(seed=42) * 40 + 10).cast("double")) \
        .withColumn("timestamp", expr("171921600 + id % 86400")) \
        .repartition(2, "factory_id")

        print(f"-> [성공] 대용량 8개 파티션을 공장별 2개 파티션으로 컴팩트 재배치 계획 수립 완료 (소요시간:{time.time() - start_time: .2f}초)")

        # [Step 3] 타겟 Iceberg 데이터베이스 및 테이블 정의 (DDL - 즉시 실행)
        spark.sql("CREATE DATABASE IF NOT EXISTS minio_lake.performance_db")
        spark.sql("DROP TABLE IF EXISTS minio_lake.performance_db.massive_sensor_logs")
        spark.sql("""
                CREATE TABLE minio_lake.performance_db.massive_sensor_logs (
                    id LONG,
                    sensor_id STRING,
                    factory_id STRING,
                    temperature DOUBLE,
                    timestamp LONG
                ) USING iceberg
                PARTITIONED BY (factory_id)
        """)

        print(f"\n=== 3. Action 호출: 최적화 파티션 데이터 레이크하우스 병렬 저장 ===")
        write_start = time.time()

        # 가공 및 정렬된 데이터프레임을 MinIO 오브젝트 스토리지로 병렬 적재
        df_repartitioned.write \
                        .format("iceberg") \
                        .mode("append") \
                        .save("minio_lake.performance_db.massive_sensor_logs")

        print(f"-> [성공] 1,000만 건 최적화 파티션 적재 완료! (소요시간: {time.time() - write_start:.2f}초)")

        print(f"\n=== 4. 셔플 파티션 최적화 제어(200개 ➡️ 4개) 후 분산 집계 SQL 실행 ===")
        query_start = time.time()

        # 💡 무의미한 200개 임시 태스크 양산을 막기 위해 클러스터 총 코어 수 스펙에 맞춰 셔플 파티션 고정
        spark.conf.set("spark.sql.shuffle.partitions", "4")

        summary_result = spark.sql("""
            SELECT 
                factory_id,
                COUNT(*) as total_count,
                ROUND(AVG(temperature), 2) as avg_temp,
                ROUND(MAX(temperature), 2) as max_temp
            FROM minio_lake.performance_db.massive_sensor_logs
            GROUP BY factory_id
        """)

        # 연산 최종 수행 및 콘솔 리포트 출력
        summary_result.show()
        print(f"-> [성공] 셔플링 제어 대시보드 연산 완료! (소요시간: {time.time() - query_start:.2f}초)")

        spark.stop()
    ```

    - 실행

    ```bash
    docker exec -it spark-client spark-submit /workspace/part_shuffle_lab.py
    ```


## 2. 셔플링 (Shuffling)

- **개념 및 정의**
    - 스파크 연산 도중 **여러 워커 노드 간에 다른 노드의 메모리에 있는 데이터를 네트워크를 통해 서로 교환하고 재배치하는 물리적 과정**
    - 데이터의 물리적 방(컨테이너)을 대대적으로 바꾸는 이사 작업과 같음

- **기술적 기반 내용**

    - 셔플링이 발생하면 스파크는
        - 데이터를 디스크에 임시로 쓰는 **Shuffle Write**를 수행하고,
        - 네트워크 포트를 통해 목적지 노드가 이를 다운로드하는 **Shuffle Read** 과정을 거침

    - 이 과정에서 대량의 디스크 I/O, 네트워크 대역폭 소모, 자바 직렬화/역직렬화(Serialization) 비용이 한꺼번에 발생
    - 스파크 파이프라인 중 **가장 자원을 많이 먹는 병목(Bottleneck) 단계**가 됨

- **중요성 및 필요성**

    - **분산 집계의 필수 관문:**
        - 단독 노드 안에서는 `FAC_SEOUL` 데이터의 전체 카운트나 평균 온도를 구할 수 없음
            - 전국 공장 데이터가 모든 워커 노드에 흩어져 있기 때문
        - 따라서 각 워커 노드가 가진 데이터를 공장명 키(Key)를 기준으로
            - 네트워크를 통해 한 노드로 몰아주어야만 최종 `GROUP BY` 연산이 성립됨
        - 이 때문에 셔플링은 분산 처리의 필수 요소

- **활용 방법 및 최적화 방향**

    - **셔플링 원천 차단 (Map-side Join):**
        - 대용량 테이블과 소용량 테이블을 결합할 때는 무거운 셔플링 대신,
        - 소용량 테이블을 모든 워커 노드의 메모리에 복제본으로 뿌려두는 브로드캐스트 조인(Broadcast Join)을 사용하여 셔플링 발생 자체를 차단

    - **적정 셔플 파티션 수 튜닝:**
        - 스파크의 `GROUP BY` 나 `JOIN` 연산 시 기본 셔플 파티션 수는 `200`개로 고정되어 있음
        - 1,000만 건 이하의 데이터에서는 200개로 쪼개면 지나치게 작은 찌꺼기 파일이 양산되므로 이를 리소스 스펙에 맞게 줄여주어야 성능이 향상됨

- **예제 및 설명**
    - 셔플 파티션을 최적화하여 분산 대시보드 집계하기
    - 기본값 200개를 ROGStrix의 워커 코어 수(총 4개 코어)에 맞춤 설정

    ```python
    spark.conf.set("spark.sql.shuffle.partitions", "4")

    # 셔플링을 유발하는 고부하 GROUP BY 연산 수행
    summary_result = spark.sql("""
        SELECT factory_id, COUNT(*) as total_count, AVG(temperature) as avg_temp
        FROM minio_lake.performance_db.massive_sensor_logs
        GROUP BY factory_id
    """)
    summary_result.show()
    ```

    - `GROUP BY factory_id` 구문이 들어가는 순간 워커 1, 2 간에 대대적인 물리적 데이터 셔플링이 가동됨
    - 이때 `spark.sql.shuffle.partitions` 값을 `4`로 낮추어 주었기 때문에,
    - 스파크는 쓸데없이 200개의 임시 가상 태스크를 만들지 않고 딱 4개의 굵직한 분산 파티션 파일만 교환하여 네트워크 전달 속도를 극대화함

    - 전체 코드

    ```python
    #//file: "ecommerce_analytics.py"
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, when, expr, round
    import time

    if __name__ == "__main__":
        # [Step 1] Spark 세션 기동 (내장 라이브러리 기반 안전 가동)
        spark = SparkSession.builder \
                .appName("Ecommerce-Data-Processing-Lab") \
                .master("spark://spark-master:7077") \
                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
                .config("spark.sql.catalog.minio_lake", "org.apache.iceberg.spark.SparkCatalog") \
                .config("spark.sql.catalog.minio_lake.type", "hadoop") \
                .config("spark.sql.catalog.minio_lake.warehouse", "s3a://warehouse/iceberg") \
                .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
                .config("spark.hadoop.fs.s3a.access.key", "admin") \
                .config("spark.hadoop.fs.s3a.secret.key", "password123") \
                .config("spark.hadoop.fs.s3a.path.style.access", "true") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                .getOrCreate()

        print("\n=== 1. 이커머스 비즈니스 분석 파이프라인 엔진 가동 완료 ===")

        TOTAL_ORDERS = 10_000_000
        print(f"\n=== 2. DataFrame API: 다중 정형 로그 결합 및 파생 변수 가공 (1,000만 건) ===")
        start_time = time.time()

        # 가상 대용량 주문 원천 데이터프레임 생성 (8개 파티션 분할)
        raw_orders = spark.range(0, TOTAL_ORDERS, numPartitions=8)

        # [DataFrame 가공 1] 정형 코드 결합 및 수식 파생 변수 유도
        processed_orders = raw_orders.withColumn("order_id", expr("concat('ORD_', lpad(id, 8, '0'))")) \
                                    .withColumn("user_id", expr("concat('USR_', lpad(id % 50000, 5, '0'))")) \
                                    .withColumn("category", when(col("id") % 4 == 0, "Electronics")
                                                            .when(col("id") % 4 == 1, "Apparel")
                                                            .when(col("id") % 4 == 2, "Home")
                                                            .otherwise("Beauty")) \
                                    .withColumn("price", round(rand(seed=10) * 200 + 10, 2)) \
                                    .withColumn("quantity", (rand(seed=20) * 5 + 1).cast("int")) \
                                    .withColumn("total_amount", round(col("price") * col("quantity"), 2))

        # [DataFrame 가공 2] 기준 정보 마스터 데이터프레임(5만 명의 유저 등급 매핑) 생성
        raw_users = spark.range(0, 50000)
        user_master = raw_users.withColumn("user_id", expr("concat('USR_', lpad(id, 5, '0'))")) \
                            .withColumn("region", when(col("id") % 3 == 0, "Seoul")
                                                    .when(col("id") % 3 == 1, "Busan")
                                                    .otherwise("Incheon"))

        # [DataFrame 가공 3] 대용량 주문 테이블과 소용량 유저 마스터 테이블의 분산 결합(Join)
        final_ecommerce_df = processed_orders.join(user_master, on="user_id", how="inner")

        print(f"-> [성공] DataFrame API 융합 및 전처리 실행 계획 수립 완료 (소요시간: {time.time() - start_time:.2f}초)")

        # [Step 3] SparkSQL 테이블 인프라 정의 (ACID 트랜잭션 Iceberg 포맷)
        spark.sql("CREATE DATABASE IF NOT EXISTS minio_lake.ecommerce_db")
        spark.sql("DROP TABLE IF EXISTS minio_lake.ecommerce_db.order_analysis_ledger")
        spark.sql("""
                CREATE TABLE minio_lake.ecommerce_db.order_analysis_ledger (
                    user_id STRING,
                    order_id STRING,
                    category STRING,
                    price DOUBLE,
                    quantity INT,
                    total_amount DOUBLE,
                    region STRING
                ) USING iceberg
                PARTITIONED BY (category)
        """)

        print(f"\n=== 3. Action 호출: 융합 전처리 데이터 레이크하우스 병렬 저장 ===")
        write_start = time.time()

        # 지연 연산 해제 및 MinIO 스토리지 적재 (카테고리별 물리 디렉토리 자동 분할 적재)
        final_ecommerce_df.write \
                        .format("iceberg") \
                        .mode("append") \
                        .save("minio_lake.ecommerce_db.order_analysis_ledger")

        print(f"-> [성공] 1,000만 건 가공 데이터 레이크하우스 커밋 완료 (소요시간: {time.time() - write_start:.2f}초)")

        print(f"\n=== 4. SparkSQL: 다차원 분산 셔플링 집계 기반 비즈니스 매출 마트 생성 ===")
        query_start = time.time()

        # 고정된 저장소 인프라 테이블을 대상으로 대규모 네트워크 데이터 재배치(Shuffling) 집계 유발
        vip_summary = spark.sql("""
            SELECT 
                region,
                category,
                COUNT(DISTINCT user_id) as unique_users,
                SUM(quantity) as total_units_sold,
                ROUND(SUM(total_amount), 2) as aggregate_revenue,
                ROUND(AVG(total_amount), 2) as average_order_value
            FROM minio_lake.ecommerce_db.order_analysis_ledger
            GROUP BY region, category
            ORDER BY region ASC, aggregate_revenue DESC
        """)

        # 최종 결과 리포트 테이블 출력
        vip_summary.show(truncate=False)
        print(f"-> [성공] 다차원 셔플링 매출 분석 마트 연산 완료 (소요시간: {time.time() - query_start:.2f}초)")

        spark.stop()
    ```

    - 실행

    ```bash
    docker exec -it spark-client spark-submit /workspace/ecommerce_analytics.py
    ```


## 3. 종합 최적화 튜닝 포인트 및 활용 방향

- 기존 실습의 성공 로그(`result.txt`)를 복기하면, 1,000만 건 적재 과정에서 다음과 같은 최적화 메커니즘이 이미 완벽하게 작동함

    ```text
    # 성공 로그 내부의 핵심 최적화 팩트 지표
    26/06/24 15:23:05 INFO SparkWrite: Requesting ClusteredDistribution(factory_id) as write distribution
    26/06/24 15:23:10 INFO SparkWrite: Committing append with 2 new data files
    ```

<br>

> - **아키텍처 가이드라인**
>   1. **`ClusteredDistribution` 메커니즘의 의의:**
>       - Iceberg 포맷은
>           - 데이터를 쓸 때 물리적인 디렉터리 분기를 위해
>           - 내부적으로 **정렬 및 셔플링 연산(`ClusteredDistribution`)을 자동으로 유발**시킴
>       - 정교한 내부 분산 정렬 알고리즘 덕분에,
>           - 워커 1과 워커 2가 무작위로 MinIO에 파일을 기록하지 않고
>           - 정확히 서울 데이터와 부산 데이터를 노드별로 정렬하여
>           - 단 2개의 대형 Parquet 정렬 파일(`2 new data files`)로 커밋할 수 있었던 것
>
>   2. **최종 가공 파이프라인 가이드 요약 테이블:**
>
>   <div class="info-table">
>   <table>
>       <thead>
>           <th style="width: 170px;">최적화 타겟</th>
>           <th style="width: 220px;">핵심 설정 및 기법</th>
>           <th style="width: 520px;">실무 적용 의미</th>
>       </thead>
>       <tbody>
>           <tr>
>               <td class="td-rowheader">적재 성능 향상</td>
>               <td>PARTITIONED BY (column)</td>
>               <td>물리적 디렉토리 분할을 통한 런타임 쿼리 스캔 비용 절감 (파티션 프루닝)</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">네트워크 병목 최소화</td>
>               <td>spark.sql.shuffle.partitions</td>
>               <td>리소스 규모에 맞는 셔플 파티션 수 튜닝으로 무의미한 가상 태스크 양산 차단</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">메모리 오버헤드 방지</td>
>               <td>.repartition() / .coalesce()</td>
>               <td>연산 완료 후 분산 파일 조각들을 병합하여 가독성 증대 및 스토리지 I/O 부하 감소</td>
>           </tr>
>       </tbody>    
>   </table>
>   </div>
{: .summary-quote}
