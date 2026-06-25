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
    - 대용량 데이터 가공의 가장 큰 병목 지점인 셔플링(Shuffling)을 물리적으로 어떻게 제어하고 최적화할 수 있는지 증명하는 예제<br><br>

    - **예제의 구성**
        - 스마트팩토리의 대용량 센서 로그(1,000만 건)와 기준 정보인 장비 마스터 데이터(1,000건)를 이종 결합하는 동일한 연산을
        - 두 가지의 서로 다른 물리적 방식으로 실행하여 성능 차이를 초 단위로 직접 비교하도록 구성

    - **기반 기술 (Underlying Technologies)**
        - Spark SQL Optimizer (Catalyst):
            - 코드 단에서 주입된 broadcast() 힌트를 해석하여
            - 물리 실행 계획을 Shuffle Join에서 Broadcast Join으로 동적으로 전환

        - Tungsten Execution Engine:
            - 브로드캐스트된 해시 테이블을
            - CPU 캐시 레벨에서 고속 탐색하여
            - 메모리 내 매핑 연산 속도를 극대화

    - **중요성, 의미와 의의 (Significance & Impact)**
        - 하드웨어 친화적 튜닝 입증:
            - 네트워크 자원(Network Bandwidth)이 분산 컴퓨팅의 가장 비싼 자원임을 학습자에게 리포트 지표로 확실하게 각인시킴

        - 코드 한 줄의 가치:
            - 인프라를 증설하거나 하드웨어를 바꾸지 않고도,
            - 오직 .join(broadcast(small_master))라는 코드 한 줄의 최적화 힌트만으로
            - 시스템 성능을 수십 배 이상 끌어올릴 수 있다는 아키텍처적 주도권을 증명

    <br>

    - **전체 코드**

    ```python
    #//file: "shuffle_join_tuning.py"
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, expr, broadcast
    import time

    if __name__ == "__main__":
        spark = SparkSession.builder \
                .appName("Spark-Shuffle-Join-Tuning-Lab") \
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

        print("\n=== 🚀 1. 셔플링 조인 튜닝 실험 엔진 가동 완료 ===")

        # 대용량 스마트팩토리 로그 생성 (1,000만 건)
        print("\n⏳ [케이스 A] 1,000만 건 대용량 센서 로그 생성 중...")
        large_logs = spark.range(0, 10000000, numPartitions=8) \
                        .withColumn("equipment_id", expr("concat('EQ_', lpad(id % 1000, 4, '0'))")) \
                        .withColumn("metrics", rand(seed=42) * 100)

        # 극소용량 마스터 데이터 생성 (장비 명칭 매핑 - 1,000건)
        print("⏳ [케이스 B] 1,000건 소용량 장비 마스터 정보 생성 중...")
        small_master = spark.range(0, 1000) \
                            .withColumn("equipment_id", expr("concat('EQ_', lpad(id, 4, '0'))")) \
                            .withColumn("equipment_name", expr("concat('Sensor_Node_Alpha_', id)")) \
                            .drop("id")

        print("\n------------------------------------------------------------")
        print("🚨 실험 1: 일반 셔플 조인 (Shuffle Hash Join) 실행")
        print("-> 데이터를 정렬하기 위해 워커 노드 간 전 대역폭 네트워크 셔플링이 발생합니다.")
        print("------------------------------------------------------------")
        
        # 일반 조인 시 강제로 셔플링 파티션을 많이 잡아서 병목을 체감하도록 유도
        spark.conf.set("spark.sql.shuffle.partitions", "200")
        
        start_shuffle = time.time()
        # 일반적인 조인 수행 (네트워크 데이터 이동 유발)
        bad_join_result = large_logs.join(small_master, on="equipment_id", how="inner")
        
        # Action을 호출하여 실제 분산 연산 및 셔플링 강제 구동
        shuffle_count = bad_join_result.count()
        duration_shuffle = time.time() - start_shuffle
        print(f"🚩 [일반 셔플 조인 완료] 총 {shuffle_count:,}건 매핑 완료 (소요시간: {duration_shuffle:.2f}초)")

        print("\n------------------------------------------------------------")
        print("✨ 실험 2: 최적화 브로드캐스트 조인 (Broadcast Hash Join) 실행")
        print("-> 소용량 마스터를 메모리에 복제 배포하여 네트워크 셔플링을 '원천 차단'합니다.")
        print("------------------------------------------------------------")
        
        start_broadcast = time.time()
        # 💡 broadcast() 힌트를 사용하여 최적화 조인 수행 (셔플링 단계를 완전히 생략)
        good_join_result = large_logs.join(broadcast(small_master), on="equipment_id", how="inner")
        
        broadcast_count = good_join_result.count()
        duration_broadcast = time.time() - start_broadcast
        print(f"🚩 [브로드캐스트 조인 완료] 총 {broadcast_count:,}건 매핑 완료 (소요시간: {duration_broadcast:.2f}초)")

        print("\n" + "="*60)
        print(f"📊 최종 성능 개선 결과 리포트")
        print(f"  - 일반 셔플 조인 소요시간: {duration_shuffle:.2f}초")
        print(f"  - 브로드캐스트 조인 소요시간: {duration_broadcast:.2f}초")
        print(f"🔥 네트워크 셔플링 제어를 통한 속도 향상: 약 {duration_shuffle / max(duration_broadcast, 0.01):.1f}배 가속")
        print("="*60 + "\n")

        spark.stop()
    ```

    - 실행

    ```bash
    docker exec -it spark-client spark-submit /workspace/ecommerce_analytics.py
    ```


    - **단계별 코드 상세 설명 (Step-by-Step Walkthrough)**

        - **[Step 1] 대소용량 소스 데이터프레임 빌드**

            ```python
            large_logs = spark.range(0, 10000000, numPartitions=8) ...
            small_master = spark.range(0, 1000) ...
            ```

            - 실험의 대칭성을 위해 8개 파티션으로 쪼개진 **1,000만 건의 헤비(Heavy) 데이터**와 단 1개의 파티션으로도 가뿐한 **1,000건의 라이트(Light) 데이터**를 메모리상에 독립적으로 생성

        - **[Step 2] 실험 1: 일반 셔플 조인 강제 구동**

            ```python
            spark.conf.set("spark.sql.shuffle.partitions", "200")
            bad_join_result = large_logs.join(small_master, on="equipment_id", how="inner")
            shuffle_count = bad_join_result.count()
            ```

            - **`spark.sql.shuffle.partitions = 200`:**
                - 스파크의 기본 셔플 파티션 수를 명시
                - 쿼리가 실행될 때 워커 노드들이 네트워크 조인 분기를 위해 무려 200개의 임시 태스크 조각을 쪼개고 나르는 무거운 물리적 이사(Shuffling) 과정을 겪도록 유도
            * **`.count()` (Action):**
                - 지연 연산되어 있던 조인 계보를 깨우고
                - 물리적인 셔플링 연산을 실시간 가동하는 트리거 역할 수행

        - **[Step 3] 실험 2: 브로드캐스트 최적화 조인 구동**

            ```python
            good_join_result = large_logs.join(broadcast(small_master), on="equipment_id", how="inner")
            broadcast_count = good_join_result.count()
            ```

            - **`broadcast(small_master)`:**
                - 본 예제의 핵심 심장부
                - 스파크 최적화 엔진에게 **"이 소형 테이블은 셔플링하지 말고 모든 워커 노드 메모리에 복제본으로 미리 얹어줘"**라고 힌트를 하달
                    - 이 명령 덕분에 200개의 셔플 파티션 네트워크 교환 단계가 물리 실행 계획에서 **통째로 소멸**
                    - 워커 노드가 가진 CPU 코어들이 로컬 메모리 안에서 단 몇 초 만에 1,000만 건을 초고속 패스 처리하게 됨

        - **[Step 4] 최종 성능 비교 리포트 출력**

            ```python
            print(f"🔥 네트워크 셔플링 제어를 통한 속도 향상: 약 {duration_shuffle / duration_broadcast:.1f}배 가속")
            ```

            - 두 연산의 순수 런타임 시간을 계산
            - 최종 효율성 지표를 콘솔 테이블로 직관적으로 출력하고
            - 자원을 반납(`spark.stop()`)


## 3. 종합 최적화 튜닝 포인트 및 활용 방향

- 기존 실습의 성공 로그(`result.txt`)를 복기하면, 1,000만 건 적재 과정에서 다음과 같은 최적화 메커니즘이 이미 완벽하게 작동함

    ```text
    # 성공 로그 내부의 핵심 최적화 팩트 지표
    26/06/24 15:23:05 INFO SparkWrite: Requesting ClusteredDistribution(factory_id) as write distribution
    26/06/24 15:23:10 INFO SparkWrite: Committing append with 2 new data files
    ```

<br>

- **셔플 조인(Shuffle Join)의 한계**
    - 분산 클러스터 환경에서 두 테이블을 결합(JOIN)할 때,
    - 스파크는 기본적으로 양쪽 테이블의 데이터를 결합 키(Key) 기준으로 재정렬하기 위해
    - 전국의 워커 노드 간 네트워크로 데이터를 교환하는 셔플 해시 조인(Shuffle Hash Join)을 수행함
    - 데이터가 1,000만 건 단위가 되면 네트워크 대역폭 폭발과 디스크 I/O 병목으로 인해 연산 속도가 급격히 떨어짐

- **브로드캐스트 조인 (Broadcast Join)의 혁신**
    - 결합하려는 두 테이블 중 하나가 메모리에 가뿐히 올라갈 정도로 작은 소용량 마스터 테이블(예: 1,000건 장비 정보)이라면, 굳이 무거운 셔플링을 할 필요가 없음
    - 스파크 드라이버가 이 작은 테이블의 복제본을 모든 워커 노드의 메모리로 미리 통째로 배포(Broadcast)해 버리는 방식
        - 각 워커 노드는 네트워크 이동 없이 자신이 가진 대용량 파티션 조각과 메모리 상의 마스터 데이터를 로컬에서 즉시 매핑(Map-side Join)
        - 분산 셔플링을 원천 차단하여 극적인 속도 향상을 이루어냄


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
