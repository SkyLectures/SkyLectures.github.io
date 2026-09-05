---
layout: page
title:  "Spark DataFrame, SparkSQL을 이용한 데이터 가공"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-04-01_01-SparkDataFrameSql
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - 개발자는 프로그래밍 방식(DataFrame API)과 선언적 방식(SQL) 중 편리한 방법을 선택하여 데이터를 처리할 수 있음
{: .common-quote}
 
## 1. Spark DataFrame

- **개념 및 정의**
    - 분산 컴퓨팅 환경에서 데이터를 정형화된 테이블 형태로 다루기 위한 최상위 수준의 분산 데이터 컬렉션(Distributed Data Collection)
        - R이나 Python의 pandas DataFrame과 유사하게 구조화된 컬럼과 행으로 이루어짐

    - 대용량 데이터 가공에 가장 최적화된 API
        - 내부적으로 가상머신(JVM) 레벨에서 작동
        - 카탈리스트 옵티마이저(Catalyst Optimizer)를 통해 연산 엔진이 실행 계획을 최적화함

- **기술적 정의와 아키텍처**
    - RDD(Resilient Distributed Dataset)
        - 초기 버전에 사용됨
        - 내부 자바 객체를 가공하는 로우 레벨 API

    - DataFrame
        - RDD에 스키마(Schema)라는 정형화된 틀을 얹은 데이터 구조
        - 관계형 데이터베이스(RDB)의 테이블이나 Python의 Pandas DataFrame과 개념적으로 유사함
        - 단일 메모리가 아닌 여러 워커 노드의 JVM(Java Virtual Machine) 메모리에 쪼개져서 관리된다는 점이 근본적인 차이

- **핵심 매커니즘**
    - **강력한 정적 타입 스키마(Schema)**
        - 컬럼명과 컬럼별 데이터 타입(LONG, STRING, DOUBLE 등)이 메타데이터로 고정되어 있음<br>
            🡲 스파크 엔진이 데이터의 물리적 구조를 미리 파악하고 최적화할 수 있는 기술적 기반

    - **지연 연산(Lazy Evaluation)과 계보(Lineage)**
        - DataFrame에 filter(), select(), withColumn() 등의 가공 명령(Transformation)을 내려도 즉시 계산되지 않음
            1. 연산들의 연결 고리인 내부 계보(Lineage)만 빌드
            2. 최종 출력이나 저장을 지시하는 액션(Action) 명령(show(), save())이 떨어지는 순간
            3. 최적의 경로로 한 번에 실행

    - **오프힙(Off-Heap) 메모리 관리 (Tungsten 프로젝트)**
        - 자바 가상머신 힙 영역 외부의 Raw Byte 배열 상태로 데이터를 관리 🡲 메모리 효율성과 연산 속도가 극대화됨
            - 기술 적용의 이유: 자바 객체 고유의 가비지 컬렉션(GC) 오버헤드를 피하기 위해서


## 2. SparkSQL

- **개념 및 정의**
    - 정형 데이터를 가공하기 위해 SQL 질의 구문을 직접 사용할 수 있도록 지원하는 스파크의 핵심 모듈
        - 구조화된 정형 데이터를 한층 더 쉽고 강력하게 다루기 위해
        - ANSI SQL 문법 및 관계형 대수(Relational Algebra) 연산을 지원
    - 데이터프레임과 SQL 스키마가 상호 완벽하게 호환됨

- **기술적 정의와 의의**
    - 코드 프로그래밍 방식(DataFrame API)과 선언적 질의 방식(SQL 구문)을 기술적으로 100% 동일하게 융합한 핵심 엔진
        - 단순히 "스파크 위에서 SQL 질의문을 던질 수 있다" 수준을 뛰어넘음
    - 개발자가 df.select("name")이라고 파이썬 코드를 짜든, spark.sql("SELECT name FROM ...")이라고 SQL 문장을 적든, SparkSQL 엔진 내부에서는 완벽하게 동일한 실행 계획으로 변환되어 처리됨

- **핵심 매커니즘**
    - **엔진의 심장: 카탈리스트 옵티마이저 (Catalyst Optimizer)**
        - SparkSQL이 고성능을 내는 이유
            - 내부에 카탈리스트 옵티마이저라는 지능형 실행 계획 최적화 엔진이 탑재되어 있기 때문
            - SQL이나 DataFrame 명령이 접수되면 🡲 다음 4단계를 거쳐 🡲 물리적 분산 명령으로 재컴파일됨

        - 프로세스
            1. **분석 (Analysis)**
                - 쿼리문 구문을 파악 🡲 카탈로그 메타데이터와 대조 🡲 실제 데이터베이스나 테이블, 컬럼명이 존재하는지 확인
                    - 미존재 시 AnalysisException 발생
            2. **논리적 계획 최적화 (Logical Plan Optimization)**
                - 관계형 대수 규칙을 적용하여 쿼리를 최적화함

                - 푸시다운 프리디케이트(Pushdown Predicate):
                    - 데이터를 다 읽은 뒤 필터링하는 것이 아니라,
                    - 파일(MinIO/Parquet)을 읽는 단계부터 조건문(WHERE)을 결합하여 필요한 데이터만 걸러서 메모리에 올림

                - 프로젝션 푸시다운(Projection Pushdown):
                    - 테이블에 컬럼이 아무리 많이 있어도,
                    - 쿼리에서 쓰는 컬럼 몇 개만 콕 집어서 파일 레이어에서 읽어 들임

            3. **물리적 계획 수립 (Physical Planning)**
                - 실제 클러스터 자원 스펙에 맞게
                - 워커 노드들이 코어별로 어떤 방식으로 데이터를 쪼개고, 조인하고, 셔플링할 것인지
                - 구체적인 물리 실행 계획들을 연산함

            4. **코드 생성 (Code Generation)**
                - 최적의 물리 계획이 선택되면
                - 자바 바이트코드(Java Bytecode)를 실시간으로 자동 생성(Tungsten 기법)하여
                - 가상머신 위에서 기계어 수준의 고속 병렬 연산이 일어나도록 만듦



## 3. 대용량 데이터 가공의 핵심 메커니즘

- 대용량 데이터 가공의 핵심 심장부인 지연 연산(Lazy Evaluation)과 분산 셔플링(Shuffling)은 스파크의 물리적 효율성을 결정짓는 양대 축

### 3.1 지연 연산 (Lazy Evaluation)

> - Spark DataFrame의 연산은 데이터 변환(Transformation)과 실행(Action)으로 구분됨
> - 데이터 가공 로직을 짤 때는 실제 연산이 수행되지 않고 실행 계획만 수립되며,
> - 최종 데이터를 수집하거나 저장하는 시점에 최적의 경로로 단 한 번에 분산 실행됨
{: .common-quote}

- **상세 설명 및 작동 방식**
    - 지연 연산
        - 데이터 가공 명령이 떨어졌을 때
        - 즉시 물리적인 연산을 수행하지 않고,
        - 최종 결과가 필요할 때까지 계산을 미루는 메커니즘

    - 스파크 API는 크게 두 가지 연산으로 나뉨
        - 변환 연산 (Transformation)
            - filter(), select(), join(), groupBy() 등 데이터를 바꾸는 작업들
            - 이 명령들은 호출되어도 실제 데이터를 1줄도 건드리지 않음
            - 내부적으로 논리적 실행 계획(Logical Plan)인 계보(Lineage)만 차분히 누적 기록
        - 액션 연산 (Action)
            - show(), count(), save() 등 최종 결과를 화면에 출력하거나 디스크에 쓰는 작업들
            - 액션 연산이 호출되는 바로 그 순간,
            - 쌓여있던 계보(Lineage)를 깨우고 최적화 엔진을 가동하여 비로소 워커 노드들에게 실제 분산 연산 명령을 하달

- **DataFrame, SparkSQL에서의 작동 위치**
    - 두 인터페이스 모두 지연 연산 바운더리 안에서 작동함
    - 내부의 카탈리스트 옵티마이저(Catalyst Optimizer)라는 하나의 지연 연산 최적화 창구로 수렴됨<br><br>

    - DataFrame API
        - 파이썬이나 자바 코드로 df.filter(...).select(...)와 같이 메서드 체이닝을 작성하는 과정 전체가
        - 지연 연산의 계보(Lineage)를 빌드하는 과정
    - SparkSQL
        - spark.sql("SELECT ... FROM ... WHERE ...") 문장을 실행하는 순간 파싱(Parsing)이 일어나며
        - 지연 연산 구조의 추상 구문 트리(AST) 뼈대가 형성됨

    
### 3.2 분산 셔플링 (Distributed Shuffling)

> - `GROUP BY` 나 `JOIN` 같이 특정 기준에 따라 데이터를 재정렬할 때,
> - 여러 워커 노드 간에 네트워크를 통해 데이터를 주고받는 물리적 재배치 현상
> - 분산 처리의 성능을 결정짓는 가장 무겁고 중요한 단계
{: .common-quote}

- **상세 설명 및 작동 방식**
    - 분산 셔플링:
        - 데이터의 기준(Key)을 재정렬하기 위해
        - 클러스터 내부의 여러 워커 노드 간에 네트워크를 통해
        - 데이터를 대대적으로 교환하고 이동시키는 물리적 메커니즘

    - 스파크의 기본 물리 파티션들은 공장 ID나 센서 ID와 상관없이 무작위 분할되어 있을 확률이 높음<br>
        🡲 이 상태에서 GROUP BY factory_id나 서로 다른 테이블의 JOIN을 수행하려면,
        - 전국의 워커 노드에 흩어져 있는 동일한 factory_id 그룹의 행들을 물리적으로 단 하나의 특정 워커 노드 메모리 공간으로 모아주어야 그룹핑 연산이 성립됨
        - 이 때문에 데이터가 네트워크망을 타고 이리저리 이동하는 셔플링이 발생함

- **DataFrame, SparkSQL에서의 작동 위치**
    - 셔플링은 API 계층이 아닌, 스파크의 가장 하단부인 물리적 런타임 레이어(Physical Runtime Layer)에서 작동
        - DataFrame의 .repartition(), .join(), .groupBy() 메서드를 호출하거나,
        - SparkSQL 구문에서 GROUP BY, JOIN, DISTINCT, ORDER BY 문장을 선언했을 때,
        - **카탈리스트 옵티마이저**가 논리 계획을 해석하여
        - 물리 실행 단계(Stage 분기점)에서 ShuffleMapStage를 강제로 생성하며 구동됨


### 3.3 지연 연산과 분산 셔플링의 연동

- 이 두 기술은 독립적으로 놀지 않고, "지연 연산이 두뇌(전략) 역할을 하고, 셔플링이 팔다리(물리 실행) 역할을 수행"하며 유기적으로 맞물려 돌아감

```
[ 연동 파이프라인 매핑 구조 ]
1. 유저가 SQL / DataFrame 가공 명령 입력 (Transformation)
   🡳 
2. 지연 연산 가동: 최적화 최적 경로 탐색 (카탈리스트 최적화 실행 계획 수립)
   🡳 [최적화 완료 및 Action 명령 투입]
3. 셔플링 경계선 분기: Stage 단층 쪼개기 (안전한 로컬 연산 구간과 무거운 네트워크 구간 분리)
   🡳 
4. 물리 실행: 무의미한 셔플링 데이터를 파일 레이어에서 사전 차단 (Predicate Pushdown)
   🡳 
5. 최종 결과물: 완벽하게 최소화된 고속 병렬 쓰기 (MinIO 레이크하우스 커밋)
```

1. **연동 프로토콜: Stage 분기와 최적화 최적화**
    - 지연 연산 덕분에 스파크는 전체 가공 파이프라인의 그림을 미리 내려다볼 수 있음
    - 액션 명령이 떨어지면 🡲 스파크는 계보를 분석하여 🡲 **"셔플링이 일어나는 지점"을 기준**으로 🡲 **연산 단계를 스테이지(Stage)라는 단위로 엄격하게 분기**함

        - **Stage 0 (Narrow Dependency):**
            - 네트워크 이동 없이,
            - 각 워커 노드가 본인 메모리에 들고 있는 파티션 조각만 가지고
            - 고속으로 필터링(`filter`)이나 가공(`withColumn`)을 수행하는 구간

        - **Stage 1 (Wide Dependency):**
            - Stage 0의 결과물들이 셔플링 경계선을 만나 네트워크 전송을 완료한 후,
            - 집계(`GROUP BY`) 연산을 완수하는 구간

2. **상호 연동이 만들어내는 최종 결과물**

    - 지연 연산이 파이프라인을 통제 🡲 "셔플링해야 할 데이터의 양 자체를 물리적으로 최소화"하는 시너지 효과 발휘
        - 예시
            - 10억 건 중 9억 건을 버리는 필터링과 셔플링 집계가 묶여 있다면,
            - 지연 연산 엔진은 필터링을 파일 레이어까지 끌고 내려가서(`Predicate Pushdown`) 1억 건만 읽어 들이고,
            - **결과적으로 네트워크 셔플링 대역폭 부하를 10분의 1로 극적으로 깎아내는 무결점의 고속 분산 연산 결과**를 도출


### 3.4 그 외의 대용량 가공을 지탱하는 메커니즘

- **AQE (Adaptive Query Execution - 적응형 쿼리 실행)**
    - 스파크 3.0부터 도입된 현대 스파크 최적화의 꽃
        - 과거: 데이터가 실행되기 전에 수립된 실행 계획을 융통성 없이 끝까지 밀고 나감
        - AQE: **지연 연산이 풀려 Stage 0번 연산(Shuffle Write)이 끝난 시점의 실제 중간 데이터 통계치(크기, 편향도)를 실시간으로 모니터링**

    - 중간 데이터가 생각보다 너무 작다면
        - 다음 스테이지의 셔플 파티션 수(기본 200개)를 자동으로 4개나 8개로 병합(`Coalesce`)하여
        - 파일 조각 찌꺼기가 남는 것을 방지함

    - 특정 워커 노드에 데이터가 쏠려 있다면
        - 파티션을 동적으로 쪼개어
        - **데이터 편향(Skew Joint) 현상을 실시간으로 자동 치유**

- **전체 단계 코드 생성 (Whole-Stage Code Generation - 텅스텐 엔진)**
    - 자바 가상머신(JVM) 기반인 스파크 🡲 데이터 가공 시 수천만 번의 객체 생성과 가비지 컬렉션(GC) 부하에 의한 시각적인 한계
    - 스파크 텅스텐(Tungsten) 엔진
        - DataFrame의 여러 가공 단계를 분리된 함수로 실행하지 않고,
        - 하나의 거대한 `for` 루프 기계어 바이트코드로 실시간 합성(Compile)하여 실행
        - 메모리 포인터 연산을 직접 수행하므로 가상머신의 한계를 초과하는 극한의 하드웨어 가속 성능을 보장함

- **저장소 레이어와의 연동 최적화: 버킷팅(Bucketing) 및 파일 스킵**
    - **Iceberg 테이블 포맷**과 결합할 때 극대화되는 기술
    - 데이터 가공 후 저장할 때 특정 키 기준으로 정렬 및 파일 분할 처리를 해두면,
        - 다음번 가공 파이프라인에서 데이터를 읽어 들일 때
        - 대규모 셔플링 과정을 생략하고 곧바로 병렬 처리에 진입 가능 🡲 토탈 인프라 처리 성능이 수십 배 이상 증가


## 4. 실습 예제

> - **시나리오: 이커머스 다중 로그 결합 및 VIP 분석**
>   - 실무에서 가장 많이 쓰이는 "대규모 고객 주문 데이터"와 "회원 기본 정보"라는 서로 다른 두 소스 데이터를 가공하는 시나리오
>   - **DataFrame API**
>       - 대량의 주문 원천 데이터(1,000만 건)의 파싱, 결합, 파생 컬럼 생성 등 정교한 전처리를 수행
>   - **SparkSQL**
>       - 전처리된 데이터 레이크하우스를 타겟으로 셔플링 연산(`GROUP BY` 및 고도화된 집계)을 수행
>       - 비즈니스 분석 마트 생성
{: .common-quote}


### 4.1 예제 소스코드

- 기존 인프라 환경(내장된 격리 경로의 Jar 파일 및 MinIO catalog)에서 그대로 동작함 🡲 워크스페이스에 이 파일만 생성하여 실행

```python
#//file: "ecommerce_analytics.py"
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, expr, round
import time

# 1. Spark 세션 기동 (내장 라이브러리 기반 안전 가동)
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

print("\n=== 1. 이커머스 분석 파이프라인 엔진 가동 완료 ===")

TOTAL_ORDERS = 10_000_000
print(f"\n=== 2. DataFrame API: 대규모 대용량 데이터프레임 가공 (1,000만 건) ===")
start_time = time.time()

# 가상 대용량 주문 raw 데이터 생성 (8개 파티션 분할)
raw_orders = spark.range(0, TOTAL_ORDERS, numPartitions=8)

# [DataFrame 가공 1] 문자열 결합, 조건 분기, 타입 캐스팅 및 수학적 파생 변수 생성
processed_orders = raw_orders.withColumn("order_id", expr("concat('ORD_', lpad(id, 8, '0'))")) \
                             .withColumn("user_id", expr("concat('USR_', lpad(id % 50000, 5, '0'))")) \
                             .withColumn("category", when(col("id") % 4 == 0, "Electronics")
                                                    .when(col("id") % 4 == 1, "Apparel")
                                                    .when(col("id") % 4 == 2, "Home")
                                                    .otherwise("Beauty")) \
                             .withColumn("price", round(rand(seed=10) * 200 + 10, 2)) \
                             .withColumn("quantity", (rand(seed=20) * 5 + 1).cast("int")) \
                             .withColumn("total_amount", round(col("price") * col("quantity"), 2))

# [DataFrame 가공 2] 분석용 마스터 데이터프레임(소용량 유저 등급 매핑 테이블) 생성
raw_users = spark.range(0, 50000)
user_master = raw_users.withColumn("user_id", expr("concat('USR_', lpad(id, 5, '0'))")) \
                       .withColumn("region", when(col("id") % 3 == 0, "Seoul")
                                             .when(col("id") % 3 == 1, "Busan")
                                             .otherwise("Incheon"))

# [DataFrame 가공 3] 대용량 주문 데이터와 소용량 마스터 데이터의 분산 결합(Join)
# 지연 연산 덕분에 이 시점에는 계획만 수립되며, 실제 데이터 결합은 최종 적재 시점에 최적화되어 처리됩니다.
final_ecommerce_df = processed_orders.join(user_master, on="user_id", how="inner").drop("id")

print(f"-> [성공] DataFrame API 조인 및 전처리 계획 빌드 완료 (소요시간: {time.time() - start_time:.2f}초)")

# 3. SparkSQL 테이블 인프라 선언 (ACID 트랜잭션 Iceberg 포맷)
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

print(f"\n=== 3. Action 호출: 가공된 분산 데이터 레이크하우스 병렬 저장 ===")
write_start = time.time()

# 지연 연산 해제 및 MinIO 스토리지 적재 (카테고리별 물리 파티션 자동 분기)
final_ecommerce_df.write \
                  .format("iceberg") \
                  .mode("append") \
                  .save("minio_lake.ecommerce_db.order_analysis_ledger")

print(f"-> [성공] 1,000만 건 정제 데이터 레이크 적재 완료 (소요시간: {time.time() - write_start:.2f}초)")

print(f"\n=== 4. SparkSQL: 대규모 셔플링 집계 연산 및 연계 비즈니스 마트 생성 ===")
query_start = time.time()

# [SparkSQL 가공] 데이터 레이크에 저장된 영구 테이블을 대상으로 고난도 다차원 집계 수행
# 대규모 워커 노드 간의 네트워크 데이터 재배치(Shuffling)를 유발하여 최종 통계 매트릭스 도출
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

# 최종 가공 데이터 리포트 출력
vip_summary.show(truncate=False)
print(f"-> [성공] 지역/카테고리별 매출 분석 마트 연산 완료 (소요시간: {time.time() - query_start:.2f}초)")

spark.stop()
```


### 4.2 단계별 데이터 가공 메커니즘 상세 설명

- **[Step 1] DataFrame API를 이용한 복합 전처리 변환**

    - **`expr("concat(...)")` 및 `lpad()`:**
        - 가상 시퀀스 숫자(`id`)를 활용
        - 실제 엔터프라이즈 환경에서 사용되는 정형화된 코드 형태(`ORD_00000001`, `USR_00001`)로 데이터를 포맷팅 가공

    - **`when().otherwise()` 구문:**
        - 다중 조건 분기를 설정
        - 카테고리 도메인 데이터와 지역 코드 데이터를 동적으로 생성하는 실무형 비즈니스 로직

    - **파생 변수 수식 계산:**
        - 단품 가격(`price`)과 구매 수량(`quantity`) 컬럼을 결합
        - 총 주문 금액(`total_amount`)이라는 새로운 비즈니스 지표 컬럼을 계산 규칙에 맞게 유도

- **[Step 2] 데이터프레임 간 대규모 분산 결합 (Join)**

    ```python
    final_ecommerce_df = processed_orders.join(user_master, on="user_id", how="inner")
    ```

    - **지연 연산의 가치 실현:**
        - 1,000만 건의 주문 내역과 5만 건의 회원 마스터를 `user_id` 기준으로 결합
        - 지연 연산 덕분에 이 시점에는 실제 조인이 일어나 자원이 낭비되지 않으며,
        - 내부 **카탈리스트 옵티마이저**가 최적의 결합 실행 계획을 수립한 뒤
            - 예: 소용량 유저 테이블을 워커 노드 메모리에 복제하여 셔플링을 최소화하는 알고리즘 등
        - 하단의 저장 단계(`write`)에서 단 한 번에 고속 연산 처리

- **[Step 3] 스토리지 파티셔닝 기반 병렬 Write**

    - DDL 선언부에서 **`PARTITIONED BY (category)`** 설정 부여
    - 워커 노드 1, 2가 1,000만 건 데이터를 나누어 처리한 후,
        - MinIO 저장소에 최종 파일들을 기록할 때
        - 상품 카테고리명 기준(`category=Electronics`, `category=Apparel` 등)으로
        - 파일 디렉토리를 물리적으로 분할 정렬하여 저장

- **[Step 4] SparkSQL 다차원 셔플링 집계 연산**

    ```sql
    SELECT region, category, COUNT(DISTINCT user_id), SUM(total_amount) ...
    GROUP BY region, category
    ```

    - 적재 완료된 대규모 데이터 레이크하우스를 대상으로 다차원 그룹핑 연산을 직접 수행
    - 전국의 워커 노드가 각자 보관 중인 로컬 Parquet 파일에서 지역별, 카테고리별 중간 합계를 계산한 뒤,
        - 이 데이터들을 네트워크망을 통해 상호 재배치하는 **물리적 셔플링(Shuffling)** 과정을 거침
    - 최종적으로 지역별 고유 사용자 수(`COUNT(DISTINCT)`), 누적 총 매출액(`SUM`), 평균 주문 단가(`AVG`) 매트릭스를 초고속으로 계산해 출력


> - 이 예제는 단순한 난수 생성을 넘어 
>   - **원천 데이터 전처리 🡲 이기종 테이블 결합(Join) 🡲 파티션 레이크하우스 구축 🡲 고부하 셔플링 대시보드 집계 SQL**로 이어지는
>   - 완벽한 데이터 엔지니어링 파이프라인의 핵심 가공 문법을 모두 담고 있음
{: .summary-quote}