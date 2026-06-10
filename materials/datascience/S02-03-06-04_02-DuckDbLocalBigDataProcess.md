---
layout: page
title:  "DuckDB를 이용한 로컬 대용량 데이터 처리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-04_02-DuckDbLocalBigDataProcess
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - **기존의 데이터 분석 아키텍처**
>   - 수십~수백 GB의 대용량 데이터를 처리하기 위해 Hadoop, Spark 같은 **분산 컴퓨팅 시스템**이나 대형 클라우드 데이터 웨어하우스를 구축<br> 🡲 이는 복잡한 인프라 설정과 높은 비용을 수반함<br><br>
> - **DuckDB의 철학**
>   - **단일 PC의 자원을 극한으로 활용하여 대용량 데이터를 처리한다**
>   - 가벼운 라이브러리인 DuckDB가 어떻게 기존의 거대한 분산 시스템(Spark, Hadoop)이나 클라우드 데이터 웨어하우스(BigQuery, Snowflake)의 전유물이었던 '빅데이터급 처리 성능'을 단 한 대의 노트북에서 이끌어내는지 **이론적 배경과 내부 아키텍처**를 이해하는 것이 좋음
{: .common-quote}


## 1. DuckDB 기반 로컬 대용량 데이터 처리 기술

- **하드웨어 트렌드의 변화: "대부분의 데이터는 생각보다 작다"**

    > - 과거 2010년대 초반 빅데이터 붐이 일었을 때는 '분산 컴퓨팅(Hadoop, Spark)'이 무조건적인 정답이었음
    > - 현대에 이르러 하드웨어 스펙이 기하급수적으로 발전하면서 상황이 바뀜

    - **하드웨어의 상향평준화:**
        - 일반적인 노트북도 16GB~32GB의 RAM과 다중 코어(Multi-core) CPU, 그리고 초고속 NVMe SSD를 탑재

    - **실제 데이터의 규모:**
        - 전 세계 대기업의 일일 분석 데이터 중 90% 이상은 수십 기가바이트(GB) 미만
        - 이는 **단일 컴퓨터의 자원으로도 충분히 소화할 수 있는 영역**

    - **분산 시스템의 오버헤드:**
        - 네트워크를 통해 여러 컴퓨터로 데이터를 쪼개고 다시 합치는 분산 시스템은 데이터 전송 속도(Network I/O) 때문에 오히려 소~중규모 데이터 분석에서 로컬 컴퓨터 한 대보다 느린 병목 현상이 발생함

    - DuckDB는 **"네트워크 비용을 없애고 단일 PC의 하드웨어 능력을 100% 쥐어짜서 분산 서버보다 빠르게 처리하자"**는 이론에서 출발함


- **행 중심(Row-oriented) vs 열 중심(Columnar) 아키텍처**
    - 데이터베이스가 디스크와 메모리에 데이터를 배치하는 구조적 차이
    - 기존 아키텍처와 열 중심 아키텍터의 비교
        - **전통적인 DB (OLTP):**
            - 데이터를 가로(Row) 방향으로 연속해서 저장
            - 하나의 레코드를 통째로 읽고 쓰는 데 유리함
            - 대량 분석 시 심각한 병목이 발생함
                - 수억 건의 데이터 중 오직 '매출액' 컬럼 하나만 합산 🡲 이름, 주소, 날짜 등 불필요한 데이터를 모두 디스크에서 읽어와야 함

        - **DuckDB (OLAP):**
            - 데이터를 세로(Column) 방향으로 묶어서 저장
            - '매출액' 컬럼만 필요하다면 디스크에서 딱 그 매출액 데이터가 저장된 블록만 집어서 읽어옴
                - **디스크 I/O(입출력)량이 수십에서 수백 분의 일로 감소**


- **벡터화 쿼리 실행 엔진 (Vectorized Query Execution)**
    - 데이터를 읽어온 후, CPU 내부에서 연산할 때 효율성을 극대화하는 이론

    - **기존 모델과 벡터화 모델의 비교**
        - **화산 모델 (전통적 방식)**
            - 데이터를 한 건(Row)씩 가져와서 계산 필터를 거치고 다음 단계로 넘김
            - 1,000만 건을 처리하려면 🡲 CPU 내부적으로 다음 데이터가 올 때까지 기다리는 오버헤드(함수 호출, 조건문 분기 예측 실패 등)가 1,000만 번 발생

        - **벡터화 모델 (DuckDB 방식)**
            - 데이터를 약 2,048개의 열 데이터 묶음인 **'벡터(Vector)'** 단위로 쪼개어 CPU 캐시에 올림
            - 현대 CPU의 **SIMD** 기술을 활용 🡲 2,048개의 데이터를 단 몇 번의 CPU 사이클만으로 한꺼번에 연산
                - SIMD (Single Instruction Multiple Data): 하나의 명령어로 여러 개의 데이터를 동시에 계산하는 하드웨어 가속 기능
            - 이를 통해 CPU가 쉬지 않고 풀가동되며 계산 능력이 극대화됨


- **스마트 메모리 관리와 디스크 스필러**
    - 로컬 대용량 데이터 처리의 가장 큰 문제는 Out-of-Memory (OOM, 메모리 부족으로 프로그램이 강제 종료되는 현상)
        - Python의 Pandas가 대용량 데이터에서 자주 터지는 고질적인 문제
        - DuckDB는 이를 알고리즘적으로 방어함

    - **처리 방식**
        - **블록 단위 스트리밍:**
            - 100GB짜리 파일이 있더라도 이를 한 번에 메모리에 올리지 않음
            - 내부 버퍼 매니저가 감당할 수 있는 크기의 블록으로 나누어 순차적으로 스트리밍하며 처리

        - **외부 정렬 및 디스크 스필러 (External Aggregation/Sort):**
            - `GROUP BY`나 `ORDER BY` 같은 무거운 연산을 수행할 때
                - 가용 RAM이 가득 차면, DuckDB는 연산 중인 데이터의 일부를 로컬 디스크(SSD)에 임시 파일 형태로 안전하게 기록(`Spilling`)
                - 연산이 끝나면 이 파편들을 다시 병합
                - 성능은 약간 떨어질 수 있지만 "죽지 않고 끝까지 대용량 쿼리를 완수하는 안정성"을 보장하는 이론적 장치


- **프로젝션 & 프레디케이트 푸시다운**
    - 외부 파일(CSV, Parquet)을 읽을 때 가해지는 데이터 다이어트 기술

    - **Projection Pushdown (열 선택 푸시다운):**
        - 쿼리에서 `SELECT age FROM 'data.parquet'`라고 선언하면,
        - DuckDB는 파일을 열기도 전에 'age' 열 외의 모든 데이터는 쳐다보지도 않고 스킵

    - **Predicate Pushdown (조건 푸시다운):**
        - `WHERE country = 'Korea'`라는 조건이 있다면,
        - 전체 데이터를 메모리로 다 들고 와서 거르는 것이 아니라
        - 파일 레벨(특히 Parquet의 메타데이터 블록 포맷 활용)에서 'Korea'가 아닌 데이터 블록은 아예 읽기 조차 하지 않고 걸러냄

    - 이 기술 덕분에 수십 GB의 원격/로컬 데이터 중 **진짜 계산에 필요한 수 MB의 데이터만 메모리로 쏙 들어오게** 됨


- **Zero-Copy 통합:**
    - 파이썬의 Pandas나 Arrow 프레임워크와 메모리 주소를 공유
    - 데이터 변환 과정에서 메모리를 새로 할당하고 복사하는 오버헤드가 전혀 없음



## 2. 대용량 분석 파이썬 예제

하고, 이를 DuckDB로 고속 집계 분석하는 전체 파이썬 코드를 작성해 보겠습니다.

1. **사전 준비**
    - 필요한 패키지 설치    
    - 가상 데이터 생성을 위해 `faker` 라이브러리를 추가로 사용

    ```bash
    pip install duckdb pandas faker pyarrow
    ```

2. **가상 데이터 생성 코드 (`generate_data.py`)**
    - 실습을 위해 1,000만 건(약 1~2GB 크기)의 가상 이커머스 주문 데이터를 생성함
    - 먼저 로컬 디스크에 대용량 대안으로 쓰일 대규모 CSV 파일을 생성

    ```python
    import csv
    import random
    from datetime import datetime, timedelta
    from faker import Faker
    import os

    def generate_large_csv(filename, num_rows):
        fake = Faker()
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Beauty', 'Sports']
        status_options = ['Completed', 'Cancelled', 'Refunded', 'Pending']
        
        start_date = datetime(2025, 1, 1)
        
        print(f"[{filename}] {num_rows:,}건의 가상 데이터 생성 시작...")
        
        # 디렉토리가 없다면 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 헤더 작성
            writer.writerow(['order_id', 'customer_id', 'order_date', 'category', 
                                'price', 'quantity', 'status'])
            
            for i in range(1, num_rows + 1):
                order_id = f"ORD_{i:08d}"
                customer_id = f"CUST_{random.randint(1, 500000):06d}" # 50만 명의 고객 고정
                # 2025년 1월 1일부터 약 1년 동안의 무작위 날짜
                order_date = (start_date + timedelta(days=random.randint(0, 365), 
                                seconds=random.randint(0, 86400))).strftime('%Y-%m-%d %H:%M:%S')
                category = random.choice(categories)
                price = round(random.uniform(10.0, 1500.0), 2)
                quantity = random.randint(1, 5)
                status = random.choices(status_options, weights=[70, 10, 5, 15], k=1)[0]
                
                writer.writerow([order_id, customer_id, order_date, category, price, quantity, status])
                
                if i % 2000000 == 0:
                    print(f" 진행률: {i:,} / {num_rows:,} 행 완료")
                    
        print(f" 데이터 생성 완료: {filename} (파일 크기를 확인해 보세요!)")

    # 1,000만 건의 대용량 로그 데이터 생성 권장
    if __name__ == "__main__":
        generate_large_csv('data/large_sales_data.csv', 10000000)
    ```


3. **DuckDB 고속 분석 코드 (`analyze_data.py`)**
    - 생성한 무거운 CSV 파일을 DuckDB를 이용해 읽어 들이고 분석하는 코드
    - 데이터를 전통적인 방식(Pandas 등)처럼 메모리에 통째로 로드하지 않고 **파일 상태 그대로 스캔**하여 연산

    ```python
    import duckdb
    import time

    # 1. DuckDB 연결 (Persistent 파일 모드 사용)
    con = duckdb.connect('ecommerce_analysis.duckdb')

    # 2. 로컬 자원 최적화 세팅
    con.execute("SET memory_limit = '4GB';") # 사용할 최대 RAM 제한
    con.execute("SET threads = 4;")           # 사용할 CPU 코어(스레드) 수 지정

    print("\n--- 분석 1: CSV 파일 직접 연결 및 가상 뷰(View) 생성 ---")
    # 데이터를 내부 테이블로 이전(Import)하지 않고 외부 파일에 직접 빨대를 꽂는 방식 (Zero-Ingestion)
    con.execute("""
        CREATE OR REPLACE VIEW v_sales AS 
        SELECT * FROM read_csv_auto('data/large_sales_data.csv')
    """)
    print("뷰 생성 완료. (실제 데이터를 읽지 않고 메타데이터만 연결된 상태)")


    print("\n--- 분석 2: 대규모 집계 연산 (카테고리별 매출 및 주문 건수) ---")
    start_time = time.time()

    # 1,000만 건 데이터 대상 대규모 집계 SQL 실행
    query1 = """
    SELECT 
        category,
        COUNT(order_id) AS total_orders,
        ROUND(SUM(price * quantity), 2) AS total_revenue,
        ROUND(AVG(price), 2) AS avg_unit_price
    FROM v_sales
    WHERE status = 'Completed'
    GROUP BY category
    ORDER BY total_revenue DESC;
    """

    result1 = con.execute(query1).df() # 결과를 바로 Pandas DataFrame으로 변환 (Zero-Copy)
    end_time = time.time()

    print(result1)
    print(f">> 1,000만 건 집계 소요 시간: {end_time - start_time:.4f} 초")


    print("\n--- 분석 3: 고성능 Parquet 포맷으로 변환 ---")
    # 대용량 처리를 위해 무거운 CSV 대신 컬럼형 파일 포맷인 Parquet로 압축 변환 보관
    start_time = time.time()
    con.execute("""
        COPY v_sales TO 'data/compressed_sales.parquet' (FORMAT PARQUET);
    """)
    print(f">> Parquet 압축 변환 소요 시간: {time.time() - start_time:.4f} 초")


    print("\n--- 분석 4: 변환된 Parquet 파일 대상 시계열 월별 집계 ---")
    start_time = time.time()

    query2 = """
    SELECT 
        STRFTIME(CAST(order_date AS TIMESTAMP), '%Y-%m') AS order_month,
        COUNT(DISTINCT customer_id) AS unique_customers,
        ROUND(SUM(price * quantity), 2) AS monthly_revenue
    FROM 'data/compressed_sales.parquet'
    GROUP BY 1
    ORDER BY order_month;
    """

    result2 = con.execute(query2).df()
    end_time = time.time()

    print(result2.head(5)) # 상위 5개 월만 출력
    print(f">> Parquet 기반 고속 월별 집계 소요 시간: {end_time - start_time:.4f} 초")

    # 연결 종료
    con.close()
    ```

<br>

> - **예제 코드의 기술적 관전 포인트**
>   - **초고속 로딩 성능 (`read_csv_auto`):**
>       - 일반적으로 Pandas에서 1,000만 건의 CSV를 `read_csv()`로 읽으면 몇 분이 걸리거나 메모리가 부족해짐
>       - DuckDB는 내부의 멀티스레드 CSV 리더가 벡터화 방식으로 파싱하여 수초 내에 쿼리를 마침
>   - **Parquet 포맷의 위력:**
>       - CSV 파일의 크기와 새로 생성된 `compressed_sales.parquet` 파일의 크기를 비교해 보면,
>       - 컬럼 기반 압축 덕분에 용량이 몇 분의 일 수준으로 줄어든 것을 확인할 수 있음
>       - 용량이 줄어든 만큼 분석 속도(`분석 4`)는 훨씬 더 빨라짐
>   - **데이터 파이프라인의 완성:**
>       - 외부 대용량 파일 가공 🡲 DuckDB 고속 집계 🡲 최종 요약본만 Pandas 데이터프레임(`df()`)으로 반환하는 구조
>       - 메모리를 매우 아끼면서 강력한 대용량 데이터 파이프라인을 로컬에 구현할 수 있게 됨
{: .summary-quote}