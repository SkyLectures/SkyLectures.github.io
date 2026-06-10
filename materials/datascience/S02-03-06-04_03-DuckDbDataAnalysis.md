---
layout: page
title:  "DuckDB를 이용한 데이터 분석"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-04_03-DuckDbDataAnalysis
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - 지금까지 살펴본 핵심 아키텍처(열 중심 저장, 벡터화 연산 등)가 DuckDB의 '하드웨어적 엔진 원리'라면,
> - 실제로 이를 사용해 데이터를 쿼리하고 인사이트를 도출하는 '데이터 분석 과정'에서는 조금 다른 차원의 배경 지식이 필요함
{: .common-quote}


## 1. DuckDB 기반 데이터 분석의 핵심 배경지식

- **Zero-Copy 기반의 데이터 생태계 통합 (Interoperability)**
    - 전통적인 데이터 분석 환경: 데이터를 툴 사이로 이동할 때마다 파일 형식 변환과 복사(Copy) 때문에 많은 시간과 메모리가 낭비됨
    - DuckDB 기반의 데이터 분석 환경: **Apache Arrow 표준 기법**을 아키텍처 저변에 깔고 있음<br><br>

    - **메모리 주소 공유:**
        - 파이썬에서 Pandas DataFrame이나 Polars, PyArrow 객체를 DuckDB로 불러올 때, 데이터를 새로 복사해서 메모리에 올리지 않음
        - DuckDB는 해당 라이브러리가 점유하고 있는 메모리 주소를 그대로 가리킨 채(Pointer) SQL 쿼리를 실행함

    - **분석가 관점의 의의:**
        - "수십 GB의 Pandas 데이터프레임을 복사하느라 컴퓨터가 멈추는 현상"이 발생하지 않음
        - DuckDB와 다른 파이썬 라이브러리 간의 결합이 지연 시간(Latency) 없이 매끄럽게 이루어짐

- **현대적 SQL(Modern SQL) 프로그래밍 패러다임**
    - DuckDB는 단순히 오래된 표준 SQL을 지원하는 레트로 엔진이 아님
    - Snowflake, BigQuery 등 최신 대규모 데이터 웨어하우스(DW)에서 채택된 '분석 생산성을 높이는 고등 SQL 구문'을 대거 탑재<br><br>

    - **복잡한 윈도우 함수 및 QUALIFY:**
        - 데이터 분석 시 누적 합계, 이동 평균, 순위(Rank) 등을 구할 때 윈도우 함수를 자주 사용
        - DuckDB는 이 결과를 필터링할 때 서브쿼리로 감쌀 필요 없이 `QUALIFY` 절을 통해 한 번에 걸러냄

    - **유연한 데이터 탐색 구문:**
        - 모든 컬럼을 다 적지 않고 특정 컬럼만 제외하는 `SELECT * EXCLUDE (password)`,
        - 컬럼의 형태를 가공하는 `SELECT * REPLACE (upper(name) AS name)`
        - 위와 같은 구문을 지원하여 대용량 테이블의 분석 코드가 획기적으로 짧아짐

- **구조적 파일 포맷(Parquet)과 메타데이터 활용 기법**
    - 대용량 데이터를 다룰 때 CSV와 Parquet 파일의 내부 구조 차이를 아는 것은 분석 성능을 결정짓는 핵심 배경지식<br><br>

    - **Parquet의 메타데이터 블록:**
        - Parquet 파일은 내부적으로 각 데이터 블록의 최솟값(Min), 최댓값(Max), 총 합계(Sum) 등의 통계 정보를 메타데이터에 미리 기록해 둠

    - **DuckDB의 지능적 스캔:**
        - DuckDB는 쿼리를 실행할 때 Parquet의 메타데이터를 가장 먼저 읽음
            - 예를 들어 `WHERE age > 50`이라는 조건이 있을 때,
                - 어떤 데이터 블록의 `Max(age)`가 42라면 DuckDB는 **그 블록 안에 있는 수백만 건의 데이터를 아예 읽지도 않고 통째로 건너뜀**
                - 이를 통해 분석 속도가 수백 배 빨라짐

- **로컬 자원 분배 및 실행 계획(Execution Plan) 최적화**
    - 분산 서버가 아닌 단일 PC(로컬)에서 대용량 데이터를 제어하기 때문에,
    - 내 컴퓨터의 자원 한계를 명확히 인지하고 조율하는 지식이 필요함<br><br>

    - **임시 디스크 스필러의 대가:**
        - 메모리가 부족하면 디스크를 쓰기 때문에 OOM 에러는 나지 않지만,
        - 디스크에 쓰고 읽는 과정에서 분석 속도가 급격히 느려짐
        - 따라서 분석가는 쿼리를 짤 때 메모리를 덜 쓰는 효율적인 쿼리를 고민해야 함
            - 예: 불필요한 `JOIN`이나 대규모 `DISTINCT` 지양

    - **EXPLAIN 명령을 통한 병목 진단:**
        - 내가 짠 SQL문 앞에 `EXPLAIN`을 붙이면 DuckDB가 데이터를 어떤 순서로 읽고 필터링하는지 실행 계획을 보여줌
        - 이를 통해 어떤 단계에서 연산이 정체되는지 파악하고, 인덱스를 생성하거나 쿼리를 수정하는 최적화 지식이 요구됨

<br>

> - **요약**
>   - DuckDB 분석 프로세스를 잘 다룬다는 것은 아키텍처적 원리를 넘어
>       - 파이썬 메모리를 공유하고,
>       - 최신 SQL 구문으로 코드를 효율화하며,
>       - Parquet 메타데이터 특성을 활용하고,
>       - 로컬 리소스 한계 안에서 쿼리를 최적화하는
>   - 일련의 소프트웨어적 감각"을 갖추는 것을 의미함


## 2. 샘플 데이터 생성 (Data Generation)

- 실습을 위해 **500만 건**의 가상 고객 행동 로그 데이터를 생성하는 Python 스크립트
- 웹서버의 로그나 대규모 서비스의 트래픽 데이터를 모방하여 작성됨

```python
# 필요 라이브러리 설치: pip install pyarrow faker pandas
import csv
import random
from datetime import datetime, timedelta
import os

def generate_behavior_logs(filename, num_rows):
    actions = ['page_view', 'search', 'add_to_cart', 'purchase', 'click_ad']
    devices = ['Mobile', 'Desktop', 'Tablet']
    categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports']
    
    start_date = datetime(2026, 1, 1)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"[{filename}] {num_rows:,}건의 샘플 로그 생성 시작...")
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['log_id', 'user_id', 'timestamp', 'action', 'category', 'duration_sec', 'device'])
        
        for i in range(1, num_rows + 1):
            log_id = f"LOG_{i:08d}"
            user_id = f"USR_{random.randint(1, 100000):06d}"  # 고유 사용자 10만 명
            # 2026년 1월부터 3개월간의 데이터 무작위 생성
            timestamp = (start_date + timedelta(days=random.randint(0, 90), seconds=random.randint(0, 86400))).strftime('%Y-%m-%d %H:%M:%S')
            action = random.choices(actions, weights=[50, 25, 15, 5, 5], k=1)[0]
            category = random.choice(categories)
            duration_sec = random.randint(5, 600) if action != 'purchase' else 0
            device = random.choice(devices)
            
            writer.writerow([log_id, user_id, timestamp, action, category, duration_sec, device])
            
            if i % 1000000 == 0:
                print(f" 진행률: {i:,} / {num_rows:,} 행 작성 완료")
                
    print(f" 데이터 생성 완료: {filename}")

if __name__ == "__main__":
    generate_behavior_logs('data/user_behavior_logs.csv', 5000000)
```


## 3. DuckDB 고속 분석 코드 (`analyze_data_advanced.py`)

```python
import duckdb
import pandas as pd
import time

# -------------------------------------------------------------
# [배경지식 1 증명 준비] 가상의 Pandas DataFrame 생성 (Zero-Copy용)
# -------------------------------------------------------------
# 유저들의 등급 정보가 담긴 소규모 매핑 데이터 (Pandas 객체)
user_meta_df = pd.DataFrame({
    'user_id': [f"USR_{i:06d}" for i in range(1, 100001)],
    'user_tier': [p for p in pd.Series(['VIP', 'GOLD', 'SILVER', 'BRONZE']).sample(100000, replace=True, random_state=42)]
})

# 1. DuckDB 인메모리 연결
con = duckdb.connect()

# 2. 로컬 대용량 CSV 파일 연결 (500만 건)
con.execute("""
    CREATE VIEW logs AS 
    SELECT * FROM read_csv_auto('data/user_behavior_logs.csv')
""")

print("="*70)
print("개정된 DuckDB 고급 분석 및 배경지식 검증")
print("="*70)


# -------------------------------------------------------------
# 검증 1: 현대적 SQL (Modern SQL) - EXCLUDE 및 QUALIFY 활용
# -------------------------------------------------------------
print("\n[실습 1] 현대적 SQL 구문 테스트 (EXCLUDE & QUALIFY)")
start = time.time()
query_1 = """
SELECT 
    * EXCLUDE (log_id, duration_sec), -- 불필요한 고유 ID와 체류시간 컬럼은 제외하고 가져옴
    RANK() OVER (PARTITION BY device ORDER BY timestamp DESC) AS recent_rank
FROM logs
WHERE action = 'purchase'
QUALIFY recent_rank <= 1 -- 서브쿼리 없이 윈도우 함수 결과를 바로 필터링
LIMIT 5;
"""
df_1 = con.execute(query_1).df()
print(df_1)
print(f">> 소요시간: {time.time() - start:.4f}초")


# -------------------------------------------------------------
# 검증 2: Zero-Copy 통합 - Pandas와 메모리 주소 공유 연산
# -------------------------------------------------------------
print("\n[실습 2] Zero-Copy 결합 테스트 (DuckDB SQL + 외부 Pandas DataFrame)")
start = time.time()

# DuckDB는 파이썬 메모리에 있는 'user_meta_df'라는 변수명을 인식하여 복사 없이 직접 JOIN합니다.
query_2 = """
SELECT 
    m.user_tier,
    COUNT(l.log_id) AS total_actions,
    COUNT(CASE WHEN l.action = 'purchase' THEN 1 END) AS purchase_count
FROM logs l
JOIN user_meta_df m ON l.user_id = m.user_id -- 파이썬 메모리 상의 Pandas 데이터프레임과 직접 조인
GROUP BY m.user_tier
ORDER BY purchase_count DESC;
"""
df_2 = con.execute(query_2).df()
print(df_2)
print(f">> 소요시간: {time.time() - start:.4f}초")


# -------------------------------------------------------------
# 검증 3: 실행 계획(Execution Plan) 분석 - EXPLAIN 활용
# -------------------------------------------------------------
print("\n[실습 3] EXPLAIN을 통한 쿼리 최적화 및 내부 연산 과정 진단")
query_3 = """
EXPLAIN 
SELECT category, COUNT(*) 
FROM logs 
WHERE action = 'add_to_cart' 
GROUP BY category;
"""
# 실행 계획을 문자열 형태로 받아와 출력합니다.
explanation = con.execute(query_3).fetchone()[1]
print(explanation)

con.close()
```

- **코드 해설 및 배경지식 매핑**

    - **현대적 SQL의 생산성 증명**
        - **`* EXCLUDE (log_id, duration_sec)`**:
            - 현업에서 대용량 테이블을 분석할 때,
                - 수십 개의 컬럼 중 몇 개만 제외하고 싶어도 일일이 컬럼명을 다 적어야 함
            - DuckDB
                - `EXCLUDE` 구문을 통해 분석에 방해되는 무거운 식별자나 불필요한 열을 선언적으로 제거해
                - 코드 가독성을 극대화

        - **`QUALIFY`**:
            - `WHERE`나 `HAVING` 절에서는 인식하지 못하는 윈도우 함수 결과(`recent_rank`)를
            - 서브쿼리 분리 없이 한 문장 안에서 필터링하여
            - 기기별 가장 최근 구매 내역 1건씩을 매우 간결하게 뽑아냄

    - **Zero-Copy 프레임워크 통합 증명**
        - **`JOIN user_meta_df`**:
            - 이 부분이 가장 놀라운 아키텍처적 포인트
            - DuckDB 내부에 생성한 테이블이 아니라,
            - **파이썬 메모리에 로드되어 있는 Pandas 데이터프레임 변수명을 SQL 안에서 테이블처럼 직접 Join**

        - 내부적으로는 메모리 복사 과정이 전혀 없는 `Zero-Copy(Apache Arrow 포맷 기반)` 형태로 주소만 참조하여 연산
            - 500만 건의 로그와 10만 건의 Pandas 데이터프레임 결합이 병목 없이 순식간에 완료됨

    - **실행 계획(Execution Plan) 진단 증명**
        - **`EXPLAIN`**:
            - DuckDB 분석 엔진이 쿼리를 수행하기 전 작성한 물리적/논리적 연산 지도를 출력

        - 출력 결과를 보면 다음을 눈으로 확인할 수 있음
            - `MULTIPLEX_CSV_SCAN`을 통해 멀티스레드로 CSV를 쪼개어 읽는 과정,
            - `FILTER` 단계에서 `action = 'add_to_cart'` 조건이 하부 레이어에서 어떻게 먼저 걸러지는지(`Pushdown`),
            - 마지막으로 `HASH_GROUP_BY`가 작동하는 파이프라인의 구조

        - 위의 내용을 통해 블랙박스 같던 DB 내부의 최적화 과정을 논리적으로 추적할 수 있게 됨

<br>

> - **[참고: 실행계획(Execution Plan)]**
>   - 데이터베이스에서 우리가 작성하는 SQL은 "무엇(What)을 가져올 것인가"만 정의할 뿐, "어떻게(How) 가져올 것인가"는 기술하지 않음
>   - 데이터베이스 내부의 두뇌 역할을 하는 옵티마이저(Optimizer)가 데이터를 가장 빠르게 찾아내기 위해 수립한 구체적인 처리 경로이자 최적의 계산 가이드라인을 실행계획(Execution Plan)이라고 함<br><br>
>   - SQL이 실행되면 데이터베이스는 내부적으로 다음과 같은 3단계를 거침
>       - **[SQL 구문 분석(Parsing)] 🡲 [최적화(Optimization): 여러 실행계획 비교] 🡲 [실행계획 확정 및 실행]**
>           - 구문 분석 (Parser): SQL 문법에 오류가 없는지, 접근하려는 테이블과 칼럼이 실제로 존재하는지 확인
>           - 통계 정보 확인: DB는 평소에 테이블의 데이터가 총 몇 건인지, 특정 칼럼에 데이터가 얼마나 균일하게 분포되어 있는지(선택도, Selectivity) 등의 통계 정보를 수집
>           - 비용 기반 최적화 (CBO): 생성 가능한 수많은 경로의 '예상 비용(디스크 I/O 횟수, CPU 연산량)'을 계산하여, 가장 비용이 적게 드는 최종 실행계획을 채택<br><br>
>   - 실행계획에서 눈여겨봐야 할 핵심 요소
>       - 스캔 방식 (Access Type): 데이터를 어떻게 찾았는가?
>           - Full Table Scan (ALL): 인덱스를 전혀 타지 못하고 테이블의 처음부터 끝까지 디스크를 다 긁은 상태 (튜닝 1순위 대상)
>           - Index Range Scan (range): 인덱스를 이용해 특정 범위만 똑똑하게 읽어낸 상태 (가장 이상적)
>           - Const / Eq_Ref:
>               - 기본키(PK)나 유니크 키를 사용해 단 1건만 정확하게 집어내어 읽은 상태
>               - 속도가 가장 빠름
>       - 조인 알고리즘 (Join Method): 테이블들을 어떻게 엮었는가?
>           - 두 개 이상의 테이블을 합칠 때 어떤 방식으로 컴퓨터가 연산했는지 보여줌
>           - Nested Loop Join:
>               - 한 테이블의 행을 하나씩 읽어가며 다른 테이블의 인덱스를 반복해서 찾는 방식
>               - OLTP 웹 서비스에서 가장 흔하고 효율적임
>           - Hash Join:
>               - 양쪽 테이블의 데이터를 메모리에 해시 테이블로 올려놓고 짝을 맞추는 방식
>               - 대용량 데이터를 한 번에 분석(OLAP)할 때 주로 발생함
>               - 속도가 매우 빠름
>       - Rows 및 Filtered: 얼마나 낭비가 심한가?
>           - Rows: 쿼리를 처리하기 위해 옵티마이저가 검사해야 할 것으로 예상한 행(Row)의 총수
>           - Filtered: 검사한 전체 행 중, WHERE 조건에 걸러져서 최종 결과로 살아남은 데이터의 비율(%)
>           - 주의 신호:
>               - Rows는 100만 건인데 Filtered가 $0.1%$라면, 100만 번을 다 뒤져서 겨우 1,000건 건졌다는 뜻
>               - 인덱스 재설계가 시급하다는 방증<br><br>
> - **엔지니어를 위한 실행계획 활용 가이드**
>   - 개발이나 리서치를 진행하실 때 다음과 같은 상황이라면 반드시 쿼리 앞에 EXPLAIN을 붙여 실행계획을 확인해야 함
>       - 로컬에선 빨랐는데 운영 서버에서 느릴 때:
>           - 데이터 양이 많아지면서 옵티마이저가 스캔 방식을 ALL로 변경해버렸을 가능성이 큼
>       - 인덱스를 분명히 걸었는데 안 탈 때:
>           - WHERE 조건절에 함수를 써서 가공했거나(WHERE LOWER(name) = 'abc'),
>           - 인덱스 칼럼의 타입을 다르게 입력하면(문자열 칼럼에 숫자 입력 등) 옵티마이저가 인덱스를 무시함
>           - 실행계획의 key 항목을 보면 인덱스가 누락되었는지 바로 파악할 수 있음
{: .common-quote}