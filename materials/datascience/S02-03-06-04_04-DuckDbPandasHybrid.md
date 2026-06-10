---
layout: page
title:  "Pandas와 DuckDB를 이용한 하이브리드 가공"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-04_04-DuckDbPandasHybrid
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 기술적 배경 및 이론 (Technical Background)

> - **하이브리드(Hybrid) 데이터 가공**
>   - **Pandas의 유연한 표현력**과 **DuckDB의 압도적인 연산 속도 및 메모리 효율성**을 결합하여,
>   - 두 도구의 장점만을 극대화하는 영리한 데이터 처리 전략<br><br>
>   - 많은 데이터 분석가들이 대용량 데이터를 다룰 때 `Pandas`만 사용하다가 메모리 부족(OOM) 에러를 겪거나,
>   - 반대로 `SQL(DuckDB)`만 사용하다가 복잡한 데이터 변환이나 시각화, 머신러닝 연동에서 한계를 느낌

- **Heavy Lifting(무거운 연산)은 DuckDB에게**
    - 500만 건, 1,000만 건이 넘어가는 원시 데이터(Raw Data)에서 특정 조건을 필터링하고, 그룹화(`GROUP BY`)하여 집계하는 작업은 디스크 I/O와 CPU 연산 속도가 생명
    - 이 영역은 컬럼 기반 저장 구조와 벡터화 엔진을 가진 **DuckDB**가 Pandas보다 수십 배 이상 빠르며 메모리도 거의 쓰지 않음

- **Delicate Touch(정교한 가공 및 활용)는 Pandas에게**
    - 집계가 완료되어 크기가 수만 건 이하로 줄어든 데이터셋은 더 이상 메모리 부담이 없음
    - 이때부터는 파이썬 생태계의 풍부한 기능이 필요함
    - 비정형 텍스트의 정규식(Regex) 처리, 시계열 데이터의 복잡한 보간법(Interpolation), 시각화(Matplotlib/Seaborn) 및 머신러닝 모델 입력 데이터 변환 등은 **Pandas**가 훨씬 직관적이고 강력함

- **융합의 핵심: Apache Arrow 기반의 Zero-Copy**
    - 하이브리드 가공이 실용적인 기술이 될 수 있는 결정적인 이유
        - 두 도구가 **Apache Arrow**라는 공통의 인메모리 데이터 표준을 공유하기 때문
        - DuckDB에서 대용량 데이터를 처리한 후 `.df()`를 호출해 Pandas로 넘기거나, 반대로 Pandas 데이터를 DuckDB SQL 내부로 가져올 때
            - 데이터 복사(Copy)가 일어나지 않고
            - 메모리 주소만 전달(Zero-Copy) 됨
            - 데이터 이동에 따른 지연 시간과 메모리 낭비가 '0'에 수렴함


## 2. 하이브리드 가공 예제 코드

- 앞에서 생성한 **500만 건의 `user_behavior_logs.csv` 데이터셋을 그대로 재활용**함

> - **시나리오**
>   1. **[DuckDB 역할]**
>       - 500만 건의 거대한 로그 파일에서 `purchase`(구매) 및 `add_to_cart`(장바구니) 행동만 골라내어,
>       - 유저별/카테고리별 총 지속시간과 행동 횟수를 초고속으로 요약 집계
>       - (데이터 크기를 수백만 건에서 수만 건 단위로 축소)
>   2. **[Pandas 역할]**
>       - 축소된 데이터셋을 Zero-Copy로 넘겨받아,
>       - 파이썬의 피벗(`pivot`) 기능을 쓰고,
>       - 결측치(NaN)를 정교하게 채우며,
>       - 머신러닝이나 통계 분석에 적합한 형태의 '유저-카테고리별 행동 매트릭스'로 최종 가공함

```python
import duckdb
import pandas as pd
import time

print("="*70)
print("Pandas & DuckDB 하이브리드 데이터 가공 파이프라인")
print("="*70)

# -------------------------------------------------------------
# STEP 1: DuckDB를 이용한 무거운 데이터 축소 (Heavy Lifting)
# -------------------------------------------------------------
start_time = time.time()

# 인메모리 DuckDB 연결 및 CSV 파일 연결
con = duckdb.connect()
con.execute("""
    CREATE VIEW raw_logs AS 
    SELECT * FROM read_csv_auto('data/user_behavior_logs.csv')
""")

# 500만 건 중 분석에 필요한 핵심 액션만 필터링하고 유저/카테고리별로 1차 요약
# 이 무거운 연산을 DuckDB의 벡터화 엔진이 초고속으로 처리합니다.
duckdb_query = """
SELECT 
    user_id,
    category,
    COUNT(CASE WHEN action = 'add_to_cart' THEN 1 END) AS cart_count,
    COUNT(CASE WHEN action = 'purchase' THEN 1 END) AS purchase_count,
    SUM(duration_sec) AS total_duration
FROM raw_logs
WHERE action IN ('add_to_cart', 'purchase')
GROUP BY user_id, category;
"""

# .df() 호출 시 Zero-Copy 방식으로 Pandas DataFrame으로 즉시 전환됩니다.
summary_df = con.execute(duckdb_query).df()

duckdb_time = time.time() - start_time
print(f"▶ [DuckDB 완료] 500만 건 -> {len(summary_df):,}건으로 축소 완료 ({duckdb_time:.4f}초)")


# -------------------------------------------------------------
# STEP 2: Pandas를 이용한 정교한 데이터 변환 (Delicate Touch)
# -------------------------------------------------------------
start_time = time.time()

# 축소된 데이터는 메모리 부담이 없으므로 Pandas의 강력한 피벗/분석 기능을 활용합니다.
# 유저별로 각 카테고리에서 구매(purchase_count)를 얼마나 했는지 2차원 매트릭스 형태로 변환
pivot_df = summary_df.pivot(
    index='user_id', 
    columns='category', 
    values='purchase_count'
)

# SQL에서는 까다로운 결측치(NaN) 처리 및 데이터 타입 변환을 Pandas로 간결하게 수행
pivot_df = pivot_df.fillna(0).astype(int)

# 유저별 총 구매 카테고리 다양성(Diversity Score) 컬럼을 파이썬 내장 기능으로 쉽게 계산
pivot_df['category_diversity'] = (pivot_df > 0).sum(axis=1)

# 결과 상위 5개 유저 확인
pandas_time = time.time() - start_time
print(f"▶ [Pandas 완료] 피벗 및 결측치 가공, 파생변수 생성 완료 ({pandas_time:.4f}초)")

print("\n[최종 하이브리드 가공 완료 데이터셋 샘플 (User-Category Purchase Matrix)]")
print(pivot_df.head())

con.close()
```

- **예제 설명: 하이브리드 가공의 기술적 장점 및 특징**
    1. **메모리 절약과 OOM 방지:**
        - 500만 건의 CSV 데이터를 Pandas의 `read_csv()`로 전량 로드한 뒤 피벗을 시도했다면,
        - 수 기가바이트의 RAM을 소모하며 컴퓨터가 느려지거나 멈췄을 것
        - 예제에서는 DuckDB가 대용량 파일에서 필요한 데이터만 스트리밍하며 뽑아냈기 때문에
        - 파이썬 프로세스의 메모리 점유율이 극도로 낮게 유지됨

    2. **코드의 가독성과 생산성 향상:**
        - 2단계에서 진행한 `pivot` 연산과 결측치를 채우는 `.fillna(0)`, 행 단위 조건 계산(`sum(axis=1)`)을 순수 SQL로만 작성하려면
        - 수십 줄의 복잡한 `CASE WHEN`과 서브쿼리가 필요함
        - 반면 Pandas를 활용하면 단 3~4줄의 직관적인 코드로 표현이 가능함

    3. **병목 없는 데이터 스위칭:**
        - 전처리 속도를 보면
        - 500만 건의 대규모 필터링, 그룹 집계, 파이썬 데이터프레임 변환 및 최종 피벗까지 전체 파이프라인이 단 **0.5초 내외**에 종료됨
        - DuckDB와 Pandas가 Apache Arrow의 메모리 포맷을 공유하여 복사 오버헤드가 전혀 없었기 때문에 가능한 속도임