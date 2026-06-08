---
layout: page
title:  "Pandas를 활용한 데이터 분석"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S02-02-03-02_01-PandasDataAnalysis
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 가상 데이터셋 생성

- **데이터셋**
    - 어느 가상의 테크 기업 마케팅 부서에서 활용하는 '광고 채널별 매출 및 고객 유입 데이터셋'
    - 분석할 데이터(`marketing_data`)는 다음과 같은 구조와 문제점을 가지고 있음
        * **Channel**: 광고 채널 (YouTube, Instagram 등 / 대소문자 혼용 및 오탈자 존재)
        * **Impressions**: 광고 노출 횟수 (결측치 존재)
        * **Clicks**: 클릭 횟수
        * **Revenue**: 해당 채널을 통해 발생한 매출액 (금액 단위에 `,` 기호 포함)

- **가상 데이터셋 생성 코드**
    - 제시된 마케팅 데이터셋(`marketing_data.csv`)의 구조와 결함 조건에 맞춰 
    - 300건의 가상 데이터를 생성하여 파일로 저장

        ```python
        import random
        import numpy as np
        import pandas as pd

        # 일관된 가상 데이터 생성을 위한 난수 시드 설정
        np.random.seed(42)
        random.seed(42)

        num_samples = 300

        # 1. Channel 생성 (대소문자 오염 및 의도적인 오탈자 'Youtub', 'insta' 포함)
        channels_pool = [
            "YouTube",
            "youtube",
            "Youtub",
            "Instagram",
            "instagram",
            "insta",
            "Facebook",
            "FACEBOOK",
        ]
        channels = [random.choice(channels_pool) for _ in range(num_samples)]

        # 2. Impressions 생성 (정상 범위 내 분포, 약 7% 확률로 결측치 처리)
        impressions = []
        for _ in range(num_samples):
            if random.random() < 0.07:  # 7% 결측치
                impressions.append(np.nan)
            else:
                impressions.append(
                    random.randint(10000, 150000)
                )  # 1만 ~ 15만 노출

        # 3. Clicks 생성 (노출 수의 1% ~ 5% 사이로 무작위 생성, 결측치인 경우 별도 범위 지정)
        clicks = []
        for imp in impressions:
            if pd.isna(imp):
                clicks.append(random.randint(500, 3000))
            else:
                # 노출 대비 전환율을 고려한 현실적인 클릭 수 계산
                click_rate = random.uniform(0.01, 0.05)
                clicks.append(int(imp * click_rate))

        # 4. Revenue 생성 (클릭 수와 연동하여 생성하되, 천단위 쉼표를 포함한 문자열 포맷팅)
        revenue = []
        for clk in clicks:
            # 클릭당 대략 800원 ~ 1500원의 매출이 발생한다고 가정
            rev_val = clk * random.randint(800, 1500)
            revenue.append(f"{rev_val:,}")

        # 데이터프레임 조립 및 CSV 저장
        marketing_df = pd.DataFrame(
            {
                "Channel": channels,
                "Impressions": impressions,
                "Clicks": clicks,
                "Revenue": revenue,
            }
        )

        file_name = "marketing_data.csv"
        marketing_df.to_csv(file_name, index=False, encoding="utf-8-sig")
        print(
            f"=== [성공] 300건의 마케팅 원본 데이터가 '{file_name}' 파일로 저장되었습니다. ===\n"
        )
        ```


## 2. 데이터 분석 실습

## 2.1 예제 코드

```python
import random
import numpy as np
import pandas as pd

# =====================================================================
# CSV 파일 로드 및 데이터 탐색 (Data Loading & EDA)
# =====================================================================

# 이제 하드디스크에 저장된 파일을 읽어와서 분석을 시작
df = pd.read_csv(file_name, encoding="utf-8-sig")

print("=== 1. 원본 파일 최초 로드 (상위 10개 행) ===")
print(df.head(10))
print("\n=== 2. 데이터 기본 타입 및 결측치 확인 ===")
print(df.info())
print("\n" + "=" * 60 + "\n")


# =====================================================================
# 데이터 정제 및 전처리 (Data Cleaning)
# =====================================================================

# 3-1. 텍스트 데이터 표준화 및 오탈자 교정
df["Channel"] = df["Channel"].str.strip().str.upper()

# 'YOUTUB' 이나 'INSTA' 같은 대표적인 오탈자를 원래 채널명으로 매핑/치환
df["Channel"] = df["Channel"].replace({"YOUTUB": "YOUTUBE", "INSTA": "INSTAGRAM"})

# 3-2. 수치형 데이터 형식 변환 (문자열 쉼표 제거 후 정수형 변환)
df["Revenue"] = df["Revenue"].str.replace(",", "").astype(int)

# 3-3. 수치형 결측치(NaN) 처리
# Impressions의 결측치를 '전체 노출 수의 평균값'으로 대체 (소수점은 반올림)
mean_impressions = round(df["Impressions"].mean())
df["Impressions"] = df["Impressions"].fillna(mean_impressions)

print("=== 3. 전처리 및 정제 완료 후 데이터 상태 ===")
print(df.head(10))
print("\n" + "=" * 60 + "\n")


# =====================================================================
# 데이터 변환 및 핵심 분석 지표 생성 (Feature Engineering)
# =====================================================================
# 마케팅 성과 평가를 위한 파생 변수 계산

# CTR (Click-Through Rate, 클릭률): 노출 대비 클릭 빈도 (%)
df["CTR(%)"] = round((df["Clicks"] / df["Impressions"]) * 100, 2)

# RPC (Revenue Per Click, 클릭당 매출): 클릭 한 번이 기여한 매출 효율 (원)
df["RPC"] = round(df["Revenue"] / df["Clicks"], 1)

print("=== 4. 마케팅 성과 지표(CTR, RPC) 파생 변수 추가 ===")
print(df.head(10))
print("\n" + "=" * 60 + "\n")


# =====================================================================
# 데이터 그룹화 및 최종 성과 집계 (Aggregation & Analysis)
# =====================================================================
# 난잡했던 채널들이 정제되었으므로, 이제 채널별 그룹화 분석이 가능합니다.
channel_analysis = (
    df.groupby("Channel")
    .agg(
        {
            "Impressions": "sum",  # 총 노출수
            "Clicks": "sum",  # 총 클릭수
            "Revenue": "sum",  # 총 매출액
            "CTR(%)": "mean",  # 평균 클릭률
            "RPC": "mean",  # 평균 클릭당 매출기여도
        }
    )
    .reset_index()
)

# 최종 성과 지표인 '총 매출액(Revenue)'을 기준으로 내림차순 정렬
channel_analysis = channel_analysis.sort_values(by="Revenue", ascending=False)

# 가독성을 위해 결과창의 수치 포맷팅을 변경하여 출력
print("=== 5. [최종 분석 리포트] 채널별 마케팅 성과 요약 ===")
pd.options.display.float_format = "{:,.2f}".format
print(channel_analysis.to_string(index=False))
```

### 2.2 코드 수정 사항 및 분석 포인트 설명

1. **오탈자 교정 로직 추가 (`.replace`)**
    - **상황:**
        - 데이터 규모가 300건으로 늘어나면서 `Youtub`, `insta` 같이 현업에서 자주 발생하는 휴먼 에러(오탈자) 데이터를 삽입
    - **해결:**
        - `str.upper()`로 대문자 통일을 먼저 수행한 뒤,
        - Pandas의 딕셔너리 매핑 기법인 `.replace({"YOUTUB": "YOUTUBE", "INSTA": "INSTAGRAM"})`을 사용하여
        - 지저분하게 흩어져 있던 광고 채널을 `FACEBOOK`, `INSTAGRAM`, `YOUTUBE` 3개의 정형화된 범주로 명확하게 통합

2. **수치 변환 안정성 확보 (`astype(int)`)**
    - **상황:**
        - 원본 파일의 매출 데이터가 쉼표(`,`)를 달고 저장되었기 때문에
        - 데이터 로드 시 기본 문자열로 분류됨
    - **해결:**
        - 문자열 치환 후,
        - `.astype(int)` 연산자를 직접 적용하여
        - 계산 역량이 있는 완전한 정수 데이터형으로 일관되게 정제

3. **유동적 결측치 보정 (`fillna`)**
    - **상황:**
        - 300건 중 약 7%의 비율로 발생한 `Impressions`(노출수)의 빈 칸을 메워야 함
    - **해결:**
        - 파일로부터 로드된 정상 데이터들의 산술 평균값(`mean()`)을 실시간으로 추적·계산하여
        - 유실 항목을 안전하게 은닉 및 대체 처리

4. **다중 성과 지표 집계 (`groupby` + `agg`)**
    - **상황:**
        - 정제된 데이터를 기반으로 비즈니스 인사이트를 도출해야 함
    - **해결:**
        - `.groupby('Channel').agg({...})` 구문을 통해
        - 각 광고 채널별로 단순 규모 합산(`sum`)이 필요한 지표와
        - 효율성 평균(`mean`)을 내야 하는 지표(CTR, RPC)를 분리하여
        - 한 번에 종합 요약 리포트를 산출















Pandas를 활용한 전체적인 데이터 분석 파이프라인을 이해할 수 있도록, [데이터 로드 ➔ 탐색 ➔ 정제 ➔ 집계/분석 ➔ 시각화 데이터 준비]의 전체 흐름을 담은 예제 코드와 상세 설명을 준비했습니다.

이번에 사용할 시나리오는 어느 가상의 테크 기업 마케팅 부서에서 활용하는 '광고 채널별 매출 및 고객 유입 데이터셋'입니다.

---

## 1. 가상 데이터셋 및 분석 시나리오

분석할 데이터(`marketing_data`)는 다음과 같은 구조와 문제점을 가지고 있습니다.

* **Channel**: 광고 채널 (YouTube, Instagram 등 / 대소문자 혼용 및 오탈자 존재)
* **Impressions**: 광고 노출 횟수 (결측치 존재)
* **Clicks**: 클릭 횟수
* **Revenue**: 해당 채널을 통해 발생한 매출액 (금액 단위에 `,` 기호 포함)

---

## 2. 데이터 분석 통합 예제 코드

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------
# [1단계] 데이터 로드 및 가상 데이터셋 생성
# --------------------------------------------------
raw_data = {
    "Channel": [
        "Youtube",
        "instagram",
        "YouTube",
        "Facebook",
        "Instagram",
        "youtube",
    ],
    "Impressions": [50000, 85000, np.nan, 120000, 95000, 60000],  # 결측치 포함
    "Clicks": [1200, 3400, 1500, 4100, 3900, 1100],
    "Revenue": [
        "1,200,000",
        "3,100,000",
        "1,600,000",
        "4,800,000",
        "3,500,000",
        "950,000",
    ],  # 쉼표 포함 문자열
}

df = pd.DataFrame(raw_data)

# --------------------------------------------------
# [2단계] 데이터 탐색 (Exploratory Data Analysis, EDA)
# --------------------------------------------------
print("=== 1. 데이터 기본 정보 확인 ===")
print(df.info())
print("\n=== 2. 수치형 데이터 기술 통계 ===")
print(df.describe())
print("\n" + "=" * 50 + "\n")

# --------------------------------------------------
# [3단계] 데이터 전처리 및 정제 (Data Cleaning)
# --------------------------------------------------
# 3-1. 텍스트 데이터 표준화 (대소문자 통일)
df["Channel"] = df["Channel"].str.upper()

# 3-2. 수치형 데이터 형식 변환 (문자열 쉼표 제거 후 정수형 변환)
df["Revenue"] = df["Revenue"].str.replace(",", "").astype(int)

# 3-3. 결측치(NaN) 처리 (Impressions의 결측치를 채널별 평균값으로 대체)
# 여기서는 간단히 전체 평균값으로 대체합니다.
mean_impressions = df["Impressions"].mean()
df["Impressions"] = df["Impressions"].fillna(mean_impressions)

print("=== 3. 정제 완료된 데이터프레임 ===")
print(df)
print("\n" + "=" * 50 + "\n")

# --------------------------------------------------
# [4단계] 데이터 변환 및 분석 지표 생성 (Feature Engineering)
# --------------------------------------------------
# 광고 효율을 평가하기 위한 핵심 지표 계산
# CTR (Click-Through Rate, 클릭률) = (Clicks / Impressions) * 100
df["CTR(%)"] = round((df["Clicks"] / df["Impressions"]) * 100, 2)

# CPC (Cost Per Click) 개념을 변형한 클릭당 매출 기여도 계산
# Revenue_per_Click = Revenue / Clicks
df["Rev_per_Click"] = round(df["Revenue"] / df["Clicks"], 1)

print("=== 4. 파생 변수(지표)가 추가된 데이터 ===")
print(df)
print("\n" + "=" * 50 + "\n")

# --------------------------------------------------
# [5단계] 데이터 그룹화 및 집계 (Aggregation)
# --------------------------------------------------
# 채널별로 데이터를 묶어서 매출 합계, 평균 CTR, 총 클릭수 계산
channel_analysis = (
    df.groupby("Channel")
    .agg({"Revenue": "sum", "Clicks": "sum", "CTR(%)": "mean"})
    .reset_index()
)

# 매출액 기준 내림차순 정렬
channel_analysis = channel_analysis.sort_values(by="Revenue", ascending=False)

print("=== 5. 최종 채널별 성과 분석 결과 ===")
print(channel_analysis)

```

---

## 3. 데이터 분석 파이프라인 단계별 상세 설명

### ① 데이터 탐색 (`info()`, `describe()`)

* **설명:** 데이터를 받자마자 무작정 연산을 시작하면 안 됩니다. `df.info()`를 통해 각 컬럼의 데이터 타입과 결측치 존재 여부를 파악하고, `df.describe()`를 통해 평균, 최솟값, 최댓값, 사분위수 등 데이터의 전반적인 분포와 이상치 징후를 확인합니다.

### ② 데이터 정제 (Data Cleaning)

* **분석의 걸림돌 제거:** `Youtube`와 `youtube`는 데이터 분석 시 서로 다른 그룹으로 분리되는 문제가 있습니다. 이를 `.str.upper()`로 모두 `YOUTUBE`로 통일합니다.
* **타입 매칭:** 연산이 불가능한 쉼표 텍스트 형태의 매출액 데이터를 정수형(`int`)으로 변환하고, 데이터가 비어 있는 `Impressions` 항목은 기존 데이터의 평균값으로 메워 데이터 손실을 방지합니다.

### ③ 파생 변수 생성 (Feature Engineering)

* **설명:** 원본 데이터에 있는 숫자 자체보다, 데이터 간의 관계를 통해 새로운 의미를 도출하는 단계입니다. 노출 대비 클릭 수의 비율인 CTR(클릭률)을 계산하여 어떤 채널이 고객의 이목을 가장 잘 끌었는지 파악할 수 있는 정량적 기준을 마련합니다.

### ④ 그룹화 및 정렬 (`groupby()`, `sort_values()`)

* **설명:** 데이터 분석의 핵심은 **비교와 요약**입니다. `groupby('Channel')`를 사용하면 각 채널(FACEBOOK, INSTAGRAM, YOUTUBE) 별로 데이터를 쪼개어 합계(`sum`)나 평균(`mean`)을 연산할 수 있습니다. 마지막으로 `sort_values()`를 사용하여 가장 매출 기여도가 높은 채널이 무엇인지 한눈에 보이도록 데이터를 정렬합니다.