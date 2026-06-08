---
layout: page
title:  "Pandas를 활용한 데이터 전처리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S02-02-02-02_01-PandasDataPreprocess
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 가상 데이터셋 생성

- **데이터셋**
    - e-커머스 플랫폼의 고객 구매 데이터(`customer_data.csv`)를 모방한 데이터
    - 고질적인 데이터 문제(결측치, 이상치, 데이터 형식 오류 등)가 포함되어 있음

    - 수집된 원본 데이터 상태
        - **Id**: 고객 식별자
        - **Join_Date**: 가입일 (날짜 형식이 통일되지 않음)
        - **Age**: 나이 (결측치 및 말도 안 되는 이상치 포함)
        - **Usage_Fee**: 이용 금액 (문자열 `$` 기호 및 쉼표 포함)
        - **Status**: 회원 상태 (대소문자 혼용 및 공백 문제)

- **가상 데이터셋 생성 코드**

```python
import random
import numpy as np
import pandas as pd

# 일관된 난수 생성을 위한 시드 설정
np.random.seed(42)
random.seed(42)

# 샘플 데이터 수
num_samples = 200

# 1. Id 생성 (1001부터 1200까지)
ids = list(range(1001, 1001 + num_samples))

# 2. Join_Date 생성 (다양한 날짜 형식 및 결측치 혼합)
date_pool = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
join_dates = []

for _ in range(num_samples):
    if random.random() < 0.05:  # 5% 확률로 결측치
        join_dates.append(None)
    else:
        dt = random.choice(date_pool)
        # 다양한 날짜 텍스트 포맷 무작위 적용
        fmt = random.choice(["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"])
        join_dates.append(dt.strftime(fmt))

# 3. Age 생성 (정상 범위, 음수 이상치, 100세 이상 노이즈, 결측치 혼합)
ages = []
for _ in range(num_samples):
    rand_val = random.random()
    if rand_val < 0.08:  # 8% 확률로 결측치
        ages.append(np.nan)
    elif rand_val < 0.12:  # 4% 확률로 음수 이상치
        ages.append(random.randint(-15, -1))
    elif rand_val < 0.16:  # 4% 확률로 비현실적인 고령 이상치
        ages.append(random.randint(120, 200))
    else:  # 정상 범위 (15세 ~ 80세)
        ages.append(random.randint(15, 80))

# 4. Usage_Fee 생성 (문자열 '$' 및 천단위 쉼표 포함, 대형 구매 이상치 포함)
usage_fees = []
for _ in range(num_samples):
    # 일반적인 유저 금액대 vs 고액 자산가 노이즈 분리
    if random.random() < 0.03:
        fee_val = random.randint(1000000, 5000000)  # 이상치급 고액
    else:
        fee_val = random.randint(0, 500000)  # 일반 금액대

    # 천단위 쉼표 및 $ 기호가 붙은 문자열 포맷팅
    formatted_fee = f"${fee_val:,}"
    usage_fees.append(formatted_fee)

# 5. Status 생성 (대소문자 오염, 앞뒤 공백 무작위 추가)
status_options = ["Active", "Silver", "Gold"]
statuses = []
for _ in range(num_samples):
    base_status = random.choice(status_options)

    # 무작위 대소문자 변형 및 공백 추가
    case_style = random.choice(["lower", "upper", "mixed"])
    if case_style == "lower":
        base_status = base_status.lower()
    elif case_style == "upper":
        base_status = base_status.upper()

    # 앞뒤 공백 삽입 패턴
    space_style = random.choice(["both", "left", "right", "none"])
    if space_style == "both":
        base_status = f"  {base_status}  "
    elif space_style == "left":
        base_status = f"   {base_status}"
    elif space_style == "right":
        base_status = f"{base_status} "

    statuses.append(base_status)

# --------------------------------------------------
# 데이터프레임 조립 및 확인
# --------------------------------------------------
customer_df = pd.DataFrame(
    {
        "Id": ids,
        "Join_Date": join_dates,
        "Age": ages,
        "Usage_Fee": usage_fees,
        "Status": statuses,
    }
)

# 생성된 가상 데이터를 CSV 파일로 저장
customer_df.to_csv("customer_data.csv", index=False, encoding="utf-8-sig")

print("=== 가상 데이터셋 200건 생성 완료 ('customer_data.csv' 저장됨) ===")
print(customer_df.head(15))  # 상위 15개 샘플 출력하여 노이즈 확인
print("\n=== 데이터셋 결측치 개수 및 데이터 타입 요약 ===")
print(customer_df.info())
```


## 2. 데이터 전처리 예제 코드

```python
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. 가상 데이터 파일(CSV) 읽어오기
# --------------------------------------------------
# encoding='utf-8-sig'는 한글 및 특수문자 깨짐과 BOM 에러를 방지합니다.
file_path = "customer_data.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

print("=== [1단계: 원본 데이터 로드 및 초기 검사] ===")
print(df.head(10))  # 처음 10개 행 시각적 확인
print("\n--- 데이터 인덱스 및 컬럼별 타입 정보 ---")
print(df.info())
print("\n--- 전처리 전 결측치 개수 현황 ---")
print(df.isnull().sum())
print("\n" + "=" * 60 + "\n")


# --------------------------------------------------
# 2. 데이터 전처리 파이프라인 수행
# --------------------------------------------------
print("=== [2단계: 데이터 전처리 파이프라인 가동] ===")

# Step 2-1. 날짜 데이터 형식 통일 (Datetime 변환)
# 혼재된 문자열 포맷(%Y-%m-%d, %Y/%m/%d 등)을 정형 Datetime 객체로 일괄 변환
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce")

# Step 2-2. 날짜형 결측치 처리 (가입일 빈 곳은 최빈값으로 대체)
most_frequent_date = df["Join_Date"].mode()[0]
df["Join_Date"] = df["Join_Date"].fillna(most_frequent_date)
print("-> 가입일(Join_Date) 포맷 변환 및 결측치 대체 완료")


# Step 2-3. 텍스트 데이터 정제 (앞뒤 공백 제거 및 대문자 변환)
# 문자열 내의 불규칙한 공백을 지우고 그룹을 하나로 통일
df["Status"] = df["Status"].str.strip().str.upper()
print("-> 회원 상태(Status) 공백 제거 및 대문자 통일 완료")


# Step 2-4. 수치형 데이터 정제 (특수문자 제거 후 데이터 타입 변환)
# '$' 기호와 천단위 쉼표(',')를 제거한 뒤, 실제 연산이 가능한 정수형(int)으로 변경
df["Usage_Fee"] = (
    df["Usage_Fee"].str.replace("$", "", regex=False).str.replace(",", "")
)
df["Usage_Fee"] = pd.to_numeric(df["Usage_Fee"])
print("-> 이용 금액(Usage_Fee) 특수문자 제거 및 수치형 변환 완료")


# Step 2-5. 이상치(Outlier) 식별 및 처리
# 정상 나이 범위를 0세 초과 100세 미만으로 정의하고, 범위를 벗어나는 이상치는 NaN 처리
df.loc[(df["Age"] <= 0) | (df["Age"] >= 100), "Age"] = np.nan

# Step 2-6. 수치형 결측치 처리 (나이 빈 곳을 정상 데이터의 평균값으로 대체)
mean_age = round(df["Age"].mean(), 1)
df["Age"] = df["Age"].fillna(mean_age)
print(f"-> 나이(Age) 이상치 정제 및 평균값({mean_age}세)으로 결측치 대체 완료")


# Step 2-7. 파생 변수 생성: 연속형 변수의 범주화 (Binning)
# 연속적인 나이 데이터를 분석 목적에 맞게 연령대 그룹으로 구간화
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ["Teens", "20s", "30s", "40s", "50s", "60s+"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)
print("-> 연령대 범주 파생 변수(Age_Group) 생성 완료")

print("\n" + "=" * 60 + "\n")


# --------------------------------------------------
# 3. 최종 정제 및 변환 결과 확인
# --------------------------------------------------
print("=== [3단계: 최종 정제 완료 데이터셋 검증] ===")
print(df.head(15))  # 상위 15개 샘플 출력하여 정제 상태 검증
print("\n--- 전처리 후 최종 데이터 정보 ---")
print(df.info())
print("\n--- 회원 상태(Status) 고유 빈도 분석 ---")
print(df["Status"].value_counts())
print("\n--- 연령대별 그룹 분포 확인 ---")
print(df["Age_Group"].value_counts().sort_index())

# 필요한 경우 정제 완료된 데이터 파일을 별도로 저장할 수 있습니다.
# df.to_csv("cleaned_customer_data.csv", index=False, encoding='utf-8-sig')
```


## 3. 데이터 전처리 단계별 설명

1. **가입일 형식 통일 및 에러 처리 (`pd.to_datetime`)**
    - **수행 코드:**

        ```python
        df['Join_Date'] = pd.to_datetime(df['Join_Date'], errors='coerce')
        ```

    - **설명:**
        - 외부 파일(`customer_data.csv`)을 읽어오면 날짜가 문자열(`object`)로 인식되며,
        - `2024-05-12`, `2025/11/02`, `2024.01.30`처럼 수집 포맷이 제각각인 경우가 많음
        - `pd.to_datetime()`은 이러한 다양한 문자열 패턴을 날짜형 데이터(`datetime64`)로 통일해 줌

    - **핵심 옵션:**
        - `errors='coerce'`를 설정하면 문자열이 너무 심하게 깨져 날짜로 파싱할 수 없는 유효하지 않은 데이터를
        - 에러 발생 없이 `NaT`(날짜형 결측치)로 안전하게 변환함

2. **날짜형 결측치 채우기 (`fillna` + `mode`)**
    - **수행 코드:**

        ```python
        most_frequent_date = df['Join_Date'].mode()[0]
        df['Join_Date'] = df['Join_Date'].fillna(most_frequent_date)
        ```

    - **설명:**
        - 1단계에서 발생한 `NaT`나 원본의 빈 칸을 처리하는 단계
        - 시계열이나 가입일 데이터는 평균값을 낼 수 없으므로,
        - 데이터셋에서 가장 빈번하게 등장한 날짜인 최빈값(`.mode()[0]`)을 동적으로 계산하여 빈 곳을 메움
        - 행을 삭제하지 않고 데이터 유실을 최소화하는 기법

3. **문자열 오염 정제 (`str.strip` + `str.upper`)**
    - **수행 코드:**

        ```python
        df['Status'] = df['Status'].str.strip().str.upper()
        ```

    - **설명:**
        - 공백과 대소문자가 뒤섞인 텍스트 데이터(`'  Active '`, `'silver '`, `'GOLD'`)를 정형화
        - Pandas의 `.str` 액세서들을 체이닝(Chaining)하여 적용

        - `strip()`: 문자열 앞뒤에 숨은 불필요한 공백 제거
        - `upper()`: 소문자나 대소문자가 섞인 문자를 모두 대문자로 통일

    - **효과:**
        - 정제 전에는 모두 다른 그룹으로 인식되던 데이터가
        - `ACTIVE`, `SILVER`, `GOLD`라는 명확한 3개의 범주(Category)로 묶이게 됨

4. **특수문자 제거 및 수치형 변환 (`str.replace` + `to_numeric`)**
    - **수행 코드:**

        ```python
        df['Usage_Fee'] = df['Usage_Fee'].str.replace('$', '', regex=False).str.replace(',', '')
        df['Usage_Fee'] = pd.to_numeric(df['Usage_Fee'])
        ```

    - **설명:**
        - 돈이나 수량 데이터에 통화 기호(`$`)나 천단위 구분 쉼표(`,`)가 포함되어 있으면 Pandas는 이를 수치가 아닌 '문자열'로 인식
        - 합계나 평균을 구할 수 없음

    * **효과:**
        - `replace()`를 통해 방해 요소를 공백으로 지워 순수한 숫자 형태의 문자열로 만든 뒤,
        - `pd.to_numeric()`을 통해 연산이 가능한 **정수/실수형 데이터 타입**으로 강제 변환

5. **논리 조건을 활용한 이상치 식별 (`df.loc`)**
    - **수행 코드:**

        ```python
        df.loc[(df['Age'] <= 0) | (df['Age'] >= 100), 'Age'] = np.nan
        ```

    - **설명:**
        - 나이 데이터에 포함된 음수(`-15`)나 비현실적인 고령(`150`) 같은 데이터는
        - 수집 오류로 발생한 이상치(Outlier)

    - **효과:**
        - 비즈니스 도메인 규칙(예: 서비스 이용 나이는 1세 이상 99세 이하)에 따라
        - 대괄호 `[]` 안에 논리 연산자(`|`, OR)로 조건식을 넣고,
        - 이에 해당하는 데이터만 콕 집어 결측치(`np.nan`)로 바꿈
        - 이상치가 통계치(평균 등)를 왜곡하는 것을 막기 위한 선행 작업

6. **수치형 결측치 처리 (`fillna` + `mean`)**
    - **수행 코드:**

        ```python
        mean_age = round(df['Age'].mean(), 1)
        df['Age'] = df['Age'].fillna(mean_age)
        ```

    * **설명:**
        - 5단계에서 이상치를 `NaN`으로 비워두었거나 원래 비어 있던 나이 값을 처리
        - 여기서는 이상치가 제거된 정상적인 나이 데이터들의 평균값(`mean()`)을
        - 소수점 첫째 자리까지 구한 뒤 `fillna()`로 일괄 대체함
        - 데이터의 전체적인 통계적 특성을 유지하는 대표적인 결측치 처리 방식

7. **데이터 구간화 및 파생 변수 생성 (`pd.cut`)**
    - **수행 코드:**

        ```python
        bins = [0, 20, 30, 40, 50, 60, 100]
        labels = ['Teens', '20s', '30s', '40s', '50s', '60s+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
        ```

    * **설명:**
        - 연속형 수치(23세, 37세, 45세 등)를 그대로 분석하기보다
        - '20대', '30대'처럼 그룹으로 묶어 분석하는 것이 효과적일 때가 있음 🡲 구간화(Binning)
        
    * **효과:**
        - `bins`에 경계값 범위를 지정하고
        - `labels`에 매핑할 그룹명을 정의하여
        - `pd.cut()`에 전달하면,
        - 나이에 맞는 연령대 범주형 데이터인 `Age_Group`이라는 새로운 파생 변수(Feature)가 만들어짐


> - **요약:**
>   - 이 7단계의 전처리 파이프라인을 거치면,
>   - 아무리 엉망으로 수집된 파일 데이터라도 **"데이터 타입 오류 없음, 결측치 없음, 이상치 정제됨, 분석하기 좋은 범주 생성됨"** 상태의 이상적인 데이터셋으로 거듭나게 됨

> - **핵심 전처리 단계별 설명**

1. 날짜 형식 통일 및 정제 (`pd.to_datetime`)
    - **문제점:**
        - `2025-01-10`, `2025/02/15`, `2025.03.20` 등 날짜 기입 방식이 제각각임
    - **해결책:**
        - Pandas의 `to_datetime()`은 유연한 파싱 능력을 가지고 있어 다양한 포맷을 정형 포맷(`YYYY-MM-DD`)으로 자동 변환함
        - `errors='coerce'` 옵션은 형식을 도저히 맞출 수 없는 깨진 데이터를 결측치(`NaT`)로 안전하게 변환해 줌

2. 텍스트 정제 (`str` 액세서 활용)
    - **문제점:**
        - `" Active "`, `"active"`, `"Active"`는 사람이 보기엔 같은 상태이지만 컴퓨터는 완전히 다른 문자열로 인식함
    - **해결책:**
        - `.str.strip()`으로 불필요한 앞뒤 공백을 자르고,
        - `.str.upper()`를 통해 전체 대문자로 통일하여
        - 범주형 데이터의 고유값(Unique Values)을 단순화함

3. 문자열 포함 수치 데이터 정형화
    - **문제점:**
        - 화폐 단위 기호(`$`)나 천단위 구분 쉼표(`,`)가 포함되면
        - 데이터 타입이 `object`(문자열)로 지정되어 연산이 불가능
    - **해결책:**
        - `.str.replace()`로 기호들을 공백으로 치환한 후,
        - `pd.to_numeric()`을 사용해 정수/실수형 데이터 타입으로 강제 변환

4. 이상치 처리 (`df.loc` 조건문)
    - **문제점:**
        - 나이 데이터에 음수(`-5`)나 비현실적인 값(`150`)이 섞여 있으면
        - 통계 분석 결과가 왜곡됨
    - **해결책:**
        - 논리 연산자(`|`, `&`)를 결합한 불리언 인덱싱을 통해
        - `df.loc[조건, 컬럼]` 형태로 접근하여
        - 이상치를 우선 결측치(`np.nan`)로 바꾼 뒤
        - 분석 도메인에 맞는 처리를 이어감

5. 결측치 대체 (`fillna`)
    - **문제점:**
        - 데이터 분석 및 머신러닝 모델 입력 시 결측치(`NaN`)가 있으면
        - 에러가 발생할 수 있음
    - **해결책:**
        - 연속형 변수(나이 등)는 분석 목적에 따라 **평균(Mean)** 또는 중앙값(Median)으로 대체하고,
        - 범주형이나 날짜 데이터는 **최빈값(Mode)** 등으로 대체하여
        - 데이터 유실(행 삭제)을 최소화함

6. 데이터 변환 및 구간화 (`pd.cut`)
    - **설명:**
        - 나이(`Age`) 같은 연속형 수치 데이터를 그대로 쓰기보다
        - `20대`, `30대`처럼 그룹화(Binning)하는 것이 분석 직관성을 높일 때가 있음
        - `pd.cut()` 함수에 경계값 리스트(`bins`)와 매핑할 레이블(`labels`)을 넘겨주면 손쉽게 파생 변수를 생성할 수 있음