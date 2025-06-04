---
layout: page
title:  "LangChain + GPT"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-06_01-LangChainGptDataAnalysis
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. LangChain + GPT로 데이터 처리하기

### 1.1 Pandas DataFrame 자연어 분석하기

- 필요한 라이브러리

    ```bash
    pip install pandas langchain langchain_experimental langchain-openai tabulate
    ```

- 예제 코드

    ```python
    import pandas as pd
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI

    # 1. 데이터프레임 준비
    df = pd.DataFrame({
        "년도": [2022, 2022, 2023, 2023],
        "제품": ["노트북", "모니터", "노트북", "모니터"],
        "판매량": [100, 150, 300, 250]
    })

    # 2. LLM 설정
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 3. dangerous_code 허용 옵션 추가
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        **{"allow_dangerous_code": True}
    )

    # 4. 자연어 질문
    query = "2023년에 가장 많이 팔린 제품은 무엇인가요?"
    response = agent.invoke(query)

    print("\n분석 결과:")
    print(response)
    ```

- ⚠️ **보안 관련 주의 사항**
    - 이 기능은 GPT가 실제로 파이썬 코드를 실행함
    - 서버 환경에서 실행할 때는 사용자 입력 제한, 샌드박스 환경 격리, 파일 접근 제한 등을 반드시 고려할 것
    - 로컬 개발/연구용으로는 괜찮지만, 웹 서비스로 외부에 공개 시 매우 위험할 수 있음
    - 🔐 공식 보안 가이드: https://python.langchain.com/docs/security/



### 1.2 실습용 데이터셋 다운로드

```python
import wget

file_url = "https://raw.githubusercontent.com/HelloDataScience/Datasets/refs/heads/main/APT_Price_GangNamGu_2023_20230731.txt"
out_filename = "./data/Apart_Price_2023.tsv"
wget.download(file_url, out_filename)
```


### 1.3 데이터셋 로드

- 해당 데이터셋은 UTF-16 코드셋으로 저장되어 있음

```python
import pandas as pd

df = pd.read_csv(out_filename, sep="\t", encoding="utf-16")
df
```


#### 1.4 예제

- 예제 1

    ```python
    import pandas as pd
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI

    # 1. 데이터프레임 준비
    df = pd.read_csv("Apart_Price_2023.tsv", sep="\t", encoding="utf-16")

    # 2. LLM 설정
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # 3. dangerous_code 허용 옵션 추가
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        **{"allow_dangerous_code": True}
    )

    # 4. 자연어 질문
    query = "전체 데이터의 갯수와 자료형 등을 확인해줄래?"
    response = agent.invoke(query)

    print("\n분석 결과:")
    print(response)
    ```

- 예제 2

    ```python
    response = agent.invoke("데이터 분석 과정에서 활용하기 어려운 컬럼들인 법정동, 지번 컬럼들을 삭제해주는 코드를 짜줄래?")
    print(response["output"])
    df
    ```
    
    ```python
    df.drop(['법정동', '지번'], axis=1, inplace=True)
    df
    ```

    ```python
    agent = reset_agent(df)
    ```

- 예제 3

    ```python
    response = agent.invoke("결측치가 있는 컬럼이 있는지 확인해서 말해줘")
    print(response["output"])
    ```

    ```python
    agent = reset_agent(df)
    ```

- 예제 4

    ```python
    response = agent.invoke("거래일 컬럼에서 거래월을 분리해서 새로운 컬럼으로 만들어주는 코드를 짜줄래?")
    print(response["output"])
    ```

    ```python
    df['거래일'].str.split('-').str[1].astype(int).rename('거래월')
    df
    ```

    ```python
    agent = reset_agent(df)
    ```

- 예제 5

    ```python
    response = agent.invoke("1평은 3.3 제곱미터이니, '전용면적' 컬럼의 값들을 3.3으로 나누고 소수점 둘째자리까지 표시한 '평'이라는 이름의 컬럼을 새롭게 만들어주는 코드를 짜줘.")
    print(response["output"])
    ```

    ```python
    df['평'] = (df['전용면적'].astype(float) / 3.3).round(2)
    df
    ```

- 예제 6

    ```python
    ref = """
    1) 평이 18 이하면 소형,
    2) 18보다 크고 25 이하면 중형,
    3) 25보다 크고 31 이하면 중대형,
    4) 31보다 크면 대형으로 분류됩니다
    """

    response = agent.invoke("{}를 참고하여, '평' 컬럼을 '유형'이라는 이름의 컬럼으로 새롭게 만들어주는 코드를 한 줄로 짜줄래?".format(ref))
    print(response["output"])
    ```

    ```python
    df['유형'] = df['평'].apply(lambda x: '소형' if x <= 18 else '중형' if 18 < x <= 25 else '중대형' if 25 < x <= 31 else '대형')
    df
    ```

    ```python
    df.to_csv("./data/pp_df.csv", index=False)
    ```

    ```python
    df = pd.read_csv("./data/pp_df.csv")
    df
    ```

- 예제 7

    ```python
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    fe = fm.FontEntry(fname='./NanumGothic.ttf', name='NanumGothic')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})
    plt.rcParams["axes.unicode_minus"] = False
    ```

    ```python
    import warnings
    warnings.filterwarnings("ignore")
    ```

    ```python
    response = agent.invoke("아파트 컬럼을 기준으로 거래금액의 평균값에 대해 가장 비싼 아파트 단지 Top10을 x축으로, 거래금액 컬럼을 y축으로 한 막대 그래프를 예쁘게 그려줘. 제목은 한글로 만들어주면 좋겠어")
    print(response["output"])
    ```

    ```python
    df['거래금액'].value_counts().plot(kind='bar', figsize=(10,6), rot=0)
    plt.title('아파트 단지별 거래금액 평균')
    plt.xlabel('아파트')
    plt.ylabel('거래금액')
    plt.show()
    ```