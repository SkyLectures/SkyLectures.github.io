---
layout: page
title:  "LangChain"
date:   2025-04-02 09:00:00 +0900
permalink: /materials/S03-05-03-04_01-LangChainDataAnalysis
categories: materials
---

## LLM + LangChain으로 데이터 처리하기

### 1. LangChain
- 대규모 언어 모델(LLM)로 구동되는 애플리케이션을 개발하기 위한 오픈소스 프레임워크
- Python과 JavaScript 라리브러리를 제공함
- 기본적으로 거의 모든 LLM을 위한 일반적인 인터페이스이므로 LLM 애플리케이션을 구축한 다음 통합할 수 있는 중앙집중식 개발 환경을 갖추고 있음
- LLM 애플리케이션 라이프사이클의 모든 단계에 대한 간소화를 지원함
- 기존의 라이브러리를 사용할 경우 각 버전의 차이 및 개발팀에 따른 호환성 문제가 자주 발생하는데 LangChain과 같이 통합을 지원하는 프레임워크를 사용하면 이러한 문제를 최소화할 수 있음


### 2. LangChain 사용해 보기

<span style="display:block; text-align:right">[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part03_AI/S03-05-03-02_01_LangChain_Pandas.ipynb)</span>

#### 2.1 필요한 라이브러리 설치

```bash
pip install numpy pandas matplotlib wget
pip install ipykernel
pip install langchain langchain_experimental langchain_community
```


#### 2.2 실습용 데이터셋 다운로드

```python
import wget

file_url = "https://raw.githubusercontent.com/HelloDataScience/Datasets/refs/heads/main/APT_Price_GangNamGu_2023_20230731.txt"
out_filename = "./data/Apart_Price_2023.tsv"
wget.download(file_url, out_filename)
```


#### 2.3 데이터셋 로드

- 해당 데이터셋은 UTF-16 코드셋으로 저장되어 있음

```python
import pandas as pd

df = pd.read_csv(out_filename, sep="\t", encoding="utf-16")
df
```


#### 2.4 LangChain 설정

- 기본 규칙
    - 기본 언어모델은 Meta의 Llama 3.2를 사용함
    - GPT, Gemini 등의 메이저 모델(주로 유료)은 전용 클래스를 사용하며 무료로 제공되는 오픈소스 모델은 주로 Ollama를 이용하여 설정함
    - 실습 예제에서는 OpenAI의 GPT-3.5-Turbo를 사용할 때의 코드를 주석으로 함께 제공함
    - agent가 생성된 이후는 모든 언어모델에 대하여 동일한 코드를 사용함

- ChatOllama를 이용하여 오픈소스 언어모델을 사용할 경우
    - 로컬 시스템에 Ollama가 설치되어 있어야 함
    - "ollama run llama3.2" 명령을 통해 llama3.2 언어모델이 로컬 시스템에 설치되어 작동하고 있어야 함

- 모듈 import

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOllama
```

- Pandas기반 에이전트 생성
    - 데이터 분석에 활용하기 위하여 Pandas를 기반으로 하여 사용함

```python
agent = create_pandas_dataframe_agent(ChatOllama(temperature=0, model='llama3.2'), df, verbose=True, allow_dangerous_code=True)
```

- 에이전트 초기화 함수
    - 작업 시 에이전트의 초기화 여부에 분석 결과가 영향을 많이 받음

```python
def reset_agent(df, temperature=0, verbose=True):
    return create_pandas_dataframe_agent(ChatOllama(temperature=0, model='llama3.2'), df, verbose=True, allow_dangerous_code=True)
```

- [참고] OpenAI의 GPT-3.5-Trubo 모델을 사용하는 예제코드
    - API_KEY를 환경 설정에 넣거나 create_pandas_dataframe_agene 함수 호출 시 파리미터로 전달해 주어야 함
    - OpenAI API와 Ollama의 사용을 비교해 볼 것

```python
#!pip install openai

# import os
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.chat_models import ChatOpenAI

# os.environ["OPENAI_API_KEY"] = "***************************************************"
# agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), df, verbose=True)

# def reset_agent(df, temperature=0, verbose=True):
#     return create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), df, verbose=verbose)
```

- create_pandas_dataframe_agent 함수를 통해 에이전트를 생성하면
    - 해당 에이전트는 Pandas의 데이터 프레임을 기반으로 작동하도록 설정됨
    - 따라서 주어진 데이터셋을 기반으로 대화, 질의응답이 가능해 짐

#### 2.5 예제

- 예제 1

```python
response = agent.invoke("전체 데이터의 갯수와 자료형 등을 확인해줄래?")
print(response["output"])
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