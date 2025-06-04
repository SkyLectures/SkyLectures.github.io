---
layout: page
title:  "LangChain 기반으로 GPT 사용하기"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-05_01-LangChainGpt
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Simple Chat

- 설치할 라이브러리

    ```bash
    pip install langchain langchain_community
    ```

- 예제 코드

    ```python
    # Simple Chat

    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    response = llm.invoke([
        HumanMessage(content="서울의 날씨는 보통 어떤가요?")
    ])
    print(response.content)
    ```

- 참고
    - langchain_core, langchain_openai, langchain_community 모듈이 분리되어 사용됨
    - invoke(), batch(), stream() 방식이 기본 호출 방식으로 통일
    - ChatOpenAI, OpenAIEmbeddings 등은 더 이상 langchain.llms에서 불러오지 않음

## 2. 문서 요약기 (Text Summarization)

- 설치할 라이브러리

    ```bash
    pip install langchain langchain_community langchain-openai
    ```

- 예제 코드

    ```python
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.summarize import load_summarize_chain
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI

    # 1. 문서 로드 및 분할
    loader = TextLoader("sample.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # 2. LLM 설정
    llm = OpenAI(temperature=0)

    # 3. 프롬프트 템플릿 (한국어 요약 요청)
    map_prompt = PromptTemplate.from_template(
        "다음 문서를 읽고 핵심 내용을 한국어로 요약하세요:\n\n{text}"
    )
    combine_prompt = PromptTemplate.from_template(
        "다음은 여러 문서의 요약입니다. 이를 종합하여 핵심 내용을 한국어로 요약하세요:\n\n{text}"
    )

    # 4. 체인 생성
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt
    )

    # 5. 실행
    summary = chain.invoke(split_docs)

    # 6. 출력
    print(summary['output_text'])
    ```

- 기존 코드 변경사항

| 이전 방식 (`.run()`)  | 최신 방식 (`.invoke()`)                 |
| ----------------- | ----------------------------------- |
| `chain.run(docs)` | `chain.invoke(docs)`                |
| 결과 직접 문자열 반환      | `dict` 형식 반환 (`'output_text'` 키 사용) |

- 참고
    - LangChain에서 응답 언어를 한국어로 고정하려면 프롬프트 또는 시스템 메시지에 "한국어로 대답하세요" 등의 지시를 포함해야 함
    - 포함하지 않아도 한국어로 잘 나오기도 하지만 가끔 한글 문서를 읽고 영문으로 대답하는 경우가 있음
    - 요약 체인을 사용할 때는 LLM에 전달되는 프롬프트에 한국어 응답 조건을 추가하는 방식으로 해결할 수 있음

## 3. 문서 기반 질문응답 (Retrieval QA)

- 설치할 라이브러리

    ```bash
    pip install langchain langchain_community langchain-openai faiss-cpu
    ```

- 예제 코드

    ```python
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import RetrievalQA

    # 1. 문서 로드 및 분할
    loader = TextLoader("sample.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # 2. 임베딩 및 벡터스토어
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 최신 모델 사용
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 3. QA 체인 생성
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 4. 질문 실행
    query = "이 문서에서 핵심 내용은 무엇인가요?"
    result = qa_chain.invoke(query)

    print(result['result'])
    ```

- FAISS (Facebook AI Similarity Search) 
    - Facebook AI Research에서 개발한 고속 유사도 검색 라이브러리
    - 대규모 벡터(벡터 임베딩) 데이터에서 가장 유사한 항목을 빠르게 찾도록 최적화된 도구
    - 주요 특징
        - 초대형 벡터 데이터셋 처리: 수백만~수십억 개 벡터에 대한 효율적 검색 가능
        - 고속 근사 최근접 이웃 검색(ANN): 정확도와 속도 간 균형을 맞춘 근사 검색 알고리즘 제공
        - CPU/GPU 지원: CPU와 GPU 모두에서 작동해 빠른 연산 지원
        - 다양한 인덱스 구조: IVF, PQ, HNSW 등 여러 검색 인덱스 구조 제공
    - FAISS가 불편하거나 설치가 어려운 경우, 대안

| 벡터스토어                                  | 특징                            |
| -------------------------------------- | ----------------------------- |
| **Chroma**                             | LangChain 기본 지원, pip만으로 설치 가능 |
| **Weaviate**, **Pinecone**, **Qdrant** | 클라우드/로컬 사용 가능, 고성능            |


## 4. 프롬프트 템플릿을 사용한 이메일 생성기

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# 1. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
당신은 이메일 작성 도우미입니다.
다음 정보를 바탕으로 정중한 한국어 이메일을 작성하세요:

받는 사람: {recipient}
주제: {subject}
요약: {summary}
""")

# 2. LLM 정의
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 3. 체인 구성: PromptTemplate → ChatOpenAI
chain: RunnableSequence = prompt | llm

# 4. 입력값 정의
inputs = {
    "recipient": "김철수 과장님",
    "subject": "회의 일정 변경 요청",
    "summary": "이번 주 금요일 오전 회의를 다음 주로 미루고자 합니다."
}

# 5. 실행
response = chain.invoke(inputs)
print(response.content)
```

- 변경사항

| 이전 방식 (`LLMChain`)              | 최신 방식 (`RunnableSequence`)    |       |
| ------------------------------- | ----------------------------- | ----- |
| `LLMChain(prompt=..., llm=...)` | \`Runnable = PromptTemplate   | LLM\` |
| `.run(input_dict)`              | `.invoke(input_dict)`         |       |
| LangChain 내부 종속성에 강함            | 더 유연한 조합 가능 (ex. 여러 단계 체인 구성) |       |
