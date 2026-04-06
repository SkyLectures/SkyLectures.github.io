---
layout: page
title:  "RAG 요소: 청킹, 검색 방법, Vector DB"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-06-01_02-RagComponents
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

> - RAG 시스템의 성능을 좌우하는 3대 핵심 요소: **청킹(Chunking), 검색(Retrieval), 그리고 벡터 DB(Vector Database)**
{: .common-quote}


## 1. 청킹 (Chunking)

> - 데이터 엔지니어링에서 대용량 데이터를 처리하기 위해 일정한 크기로 나누는 작업
> - 메모리 효율성을 높이고 병렬 처리를 가능하게 하기 위한 기본 기법
{: .common-quote}

- **RAG에서의 상세 역할**
    - RAG에서 청킹은 **'검색의 단위'**를 결정하는 매우 전략적인 단계
    - LLM은 입력 가능한 토큰(Token) 제한이 있음 🡲 방대한 문서를 의미 있는 조각으로 잘라야 함

    - **Fixed-size Chunking** 
        - 단순히 글자 수나 토큰 수로 자르는 방식
        - 구현은 쉽지만 문맥이 잘릴 위험이 큼
    - **Recursive Character Chunking**
        - 마침표, 줄바꿈 등을 기준으로 문맥을 최대한 보존하며 자르는 방식
    - **Semantic Chunking**
        - NLP 모델을 사용하여 의미가 변하는 지점을 포착해 자름
        - 가장 정교하지만 연산 비용이 높음
    - **Overlap (중복 허용)**
        - 청크와 청크 사이에 일정 부분(예: 10~20%)을 중복시켜,
        - 잘린 부분의 문맥이 검색 시 누락되지 않도록 방어하는 기술이 핵심


## 2. 검색 방법 (Retrieval Methods)

> - 전통적인 정보 검색(Information Retrieval)
>   - 사용자의 질의에 부합하는 데이터를 데이터베이스나 인덱스에서 찾아내는 과정
>   - 예시: SQL의 `LIKE` 검색이나 키워드 매칭
{: .common-quote}

- **RAG에서의 상세 역할**
    - RAG의 검색은 단순히 단어가 포함되었는지를 보는 것이 아니라,
    - **'질문의 의도'**와 **'문서의 내용'**이 얼마나 가까운지를 계산함

    - **Dense Retrieval (밀집 검색)**
        - 텍스트를 고차원 벡터로 변환하여 수치적 유사도를 계산
        - "배가 고프다"와 "식사하고 싶다"처럼 단어는 다르지만 의미가 같은 데이터를 찾아낼 수 있음
    - **Sparse Retrieval (희소 검색)**
        - BM25와 같은 알고리즘을 사용
        - 특정 핵심 키워드가 정확히 일치하는지 확인
        - 고유명사나 전문 용어 검색에 유리함
    - **Hybrid Retrieval**
        - 위 두 방식을 결합하여 의미와 키워드 정확도를 동시에 확보
    - **Top-k Retrieval**
        - 검색된 결과 중 유사도 점수가 가장 높은 상위 $$k$$개만을 선별하여 LLM에게 전달
        - 정보의 밀도를 높임


## 3. 벡터 데이터베이스 (Vector DB)

> - 수치화된 데이터(벡터)를 저장하고 관리하는 특수 목적용 데이터베이스
> - 고차원 공간에서 데이터 간의 거리를 빠르게 계산하는 데 최적화되어 있음
{: .common-quote}

- **RAG에서의 상세 역할**
    - RAG 아키텍처에서 벡터 DB는 **'장기 기억 저장소(Long-term Memory)'** 역할

    - **임베딩 저장**
        - 텍스트 청크를 벡터로 변환한 결과물($$Embedding$$)을 수만 개에서 수억 개까지 저장
    - **고속 근사 근접 이웃 검색 (ANN)**
        - 수백만 개의 데이터 사이에서 일일이 거리를 계산하면 속도가 느려짐
        - HNSW(Hierarchical Navigable Small World)와 같은 알고리즘을 사용
        - **밀리초(ms) 단위**로 유사한 데이터를 검색
    - **메타데이터 필터링**
        - 벡터 값뿐만 아니라 생성 날짜, 문서 카테고리 등 일반 데이터(Metadata)를 함께 저장
        - 특정 조건(예: '2025년 이후 문서만 검색') 하에 벡터 검색을 수행할 수 있게 함
    - **대표적 솔루션** 
        - Milvus, Pinecone, Weaviate, FAISS(Facebook), Chroma 등
        - 최근에는 PostgreSQL(pgvector) 같은 전통적 DB에도 관련 기능이 통합되고 있음


## 4. 요약

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">요소</th>
        <th style="width: 400px;">핵심 키워드</th>
        <th style="width: 400px;">비유</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">청킹</td>
            <td class="td-left">Context Window, Overlap</td>
            <td class="td-left">책의 페이지를 지능적으로 나누기</td>
        </tr>
        <tr>
            <td class="td-rowheader">검색</td>
            <td class="td-left">Similarity, Top-k</td>
            <td class="td-left">질문과 가장 관련 있는 페이지 찾기</td>
        </tr>
        <tr>
            <td class="td-rowheader">벡터 DB</td>
            <td class="td-left">Embedding, HNSW, Latent Space</td>
            <td class="td-left">페이지들이 꽂혀 있는 초고속 도서관</td>
        </tr>
    </tbody>
</table>
</div>

> - ERP 개발이나 스마트팩토리 연구 시, 수많은 기술 로그나 매뉴얼 데이터를 효율적으로 처리하려면
> - **'청킹 전략'**을 데이터의 특성(코드인지, 서술형 문장인지)에 따라 다르게 가져가는 것이 구현 단계의 핵심 노하우가 될 것
{: .expert-quote}
