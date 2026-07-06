---
layout: page
title:  "MinIO - VectorDB 기반의 Hybrid Search RAG"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-99-04-01_01-HybridSearchRag
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - **실습 소개**
>   - 앞서 구축한 MinIO(오브젝트 스토리지)와 **Qdrant(벡터 DB)** 아키텍처 위에,
>   - 로컬 LLM인 Ollama(Gemma4 계열 모델 기반)와 **LangChain** 프레임워크를 얹어
>   - 하이브리드 서치(Sparse + Dense)가 적용된
>   - 고성능 로컬 RAG 시스템을
>   - 엔드투엔드로 구현
{: .common-quote}

- **Dense 검색:** 문맥과 의미를 파악하는 벡터 유사도 검색 (Qdrant 내장)
- **Sparse 검색:** 고유명사, 숫자, 가변 면적 유량계 등의 정확한 키워드를 매칭하는 BM25 검색 (LangChain 컴포넌트 활용)
- **하이브리드 결합:** 두 검색 결과를 융합(Reciprocal Rank Fusion, RRF 등)하여 LLM에게 최적의 컨텍스트를 제공



## 1. 전체 시스템 아키텍처 및 폴더 구조

- 실습을 진행하기 전, 프로젝트 루트 디렉터리를 생성하고 구조화 수행

```text
local-rag-project/
├── data/                  # 테스트용 샘플 문서 저장 폴더
│   └── manual_sample.txt
├── docker-compose.yml     # 인프라(MinIO, Qdrant, Ollama) 설정
└── rag_pipeline.py        # LangChain 하이브리드 RAG 메인 소스 코드
```


## 2. `docker-compose.yml` 작성 및 인프라 구동

- Ollama 컨테이너를 추가하고 로컬 GPU(NVIDIA) 가속 설정을 포함한 최종 인프라 명세

```yaml
services:
    # 1. 로컬 LLM 및 임베딩을 구동할 Ollama (RTX 4080 등 GPU 가속 설정)
    ollama:
        image: ollama/ollama:latest
        container_name: ollama-local
        ports:
            - "11434:11434"
        volumes:
            - ollama_data:/root/.ollama
        deploy:
            resources:
                reservations:
                    devices:
                        - driver: nvidia
                          count: all
                          capabilities: [gpu]
        restart: unless-stopped

    # 2. 원본 문서를 저장할 MinIO (2024년 안정 버전 지정)
    minio:
        image: minio/minio:RELEASE.2024-01-11T07-46-16Z
        container_name: minio-local
        ports:
            - "9000:9000"
            - "9001:9001"
        environment:
            MINIO_ROOT_USER: minioadmin
            MINIO_ROOT_PASSWORD: minioadminpassword
        command: server /data --console-address ":9001"
        volumes:
            - minio_data:/data
        restart: unless-stopped

    # 3. 고속 벡터 검색 및 페이로드 필터링을 위한 Qdrant
    qdrant:
        image: qdrant/qdrant:latest
        container_name: qdrant-local
        ports:
            - "6333:6333"
            - "6334:6334"
        volumes:
            - qdrant_data:/qdrant/storage
        restart: unless-stopped

volumes:
    ollama_data:
    minio_data:
    qdrant_data:
```

- **인프라 구동 및 모델 다운로드**

```bash
# 1. 컨테이너 일괄 구동 (-d: 백그라운드)
docker compose up -d

# 2. Ollama 컨테이너 내부로 진입하여 Gemma 모델 다운로드 (예: gemma2 또는 gemma:9b 등 환경에 맞게 선택)
docker exec -it ollama-local ollama run gemma2
```

> *참고: 다운로드가 완료되어 대화창이 뜨면 `/bye`를 입력하여 빠져나옴*


## 3. 파이썬 개발 환경 세팅

- **라이브러리 설치**
    - LangChain의 최신 패키지 구조와 하이브리드 서치(BM25) 구현을 위해 필요한 핵심 라이브러리들을 설치

    ```bash
    pip install langchain langchain-community langchain-ollama qdrant-client minio rank-bm25
    ```


- **실습 데이터 파일 생성**
    - `data/manual_sample.txt` 경로에 테스트용 도메인 지식 문서 생성
    - 키워드 검색과 의미 검색의 차이를 극명하게 볼 수 있도록 고유 명사와 기술 스펙 위주로 작성

    ```text
    [문서 ID: 101] 스마트팩토리 가변 면적 유량계(Model: FM-2026)의 핵심 센서 정밀 보정 주기는 연 1회(매년 3월) 필수 진행해야 합니다. 
    [문서 ID: 102] HNSW 인덱싱 알고리즘은 메모리(RAM) 소모량이 타 알고리즘 대비 크지만, 대규모 벡터 공간에서 초고속 ANN 검색을 보장하는 특성이 있습니다.
    [문서 ID: 103] 기업 보안 규정 지침에 따라, 모든 사내 RAG 시스템의 데이터 적재 시 멀티테넌시 격리 수준을 등급 3단계 이상으로 적용해야 합니다.
    ```


## 4. 통합 RAG 파이프라인 구현

- **[1. 로컬 파일 읽기 → 2. MinIO 백업 → 3. 텍스트 분할 → 4. Qdrant (Dense) & BM25 (Sparse) 하이브리드 리트리버 구성 → 5. 로컬 LLM을 통한 RAG 답변 생성]** 프로세스

```python
#//file: "rag_pipeline.py"
import io
import os
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers import EnsembleRetriever

# ==========================================
# 1. 기본 인프라 환경 설정 및 클라이언트 초기화
# ==========================================
MINIO_URL = "localhost:9000"
MINIO_ACCESS = "minioadmin"
MINIO_SECRET = "minioadminpassword"
BUCKET_NAME = "rag-source-vault"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hybrid_rag_idx"

# 클라이언트 생성
minio_client = Minio(MINIO_URL, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

qdrant_client = QdrantClient(url=QDRANT_URL)

# Ollama 모델 컴포넌트 선언 (컨테이너 내부에서 다운로드한 모델 매핑)
MODEL_NAME = "gemma4" 
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url="http://localhost:11434")
llm = ChatOllama(model=MODEL_NAME, base_url="http://localhost:11434", temperature=0.1)

print("모든 인프라 및 AI 모델 컴포넌트 로드 완료.")

# ==========================================
# 2. 데이터 적재 및 이원화 파이프라인 (MinIO & 로컬 파일)
# ==========================================
# A. 로컬 원시 파일 로드 및 MinIO 백업 동기화
file_path = "data/manual_sample.txt"
file_name = os.path.basename(file_path)

with open(file_path, "rb") as f:
    file_data = f.read()
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=file_name,
        data=io.BytesIO(file_data),
        length=len(file_data),
        content_type="text/plain"
    )
print(f"원본 문서 '{file_name}' 가 MinIO [{BUCKET_NAME}] 버킷에 안전하게 백업되었습니다.")

# B. LangChain 구조로 텍스트 로드 및 청킹(Chunking)
loader = TextLoader(file_path, encoding="utf-8")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
docs = text_splitter.split_documents(raw_documents)

# 메타데이터에 MinIO 참조 키를 강제로 바인딩하여 데이터 추적성 확보
for idx, doc in enumerate(docs):
    doc.metadata["source_storage"] = "minio"
    doc.metadata["minio_bucket"] = BUCKET_NAME
    doc.metadata["minio_object_key"] = file_name
    doc.metadata["chunk_id"] = idx

print(f"원본 문서를 총 {len(docs)}개의 의미론적 청크로 분할 완료.")

# ==========================================
# 3. 하이브리드 검색기(Ensemble Retriever) 설계
# ==========================================
# Dense 검색기 기동 (Qdrant Vector Store 빌드)
vector_store = Qdrant.from_documents(
    documents=docs,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True # 실습 반복을 위해 매번 컬렉션 초기화 재생성
)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Sparse 검색기 기동 (자연어 토큰 기반 BM25 검색)
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 2

# 가중치 결합형 앙상블 리트리버 구성 (Dense 50% + Sparse 50%)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)
print("Qdrant(Dense)와 BM25(Sparse)가 결합된 하이브리드 검색기 빌드 완료.")

# ==========================================
# 4. LangChain RAG 체인 설계 (가동 스위치)
# ==========================================
# LLM에게 전달할 프롬프트 구조화
rag_prompt = ChatPromptTemplate.from_template("""
귀하는 스마트팩토리 및 고도의 AI 기술 문서를 분석하는 전문 엔지니어입니다.
반드시 아래 제공된 컨텍스트(Context) 정보만을 기반으로 사용자의 질문에 사실적인 답변을 하십시오.
만약 컨텍스트에 없는 내용이거나 답변을 유추할 수 없다면 절대 지어내지 말고 "제공된 문서에 관련 정보가 없습니다"라고 답하십시오.

[Context]
{context}

Question: {question}
Answer:""")

# 컨텍스트 문서들을 가독성 있게 포맷팅하는 유틸리티 함수
def format_docs(input_docs):
    formatted = []
    for d in input_docs:
        meta = d.metadata
        formatted.append(f"내용: {d.page_content}\n(출처: {meta.get('minio_bucket')}/{meta.get('minio_object_key')} | Chunk: {meta.get('chunk_id')})")
    return "\n\n".join(formatted)

# 데이터 흐름 제어를 위한 랭체인 표현식(LCEL) 체인 구성
rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("RAG 지식 추론 체인 활성화 완료.\n" + "="*50)

# ==========================================
# 5. 확인 테스트 및 지식 검증
# ==========================================
def run_test(query):
    print(f"\n[질문 입력]: {query}")
    # 1단계: 하이브리드 검색기가 가져온 원본 청크 데이터 확인
    retrieved_chunks = hybrid_retriever.invoke(query)
    print("--------------------------------------------------")
    print("[하이브리드 교차 매칭된 지식 청크]")
    for c in retrieved_chunks:
        print(f"- {c.page_content} (메타데이터: {c.metadata.get('minio_object_key')})")
    print("--------------------------------------------------")
    
    # 2단계: RAG 가동 및 최종 답변 추출
    response = rag_chain.invoke(query)
    print(f"[Gemma4 로컬 AI 추론 답변]:\n{response}")
    print("="*50)

# 테스트 실행 (1번: 키워드 성격이 강한 질문 / 2번: 맥락 중심의 의미론적 질문)
run_test("유량계 모델 FM-2026 정밀 보정은 언제 해야 하나요?")
run_test("메모리를 많이 쓰더라도 가장 빠른 벡터 검색 알고리즘이 뭔지 설명해줘.")
```


## 5. 구동 확인 및 테스트 검증 결과 확인

- 코드를 실행하기 전 각 인프라가 정상 상태인지 선행 확인하고 테스트 진행

    - **Step 1: 인프라 컨테이너 상태 점검**
        - 3개의 핵심 서비스가 업타임(Up) 상태인지 확인

        ```bash
        docker compose ps
        ```

        - **확인 포인트:**
            - `ollama-local`, `minio-local`, `qdrant-local` 컨테이너의 State가 모두 `Up`이어야 함

    - **Step 2: 파이프라인 스크립트 실행**
        
        ```bash
        python rag_pipeline.py
        ```

- **테스트 시나리오: 무엇을, 왜 테스트하는가?**
    - RAG 시스템의 성능을 다각도로 검증하기 위해 **성격이 전혀 다른 3가지 질문**을 던져 시스템의 핵심 메커니즘을 테스트

    - **테스트 1: 고유명사 및 정밀 스펙 매칭 테스트**
        - **질문:**
            - `"유량계 모델 FM-2026 정밀 보정은 언제 해야 하나요?"`

        - **무엇을 테스트하는가?:**
            - 알파벳과 숫자가 결합된 고유 식별자(`FM-2026`) 및 구체적인 수치 정보(`연 1회`, `3월`)를 시스템이 정확히 끄집어내는지 테스트

        - **왜 테스트하는가?:**
            - 일반적인 벡터(Dense) 검색은 '유량계 보정'이라는 전체적인 맥락은 잘 짚지만,
            - `FM-2026`, `FM-2025` 같은 **미세한 모델명 차이를 구분하지 못하고 뭉뚱그리는 경향**이 있음
            - 키워드 기반의 BM25(Sparse) 검색이 이 약점을 잘 보완하여 핀포인트로 올바른 문서를 찾아오는지 검증하기 위함

    - **테스트 2: 문맥 및 의미론적 유추 테스트**
        - **질문:**
            - `"메모리를 많이 쓰더라도 가장 빠른 벡터 검색 알고리즘이 뭔지 설명해줘."`

        - **무엇을 테스트하는가?:**
            - 질문에 'HNSW'라는 단어가 단 한 글자도 포함되어 있지 않음에도, '메모리를 많이 쓴다', '가장 빠른 벡터 검색'이라는 **인간의 문맥적 뉘앙스를 이해하여 관련 문서를 찾아내는지** 테스트

        - **왜 테스트하는가?:**
            - 전통적인 키워드 검색은 사용자가 정확한 기술 용어(HNSW)를 모르면 관련 매뉴얼을 찾을 수 없음
            - 임베딩 모델(Dense 검색)이 단어의 표면적 일치가 아닌 개념적 유사성(Semantic Similarity)을 기반으로 지식을 탐색할 수 있는지 검증하기 위함

    - **테스트 3: 환각(Hallucination) 제어 및 방어벽 테스트**
        - **질문:**
            - `"스마트팩토리 제어실의 실내 적정 온습도 기준은 어떻게 되나요?"`

        - **무엇을 테스트하는가?:**
            - 지식 베이스(`manual_sample.txt`)에 **존재하지 않는 정보**를 질문했을 때,
            - LLM이 거짓말을 지어내지 않고 거절하는지 테스트

        - **왜 테스트하는가?:**
            - 생성형 AI의 가장 큰 고질병은 모르는 것도 그럴듯하게 지어내는 '환각 현상'
            - 우리가 프롬프트에 설정한 격리 지침(`제공된 문서에 관련 정보가 없다면 ~ 라고 답하십시오`)이 제대로 작동하여 **엔터프라이즈 환경에서 요구하는 안전성**을 확보했는지 검증하기 위함


- **기대 출력**
    - 스크립트 실행 시 콘솔에 출력되는 원시 데이터와 LLM의 최종 추론 결과

    - **결과 1: 고유명사 매칭 (성공)**
        - **하이브리드 리트리버 추출 결과:**

        ```text
        - [문서 ID: 101] 스마트팩토리 가변 면적 유량계(Model: FM-2026)의 핵심 센서 정밀 보정 주기는 연 1회(매년 3월) 필수 진행해야 합니다.
        ```

        - **로컬 AI 추론 답변:**

        ```text
        스마트팩토리 가변 면적 유량계(Model: FM-2026)의 핵심 센서 정밀 보정은 연 1회 필수 진행해야 하며, 구체적인 시기는 매년 3월입니다.
        ```

    - **결과 2: 의미론적 유추 (성공)**
        - **하이브리드 리트리버 추출 결과:**

        ```text
        - [문서 ID: 102] HNSW 인덱싱 알고리즘은 메모리(RAM) 소모량이 타 알고리즘 대비 크지만, 대규모 벡터 공간에서 초고속 ANN 검색을 보장하는 특성이 있습니다.
        ```

        - **로컬 AI 추론 답변:**

        ```text
        메모리(RAM) 소모량이 상대적으로 크지만, 대규모 벡터 공간에서 초고속 ANN 검색을 보장하는 알고리즘은 HNSW 인덱싱 알고리즘입니다.
        ```

    - **결과 3: 환각 제어 방어 (성공)**
        - **하이브리드 리트리버 추출 결과:**
            - `[검색된 관련 문서 청크 없음]`

        - **로컬 AI 추론 답변:**

        ```text
        제공된 문서에 관련 정보가 없습니다.
        ```

- **도출된 결과의 기술적 의의 (What it means)**
    - 이 세 가지 테스트가 모두 성공했다는 것은 구축된 시스템이 다음과 같은 **기술적 가치와 의의**를 확보했음을 의미

    - **하이브리드 검색(Ensemble)의 실효성 증명**
        - 숫자와 모델명에 강한 BM25와 뉘앙스에 강한 Qdrant 벡터 검색이 50:50으로 융합되어,
        - 검색 누락(Miss)이나 엉뚱한 문서를 참조하는 비율을 극적으로 낮추었음을 의미

    - **이원화 저장소 매핑 및 추적성 확보**
        - 결과 데이터의 메타데이터에 `rag-source-vault/manual_sample.txt`와 같은 MinIO 주소가 명확히 유지됨
        - 이는 AI가 답변을 고안할 때 **"내가 이 답변을 스토리지의 어떤 파일에서 읽고 말하는 것인지"** 출처를 100% 추적할 수 있음을 의미

    - **완벽한 데이터 주권(Data Sovereignty) 실현**
        - 오픈AI 등 외부 API를 일절 호출하지 않고,
        - 내부 인프라(MinIO, Qdrant)와 로컬 GPU로 구동되는 Ollama(Gemma4)만으로 상용 수준의 RAG를 구현
        - 즉, **기업 내부의 대외비 매뉴얼이나 기밀 소스 코드가 외부로 단 1바이트도 유출되지 않는 안전망**이 완성되었음을 의미