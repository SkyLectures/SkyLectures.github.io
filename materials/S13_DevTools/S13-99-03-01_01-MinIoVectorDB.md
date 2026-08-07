---
layout: page
title:  "MinIO - VectorDB 연동"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-99-03-01_01-MinIoVectorDB
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}




> - **MinIO(오브젝트 스토리지)와 Qdrant(벡터 데이터베이스)를 연동하는 아키텍처**
>   - 멀티미디어(이미지, 오디오, 대용량 텍스트) 기반의 RAG(검색 증강 생성) 또는 멀티모달 AI 시스템을 구축할 때 가장 표준적으로 사용되는 조합
>   - **연동의 핵심 메커니즘**
>       - MinIO: 원본 데이터(비정형 파일)를 저장
>       - Qdrant: 해당 파일의 고차원 임베딩 벡터와 함께 **MinIO의 파일 다운로드 URL(또는 Object Key)을 메타데이터로 저장**
>       - MioIO와 Qdrant 동기화
{: .common-quote}



## 1. 개발 환경 구성 (Docker Compose)

- **docker-compose.yml 구성**
    - 두 서비스와 파이프라인 검증용 주피터 노트북(또는 파이썬 실행 환경)을 한 번에 구동할 수 있는 `docker-compose.yml`

    ```yaml
    services:
        # 1. 원본 비정형 데이터를 저장할 MinIO
        minio:
            image: minio/minio:RELEASE.2024-01-11T07-46-16Z
            container_name: minio-local
            ports:
                - "9000:9000"       # API 포트
                - "9001:9001"       # Web 콘솔 포트
            environment:
                MINIO_ROOT_USER: minioadmin
                MINIO_ROOT_PASSWORD: minioadminpassword
            command: server /data --console-address ":9001"
            volumes:
                - minio_data:/data
            restart: unless-stopped

        # 2. 고속 벡터 검색을 위한 Qdrant
        qdrant:
            image: qdrant/qdrant:latest
            container_name: qdrant-local
            ports:
                - "6333:6333"       # HTTP API 포트
                - "6334:6334"       # gRPC 포트
            volumes:
                - qdrant_data:/qdrant/storage
            restart: unless-stopped

    volumes:
        minio_data:
        qdrant_data:
    ```

    - 구성한 시스템은 MinIO(오브젝트 스토리지)와 Qdrant(벡터 DB)가 결합된 구조
        - unless-stopped 옵션이 없을 경우, 새벽에 서버가 잠깐 재부팅되거나 했을 때 DB 역할을 하는 두 서비스가 다운된 상태로 방치됨
        - 이 때 Python 애플리케이션 코드가 데이터를 적재하거나 검색하려 하면 ConnectionError를 발생시키며 전체 AI 파이프라인이 마비됨
        - unless-stopped를 넣어둠으로써, 인프라의 최소한의 가동성(Availability)과 복구 탄력성을 확보할 수 있음
        - (실습 자체에는 그다지 필요하지 않음)

    - **구동 명령:** `docker compose up -d`
    - 구동 후 브라우저에서 `http://localhost:9001`에 접속하여 `raw-documents`라는 이름의 버킷(Bucket)을 미리 생성


- **Python 필수 라이브러리 설치**
    - 연동 파이프라인 구성을 위해 MinIO 공식 SDK, Qdrant 공식 클라이언트, 그리고 임베딩 생성을 위한 라이브러리를 설치
        - 실습에서는 가장 가벼운 `SentenceTransformers`를 사용

    ```bash
    pip install minio qdrant-client sentence-transformers
    ```


## 2. MinIO-Qdrant 연동 소스 코드 (Python)

- 비정형 데이터가 AI 기술을 만나 검색 가능한 '지식 자산'으로 전환되는 엔드투엔드 파이프라인의 프로토타입
- 코드는 **`[원본 파일 준비 🡲 MinIO 업로드 🡲 임베딩 생성 🡲 Qdrant 메타데이터 연동 저장 🡲 테스트 검색]`**의 전체 수명 주기를 포함함

```python
#//file: "minio_qdrant.py"
import io
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 클라이언트 초기화 및 가상 데이터 준비
# ==========================================
# MinIO 연결
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadminpassword",
    secure=False
)
BUCKET_NAME = "raw-documents"

# Qdrant 연결
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "knowledge_base"

# 로컬 임베딩 모델 로드 (384 차원 출력)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384

# 실무 가상 데이터 (문서 ID, 파일명, 실제 본문 내용)
sample_docs = [
    {"id": 1, "filename": "smart_factory_manual.txt", "content": "스마트팩토리 가변 면적 유량계의 핵심 센서 보정 주기는 연 1회입니다."},
    {"id": 2, "filename": "ai_research_note.txt", "content": "HNSW 알고리즘은 메모리 소모가 크지만 초고속 ANN 검색을 보장합니다."},
]

# ==========================================
# 2. Qdrant 컬렉션(테이블) 생성
# ==========================================
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' 생성 완료.")

# ==========================================
# 3. 데이터 적재 파이프라인 (MinIO & Qdrant 연동)
# ==========================================
points = []

for doc in sample_docs:
    # A. 원본 텍스트 파일을 바이트 스트림으로 변환하여 MinIO에 업로드
    content_bytes = doc["content"].encode("utf-8")
    file_stream = io.BytesIO(content_bytes)
    
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=doc["filename"],
        data=file_stream,
        length=len(content_bytes),
        content_type="text/plain"
    )
    print(f"[MinIO] '{doc['filename']}' 업로드 성공.")

    # B. 본문 내용을 벡터 임베딩으로 변환
    vector = embedding_model.encode(doc["content"]).tolist()

    # C. Qdrant 페이로드(메타데이터) 구성 시 MinIO 참조 정보 기입
    payload = {
        "filename": doc["filename"],
        "minio_bucket": BUCKET_NAME,
        "minio_object_key": doc["filename"],  # 추후 MinIO에서 파일을 꺼내올 키 역할
        "preview": doc["content"][:20] + "..."  # 빠른 확인용 요약문
    }

    # D. Qdrant 데이터 구조체 생성
    points.append(
        PointStruct(id=doc["id"], vector=vector, payload=payload)
    )

# Qdrant에 벡터 및 메타데이터 일괄 Upsert
qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
print("[Qdrant] 벡터 인덱싱 완료.\n")

# ==========================================
# 4. 연동 시스템 유사도 검색 및 원본 파일 추적 테스트
# ==========================================
query_text = "유량계 센서 점검 정보가 필요해"
print(f"사용자 질의: '{query_text}'")

# 쿼리 임베딩 생성
query_vector = embedding_model.encode(query_text).tolist()

# Qdrant 검색 실행 (최신 v1.11+ 규격 매핑)
search_results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1
).points  # .points를 붙여야 하위 데이터 구조와 일치

if search_results:
    best_match = search_results[0]
    score = best_match.score
    meta = best_match.payload
    
    print(f"\n[검색 결과] 매칭 점수: {score:.4f}")
    print(f"-> 참조할 MinIO 파일명: {meta['filename']}")
    
    # E. 검색된 메타데이터를 기반으로 MinIO에서 원본 파일 가져오기
    try:
        response = minio_client.get_object(meta["minio_bucket"], meta["minio_object_key"])
        original_content = response.read().decode("utf-8")
        print(f"-> [MinIO 원본 데이터 복원 완료]:\n{original_content}")
    finally:
        response.close()
        response.release_conn()
```

- **실행 결과**

    ```text
    $ python minio_qdrant.py 
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    Loading weights: 100%|██████████████████████████████████| 103/103 [00:00<00:00, 13744.82it/s]
    [MinIO] 'smart_factory_manual.txt' 업로드 성공.
    [MinIO] 'ai_research_note.txt' 업로드 성공.
    [Qdrant] 벡터 인덱싱 완료.

    사용자 질의: '유량계 센서 점검 정보가 필요해'

    [검색 결과] 매칭 점수: 0.8104
    -> 참조할 MinIO 파일명: smart_factory_manual.txt
    -> [MinIO 원본 데이터 복원 완료]:
    스마트팩토리 가변 면적 유량계의 핵심 센서 보정 주기는 연 1회입니다.
    ```

    - **결과 내용 분석**
        - 분리되어 있던 스토리지, 인공지능 모델, 벡터 DB를 하나로 묶어주는 **'오케스트레이터(지휘자)'** 역할을 수행함<br><br>

        - **경고 구문: Hugging Face 모델 로드 관련 경고**
            - 구문 내용
                ```text
                Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
                Loading weights: 100%|██████████████████████████████████| 103/103 [00:00<00:00, 13744.82it/s]
                ```
            - 연관 소스코드

                ```python
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                ```

            - Hugging Face(HF) 허브에서 발생
            - 코드 내부에서 SentenceTransformer("all-MiniLM-L6-v2")를 호출할 때,
                - 로컬에 모델 가중치가 없거나 업데이트를 확인할 때 Hugging Face 서버(Hub)에 접속하여 모델을 다운로드/체크함
            - 이때 아무런 인증 정보(토큰) 없이 비로그인 상태로 요청을 보냈기 때문에,
                - Hugging Face 측에서 "현재 인증되지 않은 요청을 보내고 있습니다. 더 높은 요청 제한(Rate Limit)과 빠른 다운로드 속도를 원하시면 HF_TOKEN을 설정하세요."라고 안내하는 것
            - 해결 방안
                - 방법 A: Hugging Face 토큰 설정하기 (정석)

                    ```bash
                    export HF_TOKEN="발급받은_토큰_문자열"
                    python minio_qdrant.py
                    ```
                    
                - 방법 B: 코드 상단에서 경고 무시하기 (가장 간단함)

                    ```python
                    import os
                    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                    # 토큰 경고를 끄기 위해 로컬 로깅 레벨 조정
                    import logging
                    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                    ```

                - 방법 C: 그냥 무시하기
                    - 소규모 샘플 데이터나 개인 개발 환경에서 가끔 실행할 때는 인증 토큰이 없어도 모델 다운로드와 실행이 정상적으로 완료됨


        - **데이터 이원화 적재 및 인덱싱 단계**
            - 출력 내용: 

                ```text                
                [MinIO] 'smart_factory_manual.txt' 업로드 성공.
                [MinIO] 'ai_research_note.txt' 업로드 성공.
                [Qdrant] 벡터 인덱싱 완료.
                ```

            - 내용 분석
                - sample_docs에 정의된 2개의 실무 데이터가 루프를 돌며 처리됨
                    - 원본 텍스트 전체는 바이트 스트림으로 변환되어 오브젝트 스토리지(MinIO)의 raw-documents 버킷에 파일 형태로 격리 적재
                    - 본문 내용은 384차원의 실수 배열(vector)로 변환된 후, Qdrant(벡터 DB) 공간에 적재(upsert)
                    - 메타데이터인 payload에 MinIO의 버킷명과 오브젝트 키가 꼬리표로 함께 저장되어 두 저장소 간의 논리적 연결 고리 완성
                - AI가 읽을 수 없는 대용량 원본 파일(텍스트/이미지 등)은 비용이 저렴하고 안정적인 MinIO(오브젝트 스토리지)에 저장
                - AI가 의미를 해석할 수 있도록 변환한 수학적 좌표(임베딩)는 Qdrant(벡터 DB)에 저장


        - **의미론적 추상화 (Embedding Pipeline)**
            - AI 임베딩 모델(`SentenceTransformer`)을 구동하여 인간의 자연어 데이터를 고차원 벡터 공간의 숫자로 변환(Mapping)하는 핵심 연산을 수행


        - **상호 참조 메타데이터 설계 (Metadata Mapping)**
            - Qdrant에 벡터를 넣을 때,
                - 해당 벡터가 MinIO의 어떤 파일에서 나왔는지 추적할 수 있도록 `minio_object_key`를 페이로드에 저장
                - 이로써 **두 시스템 간의 강력한 데이터 링크**가 형성됨
                

        - **유사도 검색 및 매칭 점수 산출 단계**
            - 출력 내용:

                ```text
                사용자 질의: '유량계 센서 점검 정보가 필요해'

                [검색 결과] 매칭 점수: 0.8104
                -> 참조할 MinIO 파일명: smart_factory_manual.txt
                ```

            - 내용 분석
                - 사용자의 질의 텍스트가 임베딩 모델을 거쳐 query_vector로 변환됨
                - 최신 규격인 query_points를 통해 Qdrant 공간 내에서 코사인 유사도 연산이 수행됨
                - 가장 가까운 포인트 1개(limit=1)를 추출
                - 질문에 '보정 주기' 대신 '점검 정보'라는 유의어가 쓰였음에도 의미적 유연성에 의해 0.8104 (약 81%의 유사도)라는 높은 점수로 smart_factory_manual.txt와 매칭되는 포인트를 검색


        - **메타데이터 추적을 통한 원본 데이터 복원 단계**
            - 출력 내용:

                ```text
                -> [MinIO 원본 데이터 복원 완료]:
                스마트팩토리 가변 면적 유량계의 핵심 센서 보정 주기는 연 1회입니다.
                ```
            
            - 내용 분석
                - 매칭된 결과의 메타데이터를 기반으로 **MinIO에서 원본 파일을 즉시 다운로드하여 사용자에게 복원**
                - Qdrant에서 검색된 최상위 결과의 payload를 해석하여 원본 파일의 위치를 파악
                - 추출한 메타데이터(meta["minio_bucket"], meta["minio_object_key"])를 바탕으로 MinIO API를 직접 호출(get_object)
                - 파일 내부의 바이트 데이터를 가져온 뒤, utf-8 문자열로 디코딩하여 화면에 출력


- **기술적 및 아키텍처적 의의**
    - **현대적 RAG(검색 증강 생성) 시스템의 뼈대 완성**
        - LLM(대형 언어 모델)은 가짜 정보를 말하는 '환각 현상(Hallucination)'과 입력 글자 수 제한이 있음
        - 이 코드는 LLM에게 신뢰할 수 있는 외부 지식을 실시간으로 공급해 주는 **RAG 인프라의 핵심 데이터 흐름을 구현**함

    - **차원의 저주와 성능 한계 극복**
        - 수천, 수만 개의 문서 데이터를 매번 AI 모델로 전부 대조하려면 연산 비용이 폭증함
        - 이 코드는 인덱싱된 Qdrant 공간에서 근사치 검색(ANN)을 수행함으로써,
        - **대규모 데이터셋에서도 밀리초(ms) 단위의 초고속 검색이 가능함**을 증명

    - **저장소의 분산 격리(Decoupling) 🡲 느슨한 결합(Loose Coupling) 아키텍처의 실현**
        - 분산 시스템 설계 패턴을 파이썬 코드로 결합(Orchestration)
            - 쿼리 연산에 민감한 고차원 벡터 데이터는 Qdrant 메모리에 가볍게 유지
            - 무거운 원본 파일 텍스트는 오브젝트 스토리지(MinIO)가 전담
        - 대규모 엔터프라이즈 환경으로 확장할 수 있는 RAG(검색 증강 생성) 백엔드의 인프라 표준 구조 실증
        - 만약 하나의 DB에 원본 파일과 벡터를 모두 저장한다면 용량과 성능 면에서 한계에 부딪혔을 것

    - **지식 추적성(Traceability) 입증**
        - 벡터 DB 검색 결과에서 멈추지 않고,
        - 메타데이터에 심어둔 식별 정보 링크를 추적하여
        - 실제 물리 저장소(MinIO)에 있는 원본 도큐먼트의 텍스트 콘텐츠를 완벽히 복원해 낼 수 있음을 확인


> - 코드의 역할은
>   - **"스토리지(MinIO)의 무한한 저장 능력"**과 **"벡터 DB(Qdrant)의 초고속 의미 검색 능력"**을 AI 임베딩 모델로 융합하여,
>   - 언제든 원하는 지식을 실시간으로 꺼내 쓸 수 있도록 만든 **'AI 장기 기억 장치'의 가동 스위치**
{: .common-quote}



## 3. 실무 운영 시 핵심 고려사항 (Best Practices)

- **메타데이터 필터링 활용:**
    - Qdrant에 저장할 때 파일 유형(`file_type: "txt"`), 생성 부서, 혹은 보안 등급 등의 메타데이터를 함께 인덱싱하면,
    - 유저 권한이나 특정 버킷 내에서만 유사도 검색이 이루어지도록 제한할 수 있음

- **Presigned URL 발급 전략:**
    - 대규모 웹 애플리케이션 프론트엔드와 연동할 때, 프론트엔드가 MinIO에 직접 접근하게 하면 보안상 위험함
    - Qdrant 검색 결과로 나온 `minio_object_key`를 가지고 백엔드 아키텍처에서 MinIO의 `presigned_get_object()` 메서드를 호출하여 **만료 시간이 있는 임시 다운로드 URL**을 생성해 클라이언트에 반환하는 것이 정석

- **분할(Chunking) 전략과 Key 매핑:**
    - 하나의 큰 PDF/텍스트 파일이 여러 개(예: 100개)의 청크 벡터로 쪼개져 Qdrant에 들어갈 경우,
        - 100개의 벡터가 모두 동일한 하나의 `minio_object_key`를 바라보게 설계해야 함
    - `chunk_id`나 `page_number` 같은 추가 오프셋 정보를 Qdrant 페이로드에 남겨두면 원본의 정확한 위치를 추적하기 수월함