---
layout: page
title:  "LLM Streaming API 구현 (실시간 답변 출력)"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-03_01-FastApiLlmStreamingApi
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 실습 주제: "실시간 AI 에이전트 스트리밍 엔진"

- 사용자가 질문을 던졌을 때, LLM이 전체 답변을 완성할 때까지 10~20초를 기다리게 하는 것이 아니라, 
- **생성되는 즉시 한 글자씩 브라우저에 뿌려주는 방식**
- 이는 `EventSource` 프로토콜(Server-Sent Events, SSE)을 통해 이루어짐

### 1. LLM 시뮬레이션 기반

#### 1.1 통합 풀 소스코드 (`main.py`)

- 이 코드는 실제 LLM(OpenAI 등)이 없어도 작동 원리를 이해할 수 있도록 가상의 스트리밍 생성기(Generator)를 포함함

```python
#//file: "main.py"
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Streaming API 실습")

# [1] 데이터 모델: 스트리밍할 질문 정의
class ChatRequest(BaseModel):
    prompt: str = Field(..., example="인공지능의 미래에 대해 설명해줘.")

# [2] 가상 LLM 엔진: 데이터를 조각(Chunk) 단위로 생성
async def fake_llm_streamer(prompt: str):
    """
    LLM이 문장을 생성하는 과정을 시뮬레이션함
    한 글자씩 생성될 때마다 yield를 통해 즉시 밖으로 보냄
    """
    full_text = f"요청하신 '{prompt}'에 대한 분석 결과입니다. " \
                f"인공지능은 단순히 도구를 넘어 인간의 지성을 확장하는 파트너가 될 것입니다..."
    
    for char in full_text:
        yield f"data: {char}\n\n"  # SSE(Server-Sent Events) 규격 준수
        await asyncio.sleep(0.05)   # 글자 사이의 지연 시간 시뮬레이션

# [3] 스트리밍 엔드포인트
@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    StreamingResponse를 사용하여 클라이언트와의 연결을 유지한 채
    데이터를 한 조각씩(Chunk) 계속해서 전송함
    """
    # 생성기(Generator)를 전달하여 응답 시작
    return StreamingResponse(
        fake_llm_streamer(request.prompt), 
        media_type="text/event-stream"
    )
```


### 1.2 단계별 이론 및 프로세스 설명

- **Step 1: `yield` 키워드와 Generator**
    - 일반적인 함수는 `return`을 만나면 값을 반환하고 종료됨
    - 하지만 `yield`를 사용하면 함수가 실행 상태를 유지한 채 값을 밖으로 던지고 다음 호출을 기다림
    - 이를 통해 **메모리를 아끼면서 대량의 데이터를 조각내어 보낼 수 있음**

- **Step 2: `StreamingResponse`**
    - FastAPI(Starlette)가 제공하는 특수 응답 클래스
    - 일반 응답은 모든 연산이 끝나야 브라우저로 전송되지만, 
    - `StreamingResponse`는 **데이터 조각이 생성될 때마다 즉시 HTTP 패킷을 전송**함

- **Step 3: SSE (Server-Sent Events) 규격**
    - 스트리밍 시 브라우저가 데이터를 인식하게 하려면 특정 형식이 필요함
        - **형식:** `data: [내용]\n\n`
        - **Media Type:** `text/event-stream`


### 1.3 확인 및 테스트 방법

1. Swagger UI를 통한 확인 (가장 간단한 방법)
    1. `http://127.0.0.1:8000/docs` 접속
    2. `POST /v1/chat/stream` 엔드포인트 실행
    3. **확인 포인트**
        - 결과 창에서 한 번에 짠! 하고 나타나는 것이 아니라,
        - **글자가 한 글자씩 추가되며 결과가 길어지는 과정**을 눈으로 확인

2. `curl` 명령어를 통한 확인 (실무적 방법)

    ```bash
    curl -X 'POST' \
    'http://127.0.0.1:8000/v1/chat/stream' \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "안녕?"}'
    ```

3. **결과:**
    - 터미널 창에 글자가 타다닥! 하고 실시간으로 찍히는 것을 확인함


### 2. Ollama 기반의 로컬 LLM을 이용한 실습

> - **데이터 프라이버시(Local Execution)**를 보장하면서도, 
> - 현대적인 **비동기 제너레이터(Async Generator)**의 정수를 맛볼 수 있음
> - `httpx` 라이브러리를 사용하여 Ollama의 API와 비동기적으로 통신하는 구조
{: .common-quote}

#### 2.1 실습 전 준비사항

- 터미널에서 Ollama와 통신하기 위한 비동기 HTTP 클라이언트를 설치해야 함

```bash
pip install httpx
```

-(Ollama가 로컬에 설치되어 있고 `llama3` 또는 `qwen2` 등의 모델이 실행 가능한 상태여야 함)

#### 2.2 Ollama 기반 로컬 LLM 스트리밍 코드 (`main.py`)

```python
import httpx
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Ollama 로컬 LLM 스트리밍 서버")

class ChatRequest(BaseModel):
    prompt: str = Field(..., example="인공지능의 미래에 대해 한 문장으로 말해줘.")
    model: str = Field(default="llama3", example="llama3")

# [핵심 로직] Ollama API와 비동기 스트리밍 통신
async def ollama_streamer(prompt: str, model: str):
    """
    로컬 Ollama 서버에 요청을 보내고, 
    응답이 오는 대로 한 조각씩 클라이언트에게 전달함
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True  # Ollama에게 스트리밍 모드 요청
    }

    # timeout을 None으로 설정하여 LLM 생성 시간이 길어져도 연결이 끊기지 않게 함
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    # Ollama 응답 파싱 (JSON 조각)
                    chunk = json.loads(line)
                    response_text = chunk.get("response", "")
                    
                    # SSE 규격에 맞춰 데이터 전송
                    yield f"data: {response_text}\n\n"
                    
                    # 분석 완료 여부 확인
                    if chunk.get("done"):
                        break

@app.post("/v1/chat/local-stream")
async def local_chat_stream(request: ChatRequest):
    """
    사용자의 요청을 받아 로컬 Ollama 엔진의 응답을 스트리밍
    """
    return StreamingResponse(
        ollama_streamer(request.prompt, request.model),
        media_type="text/event-stream"
    )
```


#### 2.3 단계별 작동 프로세스 설명

1. `httpx.stream` (비동기 스트림 연결)
    - 일반적인 `client.post`는 모든 응답을 받을 때까지 기다리지만,
    - `client.stream`은 서버(Ollama)가 데이터를 보내기 시작하는 즉시 통로를 Open

2. `aiter_lines()` (비동기 라인 읽기)
    - Ollama는 스트리밍 시 각 토큰 정보를 JSON 한 줄씩 전송
    - `aiter_lines()`는 이 줄바꿈을 감지하여 한 줄이 완성될 때마다 코드를 활성화

3. 데이터 파싱 및 재포장
    - 로컬 엔진(Ollama)이 주는 데이터 포맷을 우리 서비스의 규격(SSE)으로 변환하여
    - 브라우저에 던져주는 **'중계기(Relay)'** 역할을 수행


#### 2.4 확인 및 테스트 방법

1. Ollama 서버 확인
    - 먼저 로컬에서 Ollama가 돌아가고 있는지 확인
    ```bash
    ollama serve
    ```

2. API 테스트 (Swagger UI)
    1. `http://127.0.0.1:8000/docs` 접속
    2. `POST /v1/chat/local-stream` 실행
    3. **확인 포인트**
        - 로컬 GPU(RTX 4080)가 작동하며 문장이 실시간으로 생성되는지 확인
        - 시뮬레이션 때보다 훨씬 역동적인 속도를 체감할 수 있음

3. 에러 대응 실습 (수강생 가이드)
    - **상황:** Ollama가 꺼져 있을 때 요청 보내기
    - **결과:** `ConnectError`가 발생
    - **교훈:** "API 서버는 혼자 존재할 수 없으며, 백엔드 엔진(Ollama)과의 연결 상태를 항상 체크해야 한다"


> - `httpx`를 이용한 비동기 외부 통신은 현대 백엔드 개발자에게 필수적인 스킬
{: .expert-quote}