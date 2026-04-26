---
layout: page
title:  "FastAPI 기초: 비동기 처리(Async/Await)와 Pydantic 모델링"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-02_03-FastApiAsyncPydanticModeling
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 실습 주제: 고성능 AI 뉴스 분석기 (Async News Analyzer)

> - 긴 뉴스 기사를 받아 **비동기적으로 분석**
> - 결과의 **데이터 규격을 보장**
{: .common-quote}


### 통합 코드

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import asyncio
import time

app = FastAPI(title="AI 뉴스 분석 에이전트")

# 1. Pydantic 모델링: 데이터 입구와 출구 정의
class NewsRequest(BaseModel):
    title: str = Field(..., min_length=5, example="인공지능 시장의 미래 전망")
    content: str = Field(..., min_length=20, example="이곳에 아주 긴 뉴스 본문이 들어갑니다. 최소 20자 이상 작성해야 합니다.")
    tags: List[str] = Field(default=[], max_items=3)

    @validator("content")
    def check_forbidden_words(cls, v):
        if "스팸" in v:
            raise ValueError("부적절한 단어가 포함되어 있습니다.")
        return v

class NewsResponse(BaseModel):
    summary: str
    sentiment: str
    processing_time: float

# 2. 비동기 처리 로직: 가상의 AI 엔진 호출
async def fake_ai_analysis(text: str):
    # AI 모델이 텍스트를 분석하는 데 2초가 걸린다고 가정 (비동기 대기)
    await asyncio.sleep(2.0)
    sentiment = "긍정" if "미래" in text or "혁신" in text else "중립"
    return f"요약문: {text[:10]}...", sentiment

# 3. 엔드포인트 구성
@app.post("/analyze", response_model=NewsResponse)
async def analyze_news(request: NewsRequest):
    start_time = time.perf_counter()
    
    # 비동기 함수 호출 (이동안 서버는 다른 일을 할 수 있음)
    summary, sentiment = await fake_ai_analysis(request.content)
    
    end_time = time.perf_counter()
    
    return NewsResponse(
        summary=summary,
        sentiment=sentiment,
        processing_time=round(end_time - start_time, 2)
    )
```

1. **Pydantic 모델링: 데이터 입구와 출구 정의**
    - 데이터가 비즈니스 로직에 들어가기 전, 입구에서 컷트되는지 확인

    - **테스트 방법**
        - Swagger UI에서 `content`를 20자 미만으로 입력하거나, `tags`를 4개 이상 넣어보기

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-001.png">
        </div>

    - **확인할 결과**
        - **422 Unprocessable Entity** 발생
        - `loc`: `["body", "content"]`, `msg`: `ensure this value has at least 20 characters` 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-002.png">
        </div>

    - **핵심 원리**
        - 개발자가 `if len(content) < 20:` 같은 코드를 짤 필요 없이, 선언만으로 데이터가 정제됨을 이해하기


2. **비동기 처리 로직: 가상의 AI 엔진 호출**
    - 단순 타입 체크가 아닌, 개발자가 직접 만든 규칙이 작동하는지 확인

    - **테스트 방법**
        - `content` 내용 중에 **"스팸"**이라는 단어를 포함해서 전송

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-003.png">
        </div>

    - **확인할 결과**
        - **422 Error** 발생
        - `msg`: `Value error, 부적절한 단어가 포함되어 있습니다.` 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-004.png">
        </div>

    - **핵심 원리**
        - `@validator`를 통해 복잡한 사내 규정을 데이터 수신 단계에서 즉시 적용할 수 있음을 체감하기


3. **비동기(Async) 처리 테스트 (효율성 확인)**
    - 서버가 "기다림"을 어떻게 처리하는지 확인

    - **테스트 방법**
        - 브라우저 탭을 2개 열고, 거의 동시에 **[Execute]**를 클릭

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-005.png">
        </div>

    - **확인할 결과**
        - 첫 번째 요청이 끝날 때까지 두 번째 요청이 멈춰있지 않고, 두 요청 모두 약 2초 후에 거의 동시에 완료됨
        - `processing_time`이 약 2.0초 내외로 찍히는지 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-006.png"><br><br>
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_03-007.png">
        </div>

    - **핵심 원리**
        - `await`가 적힌 부분에서 서버는 작업을 예약만 해두고 다른 손님(요청)을 받으러 간다는 '논블로킹' 개념 이해하기


> - **Pydantic**은 데이터의 **'관상(Type/Structure)'**을 보고 무결성을 보장함
> - **Async/Await**는 서버가 노는 시간(I/O 대기) 없이 **'멀티태스킹'**을 하게 만듦
> - 이 두 가지가 합쳐져서 **안전하고 빠른** 현대적인 API 서비스가 완성됨
{: .summary-quote}
