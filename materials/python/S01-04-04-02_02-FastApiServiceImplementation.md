---
layout: page
title:  "FastAPI 기초: 서비스 구현"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-02_02-FastApiServiceImplementation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 실습 주제: "사내 AI 도서 대출 예약 엔진 구축"

### 1. 실습 프로세스 개요

- 본 실습은 클라이언트의 요청부터 최종 응답까지의 11단계를 총 5개의 핵심 모듈로 그룹화하여 진행

1. **모듈 A (준비):** API 라우팅 및 비동기 환경 설정 (1~3단계)
2. **모듈 B (검증):** Pydantic을 이용한 데이터 게이트웨이 구축 (4단계)
3. **모듈 C (로직):** 비즈니스 규칙 및 외부 연동 처리 (5~7단계)
4. **모듈 D (출력):** 타입-세이프 응답 생성 및 문서화 (8~11단계)
5. **모듈 E (통합):** 전체 프로세스 연결 및 최종 테스트


### 2. 각 단계별 이론 및 예제 코드

#### 2.1 모듈 A: 라우팅 및 비동기 환경 (1~3단계)

- **이론**
    - FastAPI는 `Starlette`을 통해 특정 URL 경로와 Python 함수를 연결(Routing)함
    - `async` 키워드는 CPU가 I/O 작업(DB, 네트워크)을 기다리는 동안 다른 요청을 처리할 수 있게 하는 '논블로킹'의 핵심

- **예제 코드**

    ```python
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/books/loan")
    async def loan_endpoint():
        return {"message": "Request Received"}
    ```

- **테스트**
    - 서버 실행 후 `http://127.0.0.1:8000/docs`에서 해당 엔드포인트가 리스트에 뜨는지 확인

#### 2.2 모듈 B: 데이터 레이어 검증 (4단계)

- **이론**
    - `Pydantic`은 들어온 JSON 데이터를 파싱하여 Python 객체로 만듦
    - 이때 타입 힌트와 `Field` 제약 조건을 통해 데이터의 무결성을 런타임에 보장함

- **예제 코드**

    ```python
    from pydantic import BaseModel, Field

    class LoanRequest(BaseModel):
        book_id: int = Field(..., gt=0, description="도서 번호는 1 이상")
        user_id: str = Field(..., min_length=3, description="아이디는 3자 이상")
    ```

- **테스트**
    - Swagger UI에서 `book_id`에 0이나 음수를 넣었을 때 `422 Unprocessable Entity` 에러가 발생하는지 확인

#### 2.3 모듈 C: 비즈니스 로직 및 외부 연동 (5~7단계)

- **이론**
    - 실제 데이터베이스 조회(5단계), 사내 대출 규정 적용(6단계), 알림톡 발송(7단계)과 같은 핵심 업무 로직이 수행되는 구간

- **예제 코드**

    ```python
    import asyncio

    async def run_business_logic(book_id: int):
        await asyncio.sleep(0.5) # DB 조회 및 외부 API 연동 시뮬레이션
        if book_id > 1000: # 비즈니스 규칙: 1000번 이상 도서는 금서
            return False
        return True
    ```

- **테스트**
    - 내부 로직 함수가 정상적으로 `True/False`를 반환하는지 별도 유닛 테스트 수행

#### 2.4 모듈 D: 응답 생성 및 문서화 (8~11단계)
- **이론**
    - 클라이언트에 보낼 데이터를 규격화(9단계)
    - 이를 HTTP 응답 메시지로 구성(10단계)
    - 이 과정에서 `ReDoc`(8단계)은 자동으로 업데이트됨

- **예제 코드**

    ```python
    from datetime import datetime

    class LoanResponse(BaseModel):
        status: str
        processed_at: datetime
    ```

- **테스트**
    - 브라우저에서 `http://127.0.0.1:8000/redoc`에 접속하여 API 명세서가 예쁘게 정리되었는지 확인

---

### 3. 최종 통합 예제 코드 (`main.py`)

```python
#//file: "main.py"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

app = FastAPI(title="AI 도서 대출 시스템")

# [4, 9단계] 데이터 모델 정의
class LoanRequest(BaseModel):
    book_id: int = Field(..., gt=0)
    user_id: str = Field(..., min_length=3)

class LoanResponse(BaseModel):
    loan_id: int
    book_id: int
    status: str
    processed_at: datetime

# [5, 6, 7단계] 비즈니스 로직 처리 함수
async def execute_loan_process(request: LoanRequest):
    await asyncio.sleep(0.3) # 비동기 I/O 처리 (3단계)
    
    # 비즈니스 규칙 (6단계)
    if request.book_id == 404:
        raise HTTPException(status_code=404, detail="도서를 찾을 수 없습니다.")
        
    return LoanResponse(
        loan_id=20260426,
        book_id=request.book_id,
        status="대출 예약 완료",
        processed_at=datetime.now()
    )

# [1, 2, 10, 11단계] 웹 레이어 진입점 및 응답 처리
@app.post("/books/loan", response_model=LoanResponse)
async def create_loan(request: LoanRequest):
    """
    도서 대출 프로세스 통합 실행
    (입력 검증 -> 로직 처리 -> 타입 세이프 응답)
    """
    return await execute_loan_process(request)
```

### 4. 최종 확인 및 테스트 방법 (수강생 가이드)

- 자신의 서버가 완벽하게 동작하는지 다음 순서로 확인

    1. **서버 가동**
        - 터미널에서 `uvicorn main:app --reload` 실행

    2. **정상 시나리오**
        - Swagger 접속 -> `book_id: 10`, `user_id: "yang"` 입력
        * **결과:** `200 OK` 응답과 함께 오늘 날짜와 대출 완료 메시지 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-001.png"><br><br>
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-002.png">
        </div>

    3. **데이터 검증(Pydantic) 테스트**
        * `user_id: "ya"` (2글자) 입력
        * **결과:** FastAPI가 즉시 `detail` 메시지와 함께 에러를 던지는지 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-003.png"><br><br>
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-004.png">
        </div>

    4. **비즈니스 규칙 테스트**
       * `book_id: 404` 입력.
       * **결과:** 우리가 `HTTPException`으로 설정한 404 에러가 명확하게 전달되는지 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-005.png"><br><br>
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-006.png">
        </div>

    5. **문서화 시스템 감상**
       * ReDoc(/redoc)에 들어가서 본인이 쓴 주석이 실시간으로 반영된 '기업급 API 문서'를 확인

        <div class="insert-image" style="text-align: center;">
            <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-02_02-007.png">
        </div>