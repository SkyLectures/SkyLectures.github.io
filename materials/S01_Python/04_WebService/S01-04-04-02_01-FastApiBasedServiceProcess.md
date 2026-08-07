---
layout: page
title:  "FastAPI 기초: 서비스 기본 흐름"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-02_01-FastApiBasedServiceProcess
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - FastAPI의 서비스 기본 흐름을 이해하는 것은 단순히 코드를 짜는 것을 넘어,<br>
> **현대적인 웹 아키텍처가 요청(Request)을 어떻게 처리하고 응답(Response)으로 변환하는지** 그 메커니즘을 파악하는 과정
> - **FastAPI만의 "이벤트 기반 비동기 워크플로우"**를 중심으로 학습할 것을 권장함
{: .common-quote}

## 1. 전체적인 생명주기 (Lifecycle)

> - 클라이언트의 요청이 들어와서 응답이 나가는 과정은 크게 5단계로 나뉨

1. **연결 및 라우팅 (ASGI & Starlette):**
    - 웹 서버(Uvicorn)가 요청을 받아 FastAPI 앱으로 전달
    - URL에 맞는 함수(Path Operation)를 검색
2. **데이터 추출 및 변환 (Pydantic):**
    - HTTP 요청 메시지(Header, Body, Query)에서 데이터를 뽑아 Python 객체로 변환
3. **유효성 검사 (Validation):**
    - 선언된 타입 힌트에 맞는지 검사
    - 실패 시 즉시 에러를 반환
4. **비즈니스 로직 실행 (Application Logic):**
    - 개발자가 작성한 `async def` 함수가 실행됨
    - DB 조회나 AI 모델 호출 등이 이 단계에서 일어남
5. **직렬화 및 응답 (Serialization):**
    - 결과물을 JSON 등으로 변환
    - 클라이언트에게 전송


## 2. 핵심 이론 구성 요소

- **ASGI (Asynchronous Server Gateway Interface)**
    - FastAPI는 기본적으로 **ASGI** 표준을 따름
    - **이론적 배경:**
        - 과거 WSGI(Python의 전통적인 서버 규격)는 한 번에 하나의 요청만 처리하는 동기식 구조
    - **FastAPI의 선택:**
        - ASGI는 한 개의 프로세스가 수만 개의 연결을 동시에 유지할 수 있는 **이벤트 루프(Event Loop)** 방식을 사용
        - 이는 특히 대기 시간이 긴 AI 추론이나 대량의 센서 데이터 수집에 최적화된 구조임

- **Starlette: 고성능 웹 엔진**
    - FastAPI의 밑바닥에서 네트워크 통신을 전담하는 '엔진'
    - **라우팅(Routing):**
        - `/users/1`과 같은 경로를 인식하여
        - 적절한 Python 함수에 매핑
    - **컨텍스트 관리:**
        - 요청에 대한 세션, 쿠키, 상태 정보를 관리
        - 비동기적으로 안전하게 데이터를 전달

- **Pydantic: 데이터의 관문 (Gatekeeper)**
    - FastAPI가 "현대적"이라고 불리는 가장 큰 이유
    - **타입 힌트 활용:**
        - Python의 표준 타입 힌트(`name: str`)를 읽어와서 런타임에 데이터 규격을 강제
    - **Parsing, not Validation:**
        - 단순히 데이터가 맞는지 틀린지만 보는 것이 아니라,
        - 들어온 원시 데이터(Raw data)를 Python이 다루기 쉬운 **'강력한 타입의 객체'**로 재구성


## 3. 단계별 학습

### 3.1 연결 및 라우팅 (ASGI & Starlette)

- 클라이언트의 HTTP 요청이 어떻게 파이썬 함수로 변환되는지 그 "길"을 찾는 과정
- Uvicorn(서버) - ASGI(규격) - Starlette(엔진) - FastAPI(프레임워크)의 관계를 명확히 이해하기

```python
#//file: "main.py"
from fastapi import FastAPI
import datetime

# [ASGI & Starlette 계층] FastAPI 인스턴스 생성
# 내부적으로 Starlette 프레임워크를 상속받아 ASGI 규격을 준수합니다.
app = FastAPI(title="FastAPI Routing 기초")

# 1. 정적 라우팅 (Static Routing)
@app.get("/", tags=["Basic"])
async def read_root():
    """가장 기본적인 루트 경로 라우팅"""
    return {"message": "웹 서버(Uvicorn)로부터 요청을 전달받았습니다."}

# 2. 동적 라우팅 (Dynamic Routing / Path Operation)
@app.get("/items/{item_id}", tags=["Advanced"])
async def read_item(item_id: int):
    """
    URL에 포함된 {item_id} 값을 해석하여 함수로 전달합니다.
    Uvicorn -> FastAPI -> URL 매칭 -> item_id 추출 -> 함수 실행 순서로 진행됩니다.
    """
    return {
        "item_id": item_id,
        "timestamp": datetime.datetime.now(),
        "info": f"URL 경로로부터 {item_id}번 아이템 요청을 인식했습니다."
    }
```

<div class="insert-image" style="text-align: center;">
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-006.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-007.png">
</div>

### 3.2 데이터 추출 및 변환 (Pydantic)

- 클라이언트가 보낸 원시적인 HTTP 요청(텍스트)을 파이썬이 다루기 쉬운 '강력한 타입의 객체'로 변환하는 과정
- FastAPI는 요청의 세 가지 핵심 위치(Header, Body, Query)에서 데이터를 동시에 추출할 수 있음
- 예제 코드는 HTTP 요청의 각기 다른 위치에서 데이터가 어떻게 뽑혀 나와 Pydantic 객체로 합쳐지는지 보여줌

```python
#//file: "main.py"
from fastapi import FastAPI, Header, Body, Query
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Pydantic 데이터 추출 실습")

# [4단계] 데이터 레이어: Body 데이터 규격 정의 (Pydantic)
class Item(BaseModel):
    name: str = Field(..., example="AI 스피커")
    price: float = Field(..., gt=0, example=150000.0)
    description: Optional[str] = None

@app.post("/items/{item_id}")
async def create_item(
    # 1. Path Parameter: URL 경로에서 추출
    item_id: int, 
    
    # 2. Body: HTTP 본문의 JSON을 Pydantic 객체로 변환
    item: Item, 
    
    # 3. Query Parameter: URL 뒤의 ?q=... 에서 추출 (기본값 설정 가능)
    q: Optional[str] = Query(None, max_length=50), 
    
    # 4. Header: HTTP 헤더에서 사용자 정의 토큰 등을 추출
    user_agent: Optional[str] = Header(None)
):
    """
    HTTP 요청의 다양한 위치(Header, Body, Query, Path)에서 
    데이터를 동시에 추출하여 파이썬 객체로 변환합니다.
    """
    return {
        "path": {"item_id": item_id},
        "body": item,  # Pydantic 객체는 자동으로 JSON 변환됨
        "query": {"q": q},
        "header": {"user_agent": user_agent},
        "message": f"'{item.name}' 데이터가 성공적으로 파싱 및 변환되었습니다."
    }
```

- **데이터가 변환되는 논리적 단계**
    1. 추출 (Extraction)
        - FastAPI가 들어온 HTTP 요청을 훑으며, 함수 인자에 선언된 위치(Header, Body, Query)에서 값을 찾음
    2. 파싱 (Parsing)
        - 텍스트로 들어온 값(예: "150000")을 코드에 선언된 타입(예: float)으로 변환
    3. 검증 (Validation)
        - Pydantic 모델에 정의된 규칙(gt=0, max_length=50 등)에 맞는지 검사
    4. 객체 생성
        - 모든 검증이 끝나면 비로소 우리가 사용할 수 있는 Pydantic 인스턴스(item)가 함수 내부에서 생성됨

- **확인 및 테스트 방법**
    1. 정상 요청 테스트 (Swagger UI)
        - http://127.0.0.1:8000/docs 접속
        - POST /items/{item_id} 클릭 후 [Try it out] 버튼 클릭
        - 각 필드에 값을 입력 후 실행
            - item_id: 10
            - Body: {"name": "노트북", "price": 1200000}
            - q: search_keyword
        - 결과: 입력한 값들이 각각의 위치에서 정확히 추출되어 JSON으로 반환되는지 확인

    2. 데이터 변환(Casting) 확인
        - price에 숫자가 아닌 문자열 "1200000"(따옴표 포함)을 보내도, 
        - FastAPI가 자동으로 float으로 변환하여 처리하는 것을 확인

    3. 검증 실패 테스트 (Validation Fail)
        - price에 -500을 입력하거나, item_id에 문자를 넣어봄
        - 결과
            - 422 Unprocessable Entity 에러가 발생
            - 어떤 위치의 어떤 데이터가 왜 틀렸는지 알려주는 에러 메시지 확인

<div class="insert-image" style="text-align: center;">
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-008.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-009.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-010.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-011.png">
</div>

> - "데이터는 단순히 전달되는 것이 아니라, 엄격한 관문(Pydantic)을 통과해야만 비즈니스 로직에 도달할 수 있음"을 확인할 것
{: .common-quote}

### 3.3 유효성 검사 (Validation)

- 단순히 타입이 맞는지(int인지 str인지)를 넘어, 
- **값의 범위나 문자열의 패턴**까지 검사하여 비즈니스 로직에 결함이 있는 데이터가 들어오는 것을 원천 봉쇄하는 예제
- Pydantic의 `Field`와 `validator`를 사용하여 실무에서 자주 쓰이는 검증 로직 구현

```python
#//file: "main.py"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List

app = FastAPI(title="FastAPI 유효성 검사 실습")

# [4단계] 데이터 레이어: 엄격한 검증 규칙이 적용된 모델
class UserCreate(BaseModel):
    # 1. Field를 이용한 기본 검증 (길이, 범위 제한)
    username: str = Field(..., min_length=3, max_length=20, description="아이디는 3~20자")
    age: int = Field(..., ge=19, le=120, description="19세 이상 성인만 가입 가능")
    
    # 2. 복잡한 리스트 데이터 검증
    interests: List[str] = Field(default=[], max_items=5, description="관심사는 최대 5개")

    # 3. validator를 이용한 커스텀 비즈니스 로직 검증
    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("아이디는 영문과 숫자만 포함해야 합니다.")
        return v

@app.post("/users/register")
async def register_user(user: UserCreate):
    """
    모든 유효성 검사를 통과해야만 이 함수 내부의 로직이 실행됩니다.
    검사 실패 시, FastAPI는 즉시 422 에러와 상세 원인을 반환합니다.
    """
    return {"message": f"{user.username}님의 가입 처리를 시작합니다.", "data": user}
```

- **유효성 검사의 '철학'**
    - **선언적 검증:**
        - 코드로 일일이 `if age < 19:`라고 적는 대신,
        - 타입 힌트와 `Field`를 통해 데이터의 성격을 선언

    - **즉시 실패 (Fail-Fast):**
        - 잘못된 데이터가 들어오면 무거운 DB 조회나 복잡한 계산을 시작하기도 전에 입구에서 바로 쫓아냄

    - **데이터 무결성:**
        - 함수 내부(`register_user`)로 들어온 `user` 객체는 이미 모든 검증을 마친 '깨끗한 데이터'임이 보장
            - 개발자는 안심하고 로직에만 집중할 수 있음

- **확인 및 테스트 방법 (에러 메시지 분석)**
    - 422 에러를 **의도적으로** 발생시켜 내용 분석

        - 타입은 맞지만 값이 틀린 경우
            - **입력:** `{"username": "dev", "age": 15, "interests": []}`
            - **결과:** **422 Error**
            - **메시지 분석:** `loc: ["body", "age"]`, `msg: "ensure this value is greater than or equal to 19"`
            - **교훈:** `int` 타입(15)은 맞지만, `ge=19`라는 값의 범위를 어겼음을 확인<br><br>

            <div class="insert-image" style="text-align: center;">
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-012.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-013.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-014.png">
            </div>

        - 커스텀 검증(validator)을 어긴 경우
            - **입력:** `{"username": "user_#1", "age": 25, "interests": []}`
            - **결과:** **422 Error**
            - **메시지 분석:** `loc: ["body", "username"]`, `msg: "아이디는 영문과 숫자만 포함해야 합니다."`
            - **교훈:** 우리가 `@validator`에 정의한 `ValueError` 메시지가 클라이언트에게 그대로 전달되는 것을 확인<br><br>

            <div class="insert-image" style="text-align: center;">
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-015.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-016.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-017.png">
            </div>


        - 정상 케이스
            - **입력:** `{"username": "pythonista", "age": 30, "interests": ["AI", "FastAPI"]}`
            - **결과:** **200 OK**<br><br>

            <div class="insert-image" style="text-align: center;">
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-018.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-019.png"><br><br>
                <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-020.png">
            </div>

> - 유효성 검사는 단순히 '맞다 틀리다'를 넘어, 서비스의 안정성을 지키는 가장 첫 번째 방어선
{: .common-quote}


## 4. 요청 처리의 기술적 디테일

- **경로 매개변수 vs 쿼리 매개변수 (Path vs Query)**
    - FastAPI는 URL 구조를 통해 데이터를 어떻게 다룰지 이론적으로 구분함
        - **경로 매개변수 (Path):**
            - 리소스의 **고유 식별자**를 나타냄 (예: `/books/10` -> 10번 책)
            - 시스템에서 반드시 존재해야 하는 필수 정보로 취급
        - **쿼리 매개변수 (Query):**
            - 리소스의 **상태나 정렬, 필터링**을 나타냄 (예: `/books?sort=popular` -> 인기순 정렬)
            - 기본값을 가질 수 있으며, 선택적인 정보로 취급

- **의존성 주입 (Dependency Injection - DI)**
    - FastAPI는 함수형 프로그래밍의 이점을 살린 고유한 DI 시스템을 가집니다.
        - **이론:**
            - 공유 로직(인증, DB 연결 등)을 별도의 함수로 분리
            - 필요한 엔드포인트에서 `Depends()`를 통해 주입
        - **이점:**
            - 코드의 중복 제거
            - 테스트 시 가짜(Mock) 객체를 갈아 끼우기 매우 용이한 구조를 만듦

## 5. 요약: 왜 이 흐름이 중요한가?

> - 기본 흐름을 이해한다는 것은 **"내 코드가 어디서 멈출 수 있고, 어디서 병목이 생기는지"**를 아는 것
{: .common-quote}

- **Pydantic 단계:**
    - 데이터 형식이 틀리면 내 비즈니스 로직은 실행조차 되지 않으므로 안전함
- **Async/Await 단계:**
    - `await` 키워드를 만나는 순간, 서버는 놀지 않고 다른 사람의 요청을 처리하러 떠남
    - 이 덕분에 고성능이 보장됨

> - **"FastAPI는 단순히 웹 프레임워크가 아니라, 데이터의 무결성을 보장하고 비동기 효율을 극대화하는 지능형 게이트웨이"**
{: .summary-quote}


## 6. 예제 코드

1. **클라이언트 요청 (Client Request)**
    - 사용자가 데이터를 보내는 행위 자체를 정의
        - 주로 Swagger UI나 외부 도구로 대체됨
        - 예제에서는 구조를 이해하기 위해 정의함

    ```python
    # 클라이언트가 전송할 JSON 데이터 예시
    {
        "book_id": 101,
        "user_id": "seokhwan_yang"
    }
    ```

2. **웹 레이어: 라우팅 요청 (Routing)**
    - FastAPI 인스턴스를 생성
    - 특정 URL 경로를 함수와 연결

    ```python
    from fastapi import FastAPI

    app = FastAPI()

    # POST 요청을 /books/loan 경로로 연결(라우팅)
    @app.post("/books/loan")
    async def route_loan_request():
        pass # 다음 단계에서 구현
    ```

3. **웹 레이어: 비동기 I/O 처리 (Async I/O)**
    - 함수 선언 시 `async` 키워드를 사용하여 논블로킹(Non-blocking) 구조 생성

    ```python
    # 비동기 함수 선언을 통해 시스템 자원 효율화
    async def handle_loan_async():
        # 여기서 await를 사용하여 I/O 병목을 방지함
        pass
    ```

4. **데이터 레이어: 입력 데이터 파싱 및 검증 (Pydantic Validation)**
    - 들어온 데이터를 Pydantic 모델로 변환
    - 규칙 검사

    ```python
    from pydantic import BaseModel, Field

    class LoanRequest(BaseModel):
        # 타입 힌트를 통한 파싱 및 검증
        book_id: int = Field(..., gt=0, description="도서 ID는 0보다 커야 함")
        user_id: str = Field(..., min_length=3, description="사용자 ID는 3자 이상")
    ```

5. **비즈니스 로직: DB 상호작용 (Database Interaction)**
    - 데이터베이스에 접근하여 데이터를 조회하거나 저장하는 시뮬레이션

    ```python
    import asyncio

    async def get_db_data(book_id: int):
        # DB 조회 시간 시뮬레이션 (비동기 대기)
        await asyncio.sleep(0.1)
        return {"id": book_id, "title": "FastAPI Master", "stock": 5}
    ```

6. **비즈니스 로직: 비즈니스 규칙 적용 (Business Rules)**
    - 데이터가 비즈니스 정책에 맞는지 검사

    ```python
    def check_loan_policy(stock_count: int):
        # 재고가 없으면 에러 발생 (비즈니스 규칙)
        if stock_count < 1:
            return False
        return True
    ```

7. **비즈니스 로직: 외부 API/서비스 연동 (External Integration)**
    - 외부 알림 서비스나 인증 서비스와 통신

    ```python
    import asyncio

    async def call_external_notification(user_id: str):
        # 외부 API 호출 시뮬레이션
        await asyncio.sleep(0.2)
        print(f"Notification sent to {user_id}")
    ```

8. **자동 문서화 시스템: ReDoc (Documentation)**
    - 코드를 작성하면 자동으로 생성되는 API 문서 확인

    ```python
    # 별도 코드 없이 서버 실행 후 브라우저 접속
    # URL: http://127.0.0.1:8000/redoc
    ```

9. **데이터 레이어: 타입-세이프 결과 생성 (Serialization Model)**
    - 응답으로 내보낼 데이터를 안전한 규격으로 재구성

    ```python
    from pydantic import BaseModel
    from datetime import datetime

    class LoanResponse(BaseModel):
        # 클라이언트에 보낼 데이터 필드 정의
        loan_id: int
        status: str
        processed_at: datetime
    ```

10. **데이터 레이어: HTTP 응답 처리 (Response Formation)**
    - FastAPI가 Pydantic 객체를 JSON으로 변환하여 응답 메시지를 구성

    ```python
    # 응답 모델을 엔드포인트에 설정
    @app.post("/books/loan", response_model=LoanResponse)
    async def finalize_response():
        # 이 함수가 반환하는 데이터는 자동으로 LoanResponse 규격에 맞춰짐
        pass
    ```

11. **클라이언트로 HTTP 응답 (Client Delivery)**
    - 최종적으로 클라이언트가 받게 될 결과물

    ```python
    # 최종 HTTP Response Body
    {
        "loan_id": 12345,
        "status": "Success",
        "processed_at": "2026-04-26T12:00:00"
    }
    ```

## 7. 통합 예제

- 앞서 학습한 11단계의 모든 개념이 유기적으로 결합된 **완성형 서버 코드**
- 실제 파일 하나에 복사하여 즉시 실행 가능함
- 각 부분이 어느 단계에 해당하는지 주석을 통해 확인하실 수 있음

### 7.1 예제 코드

```python
#//file: "main.py"

# [필요 패키지 임포트]
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from datetime import datetime
import asyncio

# [8단계 준비] FastAPI 인스턴스 생성 및 자동 문서화 준비
app = FastAPI(title="통합 도서 대출 관리 시스템")

# ---------------------------------------------------------
# [4단계 & 9단계] 데이터 레이어: 입력(Request) 및 출력(Response) 모델 정의
# ---------------------------------------------------------

# [4단계] 입력 데이터 파싱 및 검증용 모델
class LoanRequest(BaseModel):
    book_id: int = Field(..., gt=0, description="도서 번호 (양수)")
    user_id: str = Field(..., min_length=3, description="사용자 아이디 (3자 이상)")

    @validator("user_id")
    def check_black_list(cls, v):
        if "black" in v:
            raise ValueError("대출 제한 유저입니다.")
        return v

# [9단계] 결과 데이터 생성용 모델 (Type-Safe)
class LoanResponse(BaseModel):
    loan_id: int
    book_id: int
    status: str
    processed_at: datetime

# ---------------------------------------------------------
# [5, 6, 7단계] 비즈니스 로직 레이어
# ---------------------------------------------------------

async def process_library_logic(request: LoanRequest):
    # [5단계] DB 상호작용 시뮬레이션
    await asyncio.sleep(0.3)
    db_book_stock = 5 # 가상의 재고 데이터
    
    # [6단계] 비즈니스 규칙 적용
    if db_book_stock < 1:
        raise HTTPException(status_code=400, detail="현재 재고가 없는 도서입니다.")
    
    # [7단계] 외부 API 연동 시뮬레이션 (알림 서비스 등)
    await asyncio.sleep(0.2)
    print(f"알림: {request.user_id}님에게 대출 처리 메시지 발송")

    # [9단계 반영] 내부 처리 결과를 규격에 맞는 객체로 생성
    return LoanResponse(
        loan_id=20260426,
        book_id=request.book_id,
        status="정상 대출 예약됨",
        processed_at=datetime.now()
    )

# ---------------------------------------------------------
# [2, 3, 10, 11단계] 웹 레이어 및 진입점
# ---------------------------------------------------------

# [2단계] 라우팅 설정
# [10단계] HTTP 응답 처리 설정 (response_model)
@app.post("/books/loan", response_model=LoanResponse, tags=["Library Operation"])
async def loan_book_api(request: LoanRequest): # [1, 3, 4단계 작동]
    """
    도서 대출을 처리하는 통합 엔드포인트입니다.
    """
    # [5, 6, 7, 9단계 실행 후 결과 수신]
    result = await process_library_logic(request)
    
    # [11단계] 최종 클라이언트 응답 전송
    return result
```

### 7.2 결과 확인 방법

1. **서버 실행**
    - 터미널(또는 CMD)에서 다음 명령어를 입력

    ```bash
    uvicorn main:app --reload
    ```

2. **Swagger UI를 통한 데이터 전송 (1~4단계 확인)**
    1. 브라우저에서 `http://127.0.0.1:8000/docs`에 접속
    2. `POST /books/loan` 항목을 클릭 🡲 **[Try it out]**을 클릭
    3. 아래 JSON 데이터를 입력한 후 🡲 **[Execute]**를 클릭

    ```json
    {
        "book_id": 101,
        "user_id": "seokhwan"
    }
    ```
    4. **결과:**
        - 하단의 `Responses` 섹션에서 200 성공 코드와 함께
        - 9단계에서 정의한 `LoanResponse` 형태의 결과를 확인

3. **유효성 검사 실패 확인 (Pydantic 검증 확인)**
    1. `user_id`를 `"bk"`(3자 미만)로 바꾸거나, `"black_user"`(validator 차단)로 바꿔서 다시 보냄
    2. **결과:** 
        - `422 Unprocessable Entity` 혹은 `400 Bad Request` 에러가 발생
        - 우리가 설정한 에러 메시지가 출력되는지 확인

4. **ReDoc 확인 (8단계 확인)**
    - 브라우저에서 `http://127.0.0.1:8000/redoc`에 접속
    - Swagger와는 또 다른, 깔끔하게 정리된 기업용 문서 형태를 확인
    - 본인이 쓴 주석(`description`)이 어디에 표시되는지 확인

<div class="insert-image" style="text-align: center;">
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-021.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-022.png"><br><br>
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-023.png">
</div>


> - 작성된 코드는 단순한 코드가 아니라 **'살아있는 문서'**이자 **'철저한 감시자'**
> - 타입을 정의했을 뿐인데 데이터 검증이 끝났고, 함수를 만들었을 뿐인데 웹 페이지 문서가 생성됨
> - 이것이 바로 FastAPI가 현대 백엔드 시장에서 가장 사랑받는 이유
{: .summary-quote}
