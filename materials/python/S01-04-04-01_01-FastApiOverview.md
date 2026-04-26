---
layout: page
title:  "FastAPI 웹 프레임워크 개요"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-01_01-FastApiOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. FastAPI Web Framework란?

- 현재 Python 생태계에서 가장 빠르게 성장하는 프레임워크
    - **AI/ML 모델 서빙**: 비동기 처리가 필수적인 LLM(Large Language Model) 스트리밍 응답이나 실시간 추론 API에 최적
    - **Microservices**: 가볍고 빠르며 Docker 환경에서 컨테이너화하기 매우 효율적

<div class="insert-image" style="text-align: center;">
    <img style="width: 950px;" src="/materials/python/images/S01-04-04-01_01-001.png">
</div>

### 1.1 기술적 정의

- Python 3.8+의 **표준 타입 힌트**를 기반으로 API를 구축하기 위한 현대적이고 빠른(고성능) 웹 프레임워크
    - 표준 타입 힌트(Standard Python Type Hints)란?
        - 변수와 함수의 매개변수, 반환값에 데이터 타입을 명시하는 규격
        - 파이썬 언어 자체에서 제공하는 name: str과 같은 문법을 그대로 사용하여 API의 입출력 규격을 정의하는 방식을 의미
        - Python 3.5부터 도입되어 3.8 이상에서 정착됨
- Python의 생산성과 C++/Java 수준의 엄밀함 및 성능을 결합하려는 현대적 시도의 일환

- **성능(Performance)**
    - Node.JS 및 Go와 대등한 수준의 성능을 제공 🡲 Python 프레임워크 중 가장 빠른 것 중 하나로 간주됨
    - **Asynchronous (async/await) 기반의 논블로킹 I/O 처리** 방식이 빠른 속도의 이유
        - FastAPI는 내부적으로 uvicorn과 같은 고성능 ASGI 서버 위에서 동작하기 때문

- **표준 기반**
    - API의 산업 표준인 OpenAPI(이전의 Swagger)와 JSON Schema를 100% 계승하고 준수함
    - **데이터 구조 정의(Pydantic)가 곧 JSON Schema 규격이 됨**
        - 프론트엔드나 타 시스템과의 규격 공유 시 핵심 이점으로 작용

### 1.2 구조적 정의

> - FastAPI는 바닥부터 모든 것을 새로 만들기보다, 이미 **검증된 강력한 기술들을 통합**하여 개발됨
{: .common-quote}

#### 1.2.1 Starlette (Web Layer)

- **FastAPI와 Starlette의 관계: "추상화 레이어"**
    - FastAPI는 Starlette을 상속(Inherit)받아 구축됨 🡲 FastAPI의 모든 웹 관련 기능(라우팅, 세션, 쿠키, 상태 코드 등)은 사실 Starlette의 기능임
    - 비중: FastAPI 소스 코드의 상당 부분은 Starlette의 기능을 호출하고, 그 위에 Pydantic을 통한 데이터 검증과 자동 문서화(OpenAPI) 레이어를 얹은 형태
        - 비유: Starlette이 자동차의 엔진, 변속기, 차체(기본 구동계)라면, FastAPI는 그 위에 첨단 ADAS(자율주행 보조), 대시보드 디스플레이, 고급 인테리어를 추가한 완숙된 자동차와 같음
    - 철저한 분업화의 관계
        - Starlette: "웹 프로토콜(HTTP/WebSocket)을 어떻게 가장 빠르고 안정적으로 처리할 것인가?"에 집중
        - FastAPI: "사용자가 어떻게 하면 파이썬 타입 힌트만으로 쉽고 정확하게 API를 설계하고 문서화할 것인가?"에 집중

- **FastAPI에서 Starlette의 역할**
    - 보이지 않는 곳에서 모든 실무적인 일을 처리하는 핵심 엔진
    - 비동기(Asynchronous) 웹 서비스를 구축하기 위한 경량 ASGI 프레임워크
    - 고성능 라우팅과 웹 서버 기능을 제공

- **Starlette의 핵심 특징과 중요성**
    - **순수 비동기(Native Async) 설계**
        - 처음부터 ASGI(Asynchronous Server Gateway Interface) 규격을 준수하여 설계됨
        - 모든 요청을 비동기로 처리할 수 있는 구조적 토대 제공
        - LLM 스트리밍이나 실시간 센서 데이터 처리 시 대기 시간(I/O Bound) 최소화 가능
        - 동시 접속자의 효율적 처리 가능

    - **초경량 및 고성능 (Lightweight & High Performance)**
        - 웹 프레임워크가 가져야 할 최소한의 기능만 유지
        - 극도로 최적화됨
        - 라우팅 성능: 복잡한 URL 패턴을 매칭하는 속도가 Python 생태계 내에서 최상위권
        - 저수준 제어: WebSockets, 인증(Authentication), CORS(Cross-Origin Resource Sharing) 등을 직접 처리할 수 있는 강력한 툴킷 제공

    - **견고한 웹 표준 구현**
        - HTTP 스펙을 매우 엄격하고 정교하게 구현
        - 세션 관리, 쿠키 처리, GZip 압축, 정적 파일 서빙 등 실제 웹 서비스 운영에 필요한 '기본기'가 매우 탄탄함
            - FastAPI는 이 검증된 기본기를 그대로 가져다 쓰기 때문에 안정성이 높음

    - **FastAPI에서 Starlette이 담당하는 구체적 영역**
        - FastAPI 코드 내에서 우리가 사용하는 다음 기능들은 실제로는 Starlette이 수행하는 것
            - Request & Response 객체: 클라이언트로부터 들어오는 모든 HTTP 요청 정보와 응답 처리 관리
            - Background Tasks: API 응답을 보낸 후 백엔드에서 별도로 돌아가는 작업(예: 로그 기록, 이메일 발송) 관리
            - WebSocket 지원: 양방향 실시간 통신을 위한 기반 기술 제공
            - Exception Handling: HTTP 예외 상황을 처리하고 응답을 구성하는 기본 로직 담당

#### 1.2.2 Pydantic (Data Layer)

> - Python의 Type Hint를 사용하여 데이터 검증(Validation)과 직렬화(Serialization)를 수행
> - 런타임에서 데이터 타입을 강제하여 오류를 사전에 방지
>   - 클래스 기반으로 데이터 구조를 정의하면, 런타임에 타입을 강제하고 변환해 줌
{: .common-quote}

- **Pydantic의 철학**
    - **"Parsing, not Validation" 🡲 검증이 아니라 파싱(구문분석)**
        - 단순히 데이터를 검증(Validation)하는 것을 넘어, 입력 데이터를 선언된 타입으로 변환(Parsing)하는 것에 목적이 있음
            - 예시: "123"이라는 문자열이 들어왔을 때 int 타입 힌트가 있다면 이를 숫자 123으로 자동 변환해 줌
        - 비유
            - Validation: 검문소에서 신분증이 맞는지 확인만 하고 돌려보내는 것
            - Parsing: 입국자의 옷을 그 나라의 규격에 맞게 갈아입혀서 들여보내는 것

    - **Validation (검증)의 관점**
        - 기본적인 검증 시스템은 데이터가 사전에 정의된 규칙에 부합하는지 여부만 판단
            - "이 데이터는 int인가?" 🡲 No 🡲 "에러 발생(422 Unprocessable Entity)"
        - 데이터의 형태가 조금이라도 다르면 가차 없이 거절

    - **Parsing (파싱)의 관점**
        - Pydantic이 지향하는 파싱은 데이터를 목표한 타입(Shape)으로 변환하려는 시도를 포함함
        - 유연한 수용: 입력값이 "100"(문자열)이라도, 목표 타입이 int라면 이를 숫자 100으로 변환하여 받아들임
        - 결과물의 보장: 프로세스가 끝난 시점에 데이터는 단순한 '값'이 아니라, 해당 타입의 특성과 기능을 모두 갖춘 객체가 됨

    - **왜 이 차이가 중요한가? (실무적 이점)**
        - 데이터 정제 비용의 감소
            - 스마트팩토리 센서나 외부 API로부터 데이터를 받을 때, 숫자가 문자열 형태로 유입되는 경우가 흔함
            - Parsing 방식을 사용하면 개발자가 일일이 int()나 float()로 형변환을 할 필요가 없음
            - Pydantic 모델을 통과하는 순간 이미 완벽한 타입의 데이터가 되어 있기 때문

        - 런타임 예측 가능성
            - 단순히 검증만 하는 시스템은 검증 이후에도 데이터가 어떤 형태일지 확신하기 어려움
            - Parsing 시스템은 "이 모델을 통과했다면, 무조건 이 타입의 객체임이 보장된다"는 신뢰를 부여
            - 시스템의 엄밀성과 직결됨

        - 복잡한 객체로의 확장
            - 단순 자료형뿐만 아니라, JSON 데이터를 복잡한 Python Class 객체로 즉시 변환해줌
                - Input: {"created_at": "2026-04-25T12:00:00"} (String)
                - Pydantic Parsing 결과: datetime 객체로 변환되어 즉시 날짜 연산 가능

    > - 검증(Validation)은 데이터가 틀렸음을 지적하는 데 집중하지만,
    > - 파싱(Parsing)은 데이터를 우리가 원하는 안전한 형태(Type-safe object)로 재구성하는 데 집중함<br><br>
    > - **FastAPI가 Pydantic을 채택한 이유**는, 
    >   - 개발자가 비즈니스 로직에만 집중할 수 있도록 입력 데이터의 불확실성을 완전히 제거한 깨끗한 객체를 배달해주기 위해서임
    {: .common-quote}

### 1.3 실무적 정의
- FastAPI는 다음과 같은 역할을 수행하는 엔진

- **개발 생산성 극대화**
    - 타입 힌트를 통한 IDE의 자동 완성(Auto-complete) 지원
    - 버그 발생률을 획기적으로 낮춤

- **런타임 안정성 보장**
    - Pydantic을 통해 잘못된 형식의 데이터가 비즈니스 로직으로 유입되는 것을 원천 차단

- **비동기 최적화**
    - LLM 스트리밍이나 대량의 센서 데이터를 실시간으로 처리할 때, 시스템 자원을 효율적으로 분배하는 비동기 처리를 기본으로 지원함

- **현대적 인프라 친화**
    - 비동기 처리가 기본
    - LLM 스트리밍 응답(Server-Sent Events)이나 IoT 센서 데이터의 고가용성 처리에 최적화

- **보안 안정성**
    - 입력 데이터 단계에서 타입 검증 수행
    - 인젝션 공격이나 잘못된 데이터 유입에 의한 런타임 에러(Side-effects)를 최소화

- **자동 문서화 시스템**
    - 코드를 짜면 별도의 작업 없이 실시간으로 테스트 가능한 API 명세서(Swagger UI)가 생성됨

## 2. 주요 특징 및 강점

### 2.1 비동기 프로그래밍 지원 (Async/Await)

- FastAPI는 기본적으로 **ASGI(Asynchronous Server Gateway Interface)**를 지원함
    - 전통적인 WSGI(Django, Flask 등)
        - 요청 하나당 스레드 하나를 점유하는 방식
        - I/O 대기 시간이 발생하면 자원이 낭비됨

    - ASGI(FastAPI 등)
        - `async/await`를 통해 I/O 작업(DB 조회, API 호출) 시 CPU가 다른 요청을 처리할 수 있게 하는 **이벤트 루프(Event Loop)** 방식 사용
        - Node.js나 Go와 대등한 수준의 동시 처리 성능 발휘

### 2.2 강력한 데이터 검증과 자동 문서화

- 시스템 개발 시 가장 번거로운 부분: **API 명세서 관리**와 **입력값 검증**

- **Auto-Documentation**
    - 코드를 작성함과 동시에 `/docs`(Swagger UI)와 `/redoc` 경로에 대화형 API 문서가 실시간으로 생성됨

- **Type Safety**
    - Pydantic 모델을 정의하면
        - 들어오는 JSON 데이터의 타입을 자동으로 체크
        - 에러 메시지까지 생성해 줌

### 2.3 Dependency Injection (의존성 주입) 시스템

- FastAPI는 매우 정교한 `Depends` 시스템을 갖추고 있음
- 데이터베이스 세션 관리, 사용자 인증(OAuth2, JWT), 권한 체크 등을 함수 인자 수준에서 주입 가능
- 이 방식은 코드의 결합도를 낮추고 테스트 유닛을 작성할 때 Mock 객체로 교체하기 매우 용이하게 만듦


## 3. 작동 워크플로우 (Request-Response Cycle)

1.  **Request**: 클라이언트가 JSON 데이터 전송
2.  **Validation**: Pydantic 모델이 정의된 타입과 일치하는지 확인 (틀리면 422 Error 자동 반환)
3.  **Dependency**: 필요한 의존성(예: DB 연결)을 주입
4.  **Logic**: `async` 함수 내에서 비동기 로직 수행
5.  **Response**: 결과를 다시 JSON으로 직렬화하여 클라이언트에 반환

<br>

<div class="insert-image" style="text-align: center;">
    <img style="width: 950px;" src="/materials/python/images/S01-04-04-01_01-002.png">
</div>