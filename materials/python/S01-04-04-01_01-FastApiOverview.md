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

### 1.1 기술적 정의
- Python 3.8+의 **표준 타입 힌트(Standard Python Type Hints)**를 기반으로 API를 구축하기 위한 현대적이고 빠른(고성능) 웹 프레임워크
- Python의 생산성과 C++/Java 수준의 엄밀함 및 성능을 결합하려는 현대적 시도의 일환

- **성능(Performance)**
    - Node.JS 및 Go와 대등한 수준의 성능을 제공 🡲 Python 프레임워크 중 가장 빠른 것 중 하나로 간주됨

- **표준 기반**
    - API의 산업 표준인 OpenAPI(이전의 Swagger)와 JSON Schema를 100% 계승하고 준수함

### 1.2 구조적 정의
- FastAPI는 바닥부터 모든 것을 새로 만들기보다, 이미 검증된 강력한 기술들을 통합하여 탄생함

- **Starlette (Web Layer)**
    - 비동기(Asynchronous) 웹 서비스를 구축하기 위한 경량 ASGI 프레임워크
    - 고성능 라우팅과 웹 서버 기능을 제공

- **Pydantic (Data Layer)**
    - Python의 Type Hint를 사용하여 데이터 검증(Validation)과 직렬화(Serialization)를 수행
    - 런타임에서 데이터 타입을 강제하여 오류를 사전에 방지
        - 클래스 기반으로 데이터 구조를 정의하면, 런타임에 타입을 강제하고 변환해 줌

### 1.3 실무적 정의
- FastAPI는 다음과 같은 역할을 수행하는 엔진

- **자동 문서화 시스템**
    코드를 짜면 별도의 작업 없이 실시간으로 테스트 가능한 API 명세서(Swagger UI)가 생성됨

- **런타임 안정성 보장**
    - Pydantic을 통해 잘못된 형식의 데이터(예: 주식 수량에 문자열이 들어오는 경우)가 비즈니스 로직으로 유입되는 것을 원천 차단

- **비동기 최적화**
    - LLM 스트리밍이나 대량의 센서 데이터를 실시간으로 처리할 때, 시스템 자원을 효율적으로 분배하는 비동기 처리를 기본으로 지원함


## 2. 주요 특징 및 강점

### 2.1 비동기 프로그래밍 지원 (Async/Await)

- FastAPI는 기본적으로 **ASGI(Asynchronous Server Gateway Interface)**를 지원함
    - 전통적인 WSGI(Django, Flask 등)
        - 요청 하나당 스레드 하나를 점유하는 방식
        - I/O 대기 시간이 발생하면 자원이 낭비됨

    - ASGI(FastAPI 등)
        - `async/await`를 통해 I/O 작업(DB 조회, API 호출) 시 CPU가 다른 요청을 처리할 수 있게 하는 **이벤트 루프(Event Loop)** 방식을 사용합니다. 이는 Node.js나 Go와 대등한 수준의 동시 처리 성능을 냅니다.

### 2.2 강력한 데이터 검증과 자동 문서화

- 시스템을 개발 시 가장 번거로운 부분: **API 명세서 관리**와 **입력값 검증**

- **Auto-Documentation**
    - 코드를 작성함과 동시에 `/docs`(Swagger UI)와 `/redoc` 경로에 대화형 API 문서가 실시간으로 생성됨

- **Type Safety**
    - Pydantic 모델 정의하면
        - 들어오는 JSON 데이터의 타입을 자동으로 체크
        - 에러 메시지까지 생성해 줌

### 2.3 Dependency Injection (의존성 주입) 시스템

- FastAPI는 매우 정교한 `Depends` 시스템을 갖추고 있음

- 데이터베이스 세션 관리, 사용자 인증(OAuth2, JWT), 권한 체크 등을 함수 인자 수준에서 주입 가능
- 이 방식은 코드의 결합도를 낮추고 테스트 유닛을 작성할 때 Mock 객체로 교체하기 매우 용이하게 만듦



## 3. 작동 워크플로우 (Request-Response Cycle)

1.  **Request**: 클라이언트가 JSON 데이터를 전송합니다.
2.  **Validation**: Pydantic 모델이 정의된 타입과 일치하는지 확인합니다. (틀리면 422 Error 자동 반환)
3.  **Dependency**: 필요한 의존성(예: DB 연결)을 주입합니다.
4.  **Logic**: `async` 함수 내에서 비동기 로직을 수행합니다.
5.  **Response**: 결과를 다시 JSON으로 직렬화하여 클라이언트에 반환합니다.



---

## 4. 고성능 백엔드로서의 위상
FastAPI는 현재 Python 생태계에서 가장 빠르게 성장하는 프레임워크입니다. 특히 다음과 같은 영역에서 두각을 나타냅니다.
* **AI/ML 모델 서빙**: 비동기 처리가 필수적인 LLM(Large Language Model) 스트리밍 응답이나 실시간 추론 API에 최적입니다.
* **Microservices**: 가볍고 빠르며 Docker 환경에서 컨테이너화하기 매우 효율적입니다.

박사님께서 운영하시는 **Stock-Ops** 시스템처럼 실시간 금융 데이터를 수집하고 LLM으로 분석하여 스트리밍하는 구조라면, FastAPI의 비동기 제너레이터와 Pydantic의 고속 파싱 기능이 핵심적인 역할을 할 것입니다.

혹시 이 개요 중에서 박사님의 연구 분야와 관련하여 특정 모듈(예: 비동기 DB 엔진 연동 등)의 상세 구현 방식이 궁금하신가요?