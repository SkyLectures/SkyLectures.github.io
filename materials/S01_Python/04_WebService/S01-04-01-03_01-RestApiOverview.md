---
layout: page
title:  "REST API 개요"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-01-03_01-RestApiOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. REST

### 1.1 REST란?

- Representational State Transfer(표현 상태 전송)의 약자

- 웹 서비스 간의 데이터 교환을 위한 소프트웨어 아키텍처의 하나로 웹의 디자인 원리를 따르는 분산 하이퍼미디어 시스템을 위한 지침을 제공함

- 특정 기술이나 프로토콜이 아니라, 웹과 같은 분산 시스템을 효율적으로 설계하기 위한 일련의 제약 조건(Constraints)과 원칙을 의미함
    - 자원(Resource)을 중심으로 설계
    - 자원을 이름(자원의 표현)으로 구분하여 해당 자원의 상태(정보)를 주고 받는 모든 것을 대상으로 함
    - HTTP 프로토콜을 기반으로 하며, 클라이언트와 서버 간의 상호 작용을 단순하고 효율적으로 만듦

- 기본적으로 웹의 기존 기술과 HTTP 프로토콜을 그대로 활용하기 때문에 웹의 장점을 최대한 활용할 수 있는 아키텍처 스타일이라고 할 수 있음

### 1.2 REST 탄생 배경

- HTTP의 주요 저자 중 한 사람인 로이 필딩(Roy Thomas Fielding) 의 박사 학위 논문(2000)에서 처음 소개됨
    - [논문참고: Architectural Styles and the Design of Network-based Software Architectures](https://ics.uci.edu/~fielding/pubs/dissertation/top.htm)

- 웹 설계의 우수성에 비해 웹 기술이 제대로 사용되지 못하는 모습에 안타까워하여 웹의 장점을 최대한 활용할 수 있는 아키텍처로써 REST를 발표함

### 1.3 REST 아키텍처 스타일의 특징

- 자원 중심(Resource-Based)
    - REST에서 모든 것은 '자원(Resource)'으로 간주됨
    - 자원은 URI(Uniform Resource Identifier)를 통해 식별되며, 특정 형태(예: JSON, XML, HTML 등)로 표현될 수 있음
        - 예: 사용자는 **'/users/{id}'**와 같은 URI로 식별되는 자원이 될 수 있음

- 클라이언트-서버 구조(Client-Server Architecture)
    - 클라이언트와 서버가 독립적으로 작동
        - 클라이언트는 사용자 인터페이스와 사용자 요청을 담당
        - 서버는 자원의 저장 및 처리를 담당
    - 이 분리를 통해 각 부분이 독립적으로 발전하고 확장될 수 있음

- 무상태성(Stateless)
    - 각 요청은 독립적으로 처리되어야 함
    - 서버는 클라이언트의 요청 간에 어떠한 클라이언트 상태 정보도 유지하지 않음
    - 각 요청은 그 자체로 필요한 모든 정보를 포함해야 함
    - 이는 서버의 확장성을 높이고, 장애 발생 시 복구를 용이하게 함
        - 예: 인증 토큰은 각 요청에 포함되어야 함

- 캐시 가능(Cacheable)
    - 클라이언트는 서버의 부하를 줄이고 응답 시간을 개선하기 위하여 서버 응답을 캐시할 수 있도록 설계되어야 함
    - 응답이 캐시 가능한 경우 이를 명시적으로 알려줌으로써 클라이언트가 효율적으로 캐시할 수 있도록 함

- 계층화된 시스템(Layered System)
    - 시스템은 계층화되어야 하며, 각 계층은 다른 계층의 동작을 간섭하지 않음
    - 클라이언트는 중간 서버의 존재를 알 필요 없이 서버와 직접 통신하는 것처럼 작동함
    - 로드 밸런서, 프록시, 캐시 서버 등 여러 계층을 통해 시스템을 구성할 수 있음 → 보안, 성능, 확장성 향상

- 인터페이스 일관성(Uniform Interface)
    - REST의 핵심 원칙 중 하나
    - 클라이언트와 서버 간의 상호작용 방식 통일
        - HTTP 표준을 따르는 통일된 인터페이스를 사용하여 자원을 조작함
    - 제약 조건
        - 자원의 식별(Identification of resources)
            - 모든 자원은 고유한 URI를 통해 식별
        - 표현을 통한 자원 조작(Manipulation of resources through representations)
            - 클라이언트는 자원의 표현(예: JSON, XML)을 받아 조작할 수 있음
        - 자기 서술적 메시지(Self-descriptive messages)
            - 각 메시지는 메시지를 어떻게 처리해야 하는지에 대한 충분한 정보를 포함해야 함
        - 하이퍼미디어(HATEOAS - Hypermedia as the Engine of Application State)
            - 응답에 관련된 자원에 대한 링크를 포함하여 클라이언트가 동적으로 API를 탐색할 수 있도록 함

- 주문형 코드(Code on Demand - Optional)
    - 서버가 클라이언트로 실행 가능한 코드를 전송하여 클라이언트 기능을 확장할 수 있음
        - 선택 사항이며 자주 사용되지는 않음


## 2. REST API(= RESTful API)

### 2.1 REST API란

- 개념
    - REST 아키텍처 스타일의 원칙을 준수하여 설계된 웹 API(Application Programming Interface)
    - 웹의 기본 프로토콜인 HTTP를 사용하여 자원(Resource)을 전송
    - REST의 6가지 제약 조건(특히 인터페이스 일관성)을 따름

- RESTful WEB Service Architecture<br>
    <img src="/materials/images/python/S01-04-02-03_01-001.png" width="500">

- 자원 기반의 구조(ROA: Resource Oriented Architecture)
    - 설계의 중심에 Resoure가 있음
    - 웹의 모든 자원에 고유한 ID인 HTTP URI를 부여
    - HTTP URI를 통해 자원을 명시

- 자원에 대한 작업
    - HTTP Method (POST, GET, PUT, DELETE)를 통해 해당 자원에 대한 CRUD OPERATION을 적용
    - CURD: Create, Update, Retrieve, Delete

### 2.2 REST API의 작동방식

- HTTP 메서드 활용
    - 자원에 대한 CRUD(Create, Read, Update, Delete) 작업을 수행하기 위해 표준 HTTP 메서드를 활용
        - GET: 자원 조회
        - POST: 새로운 자원 생성
        - PUT: 자원 업데이트, 자원이 없으면 생성
        - PATCH: 자원의 일부 업데이트합
        - DELETE: 자원 삭제

- URI를 통한 자원 식별
    - 자원을 고유하게 식별하기 위해 URI(Uniform Resource Identifier) 사용
    - 예: "GET /users/123" → ID가 123인 사용자 조회

- 표준 데이터 형식
    - 자원의 표현(Representation)을 위해 JSON 또는 XML과 같은 표준 데이터 형식을 사용
        - JSON(JavaScript Object Notation): 더 가볍고 파싱하기 쉬워 최근에 많이 사용
        - XML(Extensible Markup Language)

- 무상태 통신
    - 각 요청이 독립적으로 처리됨
    - 서버는 이전 요청에 대한 정보를 유지하지 않음

### 2.3 REST의 주요 구성요소

- 자원(Resource)
    - 자원은 URI(Uniform Resource Identifier)를 통해 고유하게 식별됨
        - 예: `/api/users`는 사용자 자원을 나타냄
    - 자원은 다양한 형태로 표현될 수 있음
        - 주로 JSON, XML, HTML 등이 사용됨

- 표현(Representation)
    - 자원의 상태를 표현하는 방법으로 클라이언트와 서버 간에 주고받는 데이터 형식을 의미함
        - 예: JSON 형식으로 사용자 정보를 표현할 수 있음

- 메서드(Method)
    - HTTP 메서드를 통해 자원에 대한 다양한 동작을 수행함
    - 주요 메서드
        - GET: 자원 조회
        - POST: 새로운 자원 생성
        - PUT: 자원의 전체 수정
        - PATCH: 자원의 일부 수정
        - DELETE: 자원 삭제

- 상태 전이(State Transfer)
    - 클라이언트와 서버 간의 통신에서 상태 전이를 통해 자원을 조작함
    - 각 요청은 독립적으로 처리됨
    - 서버는 요청 간의 상태를 저장하지 않음

> **[정리]**
> - REST의 구성요소는 자원, 표현, 행위(메서드)이며 그 표현 대상은 상태의 전이임
> - REST에서 하나의 자원은 JSON, XML, TEXT, RSS 등 여러 형태의 Representation으로 나타낼 수 있음

### 2.4 REST API URI 설계
- 자원은 복수형을 사용함
    - 예: `/api/users`
- 명확하고 일관된 구조를 사용함
    - 예: `/api/users/{userId}/orders/{orderId}`
- 행위는 URI가 아닌 메서드로 표현함
    - 예: `POST /api/users/{userId}/activation`
- URI는 소문자를 사용함
    - 예: `/api/users`

### 2.5 REST API의 장점

- 단순성과 직관성
    - HTTP 프로토콜을 기반으로 함 → 이해하고 사용하기 쉬움

- 확장성
    - 무상태성 → 서버 확장이 용이

- 유연성
    - 다양한 클라이언트(웹 브라우저, 모바일 앱 등)와 서버 간의 통신에 적합
    - 특정 기술에 종속되지 않음

- 성능
    - 캐싱을 통해 네트워크 트래픽을 감소 및 응답 시간을 개선

- 유지보수 용이성
    - 클라이언트와 서버가 독립적으로 개발됨 → 유지보수가 쉬움

### 2.6 REST API 등장의 필요성

- 최근의 서비스 / 애플리케이션의 개발 흐름 → 멀티 플랫폼, 멀티 디바이스 시대
- 서버 프로그램의 기대 성능 변화
    - 예전: 단순히 하나의 브라우저만 지원
    - 최근: 여러 웹 브라우저 지원 + 모바일 디바이스와의 통신에도 대응할 수 있어야 함
- 플랫폼에 맞추어 새로운 서버를 만드는 수고를 들이지 않기 위해 범용적으로 사용성을 보장하는 서버 디자인이 필요해짐

### 2.7 REST API를 만드는 이유

- 클라이언트 측면
    - 정형화된 플랫폼이 아닌 
    - 모바일, PC, 어플리케이션 등 플랫폼에 제약을 두지 않는 것이 목표

- 2010년 이전
    - 서버 측에서 데이터를 전달해주는 클라이언트 프로그램의 대상은 PC 브라우저(대상이 명확함)
    - 단순히 JSP, ASP, PHP 등을 이용한 웹페이지 구성, 개발로 충분

    > [참고] 최초의 스마트폰
    > - IBM의 사이먼(Simon Personal Communicator). 1992년 컨셉제품 발표, 1994년 출시
    > - 전화 기능외에도 주소록, 세계 시각, 계산기, 메모장, 전자우편, 팩스 송수신, 오락 등의 기능 포함
    > - 터치스크린을 사용하여 손가락으로 전화번호를 입력할 수 있었음

- 2010년 이후
    - 다양한 스마트 기기들의 등장과 TV, 스마트 폰, 테블릿 등 클라이언트 프로그램의 다양화
    - 그에 맞춰 서버를 일일이 개발하는 것이 비효율적인 일이 됨
    - 이 과정에서 개발자들은 
        - 클라이언트를 전혀 고려하지 않고
        - 메시지 기반, XML, JSON과 같이 클라이언트에서 바로 객체로 치환 가능한 형태의 데이터 통신 지향
    - 결과적으로 서버와 클라이언트의 역할이 분리됨
    - 이런 변화에 따라 HTTP 표준 규약을 지키면서 API를 만드는 방식이 요구됨