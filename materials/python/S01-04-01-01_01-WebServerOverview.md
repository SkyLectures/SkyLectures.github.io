---
layout: page
title:  "웹 서버 개발의 이해"
date:   2025-03-01 10:00:00 +0900
permalink: /material/S01-04-01-01_01-WebServerOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 웹(World Wide Web, WEB) 개요
- **월드 와이드 웹(World Wide Web, WEB)** 의 약자
- 인터넷에 연결된 사용자들이 정보를 공유할 수 있는 공간을 의미함
- **하이퍼텍스트(Hypertext)** 방식으로 정보를 연결하며
- **HTML**과 **HTTP** 프로토콜을 사용하여 다양한 콘텐츠를 제공함
- 인터넷의 대표적인 서비스 중 하나로, 정보를 쉽게 공유하고 접근할 수 있도록 지원함

## 2. 웹의 동작

### 2-1. 웹 시스템의 동작 구조

<p align="center"><img src="https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/images/S01-04-01-00_001.png" width="800"></p>

- 요청과 응답
    - 요청과 응답은 HTTP(Hypertext Transfer Protocol)라는 프로토콜을 지키면서 통신을 수행함
    - 요청: 클라이언트에서 서버로 정보를 요구하기 위해 보내는 메시지
        - GET: 서버로부터 정보를 조회할때 사용
        - POST: 새로운 정보를 서버에 생성하고 저장할때 사용
        - PUT: 서버의 기존 리소스를 전체적으로 업데이트할때 사용
        - PATCH: 서버의 기존 리소스를 부분적으로 업데이트할때 사용
        - DELETE: 서버에서 기존 리소스를 제거할때 사용
    - 응답: HTTP에서 요구된 메시지에 대한 응답, HTML, 이미지 등이 전달됨<br><br>


### 2-2. 클라이언트-서버 구조
- 네트워크 기반의 컴퓨팅 모델의 하나
- 서비스를 요청하는 클라이언트와 이를 제공하는 서버로 구성됨

<p align="center"><img src="https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/images/S01-04-01-00_002.png" width="800"></p>

- 주요 특징
    - 역할 분담
        - 클라이언트: 사용자 인터페이스와 상호작용을 담당
        - 서버: 데이터 처리, 저장, 보안 등의 백엔드 작업 수행
    - 중앙 집중화
        - 서버에서 데이터와 리소스를 중앙 관리
        - 일관성과 보안을 강화함
    - 확장성
        - 서버 추가나 업그레이드를 통해 시스템 처리 능력을 쉽게 확장할 수 있음
    - 다양한 클라이언트 지원
        - 여러 종류의 클라이언트 디바이스에서 동일한 서비스를 이용할 수 있음
    - 네트워크 효율성
        - 요청과 응답을 최적화하여 네트워크 트래픽을 감소시킴

- 각 요소의 작동 방식

    - Server의 작동 방식
    <p align="center"><img src="https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/images/S01-04-01-00_003.jpg" width="600"></p>
    <br><br>

    - Client의 작동 방식
    <p align="center"><img src="https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/images/S01-04-01-00_004.jpg" width="400"></p>
    <br><br>

    - Server-Client의 작동 방식
    <p align="center"><img src="https://raw.githubusercontent.com/SkyLectures/LectureMaterials/main/images/S01-04-01-00_005.jpg" width="600"></p>
    <br><br>


## 3. Web Server 기술 분류

### 3-1. Web Server
- HTTP Request 에 맞는 웹페이지를 Response 해주는 기능을 가짐(Static, 정적임)
    - 정적 이란?
        - Web Server 에 있는 웹페이지를 그대로 Response 함
        - 웹페이지가 변경 되지 않는 동일한 웹페이지를 Response 함

        <p align="center"><img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*s63xJ_94_5tUTUt8ruh-1Q.png"></p>

### 3-2. Web Application Server
- Web Server + Web Container( = Web Server + CGI) = WAS(Web Application Server)
- Web Server 가 동적으로 동작하면 Web Application Server
- 동적이란?
    - Request 에 따라 데이터를 가공하여 생성된 웹페이지를 Response
    - Request 에 따라 Response 되는 웹페이지가 달라짐

    <p align="center"><img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*vsUpO7CuDvG-v1T11Loq9A.png"></p>

### 3-3. WSGI (Web Server Gateway Interface)
- Web Server 의 Request 를 Python Application 으로 보내주고 Response 를 받아서 Web Server 로 보내주는 Web Server Gateway Interface 임
- Web Server 의 Request 를 Callable Object 를 통해 Application 에 전달Callable Object 는 Function 이나 Object 의 형태로 HTTP Request 에 대한 정보(Method, URL, Data, …), Callback 함수 정보를 전달
- 2003년 파이썬 표준으로 WSGI 가 나온 이후 현재까지 사용됨
- WSGI Middleware 는 WSGI 의 구현체로 Request 를 Flask, django 와 같은 Web Framework 에 연결하는 WSGI server의 역할을 함(gunincorn, uWSGI, Werkzeug 등이 있음)
- WSGI 는 Synchronous 하게 작동하기에 동시에 많은 Request 를 처리하는데 한계가 있음(Celery, Queue 를 이용하여 성능 향상 가능)

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*ODWnK8-rSRgNwd5PTm5Okg.png"></p>

### 3-4. ASGI (Asyncronous Server Gateway Interface)
- WSGI 와 비슷한 구조를 가지나 기본적으로 모든 요청을 Asynchronous 로 처리하는게 다름
- WSGI 에서 지원 되지 않는 Websocket, HTTP 2.0 을 지원함
- ASGI 는 WSGI 와 호환됨(=ASGI는 WSGI의 상위 버전임)
- WSGI 가 Synchronous 하게 작동함으로써 발생하는 한계를 해결하기 위해Uvicorn 과 같은 Asynchronous Server Gateway Interface 가 나옴 → WSGI의 단점은 요청을 받고 응답을 반환하는 단일 동기 호출 방식이라는 것 → 웹소켓을 사용할 수 없음. wsgi.websocket을 사용할 수 있지만, 표준화안됨
- 단일 astnchronous(비동기) 호출이 가능 하므로 여러 이벤트를 주고받을수 있음 →  대용량 트래픽 처리를 유연하게 할 수 있음

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*9YVDOeD0H2nzNHG2BTPAog.png"></p>


## 4. 백엔드 개발자가 하는 일

1. 서버 개발
    - 클라이언트의 요청을 접수하고 이를 처리한 후 적절한 응답을 보내는 서버의 개발
    - 백엔드 개발 언어와 프레임워크를 사용해 웹 애플리케이션의 핵심 로직 구현

2. 데이터베이스 설계 및 관리
    - 웹 애플리케이션의 데이터를 효율적으로 저장, 관리하기 위한 데이터베이스의 설계 및 관리
    - 웹 애플리케이션을 운영할 적절한 DBMS의 선택
    - 데이터베이스의성능과 확장성을 고려한 데이터 모델의 설계 및 운영

3. API 개발
    - API: 프론트엔드와 백엔드가 데이터를 효율적으로 주고받을 수 있는 인터페이스
    - 백엔드 개발자는 이러한 API를 개발하여 프론트엔드에서 특정 데이터나 기능에 접근할 수 있게 함
    - API는 다른 서비스나 플랫폼과 통합할 때에도 중요한 역할을 함

4. 보안 및 인프라 관리
    - 외부 공격으로부터 서버와 데이터를 지키기 위한 보안 정책의 수립 및 관련 기술 적용
    - 안정적인 서비스의 제공을 위해 서버 구성 및 모니터링 등의 인프라 관리 업무 수행
    - 인프라
        - 인프라스트럭처(Infrastructure)
        - 서버를 구성하는 하드웨어 기기(서버 장비, 스토리지, 네트워크 장비 등), 미들웨어, 운영체제 등

5. 네트워크 설정
    - 네트워크 설정을 통해 서버와 클라이언트 간의 효율적인 통신 보장
    - 로드 밸런싱(Load Balancing, 작업을 나누어 부하를 분산하는 것), 캐싱(Caching, 자주 사용하는 데이터의 복사본을 고속 저장소에 저장하는 것), 네트워크 보안 설정 등을 통해 사용자 경험과 서버의 성능을 향상시킴
