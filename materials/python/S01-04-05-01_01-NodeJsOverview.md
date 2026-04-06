---
layout: page
title:  "Node.JS의 이해"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-05-01_01-NodeJsOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Node.JS란?

- 브라우저 밖으로 나온 자바스크립트
- Node.js는 프레임워크나 프로그래밍 언어가 아니라, **자바스크립트 런타임(Runtime Environment)**

- **등장 배경**
    - 과거 자바스크립트는 웹 브라우저 안에서만 동작하는 보조적인 언어였음
    - 2009년 라이언 달(Ryan Dahl)이 구글의 고성능 자바스크립트 엔진인 **V8**을 브라우저에서 추출하여 서버에서도 실행할 수 있게 만든 것이 시작
* **핵심 가치**
    - 프론트엔드와 백엔드를 동일한 언어(JavaScript/TypeScript)로 개발할 수 있게 하여 **Full-stack 개발의 문**을 염


## 2. 핵심 아키텍처

- 전통적인 서버 아키텍처는 클라이언트의 요청마다 새로운 스레드를 할당하는 'Thread-per-request' 방식
- Node.js는 정반대의 길을 선택함 🡲 싱글 스레드와 비차단 I/O

- **싱글 스레드 이벤트 루프 (Single-threaded Event Loop)**
    - Node.js는 메인 스레드 하나가 모든 요청을 처리함
    - "하나가 다 처리하면 느리지 않을까?"라는 의문이 생길 수 있지만, 핵심은 **'기다리지 않는 것'**

- **비차단 I/O (Non-blocking I/O)**
    - 파일 읽기, DB 쿼리, 네트워크 요청과 같이 시간이 오래 걸리는 작업(I/O)이 발생하면,
    - Node.js는 이를 운영체제나 별도의 워커 스레드에 맡기고 즉시 다음 작업을 수행함
    - 작업이 완료되면 '이벤트'를 통해 결과를 돌려받음

    > - **비차단 I/O (Non-blocking I/O)와 비동기(Asyncronous)**
    >   - **비차단 I/O (Non-blocking I/O): "제어권"의 문제**
    >       - 호출된 함수가 자신의 작업을 다 마치지 않더라도 제어권을 즉시 호출자에게 넘겨주는 방식
    >       - 동작 방식
    >           - 시스템 콜(System Call)을 보냈을 때,
    >           - 데이터가 준비되지 않았다면 에러 코드(EWOULDBLOCK 등)를 즉시 반환
    >       - 핵심
    >           - 호출한 쪽은 제어권을 바로 돌려받기 때문에 다른 일을 할 수 있지만,
    >           - 작업이 완료되었는지 확인하기 위해 반복적으로 물어봐야(Polling) 함
    >       - 비유
    >           - 식당에서 진동벨이 없어서 손님이 점원에게 "내 음식 나왔나요?"라고 계속 확인하러 가는 상황<br><br>
    >   - **비동기 (Asynchronous): "완료 통지"의 문제**
    >       - 호출된 함수가 작업의 완료를 직접 책임지고 통보해 주는 방식
    >       - 동작 방식
    >           - 호출자는 작업을 요청하면서 콜백(Callback) 함수를 함께 전달
    >           - 호출자는 작업 완료 여부에 신경을 끄고 자기 일을 하다가,
    >           - 나중에 작업이 끝나면 신호(Signal)나 콜백을 통해 결과를 전달받음
    >       - 핵심
    >           - 작업의 완료를 확인하는 주체가 호출자가 아닌 **호출받은 쪽(커널이나 프레임워크)**에 있음
    >       - 비유
    >           - 식당에서 음식을 주문하고 자리에 앉아 있으면,
    >           - 음식이 다 되었을 때 점원이 직접 테이블로 가져다주는 상황<br><br>
    >
    >   - 왜 혼동되는가?
    >       - 현대적인 고성능 서버(Node.js, Go, Nginx 등)는 보통 Non-blocking I/O와 Asynchronous 모델을 동시에 사용하기 때문<br><br>
    >
    >   - **시스템 설계 관점에서 두 개념을 조합하면**
    >
    >   <div class="info-table">
        <table>
            <thead>
                <th style="width: 150px;">구분</th>
                <th style="width: 300px;">비차단 (Non-blocking)</th>
                <th style="width: 300px;">비동기 (Asynchronous)</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">관심사</td>
                    <td class="td-left">제어권 (나 지금 바로 돌아가도 돼?)</td>
                    <td class="td-left">완료 시점 (끝나면 네가 알려줄 거지?)</td>
                </tr>
                <tr>
                    <td class="td-rowheader">응답 방식</td>
                    <td class="td-left">결과가 없으면 없다고 즉시 응답</td>
                    <td class="td-left">일단 접수하고 나중에 통보</td>
                </tr>
                <tr>
                    <td class="td-rowheader">주요 활동</td>
                    <td class="td-left">호출자가 완료 여부를 계속 확인(Polling)</td>
                    <td class="td-left">호출자는 잊고 있다가 알림을 받음</td>
                </tr>
                <tr>
                    <td class="td-rowheader">Node.js 예시</td>
                    <td class="td-left">fs.readSync (Blocking과 대조됨)</td>
                    <td class="td-left">fs.readFile(path, callback) (Async)</td>
                </tr>
            </tbody>
        </table>
        </div>
    {: .common-quote}

## 3. 내부 구조: V8 엔진과 Libuv

- **V8 Engine**
    - 구글이 만든 자바스크립트 엔진
    - 자바스크립트 코드를 기계어로 초고속 컴파일함

- **Libuv:**
    - C++로 작성된 라이브러리
    - Node.js의 핵심인 **이벤트 루프**를 구현함
    - 시스템의 비동기 I/O를 실질적으로 담당함


## 4. Node.js의 장단점 및 적합한 유스케이스

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">구분</th>
        <th style="width: 520px;">내용</th>
        <th style="width: 280px;">비고</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">장점</td>
            <td class="td-left">압도적인 처리 속도(I/O 한정), 풍부한 생태계(NPM), 개발 생산성</td>
            <td class="td-left">실시간 채팅, 스트리밍에 최적</td>
        </tr>
        <tr>
            <td class="td-rowheader">단점</td>
            <td class="td-left">CPU 집약적 작업(복잡한 수학 계산, 영상 인코딩)에 취약</td>
            <td class="td-left">싱글 스레드 병목 현상 발생 가능</td>
        </tr>
        <tr>
            <td class="td-rowheader">생태계</td>
            <td class="td-left"><b>NPM (Node Package Manager):</b> 세계 최대 규모의 오픈소스 라이브러리 저장소</td>
            <td class="td-left">라이브러리 조합만으로 빠른 구현 가능</td>
        </tr>
    </tbody>
</table>
</div>


## 5. 현대적 실무에서의 위상

- **Microservices Architecture (MSA)**
    - 가볍고 빠르기 때문에 서비스를 작게 쪼개어 배포하는 MSA 환경에서 가장 선호되는 도구

- **Frontend Tooling**
    - 현재 우리가 사용하는 React, Next.js, Vue 등 모든 현대적 프론트엔드 도구들이 Node.js 환경 위에서 빌드되고 돌아감
    - 즉, **현대 웹 개발자의 PC에 반드시 설치되어야 하는 필수 OS**와 같은 위치


> - **ERP나 스마트팩토리**의 대규모 트랜잭션 처리에서,
>   - 실시간 데이터 수집(Sensor Data)이나 대시보드 알림 기능을 구현할 때
>   - Node.js는 Java 기반 시스템보다 훨씬 적은 리소스로 높은 효율을 낼 수 있음
>   - 단, 복잡한 비즈니스 로직이나 금융권의 엄격한 트랜잭션 관리가 필요한 영역에서는 여전히 Java/C# 계열이 강세
{: .expert-quote}
