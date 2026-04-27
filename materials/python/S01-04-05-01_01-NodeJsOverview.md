---
layout: page
title:  "Node.JS의 이해"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-05-01_01-NodeJsOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Node.js 개요

- 2009년 라이언 달(Ryan Dahl)에 의해 발표
- 현대 소프트웨어 아키텍처의 핵심이자 서버 사이드 자바스크립트의 표준
- 2026년 현재, 단순한 런타임을 넘어 엔터프라이즈급 안정성과 클라우드 네이티브 환경에 최적화된 생태계를 구축 중

### 1.1 기초 개념 및 핵심 특징

- Node.js는 **Chrome V8 자바스크립트 엔진**으로 빌드된 **비동기 이벤트 주도(Event-driven) JavaScript 런타임**

- **Runtime, not Framework**
    - Node.js는 언어나 프레임워크가 아니라
    - 자바스크립트가 브라우저 밖(서버, 데스크탑 등)에서 실행될 수 있도록 하는 **환경**

- **Single-Threaded**
    - 자바스크립트 코드를 실행하는 메인 스레드는 하나
    - 내부적으로는 멀티 스레드를 활용해 효율을 극대화 함

- **Non-blocking I/O**
    - 입출력(파일, 네트워크 등) 작업 시 결과가 올 때까지 기다리지 않고 다음 코드를 실행
    - 이는 대규모 동시 접속 처리에 매우 유리


### 1.2 핵심 기술 및 내부 구조 (Architecture)

- Node.js의 고성능은 **V8 엔진**과 **libuv** 라이브러리의 조합에서 나옴

- **V8 Engine**
    - 구글이 C++로 작성한 고성능 엔진
    - 자바스크립트 코드를 **JIT(Just-In-Time) 컴파일**을 통해 기계어로 직접 번역함
    - 2026년 현재 V8은 WebAssembly(Wasm) 지원이 더욱 강화됨 🡲 AI 모델 추론 등 연산 집약적 작업에서도 높은 성능을 보임

- **libuv와 이벤트 루프 (Event Loop)**
    - libuv
        - C++ 기반의 라이브러리
        - 시스템 커널을 사용하여 **비동기 I/O**를 구현함

    - **이벤트 루프**
        - 6개의 단계(Timers 🡲 Pending 🡲 Poll 🡲 Check 등)를 돌며 콜백 함수를 처리

    - **스레드 풀(Thread Pool)**
        - 암호화(Crypto), 압축(Zlib), 파일 I/O와 같이 무거운 작업은 메인 스레드가 아닌 내부 스레드 풀(기본 4개, 최대 1024개 확장 가능)에서 처리하여 메인 스레드의 차단(Blocking)을 방지


### 1.3 브라우저 자바스크립트와의 차이점

- 두 환경 모두 V8 엔진을 사용하지만, 제공하는 **API와 목적**이 완전히 다름

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">비교 항목</th>
        <th style="width: 400px;">브라우저 (Client-side)</th>
        <th style="width: 400px;">Node.js (Server-side)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">전역 객체</td>
            <td class="td-left">`window`, `document` (DOM 존재)</td>
            <td class="td-left">`global`, `process` (DOM 없음)</td>
        </tr>
        <tr>
            <td class="td-rowheader">파일 시스템</td>
            <td class="td-left">보안상 제한됨</td>
            <td class="td-left">`fs` 모듈을 통한 자유로운 읽기/쓰기</td>
        </tr>
        <tr>
            <td class="td-rowheader">모듈 시스템</td>
            <td class="td-left">ES Modules (import/export)</td>
            <td class="td-left">CommonJS (require) + ES Modules (혼용)</td>
        </tr>
        <tr>
            <td class="td-rowheader">주요 역할</td>
            <td class="td-left">UI 상호작용, DOM 조작</td>
            <td class="td-left">서버 로직, 데이터베이스 연결, OS 제어</td>
        </tr>
    </tbody>
</table>
</div>



### 1.4 장단점 분석 (2026년 관점)

- **장점 (Pros)**
    - **Full-stack Unity**
        - 프론트엔드와 백엔드를 하나의 언어(JS/TS)로 개발
        - 코드 공유와 생산성이 극대화됨

    - **High Concurrency**
        - 가벼운 I/O 요청이 빈번한 실시간 서비스(채팅, 스트리밍, IoT 데이터 수집)에 최적

    - **Massive Ecosystem**
        - NPM을 통해 수백만 개의 모듈을 즉시 활용할 수 있음
        - 2026년에는 AI SDK(LangChain, OpenAI 등)와의 결합이 매우 강력해짐

- **단점 (Cons)**
    - **CPU Bound Task**
        - 복잡한 수학 연산이나 AI 모델 학습 등 CPU 점유율이 높은 작업 시 이벤트 루프가 멈출 수 있음
        - Worker Threads로 보완 가능하지만 설계가 복잡해짐

    - **Callback/Async 복잡성**
        - 비동기 흐름을 잘못 관리하면 코드 가독성이 떨어짐


### 1.5 현대적 실무에서의 역할 및 위상

- 현재 Node.js는 단순한 웹 서버를 넘어 다음과 같은 영역에서 중추적인 역할을 수행 중

- **마이크로서비스 및 클라우드 네이티브 (Serverless)**
    - 가볍고 실행 속도가 빨라
    - AWS Lambda, Google Cloud Run과 같은 **서버리스(Serverless)** 환경과
    - **Docker/Kubernetes** 기반의 마이크로서비스 아키텍처(MSA)에서
    - 가장 선호되는 언어 중 하나

- **AI 연동 및 오케스트레이션**
    - 최근, AI 에이전트 개발 시
        - Python으로 모델을 만들고,
        - **Node.js(TypeScript)**로 서비스의 API와
        - 실시간 오케스트레이션을 담당하는 구조가 정착됨
    - 특히 `langchain.js`와 같은 라이브러리의 발달로 AI 서비스 구축의 핵심 툴이 됨

- **스마트팩토리 및 IoT 데이터 게이트웨이**
    - 수천 개의 센서로부터 들어오는 실시간 데이터를
    - 비차단(Non-blocking) 방식으로 수집하고 처리하는
    - **데이터 게이트웨이**로서의 위상이 매우 높음

## 2. 설치와 환경 구축

> - Node.js를 활용한 전문적인 개발 환경을 구축하기 위해서는 단순히 실행 파일을 설치하는 것을 넘어,
> - **버전 관리**와 **패키지 관리**를 효율적으로 제어할 수 있는 구조를 만드는 것이 핵심
{: .common-quote}


- 직접 홈페이지에서 설치하는 방식은 간편하지만, 여러 프로젝트의 호환성을 관리하기 어려움
- 따라서 버전 관리 도구를 먼저 설치하는 것이 실무 표준

1. **NVM (Node Version Manager) 설치**
    - **Windows**
        - [nvm-windows](https://github.com/coreybutler/nvm-windows)를 다운로드하여 설치

    - **macOS/Linux**: 터미널에서 다음 명령어를 실행
        ```bash
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        ```

2. **Node.js 버전 설치 및 선택**
    - 설치 후 터미널(또는 CMD)에서 다음 명령어로 안정적인 버전(LTS)을 설치

        ```bash
        nvm install --lts    # 최신 안정화 버전(LTS) 설치
        nvm use --lts        # 설치한 LTS 버전 활성화
        node -v              # 설치 확인 (v20.x.x 등 출력)
        ```

3. **패키지 매니저 환경 구축 (NPM & Yarn/PNPM)**

    - Node.js를 설치하면 **NPM(Node Package Manager)**이 기본 포함됨
    - 최근 실무에서는 성능과 효율성에 따라 다른 도구를 병행

    - **NPM**
        - 기본 패키지 매니저
        - 전 세계 표준

    - **PNPM (추천)**
        - 하드 링크를 사용하여 디스크 공간을 절약하고 설치 속도가 매우 빠름
        - 스마트팩토리 데이터 처리처럼 대규모 종속성이 필요한 프로젝트에 유리함
        - 설치: `npm install -g pnpm`

    - **Yarn (Berry)**
        - 대규모 프로젝트에서 의존성 추적의 무결성이 뛰어남


4. **개발 IDE 및 필수 확장 도구 (VS Code)**

    - Node.js 개발에 가장 최적화된 도구는 **Visual Studio Code (VS Code)**

    - **필수 확장 프로그램 (Extensions)**
        1. **ESLint**: 코드 품질 및 문법 오류 체크 (필수)
        2. **Prettier**: 코드 포맷팅 자동화
        3. **Thunder Client**: Node.js 서버 API를 즉시 테스트 (Postman 대체)
        4. **Error Lens**: 에러 메시지를 코드 줄 바로 옆에 표시하여 가독성 향상


## 3. 실무형 프로젝트 초기화 (Standard Structure)

- 환경 구축이 끝나면, 전문적인 개발을 위해 다음과 같이 프로젝트를 시작

- **프로젝트 폴더 생성 및 이동**
    - `mkdir my-ai-project && cd my-ai-project`

- **프로젝트 초기화**
    - `npm init -y` (package.json 생성)

- **필수 환경 변수 관리**
    - `npm install dotenv`
    - `.env` 파일을 생성하여 DB 접속 정보나 AI API Key 등을 안전하게 관리

- **GitIgnore 설정**
    - `node_modules`, `.env` 등은 Git 관리에서 제외하도록 `.gitignore`를 설정


## 4. Node.js 구동 모드: 개발 vs 프로덕션

- `npm install -D nodemon`

- **운영용 (PM2)**
    - 서버가 예기치 않게 종료되었을 때 자동 재시작 지원
    - 멀티 코어를 활용할 수 있게 **클러스터링**을 지원

- `npm install -g pm2`

