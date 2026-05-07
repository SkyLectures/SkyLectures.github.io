---
layout: page
title:  "Docker 개요와 이미지, 컨테이너의 이해"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-01-01_01-DockerOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 도커의 핵심 개념

- 애플리케이션을 **컨테이너**라는 표준화된 유닛으로 패키징하여, 개발·배포·실행을 자동화하는 오픈소스 플랫폼
- 환경의 차이로 발생하는 '내 컴퓨터에서는 되는데 서버에서는 안 돼'라는 고질적인 문제를 해결하는 핵심 기술
- 현대 소프트웨어 엔지니어링에서 인프라를 코드로 관리(IaC)하고, 빠르고 안정적인 배포 체계를 구축하기 위한 필수적인 도구로 자리 잡고 있음

### 1.1 도커 이미지 (Docker Image)

- 애플리케이션을 실행하기 위해 필요한 모든 파일 시스템과 실행 설정값을 포함하고 있는 **읽기 전용(Read-only)** 파일
    - 컨테이너를 생성하기 위한 **읽기 전용 설계도(템플릿)**

- **주요 특징**
    - **Immutable (불변성)**
        - 이미지는 한 번 생성되면 절대로 변하지 않음 🡲 이를 통해 일관된 배포 환경을 보장함
        - 설정이 바뀌어야 한다면 새로운 이미지를 빌드해야 함

    - **Layered Architecture (레이어 구조)**
        - 도커 이미지는 여러 개의 '레이어'가 쌓인 형태
            - 예시
                - `Ubuntu 레이어` + `Python 설치 레이어` + `소스 코드 레이어`가 합쳐져 하나의 이미지가 됨
        - **이점**
            - 동일한 레이어를 사용하는 다른 이미지가 있다면<br>
               🡲 해당 레이어를 재사용하여 저장 공간과 빌드 시간을 획기적으로 줄일 수 있음 (Copy-on-Write 방식)
            - 변경된 부분만 업데이트하므로 효율적

    - **Snapshot**
        - 특정 시점의 애플리케이션 상태를 그대로 보존한 스냅샷과 같음

### 1.2 도커 컨테이너 (Docker Container)

- 애플리케이션 코드와 이를 실행하는 데 필요한 모든 종속성(라이브러리, 설정 파일 등)을 하나로 묶은 소프트웨어 유닛
- 컨테이너는 이미지를 실행한 상태
- 이미지가 정적인 파일이라면, 컨테이너는 메모리 위에서 동작하는 **동적인 프로세스**

- **특징**
    - **격리성**
        - 컨테이너의 핵심 기술 (격리: Isolation)
        - 각 컨테이너는 서로 독립된 공간에서 동작함
        - 컨테이너는 호스트 OS의 커널(자원)을 공유하면서도, 논리적으로는 분리되어 있어 마치 독립된 서버처럼 보임
        - 격리하는 이유
            - **Namespaces**
                - 각 컨테이너가 독립된 네트워크, 파일 시스템, 프로세스 ID(PID) 등을 갖게 하여 서로를 볼 수 없게 만듦
            - **Cgroups (Control Groups)**
                각 컨테이너가 사용할 수 있는 CPU, 메모리 자원량을 제한하여 특정 컨테이너가 전체 시스템을 점유하지 못하게 차단
    - **이식성**
        - 도커만 설치되어 있다면 어떤 OS, 어떤 클라우드 환경에서도 동일하게 작동함

- **이미지와 컨테이너의 관계 (Writable Layer)**
    - 컨테이너가 실행될 때, 도커 엔진은 읽기 전용인 이미지 레이어들 위에 **'컨테이너 레이어(Writable Layer)'**를 한 층 얹음
        - 컨테이너 안에서 파일을 생성하거나 수정하는 모든 작업은 이 최상단의 **컨테이너 레이어**에만 기록됨
        - 컨테이너를 삭제하면 이 레이어도 함께 사라지며, 원본 이미지(하단 레이어들)는 전혀 손상되지 않음


### 1.3 이미지 vs 컨테이너 상세 비교

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">항목</th>
        <th style="width: 250px;">도커 이미지 (Image)</th>
        <th style="width: 250px;">도커 컨테이너 (Container)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">상태</td>
            <td class="td-left">정적 (파일 시스템 저장)</td>
            <td class="td-left">동적 (메모리 실행 중)</td>
        </tr>
        <tr>
            <td class="td-rowheader">수정 가능 여부</td>
            <td class="td-left">읽기 전용 (Read-only)</td>
            <td class="td-left">읽기/쓰기 가능 (Read/Write)</td>
        </tr>
        <tr>
            <td class="td-rowheader">비유</td>
            <td class="td-left">프로그램 설치 파일 (exe), 붕어빵 틀</td>
            <td class="td-left">실행된 프로세스, 붕어빵</td>
        </tr>
        <tr>
            <td class="td-rowheader">생명 주기</td>
            <td class="td-left">삭제하기 전까지 보존</td>
            <td class="td-left">정지(Stop) 및 삭제(Rm) 가능</td>
        </tr>
        <tr>
            <td class="td-rowheader">구성</td>
            <td class="td-left">여러 개의 읽기 전용 레이어</td>
            <td class="td-left">이미지 레이어 + 1개의 쓰기 가능 레이어</td>
        </tr>
    </tbody>
</table>
</div>


### 1.4 실무적 관점에서의 이해

- **왜 이미지를 레이어로 나눌까?**
    - 만약 1GB짜리 이미지를 수정해서 배포한다면,
        - 레이어 구조가 아닐 경우 매번 1GB를 전송해야 함
        - 레이어 구조에서는 **수정된 레이어(예: 10MB 내외의 소스 코드 레이어)**만 전송하면 됨<br>
           🡲 배포 속도가 비약적으로 빨라짐

- **컨테이너는 '휘발성'이다**
    - 컨테이너 레이어에 쓰인 데이터는 컨테이너 삭제 시 사라짐<br>
        🡲 데이터베이스의 데이터처럼 영구적으로 보존해야 하는 값은 컨테이너 내부가 아닌 **도커 볼륨(Docker Volume)**이라는 외부 저장소에 연결(Mount)해서 관리해야 함

> - **요약**
>   - **Dockerfile**이라는 레시피를 통해 **이미지**를 작성
>   - **이미지**는 애플리케이션 실행에 필요한 모든 것이 담긴 압축 파일이며, 결코 변하지 않음
>   - 이 이미지를 실행(run)하면 **컨테이너**가 됨
>   - **컨테이너**는 이미지 위에 얇은 '쓰기 가능 층'을 얹어 독립된 환경에서 돌아가는 실제 서비스임
{: .summary-quote}


## 2. 도커의 주요 아키텍처

- 도커는 클라이언트-서버 구조를 따름

- **Docker Client**
    - 사용자가 `docker run` 같은 명령어를 입력하는 인터페이스

- **Docker Host (Daemon)**
    - 실제로 컨테이너를 관리하고 실행하는 서버 프로세스(`dockerd`)
    - 클라이언트의 요청을 받아 이미지 빌드나 컨테이너 실행을 수행함

- **Docker Registry**
    - 도커 이미지를 저장하고 공유하는 장소
    - 'Docker Hub'가 대표적


## 3. 컨테이너 vs 가상 머신(VM)

- 도커가 기존 가상화 기술보다 가볍고 빠른 이유는 **커널 공유** 방식에 있음

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">구분</th>
        <th style="width: 250px;">가상 머신 (VM)</th>
        <th style="width: 250px;">도커 컨테이너</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">구조</td>
            <td class="td-left">하이퍼바이저 위에 각자 Guest OS 탑재</td>
            <td class="td-left">호스트 OS 위에서 도커 엔진을 통해 커널 공유</td>
        </tr>
        <tr>
            <td class="td-rowheader">성능</td>
            <td class="td-left">OS 부팅이 필요해 무겁고 느림</td>
            <td class="td-left">프로세스 실행 수준으로 매우 가볍고 빠름</td>
        </tr>
        <tr>
            <td class="td-rowheader">용량</td>
            <td class="td-left">기가바이트(GB) 단위</td>
            <td class="td-left">메가바이트(MB) 단위</td>
        </tr>
    </tbody>
</table>
</div>


## 4. 도커를 사용하는 이유 (장점)

- **환경 일관성**
    - 개발, 테스트, 운영 환경을 100% 동일하게 유지 🡲 배포 사고 감소

- **자원 효율성**
    - 하드웨어 가상화 없이 OS 커널 공유 🡲 시스템 리소스를 적게 소모

- **마이크로서비스(MSA) 최적화**
    - 서비스를 작은 단위로 쪼개어 배포하고 확장하기에 용이

- **CI/CD 가속화**
    - 코드 빌드부터 배포까지의 과정을 컨테이너 기반으로 표준화 🡲 자동화 효율 극대화


## 5. 도커 워크플로우 (Lifecycle)

- 일반적으로 다음과 같은 **Build 🡲 Ship 🡲 Run** 과정을 거침

1. **Build**
    - `Dockerfile` 작성
    - `docker build` 명령으로 이미지 생성

2. **Ship**
    - 생성된 이미지를 `docker push`를 통해 레지스트리(Docker Hub 등)에 업로드

3. **Run**
    - 서버에서 `docker pull`로 이미지를 다운로드
    - `docker run`으로 컨테이너 실행


> - 도커의 핵심인 **이미지(Image)**와 **컨테이너(Container)**
>   - 흔히 붕어빵 틀과 붕어빵, 혹은 설계도와 실제 건물에 비유(객체지향 언어에서의 클래스와 인스턴스의 관계와 비슷)
>   - 기술적인 깊이에서 이해하려면 **레이어(Layer)** 구조와 **프로세스 격리**라는 개념을 명확히 파악해야 함
{: .summary-quote}


## 6. 도커 사용하기

- 도커의 핵심은 "어디서나 동일하게 실행되는 소프트웨어 패키징"

- **핵심 명령어**
    - docker pull [이미지명]: 레지스트리에서 이미지 다운로드
    - docker run [옵션] [이미지명]: 이미지를 기반으로 컨테이너 생성 및 실행
    - docker ps: 실행 중인 컨테이너 목록 확인
    - docker exec -it [컨테이너ID] /bin/bash: 실행 중인 컨테이너 내부에 접속