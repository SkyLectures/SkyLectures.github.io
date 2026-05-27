---
layout: page
title:  "Docker 개요와 이미지, 컨테이너의 이해"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-01-01_01-DockerOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 도커의 핵심 개념 (Docker Overview)

### 1.1 도커란 무엇인가?

> - 애플리케이션과 그 실행에 필요한 모든 환경(코드, 런타임, 시스템 도구, 라이브러리 등)을 **'컨테이너'**라는 경량의 표준화된 유닛으로 패키징하여,
> - 어디서나 일관되게 실행할 수 있도록 돕고, 개발·배포·실행을 자동화하는 오픈소스 플랫폼
> - 현대 소프트웨어 엔지니어링에서 인프라를 코드로 관리(IaC)하고, 빠르고 안정적인 배포 체계를 구축하기 위한 필수적인 도구로 자리 잡고 있음
{: .common-quote}

### 1.2 도커가 해결하는 고질적인 문제

- 현대 소프트웨어 개발 환경은 매우 복잡함
- 개발자의 PC, 테스트 서버, 클라우드 운영 환경의 OS 버전이나 설치된 라이브러리가 조금만 달라도 서비스가 죽는 일이 비일비재함<br><br>
- 도커가 해결하는 대표적인 문제점
    - **"내 컴퓨터에서는 잘 되는데요?" 문제:**
        - 특정 라이브러리의 버전 충돌이나 OS 환경 차이로 발생하는 배포 오류를 원천 차단
    - **복잡한 설정의 자동화:**
        - 새로운 개발자가 팀에 합류했을 때, 수많은 환경 설정 없이 도커 명령어 한 줄로 즉시 개발 환경을 구축 가능
    - **자원 낭비 방지:**
        - 무거운 가상 머신(VM)을 여러 개 띄우지 않고도, 하나의 OS 위에서 여러 애플리케이션을 가볍고 독립적으로 실행할 수 있음

### 1.3 도커의 3대 핵심 가치

- **표준화 (Standardization):**
    - 어떤 언어로 만든 앱이든 동일한 방식으로 패키징하고 배포함
    - 마치 선박의 컨테이너가 내용물에 상관없이 동일한 규격의 크레인으로 옮겨지는 것과 같음
- **격리 (Isolation):**
    - 각 컨테이너는 서로 독립된 공간에서 실행됨
    - 한 컨테이너에 오류가 발생해도 다른 컨테이너나 호스트 OS에 영향을 주지 않음
- **이식성 (Portability):**
    - "한 번 빌드하면 어디서든 실행된다(Build Once, Run Anywhere)."
    - 개발 PC에서 만든 컨테이너 그대로 클라우드나 온프레미스 서버로 옮겨 실행할 수 있음


## 2. 도커의 아키텍처 (Architecture)

> - 도커는 '어디서나 동일하게 실행되는 환경'을 만들기 위해 클라이언트-서버(C/S) 구조의 아키텍처를 채택
> - 크게 **Docker Client**, **Docker Host**, **Docker Registry** 세 가지 핵심 구성 요소로 이루어짐
{: .common-quote}

<div class="insert-image" style="text-align: right;">
    <img src="/materials/devtools/images/S13-01-01-01_01-001_Docker_Architecture.png" style="width: 90%;">
</div>

### 2.1 도커의 전체 구성요소

- **Docker Client (도커 클라이언트)**
    - 사용자가 도커와 상호작용하는 첫 번째 관문
    - **명령어 입력**
        - 사용자가 `docker run`, `docker build`, `docker pull` 등의 명령어를 입력하는 터미널(CLI) 환경
    - **통신 주체**
        - 클라이언트는 사용자의 명령을 받아 도커 데몬(Docker Daemon)에게 REST API를 통해 전달함
        - 클라이언트와 데몬은 동일한 시스템에 있을 수도 있고, 원격으로 연결될 수도 있음

- **Docker Host (도커 호스트, Daemon)**
    - 도커 애플리케이션이 실제로 실행되는 물리적 또는 가상 머신
    - **Docker Daemon (dockerd)**
        - 호스트의 핵심 프로세스. 실제로 컨테이너를 관리하고 실행하는 서버 프로세스(`dockerd`)
        - 백그라운드에서 실행되며 클라이언트의 요청을 대기함
        - 컨테이너, 이미지, 네트워크, 볼륨 등 모든 도커 객체를 관리하고 실행하는 '두뇌' 역할 수행
            - 클라이언트의 요청을 받아 이미지 빌드나 컨테이너 실행을 수행함
    - **Images (이미지)**
        - 컨테이너를 생성하기 위한 읽기 전용 설계도(템플릿)가 호스트 내부에 저장되어 있음
    - **Containers (컨테이너)**
        - 이미지를 기반으로 생성되어 실제로 동작하는 애플리케이션의 인스턴스
    - **Networks & Volumes**
        - 컨테이너 간의 통신 경로(Network)와 데이터 영속성을 위한 저장 공간(Volume)도 호스트 내에서 관리함

- **Docker Registry (도커 레지스트리)**
    - 도커 이미지를 저장하고 배포하는 중앙 저장소
    - **이미지 공유**
        - 사용자가 만든 이미지를 업로드(Push)하거나, 필요한 이미지를 다운로드(Pull)하는 역할 수행
    - **Docker Hub**
        - 도커에서 공식적으로 운영하는 세계 최대의 공용 레지스트리(Public Registry)
        - 기업 내부에서만 사용하는 사설 레지스트리(Private Registry)를 직접 구축할 수도 있음


### 2.2 도커 이미지

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


### 2.3 도커 컨테이너

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


### 2.4 이미지 vs 컨테이너

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

<br>

> - 도커의 핵심인 **이미지(Image)**와 **컨테이너(Container)**
>   - 흔히 붕어빵 틀과 붕어빵, 혹은 설계도와 실제 건물에 비유(객체지향 언어에서의 클래스와 인스턴스의 관계와 비슷)
>   - 기술적인 깊이에서 이해하려면 **레이어(Layer)** 구조와 **프로세스 격리**라는 개념을 명확히 파악해야 함
{: .summary-quote}


### 2.5 실무적 관점에서의 이해

- **왜 이미지를 레이어로 나눌까?**
    - 만약 1GB짜리 이미지를 수정해서 배포한다면,
        - 레이어 구조가 아닐 경우 매번 1GB를 전송해야 함
        - 레이어 구조에서는 **수정된 레이어(예: 10MB 내외의 소스 코드 레이어)**만 전송하면 됨<br>
           🡲 배포 속도가 비약적으로 빨라짐

- **컨테이너는 '휘발성'이다**
    - 컨테이너 레이어에 쓰인 데이터는 컨테이너 삭제 시 사라짐<br>
        🡲 데이터베이스의 데이터처럼 영구적으로 보존해야 하는 값은 컨테이너 내부가 아닌 **도커 볼륨(Docker Volume)**이라는 외부 저장소에 연결(Mount)해서 관리해야 함


### 2.6 컨테이너 데이터 관리

- **데이터 영속성의 부여**
    - 컨테이너의 쓰기 레이어는 휘발성
    - 컨테이너가 삭제되어도 데이터를 보존하려면 호스트 OS의 저장 공간을 컨테이너와 연결(Mount)해야 함
        - DB 컨테이너를 띄울 때 볼륨 설정을 모르면 컨테이너 재시작 시 데이터가 모두 날아가는 상황을 겪게 됨

    - **Bind Mount:**
        - 호스트 시스템의 특정 경로(예: /home/user/data)를 컨테이너 내부와 직접 연결
        - 설정 파일 공유나 개발 중인 소스 코드 동기화에 주로 사용됨
        - 사용 예시 (소스 코드 동기화)
            - 호스트의 특정 폴더를 컨테이너에 연결하여, 호스트에서 코드를 수정하면 컨테이너에 즉시 반영되게 함
            - 주로 개발 환경에서 사용

            ```bash
            # 호스트의 /home/user/project 폴더를 컨테이너의 /app 폴더로 연결
            docker run -d \
                --name my-web-app \
                -v /home/user/project:/app \
                nginx
            ```

    - **Docker Volume:**
        - 도커가 관리하는 별도의 저장 공간을 생성하여 연결
        - 호스트의 특정 경로를 신경 쓸 필요가 없고, 도커 명령어로 관리하기 쉬워 DB 데이터 저장 등에 권장되는 방식
        - 사용 예시 (데이터베이스 저장)
            - 도커가 관리하는 안전한 영역에 데이터를 저장함
            - 컨테이너를 지워도 데이터는 유지됨
            - 주로 운영 환경이나 DB에서 사용

            ```bash
            # 1. 'db_data'라는 이름의 도커 볼륨 생성
            docker volume create db_data

            # 2. 생성한 볼륨을 MySQL 데이터 저장 경로에 연결하여 실행
            docker run -d \
                --name mysql-db \
                -e MYSQL_ROOT_PASSWORD=password123 \
                -v db_data:/var/lib/mysql \
                mysql:8.0
            ```

            - **주의**
                - 초기 학습 단계에서는 직관성을 위해 환경 변수에 직접 패스워드를 명시했지만,
                - 실무 운영 환경에서는 향후 배울 .env 파일 분리나 클라우드 Secret 자격 증명 서비스를 연동하는 것이 보안의 철칙

### 2.7 도커 네트워크의 이해

- 컨테이너는 기본적으로 격리된 네트워크 환경을 가짐
- 외부와 통신하거나 컨테이너끼리 대화하기 위한 설정이 필요함

- **도커의 네트워크 드라이버**
    - 컨테이너가 외부 세계 또는 다른 컨테이너와 통신하는 방식을 결정함
    - 가장 많이 쓰이는 3가지 기본 드라이버: **Bridge, Host, None**

        - **Bridge 드라이버 (기본값)**
            - 도커를 설치하면 기본적으로 제공되는 네트워크 방식
            
                - **브리지 네트워크 (Bridge Network):**
                    - 도커의 기본 네트워크 방식
                    - 같은 브리지 안에 있는 컨테이너들은 서로의 컨테이너 이름을 호스트 이름처럼 사용하여 통신할 수 있음
                        - **주의**
                            - 컨테이너 내부에서 localhost는 호스트 PC가 아니라 '자기 자신(컨테이너)'을 가리키므로,
                            - DB 접속 시에는 localhost 대신 db 같은 컨테이너 이름을 써야 함
                    - 사용 예시 (사용자 정의 네트워크와 컨테이너 간 통신)
                        - 컨테이너끼리 IP가 아닌 이름으로 서로를 찾을 수 있게 함
                        - 핵심: web-server 컨테이너 내부 설정 파일에서 DB 주소를 localhost가 아닌 db-server라는 이름으로 입력하면 도커가 자동으로 연결

                        ```bash
                        # 1. 'my-network'라는 이름의 브리지 네트워크 생성
                        docker network create my-network

                        # 2. DB 컨테이너를 네트워크에 포함하여 실행
                        docker run -d --name db-server --network my-network -e MYSQL_ROOT_PASSWORD=pass mysql

                        # 3. 웹 서버 컨테이너에서 DB 서버의 이름을 주소로 사용 가능
                        docker run -d --name web-server --network my-network nginx
                        ```

            - 컨테이너들이 호스트 내부의 가상 브리지(주로 `docker0`)에 연결되어 서로 통신

            - **동작 원리:**
                - 각 컨테이너는 자신만의 사설 IP를 할당받음
                - 외부와 통신할 때는 호스트의 IP를 통해 나가는 **NAT(Network Address Translation)** 방식을 사용함
            - **특징:**
                - 컨테이너 간의 논리적인 격리를 제공
                - 필요한 경우에만 포트 포워딩(`-p`)을 통해 외부로 노출

                    - **포트 포워딩 (-p 옵션):**
                        - 호스트의 포트와 컨테이너의 포트를 연결하는 다리
                            - 예: -p 8080:80 🡲 호스트의 8080 포트로 들어오는 요청을 컨테이너의 80 포트로 전달함
                        - 사용 예시
                            - 외부 브라우저에서 컨테이너 내부 서비스에 접속할 수 있도록 길을 열어줌
                            - 확인: 브라우저에서 http://localhost:8080 접속 시 Nginx 화면이 뜸

                            ```bash
                            # 호스트의 8080 포트로 접속하면 컨테이너의 80 포트로 전달
                            docker run -d \
                                -p 8080:80 \
                                --name my-nginx \
                                nginx
                            ```

            - **사용 사례:**
                - 가장 일반적인 웹 서버, DB 등 대부분의 서비스 운영 시 사용함

        - **Host 드라이버**
            - 컨테이너가 호스트의 네트워크 환경을 그대로 공유함
            - 컨테이너만의 독립된 네트워크 네임스페이스를 갖지 않음

            - **동작 원리:**
                - 컨테이너 내부에서 80번 포트를 열면, 별도의 포트 포워딩(`-p`) 없이도 호스트의 80번 포트가 즉시 열림
            - **특징:**
                - **성능:** 네트워크 주소 변환(NAT) 과정이 없으므로 속도가 가장 빠름
                - **포트 충돌:** 호스트에서 이미 80번 포트를 쓰고 있다면 컨테이너를 실행할 수 없음
            - **사용 사례:** 
                - 네트워크 성능이 극도로 중요하거나, 아주 많은 포트를 열어야 하는 특수 애플리케이션(예: 실시간 미디어 스트리밍)에 적합

        - **None 드라이버**
            - 네트워크를 사용하지 않는 상태
            - 컨테이너에 로컬 루프백(`lo`, 127.0.0.1) 인터페이스만 생성됨

            - **동작 원리:**
                - 컨테이너에 외부로 연결되는 네트워크 인터페이스를 아예 할당하지 않음
            - **특징:**
                - 외부와의 연결이 완벽하게 차단되므로 보안상 매우 안전함
                - 오직 `docker exec`를 통해서만 내부 명령어를 실행할 수 있음
            - **사용 사례:**
                - 네트워크가 필요 없는 단순 계산 작업
                - 보안이 매우 중요한 데이터 처리(Batch 작업)
                - 커스텀 네트워크 드라이버를 직접 구현하기 위한 베이스 단계

        > **요약 및 비교**
        > <div class="info-table">
        > <table>
        >     <thead>
        >         <th style="width: 150px;">드라이버</th>
        >         <th style="width: 150px;">격리 수준</th>
        >         <th style="width: 150px;">성능</th>
        >         <th style="width: 220px;">외부 접속 방법</th>
        >         <th style="width: 150px;">비고</th>
        >     </thead>
        >     <tbody>
        >         <tr>
        >             <td class="td-rowheader">Bridge</td>
        >             <td>높음</td>
        >             <td>보통</td>
        >             <td>포트 포워딩 (`-p`) 필요</td>
        >             <td>기본 드라이버</td>
        >         </tr>
        >         <tr>
        >             <td class="td-rowheader">Host</td>
        >             <td>없음</td>
        >             <td>매우 높음</td>
        >             <td>호스트 포트 직접 사용</td>
        >             <td>포트 충돌 주의</td>
        >         </tr>
        >         <tr>
        >             <td class="td-rowheader">None</td>
        >             <td>완벽 격리</td>
        >             <td>N/A</td>
        >             <td>접속 불가</td>
        >             <td>폐쇄망 작업용</td>
        >         </tr>
        >     </tbody>
        > </table>
        > </div>
        > - 도커 네트워크 드라이버를 선택할 때의 핵심은 **격리(Isolation)와 성능(Performance) 사이의 균형**
        > - 일반적인 개발 및 운영 환경이라면 **Bridge** 드라이버를 사용하여 포트 포워딩으로 관리하는 것이 가장 안전하고 표준적인 방법임
        {: .common-quote}

        > - **한눈에 보는 옵션 구분**
        >   - -v [호스트경로]:[컨테이너경로]: 파일 시스템을 연결할 때 (Volume/Mount)
        >   - -p [호스트포트]:[컨테이너포트]: 네트워크 입구를 열 때 (Publish)
        >   - --network [네트워크명]: 컨테이너가 소속될 망을 지정할 때
        {: .common-quote}


### 2.8 도커를 사용하는 이유 (장점)

- **환경 일관성**
    - 개발, 테스트, 운영 환경을 100% 동일하게 유지 🡲 배포 사고 감소

- **자원 효율성**
    - 하드웨어 가상화 없이 OS 커널 공유 🡲 시스템 리소스를 적게 소모

- **마이크로서비스(MSA) 최적화**
    - 서비스를 작은 단위로 쪼개어 배포하고 확장하기에 용이

- **CI/CD 가속화**
    - 코드 빌드부터 배포까지의 과정을 컨테이너 기반으로 표준화 🡲 자동화 효율 극대화

### 2.9 컨테이너 vs 가상 머신(VM)

- 도커가 기존 가상화 기술보다 가볍고 빠른 이유는 **커널 공유** 방식에 있음

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">구분</th>
        <th style="width: 300px;">가상 머신 (VM)</th>
        <th style="width: 350px;">도커 컨테이너</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">구조</td>
            <td>하이퍼바이저 위에 각자 Guest OS 탑재</td>
            <td>호스트 OS 위에서 도커 엔진을 통해 커널 공유</td>
        </tr>
        <tr>
            <td class="td-rowheader">성능</td>
            <td>OS 부팅이 필요해 무겁고 느림</td>
            <td>프로세스 실행 수준으로 매우 가볍고 빠름</td>
        </tr>
        <tr>
            <td class="td-rowheader">용량</td>
            <td>기가바이트(GB) 단위</td>
            <td>메가바이트(MB) 단위</td>
        </tr>
    </tbody>
</table>
</div>

<div class="insert-image">
    <img src="/materials/devtools/images/S13-01-01-01_01-002_Docker_vs_VM.png" style="width: 70%;">
</div>


> - **요약**
>   - **Dockerfile**이라는 레시피를 통해 **이미지**를 작성
>   - **이미지**는 애플리케이션 실행에 필요한 모든 것이 담긴 압축 파일이며, 결코 변하지 않음
>   - 이 이미지를 실행(run)하면 **컨테이너**가 됨
>   - **컨테이너**는 이미지 위에 얇은 '쓰기 가능 층'을 얹어 독립된 환경에서 돌아가는 실제 서비스임
{: .summary-quote}


## 3. 도커 설치

### 3.1 Windows 환경

> - 운영체제 수준에서 컨테이너를 실행할 수 있는 환경(Linux 커널)을 준비하는 과정이 필요함 🡲 WSL 2(Windows Subsystem for Linux 2)
>   - 과거에는 가상 머신(Hyper-V)을 기반으로 작동하여 느리고 무거웠지만, 최근에는 **WSL 2(Windows Subsystem for Linux 2)** 기술을 활용하여 리눅스와 거의 동일한 성능과 통합성을 제공함
{: .common-quote}

- **1단계: Windows 시스템 요구 사항 확인**
    - 설치 전, 내 PC가 WSL 2 및 Docker Desktop을 실행할 수 있는 환경인지 확인
        - **OS 버전:** Windows 11 (64비트) 또는 Windows 10 (64비트, Home/Pro, 빌드 19041 이상)
        - **하드웨어:** 4GB 이상의 RAM, 바이오스(BIOS)에서 CPU 가상화 기능(VT-x/AMD-v) 활성화 필수
    - **확인 방법:**
        - 터미널(PowerShell/CMD)에서 `winver` 명령어를 입력하여 빌드 번호를 확인
        - CPU 가상화는 작업 관리자의 성능 탭에서 확인할 수 있음

- **2단계: WSL 2 (Windows Subsystem for Linux 2) 설치**
    - Docker Desktop은 리눅스 커널을 기반으로 작동 🡲 Windows 내에 실제 리눅스 커널을 가볍게 띄워주는 WSL 2가 먼저 설치되어 있어야 함

    1. **관리자 권한**으로 PowerShell 또는 Windows 터미널을 실행
    2. 아래 명령어를 입력하여 WSL과 기본 리눅스 배포판(Ubuntu)을 한 번에 설치

        ```powershell
        wsl --install
        ```

    3. 설치가 완료되면 반드시 컴퓨터를 다시 시작(Reboot)함
    4. 재부팅 후 자동으로 리눅스 터미널이 열리며 초기 사용자 이름과 비밀번호를 설정

    - **팁:** 만약 이미 WSL을 쓰고 있다면, `wsl --update` 명령으로 커널 버전을 최신으로 유지하는 것이 좋음

- **3단계: Docker Desktop for Windows 설치 파일 다운로드 및 실행**
    - Windows 환경에서 도커를 편리하게 관리할 수 있게 해주는 GUI 도구인 Docker Desktop 설치

    1. [Docker 공식 홈페이지 다운로드 페이지](https://www.docker.com/products/docker-desktop/) 접속 🡲 **[Download for Windows]** 버튼
    2. 다운로드한 설치 파일(`Docker Desktop Installer.exe`) 실행
    3. **중요**
        - 설치 옵션 창에서 **[Use WSL 2 instead of Hyper-V (recommended)]** 항목이 체크되어 있는지 확인하고 [OK]를 클릭
        - Hyper-V보다 성능이 훨씬 좋음
    4. 설치가 완료되면 [Close and restart] 버튼을 클릭하여 시스템을 다시 시작

- **4단계: Docker Desktop 초기 설정 및 통합 확인**
    - 재부팅 후 도커를 사용할 준비가 되었는지 확인 후, WSL 2 리눅스 환경과 연동

    1. 컴퓨터가 켜지면 Docker Desktop이 자동으로 실행됨
        - 최초 실행 시 서비스 약관 동의가 필요할 수 있음
    2. 시스템 트레이(시계 옆)에 도커 아이콘(고래 모양)이 생기고, 녹색(Running) 상태로 변할 때까지 대기
    3. **WSL 2 통합 확인:**
        - Docker Desktop 설정(톱니바퀴) > [Resources] > [WSL Integration] 메뉴로 이동
        - [Enable integration with my default WSL distro]가 체크되어 있는지 확인
        - 아래 배포판 목록에서 내가 사용하는 리눅스(예: Ubuntu)가 켜져 있는지 확인하고 [Apply & restart]를 클릭
            - 이 과정이 있어야 리눅스 터미널 안에서도 도커 명령어를 쓸 수 있음

- **5단계: 설치 성공 여부 테스트**
    - 도커가 제대로 작동하는지 터미널에서 확인

    1. PowerShell, CMD, 리눅스(Ubuntu) 터미널을 실행
    2. 도커 버전 확인

        ```bash
        docker --version
        ```

    3. 도커 공식 테스트 이미지를 실행하여 설치 성공여부 확인

        ```bash
        docker run hello-world
        ```
        - 이 명령어를 입력했을 때, 인터넷에서 이미지를 다운로드하고, "Hello from Docker!"라는 환영 메시지가 터미널에 출력되면 성공

- **요약: 설치 흐름도**

    ```text
    BIOS 가상화 활성화 ──▶ WSL 2 설치 (--install) ──▶ 재부팅 ──▶ Docker Desktop 설치
    ──▶ 재부팅 ──▶ WSL Integration 설정 ──▶ docker run hello-world 완료!
    ```

### 3.2 Linux 환경

> - 도커(Docker)는 본질적으로 **리눅스 커널(Linux Kernel) 기술**을 기반으로 만들어진 도구
> - 따라서 Windows나 macOS처럼 중간에 가상화 레이어(WSL 2나 가상 머신)를 거치지 않고, **리눅스 운영체제 위에서 직접(Native) 구동될 때 가장 빠르고 강력한 성능**을 발휘함
> - 리눅스 환경에서는 GUI 도구인 Docker Desktop 대신, 핵심 엔진인 Docker Engine(도커 엔진)을 설치하여 서버 환경에 최적화된 방식으로 운영하는 것이 글로벌 표준
<: .common-quote>


- **1단계: 기존 구버전 삭제 (충돌 방지)**
    - 시스템에 혹시 남아있을지 모르는 구버전 도커 패키지(`docker`, `docker-engine`, `docker.io` 등) 제거
        - 새 설치 시 발생할 수 있는 충돌 방지

    ```bash
    sudo apt-get remove docker docker-engine docker.io containerd runc
    ```

    - 새로 설치한 클린 OS 상태라면 이 단계는 건너뛰어도 좋음


- **2단계: 필수 패키지 및 리포지토리 설정**
    - 도커 공식 저장소(Repository)로부터 안전하게 패키지를 다운로드하기 위해 필요한 암호화 및 네트워크 관련 도구들 설치
    - 도커의 공식 GPG 보안 키를 시스템에 등록

    ```bash
    # 1. 패키지 인덱스 업데이트 및 필수 도구 설치
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    # 2. 도커 공식 GPG 키를 저장할 디렉토리 생성 및 키 다운로드
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # 3. 도커 공식 apt 저장소(Repository)를 시스템 소스 리스트에 등록
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

- **3단계: Docker Engine 설치**
    - 저장소 등록이 완료되었으면, 패키지 목록을 한 번 더 갱신한 뒤
    - 실제 도커 엔진과 컨테이너 런타임, 그리고 최신 다중 컨테이너 관리 도구인 **Docker Compose 플러그인**을 한 번에 설치

    ```bash
    # 1. 등록된 도커 저장소 반영을 위해 업데이트
    sudo apt-get update

    # 2. 도커 엔진 및 핵심 구성요소 통합 설치
    sudo apt-get install -y docker-ceil docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

    - **`docker-ce`**: 도커 엔진(데몬)의 본체
    - **`docker-ce-cli`**: 터미널에서 도커 명령어를 제어하는 클라이언트 도구
    - **`docker-compose-plugin`**: 과거 별도로 설치하던 `docker-compose` 명령어 대신, 최신 도커 표준인 **`docker compose`**(띄어쓰기 형식)를 지원하는 플러그인


- **4단계: 서비스 활성화 및 권한 설정 (중요)**
    - 리눅스 시스템이 부팅될 때 도커 엔진이 자동으로 켜지도록 설정
    - 매번 `sudo` 명령어를 붙여야 하는 번거로움을 해결하기 위한 필수 설정

    ```bash
    # 1. 도커 서비스 시작 및 부팅 시 자동 실행 설정
    sudo systemctl start docker
    sudo systemctl enable docker

    # 2. 현재 로그인한 사용자($USER)를 docker 그룹에 추가 (sudo 생략 위함)
    sudo usermod -aG docker $USER
    ```

    > - **⚠️ 주의:**
    >   - 그룹 변경 사항을 현재 터미널 세션에 즉시 반영하려면 `newgrp docker` 명령을 입력하거나, **리눅스 세션을 로그아웃한 후 다시 로그인(또는 SSH 재접속)**해야 `sudo` 없이 도커 명령어를 사용할 수 있음


- **5단계: 설치 완료 검증**
    - 설치가 완벽히 끝났는지 버전을 확인하고 테스트 컨테이너 구동

    ```bash
    # 1. 도커 엔진 버전 확인 (sudo 없이 작동하는지 체크)
    docker --version

    # 2. 도커 컴포즈 버전 확인
    docker compose version

    # 3. 최종 연동 테스트
    docker run hello-world
    ```

    - 터미널에 환영 메시지("Hello from Docker!")가 출력되면 리눅스 원격 서버나 로컬 PC에 네이티브 도커 환경 구축이 완료된 것


> - **리눅스 환경만의 아키텍처적 장점**
>   - Windows나 macOS 환경에서는 도커를 돌리기 위해 내부에 경량 가상 머신을 띄우고 그 안의 리눅스 커널을 빌려 쓰는 간접적인 방식을 취함
>   - 반면 리눅스 네이티브 환경에서는 호스트의 실제 리눅스 커널 위에 **도커 엔진(데몬)이 프로세스 관리자 형태로 직접 상주**함.
>   - 컨테이너들은 어떠한 가상화 기술도 거치지 않고 호스트 커널의 `Namespaces`와 `cgroups` 제어를 직접 받기 때문에,
>   - **I/O 병목이 전혀 없는 제로(0)에 가까운 오버헤드로 완벽한 성능**을 보장받음
>   - 기업의 운영 서버와 고성능 AI 모델 학습/서빙 환경에서 무조건 리눅스 서버를 채택하는 이유가 바로 여기에 있음
{: .summary-quote}