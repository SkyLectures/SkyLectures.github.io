---
layout: page
title:  "Airflow 개요"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-06-01-01_01-AirflowOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. Apache Airflow 개요

### 1.1 정의 및 핵심 개념

- **Apache Airflow란?**
    - 복잡한 데이터 파이프라인(Data Pipeline)과 워크플로우(Workflow)를 효율적으로 개발, 예약(Scheduling), 모니터링하기 위해 만들어진 오픈소스 워크플로우 관리 플랫폼(Workflow Management Platform)
    - 간단히, 워크플로우(Workflow)를 작성, 스케줄링 및 모니터링하는 오픈소스 플랫폼


- **Apache Airflow의 핵심 개념**
    - Airflow의 가장 큰 특징은 "Configuration as Code (코드로 작성하는 설정)"    
    - XML이나 GUI 기반의 툴과 달리, 모든 워크플로우를 **Python 코드로 정의** 🡲 버전 관리, 테스트, 유지보수가 용이하며 유연성과 확장성을 극대화
    - Airflow의 핵심은 단순히 코드를 실행하는 것이 아니라, 복잡하게 얽힌 수많은 작업(Job)들을 **지능적으로 관리하고 시각화**하는 데 있음<br><br>


    - **DAG (Directed Acyclic Graph, 방향성 비순환 그래프)**
        - 워크플로우를 구성하는 전체적인 뼈대
        - 워크플로우를 구성하는 태스크(Task)들의 실행 순서와 의존 관계를 정의한 비순환 유향 그래프
        - Python 코드로 정의되므로 버전 관리 및 협업에 유리함
        - Airflow에서 모든 작업 흐름은 DAG로 표현됨

        - **Directed (방향성):**
            - 태스크(Task)들이 실행되는 명확한 순서와 방향이 존재함 (A 🡲 B)
        - **Acyclic (비순환):**
            - 루프(Loop)나 순환 구조를 가질 수 없음
            - 한 번 실행된 태스크로 다시 되돌아가는 경로가 존재하지 않아 교착 상태(Deadlock)를 방지
                - A가 B를 실행하고, 다시 B가 A를 실행하는 순환 구조는 허용되지 않음
                - 무한 루프 방지와 명확한 선후 관계 보장을 위함
        - **Graph (그래프):** 노드(태스크)와 엣지(의존성)로 구성된 구조체

    - **Task (태스크)**
        - DAG 내부의 가장 기본적이고 독립적인 실행 단위
        - 하나의 작업(예: 데이터 다운로드, SQL 쿼리 실행, 이메일 전송 등)을 의미
        - 크게 두 가지 형태로 정의됨
            - **Operators (오퍼레이터):** 실제로 작업을 수행하기 위해 미리 템플릿화된 클래스
                - *PythonOperator:* Python 함수를 실행할 때 사용
                - *BashOperator:* Bash 명령어나 스크립트를 실행할 때 사용
                - *MySqlOperator / PostgresOperator:* 특정 RDBMS에 쿼리를 전송할 때 사용

            - **Sensors (센서):**
                - 특정 조건(파일 생성, 특정 시간, API 응답 등)이 충족될 때까지 기다렸다가 다음 태스크를 트리거하는 특수한 형태의 오퍼레이터

        - Operator vs Task: 개념의 차이
            - **Operator (오퍼레이터):**
                - 작업을 수행하기 위한 **'템플릿'** 또는 '틀'
                    - 예: PythonOperator, BashOperator, S3ToRedshiftOperator
            - **Task (태스크):**
                - 오퍼레이터를 실제로 구체화하여 DAG에 배치한 '실행 단위'
                - 하나의 DAG 안에 여러 개의 태스크가 존재
                - 태스크는 오퍼레이터의 인스턴스

    - **Task Instance (태스크 인스턴스)**
        - DAG가 실행될 때(DAG Run), 그 안에 속한 개별 Task가 실제로 물리적인 자원을 할당받아 실행되는 상태를 의미
        - '대기(Queued)', '실행 중(Running)', '성공(Success)', '실패(Failed)' 등의 고유한 생명주기(State)를 가짐

- **AI 서비스에서의 역할:**
    - 데이터 수집 🡲 전처리 🡲 모델 학습 🡲 평가 🡲 서빙으로 이어지는 **ML Pipeline**의 각 단계를 자동화하고,
    - 특정 단계 실패 시 재시도(Retry)나 알림을 처리함


### 1.2 Apache Airflow의 개발 역사

- **2014년 ~ 2015년: 에어비앤비(Airbnb)에서 개발 시작**
    - **2014년 10월:**
        - 에어비앤비의 엔지니어였던 맥심 보슈맹(Maxime Beauchemin)
        - 사내의 복잡해지는 데이터 파이프라인과 스케줄링 문제를 해결하기 위해 처음으로 개발을 시작
        - 당시 기존 툴(Crontab, Oozie 등)의 경직성을 해결하기 위해 "코드로 정의하는 워크플로우(Configuration as Code)"라는 패러다임을 최초로 제안
    - **2015년 6월:**
        - 에어비앤비는 오픈소스 생태계 발전을 위해 Airflow를 **오픈소스로 공식 선언**하고 깃허브(GitHub)에 공개

- **2016년 ~ 2018년: 아파치 인큐베이터 진입**
    - **2016년 3월:**
        - 프로젝트의 확장성과 중립성을 확보하기 위해 **아파치 소프트웨어 재단(ASF)의 인큐베이터 프로젝트**로 등록
        - 이 시기에 에어비앤비뿐만 아니라 구글, 인튜이트(Intuit), 넷플릭스 등 글로벌 테크 기업들이 컨트리뷰터로 대거 참여하면서 커뮤니티가 급격히 성장
        - 구글 클라우드(GCP)가 자사의 RAG 및 데이터 오케스트레이션 관리형 서비스인 **Cloud Composer**의 기반 엔진으로 Airflow를 채택하면서 엔터프라이즈 시장에서의 신뢰도가 급상승

- **2019년: 탑레벨 프로젝트 승격 및 v1.10 안정화**
    - **2019년 1월:**
        - 아파치 재단:
            - Airflow의 기술적 성숙도와 커뮤니티 완성도를 인정
            - 재단의 최고 등급인 탑레벨 프로젝트(Top-Level Project, TLP)로 승격
        - 이 시기에 발매된 **v1.10 계열** 버전들은 전 세계 수많은 기업의 프로덕션 환경에 표준 데이터 파이프라인 관리 도구로 대폭 도입됨
        - 쿠버네티스 엑세큐터(KubernetesExecutor)의 도입 등 컨테이너 환경과의 결합도 이 시기에 본격화

- **2020년 말 ~ 현재: Airflow 2.0 세대와 현대적 마이그레이션**
    - **2020년 12월:**
        - 아키텍처를 전면 개편한 **Airflow 2.0** 출시
        - 고질적인 병목이었던 스케줄러를 **멀티 스레드/액티브-액티브 분산 스케줄러 구조**로 변경하여 성능을 극대화
        - 기존의 무거운 오퍼레이터 위주 방식 🡲 Python 함수 자체를 태스크로 파싱하는 **TaskFlow API** 도입 🡲 개발 생산성 혁신
        - 대규모 UI/UX 개편 및 웹서버 성능 최적화
    - **현재:**
        - 이후 대대적인 패키지 파편화 정리(Providers 패키지 분리), 보안 취약점 개선, 실시간 이벤트 기반 트리거(Deferrable Operators) 도입 등 현대적인 데이터 스택(Modern Data Stack)의 필수 관제탑으로 지속해서 진화 중


### 1.3 Airflow 아키텍처 및 구성 요소

- Airflow는 확장성과 모니터링을 위해 여러 분산 컴포넌트로 구성됨

    <div class="insert-image" style="text-align: center;">
        <img src="/materials/devtools/images/S13-06-01-01_01-001_AirflowArchitecture.jpg" style="width: 90%;"><br>
        출처: Shutterstock
    </div>


    - **Scheduler (스케줄러):**
        - Airflow의 심장부
        - 모든 DAG와 태스크의 의존성을 상시 감시
        - 실행 조건이 충족된 태스크를 큐(Queue)에 밀어 넣어 실행을 트리거

    - **Web Server (웹 서버):**
        - 사용자가 DAG의 상태를 확인하고, 수동으로 실행하거나 로그를 점검할 수 있는 GUI 제공
        - 수동 트리거, 에러 로그 확인, Task 재시도(Clear) 등을 편리하게 수행할 수 있음

    - **Metadata Database (메타데이터 DB):**
        - DAG 구조, 실행 이력, Task 상태, 연결 정보(Connections) 등 Airflow의 모든 실행 이력, 상태 값을 저장하는 중앙 저장소
        - 주로 PostgreSQL, MySQL 등의 RDBMS 활용
        
    - **Executor (엑세큐터):**
        - 태스크가 실제로 '어떻게' 실행될지 결정하는 메커니즘
            - 단일 머신에서 실행할지(Sequential/Local), 여러 서버에 분산하여 실행할지(Celery/Kubernetes)를 정의<br><br>
        - SequentialExecutor:
            - 단일 스레드로 하나씩 순차 실행 (테스트용)
        - LocalExecutor:
            - 스케줄러와 같은 장비에서 멀티프로세싱으로 병렬 실행 (소규모)
        - CeleryExecutor / KubernetesExecutor:
            - 분산 환경에서 다수의 워커(Worker) 노드에 작업을 분배하여 대규모 병렬 처리 (운영 환경)

    - **Worker (워커):**
        - Executor가 Celery나 Kubernetes 기반일 때, 실제로 분배된 태스크의 로직을 직접 할당받아 수행하는 물리적 주체


- **Airflow의 계층**
    - Airflow는 설계상 [오퍼레이터 🡲 태스크 🡲 태스크 인스턴스]라는 3단계 수직 계층을 가짐

        ```
        [ 1단계: 템플릿 ]  Operator (클래스 정의)
                                │
                                ▼ (코드 단에서 객체화)
        [ 2단계: 선언부 ]  Task (오퍼레이터의 인스턴스 / DAG 구조 정의)
                                │
                                ▼ (특정 시간에 스케줄러가 실행 / 런타임)
        [ 3단계: 실행부 ]  Task Instance (태스크 + 특정 실행 시간 / 물리적 실행 상태)

        ```

    - 일반적인 프로그래밍 개념 및 일상적인 비유와 매핑하면..

        <div class="info-table">
        <table>
            <thead>
                <th style="width: 150px;">계층</th>
                <th style="width: 150px;">Airflow 용어</th>
                <th style="width: 200px;">프로그래밍(OOP) 비유</th>
                <th style="width: 150px;">일상 비유</th>
                <th style="width: 330px;">특징</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">1단계</td>
                    <td class="td-left">Operator</td>
                    <td class="td-left">Class (클래스)</td>
                    <td class="td-left">붕어빵 틀</td>
                    <td class="td-left">작업을 어떻게 할지 정의한 <b>설계도</b> 그 자체</td>
                </tr>
                <tr>
                    <td class="td-rowheader">2단계</td>
                    <td class="td-left">Task</td>
                    <td class="td-left">Object (객체)</td>
                    <td class="td-left">메뉴판에 등록된 붕어빵</td>
                    <td class="td-left">코드 단에서 변수에 할당하여 <b>DAG 파일에 등록한 상태</b></td>
                </tr>
                <tr>
                    <td class="td-rowheader">3단계</td>
                    <td class="td-left">Task Instance</td>
                    <td class="td-left">Runtime Memory (메모리 적재)</td>
                    <td class="td-left">2026년 7월 7일 16시에 구워낸 <b>'그 붕어빵'</b></td>
                    <td class="td-left">특정 날짜/시간(Logical Date)을 부여받아 <b>실제 실행되는 물리적 실체</b></td>
                </tr>
            </tbody>
        </table>
        </div>

    - **무엇이 다른가? (핵심 차이점 비교)**
        - **Task (태스크)**
            - **어디에 존재하는가?:**
                - 작성하는 `dag.py` **코드 소스 파일 내부**에 존재
            - **언제 생성되는가?:**
                - 파이썬 스크립트가 파싱되어 웹서버나 스케줄러가 DAG 그래프를 그릴 때 정의됨
            - **역할:**
                - "이 작업은 `BashOperator`를 쓸 거고, command는 `echo 'Hello'`이며, 앞뒤에 어떤 태스크와 연결된다"라는
                - **정적(Static)인 의존성 지도** 역할
            - **문맥적 인스턴스:**
                - "태스크는 오퍼레이터의 인스턴스"라고 부르는 것 🡲 파이썬 코드상에서 `task_A = BashOperator(...)`와 같이
                - **오퍼레이터 클래스의 생성자를 호출하여 '객체화'했다는 뜻**

        - **Task Instance (태스크 인스턴스)**
            - **어디에 존재하는가?:**
                - Airflow의 중앙 **메타데이터 데이터베이스(DB)** 및 **실제 워크(Worker)의 메모리/CPU 공간**에 존재
            - **언제 생성되는가?:**
                - 스케줄러가 작동하여 "오늘 치(예: 2026-07-07) 배치를 돌려라!" 하고 DAG Run(정해진 시간의 실행 회차)을 트리거하는 순간 생성됨
            - **역할:**
                - "오늘 오후 4시에 실행된 `task_A`"라는 명확한 **시공간적 컨텍스트**를 가짐
                - 따라서 이 단계에 와서야 비로소 **'대기(Queued)', '실행 중(Running)', '성공(Success)', '실패(Failed)'** 같은 동적(Dynamic)인 생명주기(State)를 가질 수 있음


- **Airflow의 핵심 동작 메커니즘**
    1. **DAG 작성:**
        - `./dags` 디렉토리에 Python 파일을 생성하여 워크플로우를 코딩

    2. **Parsing:**
        - 스케줄러가 작성된 Python 코드를 읽어 DAG 구조를 파악

    3. **Scheduling:**
        - `schedule_interval` 설정을 통해 주기적(예: 매일 자정)으로 실행되도록 설정
        - `start_date`와 `schedule_interval`을 계산하여 실행 시점이 된 태스크를 'Scheduled' 상태로 바꿈

    4. **Queuing:**
        - 실행 가능한 태스크를 Executor에게 전달
        - Executor는 이를 큐에 넣음

    5. **Execution:**
        - 워커가 큐에서 태스크를 가져와 실제 로직 실행

    6. **State Update:**
        - 실행 결과(Success/Failed)를 메타데이터 DB에 업데이트
        - 웹 서버는 이를 화면에 출력

    7. **모니터링:**
        - 웹 UI를 통해 각 태스크의 성공/실패 여부, 로그, 실행 시간을 실시간으로 확인

    8. **연동:**
        - 다양한 Operator(Python, Bash, SQL, Docker, Kubernetes 등)를 사용하여 외부 시스템과 상호작용

<br>

> - **요약**
>   - 코드에 `task_1 = PythonOperator(...)`라고 적는 행위 🡲 오퍼레이터라는 설계도를 가지고 Task(정적 객체)를 만든 것
>   - 배치가 돌 때, 스케줄러가 DB에 **"2026년 7월 7일 자 `task_1` 실행해라"** 하고 레코드를 파싱하여 물리 자원을 할당하는 순간 생성되는 것이 Task Instance(동적 실행 상태)
>   - 즉, **Task**는 1개만 정의되어 있어도, 배치가 매일 도는 스케줄이라면 **Task Instance**는 매일 1개씩(한 달이면 30개) 쌓이게 됨
{: .common-quote}


### 1.4 Airflow의 주요 장점과 한계점

- **장점**
    - **강력한 확장성(Dynamic Pipeline):**
        - Python 언어의 생태계를 그대로 사용
            - 수많은 라이브러리가 미리 구현되어 있어, AWS, GCP, Azure, Slack, Docker 등과의 연동이 매우 쉬움
            - 모든 라이브러리를 임포트하여 커스텀 오퍼레이터나 플러그인을 무한히 확장할 수 있음
        - 반복문을 통해 수백 개의 태스크를 동적으로 생성할 수 있음

    - **직관적인 UI와 모니터링:**
        - 웹 GUI가 매우 직관적
        - 파이프라인의 병목 구간이나 에러 발생 지점의 로그를 클릭 한 번으로 즉시 추적 가능        

    - **유연한 에러 핸들링:**
        - 특정 단계에서 에러가 나면 자동으로 재시도하도록 설정하거나, 실패 즉시 담당자에게 메시지를 보낼 수 있음
        - 태스크 실패 시 재시도 횟수(`retries`), 재시도 간격(`retry_delay`), 실패 시 알림(Slack, Email 트리거) 등을 DAG 단에서 정교하게 제어 가능

    - **Backfill:**
        - 과거 특정 시점의 데이터를 다시 처리해야 할 때, 코드 수정 없이 명령 하나로 과거 날짜의 작업들을 일괄 실행할 수 있음

- **한계점 및 주의사항 (ETL vs ELT)**
    - **데이터 이동 툴이 아님:**
        - Airflow는 작업을 제어하고 순서를 조율하는 '오케스트레이터(Orchestrator)'
        - 자체적으로 기가바이트 단위의 데이터를 가공하고 실어나르는 '데이터 처리 엔진'이 아님

    - **무거운 데이터 처리 배제:**
        - 대용량 데이터를 처리할 때는 Airflow 워커 내부에서 직접 처리하는 방식을 피해야 함
        - Airflow는 **Spark, Databricks, BigQuery, Snowflake 같은 외부 연산 인프라에 명령(쿼리/트리거)만 내리고, 그 작업이 끝났는지 제어하는 신호등 역할**로 쓰는 것이 정석


### 1.5 주요 용어 정리

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">용어</th>
        <th style="width: 780px;">설명</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">Execution Date</td>
            <td class="td-left">DAG가 실행되기로 예약된 논리적 시점 (실제 실행 시간과 다를 수 있음)</td>
        </tr>
        <tr>
            <td class="td-rowheader">XComs</td>
            <td class="td-left">태스크 간에 작은 데이터(메시지, 경로 등)를 공유하기 위한 통신 메커니즘</td>
        </tr>
        <tr>
            <td class="td-rowheader">Variables</td>
            <td class="td-left">Airflow 전역에서 공통으로 사용하는 설정값 (ID, 경로 등)</td>
        </tr>
        <tr>
            <td class="td-rowheader">Connections</td>
            <td class="td-left">외부 시스템(DB, 클라우드) 접속 정보 관리 (암호화되어 저장됨)</td>
        </tr>
    </tbody>
</table>
</div>


> - **요약**
>   - **Apache Airflow는 데이터 파이프라인의 '중앙 관제탑'**
>       - 복잡하게 얽혀 있는 엔터프라이즈 환경의 데이터 흐름을 **하나의 방향성 그래프(DAG)로 묶어 자동화**하고,
>           - 예: 매일 새벽 2시에 서비스 DB에서 데이터 추출 🡲 MinIO/S3에 백업 🡲 Spark로 데이터 정제 🡲 VectorDB 및 Data Warehouse에 로드 🡲 마케팅 팀에 완료 알림
>       - **실패 시 가이드라인을 코드로 통제할 수 있게 해주는 데이터 엔지니어링의 핵심 인프라 기술**로 정의
{: .summary-quote}



## 2. Airflow 환경 구축 (Docker 기반)


1. **준비 단계**

    ```bash
    # 작업 디렉토리 생성
    mkdir airflow-docker && cd airflow-docker

    # 공식 docker-compose.yaml 파일 다운로드
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'

    # 필요한 디렉토리 생성
    mkdir -p ./dags ./logs ./plugins ./config
    ```

2. **환경 변수 설정**

    - Airflow에 필요한 유저 ID 정보를 `.env` 파일에 기록

        ```bash
        echo -e "AIRFLOW_UID=$(id -u)" > .env
        ```

3. **서비스 초기화 및 실행**

    ```bash
    # DB 초기화
    docker compose up airflow-init

    # 서비스 실행 (-d는 백그라운드 실행)
    docker compose up -d
    ```

    - docker compose up airflow-init는 파일에 등록된 컨테이너 중에서 airflow-init만 실행한다는 의미
        - airflow-init 컨테이너:
            - Airflow가 정상적으로 돌 수 있도록 기반 인프라 환경을 딱 한 번만 세팅하고 스스로 종료(Exit)되는 일회성(Transient) 컨테이너
            - 내부적으로 다음과 같은 작업을 수행함
                - 메타데이터 DB 스키마 생성 및 마이그레이션 (DB Init):
                    - Airflow는 데이터베이스(PostgreSQL 등)에 태스크 상태와 실행 이력을 저장함
                    - 처음에는 DB가 텅 비어 있으므로,
                    - airflow-init가 진입하여 Airflow 구동에 필요한 수십 개의 테이블 스키마를 자동으로 생성(db init 또는 db upgrade)함
                - 기본 관리자 계정(Admin User) 자동 생성:
                    - 웹 UI(localhost:8080)에 로그인할 때 사용할 기본 유저(ID: airflow / PW: airflow)를 DB 레코드에 적재
                - 호스트 디렉토리 권한 검증 및 세팅:
                    - .env 파일의 AIRFLOW_UID 설정을 기반으로,
                    - 호스트의 ./dags, ./logs, ./plugins 디렉토리에 Airflow 컨테이너가 정상적으로 접근할 수 있는지 파일 시스템 권한을 최종 조율
        - 해당 명령을 수행한 후 결과 확인

            ```text
            airflow-init_1  | User "airflow" created with role "Admin"
            airflow-init_1  | ...
            airflow-init_1  | [SUCCESS] Airflow 기동 환경 초기화 완료.
            airflow-docker_airflow-init_1 exited with code 0
            ```

    - 실행 후 `http://localhost:8080`으로 접속 (기본 계정: `airflow` / `airflow`)

        <div class="insert-image" style="text-align: center; border: solid 1px lightgray;">
            <img src="/materials/devtools/images/S13-06-01-01_01-002_AirflowWebUi.png" style="width: 100%;">
        </div>


## 3. 예제 코드 및 상세 설명

- 간단한 데이터 전처리 및 AI 모델 학습 단계를 가정한 Python 기반 DAG 예제

    ```python
    #//file: "dags/example_ai_pipeline.py"
    # file: dags/example_ai_pipeline.py
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    # 1. 태스크 함수 정의
    def preprocess_data():
        print("데이터 전처리 중... (Cleaning, Normalization)")
        return "Data cleaned"

    def train_model(ti):
        # 이전 태스크의 반환값을 XCom으로 전달받음
        status = ti.xcom_pull(task_ids='preprocess_task')
        print(f"{status} 완료. 모델 학습 시작...")

    # 2. DAG 설정
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': datetime(2026, 5, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    with DAG(
        'ai_service_orchestration_v1',
        default_args=default_args,
        description='간단한 AI 파이프라인 예제',
        schedule='@daily',  # <--- schedule_interval을 schedule로 변경!
        catchup=False
    ) as dag:

        # 3. 태스크 정의 (Operators) - 정확히 4칸 들여쓰기 반영
        preprocess_task = PythonOperator(
            task_id='preprocess_task',
            python_callable=preprocess_data,
        )

        train_task = PythonOperator(
            task_id='train_task',
            python_callable=train_model,
        )

        # 4. 의존성 설정 (순서 정의)
        preprocess_task >> train_task
    ```

    - **예제 코드 설명**
        - **`DAG` 객체:**
            - 워크플로우의 본체
            - `ai_service_orchestration_v1`이라는 ID를 가짐
            - 매일 자정에 실행되도록 설정됨
        - **`PythonOperator`:**
            - Python 함수를 Airflow 태스크로 실행하기 위해 사용
        - **`XCom` (Cross-Communication):**
            - `ti.xcom_pull`을 통해 태스크 간에 데이터를 주고받음
            - 실제 환경에서는 대용량 데이터 대신 경로 정보나 상태값 등을 전달
        - **의존성 (`>>`):**
            - `preprocess_task >> train_task`: "전처리가 성공해야 학습을 시작한다"는 명확한 순서를 보장


> - **활용 팁**
>   - **Idempotency (멱등성):**
>       - 동일한 입력에 대해 항상 동일한 결과가 나오도록 DAG를 설계해야 함
>           - 실패 후 재실행 시 데이터 중복 방지
>   - **Dynamic Task Generation:**
>       - 리스트나 딕셔너리를 활용해 수십 개의 태스크를 반복문으로 자동 생성할 수 있음
>   - **Docker/Kubernetes Operator:**
>       - Airflow 워커 환경에 구애받지 않음
>       - 각 태스크마다 독립적인 컨테이너 환경에서 학습이나 추론을 수행할 수 있어 AI 서비스 구축 시 매우 강력함
{: .common-quote}


## 4. 실습 예제 코드 결과 확인 및 검증 프로세스

- Airflow에서 DAG는 스스로 실행되는 프로그램이 아니라 **스케줄러가 읽어가는 설계도**
- 따라서 실행 및 결과 확인은 로컬 터미널이 아닌 **Airflow Web UI**에서 진행해야 함

- **1단계: DAG 설계도 배포 (호스트 ➔ 컨테이너 전달)**
    1. **로컬 터미널 실행 중단:**
        - 로컬 가상환경 터미널에서 `python example_ai_pipeline.py`로 직접 실행하던 작업을 중단
    2. **파일 복사:**
        - 작성한 `example_ai_pipeline.py` 파일을 Docker Compose 볼륨 마운트가 연결된 **`dags/`** 디렉토리로 복사

    ```bash
    cp example_ai_pipeline.py ~/workspace/airflow/dags/
    ```

    3. **백엔드 직렬화 대기:**
        - 컨테이너 내부의 `DagFileProcessor`가 이 코드를 파싱하여 메타데이터 DB에 JSON 구조로 등록할 때까지 **약 30초~1분** 정도 대기

- **2단계: Web UI 접속 및 파이프라인 활성화**
    1. **웹 콘솔 접속:**
        - 브라우저를 열고 `http://localhost:8080`에 접속 (기본 계정: `airflow` / `airflow`)

    2. **DAG 등록 확인:**
        - 메인 화면 리스트에 `ai_service_orchestration_v1`이라는 이름의 DAG가 새로 추가되었는지 확인

    3. **잠금 해제 (Active):**
        - 처음 등록된 DAG는 일시정지 상태
        - DAG 이름 왼쪽에 있는 **회색 토글 스위치를 클릭하여 파란색(`Active`) 상태**로 전환

- **3단계: 수동 트리거 및 실행 상태 모니터링**
    1. **파이프라인 구동:**
        - DAG 우측 끝에 있는 `Trigger DAG` (재생 버튼 ▶)을 클릭하여 컨테이너 내부 워커들에게 실행 명령을 내림

    2. **그래프 뷰(Graph View) 진입:**
        - 상단 메뉴에서 `Graph` 뷰를 클릭하면 우리가 설계한 의존성 구조를 시각적으로 볼 수 있음

        ```text
        [preprocess_task] ── (성공 시 이동) ──➔ [train_task]
        ```

    3. **태스크 상태 변화 관측:**
        - 스케줄러가 위상 정렬 알고리즘에 따라 각 태스크의 테두리 색상을 실시간으로 변화시킴
            - **연두색 (`Running`):** 컨테이너 내부의 일꾼(Worker)이 Python 오퍼레이터를 실제 실행 중인 상태
            - **진녹색 (`Success`):** 에러 없이 정상적으로 실행 완료된 상태

- **4단계: 콘솔 로그(Log)를 통한 비즈니스 결과 검증**
    - 태스크가 성공적으로 끝났다면,
    - 파이썬 코드 내부의 `print()` 문과 **XCom 데이터 통신**이 완벽히 작동했는지 태스크별 상세 로그를 확인

    1. `preprocess_task` 출력 결과 확인
        - `Graph` 뷰에서 첫 번째 노드인 **`preprocess_task`** 블록을 클릭한 후 **`Log`** 버튼을 선택
        - 로그 덤프 중에서 파이썬 함수가 호출되어 찍은 아래 출력 패턴을 확인

        ```text
        [2026-07-08 17:35:10,123] {logging_mixin.py:115} INFO - 데이터 전처리 중... (Cleaning, Normalization)
        [2026-07-08 17:35:10,125] {python.py:177} INFO - Done. Returned value was: Data cleaned
        ```

        - 마지막 줄을 통해 "Data cleaned"라는 문자열이 다음 태스크로 넘겨주기 위해 메타데이터 DB(XCom)에 안전하게 임시 저장되었음을 알 수 있음

    2. `train_task` 출력 결과 확인
        - 뒤이어 실행된 **`train_task`** 블록을 클릭하고 **`Log`** 버튼을 선택
        - 이전 태스크의 리턴값을 무사히 가로채어 연산에 활용했는지 아래 최종 출력 결과를 확인

        ```text
        [2026-07-08 17:35:12,456] {logging_mixin.py:115} INFO - Data cleaned 완료. 모델 학습 시작...
        ```

    <div class="insert-image" style="text-align: center; border: solid 1px lightgray;">
        <img src="/materials/devtools/images/S13-06-01-01_01-003.png" style="width: 100%;">
    </div>

<br>

> - **[결과 요약]:**
>   - 이 로그가 명확하게 찍혀 있다면,
>   - 로컬에 `airflow` 패키지를 일일이 깔지 않아도 **Docker 격리 인프라가 설계도를 완벽하게 해석하여 분산 실행을 완료**했음이 증명된 것
{: .common-quote}