---
layout: page
title:  "DAG 이해 및 유즈케이스 연구"
date:   2026-07-07 10:00:00 +0900
permalink: /materials/S13-06-02-01_01-AirflowDag
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. DAG의 이해

> - DAG (Directed Acyclic Graph, 방향성 비순환 그래프)
>   - 아파치 에어플로우(Apache Airflow)에서 워크플로우를 설계하는 가장 핵심적인 논리적 단위
>   - 복잡한 데이터 파이프라인의 작업 순서와 의존 관계를 프로그래밍 코드로 추상화한 지도(Map)
{: .common-quote}

### 1.1 DAG를 구성하는 3대 속성

- **Graph (그래프):**
    - 노드(Node)와 간선(Edge)으로 이루어진 구조
        - 노드: 구체적인 작업(Task)을 의미
        - 간선: 작업 간의 실행 순서(Dependency)를 의미

- **Directed (방향성):**
    - 모든 간선에는 화살표가 존재함
    - Task A가 끝나야 Task B가 실행된다는 명확한 인과관계와 데이터 흐름의 방향이 고정되어 있음

- **Acyclic (비순환):**
    - 그래프 내에 루프(Loop)나 순환 경로가 절대 존재할 수 없음
        - 예를 들어
            - `Task A 🡲 Task B 🡲 Task C 🡲 Task A`와 같은 순환 구조가 만들어지면
            - 워크플로우가 무한 루프에 빠져 영원히 종료되지 않기 때문
    - Airflow 엔진은 DAG 파싱 단계에서 이를 에러로 차단


### 1.2 DAG의 이면에 숨겨진 진짜 기술적 난이도

> - DAG 자체의 개념 구조는 단순함
>   - 그러나 그것을 분산 인프라 위에서 안정적으로 실행하고 관리하는 '비하인드 엔진 구조(엔지니어링)'가 매우 복잡함
>   - 이런 이유로 여러 학습 자료들이 개념은 짧게 쓰고 실무 트러블슈팅 위주로 구성되어 있기때문에
>   - DAG 기술이 매우 간단하고 쉬운 것으로만 생각하기 쉬움
{: .common-quote}

- **개념이 단순한 이유: 추상화(Abstraction)의 성공**
    - Apache Airflow가 복잡한 하부 메커니즘을 사용자(개발자)로부터 완벽하게 숨겨주었기 때문(추상화)
    - 개발자는 그저 Python 코드로 `task1 >> task2`라고만 적으면 끝
        - 이 단순함 덕분에 프로그래밍을 조금만 할 줄 알면 "DAG는 별거 없네"라고 느끼기 쉬움
        - **개념이 쉬운 것이 아니라 Airflow가 인터페이스를 극도로 잘 설계한 결과**

- **분량이 적어 보이는 착시: "어플리케이션"과 "인프라"의 경계**
    - 많은 학습 자료들이 DAG를 설명할 때 순서도 수준에서 멈추는 이유
    - 진짜 기술적 난이도가 DAG 그 자체가 아니라 **스케줄러와 워커(Worker) 간의 분산 아키텍처**에서 발생하기 때문<br><br>

- **실무 진입 시 겪는 진짜 기술적 난관**

    > - 만약 DAG가 정말 단순한 기술이라면 데이터 엔지니어들이 밤을 새우며 파이프라인을 고칠 일이 없을 것
    > - DAG의 수학적, 개념적 정의를 이해하는 것은 1시간이면 충분하지만,
    > - 대규모 트래픽과 데이터가 흐르는 분산 시스템 위에서 '죽지 않고 완벽하게 통제되는 DAG'를 코딩하는 것은 시니어 백엔드/인프라 엔지니어의 영역
    {: .common-quote}

    - **상태 동기화 레이턴시 (Race Condition):**
        - 분산 환경에서 1초에 수천 개의 태스크가 쏟아질 때,
        - 메타데이터 DB(PostgreSQL 등)와 스케줄러, 워커 간의 상태(Queued 🡲 Running 🡲 Success)가 미세하게 꼬이면
        - 파이프라인이 멈추거나 중복 실행됨

    - **리소스 격리와 좀비 태스크 (Zombie Tasks):**
        - 대용량 AI 모델을 학습시키는 태스크가 메모리 부족(OOM)으로 OS 단에서 기습적으로 죽었을 때,
        - Airflow 스케줄러는 이를 어떻게 감지하고 자원을 회수할 것인가?<br>
            🡲 Heartbeat 매커니즘, 킵어라이브 이론 등 고도의 시스템 프로그래밍 영역으로 진입

    - **Dynamic DAG Generation (동적 DAG 생성)의 부하:**
        - 실무에서는 설정 파일(유저 정보, DB 테이블 목록 등)을 읽어와서 실시간으로 100개의 DAG를 코드로 자동 생성하는 기법을 사용
        - 이때 `DagFileProcessor`가 수십 초마다 이 코드를 파싱하면서 CPU가 100%를 치는 병목이 발생
        - "코드가 어떻게 컴파일되고 직렬화(Serialization)되는가"에 대한 컴파일러적 이해가 없으면 스케줄러 시스템 전체를 마비시키기 쉬움

    - **분산 트랜잭션과 Idempotency(멱등성) 구현의 한계:**
        - DAG의 선후 관계를 맞추는 건 쉽지만,
        - 중간에 50번째 태스크가 실패해서 재실행(Retry)했을 때,
        - 1번부터 49번까지 이미 DB에 적재된 데이터가 중복 저장되지 않도록
        - 'Upsert'나 'Write-Ahead Log' 기법을 DAG 코드 내에 녹여내는 설계 감각은 고도의 백엔드 아키텍처 역량을 요구함

    - 많은 자료에서 `DAG`라는 단어로는 겉핥기식 개념만 설명
    - 진짜 어려운 내용들은 **"CeleryExecutor 설정", "KubernetesPodOperator 메모리 관리", "XCom 백엔드 최적화"** 같은 다른 키워드 파트에 숨겨둠


### 1.3 DAG의 심층적 이해와 이론적 기반

- **알고리즘 및 그래프 이론 기반의 해석**
    - Airflow의 DAG는 단순히 "작업의 순서도"를 넘어, 컴퓨터 공학의 그래프 이론(Graph Theory)에 엄격히 기반을 두고 있음

    - **위상 정렬 (Topological Sort):**
        - Airflow 스케줄러는 DAG 내부의 Task들을 무작위로 실행하지 않음
        - 의존성 엣지(Edge)를 바탕으로 의존성이 없는 노드부터 순차적으로 정렬하는 **위상 정렬 알고리즘**을 수행
        - 위상 정렬이 가능하다는 것 자체가 그래프 내에 사이클(Cycle)이 없음을 증명함

    - **진입차수(In-degree)와 진출차수(Out-degree):**
        - **In-degree (진입차수):**
            - 한 노드로 들어오는 엣지의 개수
            - 특정 Task의 In-degree가 0이 되는 순간(선행 작업 완료 또는 선행 작업 없음),
                - 스케줄러는 해당 Task를 `Runnable` 상태로 전환함
        - **Out-degree (진출차수):**
            - 한 노드에서 나가는 엣지의 개수
            - 한 Task가 완료되면
                - Out-degree 경로를 따라 후속 Task들의 In-degree를 1씩 감소시키며
                - 파이프라인을 전진시킴

    - **사이클 검사 (Cycle Detection):**
        - Airflow는 사용자가 정의한 Python 코드를 파싱할 때
            - **DFS(깊이 우선 탐색, Depth-First Search)** 알고리즘을 활용하여
            - 그래프 내의 역방향 간선(Back Edge) 존재 여부를 검사함
        - 만약 순환 구조가 발견되면 
            - `AirflowDagCycleException`을 발생시키고
            - 웹 UI에 컴파일 에러를 명시하며
            - 스케줄링 대상에서 제외함

- **Airflow 스케줄러 백엔드에서의 DAG 파싱 메커니즘**
    - DAG는 고정된 설정 파일(JSON, YAML 등)이 아닌 **실행 가능한 Python 코드**
    - 이 특성 때문에 내부 엔진은 다음과 같이 동작함

        - **DAG File Processor의 주기적 루프:**
            - 스케줄러 내부의 `DagFileProcessorManager`는
                - 설정된 주기(기본값 `min_file_process_interval`: 30초)마다
                - `dags/` 폴더 내의 모든 Python 파일을
                - 백그라운드에서 실행(Exec)함

        - **직렬화 (Serialization):**
            - 파싱된 DAG 구조는 매번 파이프라인을 읽는 오버헤드를 줄이기 위해
                - JSON 형태로 직렬화되어
                - 메타데이터 데이터베이스(`serialized_dag` 테이블)에 저장됨
            - 웹 서버는 Python 코드를 직접 읽지 않고 이 DB 테이블을 조회하여 UI를 렌더링함으로써 성능을 보존함

        - **Top-level Code의 제약 (Anti-Pattern 방지):**
            - DAG 파일 내에서 함수나 오퍼레이터 내부가 아닌, 최상위 레벨(Top-level)에 무거운 DB 쿼리나 외부 API 호출 코드의 작성 금지
            - 스케줄러가 수십 초마다 이 코드를 실행하므로, 스케줄러 엔진 자체가 마비되는 병목 현상이 발생함


### 1.4 데이터 엔지니어링에서의 고급 아키텍처 이론

- **결함 격리 (Fault Isolation)와 트리거 룰 (Trigger Rules)**
    - 단순히 "멈추고 대기한다"는 개념을 넘어,
    - Airflow는 Task 간의 관계를 제어하기 위해 선행 Task의 상태에 따른 조건부 분기 메커니즘(Trigger Rules)을 제공함

    - **all_success (기본값):**
        - 모든 부모 Task가 성공해야 자신을 실행함

    - **one_failed / at_least_one_failed:**
        - 부모 중 하나라도 실패하면
        - 즉시 복구(Rollback) Task나 알림(Slack/PagerDuty) Task를 구동하기 위한 결함 격리 전략

    - **all_done:**
        - 부모 Task의 결과(성공/실패/인프라 에러로 인한 Skip)와 상관없이
        - 무조건 자원을 해제(Clean-up)해야 하는 인프라 관리형 Task(예: 클라우드 EMR 클러스터 종료)에 필수적으로 사용됨

- **멱등성(Idempotency)과 논리적 실행 날짜 (Logical Date)**
    - Airflow 파이프라인 설계의 핵심은 **과거 특정 시점의 시공간을 시뮬레이션할 수 있는가**의 여부

    - **Logical Date (과거명 execution_date):**
        - 이 값은 "DAG가 실제로 물리적으로 실행된 시각"이 아니라,
        - "이 DAG가 처리해야 하는 데이터의 논리적 기준 시각"을 의미

    - **Data Interval (데이터 구간):**
        - Airflow 2.2 이후부터는 `data_interval_start`와 `data_interval_end`로 명확히 분리됨
            - 예: 
                - 매일 자정에 도는 스케줄러가 2026년 7월 8일 00:00에 구동되었다면,
                - 이 스케줄러가 처리해야 하는 데이터 구간은 `2026-07-07 00:00:00`부터 `2026-07-07 23:59:59`까지

    - **과거 데이터 재처리 (Backfill 및 Catchup):**
        - 파이프라인 소스 코드가 수정되었거나 데이터 유실이 발생했을 때,
            - `catchup=True` 설정이나 `airflow dags backfill` 명령을 통해
            - 과거 1년 치 `Logical Date`를 순차적으로 주입하며 안전하게 데이터를 재생성할 수 있음
        - 모든 Task가 외부 환경 변수(예: `sysdate`)가 아닌 Airflow가 제공하는 `{ { ds }}`(Data Stamp 템플릿 변수)만을 참조하여 쿼리를 보내도록 strict하게 설계되어야 멱등성이 유지됨

- **분산 병렬 처리 (Parallelism)와 동시성 제어 백엔드**
    - 병렬 처리는 인프라 자원의 한계와 타겟 시스템(DB, API 서버)의 부하를 고려하여 다차원적으로 통제됨

    - **Executor 계층 아키텍처:**
        - **CeleryExecutor:**
            - Redis나 RabbitMQ 같은 메시지 브로커를 두고,
            - 고정된 크기의 Worker 들이 태스크를 나누어 처리함
            - 처리 속도가 빠르고 안정적

        - **KubernetesExecutor:**
            - 태스크가 생성될 때마다 쿠버네티스 클러스터 위에 독립된 **Pod**를 동적으로 실행
            - 테스크마다 상이한 라이브러리 의존성(예: CUDA 환경이 필요한 AI 모델 학습 태스크 vs 가벼운 SQL 태스크)을 완벽히 격리할 수 있어 고도화된 AI 인프라에 필수적

    - **자원 제어 파라미터 (Concurrency 임계치):**
        - `parallelism`
            - Airflow 전체 클러스터에서 동시에 실행될 수 있는 최대 Task 인스턴스 수
        - `max_active_tasks_per_dag`
            - 하나의 DAG 내에서 동시에 실행될 수 있는 Task 수
                - 예: 독립적인 100개의 로그 수집 task가 있어도 이 값이 10이면 10개씩 나누어 병렬 처리
        - `max_active_runs_per_dag`
            - 동일한 DAG가 서로 다른 Logical Date로 동시에 구동될 수 있는 최대 횟수
            - 과거 Backfill 수행 시 타겟 DB가 뻗는 것을 방지하기 위해 제약


### 1.5 데이터 엔지니어링에서 DAG가 필수적인 이유

- **결함 격리 (Fault Isolation) 및 지능형 재시도:**
    - 특정 단계(예: LLM API 전송 실패, 인프라 OOM)에서 작업이 멈추었을 때,
        - 파이프라인 전체를 중단시키지 않고 후속 작업들만 안전하게 대기 상태로 유지, 앞선 성공 단계는 재실행하지 않도록 통제함
        - 실패한 특정 노드만 타겟팅하여 '지수 백오프(Exponential Backoff)' 기반의 자동 재시도(Retry)를 수행하거나,
        - 웹 UI에서 해당 태스크만 'Clear'하여 실패 지점부터 파이프라인을 부분 재시작(Resume)할 수 있음

- **멱등성(Idempotency) 보장과 시점 복구:**
    - 동일한 논리적 실행 날짜(Logical Date)에 DAG를 몇 번을 다시 돌려도
        - 항상 같은 결과 데이터셋이 나오도록 파이프라인의 안전망을 구축할 수 있음
    - Airflow의 내장 템플릿 변수(예: `{ { ds }}`)를 활용하여 데이터 소스를 동적으로 격리함으로써,
        - 과거 특정 시점의 데이터를 오차 없이 재현(Backfill)할 수 있는 기반이 됨
    - AI 서비스에서는 어제 학습한 모델과 오늘 학습한 모델의 데이터 정합성을 유지하기 위해 필수적임

- **분산 병렬 처리 (Parallelism)와 리소스 최적화:**
    - 선후 관계가 없는 독립적인 노드들(예: 서로 다른 3개 마이크로 서비스의 로그 수집, 또는 여러 스케일의 이미지 전처리)을
    - 스케줄러가 자동으로 판단하여 다수의 Worker 노드에 동시에 지시, 병렬 처리를 극대화
    - 대규모 하드웨어 자원이 필요한 AI 워크로드에서, 
        - **Celery나 Kubernetes 기반의 Worker**에 태스크를 동적으로 배정하여
        - GPU 가속이 필요한 모델 추론 태스크와 일반 CPU 가벼운 전처리 태스크를 분리하여 인프라 비용을 극적으로 절감


## 2. DAG 유즈케이스 연구 (Use Cases)

### 2.1 Case A: 선형 및 분기형 데이터 동기화 파이프라인 (ETL/ELT)

- 가장 보편적인 유즈케이스
- 대용량 트랜잭션 데이터를 분석 저장소로 안전하게 이관하는 구조

- **흐름 및 기술적 포인트:**
    1. 새벽 2시에 Production DB에서 어제 자 정산 데이터를 백업 처리 (`BashOperator`)
        - 운영 DB의 성능 저하(Read 부하)를 방지하기 위해 반드시 트래픽이 가장 적은 새벽 시간대에 스케줄링
        - 대용량일 경우 DB Direct 조회 대신 복제본(Read Replica)을 대상으로 스크립트를 수행해야 함

    2. 동시에 마케팅 API 서버로부터 광고 효율 로그를 다운로드 (`PythonOperator`)
        - 1번과 2번 태스크는 서로 선후 관계가 없으므로 병렬 실행(Parallelism) 됨
        - API 호출 시 rate limit(요청 제한)이나 일시적 네트워크 타임아웃에 대비해 오퍼레이터 내에 `retries`와 `retry_delay` 설정을 필수로 적용

    3. 두 원시 데이터가 모두 스토리지(MinIO/S3)에 적재 완료되었는지 검증 (`MinioSensor`)
        - `Sensor`는 특정 조건(파일 생성, DB 레코드 생성 등)이 충족될 때까지 대기하는 특수 오퍼레이터
        - 워커의 자원을 계속 점유하는 `poke` 모드 대신, 주기적으로만 확인하고 자원을 반환하는 **`reschedule` 모드**를 사용해야 분산 인프라의 자원 낭비를 막을 수 있음

    4. 정제 및 병합 쿼리를 실행하여 데이터 웨어하우스(빅쿼리/Postgres)에 적재 (`PostgresOperator`)
        - Airflow 워커 자체에서 대용량 데이터를 메모리에 올려 가공하면 OOM(Memory 부족)이 발생함
        - Airflow는 단순 '트리거' 역할만 수행하고,
        - 실제 무거운 연산(Join, Aggregation)은 데이터 웨어하우스(빅쿼리, Postgres 엔진 등)의 컴퓨팅 자원을 사용하는 **ELT 방식**을 취해야 함

    5. 성공/실패 여부를 사내 메신저로 전송 (`SlackWebhookOperator`)
        - 모든 태스크마다 알림 오퍼레이터를 연결하면 코드가 지저분해짐
        - DAG의 `default_args`에 `on_failure_callback` 함수를 지정하여, 파이프라인 내 **어느 한 곳이라도 실패하면 자동으로 알림이 가도록 공통 인터셉터 구조**로 설계함


### 2.2 Case B: 주기적 인프라 상태 점검 및 자동 스케일링 체인

- 데이터 분석 외에 시스템 운영 인프라를 관제하고 최적화(FinOps/CloudOps)하는 유즈케이스

- **흐름 및 기술적 포인트:**
    1. 매주 일요일 가상 머신(VM) 및 쿠버네티스(K8s) 클러스터의 미사용 자원 스캔
        - Prometheus나 CloudWatch의 메트릭 API를 호출하여 최근 7일간 CPU/Memory 사용률이 평균 5% 미만인 노드를 필터링
        - 주말 인프라 비용 절감(FinOps)을 위한 필수 관제 단계

    2. 임계치 이하의 노드가 발견되면 리소스 축소(Downscaling) 스크립트 트리거
        - 자원을 줄이는 행위는 운영 중인 마이크로 서비스에 영향을 줄 수 있는 고위험 작업
        - Airflow DAG 내부에 `ShortCircuitOperator`나 `BranchPythonOperator`를 배치하여,
        - '축소 대상 노드가 존재할 때만' 다운스케일링 태스크로 진입하도록 조건부 분기를 엄격히 제어해야 함

    3. 백업 볼륨 복구 테스트 진행 및 디스크 정제 작업 수행
        - 정기적인 백업본이 깨지지 않았는지 자동 복구(Restore) 시뮬레이션을 실행
        - 오래된 캐시나 로그 파일(Docker dangling 이미지 등)을 삭제하여 스토리지 비용을 최적화


### 2.3 Case C: 로컬 AI 모델 재학습 및 하이브리드 RAG 지식 임베딩 자동화

- 지속적으로 발생하는 사내 기술 매뉴얼과 매일 쌓이는 고객 상담 이력을
- 로컬 AI 인프라에 동적으로 반영하기 위한 최신 AI 엔지니어링 파이프라인

- **흐름 및 기술적 포인트:**
    1. 현장 엔지니어가 작성한 최신 장비 가이드 문서(`*.txt`)를 MinIO 특정 버킷에서 감지
        - 주기적 배치 스케줄링(`schedule_interval`) 외에도,
        - MinIO의 웹훅(Webhook) 이벤트를 수신하여 데이터가 들어오는 즉시 DAG를 구동하는 **`TriggerDagRunOperator` 이벤트 기반 아키텍처**를 적용하기에 적합한 단계

    2. 텍스트 문서를 로드하여 토큰 제한에 맞게 의미론적 청킹(Chunking) 수행
        - AI LLM의 콘텍스트 윈도우 특성을 고려하여 문맥이 깨지지 않게 자르는 알고리즘(예: RecursiveCharacterTextSplitter)이 수행되는 단계
        - CPU 연산이 집중되는 구간이므로 Airflow 기본 워커와 분리하는 것이 좋음

    3. 로컬 Ollama 임베딩 모델을 호출하여 지식 데이터를 벡터화
        - 이 단계는 GPU 자원을 집중적으로 소모함
        - Airflow의 일반 가벼운 태스크들과 동거하면 인프라가 마비될 수 있음
        - **`KubernetesPodOperator`를 통해 GPU 노드가 할당된 독립된 Pod에서 실행**하거나,
        - 외부 Ollama 전용 서버의 REST API를 호출하는 비동기 방식으로 구현하여
        - 태스크 격리(Isolation)를 달성해야 함

    4. Qdrant와 같은 VectorDB의 지정된 컬렉션에 하이브리드 인덱스로 신규 업서트(Upsert) 반영
        - 멱등성(Idempotency)이 가장 중요한 단계
        - 문서가 수정되어 DAG가 재실행되었을 때 동일한 지식이 중복 저장되면 RAG 검색 시 답변 왜곡(Hallucination)이 발생함
        - 고유한 문서 ID 파싱을 통해 기존 벡터를 덮어쓰는 **Upsert(Insert or Update) 메커니즘**을 엄격하게 구현해야 파이프라인 안정성이 보장됨


<div class="insert-image" style="text-align: center;">
    <img src="/materials/devtools/images/S13-06-02-01_01-003_AirflowDagUsecase.png" style="width: 100%;">
</div>


## 3. 실습 예제: 

- **이원화 저장소 백업 및 텍스트 데이터 정제 파이프라인**
    - 로컬 파일 시스템에서 원시 로그 데이터를 읽어와 **MinIO 객체 스토리지에 백업**한 뒤,
    - 의미론적으로 데이터를 가공 및 변환(Transform)하여
    - RAG나 분석 시스템이 사용할 수 있도록 정제하는
    - 가상의 DAG 아키텍처를 파이썬 코드로 구현


### 3.1 시나리오 구성도

```text
[Task 1: check_file] 🡲 [Task 2: backup_to_minio] 🡲 [Task 3: transform_data]
```

<div class="info-table">
<table>
    <thead>
        <th style="width: 250px;">구분</th>
        <th style="width: 250px;">Task 1 (진입/검증)</th>
        <th style="width: 250px;">Task 2 (백업/적재)</th>
        <th style="width: 250px;">Task 3 (가공/정제)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">시나리오 구성도 명칭</td>
            <td class="td-left">check_file</td>
            <td class="td-left">backup_to_minio</td>
            <td class="td-left">transform_data</td>
        </tr>
        <tr>
            <td class="td-rowheader">소스코드 내 task_id</td>
            <td class="td-left">check_local_file_exists</td>
            <td class="td-left">backup_raw_data_to_minio</td>
            <td class="td-left">transform_and_extract_warnings</td>
        </tr>
        <tr>
            <td class="td-rowheader">연동 파이썬 함수명</td>
            <td class="td-left">(Bash 명령어 수행)</td>
            <td class="td-left">fn_backup_to_minio</td>
            <td class="td-left">fn_transform_and_clean_data</b></td>
        </tr>
        <tr>
            <td class="td-rowheader">Airflow 내부 실행 단계</td>
            <td class="td-left">1단계: 인프라 상태 검증</td>
            <td class="td-left">2단계: 스토리지 복제</td>
            <td class="td-left">3단계: 비즈니스 가공</b></td>
        </tr>
    </tbody>
</table>
</div>


- **시나리오와 실행 단계의 연계 원리 해설**
    - 개발자가 파이썬 코드로 `Task 1 >> Task 2 >> Task 3` 구조를 빌드하면, Airflow 스케줄러 엔진은 이를 백엔드에서 다음과 같은 원리로 연계하여 실행합니다.

    - 1단계 연계: `check_file` (소스코드 id: `check_local_file_exists`)
        - **연계 원리 (관문 역할):**
            - 파이프라인 기동 시 가장 먼저 실행되는 **진입점**
            - 호스트와 컨테이너 간의 파일 시스템 및 런타임 환경이 정상적인지 Bash 셸을 통해 물리적으로 검증
            - 이 단계가 성공해야만 비로소 데이터가 흐르기 시작함

    - 2단계 연계: `backup_to_minio` (소스코드 id: `backup_raw_data_to_minio`)
        - **연계 원리 (멱등성 확보):**
            - 1단계 검증이 끝나면 의존성 간선(`>>`)을 타고 데이터 복제 단계가 트리거됨
            - 로컬 디스크의 원시 로그를 분산 오브젝트 스토리지(MinIO)로 이관하여
            - "과거 특정 시점의 원시 스냅샷 데이터를 언제든 재현할 수 있는 안전망(멱등성)"을 확보하는 핵심 연계 구간

    - 3단계 연계: `transform_data` (소스코드 id: `transform_and_extract_warnings`)
        - **연계 원리 (디커플링 및 가공):**
            - 2단계에서 스토리지 적재가 완료된 것을 확인한 후 최종 기동
            - 저장된 원시 파일에서 `WARNING` 로그만 추출하여 RAG 시스템이나 분석 DB가 읽을 수 있는 콘텍스트 형태로 변환
            - 백업(2단계)과 가공(3단계)을 분리함으로써,
                - **가공 규칙이 변경되어 이 단계만 실패하더라도 앞선 백업 단계는 건드리지 않고 3단계만 재실행(Clear)할 수 있는 결함 격리**가 완성됨


### 3.2 파이프라인 실행 메커니즘 및 단계별 원리 분석

- Airflow 스케줄러 아키텍처 관점에서 이 DAG가 구동될 때 백엔드에서 일어나는 현상과 그 기술적 배경
- "시나리오 구성(3개 태스크)"이 실제로 살아 움직이기 위해 거치는 시스템 내부의 '런타임 라이프사이클(Runtime Lifecycle)'

```text
[ Airflow 백엔드 엔진의 타임라인 ]

  1단계: 선행 환경 점검 및 DAG 직렬화 (Parsing & Serialization)
    │  (시나리오 외적인 인프라 준비 단계 / 백그라운드 상시 구동)
    ▼
  2단계: task_check_file
    │  ➔ [시나리오 Task 1: check_file] 실행 단계
    │       - 진입점 확보 및 인프라 정합성 확보
    │       - 스케줄러의 실제 Task 큐 배정 및 워커 할당
    ▼
  3단계: task_backup_minio
    │  ➔ [시나리오 Task 2: backup_to_minio] 실행 단계
    │       - 원시 데이터의 멱등성 보장 및 격리
    │       - 성공 인지 후 다음 노드 인스턴스화 및 Python 실행
    ▼
  4단계: task_transform_clean
       ➔ [시나리오 Task 3: transform_data] 실행 단계
            - 컴퓨팅 자원의 분리와 비즈니스 로직 적용
            - 최종 노드 구동 및 스트리밍 메모리 분리       
```

- **1단계: 선행 환경 점검 및 DAG 직렬화 (Parsing & Serialization)**
    - 사용자가 시나리오를 구동하기 전, 
        - Airflow 엔진이 코드가 문법적으로 올바른지, 
        - 사이클(Loop)은 없는지 검사하여 메모리에 올리는 단계
    - 시나리오의 태스크가 실행되기도 전에 백엔드에서 상시 수행되는 엔진 고유의 영역<br><br>

    - **실행 과정:**
        - Airflow 스케줄러의 `DagFileProcessor`가 주기적으로 코드를 파싱
        - `factory_log_hybrid_processing_v1`이라는 DAG 구조를 메타데이터 DB에 JSON 형태로 직렬화하여 동기화
        - Web UI에 DAG가 노출되는 시점

    - **엔지니어링 원리:**
        - 설정 구역에 선언된 `MINIO_URL`, `LOCAL_FILE_PATH` 같은 전역 변수들은 스케줄러가 파싱할 때 메모리에 로드됨
        - **주의할 점 (Anti-Pattern 방지):**
            - 만약 전역 변수 구역에
                - `Minio(...)` 클라이언트를 직접 생성하거나
                - 파일 시스템의 존재 여부를 체크하는 `os.path.exists()` 코드를 배치했다면,
                    - 스케줄러가 30초마다 이 외부 인프라에 접근하려다 병목이 발생
            - **모든 무거운 무작위 연산과 연결(Connection) 수립은 반드시 태스크 함수(`fn_...`) 내부로 격리**해야 함


- **2단계: `task_check_file` (진입점 확보 및 인프라 정합성 확보)**
    - 시나리오 1단계인 파일 체크가 실행되는 순간
    - 엔진 관점에서는 진입차수(In-degree)를 계산해 Worker 노드에 배정하는 아키텍처적 메커니즘이 작동함<br><br>

    - **실행 과정:**
        - DAG가 트리거되면 스케줄러는
            - 이 태스크의 진입차수(In-degree)가 0임을 확인
            - `Queued` 상태를 거쳐
            - Worker에게 배정

    - **엔지니어링 원리:**
        - 실무에서 첫 번째 단계는 단순 로그 출력이 아닌 "이 작업을 시작할 최소한의 요건이 되었는가?"를 검증하는 관문(Gateway) 역할
        - 만약 하드웨어나 디스크 가용 용량, 컨테이너 네트워크에 결함이 있다면
            - 이 단계에서 파이프라인이 즉시 차단(Fault Isolation)되어
            - 후속 백업 및 데이터 오염을 원천 차단

- **3단계: `task_backup_minio` (원시 데이터의 멱등성 보장 및 격리)**
    - 시나리오 2단계인 MinIO 백업이 실행되는 순간
    - 엔진 관점에서는
        - 앞선 태스크의 Success 상태를 메타데이터 DB에서 확인하고,
        - 파이썬 런타임을 띄워
        - 외부 스토리지와 핸드셰이크를 맺는 물리적 연산이 수행됨<br><br>

    - **실행 과정:**
        - 1단계가 `Success`로 마크되면 엣지(`>>`)를 타고 `task_backup_minio`가 인스턴스화
        - 파이썬 런타임이 구동되며 MinIO API와 핸드셰이크를 수행

    - **엔지니어링 원리:** 
        - **데이터 소스 격리:**
            - 로컬 디스크는 언제든 유실되거나 덮어써질 수 있는 불안정한 저장소
            - 이를 분산 오브젝트 스토리지(MinIO/S3)로 신속히 복제함으로써 원시 데이터(Raw Data)의 스냅샷을 영구 보존

    - **방어적 코드(Defensive Coding)와 멱등성:**
        - `client.bucket_exists`로 버킷의 유무를 먼저 파악하고 동적으로 대응하는 코드
            - 인프라 상태가 초기화되었더라도
            - 파이프라인이 에러 없이 동일한 결과(버킷 생성 및 파일 업로드)를 보장하게 만드는 **멱등성 설계**의 기초

- **4단계: `task_transform_clean` (컴퓨팅 자원의 분리와 비즈니스 로직 적용)**
    - 시나리오 3단계인 데이터 가공이 실행되는 순간
    - 엔진 관점에서는
        - 이전 단계들과 메모리를 공유하지 않도록
        - 컨테이너/프로세스를 완전히 분리(Decoupling)하여
        - 독립된 자원으로 비즈니스 로직만 깔끔하게 수행하고 종료하는 단계<br><br>

    - **실행 과정:**
        - 스토리지 적재가 확인되면
        - 마지막 태스크가 기동되어
        - MinIO로부터 데이터를 스트리밍으로 읽어와
        - `WARNING` 패턴 매칭을 수행하고
        - 표준 출력으로 리포팅

    - **엔지니어링 원리:** 
        - **Decoupling (디커플링):**
            - 만약 백업과 정제를 하나의 태스크 함수에 몰아서 코딩했다면,
                - 정제 로직(오타 수정, 필터링 규칙 변경)을 고칠 때마다 백업 로직까지 같이 위험에 노출됨
            - 태스크를 쪼갬으로써
                - 정제 단계가 실패하더라도
                - 백업 단계는 터치하지 않고
                - 실패한 정제 단계만 Airflow UI에서 `Clear`하여 재실행 할 수 있게 됨

### 3.3 실습 환경 구축 및 코드 작업

- 환경파일 수정
    - 프로젝트 루트 디렉토리의 .env 파일에 minio를 추가함

    ```env
    AIRFLOW_UID=1000
    _PIP_ADDITIONAL_REQUIREMENTS=minio
    ```

- **선행 환경 점검**
    - airflow-scheduler, airflow-webserver, minio 컨테이너의 State가 모두 Up 상태여야 함

    ```bash
    docker compose ps
    ```

- **DAG 소스코드 작성**

```python
#//file: "dags/minio_processing_pipeline.py"
from datetime import datetime, timedelta
import os
import io
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from minio import Minio

# ============================================================
# [설정 구역] 인프라 연결 정보 정의
# ============================================================
# MINIO_URL = "localhost:9000"
MINIO_URL = "192.168.0.6:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadminpassword"
BUCKET_NAME = "factory-raw-logs"

LOCAL_FILE_DIR = "/opt/airflow/data"
FILE_NAME = "factory_sensor_logs.txt"
LOCAL_FILE_PATH = os.path.join(LOCAL_FILE_DIR, FILE_NAME)

# ============================================================
# [비즈니스 로직 함수 정의]
# ============================================================
def fn_backup_to_minio():
    """로컬의 원시 센서 데이터를 MinIO 스토리지 버킷에 안전하게 백업합니다."""
    client = Minio(MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    
    # 버킷이 없을 경우 자동 생성
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"[MinIO] '{BUCKET_NAME}' 버킷이 새로 생성되었습니다.")

    # 테스트를 위한 더미 데이터 생성 (물리 파일이 없을 경우 방어 코드)
    if not os.path.exists(LOCAL_FILE_PATH):
        os.makedirs(LOCAL_FILE_DIR, exist_ok=True)
        with open(LOCAL_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("TIMESTAMP:2026-07-07 15:00:00 | SENSOR:FM-2026 | VAL:45.2 | STATUS:NORMAL\n"
                    "TIMESTAMP:2026-07-07 15:05:00 | SENSOR:FM-2026 | VAL:99.9 | STATUS:WARNING_OVERFLOW\n")
        print(f"[Local] 테스트용 더미 로그 파일 생성 완료: {LOCAL_FILE_PATH}")

    # 파일 업로드 실행
    with open(LOCAL_FILE_PATH, "rb") as file_data:
        data_bytes = file_data.read()
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=FILE_NAME,
            data=io.BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="text/plain"
        )
    print(f"[성공] 원시 파일이 MinIO [{BUCKET_NAME}/{FILE_NAME}]에 백업 동기화되었습니다.")


def fn_transform_and_clean_data():
    """MinIO에서 백업된 데이터를 다운로드하여 WARNING 상태인 라인만 추출해 정제합니다."""
    client = Minio(MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    
    # 스토리지로부터 데이터 가져오기
    try:
        response = client.get_object(BUCKET_NAME, FILE_NAME)
        raw_content = response.read().decode('utf-8')
    finally:
        response.close()
        response.release_conn()

    # 데이터 변환 변형 (Transform Logics)
    lines = raw_content.strip().split("\n")
    cleaned_records = []
    
    for line in lines:
        if "WARNING" in line:  # 위험 감지 데이터만 필터링하는 규칙
            parts = line.split(" | ")
            refined_line = f"[위험 알림] 시간: {parts[0].split(':')[1]}, 센서명: {parts[1].split(':')[1]}, 측정값: {parts[2].split(':')[1]}"
            cleaned_records.append(refined_line)

    # 변환된 최종 데이터 출력 (운영 환경에서는 이 결과를 다른 DB나 하위 파일로 이관합니다.)
    print("=" * 60)
    print("[정제 완료된 하이브리드 컨텍스트 데이터 목록]")
    print("=" * 60)
    if cleaned_records:
        for record in cleaned_records:
            print(record)
    else:
        print("정제 대상(WARNING) 데이터가 존재하지 않습니다.")
    print("=" * 60)


# ============================================================
# [DAG 정의 및 오케스트레이션 설계]
# ============================================================
default_args = {
    'owner': 'seokhwan_yang',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    dag_id='factory_log_hybrid_processing_v1',
    default_args=default_args,
    description='MinIO 백업 및 데이터 텍스트 정제를 위한 기본 하이브리드 파이프라인 DAG',
    schedule='@daily',  # 매일 자정에 정기 실행
    catchup=False,               # 과거 누적 배치 미실행 (과부하 방지)
    tags=['production', 'factory', 'minio'],
) as dag:

    # Task 1: Bash 명령어를 통한 파일 시스템 존재 상태 선행 점검
    task_check_file = BashOperator(
        task_id='check_local_file_exists',
        bash_command='echo "워크플로우 기동 시작" && date',
    )

    # Task 2: Python 커스텀 함수를 활용한 MinIO 스토리지 적재 백업 작업
    task_backup_minio = PythonOperator(
        task_id='backup_raw_data_to_minio',
        python_callable=fn_backup_to_minio,
    )

    # Task 3: 적재 완료된 파일의 비즈니스 규칙 기반 필터링 및 텍스트 트랜스폼 작업
    task_transform_clean = PythonOperator(
        task_id='transform_and_extract_warnings',
        python_callable=fn_transform_and_clean_data,
    )

    # ============================================================
    # [의존성 선언 구역] 상하 흐름 명시적 정의 (DAG 뼈대 빌드)
    # ============================================================
    task_check_file >> task_backup_minio >> task_transform_clean
```


- **실습 코드 구동 분석 및 디버깅 가이드**

    - **Step 1: 환경 구성 확인 🡲 DAG 파일 배포 및 스케줄러 인식 확인**
        - 사용자가 작성한 `minio_processing_pipeline.py` 파일이 Airflow 엔진에 정상적으로 로드되는지 확인하는 단계

        1. **DAG 파일 복사:**
            - 작성한 파이썬 파일을 Airflow 컨테이너가 참조하는 호스트의 dags/ 디렉토리로 이동

        2. **백엔드 직렬화 대기:**
            - Airflow 스케줄러의 DagFileProcessor가 파일의 문법을 검사하고 DB에 등록할 때까지 약 30초~1분 정도 대기

        3. **웹 GUI 접속 및 활성화:**
            - 브라우저를 열고 http://localhost:8080에 접속
            - DAGs 메인 화면에 `factory_log_hybrid_processing_v1` 레이블이 새로 나타났는지 확인
            - 주의:
                - 처음 등록된 DAG는 누적 배치가 도는 것을 방지하기 위해 Pause(비활성화) 상태로 되어 있음
                - 토글 스위치를 클릭하여 Active (파란색 상태)로 전환

    - **Step 2: 파이프라인 수동 트리거 및 실행 메커니즘 관측**
        - 스케줄러가 정해진 순서(>>)에 맞춰 작업을 분산 제어하는 과정을 모니터링

        1. **DAG 수동 실행:**
            1. 우측 상단의 Trigger DAG (재생 아이콘) 버튼을 클릭하여 파이프라인을 기동
            2. 스케줄러가 간선(`>>`) 순서에 맞춰 `task_check_file`을 가장 먼저 큐에 배치함

        2. **태스크 인스턴스 라이프사이클 관측:**
            - 'Graph' 뷰 또는 'Grid' 뷰를 선택하면 각 태스크 테두리의 색상이 실시간으로 변하는 것을 볼 수 있음
                - 테두리 연회색 (Queued): 스케줄러가 자원 배정을 위해 대기열에 넣은 상태
                - 테두리 연두색 (Running): Worker 프로세스가 할당되어 실제 코드가 실행 중인 상태
                - 테두리 진녹색 (Success): 에러 없이 성공적으로 처리가 완료된 상태

        3. **순차 실행 검증:**
            1. 첫 번째 태스크가 정상 종료(Success)되면 그 결과를 인지하고
                - `check_local_file_exists`가 녹색으로 변함
            2. 즉시 다음 노드인 `backup_raw_data_to_minio` 태스크 인스턴스를 활성화(기동)하여 물리 자원을 배정
            3. 파이썬 함수 내부에 구현된 `Minio` 클라이언트 프로토콜에 의해 파일이 버킷에 동기화 완료
            4. 최종적으로 `transform_and_extract_warnings`가 기동되어 로그를 파싱하고 정제 결과를 콘솔에 출력
            - 전체적으로 제어권이 넘어가는 위상 정렬(Topological Sort) 흐름을 확인

    - **Step 3: 최종 연동 결과 및 런타임 로그 검증**
        - 비즈니스 로직과 외부 인프라(MinIO)에 데이터가 정상적으로 가공되어 적재되었는지 데이터 정합성을 확인

        1. MinIO 스토리지 적재 결과 확인 (Task 2의 결과)
            1. MinIO 웹 콘솔(http://localhost:9001)에 접속
            2. fn_backup_to_minio 함수의 방어 코드에 의해 factory_raw_logs라는 이름의 버킷이 자동으로 생성되었는지 확인
            3. 버킷 내부로 진입하여 factory_sensor_logs.txt 파일이 정상적으로 업로드되었는지 확인

        2. Airflow 콘솔 로그를 통한 가공 텍스트 확인 (Task 3의 결과)
            1. Airflow Web UI에서 마지막 노드인 transform_and_extract_warnings 태스크 블록을 클릭
            2. 상단 메뉴 중 Log 버튼을 클릭
            3. 출력된 표준 출력(Standard Output) 로그 내부에서 finally 구문에 의해 MinIO 커넥션 풀이 안전하게 닫혔는지(release_conn()) 확인
            4. 비즈니스 규칙에 의해 필터링된 데이터가 아래 예시와 같이 포맷팅되어 찍혔는지 확인

        - 마지막 `task_transform_clean` 인스턴스의 텍스트 로그를 클릭하여 열어보면 다음과 같은 출력 구조가 표시되는 것을 확인할 수 있음
        - 이를 통해 DAG 파이프라인이 정상적으로 원인-결과 흐름을 조율했음이 증명됨

        ```text
        Found local logs:
        [2026-07-07 17:15:22,345] {python.py:177} INFO - Done. Integration tasks succeeded.
        ============================================================
        [정제 완료된 하이브리드 컨텍스트 데이터 목록]
        ============================================================
        [위험 알림] 시간: 2026-07-07 15:05:00, 센서명: FM-2026, 측정값: 99.9
        ============================================================
        ```

        - 만약 3단계가 실패하여 빨간색(Failed)으로 변했다면,
            - 로그 상단에서 Python의 Traceback 에러 메시지를 추적하여 MinIO 접근 권한(Access Key) 오타나 로컬 디스크 볼륨 마운트 차단 여부를 점검
            
    <br>

    <div class="insert-image" style="text-align: center; border: solid 1px lightgray;">
        <img src="/materials/devtools/images/S13-06-02-01_01-004.png" style="width: 100%;">
    </div>