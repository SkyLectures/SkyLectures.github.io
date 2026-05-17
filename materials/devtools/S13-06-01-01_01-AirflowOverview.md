---
layout: page
title:  "Airflow 개요"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-06-01-01_01-AirflowOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## Orchestration & AI Service: Apache Airflow 구축 가이드

AI 서비스와 데이터 파이프라인의 복잡도가 증가함에 따라, 각 태스크의 의존성을 관리하고 실행 순서를 보장하는 Orchestration(오케스트레이션)은 필수적인 요소입니다. 그 중 가장 대표적인 도구인 **Apache Airflow**에 대해 상세히 정리해 드립니다.

---

### 1. Airflow의 개념 및 개요

**Apache Airflow**는 워크플로우(Workflow)를 작성, 스케줄링 및 모니터링하는 오픈소스 플랫폼입니다.

* **DAG (Directed Acyclic Graph):** 워크플로우를 구성하는 태스크(Task)들의 실행 순서와 의존 관계를 정의한 비순환 유향 그래프입니다. Python 코드로 정의되므로 버전 관리 및 협업에 유리합니다.
* **핵심 철학:** "Configuration as Code". 즉, 워크플로우를 코드로 관리하여 유연성과 확장성을 극대화합니다.
* **AI 서비스에서의 역할:** 데이터 수집 → 전처리 → 모델 학습 → 평가 → 서빙으로 이어지는 **ML Pipeline**의 각 단계를 자동화하고, 특정 단계 실패 시 재시도(Retry)나 알림을 처리합니다.

---

### 2. Airflow 환경 구축 (Docker 기반)

가장 권장되는 방식인 Docker Compose를 활용한 구축 방법입니다. 격리된 환경에서 Celery Executor 기반의 분산 처리가 가능하도록 구성할 수 있습니다.

#### ① 준비 단계

```bash
# 작업 디렉토리 생성
mkdir airflow-docker && cd airflow-docker

# 공식 docker-compose.yaml 파일 다운로드
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'

# 필요한 디렉토리 생성
mkdir -p ./dags ./logs ./plugins ./config

```

#### ② 환경 변수 설정

Airflow에 필요한 유저 ID 정보를 `.env` 파일에 기록합니다.

```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env

```

#### ③ 서비스 초기화 및 실행

```bash
# DB 초기화
docker-compose up airflow-init

# 서비스 실행 (-d는 백그라운드 실행)
docker-compose up -d

```

* 실행 후 `http://localhost:8080`으로 접속 가능합니다. (기본 계정: `airflow` / `airflow`)

---



Airflow의 핵심은 단순히 코드를 실행하는 것이 아니라, 복잡하게 얽힌 수많은 작업(Job)들을 **지능적으로 관리하고 시각화**하는 데 있습니다. 이를 더 깊이 이해하기 위해 핵심 아키텍처와 구성 요소별 역할을 상세히 정리해 드립니다.

---

### 1. Airflow의 5대 핵심 구성 요소 (Architecture)

Airflow는 여러 서비스가 유기적으로 결합된 분산 시스템 구조를 가집니다.

* **Scheduler (스케줄러):** 전체 워크플로우의 '두뇌'입니다. 모든 DAG와 태스크를 모니터링하며, 실행 조건이 충족된 태스크를 큐(Queue)에 보냅니다.
* **Executor (실행기):** 태스크가 '어떻게' 실행될지 결정합니다. 단일 머신에서 실행할지(Sequential/Local), 여러 서버에 분산하여 실행할지(Celery/Kubernetes)를 정의합니다.
* **Worker (워커):** 실제로 태스크를 수행하는 일꾼입니다. Executor가 할당한 로직을 직접 실행합니다.
* **Web Server (웹 서버):** 사용자가 DAG의 상태를 확인하고, 수동으로 실행하거나 로그를 점검할 수 있는 GUI 인터페이스를 제공합니다.
* **Metadata Database (메타데이터 DB):** DAG, 태스크 상태, 사용자 정보 등 모든 실행 이력을 저장하는 저장소입니다. (주로 PostgreSQL이나 MySQL 사용)

---

### 2. DAG (Directed Acyclic Graph)의 심층 이해

Airflow에서 모든 작업 흐름은 DAG로 표현됩니다.

* **Directed (유향):** 작업 간에 명확한 방향이 있습니다. (A → B)
* **Acyclic (비순환):** 루프가 없습니다. A가 B를 실행하고, 다시 B가 A를 실행하는 순환 구조는 허용되지 않습니다. 이는 무한 루프 방지와 명확한 선후 관계 보장을 위함입니다.
* **Graph (그래프):** 노드(태스크)와 엣지(의존성)로 구성된 구조체입니다.

---

### 3. Operator vs Task: 개념의 차이

초보자가 가장 많이 혼동하는 개념입니다.

* **Operator (오퍼레이터):** 작업을 수행하기 위한 **'템플릿'** 또는 '틀'입니다. (예: PythonOperator, BashOperator, S3ToRedshiftOperator)
* **Task (태스크):** 오퍼레이터를 실제로 구체화하여 DAG에 배치한 '실행 단위'입니다. 하나의 DAG 안에 여러 개의 태스크가 존재하며, 각 태스크는 오퍼레이터의 인스턴스입니다.

---

### 4. Airflow의 핵심 동작 메커니즘

1. **Parsing:** 스케줄러가 작성된 Python 코드를 읽어 DAG 구조를 파악합니다.
2. **Scheduling:** `start_date`와 `schedule_interval`을 계산하여 실행 시점이 된 태스크를 'Scheduled' 상태로 바꿉니다.
3. **Queuing:** 실행 가능한 태스크를 Executor에게 전달하고, Executor는 이를 큐에 넣습니다.
4. **Execution:** 워커가 큐에서 태스크를 가져와 실제 로직을 실행합니다.
5. **State Update:** 실행 결과(Success/Failed)를 메타데이터 DB에 업데이트하고, 웹 서버는 이를 화면에 출력합니다.

---

### 5. 왜 Airflow를 쓰는가? (장점 및 특징)

* **Dynamic Pipeline:** Python 코드를 사용하므로, 반복문을 통해 수백 개의 태스크를 동적으로 생성할 수 있습니다.
* **Extensibility:** 수많은 라이브러리가 미리 구현되어 있어, AWS, GCP, Azure, Slack, Docker 등과의 연동이 매우 쉽습니다.
* **Retry & Monitoring:** 특정 단계에서 에러가 나면 자동으로 재시도하도록 설정하거나, 실패 즉시 담당자에게 메시지를 보낼 수 있습니다.
* **Backfill:** 과거 특정 시점의 데이터를 다시 처리해야 할 때, 코드 수정 없이 명령 하나로 과거 날짜의 작업들을 일괄 실행할 수 있습니다.

### 6. 주요 용어 정리 (Cheatsheet)

| 용어 | 설명 |
| --- | --- |
| **Execution Date** | DAG가 실행되기로 예약된 논리적 시점 (실제 실행 시간과 다를 수 있음) |
| **XComs** | 태스크 간에 작은 데이터(메시지, 경로 등)를 공유하기 위한 통신 메커니즘 |
| **Variables** | Airflow 전역에서 공통으로 사용하는 설정값 (ID, 경로 등) |
| **Connections** | 외부 시스템(DB, 클라우드) 접속 정보 관리 (암호화되어 저장됨) |

Airflow는 단순한 스케줄러를 넘어, 데이터의 흐름을 하나의 제품(Product)처럼 관리할 수 있게 해주는 강력한 플랫폼입니다. 구축 단계에서 어떤 부분이 가장 궁금하신가요?



### 3. 활용 방법 및 워크플로우 흐름

1. **DAG 작성:** `./dags` 디렉토리에 Python 파일을 생성하여 워크플로우를 코딩합니다.
2. **스케줄링:** `schedule_interval` 설정을 통해 주기적(예: 매일 자정)으로 실행되도록 설정합니다.
3. **모니터링:** 웹 UI를 통해 각 태스크의 성공/실패 여부, 로그, 실행 시간을 실시간으로 확인합니다.
4. **연동:** 다양한 Operator(Python, Bash, SQL, Docker, Kubernetes 등)를 사용하여 외부 시스템과 상호작용합니다.

---

### 4. 예제 코드 및 상세 설명

간단한 데이터 전처리 및 AI 모델 학습 단계를 가정한 Python 기반 DAG 예제입니다.

#### [example_ai_pipeline.py]

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

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
    schedule_interval='@daily',  # 매일 실행
    catchup=False
) as dag:

    # 3. 태스크 정의 (Operators)
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

#### [예제 코드 상세 설명]

* **`DAG` 객체:** 워크플로우의 본체입니다. `ai_service_orchestration_v1`이라는 ID를 가지며, 매일 자정에 실행되도록 설정되었습니다.
* **`PythonOperator`:** Python 함수를 Airflow 태스크로 실행하기 위해 사용합니다.
* **`XCom` (Cross-Communication):** `ti.xcom_pull`을 통해 태스크 간에 데이터를 주고받습니다. 실제 환경에서는 대용량 데이터 대신 경로 정보나 상태값 등을 전달합니다.
* **의존성 (`>>`):** `preprocess_task >> train_task`는 "전처리가 성공해야 학습을 시작한다"는 명확한 순서를 보장합니다.

---

### 5. 핵심 활용 팁

* **Idempotency (멱등성):** 동일한 입력에 대해 항상 동일한 결과가 나오도록 DAG를 설계해야 합니다. (실패 후 재실행 시 데이터 중복 방지)
* **Dynamic Task Generation:** 리스트나 딕셔너리를 활용해 수십 개의 태스크를 반복문으로 자동 생성할 수 있습니다.
* **Docker/Kubernetes Operator:** Airflow 워커 환경에 구애받지 않고, 각 태스크마다 독립적인 컨테이너 환경에서 학습이나 추론을 수행할 수 있어 AI 서비스 구축 시 매우 강력합니다.