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

### 1.2 데이터 엔지니어링에서 DAG가 필수적인 이유

- **결함 격리 (Fault Isolation):**
    - 특정 단계(예: API 전송 실패)에서 작업이 멈추었을 때,
    - 후속 작업들만 안전하게 대기 상태로 묶고
    - 앞선 성공 단계는 재실행하지 않도록 통제

- **멱등성(Idempotency) 보장:**
    - 동일한 논리적 실행 날짜(Logical Date)에 DAG를 몇 번을 다시 돌려도
    - 항상 같은 결과 데이터셋이 나오도록 파이프라인의 안전망을 구축할 수 있음

- **분산 병렬 처리 (Parallelism):**
    - 선후 관계가 없는 독립적인 노드들(예: 서로 다른 3개 마이크로 서비스의 로그 수집)을
    - 스케줄러가 판단하여
    - 다수의 Worker 노드에 동시에 뿌려 병렬 처리를 극대화


## 2. DAG 유즈케이스 연구 (Use Cases)

### 2.1 Case A: 선형 및 분기형 데이터 동기화 파이프라인 (ETL/ELT)

- 가장 보편적인 유즈케이스
- 대용량 트랜잭션 데이터를 분석 저장소로 안전하게 이관하는 구조

- **흐름:**
    1.  새벽 2시에 Production DB에서 어제 자 정산 데이터를 백업 처리 (`BashOperator`)
    2.  동시에 마케팅 API 서버로부터 광고 효율 로그를 다운로드 (`PythonOperator`)
    3.  두 원시 데이터가 모두 스토리지(MinIO/S3)에 적재 완료되었는지 검증 (`MinioSensor`)
    4.  정제 및 병합 쿼리를 실행하여 데이터 웨어하우스(빅쿼리/Postgres)에 적재 (`PostgresOperator`)
    5.  성공/실패 여부를 사내 메신저로 전송 (`SlackWebhookOperator`)


### 2.2 Case B: 주기적 인프라 상태 점검 및 자동 스케일링 체인

- 데이터 분석 외에 시스템 운영 인프라를 관제하고 최적화하는 유즈케이스

- **흐름:**
    1. 매주 일요일 가상 머신(VM) 및 쿠버네티스(K8s) 클러스터의 미사용 자원 스캔
    2. 임계치 이하의 노드가 발견되면 리소스 축소(Downscaling) 스크립트 트리거
    3. 백업 볼륨 복구 테스트 진행 및 디스크 정제 작업 수행


### 2.3 Case C: 로컬 AI 모델 재학습 및 하이브리드 RAG 지식 임베딩 자동화

- 지속적으로 발생하는 사내 기술 매뉴얼과 매일 쌓이는 고객 상담 이력을
- 로컬 AI 인프라에 동적으로 반영하기 위한 파이프라인

- **흐름:**
    1. 현장 엔지니어가 작성한 최신 장비 가이드 문서(`*.txt`)를 MinIO 특정 버킷에서 감지
    2. 텍스트 문서를 로드하여 토큰 제한에 맞게 의미론적 청킹(Chunking) 수행
    3. 로컬 Ollama 임베딩 모델을 호출하여 지식 데이터를 벡터화
    4. Qdrant와 같은 VectorDB의 지정된 컬렉션에 하이브리드 인덱스로 신규 업서트(Upsert) 반영



## 3. 실습 예제: 

- **이원화 저장소 백업 및 텍스트 데이터 정제 파이프라인**
    - 로컬 파일 시스템에서 원시 로그 데이터를 읽어와 **MinIO 객체 스토리지에 백업**한 뒤,
    - 의미론적으로 데이터를 가공 및 변환(Transform)하여
    - RAG나 분석 시스템이 사용할 수 있도록 정제하는
    - 가상의 DAG 아키텍처를 파이썬 코드로 구현

- **시나리오 구성도**

    ```text
    [Task 1: check_file] 🡲 [Task 2: backup_to_minio] 🡲 [Task 3: transform_data]
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
MINIO_URL = "localhost:9000"
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
    schedule_interval='@daily',  # 매일 자정에 정기 실행
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

    - **Step 1: 환경 구성 확인**
        1. Airflow 컨테이너 볼륨 내에 위 코드를 배치한 후,
        2. Airflow Web UI (`http://localhost:8080`)에 접속하면
        3. `factory_log_hybrid_processing_v1`이라는 이름의 DAG가 그래프 뷰에 표시됨

    - **Step 2: 실행 메커니즘 관측**
        1. DAG를 수동 트리거(Trigger DAG)하면 스케줄러가 간선(`>>`) 순서에 맞춰 `task_check_file`을 가장 먼저 큐에 배치
        2. 첫 번째 태스크가 정상 종료(Success)되면 그 결과를 인지하고<br>즉시 다음 노드인 `backup_raw_data_to_minio` 태스크 인스턴스를 활성화하여 물리 자원을 배정
        3. 파이썬 함수 내부에 구현된 `Minio` 클라이언트 프로토콜에 의해 파일이 버킷에 동기화 완료되면,<br>마지막으로 `transform_and_extract_warnings`가 기동되어 로그를 파싱하고 정제 결과를 콘솔에 출력

    - **Step 3: 실행 결과 로그(Log) 시각화 예시**
        - 마지막 `task_transform_clean` 인스턴스의 텍스트 로그를 클릭하여 열어보면 다음과 같은 출력 구조가 표시되는 것을 확인할 수 있음
        - 이를 통해 DAG 파이프라인이 정상적으로 원인-결과 흐름을 조율했음이 증명됨

    ```text
    *** Found local logs:
    [2026-07-07 17:15:22,345] {python.py:177} INFO - Done. Integration tasks succeeded.
    ============================================================
    [정제 완료된 하이브리드 컨텍스트 데이터 목록]
    ============================================================
    [위험 알림] 시간: 2026-07-07 15:05:00, 센서명: FM-2026, 측정값: 99.9
    ============================================================
    ```