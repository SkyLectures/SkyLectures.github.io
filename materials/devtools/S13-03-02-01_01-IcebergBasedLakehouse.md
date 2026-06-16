---
layout: page
title:  "Apache Iceberg 기반 레이크하우스 구축"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-03-02-01_01-IcebergBasedLakehouse
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - Apache Iceberg 기반의 레이크하우스(Lakehouse)를 구축한다는 것은
>   - 값싸고 유연한 오브젝트 스토리지(Data Lake) 위에,
>   - RDBMS처럼 고성능 인덱싱과 ACID 트랜잭션이 가능한 관리 계층(Data Warehouse)을 결합하는 것
{: .common-quote}


## 1. Apache Iceberg 기반 레이크하우스 아키텍처 개요

- **[Python(PyIceberg) + DuckDB + MinIO]** 조합으로 모던 레이크하우스 구축하기
    - **DuckDB**는 초고속 가상 데이터 웨어하우스 엔진 역할
    - **PyIceberg**는 테이블 포맷 계층 담당

- 구축할 아키텍처는 현대적인 오픈 레이크하우스의 표준 3계층 구조를 따름
    - **Storage Layer (MinIO):**
        - 대용량 원시 데이터(Raw Parquet)와 Iceberg 메타데이터 파일들이 저장되는 오픈 데이터 레이크
    - **Table Format Layer (SQLite Virtual Catalog):**
        - 외부 자바 브릿지 서버 없이,
        - 파이썬 가상환경 내부에서 SQL 데이터베이스 원장 파일(sqlite:///)을 통해 어떤 데이터 파일이 유효한지
        - 스냅샷과 트랜잭션을 격리하고 제어하는 두뇌 역할
    - **Compute/Query Layer (DuckDB + PyIceberg):**
        - 저장된 데이터 레이크하우스 테이블을 고속으로 가상 분석(OLAP)하고 SQL을 실행하는 엔드포인트


## 2. 실습 환경 준비 (Docker Compose)

### 2.1 Docker 환경 구성

- 작업 디렉터리에 `docker-compose.yml` 파일을 생성
    - 데이터 레이크하우스의 핵심 인프라(스토리지 및 카탈로그)를 구동
        - 복잡하고 불투명한 외부 자바 카탈로그 서버 엔진을 완전히 걷어내고,
        - 오직 데이터 레이크하우스의 핵심 저장소 인프라(순수 S3 오브젝트 스토리지)만 구동

    ```yaml
    version: '3.8'

    services:
    # 1. 스토리지 계층 (S3 호환 MinIO)
    minio:
        image: minio/minio:latest
        container_name: minio
        ports:
            - "9000:9000"       # API 엔드포인트 포트
            - "9001:9001"       # 웹 관리 콘솔 포트
        environment:
            - MINIO_ROOT_USER=admin
            - MINIO_ROOT_PASSWORD=password
        command: server /data --console-address ":9001"

    # 2. MinIO 기동 시 'warehouse' 버킷을 자동으로 생성해주는 유틸리티
    mc:
        image: minio/mc:latest
        container_name: minio-mc
        depends_on:
            - minio
        entrypoint: >
            /bin/sh -c "
            until (/usr/bin/mc alias set myminio http://minio:9000 admin password); do 
                echo 'Waiting for MinIO...'; 
                sleep 1; 
            done;
            /usr/bin/mc mb myminio/warehouse;
            exit 0;"
    ```

- 터미널에서 `docker compose up -d`를 실행하여 서버 활성화


### 2.2 로컬 파이썬 환경 설정

- **가상환경 생성 및 활성화**

```bash
python -m venv datalake
cd datalake
source ./bin/activate
```


- **레이크하우스를 제어할 라이브러리 설치**
    - 자바 서버 없이 파이썬 내장에서 자체적으로 SQL 카탈로그 이정표 원장을 핸들링할 수 있도록
    - [sql-sqlite] (SQLAlchemy 코어 엔진 라이브러리 세트) 확장 팩을 함께 주입

    ```bash
    pip install "pyiceberg[s3fs,pyarrow]" duckdb python-faker numpy pandas
    ```


### 2.3 실습 데이터 생성 및 레이크하우스 구축

- **전체 파이프라인**
    1. 대규모 가상 센서 데이터(1만 건)를 실시간으로 생성
    2. Iceberg 테이블 스키마를 정의하여 레이크하우스에 적재
    3. DuckDB를 결합하여 고속 SQL 분석 및 타임 트래블을 수행

- **예제 코드**

```python
#//file: "lakehouse_demo.py"
import datetime
import random
from faker import Faker
import pyarrow as pa
from pyiceberg.catalog import load_catalog
import duckdb

# 라이브러리 초기화
fake = Faker()
Faker.seed(42)
random.seed(42)

# ==========================================
# [STEP 1] 가상 데이터 생성기 (Data Generator)
# ==========================================
def generate_sensor_data(num_records=10000):
    """스마트팩토리 센서 로그 데이터를 대량 생성하는 함수"""
    sensor_ids = [f"SNS-{i:03d}" for i in range(1, 21)]  # 20개의 센서 기기
    locations = ["Line-A", "Line-B", "Line-C", "Packaging"]
    
    data = {
        "log_id": [fake.uuid4() for _ in range(num_records)],
        "sensor_id": [random.choice(sensor_ids) for _ in range(num_records)],
        "location": [random.choice(locations) for _ in range(num_records)],
        "reading": [round(random.uniform(10.0, 95.0), 2) for _ in range(num_records)],
        "status": [random.choice(["NORMAL", "NORMAL", "NORMAL", "WARN", "ERROR"]) for _ in range(num_records)],
        "timestamp": [
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=random.randint(0, 10000))
            for _ in range(num_records)
        ]
    }
    # PyArrow Table 형태로 변환 (Iceberg와 직접 호환)
    return pa.Table.from_pydict(data)

print("1. 10,000건의 스마트팩토리 가상 데이터 생성 중...")
raw_arrow_data = generate_sensor_data(10000)


# ==========================================
# [STEP 2] Iceberg 레이크하우스 테이블 구축 
# ==========================================
print("2. 내장 SQL 카탈로그 및 물리 오브젝트 스토리지 다이렉트 바인딩 중...")
# 외부 자바 이미지의 버그 및 상용화 꼼수 간섭을 배제하기 위해 로컬 가상 SQLite 원장으로 우회 연결 설정
catalog = load_catalog(
    "default",
    **{
        "type": "sql",
        "uri": "sqlite:///iceberg_catalog.db",        # 실행하는 현재 폴더 밑에 가볍고 무결한 .db 파일 관리 원장 생성
        "warehouse": "s3://warehouse",                # 도커 MinIO 내부의 물리 데이터 영토 버킷 지정
        "s3.endpoint": "http://localhost:9000",       # 호스트에서 도커 스토리지로 직격하는 인터페이스 통로
        "s3.path-style-access": "true",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO"
    }
)

# 네임스페이스(DB) 및 테이블 구조(Schema) 정의
catalog.create_namespace("factory_analytics")

iceberg_schema = pa.schema([
    ("log_id", pa.string()),
    ("sensor_id", pa.string()),
    ("location", pa.string()),
    ("reading", pa.float64()),
    ("status", pa.string()),
    ("timestamp", pa.timestamp('us', tz='UTC'))
])

# Iceberg 테이블 생성
iceberg_table = catalog.create_table(
    identifier="factory_analytics.sensor_history",
    schema=iceberg_schema
)

# 첫 번째 데이터 적재 (Batch 1 - 최초 1만 건) -> 스냅샷 V1 발생
iceberg_table.append(raw_arrow_data)
print("최초 데이터 10,000건이 레이크하우스에 안정적으로 적재되었습니다. (Snapshot V1)")


# ==========================================
# [STEP 3] 데이터 추가 적재 (트랜잭션 및 스냅샷 V2 생성)
# ==========================================
print("\n3. 실시간 추가 데이터(이상치 데이터 5건) 적재 중...")
extra_data = pa.Table.from_pydict({
    "log_id": [fake.uuid4() for _ in range(5)],
    "sensor_id": ["SNS-999"] * 5,  # 추적을 용이하게 하기 위한 가상 테스트 ID
    "location": ["Control-Room"] * 5,
    "reading": [999.9] * 5,
    "status": ["CRITICAL"] * 5,
    "timestamp": [datetime.datetime.now(datetime.timezone.utc) for _ in range(5)]
})
iceberg_table.append(extra_data)
print("추가 데이터 5건이 적재되었습니다. (Snapshot V2)")


# ==========================================
# [STEP 4] DuckDB를 이용한 고속 가상 DW 분석
# ==========================================
print("\n4. DuckDB 엔진을 결합하여 가상 데이터 웨어하우스 SQL 분석 시작...")

# 최신 상태(Snapshot V2)의 테이블 스캔 데이터를 Arrow 객체로 로드
current_snapshot_data = iceberg_table.scan().to_arrow()

# DuckDB가 메모리 상에서 Iceberg Arrow 데이터를 직접 SQL로 쿼리 (복사 없음, 제로카피)
con = duckdb.connect(database=':memory:')

print("\n[SQL 결과 1] 라인(Location)별 평균 센서 값 및 로그 개수 요약:")
query1 = """
    SELECT location, COUNT(*) as log_count, ROUND(AVG(reading), 2) as avg_reading
    FROM current_snapshot_data
    GROUP BY location
    ORDER BY log_count DESC;
"""
con.execute(query1).show()

print("\n[SQL 결과 2] 새로 추가된 'CRITICAL' 상태 데이터 검증:")
query2 = "SELECT sensor_id, location, reading, status FROM current_snapshot_data WHERE status = 'CRITICAL';"
con.execute(query2).show()


# ==========================================
# [STEP 5] 데이터 웨어하우스 핵심: 타임 트래블(Time Travel) 검증
# ==========================================
print("\n5. 레이크하우스 기능 검증: 최초 데이터 적재 시점(Snapshot V1)으로 타임 트래블...")

# 테이블 변경 이력에서 첫 번째 스냅샷 ID 추출
history = iceberg_table.history()
v1_snapshot_id = history[0].snapshot_id

# 과거 시점의 스냅샷 데이터를 조회하도록 지정
v1_snapshot_data = iceberg_table.scan(snapshot_id=v1_snapshot_id).to_arrow()

print(f"\n[SQL 결과 3] Snapshot V1 시점 기준 'CRITICAL' 데이터 조회 (결과는 0건이어야 함):")
query3 = "SELECT COUNT(*) as critical_count FROM v1_snapshot_data WHERE status = 'CRITICAL';"
con.execute(query3).show()
```


- **예제 코드의 중요사항 해설**
    - 기존의 단순 파일 저장 방식(Data Lake)과 Iceberg 레이크하우스의 차별점을 명확히 보여주는 몇 가지 핵심 설계 요소를 포함하고 있음

    1. **`load_catalog()`의 역할: 가볍고 통제력이 강한 임베디드 카탈로그 레이어**

        ```python
        catalog = load_catalog("default", **{"type": "sql", "uri": "sqlite:///iceberg_catalog.db", ...})
        ```

        - 전통적인 데이터 레이크는 분석가가 파일 경로(s3://warehouse/path/to/parquet/)를 직접 알아야 쿼리를 할 수 있어 보안 및 관리가 파편화됨
        - Iceberg 레이크하우스는 무겁고 네트워크 간섭이 심한 중앙 자바 서버 대신, **파이썬 내부의 가볍고 무결한 SQLite 관계형 메타데이터 원장을 독점적(Embedded)으로 빌드**하여 제어권을 일원화함
        - 분석가는 복잡한 파일 토폴로지 대신 **RDBMS처럼 카탈로그에 테이블 명세**(factory_analytics.sensor_history)만 요청하면, 카탈로그 내부 원장이 물리적 파일 위치를 자동 스캔하여 숨겨주므로 아키텍처 보안 및 거버넌스 통제가 극도로 명쾌해짐

    2. **`pa.Table`과 `table.append()`: 원자적 트랜잭션(Atomic Commit)**

        ```python
        iceberg_table.append(raw_arrow_data)
        ```

        - 데이터를 저장소에 밀어 넣을 때, Iceberg는 낙관적 동시성 제어(OCC) 메커니즘을 사용
            - Parquet 파일 쓰기가 완벽하게 끝나고 메타데이터 포인터가 새 `.json` 파일로 교체(Swap)되는 순간에만 데이터가 외부에 노출됨
        - 중간에 네트워크 장애가 나거나 쓰기가 실패해도 **기존 데이터 레이크하우스의 정합성이 절대 깨지지 않고**, 롤백이나 오염이 발생하지 않음

    3. **DuckDB와의 융합: 복사 없는 고속 쿼리 (Zero-Copy 통합)**

        ```python
        current_snapshot_data = iceberg_table.scan().to_arrow()
        con.execute("SELECT ... FROM current_snapshot_data")
        ```

        - 현대 레이크하우스 아키텍처의 핵심
            - Iceberg 메타데이터 파일 구조 덕분에
            - 엔진은 필요한 파일 목록과 Min/Max 인덱스 통계를 메모리에 즉시 로드
        - 이를 메모리 공유 포맷인 **PyArrow** 데이터 구조로 받아 DuckDB에 넘기면,
            - DuckDB는 데이터를 디스크에서 다시 읽거나 포맷을 변환하는 가속 비용 없이
            - **메모리 상에서 C++ 속도로 무복사(Zero-Copy) 대화형 SQL을 수행**함
            - 대규모 비싼 DW(예: Snowflake, Oracle)를 대체할 수 있는 가성비 아키텍처의 근간

    4. **`snapshot_id` 기반 타임 트래블: 완벽한 데이터 재현성**

        ```python
        v1_snapshot_data = iceberg_table.scan(snapshot_id=v1_snapshot_id).to_arrow()
        ```

        - 분석 중 데이터 소스가 손상되었거나 ML 모델의 과거 학습 데이터셋을 그대로 재현해야 할 때, 파일 시스템을 백업해둘 필요가 없음
        - Iceberg 테이블에 **"과거 특정 시점의 스냅샷 상태로 보여달라"고 명시**하는 것만으로 가상 타임 트래블 뷰를 즉시 확보할 수 있음