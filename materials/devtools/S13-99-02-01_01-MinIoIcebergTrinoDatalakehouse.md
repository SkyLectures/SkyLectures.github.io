---
layout: page
title:  "데이터 카탈로그 연결 및 레이크하우스 통합"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-99-02-01_01-MinIoIcebergTrinoDatalakehouse
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 실습의 핵심 목적

- **주제:**
    - MinIO와 Apache Iceberg를 활용한 클라우드 네이티브 데이터 레이크하우스 구축 및 분석 엔진 통합

- **세부내용:**
    - MinIO를 이용한 S3 호환 오브젝트 스토리지 환경 조성
    - Apache Iceberg를 활용한 트랜잭션 지원 스토리지 레이어 구축
    - Trino(Presto)를 통한 고성능 분산 SQL 쿼리 엔진 연결
    - 데이터 카탈로그 기반의 통합 메타데이터 관리 실습

<br>

1. **데이터 저장의 표준화 및 인프라 자립성 확보 (MinIO/S3)**
    - **목적:**
        - 특정 퍼블릭 클라우드 벤더에 종속되지 않고,
        - 사내(On-Premise) 및 로컬 환경에 독립적인 고성능 S3 호환 오브젝트 스토리지를 구축

    - **의의:**
        - 클라우드 비용(Egress 요금 및 저장 비용) 부담 없이
        - 대규모 비정형 데이터를 표준화된 규격으로 안전하게 저장·관리하는 인프라 자립 능력을 배양

<br>

2. **데이터 정합성 확보 및 완결성 있는 스토리지 레이어 구현 (Apache Iceberg)**
    - **목적:**
        - 대용량 오브젝트 스토리지의 한계(데이터 수정/삭제의 비효율성 및 일관성 결여)를 극복하기 위해,
        - 오픈 테이블 포맷인 Apache Iceberg를 도입하여
        - ACID 트랜잭션과 타임 트래블(이력 관리) 기능을 구현

    - **의의:**
        - 단순히 파일을 쌓아두는 '데이터 레이크'의 유연성과
        - 구조적 쿼리가 가능한 '데이터 웨어하우스'의 엄격한 데이터 관리 능력을 결합한
        - '레이크하우스(Lakehouse)'의 원리를 이해하고 데이터 신뢰성을 보장

<br>

3. **연합 쿼리 최적화 및 고성능 분산 분석 환경 구축 (Trino)**

* **목적:** 대규모 레이크하우스에 저장된 데이터를 물리적인 이동이나 복사 없이(In-place), 표준 ANSI SQL을 사용하여 초고속으로 조회·분석할 수 있는 분산 컴퓨팅 엔진을 연결합니다.
* **의의:** 컴퓨팅(Trino)과 스토리지(MinIO)가 철저히 분리된 대규모 분산 아키텍처를 이해하고, 다양한 데이터 소스를 단일 인터페이스로 통합하는 연합 쿼리(Federated Query) 역량을 갖춘다.

---

## 2. 학습 단계별 연결 고리

이 과정은 무형의 비정형 데이터가 유의미한 비즈니스 인사이트로 정제되는 전 과정을 아키텍처 계층별로 수직 통합합니다.

1. **Storage Layer (MinIO):** raw 데이터(로그, 이미지, CSV 등)를 물리적으로 안전하고 빠르게 저장할 수 있는 하부 토대를 마련합니다.
2. **Table Format Layer (Apache Iceberg):** 단순 파일 덩어리에 고성능 메타데이터 계층을 입혀 고성능 검색이 가능한 '구조적 테이블' 형태로 승격시키고 데이터의 원자성을 부여합니다.
3. **Catalog Layer (Hive/REST Catalog 등):** 데이터의 스키마와 위치 정보를 중앙에서 관리하여, 상위 컴퓨팅 엔진이 데이터를 정확하게 찾아갈 수 있는 이정표를 제공합니다.
4. **Compute Layer (Trino):** 최종 분석가나 애플리케이션이 익숙한 SQL 표준 문법을 통해 초고속으로 데이터에 접근하고 가치를 추출하는 최상위 관문을 완성합니다.

---

## 3. 기대 효과

* **개발자:** Amazon AWS 계정이나 비용 지출 없이도 로컬 환경에서 클라우드 급 대용량 스토리지 연동 규격(S3 API)을 완벽히 마스터하고, 대규모 파일 업로드/다운로드 아키텍처를 독립적으로 설계할 수 있습니다.
* **데이터 엔지니어:** 데이터 레이크의 고질적인 문제인 정합성 오류를 해결하고, 컴퓨팅과 스토리지를 분리하여 독립적으로 확장(Scale-out)할 수 있는 차세대 데이터 플랫폼(Modern Data Stack)의 엔드투엔드(End-to-End) 구축 역량을 확보합니다.

---

## 4. 향후 과정을 위한 총괄평가

본 4주차 과정은 단순한 오픈소스 도구의 사용법을 넘어, "엔터프라이즈 환경에서 비용 효율적이고 유연한 대용량 데이터 플랫폼을 어떻게 자립적으로 설계할 것인가?"에 대한 구조적 해답을 제시합니다.

여기서 확보한 'MinIO-Iceberg-Trino' 파이프라인 구축 경험은, 향후 대규모 인공지능 모델 학습을 위한 데이터 공급망(LLMOps/MLOps)을 고도화하거나 기업 내 파편화된 데이터 인프라를 하나로 통합하는 고성능 데이터 아키텍처 설계의 강력한 이정표가 될 것입니다.






---

## 1. 개념 설명 및 두 도구의 연관성

### ① 왜 MinIO인가? (오픈 데이터 레이크)

MinIO는 대규모 고성능 프라이빗 클라우드 인프라를 위한 오픈소스 **오브젝트 스토리지**입니다. AWS S3 API와 100% 호환되며, 파일 시스템(HDFS)과 달리 비정형·반정형 데이터를 페타바이트(PB) 규모로 저렴하고 안전하게 저장할 수 있어 **데이터 레이크의 물리적 저장소** 역할을 합니다.

### ② 왜 Apache Iceberg인가? (트랜잭션 계층)

오브젝트 스토리지는 본질적으로 '단순히 파일을 밀어 넣는 공간'일 뿐입니다. 데이터 수정/삭제가 어렵고, 여러 컴포넌트가 동시에 접근하면 데이터가 꼬입니다. Iceberg는 이 MinIO 위에서 작동하며 "RDBMS처럼 테이블 단위 관리, ACID 트랜잭션 보장, 고속 인덱싱(Data Skipping)"을 가능하게 만드는 테이블 규격(Table Format)입니다.

### ③ 두 도구의 연관성 및 시너지

* **디스크와 OS의 관계:** MinIO가 하드디스크(물리적 저장 공간)라면, Iceberg는 파일을 논리적으로 관리하고 주소를 지정해 주는 파일 시스템(예: NTFS, ext4)과 같습니다.
* **스토리지와 컴퓨팅의 분리:** MinIO + Iceberg 구조를 가져가면 데이터는 한곳(MinIO)에 안전하게 묶여 있고, 데이터를 처리하는 엔진(Spark, DuckDB, Flink 등)은 필요에 따라 자유롭게 붙였다 뗄 수 있는 **완벽한 엔진 독립성**을 확보하게 됩니다.

---

## 2. 아키텍처 및 핵심 기술

이 파이프라인을 관통하는 아키텍처는 아래와 같이 명확한 계층 분리를 가집니다.

* **데이터 소스 (Ingestion Layer):** 소스 데이터를 파이프라인으로 밀어 넣어주는 주체 (Python, Logstash 등).
* **컴퓨팅/쿼리 엔진 (Compute Layer):** 데이터 분석 및 변환을 수행하는 가벼운 OLAP 엔진 (DuckDB, PyArrow).
* **메타데이터 카탈로그 (Catalog Layer):** Iceberg REST Catalog가 현재 최신 스냅샷 위치를 메모리/DB에 포인터로 쥐고 관리함.
* **오브젝트 스토리지 (Storage Layer):** 최하단에서 실제 메타데이터 파일(`.json`, `.avro`)과 원시 데이터 파일(`.parquet`)을 영구 보관하는 MinIO.

---

## 3. 데이터 파이프라인 연동 환경 설정 (`docker-compose.yml`)

엔지니어링 환경을 독립적으로 검증하기 위해 MinIO 스토리지와 Iceberg REST 카탈로그를 도커로 연동합니다.

```yaml
version: '3.8'

services:
  # 1. 뼈대 저장소: MinIO
  minio:
    image: minio/minio:RELEASE.2024-01-11T06-46-16Z
    container_name: pipeline-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=pipeline_admin
      - MINIO_ROOT_PASSWORD=pipeline_password
    command: server /data --console-address ":9001"

  # MinIO 초기 버킷 생성 유틸
  mc:
    image: minio/mc:RELEASE.2024-01-11T06-46-16Z
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      until (/usr/bin/mc alias set myminio http://minio:9000 pipeline_admin pipeline_password) do echo 'Waiting for MinIO...' && sleep 1; done;
      /usr/bin/mc mb myminio/telemetry-lake;
      exit 0;
      "

  # 2. 메타데이터 통제 센터: Iceberg REST Catalog
  iceberg-catalog:
    image: tabulario/iceberg-rest:0.6.0
    container_name: pipeline-catalog
    ports:
      - "8181:8181"
    environment:
      - CATALOG_WAREHOUSE=s3a://telemetry-lake/
      - CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO
      - CATALOG_S3_ENDPOINT=http://minio:9000
      - CATALOG_S3_PATH_STYLE_ACCESS=true
      - AWS_ACCESS_KEY_ID=pipeline_admin
      - AWS_SECRET_ACCESS_KEY=pipeline_password
    depends_on:
      - minio

```

터미널 실행 명령어: `docker-compose up -d`
필수 파이썬 라이브러리: `pip install "pyiceberg[s3fs,pyarrow]" duckdb`

---

## 4. 데이터 파이프라인 시나리오 및 통합 예제 코드

**[시나리오]**
스마트시티의 교량/도로에 설치된 붕괴 위험 감지 IoT 센서 데이터 파이프라인을 구축합니다.

1. **1차 적재 (Batch 1):** 정상 작동 중인 센서 로그 데이터가 파이프라인을 통해 들어옵니다.
2. **2차 적재 (Batch 2):** 특정 시점에 위험 징후를 나타내는 급격한 변동 데이터가 유입됩니다.
3. **가상 DW 분석:** 고속 OLAP 엔진(DuckDB)을 연동해 이상 징후 센서를 즉시 격리 추출하는 SQL 파이프라인을 구동합니다.
4. **타임 트래블 분석:** 분석가는 사고 발생 이전 상태(스냅샷 V1)의 데이터와 현재 상태를 교차 검증합니다.

```python
import datetime
import pyarrow as pa
from pyiceberg.catalog import load_catalog
import duckdb

# =========================================================================
# [단계 1] 연동 설정 - PyIceberg를 통한 MinIO 및 REST 카탈로그 결합
# =========================================================================
print("🔗 1. MinIO 스토리지 및 Iceberg 카탈로그 연결 파이프라인 초기화...")
catalog = load_catalog(
    "city_telemetry_catalog",
    **{
        "type": "rest",
        "uri": "http://localhost:8181",
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "pipeline_admin",
        "s3.secret-access-key": "pipeline_password",
    }
)

# 네임스페이스(DB 구조) 및 원천 스키마 정의
catalog.create_namespace("infrastructure")

bridge_schema = pa.schema([
    ("bridge_id", pa.string()),
    ("vibration_level", pa.float64()),
    ("stress_index", pa.float64()),
    ("status", pa.string()),
    ("checked_at", pa.timestamp('us', tz='UTC'))
])

# Iceberg 파이프라인 테이블 생성
table = catalog.create_table(
    identifier="infrastructure.bridge_telemetry",
    schema=bridge_schema
)
print("✅ 테이블 규격 생성 완료. (MinIO 버킷 내 메타데이터 트리 루트 확보)")


# =========================================================================
# [단계 2] 데이터 유입 (Batch 1 - 정상 주기 데이터 파이프라인)
# =========================================================================
print("\n📥 2. [Batch 1] 사물인터넷(IoT) 센서 주기 데이터 파이프라인 구동...")

batch_1_raw = {
    "bridge_id": ["BRG-01", "BRG-02", "BRG-01", "BRG-03"],
    "vibration_level": [1.2, 0.8, 1.5, 2.1],
    "stress_index": [10.5, 8.2, 11.1, 14.0],
    "status": ["STABLE", "STABLE", "STABLE", "STABLE"],
    "checked_at": [datetime.datetime.now(datetime.timezone.utc) for _ in range(4)]
}
# 고속 파일 처리를 위해 PyArrow 레코드셋 변환
batch_1_arrow = pa.Table.from_pydict(batch_1_raw)

# MinIO에 파일 쓰기 및 커밋 동시 수행
table.append(batch_1_arrow)
print("✅ Batch 1 데이터가 실시간 적재되었습니다. (스냅샷 ID 갱신 완료)")


# =========================================================================
# [단계 3] 데이터 유입 (Batch 2 - 이상 데이터 감지 및 누적 적재)
# =========================================================================
print("\n⚠️ 3. [Batch 2] 교량 센서 이상 진동 데이터 실시간 유입 트래킹...")

batch_2_raw = {
    "bridge_id": ["BRG-01", "BRG-04"],
    "vibration_level": [8.9, 1.1],  # BRG-01 센서의 위험 수치 급증
    "stress_index": [78.4, 9.0],
    "status": ["DANGER", "STABLE"],
    "checked_at": [datetime.datetime.now(datetime.timezone.utc) for _ in range(2)]
}
batch_2_arrow = pa.Table.from_pydict(batch_2_raw)

# 추가 커밋 발생 (기존 데이터와 완전 격리된 새 스냅샷 파일 생성)
table.append(batch_2_arrow)
print("✅ Batch 2 데이터 적재 완료. (안전하게 트랜잭션 격리됨)")


# =========================================================================
# [단계 4] 데이터 레이크하우스 쿼리 분석 파이프라인 (DuckDB 연동)
# =========================================================================
print("\n🦅 4. 고속 DW 엔진(DuckDB) 가속화를 통한 분석 파이프라인 작동...")

# Iceberg가 메타데이터를 기반으로 최신 유효 파켓 파일 구조만 스캔
latest_lakehouse_view = table.scan().to_arrow()

# DuckDB를 사용하여 제로카피 고속 분석 쿼리 수행
db_client = duckdb.connect(database=':memory:')

print("\n[분석 리포트 1] 실시간 교량 상태별 평균 스트레스 지수 점검:")
m_query1 = """
    SELECT status, COUNT(*) as log_count, ROUND(AVG(stress_index), 2) as avg_stress
    FROM latest_lakehouse_view
    GROUP BY status;
"""
db_client.execute(m_query1).show()


# =========================================================================
# [단계 5] 오염 역추적을 위한 파이프라인 타임 트래블
# =========================================================================
print("\n🕰️ 5. 파이프라인 역추적 검증: 이상 징후가 보고되기 전(Batch 1) 시점으로 롤백 쿼리...")

# 변경 이력 트랙커 가동
pipeline_history = table.history()
initial_version_id = pipeline_history[0].snapshot_id

# 과거 스냅샷 기준 파일 필터링 로드
past_lakehouse_view = table.scan(snapshot_id=initial_version_id).to_arrow()

print(f"\n[분석 리포트 2] 초기 버전(Snapshot: {initial_version_id}) 내부 DANGER 로그 건수:")
m_query2 = "SELECT COUNT(*) as danger_count FROM past_lakehouse_view WHERE status = 'DANGER';"
db_client.execute(m_query2).show()

```

---

## 5. 파이프라인 핵심 메커니즘 심층 해설

위의 연동 코드와 아키텍처가 실제 백엔드(MinIO 내부)에서 동작할 때 일어나는 **데이터 엔지니어링의 핵심 사항**입니다.

### ① 파이프라인의 데이터 파일 저장 흐름 (MinIO 내부의 실체)

`table.append()`가 호출되면 다음과 같은 연쇄 반응이 오브젝트 스토리지 내부에서 일어납니다.

* 데이터가 압축된 `.parquet` 파일로 MinIO의 `data/` 경로에 먼저 안착합니다.
* 파이프라인은 이 파켓 파일의 컬럼별 통계 정보(예: `vibration_level`의 최솟값/최댓값)를 뽑아서 `.avro` 포맷의 **Manifest 파일**로 저장합니다.
* 최종적으로 카탈로그 서버에 "새로운 상태 포인터(.json 파일)"를 원자적(Atomic)으로 갱신 요청합니다.

### ② Data Skipping (데이터 스키핑)을 통한 고속화 기술

DuckDB 쿼리 엔진이 `WHERE status = 'DANGER'`를 요청할 때, 일반 데이터 레이크는 MinIO에 있는 모든 파켓 파일을 다운로드해서 스캔해야 하므로 엄청난 I/O 비용이 듭니다.
반면 Iceberg 파이프라인은 **Manifest 레이어의 통계 정보(Min/Max)만 먼저 스캔**하여, `status` 컬럼에 'DANGER'라는 단어가 포함될 가능성이 전혀 없는 파일들을 무더기로 걸러내고(Skipping), **오직 해당 레코드가 있는 단 하나의 파켓 파일만 MinIO에서 콕 집어 가져옵니다.** 이것이 기가바이트(GB) 대역에서 수 밀리초(ms) 만에 SQL 처리가 끝나는 비결입니다.

### ③ 완벽한 데이터 엔지니어링 파이프라인의 혜택

* **무중단 스키마 진화:** IoT 센서가 업그레이드되어 데이터 구조에 새로운 컬럼(예: `temperature`)이 생기더라도, 과거 파켓 파일을 단 하나도 건드리지 않고 카탈로그의 메타데이터만 갱신하여 신구 데이터를 즉시 병합해 냅니다.
* **배치 및 실시간 파이프라인의 조화:** 쓰는 녀석(Ingestion)이 계속 뒤에서 파일을 밀어 넣고 있어도, 읽는 분석가(DuckDB)는 자기가 쿼리를 시작한 시점의 스냅샷만 바라보기 때문에 **"더러운 읽기(Dirty Read)" 현상이 완벽하게 예방**됩니다.