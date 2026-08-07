---
layout: page
title:  "데이터 카탈로그 연결 및 레이크하우스 통합"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-99-02-01_01-MinIoIcebergTrinoDatalakehouse
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 실습 개요 및 기술적 배경

- 과거의 대용량 데이터 레이크(Data Lake) 구조
    - S3나 MinIO 같은 오브젝트 스토리지에 파일(`CSV`, `Parquet`)을 무작위로 던져두고 Hive Metastore를 통해 조회하는 방식
    - 파일 단위의 접근만 허용 🡲 **행(Row) 단위의 수정/삭제가 불가능**
    - 데이터가 수정되는 와중에 읽기 작업이 들어오면 데이터가 오염되는 비원자성(Non-atomic) 문제가 심각

- **실습 개요: 모던 데이터 스택의 조합**
    - 본 실습은 기존 구조의 한계를 극복하기 위해 **"Compute-Catalog-Storage 분리 아키텍처"**를 구현함

        - **Storage (MinIO):**
            - 값싸고 무한히 확장 가능한 S3 호환 오브젝트 스토리지

        - **Catalog (Iceberg REST Catalog):**
            - 테이블의 스냅샷과 데이터 파일들의 위치 지도를 관리
            - 오브젝트 스토리지를 관계형 DB(RDBMS)처럼 다룰 수 있게 매핑하는 가상 레이어

        - **Compute Engine (Trino):**
            - 데이터를 저장하지 않고 오직 분산 병렬(MPP) 메모리 연산만 수행
            - 수십억 건의 데이터를 초고속으로 SQL 질의·집계하는 엔진


## 2. 실습 시나리오 및 목표 (Scenario & Objectives)

- **시나리오 환경**
    - 스마트팩토리 내의 여러 생산 라인(`Line_A`, `Line_B`)에 탑재된 대규모 IoT 센서들이 매 순간 온도와 상태 데이터를 생성
    - 이 데이터는 백엔드 스토리지에 원시 파일 형태로 지속적으로 적재되는 상황

- **무엇을 하고 싶은가? (실습의 의도)**
    - **인프라 자립화 검증:**
        - 수동으로 웹 관리 창을 켜서 스토리지를 조작하는 구식 방식을 버리고,
        - 애플리케이션(Python) 레벨에서 스토리지(MinIO 버킷 생성), 메타데이터(Trino 스키마/테이블 선언), 데이터 파이프라인(DML 적재 및 OLAP 집계)을 **단 하나의 스크립트로 완전 자동화 제어**하는 통합 파이프라인 구축

    - **Heavy Lifting의 위임:**
        - 파이썬 메모리가 대용량 시계열 연산을 감당하지 않도록,
        - 모든 무거운 집계 연산(`AVG`, `COUNT`, `GROUP BY`)은 도커 내부의 **Trino 분산 엔진에게 전적으로 위임**
        - 파이썬은 오직 최종 정제된 요약본 결과(Pandas DataFrame)만 수급하여 가볍게 활용하는 아키텍처를 구현


## 3. 단계별 작업 내용 및 기술적 의미

- **0단계: MinIO 'warehouse' 물리 버킷 자동 개설**
    - **작업 내용:**
        - `boto3`를 이용해 MinIO에 Iceberg의 원천 저장소가 될 `warehouse` 버킷을 물리적으로 선행 생성

    - **기술적 의미:**
        - `iceberg-rest` 서버가 최초 기동 시 버킷을 직접 파주지 않는 특성을 파이썬 레이어에서 극복
        - 인프라 부재로 인한 파일 시스템 체킹 에러(`ICEBERG_FILESYSTEM_ERROR`)를 원천 차단하는 자립형 코드 구조 확보

- **1단계: 가상 스키마(Database) 선언**
    - **작업 내용:**
        - Trino를 통해 `iceberg.factory_db` 스키마 생성

    - **기술적 의미:**
        - 물리적인 폴더를 파는 행위를 넘어,
        - 중앙 REST Catalog 대장에 "앞으로 이 스토리지 영역을 factory_db라는 논리적 격리 공간으로 관리하겠다"라고 영구 등록하는 단계

- **2단계: Iceberg 포맷 테이블 구조 정의 (Schema Definition)**
    - **작업 내용:**
        - 데이터 타입, 타임스탬프 정의
        - `location` 컬럼을 기준으로 파티셔닝(`partitioning = ARRAY['location']`)을 명시하여 테이블 생성
        
    - **기술적 의미:**
        - **데이터 레이크에 던져질 파일 덩어리들에 '관계형 테이블'의 옷을 입히는 작업**
        - 향후 Trino가 `WHERE location = 'Line_A'`라는 쿼리를 받으면,
            - Iceberg의 메타데이터 지도를 보고
            - Line_B 관련 파일들은 네트워크 다운로드조차 하지 않고 스킵하는
            - 고속 파일 프루닝(Pruning)의 기반 마련

- **3단계: IoT 센서 데이터 정석 적재 (Data Ingestion)**
    - **작업 내용:**
        - 4건의 스마트팩토리 센서 시뮬레이션 데이터를 `INSERT` 문으로 주입

    - **기술적 의미:**
        - Iceberg 포맷의 **ACID 트랜잭션** 발동
        - 데이터는 MinIO 내부에 고도로 압축된 읽기 전용 `Parquet` 파일로 쪼개져 저장됨
        - 카탈로그 서버는 이 파일들의 주소를 가리키는 최신 '스냅샷(Snapshot) 대장'을 원자적(Atomic)으로 갱신

- **4단계: 분산 OLAP 집계 및 판다스 수송 (Data Delivery)**
    - **작업 내용:**
        - Trino에 `GROUP BY` 대용량 집계 쿼리를 날려,
        - 파이썬 판다스 데이터프레임으로 가공된 표를 출력

    - **기술적 의미:**
        - AI/ML 모델 학습이나 실시간 BI 대시보드(Tableau 등)로 데이터를 공급하는 최종 관문
        - 파이썬 백엔드는 8080 포트 하나만 바라보고 표준 SQL을 던졌을 뿐
        - 내부에서는 Trino가 분산 병렬로 파일 시스템을 긁어 계산한 최적의 데이터셋이 리턴됨


## 4. 실행 환경 설정

- **가상환경 생성 및 활성화**

```bash
python -m venv datalakehouse
cd datalakehouse
source ./bin/activate
```

- **기존 도커 내용 정리**

```bash
docker-compose down
docker volume prune -f
docker network prune -f
```
 
- **docker-compose.yml 구성**

```yaml
version: "3.9"

services:
  minio:
    image: minio/minio:RELEASE.2024-01-11T07-46-16Z
    container_name: demo-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"

  iceberg-rest:
    image: tabulario/iceberg-rest:latest
    container_name: demo-iceberg-rest
    ports:
      - "8181:8181"
    environment:
      CATALOG_WAREHOUSE: s3://warehouse/
      CATALOG_IO__IMPL: org.apache.iceberg.aws.s3.S3FileIO
      CATALOG_S3_ENDPOINT: http://demo-minio:9000
      CATALOG_S3_PATH__STYLE__ACCESS: "true"
      CATALOG_S3_ACCESS__KEY__ID: minioadmin
      CATALOG_S3_SECRET__ACCESS__KEY: minioadmin
      AWS_REGION: us-east-1
    depends_on:
      - minio

  trino:
    image: trinodb/trino:435
    container_name: demo-trino
    ports:
      - "8080:8080"
    volumes:
      - ./configs/trino/catalog:/etc/trino/catalog
    depends_on:
      - iceberg-rest
      - minio
```

- **iceberg.properties 구성**

```properties
connector.name=iceberg

iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://demo-iceberg-rest:8181
iceberg.rest-catalog.warehouse=s3://warehouse/

fs.native-s3.enabled=true
s3.endpoint=http://demo-minio:9000
s3.path-style-access=true
s3.aws-access-key=minioadmin
s3.aws-secret-key=minioadmin
s3.region=us-east-1
```

- Docker Compose 환경 실행

```bash
docker-compose up -d
docker ps
```


## 5. 자동 버킷 생성이 추가된 자립형 스크립트

- **라이브러리 설치 (터미널)**

```bash
pip install pandas boto3 trino
```
<br>

- **파이썬 코드 작성**

```python
#//file: "integrate.py"
import sys
import pandas as pd
import boto3
from botocore.client import Config
from trino.dbapi import connect

def init_minio_bucket():
    """MinIO에 Iceberg가 사용할 'warehouse' 버킷이 없다면 자동으로 물리 생성합니다."""
    print("\n[인프라 사전 검증] MinIO 'warehouse' 물리 버킷 확인 및 자동 생성")
    try:
        s3 = boto3.resource(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        bucket = s3.Bucket('warehouse')
        
        # 버킷이 존재하는지 체크 후, 없으면 생성
        if bucket.creation_date:
            print(" -> [SKIP] MinIO 내부에 'warehouse' 버킷이 이미 안전하게 존재합니다.")
        else:
            bucket.create()
            print(" -> [SUCCESS] MinIO 내부에 'warehouse' 버킷을 물리적으로 자동 개설했습니다.")
    except Exception as e:
        # 가끔 존재하는 버킷을 다시 헤드 체크할 때 발생하는 404 예외 등 방어
        try:
            s3.create_bucket(Bucket='warehouse')
            print(" -> [SUCCESS] MinIO 내부에 'warehouse' 버킷을 강제 생성 완료했습니다.")
        except Exception as ex:
            print(f" -> [INFO] 버킷 생성 확인 완료: {ex}")

def run_query(sql, fetch_res=True):
    """Trino 연결을 통해 SQL을 수행하고 결과를 반환합니다."""
    with connect(
        host='localhost', port=8080, user='python-agent',
        catalog='iceberg', schema='factory_db'
    ) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        if fetch_res and cursor.description:
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)
        return None

# ========================================================
# 스토리지 원천 인프라 제어까지 포함된 정석 파이프라인
# ========================================================
if __name__ == "__main__":
    print("[START] 스토리지 인프라 기반 스마트팩토리 파이프라인 정석 테스트를 시작합니다.")
    print("=" * 60)

    # 0단계: 물리 버킷 유무 조회 및 자동 생성 개설
    init_minio_bucket()

    # 1단계: 깨끗해진 공간에 스키마(Database) 신규 생성
    print("\n[1단계] 실습 스키마(Database) 신규 생성 단계")
    run_query("CREATE SCHEMA IF NOT EXISTS iceberg.factory_db", fetch_res=False)
    print("    -> [SUCCESS] factory_db 스키마가 성공적으로 생성되었습니다.")

    # 2단계: Iceberg 포맷 테이블 구조 정의 단계 (정석 명세)
    print("\n[2단계] Iceberg 포맷 테이블 구조 정의 단계")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS iceberg.factory_db.sensor_logs (
        device_id INT,
        location VARCHAR,
        temperature DOUBLE,
        status VARCHAR,
        timestamp TIMESTAMP,
        vibration DOUBLE
    )
    WITH (
        format = 'PARQUET',
        partitioning = ARRAY['location']
    )
    """
    run_query(create_table_sql, fetch_res=False)
    print("    -> [SUCCESS] 정상적인 오피셜 이름 'sensor_logs' 테이블이 정의되었습니다.")

    # 3단계: IoT 센서 데이터 정석 적재 단계
    print("\n[3단계] IoT 센서 데이터 적재 단계")
    # 중복 적재 방지를 위한 카운트 체크
    cnt_df = run_query("SELECT COUNT(*) as cnt FROM iceberg.factory_db.sensor_logs")
    if cnt_df['cnt'].iloc[0] == 0:
        insert_sql = """
        INSERT INTO iceberg.factory_db.sensor_logs VALUES 
        (101, 'Line_A', 72.5, 'NORMAL', TIMESTAMP '2026-06-17 17:00:00', NULL),
        (102, 'Line_B', 88.1, 'WARN',   TIMESTAMP '2026-06-17 17:01:00', NULL),
        (103, 'Line_A', 69.8, 'NORMAL', TIMESTAMP '2026-06-17 17:02:00', NULL),
        (105, 'Line_B', 82.3, 'NORMAL', TIMESTAMP '2026-06-18 11:30:00', NULL)
        """
        run_query(insert_sql, fetch_res=False)
        print("    -> [SUCCESS] 4건의 시나리오 센서 데이터가 MinIO 레이크하우스에 깔끔하게 적재되었습니다.")
    else:
        print(f"    -> [SKIP] 이미 데이터가 {cnt_df['cnt'].iloc[0]}건 적재되어 있어 인서트를 건너뜁니다.")

    # 4단계: [AI/BI 연계] 최종 OLAP 집계 통계 결과 수급
    print("\n[4단계] 최종 분산 OLAP 집계 및 판다스 변환 데이터 마트 출력")
    analysis_sql = """
    SELECT 
        location, 
        ROUND(AVG(temperature), 2) as avg_temp, 
        COUNT(*) as total_logs
    FROM iceberg.factory_db.sensor_logs
    GROUP BY location
    ORDER BY total_logs DESC
    """
    final_df = run_query(analysis_sql)
    print("-" * 50)
    print(final_df.to_string(index=False))
    print("-" * 50)
    print("    -> [SUCCESS] Trino 분산 엔진이 정제 결과를 판다스로 완벽히 수송했습니다.")

    print("\n[FINISH] 클린 인프라 파이프라인 실습이 완벽하게 성공 종료되었습니다.")
```

<br>

- 파이썬 코드 실행

```bash
python integrate.py
```


## 6. 실습 결과와 피드백

- **실행 로그**는 **모던 데이터 레이크하우스 아키텍처의 이론**들이 물리적인 컴퓨팅 환경에서 어떻게 작동했는지를 증명하는 **명세서**

- **실습 단계별 출력 결과 분석 및 기술적 가치**
    > [인프라 사전 검증] MinIO 'warehouse' 물리 버킷 확인 및 자동 생성
    >  -> [SUCCESS] MinIO 내부에 'warehouse' 버킷을 물리적으로 자동 개설했습니다.
    - **결과가 말해 주는 것:**
        - 파이썬 애플리케이션이 `boto3(S3 API)`를 통해 로컬의 오브젝트 스토리지(MinIO) 내부를 직접 제어하는 데 성공함

    - **엔지니어링적 가치:**
        - 데이터 카탈로그(`iceberg-rest`)는 테이블의 메타데이터 대장만 관리할 뿐 물리 저장소(Bucket)를 직접 파주지 못한다는 특성을 파이썬 코드로 극복
        - 인프라 종속성으로 인한 파일 시스템 체킹 에러를 원천 차단
        - **애플리케이션의 100% 완전한 기동 자립화**를 확보했음을 의미

    > [1단계] 실습 스키마(Database) 신규 생성 단계
    >  -> [SUCCESS] factory_db 스키마가 성공적으로 생성되었습니다.
    - **결과가 말해 주는 것:**
        - 호스트 PC의 파이썬이 **8080 포트**를 경유해 도커 내부의 Trino 분산 엔진과 제어 세션 연결을 성공함

    - **엔지니어링적 가치:**
        - 흩어져 있는 오브젝트 스토리지의 파일 시스템 위에
        - 논리적인 전사 데이터 격리 공간(Database)이
        - 중앙 데이터 카탈로그 대장에 영구적으로 등록되었음을 의미

    > [2단계] Iceberg 포맷 테이블 구조 정의 단계
    >  -> [SUCCESS] 정상적인 오피셜 이름 'sensor_logs' 테이블이 정의되었습니다.
    - **결과가 말해 주는 것:**
        - S3 스토리지에 무작위로 던져질 원시(Raw) 파일들을
        - 일정한 규칙(데이터 타입, 타임스탬프)과 파티션 구조(`location`)로 해석하겠다는
        - **테이블 명세서가 데이터 카탈로그에 안전하게 저장**됨

    - **엔지니어링적 가치:**
        - 값싼 파일 스토리지(MinIO)가 수억 원짜리 관계형 DB처럼 행 단위 제어가 가능한 가상 레이크하우스로 진화할 준비를 마친 것
        - `location` 기준의 파티셔닝 명세는 향후 수십억 건의 데이터 중 필요한 라인의 파일만 골라 읽는 고속 파일 스킵(Pruning)의 기준점이 됨

    > [3단계] IoT 센서 데이터 적재 단계
    >  -> [SUCCESS] 4건의 시나리오 센서 데이터가 MinIO 레이크하우스에 깔끔하게 적재되었습니다.
    - **결과가 말해 주는 것:**
        - 4건의 스마트팩토리 센서 로그가
        - MinIO 내부의 물리 디스크 영역에
        - 압축률과 읽기 성능이 극대화된 **`Parquet` 포맷 파일로 정상 분할 저장**됨

    - **엔지니어링적 가치:**
        - Iceberg의 ACID 트랜잭션과 원자적 커밋(Atomic Commit)이 완벽히 작동했음을 의미
        - 데이터가 쓰이는 와중에도 카탈로그 대장이 최신 스냅샷 ID를 보장
        - 분산 환경에서 데이터가 찢어지거나 오염되는 현상 없이 안전하게 영속화되었음을 증명함


    > [4단계] 최종 분산 OLAP 집계 및 판다스 변환 데이터 마트 출력
    > 
    > 카탈로그 통합 검증: Iceberg Catalog에 등록된 테이블 명세 조회
    >          Table
    > 0  sensor_logs
    >         Column          Type Extra Comment
    > 0    device_id       integer              
    > 1     location       varchar              
    > 2  temperature        double              
    > 3       status       varchar              
    > 4    timestamp  timestamp(6)              
    > 5    vibration        double              
    > --------------------------------------------------
    > location  avg_temp  total_logs
    >   Line_A     71.15           2
    >   Line_B     85.20           2
    > --------------------------------------------------
    >  -> [SUCCESS] Trino 분산 엔진이 정제 결과를 판다스로 완벽히 수송했습니다.

    * **결과가 말해 주는 것:**
        - 수만~수십억 건으로 늘어날 수 있는 대용량 센서 로그에 대해
        - `Line_A`, `Line_B` 별로 평균 온도를 구하고 카운트를 세는 **대규모 분석 연산(Heavy Lifting)을 Trino 분산 엔진이 완벽하게 대리 수행**함
        - `카탈로그 통합 검증: Iceberg Catalog에 등록된 테이블 명세 조회` 쿼리 내용을 통해
        - "파이썬이 S3 파일을 직접 뒤진 게 아니라, 중앙 데이터 카탈로그가 통합 관리하는 가상 테이블 정보를 Trino를 통해 조회해 온 것"임을 명확히 시각화

    * **엔지니어링적 가치:**
        - **파이썬 데이터 과학 스택(`Pandas`, AI/ML 모델)과의 완벽한 통합 가치**를 보여줌
        - 파이썬 메모리가 터질 걱정 없이,
        - 분산 연산 결과물인 단 2줄짜리 최적화 데이터프레임 요약본만 네트워크를 통해 받아오는
        - '가장 이상적인 모던 빅데이터 파이프라인의 종착지'를 시각적으로 확인

- **총평 (Conclusion)**

> - "값싼 스토리지(MinIO) + 논리적 카탈로그(Iceberg) + 고속 분산 연산기(Trino) + 애플리케이션(Python)"이 유기적으로 톱니바퀴처럼 맞물려 들어간 최종 결과<br><br>
> - **실습을 통해 얻을 수 있는 가치**
>   - 우리는 파일만 저장하는 가장 값싼 저장소(MinIO)를 썼다.
>   - 하지만 그 위에 Iceberg와 Trino를 얹음으로써,
>       - 수억 원짜리 Oracle이나 대형 데이터웨어하우스처럼 작동하는
>       - 최고 사양의 가상 대용량 SQL DB 마트를 공짜로 만들어 냈다.
>   - 그리고 파이썬은 이 거대한 분산 인프라를 코드 몇 줄로 완벽하게 쥐고 흔들 수 있다.
{: .summary-quote}


## 7. 적용 가능한 실무 상황

- **초당 수만 건의 대규모 시계열/IoT 로그 적재 및 분석**
    - 스마트팩토리 센서, 자율주행 차량, 서비스 로그(App Click Stream) 등 끊임없이 데이터가 쏟아지는 환경<br><br>

    - **기존의 한계:**
        - 일반 DB(MySQL 등)에 이 데이터를 그대로 밀어 넣으면 쓰기 병목이 걸려 DB 전체가 마비됨
        - NoSQL(HBase, Cassandra)을 사용하면 분석가들이 좋아하는 SQL 통계 질의(`JOIN`, `GROUP BY`) 성능이 처참해짐
    - **이 구조의 활용성:**
        - 데이터 수집 에이전트(Fluentd, Kafka 등)가 데이터를 저렴한 S3(MinIO)에 `Parquet` 파일로 마구 던져놓아도 **쓰기 병목이 전혀 없음**
        - 이렇게 쌓인 날것의 파일들을 Iceberg가 실시간으로 테이블화하고, Trino가 병렬로 긁어가 가볍게 쿼리하므로 **"대용량 저적재 비용"과 "고속 OLAP 분석 성능"을 동시에 충족**

- **'데이터 사일로(Silo)' 해결을 위한 전사 연합 쿼리 (Data Federation)**
    - 기업 내부에 마케팅 데이터는 PostgreSQL에, 생산 데이터는 MySQL에, 정형 로그는 S3에 각각 찢어져 있어 통합 분석이 불가능한 상황<br><br>

    - **기존의 한계:**
        - 이를 통합하려면 모든 데이터를 하나의 거대한 데이터 웨어하우스(예: Snowflake, BigQuery)로 복사하고 변환하는 무거운 ETL 파이프라인을 몇 달 동안 구축해야 함
        - 비용도 이중으로 소요
    - **이 구조의 활용성:**
        - Trino는 소스 코드가 어디에 있든 연결할 수 있는 커넥터를 제공
        - 데이터 레이크(Iceberg)에 적재된 대용량 센서 파일과 MySQL에 있는 '공장 설비 마스터 정보' 테이블을
        - **데이터 이동 없이 Trino 안에서 한 줄의 SQL로 다이렉트 `JOIN**` 하여
        - "어느 제조사 설비가 대용량 로그 상에서 온도 상승이 가장 빈번했는가?"를 즉각 뽑아낼 수 있음

- **기계학습(ML/AI) 모델의 피처 엔지니어링 및 학습 데이터 수급**
    - 예지보전 AI 모델, 추천 시스템 모델 등을 학습시키기 위해 과거 수개월~수년 치의 대용량 이력 데이터를 정제해야 하는 상황<br><br>

    - **기존의 한계:**
        - 파이썬 메모리(`Pandas`)에 수백 기가바이트의 데이터를 한 번에 올리면 로컬 PC나 AI 서버의 메모리가 터짐(`OOM Error`).
    - **이 구조의 활용성:**
        - `integrate.py` 방식이 이 문제를 해결 가능함
        - "최근 1년간의 라인별 평균 온도와 표준편차를 구하라" 같은 **대규모 연산(Heavy Lifting)은 도커 내부의 Trino 분산 컴퓨터 클러스터가 수행**하게 하고,
        - 파이썬(AI 파이프라인)은 딱 정제된 최종 가벼운 데이터프레임만 받아서
        - 즉시 모델 학습(`XGBoost`, `PyTorch` 등)의 인풋 피처로 입력할 수 있음

- **데이터 규제 준수 및 시점 분석을 위한 데이터 감사 (Audit & Time Travel)**
    - 금융 결제 이력, 팩토리 사고 이력 분석 등 "특정 시점에 데이터가 어떤 상태였는지" 과거를 완벽하게 증명해야 하거나,
    - 잘못 업데이트된 데이터를 긴급 롤백해야 하는 상황<br><br>

    - **기존의 한계:**
        - 일반 DB에서 과거 상태를 보려면 날짜별로 테이블을 계속 복사해 두는 백업 지옥에 빠지거나,
        - 데이터 용량이 몇 배로 늘어나는 CDC(Change Data Capture) 시스템을 무겁게 얹어야 함
    - **이 구조의 활용성:**
        - Iceberg의 **타임 트래블(Time Travel)** 기능이 기본으로 작동
        - 데이터가 아무리 실시간으로 `UPDATE` 되거나 `DELETE` 되어도,
        - 물리 파일이 꼬이지 않고 과거 스냅샷 ID를 기억하고 있으므로
        - "사고가 발생하기 직전인 어제 오후 3시 상태의 마트 구조로 조회해라"가 SQL 문 한 줄로 성립됨


> - 초당 수만 건의 로그가 쏟아져서 DB는 터지기 일보 직전이고,
> - 정작 분석가들은 데이터가 파일로 흩어져 있어서 SQL 조회를 못 해 발을 동동 구르고 있으며,
> - 대형 클라우드 DW를 도입하자니 예산이 부족한 기업에게
> - **'가장 비용 효율적이면서도 압도적인 성능을 내는 오픈소스 표준 아키텍처'**