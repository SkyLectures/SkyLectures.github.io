---
layout: page
title:  "Trino(Presto) 기반 S3 데이터 SQL 엔진 구축"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-04-02-01_01-TrinoS3SqlEngine
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - Trino-AWS S3(또는 MinIO 등) 연동 🡲 대용량 데이터 레이크하우스의 쿼리 엔진 구축 🡲 현대 데이터 아키텍처의 핵심 패턴
> - MinIO(S3 대용), Trino(쿼리 엔진), Iceberg(테이블 포맷/카탈로그), Python/Pandas(데이터 생성), DuckDB(데이터 검증용)를 조합하여, 
>   - 복잡한 하둡 생태계 도구(Hive) 없이도 깔끔하게 가상 S3 데이터 레이크하우스를 구축하는 실습 과정
{: .common-quote}


## 1. 아키텍처 및 도구의 역할

- **MinIO (S3 대용):**
    - 실제 데이터 파일(Parquet)과 메타데이터(Iceberg 스키마 정보)가 저장되는 가상 S3 스토리지
- **Apache Iceberg:**
    - S3의 파일들을 일반 RDBMS 테이블처럼 트랜잭션 처리(ACID)할 수 있게 묶어주는 고성능 오픈 테이블 포맷
- **Trino:**
    - SQL 엔진 역할을 하며, MinIO에 저장된 Iceberg 테이블을 초고속으로 분석
- **Python (Pandas):**
    - 실습에 사용할 대용량 원천 가상 데이터를 생성하여 MinIO에 업로드할 때 사용


<div class="insert-image">
    <h3>Trino Architecture</h3>
    <img src="/materials/devtools/images/S13-04-02-01_01-001_Trino_S3_Sql_Engine.png" style="width: 90%;">
</div>



## 2. 실습 환경 구성 

- Docker Compose 기반

1. **파일 구조 준비**
    - 로컬 환경에 작업 폴더를 만들고 구조 설정

        ```bash
        mkdir trino-iceberg-demo && cd trino-iceberg-demo
        mkdir trino-catalog
        ```

2. **`docker-compose.yml` 작성**
    - Hive Metastore 서버 없이, 오직 **MinIO**와 **Trino** 두 가지만으로 가볍게 클러스터 구성

        ```yaml
        version: '3.8'

        services:
        # 1. S3 역할을 할 오브젝트 스토리지 (MinIO)
        minio:
            image: minio/minio:latest
            container_name: demo-minio
            ports:
            - "9000:9000"
            - "9001:9001"
            environment:
            MINIO_ROOT_USER: minioadmin
            MINIO_ROOT_PASSWORD: minioadminpassword
            command: server /data --console-address ":9001"

        # 2. 고속 분산 SQL 쿼리 엔진 (Trino)
        trino:
            image: trinodb/trino:latest
            container_name: demo-trino
            ports:
            - "8080:8080"
            volumes:
            - ./trino-catalog:/etc/trino/catalog
            depends_on:
            - minio
        ```

3. Trino의 Iceberg 카탈로그 설정 (`iceberg.properties`)
    - Trino가 Hive 서버 없이 MinIO(S3) 내부의 특정 경로를 직접 메타데이터 저장소(카탈로그 타입: `file`)로 쓰도록 설정
    - **경로 및 파일명:** `trino-catalog/iceberg.properties`

        ```properties
        connector.name=iceberg
        # Iceberg 메타데이터를 별도 서버 없이 S3(MinIO) 파일 기반으로 관리
        iceberg.catalog.type=file
        iceberg.catalog.warehouse=s3a://lakehouse/
        # MinIO(S3) 연결 정보 설정
        hive.s3.aws-access-key=minioadmin
        hive.s3.aws-secret-key=minioadminpassword
        hive.s3.endpoint=http://minio:9000
        hive.s3.path-style-access=true
        hive.s3.ssl.enabled=false
        ```


## 3. 단계별 실습 과정

- **[1 단계] 서비스 시작 및 스토리지 버킷 생성**

    1. 터미널에서 서비스 실행

        ```bash
        docker-compose up -d
        ```


    2. 브라우저로 `http://localhost:9001` (MinIO 콘솔)에 접속 (ID/PW: `minioadmin`)
    3. **Buckets** 메뉴로 이동 🡲 `lakehouse` 라는 이름의 버킷 생성
        - Trino 설정에서 warehouse 경로로 지정한 이름과 같아야 함


- **[2 단계] Python (Pandas)으로 가상 원천 데이터 생성하기**
    1. 로컬 환경에서 Python을 사용하여 분석에 사용할 가상 센서 로그/사용자 데이터를 담은 CSV 파일 준비
        - `generate_data.py` 파일 작성 후 실행 (혹은 Jupyter Notebook 활용)

            ```python
            import pandas as pd

            # 1. 가상의 원천 데이터 생성
            data = {
                'user_id': [101, 102, 103, 104, 105],
                'username': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
                'login_count': [24, 5, 42, 11, 8],
                'status': ['Active', 'Inactive', 'Active', 'Active', 'Banned']
            }
            df = pd.DataFrame(data)

            # 2. 로컬에 CSV로 저장
            df.to_csv('raw_users.csv', index=False)
            print("가상 데이터 생성 완료: raw_users.csv")
            ```

    2. 생성된 `raw_users.csv` 파일을 MinIO 웹 콘솔의 `lakehouse` 버킷 하위에 `raw/` 라는 폴더를 만들고 업로드
        - 경로 예시: `lakehouse/raw/raw_users.csv`


- **[3 단계] Trino CLI 접속 및 S3(Iceberg) 테이블 생성**
    1. Trino 컨테이너 내부로 진입하여 대화형 CLI 프롬프트 열기

        ```bash
        docker exec -it demo-trino trino
        ```

    2. Trino 프롬프트(`trino>`)에서 Iceberg 표준 SQL을 사용하여 구조화된 데이터 레이크하우스 테이블을 선언

        ```sql
        -- 1. Iceberg 스키마(데이터베이스) 생성
        CREATE SCHEMA iceberg.demo_db;

        -- 2. Iceberg 포맷의 전용 테이블 생성
        CREATE TABLE iceberg.demo_db.users (
            user_id BIGINT,
            username VARCHAR,
            login_count BIGINT,
            status VARCHAR
        );
        ```

    > - **데이터 레이크하우스의 이점**
    >   - 과거 Hive 방식처럼 파일 경로(`external_location`)를 수동으로 매핑해 줄 필요가 없음
    >   - Iceberg가 MinIO의 `s3a://lakehouse/demo_db/users` 경로에 메타데이터와 파일 구조를 자동으로 추적·관리
    {: .common-quote}


- **[4 단계] Trino를 이용한 데이터 적재 및 SQL 분석**
    - 원천 CSV 파일에서 데이터를 읽어와 구조화된 Iceberg 고성능 테이블로 옮겨 싣는 작업 🡲 Trino SQL문으로 간단히 처리 가능
    - Trino 내부에서 원천 데이터를 임시 조회하거나 바로 `INSERT` 문으로 집어넣을 수 있음
    - 실습 내용: 정식 포맷 테이블에 데이터를 입력하고 쿼리하기

        ```sql
        -- 1. 데이터 직접 인서트 테스트 (S3 저장소에 실시간 반영됨)
        INSERT INTO iceberg.demo_db.users VALUES 
        (101, 'Alpha', 24, 'Active'),
        (102, 'Beta', 5, 'Inactive'),
        (103, 'Gamma', 42, 'Active'),
        (104, 'Delta', 11, 'Active'),
        (105, 'Epsilon', 8, 'Banned');

        -- 2. 복잡한 분석 쿼리 수행 (초고속 인메모리 처리)
        SELECT status, 
            COUNT(*) as user_count, 
            AVG(login_count) as avg_logins
        FROM iceberg.demo_db.users
        GROUP BY status
        ORDER BY avg_logins DESC;
        ```


- **[5 단계] Iceberg만의 강력한 기능 실습 (Time Travel)**
    - 과거 Hive 기반 S3 조회 엔진에서는 불가능했던 **"시간 여행(Time Travel)"** 기능 실습하기
    - Iceberg는 데이터 변경 이력을 S3에 메타데이터 스냅샷으로 남김 🡲 과거 특정 시점의 S3 데이터를 SQL로 조회할 수 있음

        ```sql
        -- 데이터를 변경해 봅니다.
        UPDATE iceberg.demo_db.users 
        SET status = 'Banned' 
        WHERE user_id = 104;

        -- 현재 상태 조회 (Delta가 Banned로 나옴)
        SELECT * FROM iceberg.demo_db.users;

        -- 테이블의 변경 스냅샷 기록 확인
        SELECT * FROM iceberg.demo_db.users$snapshots;

        -- 과거 스냅샷 ID를 복사하여 (예: 4921937102931) 업데이트 이전 과거 시점 데이터 조회
        -- (스냅샷 ID는 위 스냅샷 확인 쿼리 결과의 snapshot_id 컬럼값을 사용합니다)
        SELECT * FROM iceberg.demo_db.users FOR VERSION AS OF <스냅샷ID>;
        ```


- **[6 단계] DuckDB로 S3 결과물 교차 검증**
    - Trino가 S3(MinIO)에 생성한 실제 물리 파일인 **Parquet 파일**을 가벼운 로컬 분석 도구인 **DuckDB**로 열어서 데이터 정합성이 완벽하게 보장되었는지 교차 검증해 볼 수 있음
    - MinIO 콘솔에 들어가 보면 `lakehouse/demo_db/users/data/` 폴더 내부에 `.parquet` 파일이 생성되어 있음
    - 해당 파일을 로컬로 다운로드받은 뒤 DuckDB로 열어보기

        ```python
        import duckdb

        # 다운로드받은 물리 Parquet 파일을 DuckDB로 직접 쿼리
        # Trino가 가공하여 S3에 저장한 포맷이 온전함을 검증할 수 있습니다.
        res = duckdb.query("SELECT * FROM 'data_file_name.parquet' WHERE login_count > 10").df()
        print(res)
        ```


- **[7 단계] 리소스 정리
    - 실습 가상 환경을 완전히 종료하고 정리하려면 아래 명령어를 입력

        ```bash
        docker-compose down -v
        ```

> - 위 실습으로 복잡한 오버헤드 없이 "MinIO + Iceberg + Trino"라는 직관적인 초고속 데이터 레이크 시스템 구축 패턴을 학습할 수 있음