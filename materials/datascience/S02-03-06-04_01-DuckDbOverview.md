---
layout: page
title:  "DuckDB 개요 및 설치, 환경설정"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-04_01-DuckDbOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. DuckDB 개요

> - **DuckDB**
>   - 데이터 분석(OLAP)을 위해 설계된 **오픈소스 인프로세스(In-process) SQL 데이터베이스 관리 시스템**
>       - 흔히 "분석용 SQLite"라고 불림<br><br>
>   - **기본 철학: 간편한 설치, 빠른 분석 성능, 높은 이식성**<br><br>
>   - 별도의 서버 설치 없이 애플리케이션 내에 포함되어 실행되는 것이 가장 큰 특징
>   - 2026년 현재, 단순한 임베디드 DB를 넘어 데이터 레이크하우스(DuckLake)와 클라우드 서비스(MotherDuck)를 아우르는 강력한 분석 생태계의 핵심으로 자리 잡음
{: .common-quote}

### 1.1 개발 역사

- **탄생 배경: "분석을 위한 SQLite는 왜 없을까?" (2018년)**
    - 네덜란드의 국립 수학 및 컴퓨터 과학 연구소인 CWI(Centrum Wiskunde & Informatica)
        - CWI: 모네DB(MonetDB) 같은 혁신적인 열 중심(Columnar) DB를 개발한 연구소
    - 하네스 뮬하이젠(Hannes Mühleisen)과 **마크 라스벨트(Mark Raasveldt)** 두 연구원에 의해 2018년 처음 개발됨<br><br>

    > - 가볍게 파일 하나로 돌아가는 트랜잭션 DB(SQLite)는 세상에 지대한 공헌을 했는데,
    > - 왜 **가볍게 파일 하나로 돌아가는 분석용(OLAP) DB**는 없을까?

    - 당시 파이썬을 쓰던 데이터 과학자들은 대용량 데이터를 처리할 때 메모리 부족(OOM) 문제에 시달림
    - 이를 해결하려면 무거운 대기업형 데이터 웨어하우스(DW)를 구축해야만 했음 🡲 이 틈새시장을 노리고 DuckDB 개발이 시작됨

<br>

- **오픈소스 공개와 '오리'의 유래 (2019년~2021년)**
    - **2019년 6월:** DuckDB의 첫 번째 오픈소스 버전(v0.1.0)이 GitHub에 공개
    - **이름의 유래:** 
        - 공동 창립자인 하네스 뮬하이젠이 키우던 반려오리 '윌버(Wilbur)'에서 따옴
        - 오리는 날 수도 있고, 걸을 수도 있고, 수영도 할 수 있는 다재다능한 동물 🡲 DuckDB도 어디서나 다재다능하게 쓰이길 바라는 마음
    - 파이썬 데이터 생태계(Pandas, Arrow 등)와 장벽 없이 데이터를 주고받는 'Zero-Copy' 기능 🡲 데이터 과학자들 사이에서 입소문 시작

<br>

- **상용화와 생태계의 폭발적 성장 (2021년~2023년)**
    - **2021년 DuckDB Labs 설립:** 비즈니스 지원과 지속적인 오픈소스 개발을 위해 회사 설립
    - **2022년 MotherDuck 창업:**
        - 빅데이터의 대부이자 구글 BigQuery의 창시자 중 한 명인 조던 티가니(Jordan Tigani) 🡲 DuckDB 팀과 손잡고 스타트업 **MotherDuck** 설립
        - DuckDB를 클라우드로 확장하여 "서버리스 하이브리드 분석"을 제공하겠다는 비전 🡲 순식간에 수천만 달러의 투자 유치

    - 이때부터 DuckDB는 현대적 데이터 스택(Modern Data Stack)의 핵심 도구로 인정받기 시작

<br>

- **정식 1.0 버전 출시와 대세 등극 (2024년~현재)**
    - **2024년 6월, DuckDB 1.0 공식 출시:**
        - 프로젝트 시작 후 약 6년 🡲'하위 호환성'과 '안정성'을 보장하는 정식 1.0 버전 릴리즈
        - 수많은 기업들이 프로덕션(실무 운영) 환경에 DuckDB를 적극적으로 도입 시작

    - **2025년~2026년 현재:**
        - 데이터 레이크하우스 포맷(Apache Iceberg, Delta Lake)을 완벽하게 지원
        - 로컬에서 수십~수백 GB의 데이터를 처리하는 표준 도구로 자리 잡음
        - 웹 브라우저(WebAssembly)부터 대규모 데이터 파이프라인(ETL)까지 DuckDB가 빠지지 않는 곳이 없을 정도로 대세가 됨

<br>

> - **요약**
>   1. **2018년:** 네덜란드 연구소(CWI)에서 "분석용 SQLite를 만들자"며 개발 시작.
>   2. **2022년:** MotherDuck 설립 등 상용화 단계를 거치며 클라우드/하이브리드 생태계로 확장.
>   3. **2024년 이후:** 1.0 정식 버전 출시와 함께 빅데이터 분석가·엔지니어들의 필수 라이브러리로 등극.
{: .summary-quote}


### 1.2  핵심 아키텍처 및 주요 특징

> - DuckDB는 내부적인 구조적 혁신(Architecture)을 통해 사용자가 실무에서 체감할 수 있는 강력한 기능(Features)을 제공함
{: .common-quote}

- **대규모 데이터 분석을 위한 OLAP 아키텍처**
    - DuckDB는 트랜잭션(OLTP)이 아닌, 대규모 데이터 집계 및 통계 분석(OLAP)에 맞춰 뼈대부터 설계됨
    - **컬럼 기반 벡터화 엔진 (Columnar-Vectorized Execution):**
        - 데이터를 행(Row)이 아닌 열(Column) 단위로 저장하여 I/O 효율을 극대화
        - 현대 CPU의 SIMD 명령어를 활용해 데이터를 수천 개씩 묶어(Vector) 초고속으로 연산
    - **분석 특화 강력한 SQL 지원:**
        - PostgreSQL과 호환되는 표준 SQL 지원
        - 실무 분석에 필수적인 `PIVOT`, `UNPIVOT`, 시계열 분석을 위한 `AS OF JOIN` 등 강력한 독자 구문 제공

- **경량화 및 유연성을 위한 In-Process 아키텍처**
    - 별도의 DB 서버 프로세스 없이, 애플리케이션(Python, R 등) 내부에서 라이브러리 형태로 구동되는 임베디드 구조
    - **외부 파일 직접 쿼리 (Zero-Ingestion):**
        - 가벼운 엔진 특성을 살려, 데이터를 DB 내보내기/가져오기(Import) 할 필요 없이 로컬이나 S3에 있는 `.csv`, `.parquet`, `.json` 파일을 SQL 구문으로 즉시 조회
    - **플러그인 중심의 확장성:**
        - 외부 파일 및 다양한 데이터 소스에 접근할 수 있도록 HTTP/S3, JSON, Iceberg 연동 기능이 플러그인 형태로 확장

- **파이썬 생태계 융합을 위한 Zero-Copy 아키텍처**
    - **메모리 공유 및 고속 데이터 교환:**
        - Pandas, Polars, Arrow 등 현대 데이터 과학 도구들과 메모리 주소를 공유
        - 대용량 데이터를 다룰 때 가장 큰 병목이었던 '데이터 복사(Copy)' 프로세스 제거<br> 🡲 데이터 프레임 간의 전환과 쿼리가 지연 시간 없이 즉각적으로 이루어짐


### 1.3 장점과 단점

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">구분</th>
        <th style="width: 400px;">장점 (Pros)</th>
        <th style="width: 400px;">단점 (Cons)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">성능</td>
            <td>OLAP 쿼리에서 수백 배 빠른 속도 (벡터화 엔진)</td>
            <td>고빈도 쓰기/수정(OLTP) 작업에는 부적합</td>
        </tr>
        <tr>
            <td class="td-rowheader">운영</td>
            <td>설치가 필요 없음 (`pip install duckdb`로 끝)</td>
            <td>다중 사용자 동시 쓰기 제한 (Single-Writer)</td>
        </tr>
        <tr>
            <td class="td-rowheader">호환</td>
            <td>Python, R, Arrow 등과 완벽한 통합</td>
            <td>대규모 서버 분산 환경(Distrubuted) 미지원</td>
        </tr>
        <tr>
            <td class="td-rowheader">비용</td>
            <td>오픈소스(MIT 라이선스), 인프라 비용 거의 없음</td>
            <td>서버 리소스(RAM/CPU)를 앱과 공유하므로 관리 주의</td>
        </tr>
    </tbody>    
</table>
</div>


### 1.4 주요 활용 사례 (Use Cases)

- **데이터 과학 및 머신러닝:**
    - 로컬에서 수 기가바이트(GB) 규모의 데이터를 전처리하거나
    - 탐색적 데이터 분석(EDA)을 수행할 때 유용

- **서버리스(Serverless) 분석:**
    - AWS Lambda나 Google Cloud Functions에서 짧은 시간 안에 대량의 Parquet 파일을 분석하고 결과를 반환하는 용도로 최적

- **임베디드 분석 대시보드:**
    - 데스크톱 앱이나 BI 툴 내부에 탑재되어 실시간 데이터 집계 기능을 수행

- **데이터 레이크 쿼리:**
    - 데이터 레이크(S3, GCS 등)에 흩어진 파일들을 별도의 데이터 웨어하우스 이동 없이 즉시 SQL로 통합 조회할 때 사용


### 1.5 최신 동향 (2026년 기준)

> - DuckDB는
>   - 데이터가 "너무 커서 엑셀로는 안 되고, 그렇다고 Snowflake나 BigQuery를 쓰기엔 과한"
>   - **중간 규모 데이터(Small-to-Medium Data)** 분석 시장에서 독보적인 위치를 차지하고 있음
{: .common-quote}

- **DuckLake 1.0 출시:**
    - DuckDB를 기반으로 한 로컬 레이크하우스 포맷이 공식화
    - ACID 트랜잭션을 보장하면서도 Iceberg와 호환되는 테이블 형식을 지원

- **Wasm(WebAssembly) 확장:**
    - 웹 브라우저 내에서 DuckDB를 실행하여 클라이언트 측 분석 기능을 강화

- **하이브리드 실행(MotherDuck):**
    - 로컬 자원과 클라우드 자원을 결합하여,
        - 작은 쿼리는 내 PC에서,
        - 큰 쿼리는 클라우드에서 실행하는 하이브리드 분석 모델이 대중화


## 2. DuckDB의 설치와 환경설정

> - DuckDB의 가장 큰 매력은 "설치와 환경설정이랄 게 거의 없다"는 점
> - 일반적인 데이터베이스처럼 서버를 다운로드하고, 포트를 열고, 계정을 생성하는 복잡한 과정이 전혀 필요 없음


### 2.1 환경별 설치 방법

- **Python 환경 (가장 보편적인 방법)**
    - 일반 라이브러리를 설치하듯 `pip` 명령어로 설치 완료

        ```bash
        pip install duckdb
        ```

    - **설치 확인 및 버전 체크:**

        ```python
        import duckdb
        print(duckdb.__version__)
        ```

- **CLI(명령줄 인터페이스) 독립 실행형 설치**
    - 파이썬 없이 터미널이나 명렁 프롬프트에서 바로 SQL을 쓰고 싶을 때 사용
    - 운영체제별 패키지 관리자로 간단히 설치할 수 있음
        - **macOS (Homebrew):** `brew install duckdb`
        - **Windows (Winget):** `winget install DuckDB.DuckDB`
        - **Linux (Direct Download):**
            - 공식 홈페이지에서 바이너리 파일을 받아 압축을 풀면 `duckdb` 실행 파일 하나만 나옴
            - 이를 환경변수(PATH)에 등록하면 완료

- **Google Colab / Jupyter Notebook**
    - 클라우드 환경에서도 별도의 설정 없이 코드 셀에 다음을 입력하면 즉시 사용할 수 있음

        ```python
        !pip install duckdb
        ```


### 2.2 데이터베이스 연결 및 초기 설정 (Python)

- DuckDB를 사용할 때는 '메모리 모드'로 시작할지, '파일 저장 모드'로 시작할지 선택해야 함

- **인메모리(In-Memory) 모드 (휘발성)**
    - 데이터를 디스크에 저장하지 않고 메모리 위에서만 처리함
    - 간단한 테스트나, 외부 Parquet 파일을 읽어와 한 번만 집계하고 끝낼 때 최적
    - 파이썬 프로세스가 종료되면 데이터도 사라짐

        ```python
        import duckdb

        # 연결 시 아무것도 넣지 않거나 ':memory:'를 입력하면 인메모리 모드가 됩니다.
        con = duckdb.connect() 
        # 또는 con = duckdb.connect(':memory:')
        ```

- **영구 파일(Persistent File) 모드 (비휘발성)**
    - 작업 내용을 파일로 저장하여 나중에 다시 불러오고 싶을 때 사용
    - 지정한 경로에 `.duckdb` 파일이 생성됨 🡲 SQLite처럼 이 파일 하나에 모든 데이터가 들어감

        ```python
        import duckdb

        # 지정한 파일명으로 연결 (파일이 없으면 자동으로 생성됨)
        con = duckdb.connect('my_project.duckdb')
        ```


### 2.3 실무 필수 환경설정 (Configuration)

- DuckDB는 기본 설정만으로도 훌륭하게 작동함
- 대용량 데이터를 처리할 때는 **로컬 컴퓨터의 자원을 효율적으로 쓰기 위한 환경설정**이 필수적
    - SQL 쿼리를 통해 설정을 변경할 수 있음

- **메모리 사용량 제한 (`memory_limit`)**
    - 기본적으로 DuckDB는 컴퓨터 가용 RAM의 80% 가량을 사용하려고 함
    - 웹 브라우저나 다른 프로그램이 멈추는 것을 방지하려면 상한선을 정해두는 것이 좋음

        ```python
        # 최대 메모리를 8GB로 제한
        con.execute("SET memory_limit = '8GB';")
        ```

- **CPU 스레드 수 제한 (`threads`)**
    - DuckDB는 가용한 모든 CPU 코어를 써서 연산을 병렬 처리(벡터화)함
    - 만약 다른 작업과 병행해야 한다면 스레드 수를 제한할 수 있음

        ```python
        # CPU 스레드를 4개만 사용하도록 설정
        con.execute("SET threads = 4;")
        ```

- **임시 디스크 공간 설정 (`temp_directory`)**
    - 메모리 크기를 초과하는 대용량 데이터를 처리할 때, DuckDB는 디스크를 임시 메모리 공간(Spilling to disk)으로 활용함
        - 임시 파일이 저장될 경로를 지정할 수 있음
        - 가급적 속도가 빠른 **SSD 경로**로 지정하는 것이 성능에 유리함

            ```python
            # 임시 작업 디스크 경로 지정
            con.execute("SET temp_directory = '/path/to/fast_ssd/duckdb_tmp';")
            ```


### 2.4 확장 기능(Extensions) 설치

- DuckDB는 기본 엔진을 가볍게 유지하는 대신, 필요한 기능을 플러그인처럼 설치하는 구조를 가지고 있음
- S3 클라우드 접근이나 JSON, Parquet 파싱 등을 위해 아래 설정을 첫 장에 넣어두는 것이 관례

    ```python
    # S3, HTTP 원격 파일 접근을 위한 확장 기능 설치 및 로드
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # JSON 파일 처리를 위한 확장 기능
    con.execute("INSTALL json; LOAD json;")
    ```

<br>

> - **요약 가이드라인**
>   - 다음 "3줄 코드"만 실행하면 모든 환경설정이 완료됨
{: .summary-quote}

```python
import duckdb
con = duckdb.connect('analytics.duckdb')
con.execute("SET memory_limit = '8GB'; SET threads = 4; INSTALL httpfs; LOAD httpfs;")
```