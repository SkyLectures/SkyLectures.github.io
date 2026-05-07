---
layout: page
title:  "DuckDB 개요 및 설치, 환경설정"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-04_01-DuckDbOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



**DuckDB**는 데이터 분석(OLAP)을 위해 설계된 **오픈소스 인프로세스(In-process) SQL 데이터베이스 관리 시스템**입니다. 흔히 "분석용 SQLite"라고 불리며, 별도의 서버 설치 없이 애플리케이션 내에 포함되어 실행되는 것이 가장 큰 특징입니다.

2026년 현재, DuckDB는 단순한 임베디드 DB를 넘어 데이터 레이크하우스(DuckLake)와 클라우드 서비스(MotherDuck)를 아우르는 강력한 분석 생태계의 핵심으로 자리 잡았습니다.

---

## 1. DuckDB 개요 및 철학
DuckDB는 네덜란드 CWI(Centrum Wiskunde & Informatica) 연구소에서 시작되었으며, **"간편한 설치, 빠른 분석 성능, 높은 이식성"**을 철학으로 합니다.

*   **OLAP 최적화:** 트랜잭션 처리(OLTP)가 아닌 대규모 데이터 집계 및 분석에 특화되어 있습니다.
*   **In-Process 방식:** 별도의 DB 서버 프로세스가 필요 없으며, Python, R, Java, C++ 등 호스트 애플리케이션 내에서 라이브러리 형태로 동작합니다.
*   **Zero-Copy 분석:** Pandas, Polars, Arrow와 같은 데이터 프레임워크와 메모리를 공유하여 복사 과정 없이 즉시 쿼리가 가능합니다.

---

## 2. 주요 특징 (Key Features)

### ① 컬럼 기반 벡터화 엔진 (Columnar-Vectorized Execution)
데이터를 행(Row)이 아닌 열(Column) 단위로 저장하고 처리합니다. 특정 컬럼만 읽기 때문에 I/O 효율이 극대화되며, CPU의 SIMD 명령어를 활용해 대량의 데이터를 한 번에 처리(벡터화)함으로써 압도적인 속도를 제공합니다.

### ② 강력한 SQL 지원 및 확장성
*   **PostgreSQL 호환:** 대부분의 표준 SQL과 윈도우 함수, CTE(Common Table Expressions)를 지원합니다.
*   **특수 구문:** `PIVOT`, `UNPIVOT`, `AS OF JOIN`(시계열 분석용) 등 분석에 편리한 독자적 구문을 제공합니다.
*   **확장 프로그램:** JSON, Parquet, Iceberg, HTTP/S3 연동 등이 플러그인 형태로 제공되어 외부 데이터 소스를 직접 쿼리할 수 있습니다.

### ③ 외부 파일 직접 쿼리
데이터를 DB 안으로 **임포트(Import)하지 않고도** 로컬이나 S3에 저장된 `.csv`, `.parquet`, `.json`, `.arrow` 파일을 SQL로 직접 조회할 수 있습니다.

---

## 3. 장점과 단점

| 구분 | 장점 (Pros) | 단점 (Cons) |
| :--- | :--- | :--- |
| **성능** | OLAP 쿼리에서 수백 배 빠른 속도 (벡터화 엔진) | 고빈도 쓰기/수정(OLTP) 작업에는 부적합 |
| **운영** | 설치가 필요 없음 (`pip install duckdb`로 끝) | 다중 사용자 동시 쓰기 제한 (Single-Writer) |
| **호환** | Python, R, Arrow 등과 완벽한 통합 | 대규모 서버 분산 환경(Distrubuted) 미지원 |
| **비용** | 오픈소스(MIT 라이선스), 인프라 비용 거의 없음 | 서버 리소스(RAM/CPU)를 앱과 공유하므로 관리 주의 |

---

## 4. 주요 활용 사례 (Use Cases)

*   **데이터 과학 및 머신러닝:** 로컬에서 수 기가바이트(GB) 규모의 데이터를 전처리하거나 탐색적 데이터 분석(EDA)을 수행할 때 유용합니다.
*   **서버리스(Serverless) 분석:** AWS Lambda나 Google Cloud Functions에서 짧은 시간 안에 대량의 Parquet 파일을 분석하고 결과를 반환하는 용도로 최적입니다.
*   **임베디드 분석 대시보드:** 데스크톱 앱이나 BI 툴 내부에 탑재되어 실시간 데이터 집계 기능을 수행합니다.
*   **데이터 레이크 쿼리:** 데이터 레이크(S3, GCS 등)에 흩어진 파일들을 별도의 데이터 웨어하우스 이동 없이 즉시 SQL로 통합 조회할 때 사용합니다.

---

## 5. 최신 동향 (2026년 기준)

*   **DuckLake 1.0 출시:** DuckDB를 기반으로 한 로컬 레이크하우스 포맷이 공식화되어, ACID 트랜잭션을 보장하면서도 Iceberg와 호환되는 테이블 형식을 지원합니다.
*   **Wasm(WebAssembly) 확장:** 웹 브라우저 내에서 DuckDB를 실행하여 클라이언트 측 분석 기능을 강화하고 있습니다.
*   **하이브리드 실행(MotherDuck):** 로컬 자원과 클라우드 자원을 결합하여, 작은 쿼리는 내 PC에서, 큰 쿼리는 클라우드에서 실행하는 하이브리드 분석 모델이 대중화되었습니다.



DuckDB는 데이터가 "너무 커서 엑셀로는 안 되고, 그렇다고 Snowflake나 BigQuery를 쓰기엔 과한" **중간 규모 데이터(Small-to-Medium Data)** 분석 시장에서 독보적인 위치를 차지하고 있습니다.

사용하시는 개발 언어나 현재 분석 환경(로컬 vs 클라우드)에 맞춰 DuckDB를 어떻게 도입하면 좋을지 구체적인 가이드가 필요하신가요?