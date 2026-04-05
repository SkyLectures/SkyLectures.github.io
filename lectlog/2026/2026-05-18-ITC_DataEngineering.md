---
layout: page
title:  "데이터 엔지니어링 과정(160H)"
date:   2026-04-02 10:50:00 +0900
permalink: /lectlog/2026-05-18-ITC_DataEngineering
categories: lectlog
---
* toc
{:toc .large-only .toc-sticky:true}


<h2>● 강의 개요</h2>

> - **교육분야:** 데이터 엔지니어링
> - **교육대상:** 데이터 엔지니어링 및 AI 인프라 구축에 관심이 있는 수강생
> - **강의기간:** 2026.05.18~07.14 (19:00 ~ 23:00) (164H)
> - **운영방식:** 온라인 강의

> - **학습목표**
>   - Docker 기반의 모던 데이터 스택 구축
>   - RAG를 위한 벡터 스토어 구축
{: .common-quote}


<h2>● 강의 내용(New)</h2>

- **1주차: 리눅스 & Docker**
    - **리눅스**
        - 리눅스 커맨드 기초
        - 리눅스쉘 스크립트
    - **Docker**
        - 이미지, 컨테이너에 대한 이해와 도커 사용하기
        - Dockerfile 작성 및 이미지 최적화
        - Docker Compose 이해하고 사용하기

- **2주차: Database**
    - MySQL 8.0 설치 및 기초 쿼리 이해하기
    - CTE, Window Function을 활용한 분석 쿼리
    - MongoDB 구축 및 비정형 로그 데이터 적재
    - ERD 설계 및 정규화/반정규화 전략
    - Python 기반의 크롤러

- **3주차: Python ETL**
    - Pandas를 활용한 데이터 처리
    - Pandas를 활용한 데이터 분석
    - DuckDB를 이용한 로컬 대용량 데이터 처리
    - DuckDB를 이용한 데이터 분석
    - Pandas와 DuckDB를 이용한 하이브리드 가공

- **4주차: DataLake**
    - MiniIO, S3 기반 오브젝트 스토리지 구축
    - Apache Iceberg 기반 레이크하우스 구축
    - MiniO – Iceberg 데이터 파이프라인 구축
    - Trino(Presto) 기반 S3 데이터 SQL 엔진 구축
    - 데이터 카탈로그 연결 및 레이크하우스 통합

- **5주차: Spark**
    - 분산 데이터 처리 이해 및 Spark 아키텍처 이해
    - Docker Compose를 이용한 Spark M-W 구조 구축 
    - Iceberg, MinIO 연결 및 대용량 데이터 분석 준비 
    - Spark DataFrame, SparkSQ을 이용한 데이터 가공 
    - 파티셔닝과 셔플링 최적화 이해하기

- **6주차: Streaming**
    - 실시간 데이터 특징 이해화 Kafka 이해하기
    - Docker 기반 Kafka 클러스터 구축하기
    - Producer / Consumer 애플리케이션 만들기
    - CDC(Debezium) DB 변경분 실시간 캡처
    - Kafka Connect 이용한 실시간 데이터 MiniO 적재 

- **7주차: RAG**
    - LangChain을 이용한 챗봇 기초
    - RAG를 위한 텍스트 데이터 Chunking
    - Embedding기법과 Vector DB
    - MiniO와 VectorDB(Qdrant) 연동
    - Hybrid Search 구현

- **8주차: Orchestration & AI Service**
    - **Orchestration**
        - Airflow 서비스 구축,
        - DAG 이해 및 유즈케이스 연구
        - 수집->Lake->Spark->VectorDB 흐름 자동화
    - **AI Service**
        - Streamlit 기반의 챗봇 인터페이스 개발
        - 데이터 파이프라인 구축 및 최종 RAG 챗봇 개발 