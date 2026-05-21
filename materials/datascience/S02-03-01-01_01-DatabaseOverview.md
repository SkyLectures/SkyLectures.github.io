---
layout: page
title:  "Database 개요"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-01-01_01-DatabaseOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 개요 및 기본 용어

- **데이터베이스(Database, DB):**
    - 여러 사람이 공유하여 사용할 목적으로 통합·관리되는 데이터의 집합
    - 단순히 데이터를 모아둔 것을 넘어, 컴퓨터 시스템에 전자적으로 저장되며 효율적인 검색과 수정이 가능하도록 구조화되어 있다는 점이 핵심

    - **데이터베이스의 4대 핵심 특징**
        - 실시간 접근성 (Real-time Accessibility)
            - 사용자가 요청하는 순간 실시간으로 데이터를 처리하여 응답함
        - 계속적인 변화 (Continuous Evolution)
            - 데이터의 삽입(Insert), 삭제(Delete), 수정(Update)을 통해 항상 최신 상태를 유지함
        - 동시 공유 (Concurrent Sharing)
            - 여러 사용자가 서로 다른 목적으로 동시에 동일한 데이터에 접근할 수 있음
        - 내용에 의한 참조 (Content Reference)
            - 데이터의 저장 위치나 주소가 아닌, 데이터의 '값(내용)'을 가지고 원하는 데이터를 찾음
    
- **데이터베이스 관리 시스템(Database Management System, DBMS):**
    - 사용자와 데이터베이스 사이에서 데이터를 관리하고 사용자의 요구에 따라 효율적으로 정보를 생성해 주는 소프트웨어
        - 예: MySQL, Oracle, PostgreSQL 등
    - 데이터베이스가 데이터를 모아둔 '창고'라면, DBMS는 그 창고를 안전하게 관리하고 물건을 대신 꺼내주는 '창고 관리인'
        - 엑셀 파일이나 단순 텍스트 파일과 달리, 대용량의 데이터를 여러 사람이 동시에 안전하게 쓸 수 있도록 통제해 주는 역할을 수행

    - **DBMS의 3대 필수 기능**
        - 정의 (Definition)
            - 데이터의 형태, 구조, 데이터가 가질 수 있는 조건(제약조건)을 설정
        - 조작 (Manipulation)
            - 데이터를 조회(Select), 삽입(Insert), 수정(Update), 삭제(Delete)할 수 있는 수단(예: SQL) 제공
        - 제어 (Control)
            - 데이터의 정확성 유지(무결성)
            - 여러 사람이 동시에 접근해도 무너지지 않게 통제
            - 권한이 있는 사람만 접근할 수 있도록 보안 유지

    - **대표적인 DBMS 종류**
        - 관계형(RDBMS)
            - 데이터를 표(Table) 형태로 관리하는 가장 표준적인 시스템
            - 종류: MySQL, Oracle, PostgreSQL, MS SQL Server

        - 비관계형(NoSQL)
            - 정해진 틀 없이 자유로운 형태(JSON, Key-Value 등)로 데이터를 대량 적재할 때 사용
            - 종류: MongoDB, Redis, Cassandra

- **SQL (Structured Query Language, 구조화 질의어):**
    - 관계형 데이터베이스(RDBMS)에 저장된 데이터를 관리하고 소통하기 위해 사용하는 표준 컴퓨터 언어
    - DBMS와 대화하기 위한 표준 언어
    - **데이터베이스라는 창고(RDBMS)에 있는 데이터를 조회하고, 넣고, 수정하고, 지우기 위해 창고 관리인(DBMS)에게 건내는 '표준 명령어(대화 수단)'**

    - **SQL의 3대 핵심 기능 분류**
        - SQL은 목적에 따라 크게 세 가지 종류의 명령어로 나뉨
            - DML (Data Manipulation Language - 데이터 조작어)
                - 역할: 데이터를 실제로 다룰 때 사용하며, 실무에서 가장 많이 사용됨
                - 명령어: SELECT (조회), INSERT (삽입), UPDATE (수정), DELETE (삭제)

            - DDL (Data Definition Language - 데이터 정의어)
                - 역할: 테이블이나 데이터베이스 같은 '데이터의 틀'을 만들고 변경할 때 사용
                - 명령어: CREATE (생성), ALTER (수정), DROP (삭제), TRUNCATE (초기화)

            - DCL (Data Control Language - 데이터 제어어)
                - 역할: 데이터베이스에 대한 접근 권한을 주거나 빼앗을 때 사용
                - 명령어: GRANT (권한 부여), REVOKE (권한 회수)

    - **SQL의 주요 특징**
        - 세계적 표준
            - ISO(국제표준화기구)에서 지정한 표준 언어
            - MySQL, Oracle, PostgreSQL 등 어떤 DBMS를 쓰더라도 기본 문법은 거의 동일하게 작동함
        - 선언적 언어
            - "데이터가 어디에 어떻게 저장되어 있는지" 과정을 지시하는 것이 아니라,
            - "내가 원하는 결과 데이터가 무엇인지(What)"만 선언하면 DBMS가 알아서 찾아옴
        - 영어와 유사한 문법
            - 대문자/소문자를 가리지 않으며,
            - 영어 문장 구조(SELECT name FROM users WHERE id = 1)와 매우 비슷하여 직관적이고 배우기 쉬움