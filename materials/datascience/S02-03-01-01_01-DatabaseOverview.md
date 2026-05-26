---
layout: page
title:  "Database 개요"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-01-01_01-DatabaseOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 데이터(Data)

> - **정의:** 현실 세계에서 관찰되거나 측정된 사실이나 값(Fact)
> - 현대 정보 사회의 가장 가치 있는 원자재이자, 모든 의사결정과 시스템의 기반이 되는 핵심 자산
{: .common-quote}

- **특징**
    - 단독으로 존재할 때는 단순한 수치나 문자에 불과하여 큰 의미를 갖지 못함
    - 특정 목적에 맞게 가공되고 문맥(Context)이 부여되면 비로소 가치 있는 '정보(Information)'로 전환됨
        - 예시
            - 데이터 (Data): 38 (단순한 숫자)
            - 정보 (Information): "오늘 서울의 최고 기온은 38°C이다." (의미가 부여된 데이터)

- **데이터의 분류**
    - **형태 및 구조화 수준에 따른 분류**
        - 데이터가 얼마나 일정한 규칙과 틀을 가지고 저장되어 있는지에 따라 구분됨
        - 이 분류에 따라 어떤 DBMS(MySQL인지, MongoDB인지)를 사용할지를 결정함<br><br>

        - **정형 데이터 (Structured Data)**
            - 정의: 고정된 필드(틀)에 정해진 형식으로 저장된 데이터
            - 특징:
                - 연산과 검색이 매우 빠름
                - 주로 관계형 데이터베이스(RDBMS)의 표(Table) 형태로 관리됨
            - 예시: 이름, 나이, 결제 금액, 날짜, 주소록 등

        - **반정형 데이터 (Semi-structured Data)**
            - 정의: 고정된 틀은 없지만, 데이터 내에 구조를 설명하는 메타데이터나 태그(Tag)가 포함된 데이터
            - 특징
                - 스키마(틀) 변경이 자유로움
                - 파일 형태로 교환하기 쉬움
            - 예시: JSON, XML, HTML 파일, 설정 파일 등

        - **비정형 데이터 (Unstructured Data)**
            - 정의: 형태가 전혀 정해져 있지 않고, 규칙성이 없는 데이터
            - 특징
                - 텍스트나 바이너리 형태로 존재
                - 형태가 다양해 일반적인 테이블 구조에 담을 수 없음
                - NoSQL이나 데이터 레이크(Data Lake)에 저장
            - 예시: 이미지, 영상, 오디오 파일, SNS 게시글 원문, 이메일 내용 등

    - **속성과 측정 기준에 따른 분류 (통계 및 분석 기준)**
        - 데이터가 나타내는 값의 성격에 따라 질적 데이터와 양적 데이터로 나뉨
        - 이는 주로 SQL로 통계 및 분석 쿼리를 작성할 때 집계 방식을 결정하는 기준이 됨
        
<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">분류</th>
        <th style="width: 250px;">정의</th>
        <th style="width: 250px;">특징</th>
        <th style="width: 250px;">예시</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">질적 데이터<br>(Qualitative / 범주형)</td>
            <td class="td-left">숫자로 표현할 수 없거나, 숫자로 표현해도 크기 비교가 불가능한 데이터</td>
            <td class="td-left">주로 분류나 그룹화를 할 때 사용함 (GROUP BY 대상)</td>
            <td class="td-left">성별(남/여), 혈액형, 상품 카테고리, 거주 지역</td>
        </tr>
        <tr>
            <td class="td-rowheader">양적 데이터<br>(Quantitative / 수치형)</td>
            <td class="td-left">숫자로 표현되며, 더하기·빼기 등 산술 연산이 의미가 있는 데이터</td>
            <td class="td-left">평균, 합계 등 통계량을 낼 때 사용함 (집계 함수 대상)</td>
            <td class="td-left">매출액, 회원 수, 방문 횟수, 기온, 몸무게</td>
        </tr>
    </tbody>
</table>
</div>        
         

## 2. 데이터베이스(Database, DB)

> - 여러 사람이 공유하여 사용할 목적으로 통합·관리되는 데이터의 집합
> - 단순히 데이터를 모아둔 것을 넘어, 컴퓨터 시스템에 전자적으로 저장되며 효율적인 검색과 수정이 가능하도록 구조화되어 있다는 점이 핵심
{: .common-quote}

- **데이터베이스의 4대 핵심 특징**
    - 실시간 접근성 (Real-time Accessibility)
        - 사용자가 요청하는 순간 실시간으로 데이터를 처리하여 응답함
    - 계속적인 변화 (Continuous Evolution)
        - 데이터의 삽입(Insert), 삭제(Delete), 수정(Update)을 통해 항상 최신 상태를 유지함
    - 동시 공유 (Concurrent Sharing)
        - 여러 사용자가 서로 다른 목적으로 동시에 동일한 데이터에 접근할 수 있음
    - 내용에 의한 참조 (Content Reference)
        - 데이터의 저장 위치나 주소가 아닌, 데이터의 '값(내용)'을 가지고 원하는 데이터를 찾음

## 3. 데이터베이스 관리 시스템(DBMS)

> - **데이터베이스 관리 시스템(Database Management System, DBMS)**
>   - 사용자와 데이터베이스 사이에서 데이터를 관리하고 사용자의 요구에 따라 효율적으로 정보를 생성해 주는 소프트웨어
>       - 예: MySQL, Oracle, PostgreSQL 등
>   - 데이터베이스가 데이터를 모아둔 '창고'라면, DBMS는 그 창고를 안전하게 관리하고 물건을 대신 꺼내주는 '창고 관리인'
>       - 엑셀 파일이나 단순 텍스트 파일과 달리, 대용량의 데이터를 여러 사람이 동시에 안전하게 쓸 수 있도록 통제해 주는 역할을 수행
{: .common-quote}

<br>

<div class="insert-image">
    <img src="/materials/datascience/images/S02-03-01-01_01-001.png" style="width: 70%;"><br><br><br>
    <b>DBMS 개념도</b>
</div>        

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


## 4. 구조화 질의어(SQL)

> - **SQL (Structured Query Language, 구조화 질의어)**
>   - 관계형 데이터베이스(RDBMS)에 저장된 데이터를 관리하고 소통하기 위해 사용하는 표준 컴퓨터 언어
>   - DBMS와 대화하기 위한 표준 언어
>   - **데이터베이스라는 창고(RDBMS)에 있는 데이터를 조회하고, 넣고, 수정하고, 지우기 위해 창고 관리인(DBMS)에게 건내는 '표준 명령어(대화 수단)'**
{: .common-quote}

- **SQL의 3대 핵심 기능 분류**
    - SQL은 목적에 따라 크게 세 가지 종류의 명령어로 나뉨
        - **DML (Data Manipulation Language - 데이터 조작어)**
            - 역할: 데이터를 실제로 다룰 때 사용하며, 실무에서 가장 많이 사용됨
            - 명령어: SELECT (조회), INSERT (삽입), UPDATE (수정), DELETE (삭제)

        - **DDL (Data Definition Language - 데이터 정의어)**
            - 역할: 테이블이나 데이터베이스 같은 '데이터의 틀'을 만들고 변경할 때 사용
            - 명령어: CREATE (생성), ALTER (수정), DROP (삭제), TRUNCATE (초기화)

        - **DCL (Data Control Language - 데이터 제어어)**
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