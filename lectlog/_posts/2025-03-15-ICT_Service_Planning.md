---
layout: post
title:  "[03.18] AI 기반 서비스의 기획 및 구현(80H)"
date:   2025-03-15 09:00:00 +0900
categories: lectlog
---
- toc
{:toc .large-only .toc-sticky:true}

## [강의 개요]

* 고객사: 동북 ICT
* 주제: AI 기반 서비스의 기획 및 구현
* 강의기간: 2025.03.18~2025.04.30
* 강의시간: 80H

## [강의 내용]

### 1. 파이썬 기초

- [Python 개요](/material/S01-01-01-01_01-PythonOverview)
- [Python 가상환경 설정](/material/S01-01-02-01_01-VirtualEnvironment)
- 기본 문법
    - [변수와 자료형](/material/S01-01-03-01_01-VariablesDataTypes)
    - [제어문](/material/S01-01-03-02_01-ControlStatements)
    - [문자열 처리](/material/S01-01-03-03_01-StringProcess)
    - [함수, 클래스](/material/S01-01-03-04_01-FunctionsClasses)
    - [모듈, 패키지, 라이브러리](/material/S01-01-03-05_01-Modules)
    - [예외처리](/material/S01-01-03-06_01-Exceptions)

### 2. 파이썬 라이브러리 활용
- [파이썬 표준 라이브러리](/material/S01-01-04-01_01-StandardLibrary)
- [Numpy](/material/S01-01-04-02_01-Numpy)
- [Pandas](/material/S01-01-04-03_01-Pandas)
- [Matplotlib](/material/S01-01-04-04_01-Matplotlib)

### 2. 웹서비스 시스템 구현

- [웹서비스 시스템 기초]
- [Simple Web Server 실습]

- Django WebFramework 프로그래밍
    - [Django WebFramework 개요]
    - [Django 기반 서비스 기본 흐름]
    - Django 기반 서비스 구현 실습 (코드출처: 백엔드를 위한 Django REST Framework with 파이썬)
        - [사진 목록 보기]
        - [Todo List 웹서비스]

- Django REST Framework(DRF)
    - [RESTful API 개요]
    - [Django REST Framework 개요]
    - DRF 활용 실습: 게시판 만들기 (코드출처: 백엔드를 위한 Django REST Framework with 파이썬)
        - [게시판 사용자 관리]
        - [게시판 글 관리]


### 3. LLM 활용

- [LLM 개요]
- [LLM 활용(ChatGPT API / Ollama 사용 실습)]
- LangChain
    1. [LangChain 개요]
    2. [LangChain 활용]
        - [LLM + LangChain으로 데이터 처리하기]
        - [LLM + LangChain으로 대화하기]


### 4. 서비스 기획 구현

- Pystagram 만들기 (코드출처: 이한영의 Django 입문)
    - [환경구축 및 기본 정보 설정]
    - [로그인/로그아웃 기능 구현]
    - [회원가입 기능 구현]
    - [글 관리 기능 구현]
    - [부가 기능 구현]
    - [글 상세 관리 기능 구현]