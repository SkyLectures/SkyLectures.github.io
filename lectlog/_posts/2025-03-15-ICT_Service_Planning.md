---
layout: post
title:  "AI 기반 서비스의 기획 및 구현(80H)"
date:   2025-03-15 09:00:00 +0900
categories: lectlog
---

# [03.18] AI 기반 서비스의 기획 및 구현(80H)

## [강의 개요]

* 고객사: 동북 ICT
* 주제: AI 기반 서비스의 기획 및 구현
* 강의기간: 2025.03.18~2025.04.30
* 강의시간: 80H

## [강의 내용]

### 1. 파이썬 기초

- [Python 개요](/materials/S01-01-01-00_Python_Overview)
- [Python 가상환경 설정](/materials/S01-01-02-00_Virtual_Environment)
- [기본 문법](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-01-03-001_Basic.ipynb)

### 2. 파이썬 라이브러리 활용
- [Numpy](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-01-03-017_Library_Numpy.ipynb)
- [Pandas](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-01-03-018_Library_Pandas.ipynb)
- [Matplotlib](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-01-03-019_Library_Matplotlib.ipynb)

### 2. 웹서비스 시스템 구현

- [웹서비스 시스템 기초](/materials/S01-04-01-00_Web_Service_Development_Overview)
- [Simple Web Server 실습](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-01-01_Simple_Web_Server.ipynb)

- Django WebFramework 프로그래밍
    - [Django WebFramework 개요](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-02-01_Django_Overview.ipynb)
    - [Django 기반 서비스 기본 흐름](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-02-02_Django_Basic_Service.ipynb)
    - Django 기반 서비스 구현 실습 (코드출처: 백엔드를 위한 Django REST Framework with 파이썬)
        - [사진 목록 보기](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-02-03_Django_Picture_List.ipynb)
        - [Todo List 웹서비스](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-02-04_Django_Todo_List.ipynb)

- Django REST Framework(DRF)
    - [RESTful API 개요](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-03-01_RESTful_API_Overview.ipynb)
    - [Django REST Framework 개요](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-03-02_DRF_Overview.ipynb)
    - DRF 활용 실습: 게시판 만들기 (코드출처: 백엔드를 위한 Django REST Framework with 파이썬)
        - [게시판 사용자 관리](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-03-03_01-DRF_BBS_Users.ipynb)
        - [게시판 글 관리](https://colab.research.google.com/github/SkyLectures/LectureMaterials/blob/main/Part01_Python/S01-04-03-03_02-DRF_BBS_Posts.ipynb)


### 3. LLM 활용

- [LLM 개요](/materials/S03-05-01-00_LLM_Overview)
- [LLM 활용(ChatGPT API / Ollama 사용 실습)](/materials/S03-05-02-00_LLM_Applications)
- LangChain
    1. [LangChain 개요](/materials/S03-05-03-01_LangChain_Overview)
    2. [LangChain 활용]
        - [LLM + LangChain으로 데이터 처리하기](/materials/S03-05-03-02_01_LangChain_Pandas)
        - [LLM + LangChain으로 대화하기](/materials/S03-05-03-02_02_LangChain_Chat)


### 4. 서비스 기획 구현

- Pystagram 만들기 (코드출처: 이한영의 Django 입문)
    - [개발환경 구성 및 인증시스템 구현](/materials/S01-04-03-04_01-WebService_Pystagram_01)
    - [글 관리 기능 구현](/materials/S01-04-03-04_02-WebService_Pystagram_02)
    - [부가 기능 구현](/materials/S01-04-03-04_03-WebService_Pystagram_03)