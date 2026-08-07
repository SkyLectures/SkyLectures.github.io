---
layout: page
title:  "Django 기반 서비스 기본 흐름"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-01_02-DjangoBasedServiceProcess
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## **1. Django 시작하기**

- 가상환경 설정

    ```bash
    python -m venv django_photolist
    cd django_photolist
    source ./bin/activate

    # Windows의 경우: ./Scripts/activate
    ```

- Django 설치 및 확인

    ```bash
    pip install django
    python -m django --version
    python --version
    ```

- Project 시작

    ```bash
    django-admin startproject photoweb .
    ```

- 앱 추가
    - Django에서 앱은 프로젝트 내에서 특정 기능을 수행하는 독립적인 모듈을 가리킴
    - 앱은 프로젝트의 기능을 더 작은 구성 요소로 나누어 코드베이스를 구성하는 데 도움을 줌
        - 예: 블로그 프로젝트
            - 인증 및 권한 부여 전용 앱
            - 블로그 게시물 전용 앱 등
    - 앱의 특징
        - 모듈화: 앱은 특정 기능을 수행하는 독립적인 코드와 리소스의 모음
        - 재사용성: 앱은 다른 프로젝트에서도 쉽게 재사용할 수 있음
        - 구성 요소: 앱은 템플릿, URL, 모델, 뷰 등을 포함하는 모듈

            ```bash
            python manage.py startapp photolist
            ```

- 웹서버 실행

    ```bash
    python manage.py runserver
    ```

- 웹 브라우저에서 웹서버 시작 확인
    - https://127.0.0.1:8000
    - 8000은 Django가 사용하는 포트 번호

- 생성된 폴더-파일 구조

<p align="center"><img src="/materials/images/python/S01-04-02-002_001.png" width="700"></p>

- 웹서버 설정 수정

    ```python
    #//file: "./photoweb/settings.py"
    # Application definition

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'photolist',    # 추가할 것
    ]

    #TIME_ZONE = 'UTC'            # 시간대를 한국으로 바꿀 것
    TIME_ZONE = 'Asia/Seoul'
    ```

- 웹서버 설정 URL 확인
    - ./photoweb/urls.py

## **2. Django 프로젝트 구조 살펴보기**

- MTV (Model-Template-View) 패턴
    - Django에서는 MTV 패턴으로 전반적인 개발을 진행함
    - 어떤 패턴으로 개발을 진행 → 작업 시, 규칙처럼 정해진 방식이 있고, 그 방식을 따라가며 요구하는 내용을 순서대로 채워나가는 것으로 개발을 진행한다는 의미
    <br><br>
    - Model: 앱의 데이터와 관련된 부분을 다룸
    - Template: 사용자에게 보이는 부분을 다룸
    - View: 
        - Model과 Template의 사이에서 Model의 메시지를 Template으로 전달, 
        - Template에서 발생하는 이벤트를 처리하는 부분

<p align="center"><img src="https://velog.velcdn.com/images%2Fkylehan91%2Fpost%2F7e6acd8e-594d-4b7c-aa7e-1c67870b949e%2Fimage.png" width="700"></p>

<p align="center"><img src="https://velog.velcdn.com/images/sossont/post/5e9c5550-a86f-4189-af5f-3303a7cda90e/image.png" width="700"></p>

- https://127.0.0.1:8000/admin/ 접속해보기

- 계정 만들기

    ```bash
    python manage.py createsuperuser
    ```

- 오류 발생
    - Migration 오류 수정

        ```bash
        python manage.py migrate
        ```

- 계정 만들기 재시도

    ```bash
    python manage.py createsuperuser
    ```

    - ID : seokhwan
    - email : seokhwan@1.1.1
    - password : 1234

## 3. Django Model 알아보기

### 3.1 Model
- 앱의 데이터와 관련된 부분을 다루는 영역
- 데이터베이스에 저장될 데이터의 모양을 정의하고 관련된 일부 기능들을 설정해주는 영역
- 현실 세상을 코드로 옮기는 과정이라고 생각할 수 있음

- Model 만들기

    ```python
    #//file: "./photolist/models.py"

    from django.db import models

    class Photo(models.Model):
        title = models.CharField(max_length=50)
        author = models.CharField(max_length=50)
        image = models.CharField(max_length=200)
        description = models.TextField()
        price = models.IntegerField()
    ```

- Model 적용시키기

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

- Model을 Admin 페이지에 적용시키기

    ```python
    #//file: "./photolist/admin.py"

    from django.contrib import admin
    from .models import Photo

    # Register your models here.
    admin.site.register(Photo)
    ```

## 4. Django Template 알아보기

- Template
    - 사용자에게 보이는 부분 → 웹페이지의 골격이라고 할 수 있는 HTML로 작성된 부분
    - 일반 HTML 작성 방법과 거의 동일하나 Template Tag를 사용한다는 점이 다름
    - Template Tag: HTML이 파이썬 코드로부터 데이터를 바로 넘겨받아서 처리할 수 있는 도구

## 5. Django View, URL 알아보기

- View
    - Model과 Template을 이어주는 다리와 같은 역할
    - Model을 통해 데이터에 접근하여 Template으로부터 요청받은 데이터를 뽑아와 Template에게 답변으로 보내줌
    - **Model**이 Django 프로젝트의 핵심이라면 **View**는 코드 중에서 가장 많은 비중을 차지하는 요소
<br><br>

- URL
    - 라우팅의 역할과 동시에 서버로 해당 주소에 할당된 리소스를 요청하는 역할을 담당
    - 여기서의 리소스는 HTML 페이지뿐만 아니라 내부를 채우는 데이터 등을 포함하는 개념임