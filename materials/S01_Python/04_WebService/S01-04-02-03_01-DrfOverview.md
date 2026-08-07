---
layout: page
title:  "Django REST Framework(DRF)"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-03_01-DrfOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Django REST Framework 개요

### 1.1 Django REST Framework(DRF)란?

- Django를 기반으로 REST API 서버를 만들기 위한 라이브러리
- REST API는 웹뿐만 아니라 앱과 같은 다양한 플랫폼의 백엔드 서비스를 위해 JSON과 같은 규격화된 데이터를 제공함
- `pip install djangorestframework` 명령으로 설치할 수 있음

### 1.2 주요 기능

#### 1.2.1 직렬화(Serialization)

- 모델 인스턴스를 JSON, XML 등 다양한 형식으로 변환할 수 있음
- `serializers.py` 파일을 통해 직렬화 로직을 정의함

    ```python
    #//file: "serializers.py"

    from rest_framework import serializers
    from .models import Post

    class PostSerializer(serializers.ModelSerializer):
        class Meta:
            model = Post
            fields = '__all__'
    ```

#### 1.2.2 뷰(Views)

- DRF는 다양한 뷰 클래스를 제공하여 API 엔드포인트를 쉽게 정의할 수 있음
- `APIView`, `GenericAPIView`, `ViewSet` 등을 사용하여 다양한 요구사항을 충족할 수 있음

    ```python
    #//file: "views.py"

    from rest_framework import generics
    from .models import Post
    from .serializers import PostSerializer

    class PostListCreate(generics.ListCreateAPIView):
        queryset = Post.objects.all()
        serializer_class = PostSerializer
    ```

#### 1.2.3 라우팅(Routing)

- Django의 URL 라우팅 시스템을 확장하여 API 엔드포인트를 쉽게 정의할 수 있음
    - 라우팅 시스템
        - 네트워크에서 데이터 패킷이 목적지까지 가장 효율적으로 전달될 수 있도록 경로를 선택하는 프로세스
        - 네트워크의 여러 노드(컴퓨터, 라우터 등) 간의 통신을 가능하게 하며, 인터넷과 같은 대규모 네트워크에서 특히 중요함
- `urls.py` 파일에서 라우터를 설정함

    ```python
    #//file: "urls.py"

    from django.urls import path, include
    from rest_framework.routers import DefaultRouter
    from .views import PostViewSet

    router = DefaultRouter()
    router.register(r'posts', PostViewSet)

    urlpatterns = [
        path('', include(router.urls)),
    ]
    ```

#### 1.2.4 인증 및 권한(Authentication and Permissions)

- 다양한 인증 방법(JWT, OAuth 등)을 지원하며, 사용자 권한을 세밀하게 설정할 수 있음
- `permissions.py` 파일에서 권한 로직을 정의함

    ```python
    #//file: "permissions.py"

    from rest_framework import permissions

    class IsOwnerOrReadOnly(permissions.BasePermission):
        def has_object_permission(self, request, view, obj):
            if request.method in permissions.SAFE_METHODS:
                return True
            return obj.owner == request.user
    ```

#### 1.2.5 필터링 및 페이지네이션(Filter and Pagination)

- 쿼리셋을 필터링하고, 페이지네이션을 통해 데이터를 효율적으로 관리할 수 있음
- `views.py` 파일에서 필터 및 페이지네이션 설정을 추가함

    ```python
    #//file: "views.py"

    from rest_framework import generics, filters
    from .models import Post
    from .serializers import PostSerializer

    class PostList(generics.ListAPIView):
        queryset = Post.objects.all()
        serializer_class = PostSerializer
        filter_backends = [filters.SearchFilter]
        search_fields = ['title', 'content']
    ```

## **2. Django REST Framework 예제 프로젝트**

### 2.1 프로젝트 생성

- 가상환경 설정

    ```bash
    python -m venv drf
    cd drf
    source ./bin/activate
    # Windows의 경우: ./Scripts/activate
    ```

- Django 설치 및 확인

    ```bash
    pip install django djangorestframework
    ```

- Project 시작

    ```bash
    django-admin startproject drfweb .
    ```

- 앱 추가

    ```bash
    python manage.py startapp drf
    python manage.py migrate
    ```

- 웹서버 실행

    ```bash
    python manage.py runserver
    ```

- 웹 브라우저에서 웹서버 시작 확인
    - https://127.0.0.1:8000

- 웹서버 설정 수정

    ```python
    #//file: "./restweb/settings.py"

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'rest_framework',   # 추가할 것
        'drf',          # 추가할 것
    ]

    TIME_ZONE = 'Asia/Seoul'    # 시간대를 한국으로 바꿀 것
    ```

### 2.2 Django REST Framework 프로젝트 구조

- HelloAPI 만들기

    ```python
    #//file: "./drf/views.py"

    from rest_framework.response import Response
    from rest_framework.decorators import api_view

    # Create your views here.
    @api_view(['GET'])
    def helloAPI(request):
        return Response("hello world!")
    ```

- URL 연결

    ```python
    #//file: "./drfweb/urls.py"

    from django.contrib import admin
    from django.urls import path, include

    urlpatterns = [
        path('admin/', admin.site.urls),
        path("drf/", include("drf.urls"))
    ]
    ```

    ```python
    #//file: "./drf/urls.py"

    from django.urls import path
    from .views import helloAPI

    urlpatterns = [
        path("hello/", helloAPI),
    ]
    ```

- REST API 확인
    - http://127.0.0.1:8000/drf/hello/

- Django vs DRF

|특징|Pure Django|Django REST Framework|
|------|---|---|
|개발 목적|웹 풀스택 개발|백엔드 API 서버 개발|
|개발 결과|웹 페이지를 포함한 웹 서비스|여러 클라이언트에서 사용할 수 있는 API 서버|
|응답 형태|HTML|JSON|
|다른 파일|templates|Serializers.py|
