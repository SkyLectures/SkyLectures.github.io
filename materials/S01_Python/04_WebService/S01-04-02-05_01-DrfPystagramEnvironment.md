---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-02-05_01-DrfPystagramEnvironment
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 환경구축 및 기본 정보 설정

### 1. 기능 설정

- 인증 시스템
- 피드 페이지
- 글과 댓글
- 동적 URL
- 해시 태크
- 글 상세 페이지
- 좋아요 기능
- 팔로우/팔로잉 기능
<br><br>

### 2. 기본 환경 구축
- 가상환경 만들기

    ```bash
    python -m venv pystagram
    cd pystagram
    source ./bin/activate
    ```

- 필요한 라이브러리 설치

    ```bash
    pip install django Pillow
    ```

- 프로젝트 생성
    - 프로젝트의 기본 구조 및 환경을 설정하는 역할이므로 "config"라는 이름으로 고정할 것을 권장하는 개발자도 있음

    ```bash
    django-admin startproject config .
    ```

- 기능별 디렉토리 설정

    ```bash
    mkdir templates
    mkdir static
    mkdir media
    ```

- 웹서버 설정 수정
    - config/settings.py

        ```python
        ...
        TEMPLATES = [
            {
                ...
                "DIRS": [BASE_DIR / "templates"],
            }
        ]
        ...
        LANGUAGE_CODE = 'ko-kr'
        TIME_ZONE = 'Asia/Seoul'
        ...
        STATIC_URL = "static/"
        STATICFILES_DIRS = [BASE_DIR / "static"]

        MEDIA_URL = "media/"
        MEDIA_ROOT = BASE_DIR / "media"
        ```

- 연결 경로 설정
    - config/urls.py

        ```python
        from django.conf import settings
        from django.conf.urls.static import static

        # 기존에 등록된 urlpatterns에 추가 설정
        urlpatterns += static(
            prefix=settings.MEDIA_URL,
            document_root=settings.MEDIA_ROOT,
        )
        ```

<br>

### 3. 인덱스 페이지 구성

- localhost:8000 뒤에 아무런 경로 추가 없이 기본적으로 보여 줄 인덱스 페이지 구성

- Template 만들기
    - templates/index.html

        ```html
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pystagram</title>
        </head>
        <body>
            <h1>Pystagram</h1>
        </body>
        </html>
       ```

- Views 만들기
    - config/views.py

        ```python
        from django.shortcuts import render

        def index(request):
            return render(request, 'index.html')
        ```

- URL 경로 연결
    - config/urls.py

        ```python
        from config.views import index

        urlpatterns = [
            path('admin/', admin.site.urls),
            path('', index),
        ]
        ...
        ```

- Web Server 작동 확인하기

    ```bash
    python manage.py runserver
    ```

- 웹브라우저에서 확인하기
    - 서버 작동 확인: http://127.0.0.1:8000/

- CustomUser를 사용할 것이므로 이 시점에서 Migration을 할 경우, User 모델 등록 후 Migration 시 오류가 발생할 때가 있음
<br><br>

### 4. 사용자 관리 기능 구현

#### 4.1 CustomUser 모델 설정

- APP 생성

    ```bash
    python manage.py startapp users
    ```

    - config/settings.py에 추가

        ```python
        INSTALLED_APPS = [
            ...
            'users',
        ]
        ```

- User 모델 생성
    - AbstractUser 모델의 필드
        - username (사용자명, 로그인 할 때의 아이디)
        - password (비밀번호)
        - first_name (이름)
        - last_name (성)
        - email (이메일)
        - is_staff (관리자 여부)
        - is_active (활성화 여부)
        - date_joined (가입일시)
        - last_login (마지막 로그인 일시)

    - users/models.py

        ```python
        from django.db import models
        from django.contrib.auth.models import AbstractUser

        class User(AbstractUser):
            pass
        ```
    - config/settings.py

        ```python
        AUTH_USER_MODEL = 'users.User'
        ```

    - 시스템에 반영

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

- 관리자 페이지에 모델 등록
    - 관리자 생성

        ```bash
        python manage.py createsuperuser
        ```

    - Web Server 작동 확인하기

        ```bash
        python manage.py runserver
        ```

    - 웹브라우저에서 확인하기
        - 관리자 페이지 확인: http://127.0.0.1:8000/admin/ 

    - users/admin.py

        ```python
        from django.contrib import admin
        from django.contrib.auth.admin import UserAdmin

        from users.models import User

        @admin.register(User)
        class CustomUserAdmin(UserAdmin):
            pass
        ```

- CustomUser 모델에 Field 추가
    - users/models.py

        ```python
        class User(AbstractUser):
            profile_image = models.ImageField("프로필 이미지", upload_to="users/profile", blank=True)
            short_description = models.TextField("소개글", blank=True)
        ```
       
    - 시스템에 반영

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

    - 관리자 페이지에 모델 등록
        - users/admin.py

            ```python
            @admin.register(User)
            class CustomUserAdmin(UserAdmin):
                fieldsets = [
                    (None, {"fields": ("username", "password")}),
                    ("개인정보", {"fields": ("first_name", "last_name", "email")}),
                    ("추가필드", {"fields": ("profile_image", "short_description")}),
                    ("권한", {"fields": ("is_active", "is_staff", "is_superuser")}),
                    ("중요한 일정", {"fields": ("last_login", "date_joined")}),
                ]
           ```
