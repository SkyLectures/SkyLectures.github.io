---
layout: page
title:  "LangChain"
date:   2025-04-03 14:30:00 +0900
permalink: /materials/S03-05-03-03_LangChain_Web_Service
categories: materials
---

## LangChain 기반 LLM의 WebService 활용: 영화정보 서비스 + LLM 챗봇


### 1. 기본 환경 구축
- 가상환경 만들기

    ```bash
    python -m venv movie_service
    cd movie_service
    source ./bin/activate
    ```

- 필요한 라이브러리 설치

    ```bash
    pip install numpy pandas matplotlib
    pip install django djangorestframework 
    pip install langchain langchain-community ollama
    ```

- 프로젝트 생성

    ```bash
    django-admin startproject movieweb .
    ```

- 웹서버 설정 수정
    - movieweb/settings.py

        ```python
        INSTALLED_APPS = [
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
            'rest_framework',
        ]
        ...
        LANGUAGE_CODE = 'ko-kr'
        TIME_ZONE = 'Asia/Seoul'
        ```

- Migration

    ```bash
    python manage.py migrate
    ```

- SuperUser 만들기

    ```bash
    python manage.py createsuperuser
    ```

- Web Server 작동 확인하기

    ```bash
    python manage.py runserver
    ```

- 웹브라우저에서 확인하기
    - 서버 작동 확인: http://127.0.0.1:8000/
    - 관리자 페이지 확인: http://127.0.0.1:8000/admin/ 

### 2. Home

- Static 설정하기

    ```python
    # movieweb/settings.py
    import os
    ...
    TEMPLATES = [
        {
            ...
            'DIRS': [os.path.join(BASE_DIR, 'templates'),],
            ...
        },
    ]
    ...
    STATIC_URL = '/static/'
    STATICFILES_DIRS = [
        os.path.join(BASE_DIR, 'static'),
    ]
    ```


- Template 만들기
    - movieweb/templates/base.html


    - movieweb/templates/home.html


- Views 만들기

    ```python
    import requests
    from django.shortcuts import render

    def home(request):
        context = {

        }
        return render(request, 'home.html', context)        
    ```

- 경로 설정
    - movieweb/urls.py
    
        ```python
        from django.contrib import admin
        from django.urls import path
        from . import views

        urlpatterns = [
            path('admin/', admin.site.urls),
            path('', views.home, name='home'),
        ]
        ```

### 3. Users
- 회원 가입, 로그인, 로그아웃을 처리함

- 앱 생성

    ```bash
    python manage.py startapp users
    ```

- Model 만들기
    - Django가 제공하는 User 모델 사용
        - class AbstractBaseUser
            - password = models.CharField(max_length=128)
            - last_login = models.DateTimeField()
            - is_active = True

        - class AbstractUser(AbstractBaseUser, PermissionsMixin):
            - username = models.CharField(max_length=150, unique=True)
            - first_name = models.CharField(max_length=150)
            - last_name = models.CharField(max_length=150)
            - email = models.EmailField()
            - is_staff = models.BooleanField()
            - is_active = models.BooleanField()
            - date_joined = models.DateTimeField()

        - class User(AbstractUser)

- Form 만들기(회원 가입)
    - movieweb/users/forms.py

        ```python
        from django import forms
        from django.contrib.auth.models import User

        class UserSignUpForm(forms.ModelForm): # 회원 로그인
            user_password = forms.CharField(widget=forms.PasswordInput, label="비밀번호")
            password_confirm = forms.CharField(widget=forms.PasswordInput, label="비밀번호 확인")

            class Meta:
                model = User
                fields = ['user_name', 'user_email', 'user_password']
                labels = {
                    'user_name': '사용자 이름',
                    'user_email': '이메일',
                    'user_password': '비밀번호',
                }

            def clean(self):
                cleaned_data = super().clean()
                password = cleaned_data.get("user_password")
                password_confirm = cleaned_data.get("password_confirm")

                if password != password_confirm:
                    raise forms.ValidationError("비밀번호가 일치하지 않습니다.")
                return cleaned_data
        ```

- Templates 만들기(회원 가입)
    - movieweb/users/templates/signup.html


- Views 만들기(회원 가입)
    - movieweb/users/views.py

        ```python
        from django.shortcuts import render
        from django.urls import reverse_lazy
        from django.views.generic.edit import FormView

        from users.forms import UserSignUpForm


        # Create your views here.
        class SignUpView(FormView):  # 클래스 이름을 SignUpView로 수정
            template_name = 'signup.html'
            form_class = UserSignUpForm
            success_url = reverse_lazy('login')

            def form_valid(self, form):
                user = form.save(commit=False)
                user.set_password(form.cleaned_data['user_password'])
                user.save()
                return super().form_valid(form)
        ```

- URL 연결하기
    - movieweb/users/urls.py

    ```python
    from django.urls import path
    from . import views

    app_name = 'users'

    urlpatterns = [
        path('signup/', views.SignUpView.as_view(), name='signup'),
    ]
   ```