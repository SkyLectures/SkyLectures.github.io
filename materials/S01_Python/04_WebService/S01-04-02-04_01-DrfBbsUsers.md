---
layout: page
title:  "DRF: 게시판 만들기-사용자 관리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-04_01-DrfBbsUsers
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

- 코드출처: 백엔드를 위한 Django REST Framework with 파이썬

## 1. 기능 구성

- **회원 관련 기능**
    - 회원 프로필 관리(닉네임, 관심사, 프로필 사진 등)
    - 회원 가입 기능
    - 로그인 기능
    - 프로필 수정하기 기능

- **게시글 관련기능**
    - 게시글 생성
    - 게시글 1개 가져오기
    - 게시글 목록 가져오기(가져오는 개수 제한하기)
    - 게시글 수정하기
    - 게시글 삭제하기
    - 게시글 좋아요 기능
    - 게시글 필터링(좋아요 누른 글/내가 작성한 글)
    - 게시글 각 기능마다 권한 설정

- **댓글 관련 기능**
    - 댓글 생성
    - 댓글 1개 가져오기
    - 댓글 목록 가져오기
    - 댓글 수정하기
    - 댓글 삭제하기
    - 게시글을 가져올 때 댓글도 가져오게 만들기

## 2. Project Setting

- 가상환경 생성

    ```bash
    python -m venv bbs
    cd bbs
    source ./bin/activate        # ./Scripts/activate
    ```

- DRF 프로젝트 생성

    ```bash
    pip install django djangorestframework
    django-admin startproject board .
    ```

- 관리자 계정 생성

    ```bash
    python manage.py createsuperuser
    ```

- DRF 기본 설정

    ```python
    #//file: "board/settings.py"

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'rest_framework',
    ]

    TIME_ZONE = 'Asia/Seoul'
    ```

## 3. 모델 구성

- **User 모델**

    - Django 기본 User 모델
        - 회원관리: 서비스를 위한 가장 기본적인 기능
        - User: Django Framework에서 기본 제공하는 모델의 하나
            - 회원관리를 위하여 User 모델을 새롭게 만들어도 되지만 현재 시점에서는 새로운 User 모델이 큰 의미가 없으므로 Django의 User 모델을 사용함
            - django.contrib.auth.models 안에 정의되어 있음
            - django.contrib.auth: 인증 기능을 위해 Django Framework가 미리 만들어둔 모듈
            - settings.py에서 기본으로 등록되어 있음

    - Django 기본 User 모델의 대표적인 필드
        - username: 문자열
            - 흔히 아는 ID가 들어가는 필드
            - ID인만큼 다른 사용자와 겹치면 안됨
            - 필수 항목
        - first_name: 문자열
            - 영문 이름에서 사용되는 이름의 개념
            - 선택 항목
        - last_name: 문자열
            - 영문 이름에서 사용되는 성씨의 개념
            - 선택 항목
        - email: 문자열
            - 회원 이메일 주소
            - 선택 항목
        - password: 문자열
            - 비밀번호
            - 필수 항목
            - 실제로 입력한 비밀번호 문자열을 그대로 저장하지 않고 해시값을 저장함
                - Django에서 비밀번호를 안전하게 보관하는 해시 알고리즘 사용(공식문서 참고할 것)
        - 그외 기타 필드들

## 4. 회원관리 구현

### 4.1 회원 인증의 개념

- 회원 인증 개념 (1): ID/PW를 그대로 담아 보내기
    - 가장 기본적인 인증 방법
    - Django는 보안을 위하여 회원의 비밀번호를 해시값으로 저장하고 있지만
    - 클라이언트의 입장에서는 그냥 비밀번호를 적어서 보내는 것일 뿐임
    - 인증이 필요할 때마다 ID/PW를 전송해야 하며 중간에서 탈취당할 위험이 높음
    - 보안이 매우 취약한 상태

<p align="center"><img src="/materials/images/python/S01-04-02-04_01-001.png" width="400"></p>

- 회원 인증 개념 (2): 세션 & 쿠키의 사용
    - 세션(Session): 서버쪽에 저장하는 정보
    - 쿠키(Cookies): 클라이언트의 자체적인 저장소
        - 쿠키는 데이터로 구성되긴 하지만 데이터라기보다는 데이터를 저장하는 임시저장소와 같은 역할을 수행함
    - 한 번 로그인을 수행한 후, 로그인 정보를 이용하여 세션에서 발급하는 세션 ID를 보냄으로써 인증을 대체함
    - 클라이언트에서는 세션 ID를 쿠키 저장소에 저장한 후 인증 요청이 있을때마다 세션 ID를 꺼내서 HTTP 헤더에 넣고 전송함
    - 인증이 필요할 때마다 ID/PW를 전송할 필요가 없으므로 정보의 탈취 위험이 줄어들지만 세션 ID를 탈취당할 위험은 여전히 존재함

<p align="center"><img src="/materials/images/python/S01-04-02-04_01-002.png" width="500"></p>

- 회원 인증 개념 (3): 토큰 & JWT
    - 세션 & 쿠키 방식과 비슷함
    - 회원 가입 시 서버는 유저에 매칭되는 토큰을 생성하여 클라이언트에 전달하고 클라이언트는 인증 요청 시마다 해당 토큰을 HTTP 헤더에 넣어서 전송함
    - 토큰에는 사용자의 정보가 포함되어 있으므로 서버는 해당 정보를 이용하여 인증을 수행함
    - 토큰은 암호화 방식이 적용되어 있으므로 전송 데이터 패킷을 중간에서 탈취당하더라도 정보가 노출될 가능성은 낮음
    - 암호화에 사용되는 키: settings.py에 등록되어 있는 SECRET_KEY
    - 그러나 토큰 자체를 탈취하여 사용자인척 하는 위험은 여전히 존재함
        - 해결책: 토큰의 유효기간 설정
    <br><br>
    - JWT (JSON WEB Token)
        - 사용자 인증을 위해 사용하는 Open Standard(RFC 7519)
        - JSON 포맷을 이용하여 Self-Contained 방식으로 사용자에 대한 정보를 저장하는 Claim 기반의 WEB 토큰
        - 기본 컨셉
            - IdP (Identity Provider)가 사용자의 정보를 담은 내용에 서명하는 것을 통해서 토큰을 생성
            - 유저가 서버에 요청할 때 이 토큰을 사용하며, 이 때 토큰의 무결성(integrity)과 인증성(authenticity)을 보장함
                - 무결성: 정보가 원래의 내용으로 유지되는 것. 즉 정보가 변조되지 않았음을 보장하는 것
                - 인증성: 보낸 사람과 받는 사람이 서로가 맞다고 확인할 수 있는 성질. 정보의 출처를 확인하고 신뢰할 수 있는지 판단하는 과정
            - 유저가 전송하는 데이터를 숨기는 것보다 유저가 전송하는 데이터를 인증하는데 집중하는 방식

<p align="center"><img src="/materials/images/python/S01-04-02-04_01-003.png" width="500"></p>

### 4.2 구현

- **users 앱 생성**

    ```bash
    python manage.py startapp users
    ```

- **설정 추가**

    ```python
    #//file: "board/settings.py"

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'rest_framework',
        'rest_framework.authtoken',
        'users',
    ]

    ...

    REST_FRAMEWORK = {
        'DEFAULT_AUTHENTICATION_CLASSES': [
            'rest_framework.authentication.TokenAuthentication',
        ],
    }
    ```

- **Model 구현**
    - Django User 모델에서 활용할 필드 목록
        - username
            - ID로 활용
            - required=True
        - email
            - required=True
        - password
            - required=True

- **Serializer**
    - 회원가입 프로세스
        1. 사용자가 정해진 폼에 따라 데이터를 입력함
            - username, email, password, password2
        2. 해당 데이터가 들어오면 ID가 중복되지 않는지, 비밀번호가 너무 짧거나 쉽지는 않은지 검사함
        3. 2단계를 통과했다면 회원을 생성함
        4. 회원 생성이 완료되면 해당 회원에 대한 토큰을 생성함

    - Serializer는 요청으로 들어온 데이터를 Django 데이터로 변환하여 저장하는 기능을 수행함
    - 또한 Serializer는 검증(Validation) 기능을 수행하는 역할을 가지고 있음
        - 이번 Serializer에서는 검증기능을 사용하도록 함

            ```python
            #//file: "users/serializers.py"

            from django.contrib.auth.models import User
            from django.contrib.auth.password_validation import validate_password

            from rest_framework import serializers
            from rest_framework.authtoken.models import Token
            from rest_framework.validators import UniqueValidator


            class RegisterSerializer(serializers.ModelSerializer):
                email = serializers.EmailField(
                    required=True,
                    validators=[UniqueValidator(queryset=User.objects.all())],
                )
                password = serializers.CharField(
                    write_only=True,
                    required=True,
                    validators=[validate_password],
                )
                password2 = serializers.CharField(write_only=True, required=True)

                class Meta:
                    model = User
                    fields = ('username', 'password', 'password2', 'email')

                def validate(self, data):
                    if data['password'] != data['password2']:
                        raise serializers.ValidationError(
                            {"password": "Password fields didn't match."})

                    return data

                def create(self, validated_data):
                    user = User.objects.create_user(
                        username=validated_data['username'],
                        email=validated_data['email'],
                    )

                    user.set_password(validated_data['password'])
                    user.save()
                    token = Token.objects.create(user=user)
                    return user
            ```

- **View 구현**
    - Serializer가 복잡해진 대신 View가 간단해짐
    - 회원가입의 경우 회원 생성 기능만 있음(POST)
        - 굳이 ViewSet을 사용해 다른 API 요청을 처리할 필요가 없음

            ```python
            #//file: "users/views.py"
            from django.contrib.auth.models import User
            from rest_framework import generics

            from .serializers import RegisterSerializer


            class RegisterView(generics.CreateAPIView):
                queryset = User.objects.all()
                serializer_class = RegisterSerializer
            ```

- **URL 설정**
    - 클래스형 뷰를 사용할 것이므로 .as_view()를 사용함

        ```python
        #//file: "users/urls.py"
        from django.urls import path
        from .views import RegisterView

        urlpatterns = [
            path('register/', RegisterView.as_view()),
        ]
        ```

        ```python
        #//file: "board/urls.py"
        from django.urls import path, include
        from django.contrib import admin

        urlpatterns = [
            path('admin/', admin.site.urls),
            path('users/', include('users.urls')),
        ]
        ```

- **Migration & Run Server**
    - Migration

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

    - Run Server

        ```bash
        python manage.py runserver
        ```

    - 기능 테스트 수행
        - 관리자 계정을 통해 기능이 제대로 수행되었는지 확인


## 5. Login 구현

- Serializer

    ```python
    #//file: "users/serializers.py"

    from django.contrib.auth import authenticate

    class LoginSerializer(serializers.Serializer):
        username = serializers.CharField(required=True)
        password = serializers.CharField(required=True, write_only=True)

        def validate(self, data):
            user = authenticate(data)
            if user:
                token = Token.objects.get(user=user)
                return token
            raise serializers.ValidationError(
                {"error": "Unable to log in with provided credentials."})
    ```

- View

    ```python
    #//file: "users/views.py"
    from django.contrib.auth.models import User
    from rest_framework import generics, status
    from rest_framework.response import Response

    from .serializers import RegisterSerializer, LoginSerializer

    class LoginView(generics.GenericAPIView):
        serializer_class = LoginSerializer

        def post(self, request):
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            token = serializer.validated_data
            return Response({"token": token.key}, status=status.HTTP_200_OK)
    ```

- URL

    ```python
    #//file: "users/urls.py"
    from django.urls import path
    from .views import RegisterView, LoginView

    urlpatterns = [
        path('register/', RegisterView.as_view()),
        path('login/', LoginView.as_view()),
    ]
    ```

- 실행

    ```bash
    python manage.py runserver
    ```

## 6. User 모델의 확장

- 프로젝트에서 사용하는 회원 모델
    - username: 아이디(CharField, primary=True)
    - email: 이메일 주소(EmailField)
    - password: 비밀번호(CharField)
    - nickname: 닉네임(CharField)
    - position: 직종(CharField)
    - subjects: 관심사(CharField)
    - image: 프로필 이미지(ImageField)

- Profile 모델

  <p align="left"><img src="/materials/images/python/S01-04-02-04_01-004.png" width="400"></p>

    ```python
    #//file: "user/models.py"
    from django.db import models
    from django.contrib.auth.models import User
    from django.db.models.signals import post_save
    from django.dispatch import receiver

    # Create your models here.
    class Profile(models.Model):
        user = models.OneToOneField(User, on_delete=models.CASCADE)
        nickname = models.CharField(max_length=128)
        position = models.CharField(max_length=128)
        subjects = models.CharField(max_length=128)
        image = models.ImageField(upload_to='profile/', default='default.png')

    @receiver(post_save, sender=User)
    def create_user_profile(sender, instance, created, kwargs):
        if created:
            Profile.objects.create(user=instance)
    ```

- Migration

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
    ```

## 7. 사진 처리를 위한 설정 처리

- 라이브러리 설치

    ```bash
    pip install Pillow
    ```

- 미디어 파일에 대한 경로 지정
    - STATIC_URL: 정적 파일, 시스템에서 사용하는 리소스 파일의 저장 경로
    - MEDIA_ROOT: 사용자가 업로드하는 파일의 저장경로(절대경로)
    - MEDIA_URL: 사용자가 업로드하는 파일의 경로(읽기 위한 경로, 상대경로)

    ```python
    #//file: "board/settings.py"

    import os

    STATIC_URL = '/static/'
    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

    - URL

    # myboard/urls.py
    from django.urls import path, include
    from django.contrib import admin

    from django.conf import settings
    from django.conf.urls.static import static

    urlpatterns = [
        path('admin/', admin.site.urls),
        path('users/', include('users.urls')),
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    ```

- Serializer

    ```python
    #//file: "users/serializers.py"
    from .models import Profile


    class ProfileSerializer(serializers.ModelSerializer):
        class Meta:
            model = Profile
            fields = ("nickname", "position", "subjects")
            # extra_kwargs = {"image": {"required": False, "allow_null": True}}
    ```

- View + 기본 Permission
    - 개발할 프로필 관련 기능
        - 읽어오기, 수정하기 → generics.RetrieveUpdateAPIView를 이용하여 기능 구현 가능
    - 요구되는 권한
        - 프로필 조회: 모두
        - 프로필 수정: 해당 프로필의 소유자만 가능 → permisstion_class 필드 설정을 통해 구현
    - API마다 필요한 권한이 다른 경우
        - 권한이 미리 조합된 클래스 활용
        - 직접 권한 클래스를 만들어서 설정
    - Django Rest Framework에서 제공하는 권한 종류의 예시
        - AllowAny: 모든 요청을 통과시킴. 어떤 인증도 불필요함
        - IsAuthenticated: 인증된 경우에만 통과시킴. 즉 우리가 선언한 인증 방법으로 인증을 통과한 요청만 가능한 권한
        - IsAdminUser: 관리자인 경우에만 통과

    ```python
    #//file: "users/views.py"

    from .serializers import RegisterSerializer, LoginSerializer, ProfileSerializer
    from .models import Profile
    from rest_framework import generics

    class ProfileView(generics.RetrieveUpdateAPIView):
        queryset = Profile.objects.all()
        serializer_class = ProfileSerializer
    ```

    ```python
    #//file: "users/permissions.py"

    from rest_framework import permissions

    class CustomReadOnly(permissions.BasePermission):
        def has_object_permission(self, request, view, obj):
            if request.method in permissions.SAFE_METHODS:
                return True
            return obj.author == request.user
    ```

    ```python
    #//file: "users/urls.py"
    from django.urls import path
    from .views import RegisterView, LoginView, ProfileView

    urlpatterns = [
        path('register/', RegisterView.as_view()),
        path('login/', LoginView.as_view()),
        path('profile/<int:pk>/', ProfileView.as_view())
    ]
    ```

- Admin 페이지 등록
    - User 모델만 관리자 페이지에 등록하게 되면 프로필 모델은 나타나지 않음
    - 프로필 모델을 따로 등록하면 관리자 페이지에서는 볼 수 있지만 유저 테이블과 프로필 테이블이 분리되어 있으므로 불편함
    - 아래와 같은 방법으로 두 모델이 같은 모델인 것처럼 함께 볼 수 있음

    ```python
    #//file: "user/admin.py"
    from django.contrib import admin

    from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
    from django.contrib.auth.models import User
    from .models import Profile

    class ProfileInline(admin.StackedInline):
        model = Profile
        can_delete = False
        verbose_name_plural = "profile"

    class UserAdmin(BaseUserAdmin):
        inlines = (ProfileInline, )

    admin.site.unregister(User)
    admin.site.register(User, UserAdmin)
    ```

- Migration

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
    ```