---
layout: page
title:  "Django: 게시판 만들기-글 관리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-04_02-DrfBbsPosts
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Frontend(React)와 연동하기

- Backend 뿐만 아니라 Frontend의 처리 과정도 이해하여야 Frontend 개발자와 소통이 원활해짐

### 1.1 CORS 오류

- CORS (Cross-Origin Resource Sharing, 교차 출처 자원 공유)
    - Backend와 Frontend의 연동 과정에서 가장 빈번하고 가장 처음에 발생하는 오류
    - 엄밀하게는 오류가 아니라 개발 결과물을 안전하게 지켜주기 위한 것
        - 정책의 형태로 존재
        - 정책을 위반하려 할 때마다 오류를 발생시킴
    - 서로 다른 출처끼리 자원을 공유하는 것의 의미는?
        - 같은 출처란 http://127.0.0.1:8000, http://127.0.0.1:8000/users/ 와 같이 포트번호까지 동일한 경우를 의미
        - React의 경우 http://127.0.0.1:3000 이므로 호스트 주소는 같으나 포트번호가 달라서 서로 다른 출처가 됨
        - React에서 다른 출처인 Django로부터 데이터(리소스)를 가져오려는 시도가 SOP(Same Origin Policy)에 의해 차단되는 것
    - CORS는 이러한 상황에 대한 예외조항
        - 서버에서 CORS 정책을 준수하도록 설정해두면
        - SOP의 예외 조항인 CORS 정책을 준수하여 다른 출처끼리도 자원의 공유가 가능하게 됨
    - CORS는 프론트엔드에서 발생하는 오류이지만 백엔드에서 정책을 지켜야 하는 문제임

- CORS 해결을 위한 방법

    ```bash
    pip install django-cors-headers
    ```

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
        'corsheaders',
    ]

    MIDDLEWARE = [
        'corsheaders.middleware.CorsMiddleware',     # 순서가 중요함
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ]

    CORS_ORIGIN_ALLOW_ALL = True
    CORS_ALLOW_CREDENTIALS = True
    ```

- 이제 백엔드에서 할 일은 없음(Django에서 모두 처리함)

## 2. 게시글 기능 정리

- 게시글 생성
- 게시글 1개 가져오기
- 게시글 목록 가져오기(가져오는 개수 제한하기)
- 게시글 수정하기
- 게시글 삭제하기
- 게시글 좋아요 기능
- 게시글 필터링(좋아요 누른 글/내가 작성한 글)
- 게시글 각 기능마다 권한 설정

## 3. 게시글 모델 만들기 & 마이그레이션

- 앱 만들기

    ```bash
    python manage.py startapp posts
    ```

- settings.py 설정

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
        'corsheaders',
        'posts',
    ]
    ```

- Models

    - 모델의 필드 구성
        - 저자
        - 저자 프로필
        - 제목
        - 카테고리
        - 본문
        - 이미지 → 이미지가 없을 때는 default.png 파일 표시
        - 좋아요 누른 사람들 → 다대다(ManyToMany)
        - 글이 올라간 시간

        ```python
        #//file: "posts/models.py"
        from django.db import models
        from django.contrib.auth.models import User
        from django.utils import timezone
        from users.models import Profile

        class Post(models.Model):
            author = models.ForeignKey(User, on_delete=models.CASCADE)
            profile = models.ForeignKey(Profile, on_delete=models.CASCADE, blank=True)
            title = models.CharField(max_length=128)
            category = models.CharField(max_length=128)
            body = models.TextField()
            image = models.ImageField(upload_to='post/', default='default.png')
            likes = models.ManyToManyField(User)
            published_date = models.DateTimeField(default=timezone.now)
        ```
    
- 마이그레이션

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

- 마이그레이션 시 오류 발생
    - author, likes가 모두 User를 참조하는 중 → 오류 발생
    <br><br>
    - 관련 내용
        - relation_name에서의 오류 → 참조관계에서의 오류
            - 저자 → Users 모델을 ForeignKey로 참조 중(author와 연관) → post.author.username과 같이 참조 가능
            - User 모델에서는 post라는 이름을 모름 → user.post.title과 같이 참조 불가능

                ```
                # 이런 방식으로는 역관계에서도 데이터에 접근 가능
                user = User.objects.get(pk=1)
                posts = user.post_set.all()
                ```

                - 여기서 post_set 대신 사용하는 것이 related_name

                ```
                author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author')
                ```
                - 위와 같이 이름을 지정하고
                ```
                user = User.objects.get(pk=1)
                posts = user.posts.all()
                ```
                -이렇게 하면 유저가 작성한 글들을 확인할 수 있음
                <br><br>
    - 위의 오류에서는 author, likes가 모두 User를 참조하는 중<br>
        → 둘 다 related_name을 지정하지 않고 역관계로 User에서 author, likes에 참조하려고 하므로<br>
        → user.post_set.all()이 되어 어떤 것을 잠조해야 하는지 구분할 수 없음<br>
        → 오류 발생

    ```python
    #//file: "posts/models.py"
    from django.db import models
    from django.contrib.auth.models import User
    from django.utils import timezone
    from users.models import Profile

    class Post(models.Model):
        author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
        profile = models.ForeignKey(Profile, on_delete=models.CASCADE, blank=True)
        title = models.CharField(max_length=128)
        category = models.CharField(max_length=128)
        body = models.TextField()
        image = models.ImageField(upload_to='post/', default='default.png')
        likes = models.ManyToManyField(User, related_name='like_posts', blank=True)
        published_date = models.DateTimeField(default=timezone.now)
    ```

- 마이그레이션
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

## 4. 게시글 기능 만들기

- Serializers
    - PostSerializer()
        - 해당 게시글에 대한 모든 정보를 JSON으로 변환하여 전달하는 역할을 수행
        - profile 필드를 작성하지 않으면 profile 필드에는 profile의 PK 값만 나타나므로 ProfileSerializer를 포함하도록 함
            - 이런 형태를 Nested Serializer라고 함
    - PostCreateSerializer()
        - 게시물을 등록할 때 유저는 제목, 카테고리, 본문, 이미지 등만 입력함
        - 나머지는 시스템이 알아서 채워주거나 빈칸으로 남겨둠
        - 시리얼라이저는 유저가 입력한 데이터를 검증하고 Django 데이터로 변환하여 저장하게 하는 역할을 수행

        ```python
        #//file: "posts/serializers.py
        from rest_framework import serializers

        from users.serializers import ProfileSerializer
        from .models import Post


        class PostSerializer(serializers.ModelSerializer):
            profile = ProfileSerializer(read_only=True)

            class Meta:
                model = Post
                fields = ("pk", "profile", "title", "body", "image", "published_date", "likes")


        class PostCreateSerializer(serializers.ModelSerializer):
            class Meta:
                model = Post
                fields = ("title", "category", "body", "image")
        ```

- Views(CRUD) + 권한

    - 게시글은 모든 CRUD 기능이 포함되어 있으므로 ViewSet을 사용하여 쉽게 적용 가능함
    - 단, 각기 다른 시리얼라이저를 적재적소에 활용하도록 코드를 작성해야 함
    - 게시글 생성 시 유저가 입력해 주지 않는 저자 정보를 같이 넣을 수 있도록 함
    <br><br>
    - 요구되는 권한
        - 게시글의 조회: 모든 사람
        - 게시글의 생성: 인증된 유저만 가능
        - 게시글의 수정/삭제: 해당 글의 작성자만 가능
    - User에서의 CustomReadOnly와 다른 점
        - 각 객체별 권한 뿐만 아니라 전체 객체에 대한 권한도 포함해야 함(목록 조회/생성)
            - has_permission()을 함께 구현

    ```python
    #//file: "posts/permissions.py"
    from rest_framework import permissions

    class CustomReadOnly(permissions.BasePermission):
        ## 글 조회: 누구나, 생성: 로그인한 유저, 편집: 글 작성자
        def has_permission(self, request, view):
            if request.method == 'GET':
                return True
            return request.user.is_authenticated

        def has_object_permission(self, request, view, obj):
            if request.method in permissions.SAFE_METHODS:
                return True
            return obj.author == request.user
    ```

    ```python
    #//file: "posts/views.py"
    from rest_framework import viewsets

    from users.models import Profile
    from .models import Post
    from .permissions import CustomReadOnly
    from .serializers import PostSerializer, PostCreateSerializer

    class PostViewSet(viewsets.ModelViewSet):
        queryset = Post.objects.all()
        permission_classes = [CustomReadOnly]

        def get_serializer_class(self):
            if self.action == 'list' or 'retrieve':
                return PostSerializer
            return PostCreateSerializer

        def perform_create(self, serializer):
            profile = Profile.objects.get(user=self.request.user)
            serializer.save(author=self.request.user, profile=profile)
    ```

- URL
    - ViewSet을 사용하면 라우터가 함께 따라옴
        - 라우터에 등록된 url을 활용함

        ```python
        #//file: "posts/urls.py"
        from django.urls import path
        from rest_framework import routers

        from .views import PostViewSet

        router = routers.SimpleRouter()
        router.register('posts', PostViewSet)

        urlpatterns = router.urls
        ```

- 프로젝트의 urls.py에는 라우터가 이미 posts를 설정해줌
    - posts를 설정하면 실제 주소는 localhost:8000/posts/posts/와 같이 중복됨

        ```python
        #//file: "board/urls.py"

        from django.contrib import admin
        from django.urls import path, include

        from django.conf import settings
        from django.conf.urls.static import static

        urlpatterns = [
            path('admin/', admin.site.urls),
            path('users/', include('users.urls')),
            path('', include('posts.urls')),
        ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
        ```

- 실행

    ```bash
    python manage.py runserver
    ```

## 5. 필터링 기능 만들기

- 필터링 기능: 게시글 전체를 가져올 때 조건을 걸어 가져오도록 하는 기능
- Django에서 이미 지원하는 기능임
- View 등의 코드에서 따로 호출하지 않아도 프로젝트 전역에 적용됨

    ```bash
    pip install django-filter
    ```

- settings.py 설정

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
        'corsheaders',
        'posts',
        'django_filters',
    ]

    ...

    REST_FRAMEWORK = {
        'DEFAULT_AUTHENTICATION_CLASSES': [
            'rest_framework.authentication.TokenAuthentication',
        ],
        'DEFAULT_FILTER_BACKENDS': [
            'django_filters.rest_framework.DjangoFilterBackend',
        ],
    }
    ```

- Views

    ```python
    #//file: "posts/views.py"
    from django_filters.rest_framework import DjangoFilterBackend

    from rest_framework import viewsets

    from users.models import Profile
    from .models import Post
    from .permissions import CustomReadOnly
    from .serializers import PostSerializer, PostCreateSerializer


    class PostViewSet(viewsets.ModelViewSet):
        queryset = Post.objects.all()
        permission_classes = [CustomReadOnly]
        filter_backends = [DjangoFilterBackend]
        filterset_fields = ['author', 'likes']

        def get_serializer_class(self):
            if self.action == 'list' or 'retrieve':
                return PostSerializer
            return PostCreateSerializer

        def perform_create(self, serializer):
            profile = Profile.objects.get(user=self.request.user)
            serializer.save(author=self.request.user, profile=profile)
    ```

## 6. 페이징 기능

- Pagination
    - 게시글 전체 조회 페이지를 여러 페이지로 나누는 기능
    - 한 번에 모든 글을 가져오기 부담스러울 경우 한 번의 API 요청으로 가져울 수 있는 데이터의 수를 제한하는 기능
    - 별다른 작업은 필요없음 → settings.py의 REST_FRAMEWORK에 관련 기능을 추가하기만 하면 됨
    - 단, 결과 데이터는 results에 들어가서 프론트엔드에 전달되므로 프론트엔드에서는 데이터를 꺼내가는 과정이 추가로 요구됨

    ```python
    #//file: "board/settings.py"

    REST_FRAMEWORK = {
        'DEFAULT_AUTHENTICATION_CLASSES': [
            'rest_framework.authentication.TokenAuthentication',
        ],
        'DEFAULT_FILTER_BACKENDS': [
            'django_filters.rest_framework.DjangoFilterBackend',
        ],
        'DEFAULT_PAGINATION_CLASS':
        'rest_framework.pagination.PageNumberPagination',
        'PAGE_SIZE':
        3,
    }
    ```

## 7. 좋아요 기능

- Views
    - 좋아요 기능은 오직 likes 필드에만 영향을 주므로 간단한 GET 요청 하나로 처리 가능
    - 요구되는 설정
        - 데코레이터로 GET 요청을 받는 함수형 뷰라는 설정
        - 권한이 필요하다는 설정
            - 좋아요를 누르는 권한은 회원가입을 한 유저라면 모두 가능하므로 IsAuthenticated로 설정
    - 처리 내용
        - post.likes.all() 내에 request.user가 있으면 request.user 삭제
        - post.likes.all() 내에 request.user가 없으면 request.user 추가

        ```python
        #//file: "posts/views.py"

        from django_filters.rest_framework import DjangoFilterBackend

        from rest_framework import viewsets

        from rest_framework.decorators import api_view, permission_classes
        from rest_framework.generics import get_object_or_404
        from rest_framework.permissions import IsAuthenticated
        from rest_framework.response import Response

        from users.models import Profile
        from .models import Post
        from .permissions import CustomReadOnly
        from .serializers import PostSerializer, PostCreateSerializer

        class PostViewSet(viewsets.ModelViewSet):
            queryset = Post.objects.all()
            permission_classes = [CustomReadOnly]
            filter_backends = [DjangoFilterBackend]
            filterset_fields = ['author', 'likes']

            def get_serializer_class(self):
                if self.action == 'list' or 'retrieve':
                    return PostSerializer
                return PostCreateSerializer

            def perform_create(self, serializer):
                profile = Profile.objects.get(user=self.request.user)
                serializer.save(author=self.request.user, profile=profile)

        @api_view(['GET'])
        @permission_classes([IsAuthenticated])
        def like_post(request, pk):
            post = get_object_or_404(Post, pk=pk)
            if request.user in post.likes.all():
                post.likes.remove(request.user)
            else:
                post.likes.add(request.user)

            return Response({'status': 'ok'})
        ```

- URL

    ```python
    #//file: "posts/urls.py"

    from django.urls import path
    from rest_framework import routers
    from .views import PostViewSet, like_post

    router = routers.SimpleRouter()
    router.register('posts', PostViewSet)

    urlpatterns = router.urls + [
        path('like/<int:pk>/', like_post, name='like_post')
    ]
    ```

## 8. 댓글 기능 만들기

- 댓글 기능 정리
    - 댓글 생성
    - 댓글 1개 가져오기
    - 댓글 목록 가져오기
    - 댓글 수정하기
    - 댓글 삭제하기
    - 게시글을 가져올 때 댓글도 가져오게 만들기

- 댓글 모델 만들기
    - 댓글 모델에 필요한 필드들
        - 작성자, 작성자 프로필, 게시글, 내용
    - 댓글의 경우 게시글과 밀접한 연관이 있음 → 따로 모델을 만들 필요는 없음
    - Foreign Key로 유저, 프로필, 포스트와 연결됨 + 댓글 내용 텍스트만 추가

        ```python
        #//file: "posts/models.py"
        from django.db import models
        from django.contrib.auth.models import User
        from django.utils import timezone
        from users.models import Profile

        class Post(models.Model):
            author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author')
            profile = models.ForeignKey(Profile, on_delete=models.CASCADE, blank=True)
            title = models.CharField(max_length=128)
            category = models.CharField(max_length=128)
            body = models.TextField()
            image = models.ImageField(upload_to='post/', default='default.png')
            likes = models.ManyToManyField(User, related_name='like_posts', blank=True)
            published_date = models.DateTimeField(default=timezone.now)

        class Comment(models.Model):
            author = models.ForeignKey(User, on_delete=models.CASCADE)
            profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
            post = models.ForeignKey(Post, related_name='comments', on_delete=models.CASCADE)
            text = models.TextField()
        ```

- 마이그레이션

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

- Serializers
    - 댓글을 작성할 때, 가져올 때 각각 다른 시리얼라이저가 필요함 → 게시글 시리얼라이저와 비슷
    - 게시글에서도 댓글을 불러올 수 있어야 함 → Nested Serializer 개념 활용(작성해 놓은 댓글 시리얼라이저를 게시글 시리얼라이저에 넣어주기)
    - 게시글 시리얼라이저에 댓글 시리얼라이저가 포함됨 → 댓글 시리얼라이저가 더 위에 선언되어야 함

        ```python
        #//file: "posts/serializers.py"
        from rest_framework import serializers

        from users.serializers import ProfileSerializer
        from .models import Post, Comment

        class CommentSerializer(serializers.ModelSerializer):
            profile = ProfileSerializer(read_only=True)

            class Meta:
                model = Comment
                fields = ("pk", "profile", "post", "text")

        class CommentCreateSerializer(serializers.ModelSerializer):
            class Meta:
                model = Comment
                fields = ("post", "text")

        class PostSerializer(serializers.ModelSerializer):
            profile = ProfileSerializer(read_only=True)
            comments = CommentSerializer(many=True, read_only=True)

            class Meta:
                model = Post
                fields = ("pk", "profile", "title", "body", "image", "published_date", "likes", "comments")

        class PostCreateSerializer(serializers.ModelSerializer):
            image = serializers.ImageField(use_url=True, required=False)

            class Meta:
                model = Post
                fields = ("title", "category", "body", "image")
        ```

- Views

    - ViewSet 사용
    - 댓글에 필요한 권한은 게시글과 동일
        - 댓글 보기: 모두
        - 댓글 작성: 유저만
        - 댓글 수정/삭제: 해당 댓글 작성자만 → CustomReadOnly 활용

        ```python
        #//file: "posts/views.py"

        from django_filters.rest_framework import DjangoFilterBackend

        from rest_framework import viewsets
        from rest_framework.decorators import api_view, permission_classes
        from rest_framework.generics import get_object_or_404
        from rest_framework.permissions import IsAuthenticated
        from rest_framework.response import Response
        from rest_framework import generics, status

        from users.models import Profile
        from .models import Post, Comment
        from .permissions import CustomReadOnly
        from .serializers import PostSerializer, PostCreateSerializer, CommentSerializer, CommentCreateSerializer

        class PostViewSet(viewsets.ModelViewSet):
            queryset = Post.objects.all()
            permission_classes = [CustomReadOnly]
            filter_backends = [DjangoFilterBackend]
            filterset_fields = ['author', 'likes']

            def get_serializer_class(self):
                if self.action == 'list' or 'retrieve':
                    return PostSerializer
                return PostCreateSerializer

            def perform_create(self, serializer):
                profile = Profile.objects.get(user=self.request.user)
                serializer.save(author=self.request.user, profile=profile)

        @api_view(['GET'])
        @permission_classes([IsAuthenticated])
        def like_post(request, pk):
            post = get_object_or_404(Post, pk=pk)
            if request.user in post.likes.all():
                post.likes.remove(request.user)
            else:
                post.likes.add(request.user)

            return Response({'status': 'ok'})

        class CommentViewSet(viewsets.ModelViewSet):
            queryset = Comment.objects.all()
            permission_classes = [CustomReadOnly]

            def get_serializer_class(self):
                if self.action == 'list' or 'retrieve':
                    return CommentSerializer
                return CommentCreateSerializer

            def perform_create(self, serializer):
                profile = Profile.objects.get(user=self.request.user)
                serializer.save(author=self.request.user, profile=profile)
        ```

- URL

    ```python
    #//file: "posts/urls.py"

    from django.urls import path
    from django.urls import path
    from rest_framework import routers

    from .views import PostViewSet, like_post, CommentViewSet

    router = routers.SimpleRouter()
    router.register('posts', PostViewSet)
    router.register('comments', CommentViewSet)

    urlpatterns = router.urls + [
        path('like/<int:pk>/', like_post, name='like_post')
    ]
    ```

- 실행

    ```bash
    python manage.py runserver
    ```