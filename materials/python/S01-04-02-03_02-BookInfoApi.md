---
layout: page
title:  "DRF: 도서 정보 API"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-03_02-BookInfoApi
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 도서 정보 API

### 1.1 Model

- 모델 만들기

    ```python
    #//file: "./drf/models.py"

    from django.db import models

    class Book(models.Model):
        bid = models.IntegerField(primary_key=True)
        title = models.CharField(max_length=50)
        author = models.CharField(max_length=50)
        category = models.CharField(max_length=50)
        pages = models.IntegerField()
        price = models.IntegerField()
        published_date = models.DateField()
        description = models.TextField()
    ```

- Model 적용시키기

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

### 1.2 Serializer(직렬화)

- 직렬화(Serialization)란?
    - 객체를 데이터 스트림으로 변환하는 과정을 의미함
    - 이 과정을 통해 객체는 파일에 저장되거나 네트워크를 통해 전송될 수 있음
    - 직렬화는 주로 객체의 상태를 영속적으로 저장하거나 다른 시스템 간에 객체를 전송할 때 사용됨
    - Django의 경우 ORM 방식을 이용하여 파이썬이 사용하는 형태로 데이터를 저장하지만 클라이언트는 이를 이해하지 못하므로 클라이언트가 읽을 수 있는 JSON 등의 형태로 바꿔주는 기능이라고 볼 수 있음

- 직렬화의 주요 개념
    - 객체의 변환
        - 직렬화는 객체를 바이트 스트림으로 변환하여 파일에 저장하거나 네트워크를 통해 전송할 수 있게 함
        - 이를 통해 객체의 상태를 유지하면서 다른 환경에서도 동일한 객체를 사용할 수 있음
    - 역직렬화(Deserialization)
        - 직렬화된 데이터를 다시 객체로 변환하는 과정
        - 이를 통해 저장된 객체를 다시 메모리에 올릴 수 있음
        - 클라이언트가 보낸 데이터는 파이썬에서 바로 저장하거나 활용할 수 없으므로 다시 파이썬에서 사용하는 형태로 바꿔주는 기능이라고 할 수 있음

- 직렬화의 사용 예시
    - 객체의 영속화
        - 프로그램이 종료되더라도 객체의 상태를 유지하기 위해 파일에 저장할 수 있음
        - 예: 사용자 설정 정보를 직렬화하여 저장하면 프로그램 재실행 시에도 동일한 설정을 사용할 수 있음
    - 네트워크 통신
        - 직렬화된 객체를 네트워크를 통해 다른 시스템으로 전송할 수 있으며 이는 분산 시스템에서 중요한 역할을 함
        - 예: 클라이언트-서버 통신에서 객체를 직렬화하여 전송하고, 수신 측에서 역직렬화하여 객체를 복원할 수 있음

    ```python
    #//file: "./drf/serializers.py"

    from rest_framework import serializers
    from .models import Book

    class BookSerializer(serializers.ModelSerializer):
        class Meta:
            model = Book
            fields = ['bid', 'title', 'author', 'category', 'pages', 'price', 'published_date', 'description',]
    ```

- 일반적인 경우의 Serializer는
    - 파이썬 모델 데이터를 JSON으로 바꿔주는 변환기 이므로 모델 데이터의 어떤 속성을 JSON으로 넣어줄지 선언해야 함
    - 따라서 Serializer에도 필드를 선언해야 하므로 다음과 같이 복잡해지기 쉬움
    - 그래서 serializers.ModelSerializer를 사용하여 사용할 모델만을 기반으로 정의하는 방법을 사용하였음

    ```python
    #//file: "./drf/serializers.py"

    # 일반적인 경우 Serializer는 다음과 같이 복잡해짐
    class BookSerializer(serializers.Serializer):
        bid = serializers.IntegerField()
        title = serializers.CharField(max_length=50)
        author = serializers.CharField(max_length=50)
        category = serializers.CharField(max_length=50)
        pages = serializers.IntegerField()
        price = serializers.IntegerField()
        published_date = serializers.DateField()
        description = serializers.TextField()

        def create(self, validated_data):
            return Book.objects.create(**validated_data)

        def update(self, instance, validated_data):
            instance.bid = validated_data.get('bid', instance.bid)
            instance.title = validated_data.get('title', instance.title)
            instance.author = validated_data.get('author', instance.author)
            instance.category = validated_data.get('category', instance.category)
            instance.pages = validated_data.get('pages', instance.pages)
            instance.price = validated_data.get('price', instance.price)
            instance.published_date = validated_data.get('published_date', instance.published_date)
            instance.description = validated_data.get('description', instance.description)
            instance.save()

            return instance
    ```

### 1.3 DRF FBV, CBV, APIView

- FBV(Function Based View, 함수 기반 뷰), CBV(Class Based View, 클래스 기반 뷰)
    - 뷰를 작성할 때 함수를 사용했는지, 클래스를 사용했는지의 차이

- APIView
    - 여러 가지 유형의 요청에 대하여 FBV, CBV가 제대로 동작할 수 있도록 도와주는 역할
    - FBV에서 뷰를 만들 때
        - @api_view와 같이 데코레이터 형태로 생성
    - CBV에서 뷰를 만들 때
        - APIView를 상속받는 클래스의 형태로 생성

    ```python
    #//file: "./drf/views.py"

    from rest_framework.response import Response
    from rest_framework.views import APIView
    from rest_framework.decorators import api_view

    @api_view(['GET'])
    def helloAPI(request):
        return Response("hello world!")

    class HelloAPI(APIView):
        def get(self, request, format=None):
            return Response("hello world")
    ```

- 데코레이터(Decorator)
    - 개념
        - 프로그래밍에서 데코레이터는 함수나 메소드의 동작을 수정하거나 확장하는 데 사용되는 특별한 함수를 의미함
        - 데코레이터는 다른 함수를 인자로 받아, 그 함수에 새로운 기능을 추가한 후, 수정된 함수를 반환하며, 이를 통해 코드의 재사용성을 높이고, 공통적인 패턴을 캡슐화하여 다양한 함수나 메소드에 적용할 수 있음
        - 데코레이터는 특히 로깅, 권한 검사, 성능 측정 등 다양한 상황에서 유용하게 사용될 수 있음

    - 데코레이터의 주요 역할
        - 코드 재사용성 향상: 공통 기능을 데코레이터로 정의하여 여러 함수에 쉽게 적용할 수 있음
        - 코드 가독성 향상: 함수의 핵심 로직과 부가 기능을 분리하여 코드가 더 깔끔하고 이해하기 쉬워짐
        - 유지보수 용이성: 공통 기능을 한 곳에서 관리할 수 있어, 수정이 필요할 때 여러 곳을 수정할 필요가 없음

    - 데코레이터의 예시

        ```python
        def my_decorator(func):
            def wrapper():
                print("함수 호출 전")
                func()
                print("함수 호출 후")
            return wrapper

        @my_decorator
        def say_hello():
            print("안녕하세요!")

        say_hello()
        ```
        - 위 코드에서 `my_decorator`는 `say_hello` 함수에 적용되어, `say_hello` 함수 호출 전후에 메시지를 출력하는 기능을 추가함


### 1.4 Book API 만들기

- View 만들기

    ```python
    #//file: "./drf/views.py"

    from rest_framework import status
    from rest_framework.response import Response
    from rest_framework.views import APIView
    from rest_framework.decorators import api_view
    from rest_framework.generics import get_object_or_404
    from .models import Book
    from .serializers import BookSerializer

    @api_view(['GET', 'POST'])
    def booksAPI(request):
        if request.method == 'GET':
            books = Book.objects.all()
            serializer = BookSerializer(books, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        elif request.method == 'POST':
            serializer = BookSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @api_view(['GET'])
    def bookAPI(request, bid):
        book = get_object_or_404(Book, bid=bid)
        serializer = BookSerializer(book)
        return Response(serializer.data, status=status.HTTP_200_OK)


    class BooksAPI(APIView):
        def get(self, request):
            books = Book.objects.all()
            serializer = BookSerializer(books, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        def post(self, request):
            serializer = BookSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    class BookAPI(APIView):
        def get(self, request, bid):
            book = get_object_or_404(Book, bid=bid)
            serializer = BookSerializer(book)
            return Response(serializer.data, status=status.HTTP_200_OK)
    ```

- URL 연결

    ```python
    #//file: "./drf/urls.py"

    from django.urls import path
    from .views import helloAPI, bookAPI, booksAPI, BookAPI, BooksAPI, BooksAPIMixins, BookAPIMixins

    urlpatterns = [
        path("hello/", helloAPI),
        path("fbv/books/", booksAPI),
        path("fbv/book/<int:bid>/", bookAPI),
        path("cbv/books/", BooksAPI.as_view()),
        path("cbv/book/<int:bid>/", BookAPI.as_view()),
    ]
    ```

- 데이터 확인하기

    ```json
    {
        "bid": 1,
        "title": "처음 만나는 AI 수학 with 파이썬",
        "author": "아즈마 유키나가",
        "category": "프로그래밍",
        "pages": 308,
        "price": 20000,
        "published_date": "2021-01-30",
        "description": "인공지능을 공부하는데 필요한 기초 수학개념을 한 권에 모았다!"
    }

    {
        "bid": 2,
        "title": "앱 인벤터, 상상을 현실로 만드는 프로젝트",
        "author": "이준혁",
        "category": "프로그래밍",
        "pages": 328,
        "price": 20000,
        "published_date": "2020-12-10",
        "description": "블록 코딩으로 만드는 안드로이드 앱, 앱 인벤터"
    }
    ```

## 2. 도구를 이용한 API 테스트

- DRF API 테스트 방법
    - DRF 기본 페이지로 테스트
    - REST API 테스트 도구 활용: Postman, Insomnia 등
        - 사용 방법은 대동소이하므로 맘에 드는 도구를 선택하면 됨

- Insomnia 활용 예시
    - 다운로드: [Insomnia 공식 사이트](https://insomnia.rest/download)
    - 실행 화면
    <p align="center"><img src="/materials/images/python/S01-04-02-03_02-001.png" width="800"></p>


## 3. DRF 심화

### 3.1 DRF Mixins 적용해 보기

- **views.py의 중복 코드**
    - 하나의 클래스 내에서도 books, book과 같은 모델로부터 가져온 데이터나 BookSerializer와 같은 시리얼라이저가 여러번 사용되고 있음

        ```python
        #//file: "./drf/views.py"

        class BooksAPI(APIView):
            def get(self, request):
                books = Book.objects.all()
                serializer = BookSerializer(books, many=True)
                return Response(serializer.data, status=status.HTTP_200_OK)
            def post(self, request):
                serializer = BookSerializer(data=request.data)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        class BookAPI(APIView):
            def get(self, request, bid):
                book = get_object_or_404(Book, bid=bid)
                serializer = BookSerializer(book)
                return Response(serializer.data, status=status.HTTP_200_OK)
        ```

- DRF에서 제공하는 **Mixins**를 활용하여 중복을 제거할 수 있음
    - 기능
        - 도서 전체 목록 가져오기 (GET /books/) : List
        - 도서 1권 정보 등록하기 (POST /books/) : Create
        - 도서 1권 정보 가져오기 (GET /books/{bid}/) : Retrieve
        - 도서 1권 정보 수정하기 (PUT /books/{bid}/) : Update
        - 도서 1권 정보 삭제하기 (DELETE /books/{bid}/) : Destroy

            ```python
            #//file: "./drf/views.py"

            from rest_framework import generics, mixins

            class BooksAPIMixins(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
                queryset = Book.objects.all()
                serializer_class = BookSerializer

                def get(self, request, *args, **kwargs):            # GET 메소드 처리 함수(전체 목록)
                    return self.list(request, *args, **kwargs)      # mixins.ListModeMixin과 연결

                def post(self, request, *args, **kwargs):           # POST 메소드 처리 함수(1권 등록)
                    return self.create(request, *args, **kwargs)    # mixins.CreateModeMixin과 연결

            class BookAPIMixins(mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin, generics.GenericAPIView):
                queryset = Book.objects.all()
                serializer_class = BookSerializer
                lookup_field = 'bid'

                def get(self, request, *args, **kwargs):            # GET 메소드 처리 함수(1권 가져오기)
                    return self.retrieve(request, *args, **kwargs)  # mixins.RetrieveModelMixin과 연결

                def put(self, request, *args, **kwargs):            # PUT 메소드 처리 함수(1권 수정)
                    return self.update(request, *args, **kwargs)    # mixins.UpdateModelMixin과 연결

                def delete(self, request, *args, **kwargs):         # DELETE 메소드 처리 함수(1권 삭제)
                    return self.destroy(request, *args, **kwargs)   # mixins.DestroyModelMixin과 연결
            ```

- **URL 연결**

    ```python
    #//file: "./drf/urls.py"

    from django.urls import path, include
    from .views import helloAPI, bookAPI, booksAPI, BookAPI, BooksAPI, BooksAPIMixins, BookAPIMixins

    urlpatterns = [
        path("hello/", helloAPI),
        path("fbv/books/", booksAPI),
        path("fbv/book/<int:bid>/", bookAPI),
        path("cbv/books/", BooksAPI.as_view()),
        path("cbv/book/<int:bid>/", BookAPI.as_view()),
        path("mixin/books/", BooksAPIMixins.as_view()),
        path("mixin/book/<int:bid>/", BookAPIMixins.as_view()),
    ]
    ```

- **기능 확인**
    - http://127.0.0.1:8000/drf/mixin/books/
    - http://127.0.0.1:8000/drf/mixin/book/1/

### 3.2 DRF generics 적용해 보기

- **generics란?**
    - DRF에서 일반적인 CRUD 작업을 위한 API 뷰를 최소한의 코드로 구현할 수 있도록 도와주는 강력한 추상화 계층
    - 일반적인 웹 API 패턴을 빠르고 효율적으로 구현할 수 있도록 미리 정의된 클래스 기반 뷰(Class-Based Views, CBV)들의 집합
    - 반복적인 코드를 줄이고 생산성을 높이는 데 크게 기여함

- **generics 내용**
    - 반복 감소 (DRY 원칙)
        - 웹 API를 만들다 보면 목록 조회, 상세 조회, 생성, 수정, 삭제(CRUD)와 같은 작업들이 매우 빈번하게 발생
        - 이러한 일반적인 작업들은 공통적인 로직을 가지고 있음
        - `generics`는 이러한 공통 로직을 추상화하여 개발자가 매번 동일한 코드를 작성할 필요 없이 쉽게 재사용할 수 있도록 지원

    - `GenericAPIView`를 기반으로 함
        - 모든 `generics` 뷰는 `GenericAPIView`라는 기본 클래스를 상속
        - `GenericAPIView`: 다음과 같은 기본적인 기능을 제공
            - `queryset`: 어떤 모델의 데이터를 다룰지 정의
            - `serializer_class`: 데이터를 어떻게 직렬화/역직렬화할지 정의
            - `lookup_field`: 객체를 조회할 때 사용할 필드 (기본값은 `pk` 또는 `id`)를 정의
            - `get_queryset()`, `get_object()`, `get_serializer()` 등의 메서드를 통해 쿼리셋, 객체, 시리얼라이저를 유연하게 제어할 수 있게 함
            - 인증(Authentication), 권한(Permission), 스로틀링(Throttling), 필터링(Filtering), 페이지네이션(Pagination) 등 DRF의 핵심 기능들을 사용할 수 있는 기반 제공

    - Mixin과의 조합
        - `generics` 뷰들은 `mixins`와 `GenericAPIView`를 조합하여 특정 CRUD 작업을 수행하는 뷰를 생성함
            - 예
                - `ListModelMixin`: 목록을 조회하는 `list()` 액션 제공
                - `CreateModelMixin`: 객체를 생성하는 `create()` 액션 제공
                - `RetrieveModelMixin`: 특정 객체를 조회하는 `retrieve()` 액션 제공
                - `UpdateModelMixin`: 특정 객체를 수정하는 `update()` 액션 제공
                - `DestroyModelMixin`: 특정 객체를 삭제하는 `destroy()` 액션 제공

    - 다양한 "즉시 사용 가능한(plug-and-play)" 뷰 제공
        - DRF는 이러한 `mixins`와 `GenericAPIView`를 미리 조합하여 흔히 사용되는 형태의 뷰들을 제공함
            - `generics.ListAPIView`: 목록 조회 (`GET`)
            - `generics.CreateAPIView`: 객체 생성 (`POST`)
            - `generics.RetrieveAPIView`: 특정 객체 조회 (`GET` - 상세)
            - `generics.UpdateAPIView`: 특정 객체 수정 (`PUT`, `PATCH`)
            - `generics.DestroyAPIView`: 특정 객체 삭제 (`DELETE`)
            - `generics.ListCreateAPIView`: 목록 조회 및 객체 생성 (`GET`, `POST`)
            - `generics.RetrieveUpdateAPIView`: 특정 객체 조회, 수정 (`GET`, `PUT`, `PATCH`, `DELETE`)
            - `generics.RetrieveDestroyAPIView`: 특정 객체 조회, 삭제 (`GET`, `PUT`, `PATCH`, `DELETE`)
            - `generics.RetrieveUpdateDestroyAPIView`: 특정 객체 조회, 수정, 삭제 (`GET`, `PUT`, `PATCH`, `DELETE`)

- **코드 추가**

    ```python
    #//file: "./drf/views.py"

    class BooksAPIGenerics(generics.ListCreateAPIView):
        queryset = Book.objects.all()
        serializer_class = BookSerializer

    class BookAPIGenerics(generics.RetrieveUpdateDestroyAPIView):
        queryset = Book.objects.all()
        serializer_class = BookSerializer
        lookup_field = 'bid'
    ```

- **URL 연결**

    ```python
    #//file: "./drf/urls.py"

    from django.urls import path, include
    from .views import helloAPI, bookAPI, booksAPI, BookAPI, BooksAPI, BooksAPIMixins, BookAPIMixins

    urlpatterns = [
        ...
        path("mixin/books/", BooksAPIMixins.as_view()),
        path("mixin/book/<int:bid>/", BookAPIMixins.as_view()),
        path("generics/books/", BooksAPIGenerics.as_view()),
        path("generics/book/<int:bid>/", BookAPIGenerics.as_view()),
    ]
    ```

- **기능 확인**
    - http://127.0.0.1:8000/drf/generics/books/
    - http://127.0.0.1:8000/drf/generics/book/1/

### 3.3 DRF Viewset & Router 적용해 보기

- 지금까지의 방식
    - 하나의 클래스가 하나의 URL을 담당
        - URL마다 클래스 생성, 각 클래스는 해당 URL로 들어오는 다양한 메소드 처리
    - queryset, serializer_class 부분이 겹치는 현상 발생
        - 하나의 클래스로 하나의 모델을 모두 처리할 수 있다면 겹치는 부분이 사라질 것
            - Viewset을 이용하여 문제 해결 가능

- Viewset
    - View의 집합
        - 기본형태는 클래스형 뷰의 기본형태와 동일함
    - Router와 결합함으로써 강력한 성능 발휘가 가능함

- 적용하기
    - ModelViewSet

        ```python
        #//file: "./drf/views.py"

        from rest_framework import viewsets

        class BooksViewSet(viewsets.ModelViewSet):
            queryset = Book.objects.all()
            serializer_class = BookSerializer
        ```

        - 8줄이던 코드가 4줄로..
        - ModelViewSet을 가져와 클래스를 만들면 queryset과 serializer_class의 설정으로 모델에 대한 기본적인 REST API를 만들 수 있음
        - 기능은 기존의 기능을 모두 포함 

            ```python
            # DRF 내부의 ModelViewSet 코드
            class ModelViewSet(mixins.CreateModelMixin,
                                mixins.RetrieveModelMixin,
                                mixins.UpdateModelMixin,
                                mixins.DestroyModelMixin,
                                mixins.ListModelMixin,
                                GenericViewSet):
            """
            A viewset that provides default 'create()', 'retrieve()', 'update()',
            'partial_update()', 'destroy()', and 'list()' actions.
            """
            pass
            ```

            - ViewSet도 mixins를 사용함

        - url 연결

            ```python
            # 클래스형 뷰에서의 URL 설정
            urlpatterns = [
                path("cbv/books/", BooksAPI.as_view()),
                path("cbv/book/<int:bid>/", BookAPI.as_view()),
                path("mixin/books/", BooksAPIMixins.as_view()),
                path("mixin/book/<int:bid>/", BookAPIMixins.as_view()),
            ]
            ```

            ```python
            # Router를 사용한 URL 설정
            from rest_framework import routers
            from .views import BooksViewSet

            router = routers.SimpleRouter()

            router = DefaultRouter()
            router.register('books', BooksViewSet)

            urlpatterns = router.urls
            ```

- ViewSet과 Router 통합의 장점
    - 하나의 클래스로 하나의 모델에 대한 내용을 전부 작성할 수 있음
    - queryset이나 serializer_class 등 겹치는 부분의 최소화 가능
    - URL을 일일이 지정하지 않아도 라우터를 통해 일정한 규칙의 URL을 만들 수 있음