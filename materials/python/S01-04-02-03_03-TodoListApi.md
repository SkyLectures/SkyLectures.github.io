---
layout: page
title:  "DRF: TodoList API"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-03_03-TodoListApi
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 프로젝트 설정

- 가상환경 생성

    ```bash
    python -m venv drftodo
    cd drftodo
    source ./bin/activate        # ./Scripts/activate
    ```

- DRF 프로젝트/앱 생성

    ```bash
    pip install django djangorestframework
    django-admin startproject config .
    python manage.py startapp todo
    ```

- 프로젝트 설정

    ```python
    #//file: "config/settings.py"

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'rest_framework',
        'todo',
    ]

    ...

    LANGUAGE_CODE = 'ko-kr'
    TIME_ZONE = 'Asia/Seoul'
    ```

- 관리자 계정 생성

    ```bash
    python manage.py createsuperuser
    ```

## 2. 모델 생성

```python
#//file: "todo/models.py"

from django.db import models

# Create your models here.
class Todo(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    complete = models.BooleanField(default=False)
    important = models.BooleanField(default=False)

    def __str__(self):
        return self.title
```

## 3. Todo 전체 조회 API 만들기

- 시리얼라이저 만들기

```python
#//file: "todo/serializer.py"

from rest_framework import serializers
from .models import Todo

class TodoSimpleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'title', 'complete', 'important')
```

- 뷰 만들기

```python
#//file: "todo/views.py"

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets

from .models import Todo
from .serializers import TodoSimpleSerializer

class TodosAPIView(APIView):
    def get(self, request):
        todos = Todo.objects.filter(complete=False)
        serializer = TodoSimpleSerializer(todos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
```

- URL 연결하기

```python
#//file: "todo/urls.py"

from django.urls import path
from .views import TodosAPIView

urlpatterns = [
    path('todo/', TodosAPIView.as_view()),
]
```

```python
#//file: "config/urls.py"

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('todo.urls')),
]
```

- API 테스트하기
    - 관리자모드에서 미리 데이터 입력
    - GET - http://127.0.0.1:8000/todo/

## 4. Todo 상세 조회 API 만들기

- 시리얼라이저 만들기

```python
#//file: "todo/serializer.py"

from rest_framework import serializers
from .models import Todo

class TodoSimpleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'title', 'complete', 'important')

class TodoDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'title', 'description', 'created', 'complete', 'important')
```

- 뷰 만들기

```python
#//file: "todo/views.py"

from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets

from .models import Todo
from .serializers import TodoSimpleSerializer, TodoDetailSerializer

class TodosAPIView(APIView):
    def get(self, request):
        todos = Todo.objects.filter(complete=False)
        serializer = TodoSimpleSerializer(todos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class TodoAPIView(APIView):
    def get(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoDetailSerializer(todo)
        return Response(serializer.data, status=status.HTTP_200_OK)
```

- URL 연결하기

```python
#//file: "todo/urls.py"

from django.urls import path
from .views import TodoAPIView, TodosAPIView

urlpatterns = [
    path('todo/', TodosAPIView.as_view()),
    path('todo/<int:pk>/', TodoAPIView.as_view()),
]
```

- API 테스트하기
    - GET - http://127.0.0.1:8000/todo/2/


## 5. Todo 생성 API 만들기

- 시리얼라이저 만들기

```python
#//file: "todo/serializer.py"

from rest_framework import serializers
from .models import Todo

class TodoSimpleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'title', 'complete', 'important')

class TodoDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('id', 'title', 'description', 'created', 'complete', 'important')

class TodoCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Todo
        fields = ('title', 'description')
```

- 뷰 만들기

```python
#//file: "todo/views.py"

from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets

from .models import Todo
from .serializers import TodoSimpleSerializer, TodoDetailSerializer, TodoCreateSerializer

class TodosAPIView(APIView):
    def get(self, request):
        todos = Todo.objects.filter(complete=False)
        serializer = TodoSimpleSerializer(todos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = TodoCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TodoAPIView(APIView):
    def get(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoDetailSerializer(todo)
        return Response(serializer.data, status=status.HTTP_200_OK)
```

- URL 연결하기

```python
#//file: "todo/urls.py"

from django.urls import path
from .views import TodoAPIView, TodosAPIView

urlpatterns = [
    path('todo/', TodosAPIView.as_view()),
    path('todo/<int:pk>/', TodoAPIView.as_view()),
]
```

- API 테스트하기
    - POST - http://127.0.0.1:8000/todo/
    - 특정 Todo의 id가 필요하지 않은 작업
    - body에는 JSON 형태로 값 입력

## 6. Todo 수정 API 만들기

- 뷰 만들기

```python
#//file: "todo/views.py"

from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets

from .models import Todo
from .serializers import TodoSimpleSerializer, TodoDetailSerializer, TodoCreateSerializer

class TodosAPIView(APIView):
    def get(self, request):
        todos = Todo.objects.filter(complete=False)
        serializer = TodoSimpleSerializer(todos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = TodoCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TodoAPIView(APIView):
    def get(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoDetailSerializer(todo)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoCreateSerializer(todo, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

- URL 연결하기

```python
#//file: "todo/urls.py"

from django.urls import path
from .views import TodoAPIView, TodosAPIView

urlpatterns = [
    path('todo/', TodosAPIView.as_view()),
    path('todo/<int:pk>/', TodoAPIView.as_view()),
]
```

- API 테스트하기
    - PUT - http://127.0.0.1:8000/todo/2/



## 7. Todo 완료 API 만들기

- 뷰 만들기

```python
#//file: "todo/views.py"

from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets

from .models import Todo
from .serializers import TodoSimpleSerializer, TodoDetailSerializer, TodoCreateSerializer

class TodosAPIView(APIView):
    def get(self, request):
        todos = Todo.objects.filter(complete=False)
        serializer = TodoSimpleSerializer(todos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = TodoCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TodoAPIView(APIView):
    def get(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoDetailSerializer(todo)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        todo = get_object_or_404(Todo, id=pk)
        serializer = TodoCreateSerializer(todo, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 완료목록 조회용
class DoneTodosAPIView(APIView):
    def get(self, request):
        dones = Todo.objects.filter(complete=True)
        serializer = TodoSimpleSerializer(dones, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

# 완료 처리용
class DoneTodoAPIView(APIView):
    def get(self, request, pk):
        done = get_object_or_404(Todo, id=pk)
        done.complete = True
        done.save()
        serializer = TodoDetailSerializer(done)
        return Response(status=status.HTTP_200_OK)
```

- URL 연결하기

```python
#//file: "todo/urls.py"

from django.urls import path
from .views import TodoAPIView, TodosAPIView, DoneTodoAPIView, DoneTodosAPIView

urlpatterns = [
    path('todo/', TodosAPIView.as_view()),
    path('todo/<int:pk>/', TodoAPIView.as_view()),
    path('done/', DoneTodosAPIView.as_view()),
    path('done/<int:pk>/', DoneTodoAPIView.as_view()),
]
```

- API 테스트하기
    - GET - http://127.0.0.1:8000/done/
    - GET - http://127.0.0.1:8000/done/2/
