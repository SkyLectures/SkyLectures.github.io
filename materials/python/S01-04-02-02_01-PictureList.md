---
layout: page
title:  "Django 구현 기초-사진목록 보기"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-02-02_01-PictureList
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 사진 목록 보기

### 1.1 사진 목록 화면 만들기

- Template

```html
<!--file: "photolist/templates/photo_list.html"-->

<html>
    <head>
        <title>Photo App</title>
    </head>
    <body>
        <h1><a href="">사진 목록 페이지</a></h1>
        <section>

            <div>
                <h2>
                    <a href="">title</a>
                </h2>
                <img src="" alt="" width="300" />
                <p>photo.author, photo.price원</p>
            </div>

            <div>
                <h2>
                    <a href="">title</a>
                </h2>
                <img src="" alt="" width="300" />
                <p>photo.author, photo.price원</p>
                </div>

            <div>
                <h2>
                    <a href="">title</a>
                </h2>
                <img src="" alt="" width="300" />
                <p>photo.author, photo.price원</p>
            </div>

        </section>
    </body>
</html>
```


- View

```python
#//file: "photolist/views.py"
from django.shortcuts import render

def photo_list(request):
    photos = Photo.objects.all()
    return render(request, 'photo/photo_list.html', {})
```

- URL

```python
#//file: "photolist/urls.py"

from django.urls import path
from . import views

urlpatterns = [
    path('', views.photo_list, name='photo_list')
]
```

```python
#//file: "photoweb/urls.py"

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('photo.urls')),
]
```

- View

```python
#//file: "photolist/views.py"
from django.shortcuts import render, get_object_or_404, redirect
from .models import Photo
from .forms import PhotoForm

def photo_list(request):
    photos = Photo.objects.all()
    return render(request, 'photo/photo_list.html', {'photos': photos})
```

- Template 수정

```html
<!--file: "photolist/templates/photo_list.html" -->

<html>
<head>
    <title>Photo App</title>
</head>
<body>
    <h1><a href="">사진 목록 페이지</a></h1>
    <section>
        <% for photo in photos %>
            <div>
                <h2>
                    <a href="">{ { photo.title }}</a>
                </h2>
                <img src="{ { photo.image }}" alt="{ { photo.title }}" width="300" />
                <p>{ { photo.author }}, { { photo.price }}원</p>
            </div>
        <% endfor %>
    </section>
</body>
</html>
```


## **2. 사진 게시물 보기 화면 만들기**

- Template

```html
<!--file: "photolist/templates/photo_detail.html"-->

<html>
    <head>
        <title>Photo App</title>
    </head>
    <body>
        <h1>{  { photo.title }}</h1>
        <section>
            <div>
                <img src="{ { photo.image }}" alt="{ { photo.title }}" width="300" />
                <p>{ { photo.description }}</p>
                <p>{ { photo.author }}, { { photo.price }}원</p>
            </div>
        </section>
    </body>
</html>
```

- View

  ```python
  #//file: "photolist/views.py"
  from django.shortcuts import render, get_object_or_404

  def photo_list(request):
      photos = Photo.objects.all()
      return render(request, 'photo/photo_list.html', {'photos': photos})


  def photo_detail(request, pk):
      photo = get_object_or_404(Photo, pk=pk)
      return render(request, 'photo/photo_detail.html', {'photo': photo})
  ```

- URL

  ```python
  #//file: "photolist/urls.py"
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.photo_list, name='photo_list'),
      path('photo/<int:pk>/', views.photo_detail, name='photo_detail'),
  ]
  ```

- Template

  ```html
  <!--file: "photolist/templates/photo_list.html"-->

  <html>
      <head>
          <title>Photo App</title>
      </head>
      <body>
          <h1><a href="">사진 목록 페이지</a></h1>
          <section>
              {% for photo in photos %}
              <div>
                  <h2>
                  <a href="{ % url 'photo_detail' pk=photo.pk %}">{ { photo.title }}</a>
                  </h2>
                  <img src="{ { photo.image }}" alt="{ { photo.title }}" width="300" />
                  <p>{ { photo.author }}, { { photo.price }}원</p>
              </div>
              {% endfor %}
          </section>
      </body>
  </html>
  ```

- 이미지 및 사용자 업로드 파일 등의 경로 문제
    - Django 프레임워크의 경우 타 프레임워크와 달리 이미지 및 사용자 업로드 파일의 경로 지정이 Settings.py의 설정에 고정되어 있음
    - Django 프레임워크에서는 이러한 파일들을 정적 파일(Static File)로 분류, 처리하고 있음
        - 정적 파일은 동적파일과 달리 웹 서비스 시에 데이터를 가공할 필요없이 서버에 저장된 그대로를 사용하는 것
    - Django 프레임워크에서 사용하는 정적 파일은 Static 파일과 Media 파일의 2종류로 분류함
        - Static
            - 개발자가 준비해 두는 파일
            - 개발을 위한 Resource로서 취급됨
            - 응답할 때 별도의 처리없이 파일의 내용을 그대로 보여줌
            - 파일 자체가 고정되어 있으며 서비스 중에도 추가되거아 변경되지 않음
        - Media
            - 사용자가 업로드하는 파일
            - 동적으로 변하지 않고 사용자가 업로드한 그대로 변화없이 보여주거나 사용하는 파일

- Static 파일의 사용
    - settings.py
        - INSTALLED_APPS에 django.contrib.staticfiles가 포함되어 있는지 확인
        - STATIC_URL 정의하기
            - 예: STATIC_URL = '/static/'
            - STATIC_URL은 프로젝트 시작 시 만든 startapp의 경로를 ROOT로 사용하고 있음
        - 필요 시 STATIC_ROOT 정의하기
            - Django 프로젝트에서 사용하는 모든 정적 파일을 한 곳에 모아 넣기 위한 경로
            - 실제 서비스를 위한 배포환경에서는 Django를 직접 실행하는 것이 아니라 다른 서버에 의해 실행되는 경우가 많으며, 이런 경우에는 실행하는 다른 서버가 Django 프로젝트 내부의 정적 파일을 인식하지 못함
            - 따라서 프로젝트의 바깥으로 정적 파일들을 꺼낼 필요가 있음
            -이런 경우에 STATIC_ROOT가 사용됨
    - 설정된 static 폴더에 정적파일 보관

```python
#//file: "photoweb/settings.py"

STATIC_URL = '/static/'
```

- Media 파일의 사용
    - settings.py
        - MEDIA_ROOT, MEDIA_URL 정의하기
        - MEDIA_ROOT
            - 사용자가 업로드한 파일들을 보관할 디렉토리의 절대경로
            - STATIC_ROOT와 반드시 다른 경로로 지정해야 함
        - MEDIA_URL
            - MEDIA_ROOT에서 제공되는 미디어 파일을 처리하는 URL
            - 업로드된 파일의 주소(URL)를 만들어주는 역할
    - urls.py
        - settings와 static을 import
        - urlpatterns에 static 함수 추가
            - 이때, 리스트에 추가하는 것이 아니라 리스트의 밖에 '+' 연산자를 통해 추가해야 함

```python
#//file: "photoweb/settings.py"

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

```python
#//file: "photoweb/urls.py"

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('photolist.urls'))
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```


## **3. 사진 게시물 작성 기능 만들기**

- Template

  ```html
  <!--file: "photolist/templates/photo_post.html"-->
    
  <html>
      <head>
          <title>Photo App</title>
      </head>
      <body>
          <h1><a href="/">홈으로 돌아가기</a></h1>
          <section>
              <div>
                  <h2>New Photo</h2>
                  <form method="POST">
                      { % csrf_token %} { { form.as_p }}
                      <button type="submit">완료!</button>
                  </form>
              </div>
          </section>
      </body>
  </html>
  ```


- Form

  ```python
  #//file: "photolist/forms.py"

  from django import forms
  from .models import Photo

  class PhotoForm(forms.ModelForm):
      class Meta:
          model = Photo
          fields = (
              'title',
              'author',
              'image',
              'description',
              'price',
          )
  ```

- View

  ```python
  #//file: "photolist/views.py"

  from django.shortcuts import render, get_object_or_404, redirect
  from .models import Photo
  from .forms import PhotoForm


  def photo_list(request):
      photos = Photo.objects.all()
      return render(request, 'photo/photo_list.html', {'photos': photos})


  def photo_detail(request, pk):
      photo = get_object_or_404(Photo, pk=pk)
      return render(request, 'photo/photo_detail.html', {'photo': photo})


  def photo_post(request):
      if request.method == "POST":
          form = PhotoForm(request.POST)
          if form.is_valid():
              photo = form.save(commit=False)
              photo.save()
              return redirect('photo_detail', pk=photo.pk)
      else:
          form = PhotoForm()
      return render(request, 'photo/photo_post.html', {'form': form})
  ```

- URL

  ```python
  #//file: "photolist/urls.py"

  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.photo_list, name='photo_list'),
      path('photo/<int:pk>/', views.photo_detail, name='photo_detail'),
      path('photo/new/', views.photo_post, name='photo_post'),
  ]
  ```

- Templates

  ```html
  <!--file: "photolist/templates/photo_list.html"-->

  <html>
      <head>
          <title>Photo App</title>
      </head>
      <body>
          <h1><a href="">사진 목록 페이지</a></h1>
          <h3><a href="{ % url 'photo_post' %}">New Photo</a></h3>
          <section>
              { % for photo in photos %}
              <div>
                  <h2>
                      <a href="{ % url 'photo_detail' pk=photo.pk %}">{ { photo.title }}</a>
                  </h2>
                  <img src="{ { photo.image }}" alt="{ { photo.title }}" width="300" />
                  <p>{ { photo.author }}, { { photo.price }}원</p>
              </div>
              { % endfor %}
          </section>
      </body>
  </html>
  ```


## **4. 사진 게시물 수정 기능 만들기**

- Template
    - 기존과 동일

- View

```python
  #//file: "photolist/views.py"

  from django.shortcuts import render, get_object_or_404, redirect
  from .models import Photo
  from .forms import PhotoForm


  def photo_list(request):
      photos = Photo.objects.all()
      return render(request, 'photo/photo_list.html', {'photos': photos})


  def photo_detail(request, pk):
      photo = get_object_or_404(Photo, pk=pk)
      return render(request, 'photo/photo_detail.html', {'photo': photo})


  def photo_post(request):
      if request.method == "POST":
          form = PhotoForm(request.POST)
          if form.is_valid():
              photo = form.save(commit=False)
              photo.save()
              return redirect('photo_detail', pk=photo.pk)
      else:
          form = PhotoForm()
      return render(request, 'photo/photo_post.html', {'form': form})


  def photo_edit(request, pk):
      photo = get_object_or_404(Photo, pk=pk)
      if request.method == "POST":
          form = PhotoForm(request.POST, instance=photo)
          if form.is_valid():
              photo = form.save(commit=False)
              photo.save()
              return redirect('photo_detail', pk=photo.pk)
      else:
          form = PhotoForm(instance=photo)
      return render(request, 'photo/photo_post.html', {'form': form})
  ```

- URL

```python
  #//file: "photolist/urls.py"
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.photo_list, name='photo_list'),
      path('photo/<int:pk>/', views.photo_detail, name='photo_detail'),
      path('photo/new/', views.photo_post, name='photo_post'),
      path('photo/<int:pk>/edit/', views.photo_edit, name='photo_edit'),
  ]
  ```

- Templates

    ```html
    <!--file: "photolist/templates/photo_detail.html"-->
    <html>
        <head>
            <title>Photo App</title>
        </head>
        <body>
            <h1>{ { photo.title }}</h1>
            <h3><a href="{ % url 'photo_edit' pk=photo.pk %}">Edit Photo</a></h3>
            <section>
                <div>
                    <img src="{ { photo.image }}" alt="{ { photo.title }}" width="300" />
                    <p>{ { photo.description }}</p>
                    <p>{ { photo.author }}, { { photo.price }}원</p>
                </div>
            </section>
        </body>
    </html>
    ```