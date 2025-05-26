---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-02-05_04-DrfPystagramMemberRegister
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 글 관리 기능 구현

### 1. 모델 설정

#### 1.1  APP 생성

    ```bash
    python manage.py startapp posts
    ```

- config/settings.py에 추가

    ```python
    INSTALLED_APPS = [
        ...
        'posts',
    ]
    ```

#### 1.2 임시설정 변경

- templates/users/feeds.html 파일을 templates/posts/feeds.html 로 이동
- config/views.py 파일에서 users/feeds를 posts/feeds로 변경
- users/views.py 파일에서 users/feeds를 posts/feeds로 변경

- config/urls.py

    ```python
    urlpatterns = [
        path('admin/', admin.site.urls),
        path('', index),
        path("users/", include("users.urls")),
        path("posts/", include("posts.urls")),
    ]
    ```

- users/urls.py

    ```python
    from django.urls import path
    from users.views import login_view, logout_view, signup

    urlpatterns = [
        path('login/', login_view),
        path('logout/', logout_view),
        path("signup/", signup),
    ]
    ```

- posts/urls.py

    ```python
    from django.urls import path
    from users.views import feeds 

    urlpatterns = [
        path('feeds/', feeds),
    ]
    ```

#### 1.3 글/이미지/댓글 모델링

- Model 구성
    - posts/models.py

        ```python
        from django.db import models

        class Post(models.Model):
            user = models.ForeignKey("users.User", verbose_name="작성자", on_delete=models.CASCADE)
            content = models.TextField("내용")
            created = models.DateTimeField("생성일시", auto_now_add=True)

        class PostImage(models.Model):
            post = models.ForeignKey(Post, verbose_name="포스트", on_delete=models.CASCADE)
            photo = models.ImageField("사진", upload_to="post")

        class Comment(models.Model):
            user = models.ForeignKey("users.User", verbose_name="작성자", on_delete=models.CASCADE)
            post = models.ForeignKey(Post, verbose_name="포스트", on_delete=models.CASCADE)
            content = models.TextField("내용")
            created = models.DateTimeField("생성일시", auto_now_add=True)
        ```

    - 시스템에 반영

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

#### 1.4 관리자 페이지에 모델 등록

- posts/admin.py

    ```python
    from django.contrib import admin
    from posts.models import Post, PostImage, Comment

    @admin.register(Post)
    class PostAdmin(admin.ModelAdmin):
        list_display = [
            "id",
            "content",
        ]

    @admin.register(PostImage)
    class PostImageAdmin(admin.ModelAdmin):
        list_display = [
            "id",
            "post",
            "photo",
        ]

    @admin.register(Comment)
    class CommentAdmin(admin.ModelAdmin):
        list_display = [
            "id",
            "post",
            "content",
        ]
    ```

#### 1.5 admin에 연관 객체 표시

- ForeignKey로 연결된 객체 확인
    - posts/admin.py
    
        ```python
        class CommentInline(admin.TabularInline):
            model = Comment
            extra = 1

        @admin.register(Post)
        class PostAdmin(admin.ModelAdmin):
            ...
            inlines = [
                CommentInline,
            ]
        ```

        ```python
        class CommentInline(admin.TabularInline):
            ...

        class PostImageInline(admin.TabularInline):
            model = PostImage
            extra = 1

        @admin.register(Post)
        class PostAdmin(admin.ModelAdmin):
            ...
            inlines = [
                CommentInline,
                PostImageInline,
            ]
        ```

### 2. 썸네일 이미지 표시

#### 2.1 직접 admin을 조작해서 썸네일 표시 코드 추가

- posts/admin.py

    ```python
    from django.contrib.admin.widgets import AdminFileWidget
    from django.db import models
    from django.utils.safestring import mark_safe
    ...

    class CommentInline(admin.TabularInline):
        ...

    # AdminFileWidget은 관리자 페이지에서 '파일 선택' 버튼을 보여주는 부분
    # 이 widget을 커스텀하여 <img> 태그를 추가함
    class InlineImageWidget(AdminFileWidget):
        def render(self, name, value, attrs=None, renderer=None):
            html = super().render(name, value, attrs, renderer)
            if value and getattr(value, "url", None):
                html = mark_safe(f'<img src="{value.url}" width="150" height="150">') + html
            return html

    # ImageField를 표시할 때, AdminFileWidget을 커스텀한 InlineImageWidget을 사용함
    class PostImageInline(admin.TabularInline):
        model = PostImage
        extra = 1
        formfield_overrides = {
            models.ImageField: {
                "widget": InlineImageWidget,
            }
        }
    ```

#### 2.2 오픈소스 라이브러리를 사용한 썸네일 표시

```bash
pip install django-admin-thumbnails
```

```python
# 위에서 추가한 코드들은 모두 삭제하고 썸네일 라이브러리를 사용함
import admin_thumbnails

@admin_thumbnails.thumbnail("photo")
class PostImageInline(admin.TabularInline):
    model = PostImage
    extra = 1
```

<br>

### 3. 피드 페이지

#### 3.1 View 작성
- posts/views.py

    ```python
    from posts.models import Post

    def feeds(request):
        user = request.user
        if not user.is_authenticated:
            return redirect("/users/login/")

        posts = Post.objects.all()
        context = { "posts": posts }
        return render(request, "posts/feeds.html", context)
    ```

#### 3.2 작성자 정보 표시
- templates/posts/feeds.html

    ```html
    {% raw %}
    {% extends 'base_slider.html' %}
    {% block content %}
        <nav>
            <h1>Pystagram</h1>
        </nav>
        <div id="feeds" class="post-container">
            {% for post in posts %}
                <article class="post">
                    <header class="post-header">
                        {% if post.user.profile_image %}
                            <img src="{{ post.user.profile_image.url }}">
                        {% endif %}
                        <span>{{ post.user.username }}</span>
                    </header>
                </article>
            {% endfor %}
        </div>
    {% endblock %}
    {% endraw %}
    ```

#### 3.3 이미지 슬라이더 구현
- 이미지 슬라이드 자바스크립트, CSS 파일 불러오기
    - templates/base.html

        ```html
        {% raw %}
        {% load static %}
        <!doctype html>
        <html lang="ko">
        <head>
            <link rel="stylesheet" href="{% static 'css/style.css' %}">
            <link rel="stylesheet" href="{% static 'splide/splide.css' %}">
            <script src="{% static 'splide/splide.js' %}"></script>
        </head>
        <body>
        ...
        {% endraw %}
        ```

- Splide 라이브러리 사용
    - templates/posts/feeds.html

        ```html
        {% raw %}
        {% extends 'base.html' %}
        {% block content %}
        ...
            <div id="feeds" class="post-container">
                {% for post in posts %}
                    <article class="post">
                        <header class="post-header">
                            ...
                        </header>

                        <!-- 이미지 슬라이드 영역 시작 -->
                        <div class="post-images splide">
                            <div class="splide__track">
                                <ul class="splide__list">
                                    {% for image in post.postimage_set.all %}
                                        {% if image.photo %}
                                            <li class="splide__slide">
                                                <img src="{{ image.photo.url }}">
                                            </li>
                                        {% endif %}
                                    {% endfor %}
                                </ul>
                            </div>
                        </div>
                        <!-- 이미지 슬라이드 영역 종료 -->
                    </article>
                {% endfor %}
            </div>
        {% endblock %}
        {% endraw %}
        ```

#### 3.4 템플릿 하단에 자바스크립트 코드 작성
- templates/posts/feeds.html

    ```html
    {% raw %}
    {% block content %}
        <div id="feeds" class="post-container">
            ...
        </div>
        <!-- content 블록의 최하단에 작성함 -->
        <script>
            const elms = document.getElementsByClassName('splide')
            for (let i = 0; i < elms.length; i++){
                new Splide(elms[i]).mount();
            }
        </script>
    {% endblock %}
    {% endraw %}
    ```

#### 3.5 글 속성 출력
- templates/posts/feeds.html
    - 글 내용 출력

        ```html
        <article class="post">
            <header class="post-header">...</header>
            <div class="post-images">...</div>
            <div class="post-content">
                {{ post.content|linebreaksbr }}
            </div>
        ```

    - 좋아요/댓글 버튼 표시

        ```html
        <div class="post-content">...</div>
        <div class="post-buttons">
            <button>Likes(0)</button>
            <span>Comments(0)</span>
        </div>
        ```

    - 댓글 목록 표시

        ```html
        {% raw %}
        <div class="post-buttons">...</div>
        <div class="post-comments">
            <ul>
                <1-- 각 Post에 연결된 PostComment들을 순회 -->
                {% for comment in post.comment_set.all %}
                    <li>
                        <span>{{ comment.user.username }}</span>
                        <span>{{ comment.content }}</span>
                    </li>
                {% endfor %}
            </ul>
            <button>Likes(0)</button>
            <span>Comments(0)</span>
        </div>
        {% endraw %}
        ```

    - 작성일자, 댓글 입력창 표시

        ```html
        <div class="post-comments">...</div>
        <small>{{ post.created }}</small>
        <div class="post-comments-create">
            <input type="text" placeholder="댓글 달기...">
            <button type="submit">게시</button>
        </div>
        ```

#### 3.6 Template에 링크 추가
- templates/posts/feeds.html
    - 메인 링크 추가

        ```html
        <nav>
            <h1>
                <a href="/posts/feeds/">Pystagram</a>
            </h1>
        </nav>
        ```

    - 로그아웃 버튼 추가

        ```html
        <nav>
            <h1>
                <a href="/posts/feeds/">Pystagram</a>
            </h1>
            <a href="/users/logout/">Logout</a>
        </nav>
        ```

<br>

### 4. 글과 댓글

#### 4.1 댓글 작성
- CommentForm 구현
    - posts/forms.py

        ```python
        from django import forms
        from posts.models import Comment

        class CommentForm(forms.ModelForm):
            class Meta:
                model = Comment
                fields = [
                    "content",
                ]
        ```

- 오류 발생
    - posts_comment 테이블의 post_id 필드는 NULL을 허용하지 않는다는 메시지
        - Terminal

            ```bash
            python manage.py shell
            ```

            ```python
            from post.forms import CommentForm

            data = {"content": "SampleContent"}
            form = CommentForm(data=data)
            form.is_valid()
            form.save()
            ```

- 오류 해결 방법
    - CommentForm으로 Comment 객체를 일단 만들되, 메모리 상에 객체를 만들고 필요한 데이터를 나중에 채우기
    - CommentForm에 NULL을 허용하지 않는 모든 필드를 선언하고 인스턴스 생성 시 유효한 데이터를 전달

    - 첫 번째 방법으로 해결해보기

        ```bash
        python manage.py shell
        ```

        ```python
        from post.forms import CommentForm

        data = {"content": "SampleContent"}
        form = CommentForm(data=data)
        form.is_valid()
        comment = form.save(commit=False)
        print(comment.id)

        from users.models import User
        from posts.models import Post

        user = User.objects.all()[0]
        post = Post.objects.all()[0]
        print(user)
        print(post)

        comment.user = user
        comment.post = post
        comment.save()

        comment.id
        ```

    - 두 번째 방법으로 해결해보기
        - posts/forms.py

            ```python
            class CommentForm(forms.ModelForm):
                class Meta:
                    model = Comment
                    fields = [
                        "user",
                        "post",
                        "content",
                    ]
            ```

        - Terminal

        ```bash
        python manage.py shell
        ```

        ```python
        python manage.py shell

        from post.forms import CommentForm

        data = {"content": "SampleContent"}
        form = CommentForm(data=data)
        form.is_valid()
        form.errors

        from users.models import User
        from posts.models import Post

        user = User.objects.all()[0]
        post = Post.objects.all()[0]
        data = {"content": "SampleContent", "user": user, "post": post}
        form = CommentForm(data=data)
        form.is_valid()

        comment = form.save()
        comment.id
        ```

- Comment를 생성하기 위해 필요한 데이터
    - 어떤 글(Post)의 댓글인지
    - 어떤 사용자(User)의 댓글인지
    - 어떤 내용(Comment)을 가지고 있는지

- View에서 Template으로 Form 전달
    - posts/views.py

        ```python
        from posts.forms import CommentForm

        def feeds(request):
            ...

            posts = Post.objects.all()
            comment_form = CommentForm()
            context = {
                "posts": posts,
                "comment_form": comment_form,
            }
            return render(request, "posts/feeds.html", context)
        ```

    - templates/posts/feeds.html
        - 직접 작성했던 input 요소를 삭제하고 comment_form.as_p 변수 사용

            ```html
            {% raw %}
            <div class="post-comments-create">
                <form method="POST">
                    {% csrf_token %}
                    {{ comment_form.as_p }}
                    <button type="submit">게시</button>
                </form>
            </div>
            {% endraw %}
            ```

        - CommentForm에는 post, content 필드가 있고 이 둘을 as_p로 렌더링한 경우
            - 포스트의 드롭다운 요소를 클릭하면 Post 객체를 선택할 수 있음. 사용자가 어떤 글에 댓글을 다는지는 직접 입력할 필요 없이 템플릿에서 알아서 처리해 주어야 함
            - 자동으로 < label>요소와 < input>요소가 만들어짐. 여기서는 "내용:"으로 나타나는 < label>요소가 필요하지 않음. content 값을 입력받을 < input> 요소만 있으면 됨

            ```html
            {% raw %}
            <div class="post-comments-create">
                <form method="POST">
                    {% csrf_token %}
                    {{ comment_form.content }}
                    <button type="submit">게시</button>
                </form>
            </div>
            {% endraw %}
            ```

    - posts/forms.py

        ```python
        class CommentForm(forms.ModelForm):
            class Meta:
                model = Comment
                fields = [ ... ]
                widgets = {
                    "content": forms.Textarea(
                        attrs={
                            "placeholder": "댓글 달기...",
                        }
                    )
                }
        ```

<br>

#### 4.2 댓글 작성 처리를 위한 View 구현
- posts/views.py

    ```python
    from django.views.decorators.http import require_POST

    def feeds(request):
        ...

    @require_POST
    def comment_add(request):
        print(request.POST)
    ```

- posts/urls.py

    ```python
    from django.urls import path
    from posts.views import feeds, comment_add
    ...

    urlpatterns = [
        path("feeds/", feeds),
        path("comment_add/", comment_add),
    ]
    ```

<br>

#### 4.3 form에서 comment_add View로 데이터 전달 및 처리

- form의 action 속성
    - method: GET과 POST 중 어떤 방식으로 데이터를 전달할지
    - enctype: 기본값(application/x-www-form-urlencoded)과 파일 전송을 위한 값(multipart/form-data) 중 선택

- 사용자가 직접 입력하지 않는 고정된 데이터를 form 내부에 위치
    - templates/posts/feeds.html

        ```html
        {% raw %}
        <div class="post-comments-create">
            <form method="POST" action="/posts/comment_add/">
                {% csrf_token %}
                <input type="hidden" name="post" value="{{ post.id }}">
                {{ comment_form.content }}
                <button type="submit">게시</button>
            </form>
        </div>
        {% endraw %}
        ```

- 사용자 정보를 View에서 직접 할당
    - posts/views.py

        ```python
        @require_POST
        def comment_add(request):
            form = CommentForm(data=request.POST)
            if form.is_valid():
                comment = form.save(commit=False)
                comment.user = request.user
                comment.save()

                print(comment.id)
                print(comment.content)
                print(comment.user)

                return redirect{"/posts/feeds/"}
        ```

- 작성 완료 후 원하는 Post 위치로 이동
    - templates/posts/feeds.html

        ```html
        {% raw %}
        <div id="feeds" class="post-container">
            {% for post in posts %}
                <article id="post-{{ post.id }}" class="post">
        {% endraw %}
        ```

    - posts/views.py

        ```python
        from django.http import HttpResponseRedirect

        @require_POST
        def comment_add(request):
            form = CommentForm(data=request.POST)
            if form.is_valid():
                comment = form.save(commit=False)
                ...

                return HttpResponseRedirect{f"/posts/feeds/#post-{comment.post.id}"}
        ```

<br>

#### 4.4 글의 댓글 수 표시

- Terminal에서 확인

    ```bash
    python manage.py shell
    ```

    ```python
    from posts.models import Post

    for post in Post.objects.all():
        print(f"id: {post.id}, comment_count: {post.comment_set.count()}")
    ```

- templates/posts/feeds.html

    ```html
    <div class="post-buttons">
        <button type="submit">Likes(0)</button>
        <span>Comments({{ post.comment_set.count }})</span>
    </div>
    ```

<br>

#### 4.5 댓글 삭제

- posts/views.py

    ```python
    from posts.models import Post, Comment

    @require_POST
    def comment_add(request):
        ...

    @require_POST
    def comment_delete(request, comment_id):
        if request.method == "POST":
            comment = Comment.objects.get(id=comment_id)
            comment.delete()
            return HttpResponseRedirect(f"/posts/feeds/#post-{comment.post.id}")
    ```

- posts/urls.py

    ```python
    from django.urls import path
    from posts.views import feeds, comment_add, comment_delete
    ...

    urlpatterns = [
        path("feeds/", feeds),
        path("comment_add/", comment_add),
        path("comment_delete/<int:comment_id>/", comment_delete)
    ]
    ```

- 삭제할 Comment가 요청한 사용자가 작성한 것인지 확인
    - posts/views.py

        ```python
        from django.http import HttpResponseRedirect, HttpResponseForbidden

        @require_POST
        def comment_delete(request, comment_id):
            comment = Comment.objects.get(id=comment_id)
            if comment.user == request.user:
                comment.delete()
                return HttpResponseRedirect(f"/posts/feeds/#post-{comment.post.id}")
            else:
                return HttpResponseForbidden("이 댓글을 삭제할 권한이 없습니다.")
        ```

- 템플릿에 삭제 버튼 추가
    - templates/posts/feeds.html

        ```html
        {% raw %}
        <div class="post-comments">
            <ul>
                {% for comment in post.comment_set.all %}
                    <li>
                        <span>{{ comment.user.username }}</span>
                        <span>{{ comment.content }}</span>

                        <!-- 댓글 삭제 form 추가 -->
                        {% if user == comment.user %}
                            <form method="POST" action="/posts/comment_delete/{{ comment.id }}/">
                                {% csrf_token %}
                                <button type="submit">삭제</button>
                            </form>
                        {% endif %}
                    </li>
                {% endfor %}
            </ul>
        </div>
        {% endraw %}
        ```

<br>

### 5. 글 작성하기

#### 5.1 글 작성 기본 구조

- View: /posts/post_add/
- URL: post_add
- Template: templates/posts/post_add.html

<br>

#### 5.2 글 작성 기본 구조 구현

- posts/views.py

    ```python
    ...
    def post_add(request):
        return render(request, "posts/post_add.html")
    ```

- posts/urls.py

    ```python
    ...
    from posts.views import feeds, comment_add, comment_delete, post_add
    ...

    urlpatterns = [
        ...
        path("post_add/", post_add),
    ]
    ```

- templates/posts/post_add.html

    ```html
    {% raw %}
    {% externds 'base.html' %}

    {% block content %}
        <div id="post-add">
            <h1>Post Add</h1>
        </div>
    {% endblock %}
    {% endraw %}
    ```

<br>

#### 5.3 PostForm 클래스 구현

- posts/forms.py

    ```python
    from posts.models import Comment, Post

    class PostForm(forms.ModelForm):
        class Meta:
            model = Post
            fields = [
                "content",
            ]
    ```

<br>

#### 5.4 View 로직, Template 구현

- posts/views.py

    ```python
    from posts.forms import CommentForm, PostForm
    ...

    def post_add(request):
        form = PostForm()
        context = {"form": form}
        return render(request, "posts/post_add.html", context)
    ```

- templates/posts/post_add.html

    ```html
    {% raw %}
    {% externds 'base.html' %}

    {% block content %}
        <nav>
            <h1>Pystagram</h1>
        </nav>
        <div id="post-add">
            <h1>Post Add</h1>
            <form method="POST">
                {% csrf_token %}
                {{ form.as_p }}
                <button type="submit">게시</button>
            </form>
        </div>
    {% endblock %}
    {% endraw %}
    ```

<br>

#### 5.5 여러 장의 이미지 업로드

- Template에 직접 < input type="file"> 구성
    - templates/posts/post_add.html

        ```html
        {% raw %}
        <form method="POST" enctype="multipart/form-data">
            {% csrf_token %}
            <div>
                <labl for="id_images">이미지</label>
                <input id="id_images" name="images" type="file" multiple>
            </div>
            {{ form.as_p }}
            <button type="submit">게시</button>
        </form>
        {% endraw %}
        ```

- View에서 multiple 속성을 가진 file input의 데이터 받기
    - posts/views.py

        ```python
        from posts.models import Post, Comment, PostImage

        ...

        def post_add(request):
            if request.method == "POST":
                # request.POST로 온 데이터 ("content")는 PostForm으로 처리
                form = PostForm(request.POST)

                if form.is_valid():
                    # Post의 "user"값은 request에서 가져와 자동할당한다
                    post = form.save(commit=False)
                    post.user = request.user
                    post.save()

                    # Post를 생성 한 후
                    # request.FILES.getlist("images")로 전송된 이미지들을 순회하며 PostImage객체를 생성한다
                    for image_file in request.FILES.getlist("images"):
                        # request.FILES또는 request.FILES.getlist()로 가져온 파일은
                        # Model의 ImageField부분에 곧바로 할당한다
                        PostImage.objects.create(
                            post=post,
                            photo=image_file,
                        )

                    # 모든 PostImage와 Post의 생성이 완료되면
                    # 피드페이지로 이동하여 생성된 Post의 위치로 스크롤되도록 한다
                    url = reverse("posts:feeds") + f"#post-{post.id}"
                    return HttpResponseRedirect(url)

            # GET요청일 때는 빈 form을 보여주도록한다
            else:
                form = PostForm()

            context = {"form": form}
            return render(request, "posts/post_add.html", context)
        ```

#### 5.6 내비게이션 바에 링크 추가

- 피드 페이지에서 글 작성 페이지로의 작성 추가
    - templates/posts/feeds.html

        ```html
        <nav>
            <h1>
                <a href="/posts/feeds/">Pystagram</a>
            </h1>
            <a href="/posts/post_add/">Add post</a>
            <a href="/users/logout/">Logout</a>
        </nav>
        ```

- 글 작성 페이지에서 피드 페이지로 돌아오는 링크 추가
    - templates/posts/post_add.html

        ```html
        <nav>
            <h1>
                <a href="/posts/feeds/">Pystagram</a>
            </h1>
            <a href="/users/logout/">Logout</a>
        </nav>
        ```