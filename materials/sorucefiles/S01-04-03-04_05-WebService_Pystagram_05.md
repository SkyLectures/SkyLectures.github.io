---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-03-04_05-WebService_Pystagram_05
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 부가 기능 구현

### **1. 동적 URL**

#### 1.1 URL 경로 변경

- URL 경로를 변경할 때 생기는 중복작업
    - users/urls.py

        ```python
        urlpatterns = [
            ...
            path("login2/", login_view),
            ...
        ]
        ```

    - templates/users/signup.html

        ```html
        {% raw %}
        <div id="signup">
            <form method="POST" enctype="multipart/form-data">
                ...
                <a href="{% url '/users/login2/' %}">로그인 페이지로 이동</a>
            </form>
        </div>
        {% endraw %}
        ```

    - posts/views.py

        ```python
        def feeds(request):
            ...
            if not request.user.is_authenticated:
                return redirect("/users/login2/")
        ```

    - users/views.py

        ```python
        def logout_view(request):
            logout(request)
            return redirect("/users/login2/")
        ```

    - config/views.py

        ```python
        from django.shortcuts import redirect

        def index(request):
            if request.user.is_authenticated:
                return redirect("/posts/feeds/")
            else:
                return redirect("/users/login2/")
        ```

#### 1.2 Template의 동적 URL 변경

- 동적 URL 생성을 위한 요소 추가
    - 동적으로 URL을 생성해서 사용하기 위해서는 app별로 분리된 하위 urls.py에 app_name이라는 속성이 필요함
    - 일반적으로 app의 패키지명(디렉토리명)을 사용함

- users/urls.py

    ```python
    app_name = "users"
    urlpatterns = [
        ...
    ]
    ```

- posts/urls.py

    ```python
    app_name = "posts"
    urlpatterns = [
        ...
    ]
    ```

- Template을 위한 {% raw %}{% url %}{% endraw %} 태그

    - {% raw %}{% url "URL pattern name" %}{% endraw %} 태그는 Template에서 urls.py의 내용을 이용해 동적으로 URL을 생성함
        - 구조
            ```text
            {urls.py에 있는 app_name}:{path()에 지정된 name}
            ```

    - users/urls.py

        ```python
        app_name = "users"
        urlpatterns = [
            path("login2/", login_view, name="login"),
            path("logout/", logout_view, name="logout"),
            path("signup/", signup, name="sighup"),
        ]
        ```

- 실제로 동적 링크를 만들어보기
    - templates/users/signup.html

        ```html
        {% raw %}
        <a href="{% url 'users:login' %}">로그인 페이지로 이동</a>
        {% endraw %}
        ```

    - posts/urls.py

        ```python
        app_name - "posts"
        urlpatterns = [
            path("feeds/", feeds, name="feeds"),
            path("comment_add/", comment_add, name="comment_add"),
            path("comment_delete/<int:comment_id>/", comment_delete, name="comment_delete"),
            path("post_add/", post_add, name="post_add"),
        ]
        ```

- {% raw %} {% url %}{% endraw %}  태그를 사용하도록 기존 Template 코드 수정

    - 로그인 페이지
        - templates/users/login.html

            ```html
            {% raw %}
            <a href="{% url 'users:signup' %}">회원가입 페이지로 이동</a>
            {% endraw %}
            ```

    - 회원가입 페이지
        - templates/users/signup.html

            ```html
            {% raw %}
            <a href="{% url 'users:login' %}">로그인 페이지로 이동</a>
            {% endraw %}
            ```

    - 글 작성 페이지
        - templates/posts/post_add.html

            ```html
            {% raw %}
            <nav>
                <h1>
                    <a href="{% url 'posts:feeds' %}">Pystagram</a>
                </h1>
                <a href="{% url 'users:logout' %}">Logout</a>
            </nav>
            {% endraw %}
            ```

    - 피드 페이지 - 내비게이션 바 부분
        - templates/posts/feeds.html

            ```html
            {% raw %}
            <nav>
                <h1>
                    <a href="{% url 'posts:feeds' %}">Pystagram</a>
                </h1>
                <a href="{% url 'posts:post_add' %}">Add post</a>
                <a href="{% url 'users:logout' %}">Logout</a>
            </nav>
            {% endraw %}
            ```

    - 댓글 삭제 부분
        - templates/posts/feeds.html

            ```html
            {% raw %}
            <div class="post-comments">
                ...
                {% if user == comment.user %}
                    <form method="POST" action="{% url 'posts:comment_delete' comment_id=comment.id %}">
            {% endraw %}
            ```

    - 댓글 생성 부분
        - templates/posts/feeds.html

            ```html
            {% raw %}
            <div class="post-comment-create">
                <form method="POST" action="{% url 'posts:comment_add' %}">
            {% endraw %}
            ```

### 1.3 View의 동적 URL 변경

- View를 위한 reverse 함수

    - Terminal
        - Template에서 {% raw %} {% url %}{% endraw %}  태그를 사용하듯 View에서는 reverse 함수로 동적 URL을 생성할 수 있음

            ```bash
            python manage.py shell
            ```

            ```python
            from django.urls import reverse

            reverse('users:login')
            reverse('posts:feeds')

            # 추가 인수를 dict에 담아 키워드 인수로 전달
            reverse('posts:comment_delete', kwarg={'comment_id': 1})

            # 추가 인수를 list에 담아 위치 인수로 전달
            reverse('posts:comment_delete', args=[1])
            ```

- reverse 함수를 사용하도록 기존 View 코드 수정
    - config/views.py

        ```python
        from django.shortcuts import redirect

        def index(request):
            if request.user.is_authenticated:
                return redirect("posts:feeds")
            else:
                return redirect("users:login")
        ```

    - users/views.py

        ```python
        from django.urls import reverse

        def login_view(request):
            if request.user.is_authenticated:
                return redirect("posts:feeds")
            ...
                if user:
                    login(request, user)
                    return redirect("posts:feeds")
            ...

        def logout_view(request):
            logout(request)
            return redirect("users:login")

        def signup(request):
            ...
            login(request, user)
            return redirect("posts:feeds")
            ...
        ```

    - posts/views.py

        ```python
        from django.urls import reverse

        def feeds(request):
            ...
            if not user.is_authenticated:
                return redirect("users:login")
            ...

        def comment_add(request):
            ...
            if form.is_valid():
                comment = form.save(commit=False)
                comment.user = request.user
                comment.save()
                ...
                url_next = reverse("posts:feeds") + f"#post-{comment.post.id}"
                return HttpResponseRedirect(url_next)

        def comment_delete(request, comment_id):
            if request.method == "POST":
                comment = Comment.objects.get(id=comment_id)
                if comment.user == request.user:
                    comment.delete()
                    url = reverse("posts:feeds") + f"#post-{comment.post.id}"
                    return HttpResponseRedirect(url)
            ...

        def post_add(request):
            if request.method == "POST":
                ...
                if form.is_valid():
                    ...
                    url = reverse("posts:feeds") + f"#post-{post.id}"
                    return HttpResponseRedirect(url)
            ...
        ```

    - users/urls.py

        ```python
        app_name = "users"
        urlpatterns = [
            path("login/", login_view, name="login"),
            ...
        ```

### **2. 해시태그**

#### 2.1 다대다 관계 모델

- 다대다 관계모델

    - 다대일(Many-to-One, N:1) 관계
        - 한 테이블의 한 레코드가 다른 테이블의 여러 레코드와 연관됨을 나타내는 관계

    - 다대다(Many-to-Many, M2M, N:N) 관계
        - 한 테이블의 여러 레코드가 다른 테이블의 여러 레코드와 연관됨을 나타내는 관계

    - 예를 들어
        - 학생은 하나의 대학교에만 속할 수 있음: 다대일(학생:학교) 관계
        - 한 학생은 여러 개의 수업을 수강신청할 수 있으며 하나의 수업은 그 수업을 수강신청한 여러 명의 학생을 가질 수 있음: 다대다관계

- 다대다 테이블 구조

    - 학생 테이블

        |id|이름|
        |------|---|
        |1|김럭스|
        |2|최이즈리얼|
        |3|박룰루|

    - 수업 테이블

        |id|수업명|
        |--|------|
        |1|협곡|
        |2|칼바람|
        |3|TFT|

    - 학생과 수업을 중개하는 테이블

        |학생|수업|
        |------|---|
        |1|1|
        |1|2|
        |2|1|
        |2|2|
        |2|3|


- 해시태그 모델 생성, ManyToMany 연결
    - posts/models.py

        ```python
        class HashTag(models.Model):
            name = models.CharField("태그명", max_length=50)

            - posts/models.py

        class Post(models.Model):
            ...
            tags = models.ManyToManyField(HashTag, verbose_name="해시태그 목록", blank=True)
        ```

- Terminal
    - posts/models.py에서 HashTag 클래스를 Post 클래스보다 아래쪽에 선언
    - Post 클래스가 정의될 때는 HashTag 클래스를 알 수 없음
    - ForeignKey, ManyToManyField를 사용할 때 아래쪽에 선언한 모델을 참조하려면 문자열을 사용함

        ```bash
        python manage.py runserver
        ```

        ```python
        tags = models.ManyToManyField("posts.HashTag", verbose_name="해시태그 목록", blank=True)
        ```

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```


#### 2.2 다대다 모델 admin

- admin 구현
    - posts/admin.py

        ```python
        from posts.models import Post, PostImage, Comment, HashTag

        @admin.register(HashTag)
        class HashTagAdmin(admin.ModelAdmin):
            pass

            - posts/models.py

        class HashTag(models.Model):
            name = models.CharField("태그명", max_length=50)

            def __str__(self):
                return self.name
        ```

    - posts/models.py

        ```python
        from django.db.models import ManyToManyField
        from django.forms import CheckboxSelectMultiple
        ...
        @admin.register(Post)
        class PostAdmin(admin.ModelAdmin):
            ...
            formfield_overrides = {
                ManyToManyField: {"widget": checkboxSelectMultiple},
            }
        ```

- Template에 Post의 HashTag 표시
    - templates/posts/feeds.html

        ```html
        <div class="post-content">
            {{ post.content|linebreaksbr }}
            <div class="post-tags">
                {% for tag in post.tags.all %}
                    <span>#{{ tag.name }}</span>
                {% endfor %}
            </div>
        </div>
        ```

### 2.3 해시태그 검색

- 해시태그의 사용 예시

- 기본 구조
    - View: posts/views.py → tags
    - URL: /posts/tags/{tag의 name}/
    - Template: templates/posts/tags.html

- 기본 구조 구현
    - posts/views.py

        ```python
        ...
        def tags(request, tag_name):
            return render(request, "posts/tags.html")
        ```

    - posts/urls.py

        ```python
        from posts.views import ..., tags
        ...
        urlpatterns = [
            ...
            path("tags/<str:tag_name>/", tags, name="tags"),
        ]
        ```

    - templates/posts/tags.html

        ```html
        {% raw %}
        {% extends 'base.html' %}

        {% block content %}
            <nav>
                <h1>
                    <a href="{% url 'posts:feeds' %}">Pystagram</a>
                </h1>
                <a href="{% url 'posts:post_add' %}">Add Post</a>
                <a href="{% url 'users:logout' %}">Logout</a>
            </nav>
            <div id="tags">
                <header class="tags-header">
                    <h2>#{{ tag_name }}</h2>
                    <div>게시물 1,094</div>
                </header>
                <div class="post-grid-container">
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                    <div class="post-grid"></div>
                </div>
            </div>
        {% endblock %}
        {% endraw %}
        ```

- View에서 해시태그를 찾고 해당하는 Post 목록 돌려주기

    - posts/views.py

        ```python
        from posts.models import Post, Comment, PostImage, HashTag

        def tags(request, tag_name):
            tag = HashTag.objects.get(name=tag_name)
            print(tag)

            return render(request, "posts/tags.html")
        ```

    - posts/views.py

        ```python
        def tags(request, tag_name):
            tag = HashTag.objects.get(name=tag_name)
            posts = Post.objects.filter(tags=tag)

            context = {
                "tag_name": tag_name,
                "posts": posts,
            }
            return render(request, "posts/tags.html", context)
        ```

- Post 목록만큼 Grid 랜더링, tag_name 표시
    - templates/posts/tags.html

        ```html
        {% raw %}
        ...
        </nav>
        <div id="tags">
            <header class="tags-header">
                <h2>#{{ tag_name }}</h2>
                <div>게시물 {{ posts.count }}</div>
            </header>
            <div class="post-grid-container">
                {% for post in posts %}
                    <div class="post-grid"></div>
                {% endfor %}
            </div>
        </div>
        {% endraw %}
        ```

- 각각의 Post가 가진 첫 번째 이미지 보여주기
    - templates/posts/tags.html

        ```html
        {% raw %}
        <div class="post-grid-container">
            {% for post in posts %}
                {% if post.postimage_set.first and post.postimage_set.first.photo %}
                    <div class="post-grid">
                        <img src="{{ post.postimage_set.first.photo.url }}" alt="">
                    </div>
                {% endif %}
            {% endfor %}
        </div>
        {% endraw %}
        ```

    - posts/views.py

        ```python
        def tags(request, tag_name):
            try:
                tag = HashTag.objects.get(name=tag_name)
            except HashTag.DoesNotExist:
                # tag_name에 해당하는 HashTag를 찾지 못한 경우 빈 QuerySet을 돌려준다
                posts = Post.objects.none()
            else:
                posts = Post.objects.filter(tags=tag)

            context = {
                "tag_name": tag_name,
                "posts": posts,
            }
            return render(request, "posts/tags.html", context)
        ```

    - templates/posts/tags.html

        ```html
        {% raw %}
        <div class="post-grid-container">
            {% for post in posts %}
                {% if post.postimage_set.first and post.postimage_set.first.photo %}
                    <div class="post-grid">
                        <img src="{{ post.postimage_set.first.photo.url }}" alt="">
                    </div>
                {% endif %}
            {% empty %}
                <p>검색된 게시물이 없습니다</p>
            {% endfor %}
        </div>
        {% endraw %}
        ```

- 피드 페이지의 글에서 해시태그 링크 생성
    - templates/posts/feeds.html

        ```html
        {% raw %}
        <div class="post-content">
            {{ post.content|linebreaksbr }}
            <div class="post-tags">
                {% for tag in post.tags.all %}
                    <a href="{% url 'posts:tags' tag_name=tag.name %}">#{{ tag.name }}</a>
                {% endfor %}
            </div>
        </div>
        {% endraw %}
        ```

### 2.4 해시태그 생성

- ManyToManyField 항목 추가 실습
    - Terminal

        ```bash
        python manage.py shell
        ```

        ``` python
        from posts.models import Post, HashTag

        tag = HashTag.objects.create(name='테스트해시태그')
        tag

        from users.models import User

        user = User.objects.first()
        user

        post = Post.objects.create(user=user, content='HashTag 테스트용 Post')
        post
        post.tags.all

        post.tags.add(tag)
        post.tags.all()
        ```

- 해시태그 추가 Input 구현
    - templates/posts/post_add.html

        ```html
        {% raw %}
        <form method="POST" enctype="multipart/form-data">
            {% csrf_token %}
            <div>
                <!-- label의 for속성에는 가리키는 input의 id값을 입력 -->
                <label for="id_images">이미지</label>
                <input id="id_images" name="images" type="file" multiple>
            </div>
            {{ form.as_p }}
            <div>
                <label for="id_tags">해시태그</label>
                <input id="id_tags" name="tags" type="text" placeholder="쉼표(,)로 구분하여 여러 태그 입력">
            </div>
            <button type="submit">게시</button>
        </form>
        {% endraw %}
        ```

- 쉼표로 구분된 문자열 처리
    - Terminal

        ```bash
        python manage.py shell
        ```

        ```python
        tag_string = 'coffee,latte'
        tag_string.split(',')

        # 중간에 공백이 있는 경우 공백을 처리하지는 못함
        tag_string = 'coffee, latte'
        tag_string.split(',')

        # 좌우 공백을 없애는 함수는 strip()
        '   좌우 공백 포함    '.strip()

        # list comprehension으로 리스트 내의 공백 문자열 없애기
        tag_string = 'coffee,latte'
        tag_list = [tag.strip for tag in tag_string.split(',')]
        tag_list
        ```

    - posts/views.py

        ```python
        def post_add(request):
            if request.method == "POST":
                form = PostForm(request.POST)

                if form.is_valid():
                    post = form.save(commit=False)
                    ...

                    for image_file in request.FILES.getlist("images"):
                        ...

                    # "tags"에 전달 된 문자열을 분리해 HashTag생성
                    tag_string = request.POST.get("tags")
                    if tag_string:
                        tag_names = [tag_name.strip() for tag_name in tag_string.split(",")]
                        for tag_name in tag_names:
                            tag, _ = HashTag.objects.get_or_create(name=tag_name)
                            # get_or_create로 생성하거나 가져온 HashTag객체를 Post의 tags에 추가한다
                            post.tags.add(tag)

                    # 모든 PostImage와 Post의 생성이 완료되면
                    # 피드페이지로 이동하여 생성된 Post의 위치로 스크롤되도록 한다
                    url = reverse("posts:feeds") + f"#post-{post.id}"
                    return HttpResponseRedirect(url)
            ...
        ```