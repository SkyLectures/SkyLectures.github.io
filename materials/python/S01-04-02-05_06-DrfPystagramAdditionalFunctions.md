---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-02-05_06-DrfPystagramAdditionalFunctions
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 글 상세 관리 기능 구현

- Post에 대한 상세 페이지 구현
- Template의 중복된 내용을 제거하는 리팩토링 실행

### 1. 글 상세 페이지

#### 1.1 기본구조

- View: posts/views.py → post_detail
- URL: /posts/<int:post_id>/
- Template: templates/posts/post_detail.html

#### 1.2 기본구조 구현

- posts/views.py

    ```python
    def post_detail(request, post_id):
        post = Post.objects.get(id=post_id)
        context = { "post": post }
        return render(request, "posts/post_detail.html", context)
    ```

- posts/urls.py

    ```python
    from posts.views import ..., post_detail

    app_name = "posts"
    urlpatterns = [
        ...
        path("<int:post_id>/", post_detail, name="post_detail"),
    ]
    ```

- templates/posts/post_detail.html

    ```html
    {% raw %}
    {% extends 'base.html' %}

    {% block content %}
    <div id="post_detail">
        <h1>Post Detail</h1>
    </div>
    {% endblock %}
    {% endraw %}
    ```

#### 1.3 Template 내용 구현

- templates/posts/feeds.html

    ```html
    {% raw %}
    {% extends 'base.html' %}
    {% block content %}
    <nav>...</nav>
    <div id="feeds" class="post-container">
        {% for post in posts %}
            <article id="post-{{ post.id }}" class="post">
            ...
            </article>
        {% endfor %}
    </div>
    ...
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/post_detail.html
    - 피드페이지에서는 for 반복문 안의 < article> 요소가 각각 하나의 Post를 나타냄
    - Post 상세화면에서는 Post Queryset 대신 단일 Post 객체가 전달되며 나머지 모습은 피드페이지와 동일함

        ```html
        {% raw %}
        {% extends 'base.html' %}

        {% block content %}
        <div id="feeds" class="post-container">
            <article id="post-{{ post.id }}" class="post">
            ...
            </article>
        </div>
        {% endblock %}
        {% endraw %}
        ```

#### 1.4 PostForm 전달

- posts/views.py

    ```python
    def post_detail(request, post_id):
        post = Post.objects.get(id=post_id)
        comment_form = CommentForm()
        context = {
            "post": post,
            "comment_form": comment_form,
        }
        return render(request, "posts/post_detail.html", context)
    ```

#### 1.5 {% raw %}{% include %} 태그로 Template 재사용{% endraw %}

- templates/posts/post.html
    - <article> 태그를 post.html로 재사용

        ```html
        <article id="post-{{ post.id }}" class="post">
        ...
        </article>
        ```

- templates/posts/feed.html

    ```html
    {% raw %}
    {% extends 'base.html' %}
    {% block content %}
    <nav>...</nav>
    <div id="feeds" class="post-container">
        {% for post in posts %}
            {% include 'posts/post.html' %}
        {% endfor %}
    </div>
    ...
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/post_detail.html

    ```html
    {% raw %}
    {% extends 'base.html' %}
    {% block content %}
    <nav>...</nav>
    <div id="feeds" class="post-container">
        {% include 'posts/post.html' %}
    </div>
    ...
    {% endblock %}
    {% endraw %}
    ```

- templates/nav.html
    - <nav> 태그의 내용을 별도의 nav.html로 이동

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

- templates/posts/feeds.html, templates/posts/post_detail.html 공통

    ```html
    {% raw %}
    {% extends 'base.html' %}
    {% block content %}
    {% include 'nav.html' %}
    <div id="feeds" class="post-container">
        {% include 'posts/post.html' %}
    </div>
    ...
    {% endblock %}
    {% endraw %}
    ```

#### 1.6 해시태그 검색결과에 링크 추가

- templates/posts/tags.html

    ```html
    {% raw %}
    <div class="post-grid">
        <a href="{% url 'posts:post_detail' post_id=post.id %}">
            <img src="{{ post.postimage_set.first.photo.url }}" alt="">
        </a>
    </div>
    {% endraw %}
    ```

### 2. 글 작성 후 이동할 위치

#### 2.1 Post 상세 화면에서 댓글 작성 시 상세화면으로 이동

- posts/views.py
    - 기존의 댓글 작성 후 redirect

        ```python
        def comment_add(request):
            ...
            if form.is_valid():
                ...
                url_next = reverse("posts:feeds") + f"#post-{comment.post.id}"
                return HttpResponseRedirect(url_next)
        ```

- templates/posts/post_detail.html
    - 댓글은 피드 페이지와 글 상세 페이지 양쪽에서 작성할 수 있음
    - 댓글 작성 완료 후 사용자를 이동시킬 페이지를 각각의 경우에 따라 다르게 지정할 필요가 있음

        ```html
        {% raw %}
        <div id="feeds" class="post-container">
            {% url 'posts:post_detail' post.id as action_redirect_to %}
            {% include 'posts/post.html' with action_redirect_url=action_redirect_to %}
        </div>
        {% endraw %}
        ```

- templates/posts/post.html

    ```html
    {% raw %}
    <div class="post-comment-create">
        <form method="POST" action="{% url 'posts:comment_add' %}?next={{ action_redirect_url }}">
            {% csrf_token %}
            <input type="hidden" name="post" value="{{ post.id }}">
            {{ comment_form.content }}
            <button type="submit">게시</button>
        </form>
    </div>
    {% endraw %}
    ```

- posts/views.py

    ```python
    def comment_add(request):
        ...
        if form.is_valid():
            ...
            comment.save()

            # URL로 "next"값을 전달받았다면 댓글 작성 완료 후 전달받은 값으로 이동한다
            if request.GET.get("next"):
                url_next = request.GET.get("next")

            # "next"값을 전달받지 않았다면 피드페이지의 글 위치로 이동한다
            else:
                url_next = reverse("posts:feeds") + f"#post-{comment.post.id}"

            return HttpResponseRedirect(url_next)
    ```

- templates/posts/feeds.html

    ```html
    {% raw %}
    <div id="feeds" class="post-container">
        {% for post in posts %}
            {% with post.id|stringformat:"s" as post_id %}
                {% url 'posts:feeds' as action_redirect_to %}
                {% include 'posts/post.html' with action_redirect_url=action_redirect_to|add:'#post-'|add:post.id %}
            {% endwith %}
        {% endfor %}
    </div>
    {% endraw %}
    ```

#### 2.2 Custom Template Filter

- posts/templatetags/custom_tags.py

    ```python
    from django import template

    register = template.Library()

    @register.filter
    def concat(value, arg):
        return f"{value}{arg}"
    ```

- templates/posts/feeds.html

    ```html
    {% raw %}
    {% extends 'base.html' %}
    {% load custom_tags %}

    {% block content %}
    {% include 'nav.html' %}
    <div id="feeds" class="post-container">
        {% for post in posts %}
            {% url 'posts:feeds' as action_redirect_to %}
            {% include 'posts/post.html' with action_redirect_url=action_redirect_to|concat:'#post-'|concat:post.id %}
        {% endfor %}
    </div>
    ...
    {% endblock %}
    {% endraw %}
    ```

### 3. Template 중복코드 제거

#### 3.1 화면 단위 기능 정리

- 지금까지 만든 화면 단위 기능
    - 로그인
    - 회원가입
    - 피드 페이지
    - 태그 페이지
    - 글 상세 페이지
    - 글 작성 페이지

- 비슷한 레이아웃을 가진 기능 묶음
    - 상단 내비게이션 바가 없는 레이아웃
        - 로그인
        - 회원가입
    - 내비게이션 바가 있는 레이아웃
        - 이미지 슬라이더 기능이 필요한 레이아웃
            - 피드 페이지
            - 글 상세 페이지
            - 태그 페이지
        - 이미지 슬라이더가 없어도 되는 레이아웃
            - 글 작성 페이지

- 레이아웃에 따라 base 정리
    - 상단 내비게이션 바가 없는 레이아웃: base.html
    - 내비게이션 바가 있는 레이아웃: base_nav.html
    - 내비게이션 바가 있으며 이미지 슬라이더 기능이 포함된 레이아웃: base_slider.html

#### 3.2 base.html 분할

- templates/_base.html
    - 모든 기반 레이아웃의 최상단 Template

        ```html
        {% raw %}
        {% load static %}
        <!doctype html>
        <html lang="ko">
        <head>
            <link rel="stylesheet" href="{% static 'css/style.css' %}">
            <title>Pystagram</title>
            {% block head %}{% endblock %}
        </head>
        <body>
            {% block base_content %}{% endblock %}
        </body>
        </html>
        {% endraw %}
        ```

- templates/base.html
    - 로그인, 회원가입에서 사용

        ```html
        {% raw %}
        {% extends '_base.html' %}

        {% block base_content %}
            {% block content %}{% endblock %}
        {% endblock %}
        {% endraw %}
        ```

- templates/base_nav.html
    - 글 작성에서 사용

        ```html
        {% raw %}
        {% extends '_base.html' %}

        {% block base_content %}
            {% include 'nav.html' %}
            {% block content %}{% endblock %}
        {% endblock %}
        {% endraw %}
        ```

- templates/base_slider.html
    - 피드, 글 상세에서 사용

        ```html
        {% raw %}
        {% extends '_base.html' %}
        {% load static %}

        {% block head %}
            <link href="{% static 'splide/splide.css' %}" rel="stylesheet">
            <script src="{% static 'splide/splide.js' %}"></script>
        {% endblock %}

        {% block base_content %}
            {% include 'nav.html' %}
            {% block content %}{% endblock %}
            <script>
                const elms = document.getElementsByClassName('splide');
                for (let i = 0; i < elms.length; i++) {
                    new Splide(elms[i]).mount();
                }
            </script>
        {% endblock %}
        {% endraw %}
        ```

#### 3.3 분할한 Template을 사용하도록 코드 수정

- templates/posts/feeds.html

    ```html
    {% raw %}
    {% extends 'base_slider.html' %}
    {% load custom_tags %}

    {% block content %}
        <div id="feeds" class="post-container">
            {% for post in posts %}
                {% url 'posts:feeds' as action_redirect_to %}
                {% include 'posts/post.html' with action_redirect_url=action_redirect_to|concat:'#post-'|concat:post.id %}
            {% endfor %}
        </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/post_detail.html

    ```html
    {% raw %}
    {% extends 'base_slider.html' %}

    {% block content %}
        <div id="feeds" class="post-container">
            {% url 'posts:post_detail' post.id as action_redirect_to %}
            {% include 'posts/post.html' with action_redirect_url=action_redirect_to %}
        </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/tags.html

    ```html
    {% raw %}
    {% extends 'base_nav.html' %}

    {% block content %}
        <div id="tags">
            ...
        </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/post_add.html

    ```html
    {% raw %}
    {% extends 'base_nav.html' %}

    {% block content %}
        <div id="post-add">
            ...
        </div>
    {% endblock %}
    {% endraw %}
    ```

### 4. 좋아요 기능**

#### 4.1 좋아요 모델, 관리자 구성

- ManyToManyField 추가
    - users/models.py

        ```python
        class User(AbstractUser):
            ...
            like_posts = models.ManyToManyField(
                "posts.Post",
                verbose_name="좋아요 누른 Post목록",
                related_name="like_users",
                blank=True,
            )
        ```

    - Terminal

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

- admin 구성

    - users/admin.py

        ```python
        @admin.register(User)
        class CustomUserAdmin(UserAdmin):
            fieldsets = [
                ...
                (
                    "추가필드",
                    {
                        "fields": ("profile_image", "short_description"),
                    },
                ),
                (
                    "연관객체",
                    {
                        "fields": ("like_posts",),
                    },
                ),
        ```

    - posts/models.py

        ```python
        class Post(models.Model):
            ...
            def __str__(self):
                return f"{self.user.username}의 Post(id: {self.id})"
        ```

    - users/models.py

        ```python
        class User(AbstractUser):
            ...
            def __str__(self):
                return self.username
        ```

    - posts/admin.py

        ```python
        class PostImageInline(admin.TabularInline):
            ...

        class LikeUserInline(admin.TabularInline):
            model = Post.like_users.through
            verbose_name = "좋아요 한 User"
            verbose_name_plural = f"{verbose_name} 목록"
            extra = 1

            def has_change_permission(self, request, obj=None):
                return False

        @admin.register(Post)
        class PostAdmin(admin.ModelAdmin):
            ...
            inlines = [
                CommentInline,
                PostImageInline,
                LikeUserInline,
            ]
            ...
        ```

#### 4.2 좋아요 토그 액션

- View 구현
    - posts/views.py
        - URL에서 좋아요 처리할 Post의 id를 전달받는다.

            ```python
            def post_like(request, post_id):
                post = Post.objects.get(id=post_id)
                user = request.user

                # 사용자가 "좋아요를 누른 Post목록"에 "좋아요 버튼을 누른 Post"가 존재한다면
                if user.like_posts.filter(id=post.id).exists():
                    # 좋아요 목록에서 삭제한다
                    user.like_posts.remove(post)

                # 존재하지 않는다면 좋아요 목록에 추가한다.
                else:
                    user.like_posts.add(post)

                # next로 값이 전달되었다면 해당 위치로, 전달되지 않았다면 피드페이지에서 해당 Post위치로 이동한다
                url_next = request.GET.get("next") or reverse("posts:feeds") + f"#post-{post.id}"
                return HttpResponseRedirect(url_next)
            ```

- URLconf
    - posts/urls.py

        ```python
        from posts.views import ..., post_like
        ...

        app_name = "posts"
        urlpatterns = [
            ...
            path("<int:post_id>/like/", post_like, name="post_like"),
        ]
        ```

- Template의 좋아요 버튼에 form 추가

    - templates/posts/post.html

        ```html
        {% raw %}
        <div class="post-buttons">
            <form action="{% url 'posts:post_like' post_id=post.id %}?next={{ action_redirect_url }}" method="POST">
                {% csrf_token %}
                <button type="submit"
                    {% if user in post.like_users.all %}
                        style="color: red;"
                    {% endif %}>
                    Likes({{ post.like_users.count }})
                </button>
            </form>
            <span>Comments({{ post.comment_set.count }})</span>
        </div>
        {% endraw %}
        ```


### 5. 팔로우/팔로잉 기능

#### 5.1 팔로우/팔로잉 모델, 관리자 구성

- 팔로우/팔로잉 관계
    - '해시태그', '좋아요'와 마찬가지로 ManyToManyField를 사용한 다대다관계로 구성
    - '해시태그', '좋아요'와 다른 점
        - '해시태그', '좋아요': 한쪽에서의 연결은 반대쪽에서의 연결도 나타내는 대칭적 관계
        - 팔로우/팔로잉 관계: 한 쪽에서의 연결과 반대쪽에서의 연결이 별도로 구분되는 비대칭적 관계
            - 같은 테이블(User)에서의 관계를 나타내야 함
            - 예시
                - User.username = [녹턴, 럭스, 람머스]
                - 이 User의 팔로워들(Followers)
                    - 녹턴의 팔로워들: 람머스
                    - 럭스의 팔로워들: 녹턴, 람머스
                    - 람머스의 팔로워들: 없음
                - 이 User가 팔로잉하는 대상들(Folowing)
                    - 녹턴이 팔로잉하는 사용자들: 럭스
                    - 럭스가 팔로잉하는 사용자들: 없음
                    - 람머스가 팔로잉하는 사용자들: 녹턴, 럭스
            - 팔로우/팔로잉 관계를 구성하는 중개 테이블
                - 이 중개 테이블의 데이터는 방향에 따라 나타내는 관계가 다른 비대칭적 관계를 나타냄
                - From User의 사용자는 To User의 사용자를 팔로우
                - To User의 사용자에게 From User의 사용자는 자신을 팔로잉하는 사용자로 취급

                    |From User|To User|
                    |---------|-------|
                    |람머스|녹턴|
                    |람머스|럭스|
                    |녹턴|럭스|



- 팔로우 관계 모델
    - users/models.py

        ```python
        class Relationship(models.Model):
            from_user = models.ForeignKey(
                "users.User",
                verbose_name="팔로우를 요청한 사용자",
                related_name="following_relationships",
                on_delete=models.CASCADE,
            )
            to_user = models.ForeignKey(
                "users.User",
                verbose_name="팔로우 요청의 대상",
                related_name="follower_relationships",
                on_delete=models.CASCADE,
            )
            created = models.DateTimeField(auto_now_add=True)

            def __str__(self):
                return f"관계 ({self.from_user} -> {self.to_user})"

        ...
        class User(AbstractUser):
            ...
            following = models.ManyToManyField(
                "self",
                verbose_name="팔로우 중인 사용자들",
                related_name="followers",
                symmetrical=False,
                through="users.Relationship",
            )
        ```

    - Terminal

        ```bash
        python manage.py makemigrations
        python manage.py migrate
        ```

- 팔로우 관계 admin
    - users/admin.py

        ```python
        class FollowersInline(admin.TabularInline):
            model = User.following.through
            fk_name = "from_user"
            verbose_name = "내가 팔로우 하고 있는 사용자"
            verbose_name_plural = f"{verbose_name} 목록"


        class FollowingInline(admin.TabularInline):
            model = User.following.through
            fk_name = "to_user"
            verbose_name = "나를 팔로우 하고 있는 사용자"
            verbose_name_plural = f"{verbose_name} 목록"


        @admin.register(User)
        class CustomUserAdmin(UserAdmin):
            fieldsets = [
                ...
            ]
            inlines = [
                FollowersInline,
                FollowingInline,
            ]
        ```

### 6. 프로필 페이지

#### 6.1 프로필 페이지 기본구조 및 연결

- View: users/views.py → profile
- URL: /users/<int:user_id>/profile/
- Template: templates/users/profile.html

#### 6.2 프로필 페이지 기본구조 및 연결 구현

- users/views.py

    ```python
    def profile(request, user_id):
        return render(request, "users/profile.html")
    ```

- users/urls.py

    ```python
    from users.views import ..., profile
    ...

    app_name = "users"
    urlpatterns = [
        ...
        path("<int:user_id>/profile/", profile, name="profile"),
        ...
    ]
    ```

- templates/users/profile.html

    ```html
    {% raw %}
    {% extends 'base_nav.html' %}

    {% block content %}
    <div id="profile">
        <h1>Profile</h1>
    </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/posts/post.html

    ```html
    {% raw %}
    <article id="post-{{ post.id }}" class="post">
        <header class="post-header">
            <a href="{% url 'users:profile' user_id=post.user.id %}">
                {% if post.user.profile_image %}
                    <img src="{{ post.user.profile_image.url }}" alt="">
                {% endif %}
                <span>{{ post.user.username }}</span>
            </a>
        </header>
    {% endraw %}
    ```

#### 6.3 프로필 Template에 정보 전달

- users/views.py

    ```python
    from django.shortcuts import render, redirect, get_object_or_404
    ...
    from users.models import User

    def profile(request, user_id):
        user = get_object_or_404(User, id=user_id)
        context = {
            "user": user,
        }
        return render(request, "users/profile.html", context)
    ```

#### 6.4 프로필 Template 구성

- templates/users/profile.html

    ```html
    {% raw %}
    {% extends 'base_nav.html' %}

    {% block content %}
    <div id="profile">
        <div class="info">
            <!-- 프로필 이미지 영역 -->
            {% if user.profile_image %}
                <img src="{{ user.profile_image.url }}">
            {% endif %}

            <!-- 사용자 정보 영역 -->
            <div class="info-texts">
                <h1>{{ user.username }}</h1>
                <div class="counts">
                    <dl>
                        <dt>Posts</dt>
                        <dd>{{ user.post_set.count }}</dd>
                        <dt>Followers</dt>
                        <dd>{{ user.followers.count }}</dd>
                        <dt>Following</dt>
                        <dd>{{ user.following.count }}</dd>
                    </dl>
                </div>
                <p>{{ user.short_description }}</p>
            </div>
        </div>
        <!-- 사용자가 작성한 Post목록 -->
        <div class="post-grid-container">
            {% for post in user.post_set.all %}
                {% if post.postimage_set.first %}
                    {% if post.postimage_set.first.photo %}
                        <div class="post-grid">
                            <a href="{% url 'posts:post_detail' post_id=post.id %}">
                                <img src="{{ post.postimage_set.first.photo.url }}" alt="">
                            </a>
                        </div>
                    {% endif %}
                {% endif %}
            {% endfor %}
        </div>
    </div>
    {% endblock %}
    {% endraw %}
    ```

### 7. 팔로우/팔로잉 목록

#### 7.1 중개 테이블의 데이터 가져오기

- Terminal

    ```bash
    python manage.py shell
    ```

    ```python
    from user.models import User, Relationship

    user = User.objects.get(id=1)
    user.followers.all()
    user.follower_relationships.all()

    for relationship in user.follower_relationships.all():
        print(relationship, relationship.created)
    ```

#### 7.2 base_profile.html 구성

- templates/base_profile.html

    ```html
    {% raw %}
    {% extends 'base_nav.html' %}

    {% block content %}
    <div id="profile">
        <div class="info">
            <!-- 프로필 이미지 영역 -->
            {% if user.profile_image %}
                <img src="{{ user.profile_image.url }}">
            {% endif %}

            <!-- 사용자 정보 영역 -->
            <div class="info-texts">
                <h1>{{ user.username }}</h1>
                <div class="counts">
                    <dl>
                        <dt>Posts</dt>
                        <dd>{{ user.post_set.count }}</dd>
                        <dt>Followers</dt>
                        <dd>{{ user.followers.count }}</dd>
                        <dt>Following</dt>
                        <dd>{{ user.following.count }}</dd>
                    </dl>
                </div>
                <p>{{ user.short_description }}</p>
            </div>
        </div>
        {% block bottom_data %}{% endblock %}
    </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/users/profile.html

    ```html
    {% raw %}
    {% extends 'base_profile.html' %}

    {% block bottom_data %}
    <!-- 사용자가 작성한 Post목록 -->
    <div class="post-grid-container">
        {% for post in user.post_set.all %}
            {% if post.postimage_set.first %}
                {% if post.postimage_set.first.photo %}
                    <div class="post-grid">
                        <a href="{% url 'posts:post_detail' post_id=post.id %}">
                            <img src="{{ post.postimage_set.first.photo.url }}" alt="">
                        </a>
                    </div>
                {% endif %}
            {% endif %}
        {% endfor %}
    </div>
    {% endblock %}
    {% endraw %}
    ```

#### 7.3 팔로우/팔로잉 목록

- 자신을 팔로우하는 사용자 목록(Followers)
    - View: users/views.py → followers
    - URL: /users/<int:user_id>/followers/
    - Template: templates/users/followers.html
<br><br>
- 자신이 팔로우하는 사용자 목록(Following)
    - View: users/views.py → following
    - URL: /users/<int:user_id>/following/
    - Template: templates/users/following.html

- users/views.py

    ```python
    def followers(request, user_id):
        user = get_object_or_404(User, id=user_id)
        relationships = user.follower_relationships.all()
        context = {
            "user": user,
            "relationships": relationships,
        }
        return render(request, "users/followers.html", context)


    def following(request, user_id):
        user = get_object_or_404(User, id=user_id)
        relationships = user.following_relationships.all()
        context = {
            "user": user,
            "relationships": relationships,
        }
        return render(request, "users/following.html", context)
    ```

- users/urls.py

    ```python
    from users.views import ..., followers, following
    ...

    app_name = "users"
    urlpatterns = [
        ...
        path("<int:user_id>/followers/", followers, name="followers"),
        path("<int:user_id>/following/", following, name="following"),
    ]
    ```

- templates/users/followers.html

    ```html
    {% raw %}
    {% extends 'base_profile.html' %}

    {% block bottom_data %}
    <div class="relationships">
        <h3>Followers</h3>
        {% for relationship in relationships %}
            <div class="relationship">
                <a href="{% url 'users:profile' user_id=relationship.from_user.id %}">
                    {% if relationship.from_user.profile_image %}
                        <img src="{{ relationship.from_user.profile_image.url }}">
                    {% endif %}
                    <div class="relationship-info">
                        <span>{{ relationship.from_user.username }}</span>
                        <span>{{ relationship.created|date:"y.m.d" }}</span>
                    </div>
                </a>
            </div>
        {% endfor %}
    </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/users/following.html

    ```html
    {% raw %}
    {% extends 'base_profile.html' %}

    {% block bottom_data %}
    <div class="relationships">
        <h3>Following</h3>
        {% for relationship in relationships %}
            <div class="relationship">
                <a href="{% url 'users:profile' user_id=relationship.to_user.id %}">
                    {% if relationship.to_user.profile_image %}
                        <img src="{{ relationship.to_user.profile_image.url }}">
                    {% endif %}
                    <div class="relationship-info">
                        <span>{{ relationship.to_user.username }}</span>
                        <span>{{ relationship.created|date:"y.m.d" }}</span>
                    </div>
                </a>
            </div>
        {% endfor %}
    </div>
    {% endblock %}
    {% endraw %}
    ```

#### 7.4 프로필 페이지 링크 구성

- templates/base_profile.html

    ```html
    {% raw %}
    <!-- 사용자 정보 영역 -->
    <div class="info-texts">
        <h1>{{ user.username }}</h1>
        <div class="counts">
            <dl>
                <dt>Posts</dt>
                <dd>
                    <a href="{% url 'users:profile' user_id=user.id %}">{{ user.post_set.count }}</a>
                </dd>
                <dt>Followers</dt>
                <dd>
                    <a href="{% url 'users:followers' user_id=user.id %}">{{ user.followers.count }}</a>
                </dd>
                <dt>Following</dt>
                <dd>
                    <a href="{% url 'users:following' user_id=user.id %}">{{ user.following.count }}</a>
                </dd>
            </dl>
        </div>
    {% endraw %}
    ```


### 8. 팔로우 버튼

#### 8.1 팔로우 토글 View

- View: users/views.py → follow
- URL: /users/<int:user_id>/follow/
- Template: 없음

#### 8.2 팔로우 토글 View 구현

- users/views.py

    ```python
    from django.http import HttpResponseRedirect
    from django.urls import reverse
    ...

    def follow(request, user_id):
        # 로그인 한 유저
        user = request.user
        # 팔로우 하려는 유저
        target_user = get_object_or_404(User, id=user_id)

        # 팔로우 하려는 유저가 이미 자신의 팔로잉 목록에 있는 경우
        if target_user in user.following.all():
            # 팔로잉 목록에서 제거
            user.following.remove(target_user)

        # 팔로우 하려는 유저가 자신의 팔로잉 목록에 없는 경우
        else:
            # 팔로잉 목록에 추가
            user.following.add(target_user)

        # 팔로우 토글 후 이동할 URL이 전달되었다면 해당 주소로,
        # 전달되지 않았다면 로그인 한 유저의 프로필 페이지로 이동
        url_next = request.GET.get("next") or reverse("users:profile", args=[user.id])
        return HttpResponseRedirect(url_next)
    ```

- users/urls.py

    ```python
    from users.views import ..., follow
    ...

    app_name = "users"
    urlpatterns = [
        ...
        path("<int:user_id>/follow/", follow, name="follow"),
    ]
    ```

#### 8.3 팔로우 버튼 추가

- templates/posts/post.html

    ```html
    {% raw %}
    <article id="post-{{ post.id }}" class="post">
        <header class="post-header">
            <a href="{% url 'users:profile' user_id=post.user.id %}">
                ...
            </a>

            <!-- 글의 작성자가 로그인 한 사용자라면 팔로우 버튼을 표시하지 않는다 -->
            <!-- (자기 자신을 팔로우 하는것을 방지) -->
            {% if user != post.user %}
                <form action="{% url 'users:follow' user_id=post.user.id %}?next={{ action_redirect_url }}" method="POST">
                    {% csrf_token %}
                    <button type="submit" class="btn btn-primary">
                        <!-- 이 Post의 작성자가 이미 자신의 팔로잉 목록에 포함된 경우 -->
                        {% if post.user in user.following.all %}
                            Unfollow
                        <!-- 이 Post의 작성자를 아직 팔로잉 하지 않은 경우 -->
                        {% else %}
                            Follow
                        {% endif %}
                    </button>
                </form>
            {% endif %}
        </header>
    {% endraw %}
    ```
