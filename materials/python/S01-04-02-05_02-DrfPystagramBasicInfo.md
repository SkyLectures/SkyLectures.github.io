---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-02-05_02-DrfPystagramBasicInfo
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 로그인/로그아웃 기능 구현

### 1. 로그인 기능

#### 1.1 로그인 페이지

- 기본 구조 구성
    - View: login_view
    - Template: pystagram/templetes/users/login.html
    - URL: pystagram/users/login/

- templates/users/login.html

    ```html
    <!DOCTYPE html>
    <html lang="ko">
    <body>
        <h1>로그인</h1>
    </body>
    </html>
    ```

- users/views.py

    ```python
    from django.shortcuts import render

    def login_view(request):
        return render(request, 'users/login.html')
    ```

- users/urls.py

    ```python
    from django.urls import path
    from users.views import login_view

    urlpatterns = [
        path('login/', login_view),
    ]
    ```

- config/urls.py

    ```python
    from django.urls import path, include

    urlpatterns = [
        path("admin/", admin.site.urls),
        path("", index),
        path("users/", include("users.urls")),
    ]
    ```

<br>

#### 1.2 로그인 여부에 따른 접속 제한

- 조건에 따른 View 동작 제어
    - 이미 사용자가 브라우저에서 로그인을 했다면
        - 피드 페이지로 이동
    - 사용자가 로그인을 한 적이 없거나 로그아웃을 했다면
        - 로그인 페이지로 이동

- 피드페이지 만들기
    - 글 관리 앱(posts)을 생성한 후에는 posts 쪽의 페이지를 사용함 → 이 파일은 삭제될 예정임
    - templates/users/feeds.html

        ```html
        <!DOCTYPE html>
        <html lang="ko">
        <body>
            <h1>피드 페이지</h1>
        </body>
        </html>
        ```

    - users/views.py

        ```python
        from django.shortcuts import render

        def feeds(request):
            return render(request, 'users/feeds.html')
        ```

    - users/urls.py

        ```python
        from django.urls import path
        from users.views import login_view, feeds

        urlpatterns = [
            path('feeds/', feeds),
        ]
        ```

- 관리자 페이지를 사용한 로그인/로그아웃
    - View 함수에 전달된 요청(reqest)에서 사용자 정보는 request.user 속성으로 가져올 수 있음
    - request.user 속성 중 is_authenticated 속성이 True이면 로그인 된 상태임<br><br>
    
    - users/views.py

        ```python
        from django.shortcuts import render

        def feeds(request):
            user = request.user
            is_authenticated = user.is_authenticated

            print("user: ", user)
            print("is_authenticated: ", is_authenticated)

            return render(request, 'users/feeds.html')
        ```

        - 터미널 콘솔에서 내용 확인 가능

- 로그인 여부에 따라 페이지 이동시키기
    - users/views.py

        ```python
        from django.shortcuts import render, redirect

        def login_view(request):
            if request.user.is_authenticated:
                return redirect("/users/feeds/")
            
            return render(request, 'users/login.html')

        def feeds(request):
            if not request.user.is_authenticated:
                return redirect("/users/login/")

            return render(request, 'users/feeds.html')
    ```

- 루트 경로에 접근 시, 로그인 여부에 따라 페이지 이동시키기
    - config/views.py

        ```python
        from django.shortcuts import redirect

        def index(request):
            if request.user.is_authenticated:
                return redirect("/users/feeds/")
            else:
                return redirect("/users/login/")
    ```

#### 1.3 로그인 기능 구현

- Form 클래스를 사용한 로그인 페이지 구성
    - users/forms.py

        ```python
        from django import forms

        class LoginForm(forms.Form):
            username = forms.CharField(min_length=3)
            password = forms.CharField(min_length=4)
        ```

    - 터미널에서 테스트하기

        ```bash
        python manage.py shell
        ```

        ```python
        from users.forms import LoginForm

        login_data = {"username": "u", "password": "p"}
        form = LoginForm(data=login_data)
        form.is_valid()
        form.errors

        login_data2 = {"username": "Sample username", "password": "Sample password"}
        form2 = LoginForm(data=login_data2)
        form2.is_valid()
        form2.errors
        ```

    - Form 적용하기
        - users/views.py

            ```python
            from django.shortcuts import render, redirect
            from users.forms import LoginForm

            def login_view(request):
                if request.user.is_authenticated:
                    return redirect("/users/feeds/")

                form = LoginForm()
                context = {"form": form}
                return render(request, 'users/login.html', context)
           ```

        - templates/users/login.html

            ```html
            ...
            <body>
                <h1>로그인</h1>
                {{ form.as_p }}
            </body>
            ...
            ```

        - 웹 브라우저에서 확인해보기
            - http://127.0.0.1:8000
            - 생성(렌더링)된 HTML 살펴보기
                - 화면은 잘 나오지만 <form> 태그가 포함되어있지 않음
                - Form이 제대로 작동하려면 구성요소들이 <form> 태그 안에 있어야 함

        - templates/users/login.html
            - {% raw %} ... {% endraw %} 구문은 블록코드 플러그인의 표기 문제 해결을 위한 것이므로 무시하도록 함
            
            ```html
            {% raw %}
            ...
            <body>
                <h1>로그인</h1>
                <form method="POST">
                    {% csrf_token %}
                    {{ form.as_p }}
                    <button type="submit">로그인</button>
                </form>
            </body>
            ...
            {% endraw %}
            ```
            
- View에 전달된 데이터를 Form으로 처리하기
    - users/views.py

        ```python
        def login_view(request):
            if request.user.is_authenticated:
                return redirect("/users/feeds/")

            if request.method == "POST":
                form = LoginForm(data=request.POST)
                print("form.is_valid(): ", form.is_valid())
                print("form.cleaned_data: ", form.cleaned_data)
                    
                context = {"form": form}
                return render(request, 'users/login.html', context)

            else:
                form = LoginForm()
                context = {"form": form}
                return render(request, "users/login.html", context)
        ```

        - 터미널 콘솔에서 내용 확인 가능

- View에서 로그인 처리하기
    - users/views.py

        ```python
        from django.contrib.auth import authenticate, login
        from django.shortcuts import render, redirect
        from users.forms import LoginForm

        def login_view(request):
            if request.user.is_authenticated:
                return redirect("/users/feeds/")

            if request.method == "POST":
                form = LoginForm(data=request.POST)
                if form.is_valid():
                    username = form.cleaned_data["username"]
                    password = form.cleaned_data["password"]
                    user = authenticate(username=username, password=password)

                    if user:
                        login(request, user)
                        return redirect("/users/feeds/")
                    else:
                        print("로그인에 실패했습니다.")

                context = {"form": form}
                return render(request, "users/login.html", context)
            else:
                form = LoginForm()
                context = {"form": form}
                return render(request, "users/login.html", context)
        ```


    - 터미널에서 테스트하기

        ```bash
        python manage.py shell
        ```

        ```python
        from django.contrib.auth import authenticate

        user = authenticate(username='a', password='b')
        print(user)

        user = authenticate(username='pystagram', password='1234')
        print(user)
        ```

### 2. 로그아웃 기능 구현

- 로그아웃 기본 구조 구현
    - View: logout_view
    - URL: /users/logout/
    - Template: 없음

- users/views.py

    ```python
    from django.contrib.auth import authenticate, login, logout
    from django.shortcuts import redirect
    ...
    def logout_view(request):
        logout(request)
        return redirect("/users/login/")
    ```

- users/urls.py

   ```python
    from users.views import login_view, feeds, logout_view
    ...
    urlpatterns = [
        path('login/', login_view),
        path('feeds/', feeds),
        path('logout/', logout_view),
    ]
    ```

- templates/users/feeds.html

   ```html
    <!doctype html>
    <html lang="ko">
    <body>
        <h1>피드 페이지</h1>
        <a href="/users/logout/">로그아웃</a>
    </body>
    </html>
    ```

#### 3. 로그인 기능 개선

- 피드 페이지에 로그인 상태 표시
    - templates/users/feeds.html

        ```html
        <!doctype html>
        <html lang="ko">
        <body>
            <h1>피드 페이지</h1>
            <div>{{ user.username }} (ID: {{ user.id }})</div>
            <a href="/users/logout/">로그아웃</a>
        </body>
        </html>
        ```

- 로그인 실패 시 정보 표시
    - users/views.py

        ```python
        def login_view(request):
        ...
            if request.method == "POST":
                form = LoginForm(data=request.POST)
                if form.is_valid():
                    ...

                    if user:
                        ...
                    else:
                        form.add_error(None, "입력한 자격증명에 해당하는 사용자가 없습니다.")
        ```

- 로그인 페이지 CSS 스타일링, Form 기능 추가
    - templates/users/login.html

        ```html
        {% raw %}
        {% load static %}
        <!doctype html>
        <html lang="ko">
        <head>
            <link rel="stylesheet" href="{% static 'css/style.css' %}">
        </head>
        <body>
            <div id="login">
                <form method="POST">
                {% csrt_token %}
                {{ form.as_p }}
                <button type="submit" class="btn btn-login">로그인</button>
                </form>
            </div>
        </body>
        </html>
        {% endraw %}
        ```

    - [style.css 다운로드](https://raw.githubusercontent.com/SkyLectures/LectureMaterials/refs/heads/main/Part01_Python/S01-04-03-04_01-WebService_Pystagram_01_style.css)
        - 다운로드 후 /static/css/ 에 복사해 둘 것

    - users/forms.py

        ```python
        from django import forms

        class LoginForm(forms.Form):
            username = forms.CharField(
                min_length=3,
                widget=forms.TextInput(
                    attrs={"placeholder": "사용자명 (3자리 이상)"},
                ),
            )
            password = forms.CharField(
                min_length=4,
                widget=forms.PasswordInput(
                    attrs={"placeholder": "비밀번호 (4자리 이상)"},
                ),
            )
        ```