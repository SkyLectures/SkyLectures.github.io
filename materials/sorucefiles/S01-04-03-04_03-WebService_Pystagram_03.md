---
layout: page
title:  "Mini Project: Pystagram 만들기"
date:   2025-04-04 10:20:00 +0900
permalink: /materials/S01-04-03-04_03-WebService_Pystagram_03
categories: materials
---

- 코드출처: 이한영의 Django 입문(디지털북스)

---

## 회원가입 기능 구현

- 회원가입 기능 기본 구조
    - View: signup
    - URL: /users/signup/
    - Template: templates/users/signup.html

### 1. 기본 구조 생성

- users/views.py

    ```python
    from django.shortcuts import render

    def signup(request):
        return render(request, 'users/signup.html')
    ```

- users/urls.py

    ```python
    from django.urls import path
    from users.views import login_view, feeds, logout_view, signup

    urlpatterns = [
        path("login/", login_view),
        path('feeds/', feeds),
        path("logout/", logout_view),
        path("signup/", signup),
    ]
   ```

- templates/users/signup.html

    ```html
    <!doctype html>
    <html lang="ko">
    <body>
        <h1>회원가입</h1>
    </body>
    </html>
    ```

<br>

### 2. SignupForm을 사용한 Tempalte 구성

- SignupForm 클래스 정의
    - users/forms.py

        ```python
        class SignupForm(forms.Form):
            username = forms.CharField()
            password1 = forms.CharField(widget=forms.PasswordInput)
            password2 = forms.CharField(widget=forms.PasswordInput)
            profile_image = forms.ImageField()
            short_description = forms.CharField()
        ```

- View에서 Template에 SignupForm 전달
    - users/views.py

        ```python
        from users.forms import LoginForm, SignupForm

        def signup(request):
            form = SignupForm()
            context = {"form": form}
            return render(request, "users/signup.html", context)
       ```

    - templates/users/signup.html

        ```html
        {% raw %}
        <!doctype html>
        <html lang="ko">
        <body>
            <h1>Sign up</h1>
            <form method="POST" enctype="multipart/form-data">
            {% csrf_token %}
            {{ form.as_p }}
            <button type="submit">회원가입</button>
            </form>
        </body>
        </html>
        {% endraw %}
       ```
<br>

### 3. View에 회원가입 로직 구현

- Terminal에서 User 모델(클래스)의 create_user 메소드 사용 확인하기

    ```bash
    python manage.py shell
    ```

    - create 메소드
        - 생성 후 다시 확인해 보면 잘 출력함

            ```bash
            from users.models import User

            user = User.objects.create(username="sample", password="sample")
            print(user.id, user.username, user.password)
            ```

    - 인증 처리 시 제대로 된 값을 가져오지 못함
        - Django는 User의 비밀번호를 변형해서 저장하고 로드함
        - 따라서 읽어올 때에도 복호화 기능을 적용하므로 그대로 읽어오지 못함
        - 그대로 저장하는 것은 개인정보보호법 위반(국내에서는 create 메소드는 사용하면 안됨)

            ```bash
            from django.contrib.auth import authenticate

            result = authenticate(username="sample", password="sample")
            print(result)
            ```

    - create_user 메소드

        ```bash
        from users.models import User

        user2 = User.objects.create_user(username="sample2", password="sample2", short_description="sample2")
        print(user2.id, user2.short_description, user2.password)
        ```

- SignupForm의 데이터 가져오기
    - users/views.py
        - 실행 시 데이터 정상 전달 여부 확인(터미널 로그)

            ```python
            def signup(request):
                if request.method == "POST":
                    print(request.POST)
                    print(request.FILES)
                form = SignupForm()
                context = {"form": form}
                return render(request, "users/signup.html", context)
            ```

        - 문자열 데이터와 파일 데이터가 함께 전달되어야 함

            ```python
            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)
                    if form.is_valid():
                        username = form.cleaned_data["username"]
                        password1 = form.cleaned_data["password1"]
                        password2 = form.cleaned_data["password2"]
                        profile_image = form.cleaned_data["profile_image"]
                        short_description = form.cleaned_data["short_description"]
                        print(username)
                        print(password1, password2)
                        print(profile_image)
                        print(short_description)

                    context = {"form": form}
                    return render(request, "users/signup.html", context)

                form = SignupForm()
                context = {"form": form}
                return render(request, "users/signup.html", context)
            ```

        - 전달 후 로그 확인

- User 생성하기
    - User 생성 기준
        - 비밀번호(password1)와 비밀번호 확인(password2)은 값이 같아야 함
        - 같은 사용자명(username)을 사용하는 User는 생성 불가 및 오류 전달

    - Terminal에서 테스트용 User 생성하기
        - 존재하는 계정, 존재하지 않는 계정 확인

            ```bash
            python manage.py shell
            ```

            ```bash
            from users.models import User

            User.objects.filter(username="pystagram")
            User.objects.filter(username="pystagram").exists()

            User.objects.filter(username="no_user")
            User.objects.filter(username="no_user").exists()
            ```

    - users/views.py
        - 입력받은 username이 존재하면 Form에 에러 전달, 존재하지 않으면 생성
        - password1, password2가 같은지도 검사

            ```python
            from users.models import User

            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)
                    if form.is_valid():
                        username = form.cleaned_data["username"]
                        password1 = form.cleaned_data["password1"]
                        password2 = form.cleaned_data["password2"]
                        prifile_image = form.cleaned_data["profile_image"]
                        short_description = form.cleaned_data["short_description"]

                        if password1 != password2:
                            form.add_error("password2", "비밀번호와 비밀번호 확인란의 값이 다릅니다.")

                        if User.objects.filter(username=username).exists():
                            form.add_error("username", "입력한 사용자명은 이미 사용중입니다.")

                        if form.errors:
                            context = {"form": form}
                            return render(request, "users/signup.html", context)
                        else:
                            user = User.objects.create_user(
                                username = username,
                                password = password1,
                                profile_image = profile_image,
                                short_description = short_description,
                            )
                            login(request, user)
                            return redirect("/users/feeds/")
                else:
                    form = SignupForm()
                    context = {"form": form}
                    return render(request, "users/signup.html", context)
            ```

<br>

### 4. SignupForm 내부에서 데이터 유효성 검사

- clean_username 메서드 작성
    - users/forms.py

        ```python
        from django import forms
        from django.core.exceptions import ValidationError
        from users.models import User

        class SignupForm(forms.Form):
            ...
            def clean_username(self):
                username = self.cleaned_data["username"]
                if User.objects.filter(username=username).exists():
                    raise ValidationError(f"입력한 사용자명({username})은 이미 사용중입니다", code="invalid")
                return None
        ```

- clean 메서드로 password1, password2 검증
    - users/forms.py

        ```python
        class SignupForm(forms.Form):
            def clean_username(self):
            ...

            def clean(self):
                password1 = self.cleaned_data["password1"]
                password2 = self.cleaned_data["password2"]
                if password1 != password2:
                    self.add_error("password2", "비밀번호와 비밀번호 확인란의 값이 다릅니다")
        ```

- View 함수와 SignupForm 리팩토링
    - 검증로직을 Form 내부로 이동했으므로 기존 코드는 정리
        - users/views.py

            ```python
            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)

                    if form.is_valid():
                        username = form.cleaned_data["username"]
                        password1 = form.cleaned_data["password1"]
                        prifile_image = form.cleaned_data["profile_image"]
                        short_description = form.cleaned_data["short_description"]

                        user = User.objects.create_user(
                            username = username
                            password = password1
                            prifile_image = prifile_image
                            short_description = short_description
                        )

                        login(request, user)
                        return redirect("/users/feeds/")
                    else:
                        context = {"form": form}
                        return render(request, "users/signup.html", context)
                else:
                    form = SignupForm()
                    context = {"form": form}
                    return render(request, "users/signup.html", context)
            ```

    <br>

    - cleaned_data로 사용자를 생성하던 로직도 Form 내부로 이동
        - users/forms.py

            ```python
            class SignupForm(forms.Form):
                def clean(self):
                    ...

                def save(self):
                    username = self.cleaned_data["username"]
                    password1 = self.cleaned_data["password1"]
                    profile_image = self.cleaned_data["profile_image"]
                    short_description = self.cleaned_data["short_description"]
                    user = User.objects.create_user(
                        username=username,
                        password=password1,
                        profile_image=profile_image,
                        short_description=short_description,
                    )
                    return user
            ```

    <br>

    - 새로운 save()함수 적용
        - View 내부의 사용자 생성로직 삭제
        - 새로 만든 save()함수를 사용하도록 변경

        <br>

        - users/views.py
        
            ```python
            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)

                    if form.is_valid():
                        user = form.save()
                        login(request, user)
                        return redirect("/users/feeds/")

                    else:
                        context = {"form": form}
                        return render(request, "users/signup.html", context)

                else:
                    form = SignupForm()
                    context = {"form": form}
                    return render(request, "users/signup.html", context)
            ```

    - 중복 출현 로직 제거
        - users/views.py

            ```python
            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)

                    if form.is_valid():
                        user = form.save()
                        login(request, user)
                        return redirect("/users/feeds/")

                else:
                    form = SignupForm()

                context = {"form": form}
                return render(request, "users/signup.html", context)
            ```

- View에서 처리되는 프로세스의 종류(프로세스의 내용만 비교할 것. 코딩작업 하지 않음)
    - GET 요청
        - SignupForm()으로 생성된 빈 form을 사용자에게 보여줌
        - users/views.py

            ```python
            def signup(request):
                if request.method == "POST":
                    # 해당 없음

                else:
                    form = SignupForm()

                # context에 빈 Form이 전달됨
                context = {"form": form}
                return render(request, "users/signup.html", context)
            ```
    - POST 요청이며, 데이터를 받은 SignupForm이 유효한 경우
        - SignupForm(data=...)으로 생성된 form의 save() 메서드로 User 생성, redirect로 경로가 변경됨
        - users/views.py

            ```python
            def signup(request):
                # POST 요청 시 form이 유효하다면 최종적으로 rediret 처리됨
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)

                    if form.is_valid():
                        user = form.save()
                        login(request, user)
                        return redirect("/posts/feeds/")

                # 이후 로직은 실행되지 않음
            ```
    - POST 요청이며, 데이터를 받은 SignupForm이 유효하지 않은 경우
        - SignupForm(data=...)으로 생성된 form에는 error가 추가되며, 그 form을 사용자에게 보여줌
        - users/views.py

            ```python
            def signup(request):
                if request.method == "POST":
                    form = SignupForm(data=request.POST, files=request.FILES)

                    if form.is_valid():
                        # 검증에 실패하여 이 영역으로 들어오지 못함

                # context에 error를 포함한 form이 전달됨
                context = {"form": form}
                return render(request, "users/signup.html", context)
            ```

### 5. Template 스타일링과 구조 리팩토링

- templates/users/signup.html

    ```html
    {% raw %}
    {% load static %}
    <!document html>
    <html lang="ko">
    <head>
        <link rel="stylesheet" href="{% static 'css/style.css' %}">
    </head>
    <body>
        <div id="signup">
            <form method="POST" enctype="multipart/form-data">
                <h1>Pystagram</h1>
                {% csrf_token %}
                {{ form.as_p }}
                <button type="submit" class="btn btn-signup">가입</button>
            </form>
        </div>
    </body>
    </html>
    {% endraw %}
    ```

- templates/base.html
    - Template을 확장하는 {% raw %}{% extends %}{% endraw %} 태그

        ```html
        {% raw %}
        {% load static %}
        <!doctype html>
        <html lang="ko">
        <head>
            <link rel="stylesheet" href="{% static 'css/style.css' %}">
        </head>
        <body>
            {% block content %}{% endblock %}
        </body>
        </html>
        {% endraw %}
        ```

- templates/users/login.html

    ```html
    {% raw %}
    {% extends 'base.html' %}

    {% block content %}
    <div id="login">
        <form method="POST">
            <h1>Pystagram</h1>
            {% csrf_token %}
            {{ form.as_p }}
            <button type="submit" class="btn btn-login">로그인</button>
        </form>
    </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/users/signup.html

    ```html
    {% raw %}
    {% extends 'base.html' %}

    {% block content %}
    <div id="signup">
        <form method="POST" enctype="multipart/form-data">
            <h1>Pystagram</h1>
            {% csrf_token %}
            {{ form.as_p }}
            <button type="submit" class="btn btn-signup">가입</button>
        </form>
    </div>
    {% endblock %}
    {% endraw %}
    ```

- templates/users/signup.html
    - 회원가입과 로그인 페이지 간의 링크 추가

        ```html
        {% raw %}
        <div id="signup">
            <form method="POST" enctype="multipart/form-data">
                ...
                <button type="submit" class="btn btn-signup">가입</button>
                <a href="/users/login/">로그인 페이지로 이동</a>
            </form>
        </div>
        {% endraw %}
        ```

- templates/users/login.html

    ```html
    {% raw %}
    <div id="login">
        <form method="POST">
            ...
            <button type="submit" class="btn btn-login">로그인</button>
            <a href="/users/signup/">회원가입 페이지로 이동</a>
        </form>
    </div>
    {% endraw %}
    ```