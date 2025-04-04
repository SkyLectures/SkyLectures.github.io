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
    from users.views import login_view, logout_view, signup

    urlpatterns = [
        path("login/", login_view),
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

- SignupForm의 데이터 가져오기
    - users/views.py

        ```python
        def signup(request):
            form = SignupForm(data=request.POST, files=request.FILES)
            if form.is_valid():
                username = form.cleaned_data["username"]    