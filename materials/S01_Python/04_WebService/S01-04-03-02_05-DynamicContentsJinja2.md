---
layout: page
title:  "Flask에서의 동적 콘텐츠 관리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-03-02_05-DynamicContentsJinja2
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 동적 콘텐츠를 위한 Jinja2 활용 개요

### 1.1 템플릿 엔진

- **Jinja2**
    - Flask 웹 프레임워크에서 동적인 웹 페이지를 생성하고 사용자에게 다양한 정보를 제공하기 위해 템플릿 엔진
    - 동적 웹 페이지 개발에 필수적인 요소
    - Jinja2를 사용하면 
        - Python 코드와 HTML 구조를 분리하여 웹 애플리케이션의 유지보수성을 높임
        - 데이터 바인딩을 통해 서버 측에서 생성된 데이터를 HTML 템플릿에 쉽게 통합할 수 있음

- **템플릿 엔진의 필요성**
    - 순수한 HTML만으로는 웹 페이지에 동적인 데이터를 표시하거나 반복적인 구조를 효율적으로 처리하기 어려움
    - 템플릿 엔진은 
        - HTML과 유사한 문법을 사용하여 변수, 반복문, 조건문 등의 프로그래밍 요소를 HTML 구조 내에 삽입할 수 있도록 함
        - 서버 측에서 데이터를 템플릿에 주입하여 최종적인 HTML 문서를 생성하고, 이를 클라이언트에게 전송함

- **Jinja2의 특징**
    - 유연하고 강력한 문법
        - 변수 출력
        - 조건문 (`if`, `elif`, `else`)
        - 반복문 (`for`)
        - 매크로 (`macro`)
        - 필터 (`filter`)
        - (`extends`, `block`) 등

    - 보안
        - XSS(Cross-Site Scripting) 공격을 방지하기 위해 기본적으로 HTML 자동 이스케이핑 기능을 제공함

    - 확장성
        - 사용자 정의 필터, 테스트, 전역 변수 등을 추가하여 기능을 확장할 수 있음

    - Flask와의 통합
        - Flask는 Jinja2를 기본 템플릿 엔진으로 내장하고 있어 별도의 설정 없이 편리하게 사용할 수 있음

- **Jinja2 템플릿 처리 과정**
    1. Flask 애플리케이션의 뷰 함수에서 템플릿에 전달할 데이터를 Python 딕셔너리 형태로 준비
    2. `render_template()` 함수를 사용하여 템플릿 파일의 이름과 데이터를 인자로 전달
    3. Flask는 Jinja2 엔진을 사용하여 해당 템플릿 파일을 로드, 전달된 데이터를 템플릿 문법에 따라 HTML 구조 내에 삽입
    4. Jinja2 엔진은 최종적으로 완성된 HTML 문자열을 반환하며, Flask는 이 문자열을 클라이언트에게 응답으로 전송

### 1.2 필요한 모듈 및 라이브러리

- Flask (필수)
    - 웹 프레임워크의 핵심
    - Jinja2 템플릿 엔진 내장
    - `render_template()` 함수를 통해 Jinja2 템플릿을 렌더링하는 기능 제공

- Jinja2 (Flask의 의존성)
    - 강력하고 유연한 Python 템플릿 엔진
    - Flask를 설치하면 자동으로 함께 설치됨

## 2. 실습 예제

### 2.1 프로젝트 구조

```
my_app/
├── app.py         # Flask 애플리케이션 코드
└── templates/
    └── index.html   # Jinja2 템플릿 파일
```

### 2.2 프로젝트 파일

#### 2.2.1 `app.py` (Flask 애플리케이션 코드)

```python
#//file: "app.py"
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    user = {'username': 'Alice', 'age': 30}
    items = ['사과', '바나나', '체리']
    is_logged_in = True
    return render_template('index.html', user=user, items=items, logged_in=is_logged_in)

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명:
    - `/` 경로에 접근하면 `index()` 뷰 함수가 실행됨
    - `user` 딕셔너리, `items` 리스트, `logged_in` 불리언 값을 `render_template()` 함수의 인자로 전달
        - 이 데이터는 `index.html` 템플릿에서 접근할 수 있음


#### 2.2.2 `templates/index.html` (Jinja2 템플릿 파일)

```html
<!--//file: "templates/index.html" -->
<!DOCTYPE html>
<html>
<head>
    <title>Jinja2 활용 예제</title>
</head>
<body>
    <h1>안녕하세요, { { user.username }}님!</h1>
    <p>나이는 { { user.age }}세 입니다.</p>

    <h2>오늘의 추천 과일:</h2>
    <ul>
        { % for item in items %}
            <li>{{ item }}</li>
        { % endfor %}
    </ul>

    { % if logged_in %}
        <p>환영합니다! 로그인 상태입니다.</p>
    { % else %}
        <p>로그인 해주세요.</p>
    { % endif %}

    <h2>사용자 정보 (매크로 활용):</h2>
    { % macro render_user(user_info) %}
        <p>이름: { { user_info.username }}</p>
        <p>나이: { { user_info.age }}</p>
    { % endmacro %}
    { { render_user(user) }}

    <h2>아이템 목록 (필터 활용):</h2>
    <p>아이템 개수: { { items | length }}</p>
    <p>첫 번째 아이템 (대문자): { { items[0] | upper }}</p>

    <h2>템플릿 상속</h2>
    { % extends 'base.html' %}

    { % block content %}
        <p>이 내용은 base.html의 'content' 블록을 상속받아 표시됩니다.</p>
    { % endblock %}
</body>
</html>
```

- 코드 설명:
    - <span style="color: green;">&#123;&#123; user.username &#125;&#125;</span>
        - 뷰 함수에서 전달된 `user` 딕셔너리의 `username` 키 값을 출력
        - 점(.) 표기법을 사용하여 딕셔너리 속성에 접근

    - <span style="color: green;">&#123;% for item in items %&#125; ... &#123;% endfor %&#125;</span>
        - 뷰 함수에서 전달된 `items` 리스트를 순회하며 각 항목을 `<li>` 태그로 출력

    - <span style="color: green;">&#123;% if logged_in %&#125; ... &#123;% else %&#125;` ... `&#123;% endif %&#125;</span>
        - 뷰 함수에서 전달된 `logged_in` 값에 따라 조건부로 다른 내용을 출력

    - <span style="color: green;">&#123;% macro render_user(user_info) %&#125; ... &#123;% endmacro %&#125;</span>
        - `render_user`라는 이름의 매크로 정의
        - 이 매크로는 사용자 정보를 받아 특정 형식으로 출력함
        - 매크로는 코드 재사용성을 높여줌

    - <span style="color: green;">&#123;&#123; render_user(user) &#125;&#125;</span>
        - 정의된 `render_user` 매크로를 호출하고 `user` 데이터를 인자로 전달

    - <span style="color: green;">&#123;&#123; items | length &#125;&#125;</span>
        - `items` 리스트에 `length` 필터를 적용하여 리스트의 길이를 출력
        - Jinja2는 다양한 내장 필터를 제공
            - `|` 기호를 사용하여 변수에 필터 적용

    - <span style="color: green;">&#123; items[0] | upper &#125;&#125;</span>
        - `items` 리스트의 첫 번째 요소에 `upper` 필터를 적용하여 대문자로 변환하여 출력함

    - <span style="color: green;">&#123;% extends 'base.html' %&#125;</span>
        - `base.html` 템플릿을 상속받음
        - 상속을 통해 웹 페이지의 공통적인 구조 (헤더, 푸터 등)를 재사용할 수 있음

    - <span style="color: green;">&#123;% block content %&#125; ... &#123;% endblock %&#125;</span>
        - `base.html`에서 정의된 `content` 블록을 오버라이드하여 현재 템플릿의 특정 내용을 삽입함


#### 2.2.3 `templates/base.html` (템플릿 상속 예시)

```html
<!--//file: "templates/base.html" -->
<!DOCTYPE html>
<html>
<head>
    <title>기본 템플릿</title>
</head>
<body>
    <header>
        <h1>웹사이트 제목</h1>
    </header>
    <main>
        { % block content %}
            <p>여기에 기본 내용이 표시됩니다.</p>
        { % endblock %}
    </main>
    <footer>
        <p>&copy; 2025 My Website</p>
    </footer>
</body>
</html>
```

- 코드 설명:
- `templates/base.html`:
    - 웹 페이지의 기본적인 HTML 구조를 정의함
    - <span style="color: green;">&#123;% block content %&#125; ... &#123;% endblock %&#125;</span>
        - 자식 템플릿에서 내용을 채울 수 있는 `content`라는 이름의 블록을 정의함

### 2.3 실행 방법

1. 가상환경 'my_app' 생성
2. 터미널에서 `my_app` 폴더로 이동
3. 가상환경 활성화
4. `pip install Flask` 명령어를 실행하여 Flask를 설치(Jinja2는 Flask 설치 시 함께 설치됨)
5. 프로젝트 구조대로 폴더와 파일 생성
6. `python app.py` 명령어를 실행하여 Flask 개발 서버 시작
7. 웹 브라우저에서 `http://127.0.0.1:5000/` 주소로 접속하여 스타일이 적용되고 JavaScript가 실행된 웹 페이지 확인

```bash
python -m venv my_app
cd my_app
source ./bin/activate
pip install Flask

python app.py
```