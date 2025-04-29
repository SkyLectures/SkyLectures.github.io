---
layout: page
title:  "템플릿 엔진(Jinja2) 활용"
date:   2025-03-01 10:00:00 +0900
permalink: /material/S01-04-03-02_03-TemplateEngineJinja2
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 템플릿 엔진 개요

### 1.1 템플릿 엔진 (Template Engine)

- 웹 페이지의 HTML 구조와 동적으로 생성되는 데이터를 결합하여 최종적인 웹 페이지를 생성하는 도구

- **템플릿 엔진 사용 시 장점**

    - 표현 (Presentation)과 로직 (Logic)의 분리
        - HTML 코드는 템플릿 파일에
        - 애플리케이션 로직은 Python 코드에 분리하여
        - 개발 효율성과 유지보수성을 높임

    - 재사용성
        - 템플릿의 HTML 구조를 재사용하고, 
        - 필요한 데이터만 동적으로 삽입하여 
        - 효율적인 웹 페이지 관리가 가능함

    - 가독성
        - HTML 코드와 Python 코드가 분리되어 있어 
        - 웹 디자이너와 백엔드 개발자 간의 협업이 용이해짐

    - 보안
        - 템플릿 엔진은 잠재적인 보안 취약점을 방지하기 위한 기능을 제공함

### 1.2 Jinja2

- Python을 위한 현대적이고 디자이너 친화적인 템플릿 언어
- Flask와 완벽하게 통합되어 있음

- **주요 기능**

    - 변수
        - Python 코드를 통해 템플릿으로 전달된 데이터를 HTML 내에서 쉽게 출력할 수 있음 
            - <span style="color: green;">&#123;&#123; variable &#125;&#125;</span>

    - 제어 구조
        - **if**, **for** 등의 Python과 유사한 제어 구조를 사용하여 조건부 렌더링이나 반복적인 HTML 생성을 처리할 수 있음
            - <span style="color: green;">&#123;% if condition %&#125; ... &#123;% endif %&#125;</span>
            - <span style="color: green;">&#123;% for item in list %&#125; ... &#123;% endfor %&#125;</span>

    - 필터
        - 변수의 출력을 원하는 형식으로 변환할 수 있는 다양한 필터를 제공
        - <span style="color: green;">&#123;&#123; variable|upper &#125;&#125;</span>

    - 템플릿 상속
        - 공통 레이아웃을 정의하고, 하위 템플릿에서 특정 영역만 재정의하여 코드 중복을 줄이고 유지보수성을 높임
        - <span style="color: green;">&#123;% extends 'layout.html' %&#125;</span>
        - <span style="color: green;">&#123;% block content %&#125; ... &#123;% endblock %&#125;</span>
    - 매크로 
        - 자주 사용되는 HTML 구조를 함수 형태로 정의하여 재사용성을 높임
        - <span style="color: green;">&#123;% macro input(name, value='', type='text') %&#125; ... &#123;% endmacro %&#125;</span>

## 2. 구조

1.  **Flask 애플리케이션 객체 생성**
    - Flask 프레임워크를 사용하여 웹 애플리케이션 객체를 생성합니다.
2.  **라우팅 설정**
    - 특정 URL 경로에 접근했을 때 실행될 Python 함수 (뷰 함수)를 정의합니다.
3.  **데이터 전달**
    - 뷰 함수에서 템플릿으로 렌더링할 데이터를 Python 딕셔너리 형태로 준비합니다.
4.  **템플릿 렌더링**
    - `render_template()` 함수를 사용하여 템플릿 파일과 전달할 데이터를 결합하여 최종 HTML 코드를 생성하고 클라이언트에 응답합니다.
5.  **템플릿 파일 작성**
    - `.html` 확장자를 가진 Jinja2 템플릿 파일을 `templates` 폴더에 저장합니다. 이 파일에는 HTML 구조와 Jinja2 템플릿 언어 문법이 사용됩니다.

### 3. 프로세스

Flask 애플리케이션에서 Jinja2 템플릿이 렌더링되는 일반적인 프로세스는 다음과 같습니다.

1.  **클라이언트 요청**
    - 웹 브라우저와 같은 클라이언트가 특정 URL로 HTTP 요청을 보냄

2.  **라우팅** 
    - Flask 애플리케이션은 요청된 URL과 매칭되는 뷰 함수를 검색

3.  **뷰 함수 실행** 
    - 매칭된 뷰 함수 실행

4.  **데이터 준비** 
    - 뷰 함수 내에서 템플릿에 전달할 동적인 데이터를 Python 자료형 (주로 딕셔너리)으로 준비

5.  **템플릿 렌더링 요청** 
    - 뷰 함수는 **render_template()** 함수를 호출하면서 템플릿 파일 이름과 전달할 데이터를 인자로 전달

6.  **Jinja2 엔진** 
    - Flask는 Jinja2 템플릿 엔진에게 지정된 템플릿 파일과 전달받은 데이터를 전달

7.  **템플릿 파싱 및 평가** 
    - Jinja2 엔진은
        - 템플릿 파일을 파싱하고,
        - 템플릿 언어 문법 <span style="color: green;">&#123;&#123; ... &#125;&#125;, &#123;% ... %&#125;</span>을 해석하여
        - Python 코드를 통해 전달된 데이터를 HTML 구조에 삽입하거나,
        - 조건부 로직 또는 반복문 실행

8.  **HTML 생성** 
    - Jinja2 엔진은 데이터가 결합된 최종 HTML 코드를 생성함

9.  **응답** 
    - Flask는 생성된 HTML 코드를 HTTP 응답으로 클라이언트에게 전송

10. **클라이언트 렌더링** 
    - 클라이언트 (웹 브라우저)는 수신된 HTML 코드를 해석하여 웹 페이지를 화면에 표시


### 4. 코드 예제

- Flask 웹 프레임워크에서 Jinja2 템플릿 엔진을 활용하는 간단한 예제 코드

**1. Flask 애플리케이션 파일 (`app.py`):**

```python
#//file: "app.py"

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    name = "Alice"
    items = ["Apple", "Banana", "Cherry"]
    user = {'username': 'bob123', 'email': 'bob@example.com'}
    return render_template('index.html', user_name=name, item_list=items, user_info=user)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
```

**코드 설명:**

- `from flask import Flask, render_template`
    - Flask 프레임워크와 템플릿 렌더링을 위한 `render_template` 함수를 임포트

- `app = Flask(__name__)`
    - Flask 애플리케이션 객체 생성

- `@app.route('/')`
    - 루트 URL (`/`)에 접근했을 때 `index()` 함수를 실행하도록 라우팅 설정

- `index()` 함수
    - `name`, `items`, `user` 변수에 템플릿으로 전달할 데이터를 저장
    - `render_template('index.html', user_name=name, item_list=items, user_info=user)`
        - `templates` 폴더에 있는 `index.html` 템플릿 파일을 렌더링하고, 
        - `user_name`, `item_list`, `user_info`라는 이름으로 데이터를 템플릿에 전달

- `@app.route('/about')`
    - `/about` URL에 접근했을 때 `about()` 함수를 실행하도록 라우팅 설정

- `about()` 함수
    - `about.html` 템플릿 파일을 렌더링
    
- `if __name__ == '__main__':`
    - 스크립트가 직접 실행될 때 Flask 개발 서버를 시작함
    - `debug=True`는 디버그 모드를 활성화하여 개발 중 오류를 쉽게 확인할 수 있도록 함

**2. 템플릿 파일 (`templates/index.html`):**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Flask Jinja2 Example</title>
</head>
<body>
    <h1>Welcome, {{ user_name }}!</h1>

    <h2>Item List:</h2>
    <ul>
        {% for item in item_list %}
            <li>{{ item }}</li>
        {% endfor %}
    </ul>

    <h2>User Information:</h2>
    <p>Username: {{ user_info.username|upper }}</p>
    <p>Email: {{ user_info.email }}</p>

    {% if user_info.email.endswith('@example.com') %}
        <p>This user is from example.com.</p>
    {% else %}
        <p>This user is from a different domain.</p>
    {% endif %}

    <p><a href="/about">About Page</a></p>
</body>
</html>
```

**설명:**

* `{{ user_name }}`: 뷰 함수에서 `user_name`이라는 이름으로 전달된 변수의 값을 HTML 내에 출력합니다.
* `{% for item in item_list %}` ... `{% endfor %}`: 뷰 함수에서 `item_list`라는 이름으로 전달된 리스트의 각 항목을 반복하여 `<li>` 태그 내에 출력합니다.
* `{{ user_info.username|upper }}`: 뷰 함수에서 `user_info`라는 이름으로 전달된 딕셔너리의 `username` 키에 접근하여 값을 출력하고, `upper` 필터를 적용하여 대문자로 변환합니다.
* `{{ user_info.email }}`: 뷰 함수에서 `user_info` 딕셔너리의 `email` 키 값을 출력합니다.
* `{% if user_info.email.endswith('@example.com') %}` ... `{% else %}` ... `{% endif %}`: 뷰 함수에서 전달된 `user_info.email` 값에 따라 조건부로 다른 내용을 출력합니다.
* `<a href="/about">About Page</a>`: `/about` URL로 이동하는 링크를 생성합니다.

**3. 템플릿 파일 (`templates/about.html`):**

```html
<!DOCTYPE html>
<html>
<head>
    <title>About Us</title>
</head>
<body>
    <h1>About This Application</h1>
    <p>This is a simple example demonstrating the use of Flask and Jinja2.</p>
    <p><a href="/">Go Back to Home</a></p>
</body>
</html>
```

**실행 방법:**

1.  위의 `app.py`, `templates/index.html`, `templates/about.html` 파일을 동일한 프로젝트 폴더 구조로 저장합니다 (`templates` 폴더를 `app.py`와 같은 레벨에 생성해야 합니다).
2.  터미널 또는 명령 프롬프트에서 해당 프로젝트 폴더로 이동합니다.
3.  `pip install Flask` 명령어를 실행하여 Flask를 설치합니다 (이미 설치되어 있다면 생략).
4.  `python app.py` 명령어를 실행하여 Flask 개발 서버를 시작합니다.
5.  웹 브라우저에서 `http://127.0.0.1:5000/` 또는 `http://localhost:5000/` 주소로 접속하면 `index.html` 페이지가 렌더링된 것을 확인할 수 있습니다.
6.  `http://127.0.0.1:5000/about` 주소로 접속하면 `about.html` 페이지가 렌더링된 것을 확인할 수 있습니다.

이 예제를 통해 Flask 웹 프레임워크에서 Jinja2 템플릿 엔진이 어떻게 작동하는지 기본적인 개념, 구조, 프로세스, 그리고 실제 코드를 통해 이해할 수 있습니다. Jinja2의 더 많은 기능 (템플릿 상속, 매크로, 필터 등)을 학습하면 더욱 강력하고 효율적인 웹 애플리케이션 개발이 가능해집니다.