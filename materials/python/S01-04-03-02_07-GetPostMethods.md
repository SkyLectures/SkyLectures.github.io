---
layout: page
title:  "GET / POST 메서드 활용"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-03-02_07-GetPostMethods
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


- Flask 웹 프레임워크를 사용하여 웹 서비스를 개발할 때, 
    - 클라이언트(웹 브라우저 또는 다른 애플리케이션)와 서버 간의 통신은 HTTP (Hypertext Transfer Protocol) 메서드를 통해 이루어짐
    - 가장 기본적이고 중요한 메서드가 **GET**과 **POST**
    - 이 두 메서드의 이해와 적절한 활용은 웹 애플리케이션의 기능 구현 및 효율적인 데이터 처리에 필수적임

## 1. GET / POST 메서드 활용 개요

### 1.1 HTTP 메서드의 종류와 역할

- HTTP 메서드
    - 클라이언트가 서버에게 수행하고자 하는 액션을 명시적으로 나타냄
    - GET, POST 외에도 PUT, DELETE, PATCH 등 다양한 메서드가 존재
    - 웹 개발의 기본적인 흐름에서는 GET과 POST가 가장 많이 사용됨

    - **GET 메서드**
        - 목적
            - 서버로부터 특정 리소스를 **요청**하고 **조회**하는 데 사용

        - 특징
            - 요청 데이터는 URL의 쿼리 파라미터 형태로 서버에 전달 (예: `/search?keyword=flask&page=1`).
            - URL에 데이터가 노출되므로 민감한 정보를 전달하는 데는 부적절함
            - 서버 상태를 변경하지 않는 **멱등성(Idempotent)을 가짐**
                - 즉, 동일한 GET 요청을 여러 번 보내도 서버의 상태는 동일하게 유지됨
            - 브라우저는 GET 요청 결과를 캐싱할 수 있어 성능 향상에 유리함
            - 전송할 수 있는 데이터 길이에 제한이 있을 수 있음

        - Flask에서의 활용
            - 주로 데이터를 조회하여 웹 페이지에 표시하거나, 
            - API에서 특정 리소스를 가져올 때 사용됨
            - `request.args` 객체를 통해 URL 파라미터에 접근

    - **POST 메서드:**
        - 목적
            - 서버에 데이터를 **전송**하여 새로운 리소스를 **생성**하거나, 
            - 서버의 상태를 **변경**하는 데 사용

        - 특징
            - 요청 데이터는 HTTP 요청 본문에 담겨서 서버에 전달됨
                - URL에는 데이터가 노출되지 않아 GET 방식보다 보안에 유리함
            - 서버 상태를 변경할 수 있으므로 **멱등성을 가지지 않음**
                - 동일한 POST 요청을 여러 번 보내면 서버에 여러 개의 동일한 리소스가 생성될 수 있음
            - 전송할 수 있는 데이터 길이에 비교적 제한이 없음
            - 브라우저는 POST 요청 결과를 일반적으로 캐싱하지 않음

        - Flask에서의 활용
            - 주로 HTML Form을 통해 사용자가 입력한 데이터를 서버에 전송하여 처리하거나, 
            - API에서 새로운 리소스를 생성할 때 사용됨
            - `request.form` 객체를 통해 Form 데이터에 접근
            - `request.get_json()` 등을 통해 JSON 형태의 요청 본문 데이터에 접근

### 1.2 필요한 모듈 및 라이브러리

- Flask (필수)
    - 웹 프레임워크의 핵심
    - `request` 객체를 통해 클라이언트 요청 정보를 접근하고, 라우팅 기능을 제공
    - `render_template()` 함수 등을 사용하여 응답을 생성

## 2. 실습 예제 코드

### 2.1 프로젝트 구조

```
my_app/
├── app.py
└── templates/
    ├── get_form.html
    ├── post_form.html
    └── result.html
```

### 2.2 프로젝트 코드

- **`app.py` (Flask 애플리케이션 코드)**

```python
#//file: "app.py"
from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.route('/get_form')
def get_form():
    return render_template('get_form.html')

@app.route('/process_get', methods=['GET'])
def process_get():
    if request.method == 'GET':
        name = request.args.get('name')
        age = request.args.get('age')
        return render_template('result.html', method='GET', name=name, age=age)
    return "잘못된 접근입니다."

@app.route('/post_form')
def post_form():
    return render_template('post_form.html')

@app.route('/process_post', methods=['POST'])
def process_post():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        return render_template('result.html', method='POST', name=name, age=age)
    return "잘못된 접근입니다."

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `/get_form`
        - GET 요청을 처리하여 `get_form.html` 템플릿을 렌더링
    - `/process_get`
        - GET 메서드만 허용
        - (`methods=['GET']`). `request.args.get()`을 사용하여 
            - URL 파라미터 (`name`, `age`) 값을 추출하고, 
            - 결과를 `result.html` 템플릿에 전달
    - `/post_form`
        - GET 요청을 처리하여 `post_form.html` 템플릿을 렌더링
    - `/process_post`
        - POST** 메서드만 허용
        - (`methods=['POST']`). `request.form['name']` 및 `request.form['age']`를 사용하여 
            - Form 데이터를 추출하고, 
            - 결과를 `result.html` 템플릿에 전달

- **`templates/get_form.html` (GET 방식 Form 템플릿)**

```html
<!--//file: "templates/get_form.html"-->
<!DOCTYPE html>
<html>
<head>
    <title>GET 방식 Form</title>
</head>
<body>
    <h1>GET 방식으로 정보 보내기</h1>
    <form action="{ { url_for('process_get') }}" method="GET">
        <div>
            <label for="name">이름:</label>
            <input type="text" id="name" name="name" required>
        </div>
        <div>
            <label for="age">나이:</label>
            <input type="number" id="age" name="age" required min="0" max="150">
        </div>
        <button type="submit">제출 (GET)</button>
    </form>
</body>
</html>
```

- 코드 설명
    - `templates/get_form.html`
        - Form 태그의 `method` 속성이 `GET`으로 설정되어 있음
        - 데이터를 URL 파라미터 형태로 서버에 전송


- **`templates/post_form.html` (POST 방식 Form 템플릿)**

```html
<!DOCTYPE html>
<html>
<head>
    <title>POST 방식 Form</title>
</head>
<body>
    <h1>POST 방식으로 정보 보내기</h1>
    <form action="{ { url_for('process_post') }}" method="POST">
        <div>
            <label for="name">이름:</label>
            <input type="text" id="name" name="name" required>
        </div>
        <div>
            <label for="age">나이:</label>
            <input type="number" id="age" name="age" required min="0" max="150">
        </div>
        <button type="submit">제출 (POST)</button>
    </form>
</body>
</html>
```

- 코드 설명
    - `templates/post_form.html`
        - Form 태그의 `method` 속성이 `POST`로 설정되어 있음
        - 데이터를 HTTP 요청 본문에 담아 서버에 전송

- **`templates/result.html` (결과 템플릿)**

```html
<!--//file: "templates/result.html"-->
<!DOCTYPE html>
<html>
<head>
    <title>처리 결과</title>
</head>
<body>
    <h1>처리 결과</h1>
    <p>전송 방식: { { method }}</p>
    <p>이름: { { name }}</p>
    <p>나이: { { age }}</p>
    <a href="{ { url_for('get_form') }}">GET Form으로 돌아가기</a> |
    <a href="{ { url_for('post_form') }}">POST Form으로 돌아가기</a>
</body>
</html>
```

- 코드 설명
    - `templates/result.html`
        - 서버로부터 받은 전송 방식 (`method`), 이름 (`name`), 나이 (`age`)를 표시

### 2.3 실행 방법

5.  웹 브라우저에서 `http://127.0.0.1:5000/get_form` 또는 `http://127.0.0.1:5000/post_form` 주소로 접속하여 각각의 Form을 확인하고 데이터를 제출하여 결과를 확인합니다.


1. 가상환경 'my_app' 생성
2. 터미널에서 `my_app` 폴더로 이동
3. 가상환경 활성화
4. `pip install Flask` 명령어를 실행하여 Flask를 설치
5. 프로젝트 구조대로 폴더와 파일 생성
6. `python app.py` 명령어를 실행하여 Flask 개발 서버 시작
7. 웹 브라우저에서 `http://127.0.0.1:5000/get_form` 또는 `http://127.0.0.1:5000/post_form`  주소로 접속하여 각각의 Form을 확인하고 데이터를 제출하여 결과를 확인

```bash
python -m venv my_app
cd my_app
source ./bin/activate
pip install Flask

python app.py
```
