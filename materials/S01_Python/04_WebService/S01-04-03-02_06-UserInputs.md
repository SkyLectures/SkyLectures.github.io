---
layout: page
title:  "사용자 입력처리(Form 및 API)"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-03-02_06-UserInputs
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


- Flask에서 사용자로부터 데이터를 입력받고 처리하는 방법은 **Form**을 이용한 방식과 **API(RESTful API)**를 이용한 방식으로 구분됨

## 1. Form을 이용한 사용자 입력 처리

### 1.1 기본 구성

- **HTML Form** 
    - 웹 페이지에서 사용자로부터 텍스트, 선택, 파일 등의 입력을 받을 수 있는 표준 HTML 요소
    - `<form>` 태그 내에 `<input>`, `<textarea>`, `<select>` 등의 입력 필드를 포함

- **HTTP Method (GET, POST)** 
    - Form 데이터를 서버로 전송하는 방식을 결정
        - GET 
            - 데이터를 URL 파라미터 형태로 전송
            - 주로 데이터를 조회할 때 사용
            - 보안에 취약하고 전송 데이터 길이에 제한이 있음

        - POST 
            - 데이터를 HTTP 요청 본문에 담아 전송
            - 주로 데이터를 생성, 수정, 삭제할 때 사용
            - GET 방식보다 보안에 유리하고 더 많은 데이터를 전송할 수 있음

### 1.2 Flask의 Form 처리

- Flask는 
    - `request` 객체를 통해 클라이언트로부터 전송된 Form 데이터를 접근할 수 있도록 제공함
        - `request.form`: POST 또는 PUT 방식으로 전송된 Form 데이터를 담고 있는 딕셔너리 형태의 객체
        - `request.args`: GET 방식으로 전송된 URL 파라미터 데이터를 담고 있는 딕셔너리 형태의 객체

- WTForms (선택 사항, 권장)
    - Form 처리를 더욱 편리하고 안전하게 만들어주는 Flask 확장 기능
    - Form 클래스를 정의하고 유효성 검사를 쉽게 구현할 수 있도록 도와줌

### 1.3 필요한 모듈 및 라이브러리

- Flask (필수)
    - 웹 프레임워크의 핵심
    - `request` 객체 제공

- WTForms (선택 사항, 권장)
    - Form 정의 및 유효성 검사를 위한 라이브러리

- Flask-WTF (WTForms와 Flask 통합)
    - WTForms를 Flask 애플리케이션에서 쉽게 사용할 수 있도록 해주는 확장 기능
    - CSRF(Cross-Site Request Forgery) 보호 기능 제공


## 2. 실습 코드(Form)

### 2.1 프로젝트 구조

```
my_app/
├── app.py
└── templates/
    ├── form.html
    └── result.html
```

### 2.2 프로젝트 코드

#### 2.2.1 WTForms 사용 예시

- `app.py` 

```python
#//file: "app.py"
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # CSRF 보호를 위한 Secret Key 설정

class InputForm(FlaskForm):
    name = StringField('이름', validators=[DataRequired('이름을 입력해주세요.')])
    age = IntegerField('나이', validators=[DataRequired('나이를 입력해주세요.'), NumberRange(min=0, max=150, message='나이는 0에서 150 사이여야 합니다.')])
    submit = SubmitField('제출')

@app.route('/form', methods=['GET', 'POST'])
def input_form():
    form = InputForm()
    if form.validate_on_submit():
        name = form.name.data
        age = form.age.data
        return render_template('result.html', name=name, age=age)
    return render_template('form.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `InputForm` 클래스를 정의하여 Form 필드 (`name`, `age`, `submit`)와 유효성 검사 규칙 (`DataRequired`, `NumberRange`)을 설정

    - `/form` 라우트에서 
        - GET 요청 시
            - Form 객체를 생성하여 템플릿에 전달
        - POST 요청 시 
            - `form.validate_on_submit()`으로 유효성 검사를 수행
            - 유효성 검사를 통과하면 결과를 `result.html`에 렌더링

- `templates/form.html` (Form 템플릿)

```html
<!--//file: "templates/form.html"-->
<!DOCTYPE html>
<html>
<head>
    <title>사용자 입력 Form</title>
</head>
<body>
    <h1>사용자 정보 입력</h1>
    <form method="POST" action="{ { url_for('input_form') }}">
        { { form.csrf_token }}
        <div>
            { { form.name.label }} { { form.name() }}
            { % if form.name.errors %}
                <ul class="errors">
                    { % for error in form.name.errors %}
                        <li>{{ error }}</li>
                    { % endfor %}
                </ul>
            { % endif %}
        </div>
        <div>
            { { form.age.label }} { { form.age() }}
            { % if form.age.errors %}
                <ul class="errors">
                    { % for error in form.age.errors %}
                        <li>{ { error }}</li>
                    { % endfor %}
                </ul>
            { % endif %}
        </div>
        { { form.submit() }}
    </form>
</body>
</html>
```

- `templates/result.html` (결과 템플릿)

```html
<!--//file: "templates/result.html"-->
<!DOCTYPE html>
<html>
<head>
    <title>입력 결과</title>
</head>
<body>
    <h1>입력하신 정보</h1>
    <p>이름: { { name }}</p>
    <p>나이: { { age }}</p>
    <a href="{ { url_for('input_form') }}">다시 입력하기</a>
</body>
</html>
```

- 코드 설명
    - `form.html` 템플릿: Form 필드와 에러 메시지를 Jinja2를 사용하여 출력
    - `{{ form.csrf_token }}`: CSRF 공격을 방지하기 위한 숨겨진 필드를 생성

#### 2.2.2 `request` 객체 사용 예시

- `app.py`

```python
#//file: "app.py"
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/basic_form', methods=['GET', 'POST'])
def basic_input_form():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        return render_template('result.html', name=name, age=age)
    return render_template('basic_form.html')

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `/basic_form` 라우트에서 POST 요청 시 
        - `request.form` 딕셔너리에서 `name`과 `age` 값을 추출하여 `result.html`에 렌더링

-`templates/basic_form.html` (기본 Form 템플릿)

```html
<!--//file: "templates/basic_form.html"-->
<!DOCTYPE html>
<html>
<head>
    <title>기본 사용자 입력 Form</title>
</head>
<body>
    <h1>기본 사용자 정보 입력</h1>
    <form method="POST" action="{{ url_for('basic_input_form') }}">
        <div>
            <label for="name">이름:</label>
            <input type="text" id="name" name="name" required>
        </div>
        <div>
            <label for="age">나이:</label>
            <input type="number" id="age" name="age" required min="0" max="150">
        </div>
        <button type="submit">제출</button>
    </form>
</body>
</html>
```

- 코드 설명
    - `basic_form.html`
        - 기본적인 HTML Form 구조를 가짐

## 3. API(RESTful API)를 이용한 사용자 입력 처리

### 3.1 기본 구성

- **API (Application Programming Interface)** 
    - 서로 다른 소프트웨어 시스템 간의 상호작용을 위한 규칙과 인터페이스 정의
    - 웹 API는 HTTP 프로토콜을 사용하여 데이터를 주고받음

- **HTTP Method (POST, PUT, DELETE 등)** 
    - API 요청의 목적을 나타냄
        - POST: 데이터를 생성할 때
        - PUT: 데이터를 수정할 때
        - DELETE: 데이터를 삭제할 때

- **데이터 형식 (JSON, XML 등)**
    - API를 통해 주고받는 데이터의 형식 정의
        - JSON (JavaScript Object Notation)은 가볍고 널리 사용되는 형식

### 3.2 Flask의 API 처리

- Flask는 `request` 객체를 통해 API 요청 데이터를 접근할 수 있도록 제공함
    - `request.get_json()`
        - 요청 본문의 데이터가 JSON 형식일 경우 파싱하여 Python 딕셔너리 형태로 반환

    - `jsonify()`
        - Python 딕셔너리 등의 데이터를 JSON 형식의 응답으로 변환하여 반환하는 Flask 유틸리티 함수

- Flask-RESTful (선택 사항, API 개발 편의성 향상)
    - RESTful API 개발을 위한 Flask 확장 기능
    - 리소스 기반의 API 설계를 용이하게 함
    - 요청 파싱 및 응답 포맷팅 등을 편리하게 처리할 수 있도록 도와줌

### 3.3 필요한 모듈 및 라이브러리

- **Flask (필수)** 
    - 웹 프레임워크의 핵심
    - `request` 객체와 `jsonify()` 함수를 제공

- **Flask-RESTful (선택 사항, API 개발 편의성 향상)
    - RESTful API 개발을 위한 확장 기능


## 4. 실습 코드(RESTful API)

### 4.1 프로젝트 구조

```
my_app/
└── app.py
```

### 4.2 프로젝트 코드

#### 4.2.1 `app.py` (Flask 기본 API 처리)

```python
#//file: "app.py"
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/user', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data or 'age' not in data:
        return jsonify({'error': '이름과 나이를 포함한 JSON 데이터를 보내주세요.'}), 400

    name = data['name']
    age = data['age']
    # 데이터베이스 저장 등의 로직 처리 (여기서는 생략)
    return jsonify({'message': f'{name}님, 환영합니다! 나이는 {age}세입니다.'}), 201

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    # 데이터베이스에서 user_id에 해당하는 사용자 정보 조회 (여기서는 임의 데이터 사용)
    users = {
        1: {'name': 'Bob', 'age': 25},
        2: {'name': 'Charlie', 'age': 35}
    }
    if user_id in users:
        return jsonify(users[user_id]), 200
    return jsonify({'error': '사용자를 찾을 수 없습니다.'}), 404

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `/api/user` POST 요청 시 
        - `request.get_json()`으로 요청 본문의 JSON 데이터를 파싱함
        - 필수 필드 (`name`, `age`)의 유무 확인
        - 데이터를 처리한 후 JSON 응답을 반환
            - HTTP 상태 코드 201 (Created)을 함께 전송

    - `/api/users/<int:user_id>` GET 요청 시
        - URL 경로에서 `user_id`를 추출하여 해당 사용자 정보를 조회하고
        - JSON 응답을 반환함
        - 사용자가 없을 경우
            -  HTTP 상태 코드 404 (Not Found)를 전송
            - `jsonify()` 함수를 사용하여 Python 딕셔너리를 JSON 형식으로 변환

#### 4.2.3 `app.py` (Flask-RESTful 사용)

```python
#//file: "app.py"
from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('name', type=str, required=True, help='이름을 입력해주세요.')
parser.add_argument('age', type=int, required=True, help='나이를 입력해주세요.')

class User(Resource):
    def post(self):
        args = parser.parse_args()
        name = args['name']
        age = args['age']
        # 데이터베이스 저장 등의 로직 처리 (여기서는 생략)
        return {'message': f'{name}님, 환영합니다! 나이는 {age}세입니다.'}, 201

    def get(self, user_id):
        # 데이터베이스에서 user_id에 해당하는 사용자 정보 조회 (여기서는 임의 데이터 사용)
        users = {
            1: {'name': 'Bob', 'age': 25},
            2: {'name': 'Charlie', 'age': 35}
        }
        if user_id in users:
            return users[user_id], 200
        return {'error': '사용자를 찾을 수 없습니다.'}, 404

api.add_resource(User, '/api/user', '/api/users/<int:user_id>')

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `reqparse.RequestParser`
        - 요청 파라미터를 정의하고 
        - 유효성 검사 규칙을 설정

    - `User` 클래스
        - `Resource`를 상속받아 API 엔드포인트를 정의함
        - `post()` 메서드: POST 요청 처리
        - `get()` 메서드: GET 요청 처리

    - `api.add_resource()`
        - `User` 리소스를 URL 경로와 연결

- Flask-RESTful은 요청 파싱, 응답 포맷팅 등을 자동으로 처리하여 API 개발을 더욱 편리하게 만들어줌

### 4.3 테스트 방법

- API 엔드포인트
    - 웹 브라우저 주소창에서 직접 접근하는 GET 요청을 테스트
    - POST, PUT, DELETE 등의 요청은 `curl`, `Postman` 등의 HTTP 클라이언트 도구를 사용하여 테스트
