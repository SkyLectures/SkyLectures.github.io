---
layout: page
title:  "Flask에서의 정적 파일 관리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-03-02_04-StaticFiles
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 정적 파일 관리 개요

- 웹 프레임워크를 사용하여 웹 서비스를 개발할 때, CSS, JavaScript, 이미지, 폰트 등의 정적 파일들을 효율적으로 관리하는 것은 사용자 경험(UX) 향상과 웹 애플리케이션 성능 최적화에 매우 중요함

### 1.1 정적 파일의 역할

- CSS (Cascading Style Sheets): 웹 페이지의 스타일과 레이아웃을 정의하여 시각적인 표현 담당
- JavaScript: 웹 페이지에 동적인 기능과 상호작용성 부여
- 이미지: 웹 페이지의 시각적인 콘텐츠 제공
- 폰트: 웹 페이지의 텍스트 스타일 결정

### 1.2 Flask의 정적 파일 제공 방식

- Flask는 기본적으로 애플리케이션의 루트 디렉토리 내에 위치한 `static` 폴더를 통해 정적 파일을 제공함
- Flask 애플리케이션 객체가 생성될 때, `static_folder` 파라미터를 통해 다른 이름의 폴더나 여러 폴더를 정적 파일 제공 경로로 설정할 수 있음
- `url_for()` 함수를 사용하여 정적 파일의 URL을 생성
    - 이는 파일 경로가 변경되더라도 템플릿 코드의 수정 없이 유연하게 관리할 수 있도록 해줌

### 1.3 정적 파일 관리의 중요성

- 성능 향상
    - 브라우저는 정적 파일을 한 번 로드한 후에는 캐시에 저장함
        - subsequent 요청 시 서버에 다시 요청하지 않음
        - 효율적인 정적 파일 관리는 캐싱을 극대화하여 페이지 로딩 속도를 향상시킴

- 유지보수 용이성
    - 정적 파일을 별도의 폴더에서 관리하면 코드와 디자인 요소를 분리하여 유지보수가 용이해짐

- 보안
    - 정적 파일에 대한 접근 권한을 관리하여 보안을 강화할 수 있음

### 1.4 필요한 모듈 및 라이브러리

- Flask 자체에 정적 파일 제공 기능이 내장되어 있어 별도의 핵심 모듈이나 라이브러리가 필수적이지 않으나 개발 편의성 및 성능 최적화를 위해 다음과 같은 도구를 활용할 수 있음

- **Flask (필수)** 
    - 웹 프레임워크의 핵심
    - 정적 파일 제공 기능 내장

- **Werkzeug (Flask의 의존성)**
    - Flask의 기반이 되는 WSGI 유틸리티 라이브러리
    - URL 라우팅 및 요청/응답 처리를 담당함
    - 정적 파일 제공에도 일부 관여함

- **WhiteNoise (선택 사항, 배포 환경 권장)**
    - production 환경에서 정적 파일을 효율적으로 제공하기 위한 WSGI 미들웨어
    - Flask의 기본 정적 파일 제공 방식보다 성능이 뛰어남
    - CDN(Content Delivery Network)과 함께 사용하기 용이함

- **Flask-Assets (선택 사항, 개발 편의성 향상)**
    - CSS, JavaScript 파일을 번들링(bundling) 및 압축(minification)하여 로딩 속도 최적화
    - 개발 과정을 편리하게 해주는 Flask 확장 모듈

## 2. 실습 예제

### 2.1 프로젝트 구조

```text
my_app/
├── app.py         # Flask 애플리케이션 코드
├── templates/
│   └── index.html   # 템플릿 파일
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

### 2.2 프로젝트 파일

#### 2.2.1 `app.py` (Flask 애플리케이션 코드)

```python
#//file: "app.py"
from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

- 코드 설명
    - `app.py`: Flask 애플리케이션을 생성하고, 루트 경로 (`/`)에 대한 뷰 함수 `index()`를 정의함
    - `render_template('index.html')`: `templates` 폴더의 `index.html` 파일 렌더링함


#### 2.2.2 `templates/index.html` (템플릿 파일)

```html
<!--//file: "templates/index.html" -->
<!DOCTYPE html>
<html>
<head>
    <title>Flask 정적 파일 관리 예제</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <h1>안녕하세요!</h1>
    <p id="message"></p>
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
```

- 코드 설명
    - `templates/index.html`: HTML 템플릿 파일
        - `<link>` 태그
            - `href` 속성에서 `url_for('static', filename='css/style.css')` 함수를 사용하여 
            - `static` 폴더 내의 `css/style.css` 파일의 URL을 생성함
        - `<script>` 태그
            - `src` 속성에서 `url_for()` 함수를 사용하여 
            - `static` 폴더 내의 `js/script.js` 파일의 URL을 생성함


#### 2.2.3 `static/css/style.css` (CSS 파일)

```css
body {
    font-family: sans-serif;
    background-color: #f0f0f0;
    text-align: center;
    padding-top: 50px;
}

h1 {
    color: navy;
}
```

- 코드 설명
    - * `static/css/style.css`: 웹 페이지의 기본적인 스타일을 정의합니다.


#### 2.2.4 `static/js/script.js` (JavaScript 파일)

```javascript
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('message').textContent = 'JavaScript가 성공적으로 로드되었습니다!';
});
```

- 코드 설명
    - `static/js/script.js`: 문서가 로드되면 지정된 메시지를 `<p>` 태그에 표시하는 간단한 JavaScript 코드

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


### 2.4 추가적인 관리 방법 (선택 사항)

* **WhiteNoise 활용 (배포 환경):**
    ```python
    from flask import Flask, render_template, url_for
    from whitenoise import WhiteNoise

    app = Flask(__name__)
    app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

    @app.route('/')
    def index():
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug=True)
    ```
    - production 환경에서는 
        - WhiteNoise와 같은 WSGI 미들웨어를 사용하여 
        - 정적 파일 제공 성능을 향상시키는 것을 고려해볼 수 있음

- Flask-Assets 활용 (개발 편의성)

    ```python
    from flask import Flask, render_template, url_for
    from flask_assets import Environment, Bundle

    app = Flask(__name__)
    assets = Environment(app)

    css_bundle = Bundle('css/style.css', output='gen/packed.css')
    js_bundle = Bundle('js/script.js', output='gen/packed.js')

    assets.register('main_css', css_bundle)
    assets.register('main_js', js_bundle)

    @app.route('/')
    def index():
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug=True)
    ```

   - `templates/index.html`에서는 다음과 같이 번들링된 파일을 참조함

    ```html
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flask 정적 파일 관리 예제 (Flask-Assets)</title>
        { % assets 'main_css' %}
            <link rel="stylesheet" type="text/css" href="{ { ASSET_URL }}">
        { % endassets %}
    </head>
    <body>
        <h1>안녕하세요!</h1>
        <p id="message"></p>
        { % assets 'main_js' %}
            <script src="{ { ASSET_URL }}"></script>
        { % endassets %}
    </body>
    </html>
    ```
    - Flask-Assets는 
        - CSS 및 JavaScript 파일을 묶고 압축하여 
        - 네트워크 요청 수를 줄이고 
        - 파일 크기를 최적화하는 데 유용함
