---
layout: page
title:  "Flask 기반 웹서비스 기본 흐름"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-04-03-02_01-FlaskBasedServiceProcess
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Flask의 작동 방식 (기본 개념)

1. **요청 (Request)**
    - 사용자의 웹 브라우저가 서버로 보내는 정보 (URL, HTTP 메서드, 데이터 등)를 담고 있는 객체
    - `request` 객체를 통해 이 정보에 접근할 수 있도록 기능을 제공함

2. **라우팅 (Routing)**
    - 사용자의 웹 브라우저 요청 URL과 파이썬 함수를 연결하는 과정
    - `@app.route('/경로')` 데코레이터를 사용하여 특정 URL에 접근했을 때 실행될 함수를 정의함

3. **뷰 함수 (View Function)**
    - 라우팅된 URL에 대한 실제 로직을 처리하는 파이썬 함수
    - 요청을 처리하고, 필요한 데이터를 가져오거나 조작함
    - 최종적으로 사용자에게 보여줄 응답 (HTML, JSON 등)을 생성하여 반환함

4. **모델 (Models)**
    - ORM 등을 사용하여 데이터베이스와 상호 작용하는 객체

5. **템플릿 (Templates)**
    - 동적인 웹 페이지를 생성하기 위한 파일
    - Jinja2 템플릿 엔진을 사용하여 파이썬 변수와 제어 구조를 HTML 문서 내에 삽입할 수 있음

6. **컨텍스트 (Context)**
    - Flask 애플리케이션과 요청에 대한 정보를 담고 있는 객체
    - 뷰 함수 내에서 `request`, `session`, `g` 등의 객체에 접근할 수 있도록 함

7. **응답 (Response)**
    - 서버가 사용자의 요청에 대한 처리 결과를 담아 웹 브라우저로 보내는 정보
    - 뷰 함수는 문자열, 템플릿 렌더링 결과, JSON 데이터 등을 응답으로 반환할 수 있음

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-02_01-001.png' width="700"/></p>

    Flask 기반 애플리케이션의 전체적인 프로세스
    {:.figcaption}

## 2. Flask 기반의 간단한 웹 서버

- Flask 웹 서버의 기본적인 구조와 라우팅 기능을 보여주는 간단한 예시

    ```python
    #//file: "hello.py"
from flask import Flask

    # Flask 애플리케이션 객체 생성
    app = Flask(__name__)

    # 라우팅 규칙 정의: '/' 경로에 접근했을 때 실행될 함수
    @app.route('/')
    def hello_world():
        return '안녕하세요, Flask!'

    # 라우팅 규칙 정의: '/user/<username>' 경로에 접근했을 때 실행될 함수
    @app.route('/user/<username>')
    def show_user_profile(username):
        # username 변수를 사용하여 사용자 프로필 페이지를 보여줄 수 있습니다.
        return f'사용자 이름: {username}'

    # 서버를 실행하는 코드
    if __name__ == '__main__':
        app.run(debug=True)
    ```

- **코드 설명**:

    - <span style="color: #090">app = Flask(__name__)</span>
        - Flask 애플리케이션 객체 생성
        - `__name__`은 현재 모듈의 이름을 나타냄

    -  <span style="color: #090">@app.route('/')</span>
        - 데코레이터(Decorator)
        - 바로 아래에 정의된 함수 (`hello_world`)를 웹 서버의 특정 URL 경로('/')와 연결함
        - 사용자가 웹 브라우저에서 `http://<서버 주소>/`에 접속하면 `hello_world` 함수가 실행됨

    -  <span style="color: #090">def hello_world():</span>
        - `/` 경로에 접근했을 때 실행될 함수
        - 단순히 문자열 `'안녕하세요, Flask!'`를 반환
        - 이 문자열이 웹 브라우저에 표시됨

    -  <span style="color: #090">@app.route('/user/<username>')</span>
        - 또 다른 라우팅 규칙 정의
        - `<username>`: *동적 URL 변수*
        - 사용자가 `http://<서버 주소>/user/john` 또는 `http://<서버 주소>/user/jane`과 같이 `/user/` 뒤에 어떤 값을 입력하면, 그 값이 `username` 변수에 전달되어 `show_user_profile` 함수에서 사용할 수 있음
        
    -  <span style="color: #090">def show_user_profile(username):</span>
        - `/user/<username>` 경로에 접근했을 때 실행될 함수
        - 전달받은 `username`을 사용하여 사용자 이름을 표시하는 문자열 반환

    -  <span style="color: #090">if __name__ == '__main__':</span>
        - 스크립트가 직접 실행될 때만 아래의 코드를 실행하도록 함
        - `__main__`이 아닌 경우는 모듈로서 임포트되어 사용되는 코드임

    -  <span style="color: #090">app.run(debug=True)</span>
        - Flask 개발 서버를 시작함
        - `debug=True`
            - 개발 모드 활성화
            - 개발 모드에서는 코드 변경 사항이 자동으로 서버에 반영되고, 오류 발생 시 자세한 디버그 정보를 웹 페이지에서 확인할 수 있음
            - 실제 운영 환경에서는 `debug=False`로 설정

- **실행 방법**:

    1.  서버 실행: 터미널 또는 명령 프롬프트에서 해당 파일이 있는 디렉토리로 이동한 후 다음 명령을 실행

        ```bash
        python app.py
        ```

    2.  웹 브라우저 접속
        - 웹 브라우저를 열고 `http://127.0.0.1:5000/` 또는 `http://localhost:5000/`에 접속
            - "안녕하세요, Flask!" 메시지 확인
        - `http://127.0.0.1:5000/user/yourname`과 같이 `/user/` 뒤에 원하는 이름을 붙여 접속
            - "사용자 이름: yourname"과 같은 메시지 확인