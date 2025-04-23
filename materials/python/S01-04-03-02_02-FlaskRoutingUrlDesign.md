---
layout: page
title:  "Flask의 라우팅 및 URL 설계"
date:   2025-03-01 10:00:00 +0900
permalink: /material/S01-04-03-02_02-FlaskRoutingUrlDesign
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 라우팅(Routing)

### 1.1 라우팅의 개념

- 어떤 네트워크 안에서 통신 데이터를 보낼 때 최적의 경로를 선택하는 과정
    - 최적의 경로: 주어진 데이터를 가장 짧은 거리로, 또는 가장 적은 시간 안에 전송할 수 있는 경로

- 라우팅 과정은 보통 다양한 네트워크 목적지에 대한 기록을 관리하는 라우팅 테이블을 기초로 하여 수행됨
    - 그러므로, 라우터의 메모리에 기록된 라우팅 테이블의 구성은 효과적인 라우팅에 매우 중요함

- 대부분의 라우팅 알고리즘은 한번에 한가지 네트워크 경로를 사용하지만, 다중 경로 라우팅 기술은 다양한 대체 경로의 사용을 가능하게 함 

- 현대의 웹 어플리케이션은 잘 구조화된 URL로 구성되어 있으며, 이것으로 사람들은 URL을 쉽게 기억할 수 있고, 열악한 네트워크 연결 상황 하의 기기들에서 동작하는 어플리케이션에서도 사용하기 좋음

### 1.2 라우팅의 중요성

- 라우팅은 네트워크 통신의 효율성을 높임

- 네트워크 통신 장애가 발생하면 웹 사이트 페이지가 로드될 때까지 사용자가 기다리는 시간이 길어지며, 또한 웹 사이트 서버에서 많은 수의 사용자를 처리하지 못해 서버의 작동이 중단될 수 있음

- 라우팅은 네트워크가 정체 없이 최대한 많은 용량을 사용할 수 있도록 데이터 트래픽을 관리함으로써, 네트워크 장애를 최소화함

## 2. URL 라우팅 설계

### 2.1 URL 라우팅의 개념

- 웹 애플리케이션 내의 특정 기능이나 리소스에 URL을 매핑(바인딩)하는 것을 포함하는 웹 개발의 기본 개념
- 들어오는 요청이 어떻게 처리되고 어떤 뷰 함수에 의해 처리되는지를 결정하는 방법
- 라우팅은 요청을 수락하고 이를 처리하고 원하는 응답을 생성할 적절한 뷰 함수로 안내하는 것

### 2.2 Flask에서의 기본 URL 라우팅

- Flask에서의 라우팅은 사용자가 요청한 URL을 기반으로 수신 요청이 처리되는 방식을 결정함
- Flask는 Flask 애플리케이션 인스턴스의 route() 데코레이터 메서드를 사용하여 경로를 정의한 다음 적절한 뷰 함수에 바인딩하는 방식

#### 2.2.1 route() 데코레이터
- Flask의 함수에 적용될 때 웹 요청을 처리할 수 있는 보기 함수로 변환하는 특수 메서드
- 필수 URL 패턴과 선택적인 HTTP 메서드를 인수로 사용
- URL 패턴을 데코레이터 함수와 연결할 수 있게 해주며, 사용자가 데코레이터에 정의된 URL을 방문하면 해당 함수가 이 요청을 처리하도록 트리거 됨

    ```python
    @app.route('/')
    def index():
        return "이것은 기본 Flask 애플리케이션입니다"
    ```

- 루트 데코레이터를 사용하면 다양한 URL에 대한 경로를 정의하고 원하는 응답을 생성할 수 있는 적절한 뷰 함수에 매핑할 수 있음
- 이를 통해 다양한 경로에 대해 고유한 기능을 갖춘 체계적이고 체계적인 웹 애플리케이션을 만들 수 있음

    ```python
    #//file: "app.py"
from flask import Flask

    app = Flask(__name__)

    @app.route('/')
    def index():
        return "This is a basic flask application"

    if __name__ == '__main__':
        app.run()
    ```

#### 2.2.2 변수 규칙

- Flask를 사용하면 다양한 시나리오, 사용자 입력 및 특정 요구 사항에 대응할 수 있는 동적 URL을 만들 수 있음
- URL 패턴에 변수를 전달함으로써 동적으로 적용하고 사용자에게 개인화된 경험을 제공하는 경로를 설계할 수 있음

- Flask에서의 경로 정의
    - URL 패턴에 <variable_name>으로 표시된 변수 자리 표시자를 포함할 수 있음
    - 기본적으로 이러한 변수는 문자열 값을 포함함
    - 다른 유형의 데이터를 전달해야 하는 경우, 캡처할 데이터 유형을 지정할 수 있는 변환기를 제공함
        - < converter:variable_name >

            <img src='/materials/images/python/S01-04-03-02_02-001.png' width="400"/>

    - 블로그 애플리케이션에서 작성자 프로필을 표시하는 보기용 URL을 만들때, 작성자 이름 전달하기

        ```python
        @app.route('/authors/<username>')
        def show_author(username):
            return f"Return the author profile of {username}"
        ```

        - 이 예제에서 URL 패턴은 '/authors/<username>'이며, 여기서 <username>은 저자의 실제 사용자 이름으로 대체될 변수
        - 이 값은 매개변수로 show_author() 함수에 전달됨
        - 이 값을 사용하여 데이터베이스에서 저자 데이터를 검색하고
        - 데이터베이스에서 얻은 정보를 기반으로 응답을 생성하는 등의 추가 작업을 수행할 수 있음

    - URL을 통해 여러 변수 전달하기

        ```python
        @app.route('/posts/<int:post_id>/<slug>')
        def show_post(post_id, slug):
            # DB검색과 같은 처리를 수행
            return f"Post {post_id} - Slug: {slug}"
        ```

        - 변수 규칙이 여러 개 있는 URL 패턴
        - post_id 변환기를 <int:post_id> 형태로 통합하여 변수에 정수가 필요함을 표현
        - 또한 이 변수에 대한 슬러그 문자열을 캡처하는 <slug>도 제공

        - 처리 내용
            - 이 URL에 요청이 있을 때, Flask는 URL에 지정된 값을 캡처하여 추가 처리를 위해 show_post () 함수에 인수로 전달함
                - 예를 들어, /posts/456/flask-intro에 요청이 있을 경우 Flask는 post_id=456 및 slung=flask-intro를 캡처하고 이러한 값을 show_post () 함수에 인수로 전달
            - 함수 내에서 post_id 및 slung 값을 기반으로 데이터베이스에서 해당 게시물을 검색하는 등 다양한 작업을 수행할 수 있음
            - 예제 코드에서는 캡처된 값을 포함하는 문자열 응답을 간단히 반환함

- 이상의 예제들은 Flask에서 
    - 변수 규칙과 변환기를 사용하여 정수와 문자열 등 다양한 유형의 데이터를 캡처하고, 뷰 함수 내에서 처리하는 동적 URL을 생성하는 방법을 보여줌
    - 이를 통해 특정 사용자 입력에 응답하고 사용자 지정 콘텐츠를 제공하는 앱을 구축할 수 있음


#### 2.2.3 URL 빌딩

- URL 패턴을 정의하고 이를 뷰 함수에 매핑하면 URL을 하드 코딩하는 대신 코드의 다른 곳이나 템플릿에서 URL을 사용할 수 있음
- url_for() 함수
    - 제공하는 인수에 따라 자동으로 URL을 빌드함
    - url_for() 함수 사용의 이점
        - /login 대신 url_for('login')와 같은 것이 있으면 코드를 더 잘 설명할 수 있음
        - 모든 하드코딩된 URL을 수동으로 변경할 필요 없이 한 번에 URL을 변경할 수 있음
        - 특수 문자를 자동으로 투명하게 처리함
        - 생성된 URL은 절대 경로가 되어 브라우저의 상대 경로 문제를 제거함(애플리케이션 구조에 관계없이 함수가 제대로 처리해줌)
        - 합니다.

        - 사용 예시

            ```python
            from flask import Flask, url_for

            app = Flask(__name__)

            @app.route('/')
            def index():
                return "This is a basic flask application"

            @app.route('/authors/<username>')
            def show_author(username):
                return f"Return the author profile of {username}"

            @app.route('/post/<int:post_id>/<slug>')
            def show_post(post_id):
                return f"Showing post with ID: {post_id}"

            if __name__ == '__main__':
                with app.test_request_context():
                    # Generate URLs using url_for()
                    home_url = url_for('index')
                    profile_url = url_for('show_author', username='antony')
                    post_url = url_for('show_post', post_id=456, slug='flask-intro' )

                    print("Generated URLs:")
                    print("Home URL:", home_url)
                    print("Author profile URL:", profile_url)
                    print("Post URL:", post_url)
            ```

            - 코드 설명
                - '/', '/writers/<username>', '/post/<int:post_id>/<slug>의 세 가지 경로를 정의
                - __name__ == '__main__:' 블록 내에서 app.test_request_context()를 사용하여 url_for () 함수에 접근할 수 있는 테스트 요청 컨텍스트가 생성됨
                    - Flask에게 셸을 사용하는 동안에도 요청을 처리하는 것처럼 행동하라고 지시함
                - 생성된 URL을 변수(home_url, profile_url, post_url)에 저장한 다음 인쇄하여 아래에 표시된 것과 유사한 출력을 제공

                    ```text
                    Generated URLs:
                    Home URL: /
                    Author profile URL: /authors/antony
                    Post URL: /post/456/flask-intro
                    ```

    - 템플릿에서의 사용 예시

        ```python
        #//file: "app.py"

        from flask import Flask, render_template

        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template("index.html")

        @app.route('/authors/<username>')
        def show_author(username):
            return render_template('profile.html', username=username)

        if __name__ == '__main__':
            app.run()
        ```

        ```html
        <!--//file: "index.html"-->
        <!DOCTYPE html>
        <html>
        <head>
            <title>Home Page</title>
        </head>
        <body>
            <h1>Welcome to the home page!</h1>
            <a href="{{ url_for('show_author', username='Antony') }}">Visit Antony's profile</a>
        </body>
        </html>
        ```

        ```html
        <!--//file: "profile.html"-->
        <!DOCTYPE html>
        <html>
        <head>
            <title>User Profile</title>
        </head>
        <body>
            <h1>Welcome, {{ username }}!</h1>
            <a href="{{ url_for('index') }}">Go back to home page</a>
        </body>
        </html>
        ```

        - 코드 설명
            - 링크 생성
                - index.html 템플릿
                    - url_for() 함수를 사용하여 'profile'을 위한 URL을 생성
                    - username 인수를 'Antony'로 전달하면 'Antony'의 프로필 페이지 링크가 생성됨
                - profile.html 템플릿
                    - url_for('index')를 사용하여 홈 페이지를 위한 URL 생성
            - 템플릿이 렌더링되면 Plask는 {{url_for(...) }}를 해당 생성된 URL로 대체
                - 이를 통해 템플릿 내에서 동적 URL 생성이 가능해져 다른 경로로 링크를 만들거나 해당 경로에 인수를 쉽게 전달할 수 있음

        - 템플릿에서 url_for()를 사용하면
            - 생성된 URL이 올바르게 유지 관리가 가능하도록 하는 데 도움이 되며
            - 이는 Flask 애플리케이션의 현재 상태에 적응하는 동적 링크를 생성하는 편리한 방법을 제공함

    - 전반적으로 url_for() 함수를 사용하면 
        - 경로 끝점(뷰 함수의 이름) 및 기타 선택적 인수를 기반으로 Flask에서 URL을 동적으로 생성할 수 있음
        - 이를 통해 웹 앱의 특정 기능에 적응하는 유연하고 신뢰할 수 있는 URL을 생성할 수 있음

### 2.3 HTTP 메서드

- 웹 애플리케이션은 URL에 액세스할 때 서로 다른 HTTP 메서드를 사용함

- Flask는 <span style="color: red">**GET, POST, PUT, DELETE**</span> 등 다양한 HTTP 메서드를 지원함
    - HTTP 메서드는 애플리케이션에서 정의한 다양한 URL에 액세스할 때 사용할 수 있는 리소스에 대해 수행할 수 있는 작업을 정의함
    - 다양한 HTTP 메서드를 사용하여 Flask 애플리케이션에서 다양한 유형의 요청을 처리하고 관련 작업을 수행할 수 있음

- Flask에서 경로 데코레이터의 메서드 매개변수를 사용하여 경로가 허용할 수 있는 HTTP 메서드를 정의할 수 있음
    - 경로가 허용하는 메서드를 지정하면 Flask는 해당 메서드에 대해서만 경로에 액세스할 수 있도록 보장함

- 예시

    ```python
    #//file: "app.py"

    from flask import Flask, request

    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        return "This is the home page"

    @app.route('/authors', methods=['GET', 'POST'])
    def authors():
        if request.method == 'GET':
            return "Get all authors"
        elif request.method == 'POST':
            # Create a new author
            return "Create a new author"

    @app.route('/authors/<int:author_id>', methods=['GET', 'PUT', 'DELETE'])
    def author(author_id):
        if request.method == 'GET':
            return f"Get author with ID: {author_id}"
        elif request.method == 'PUT':
            # Update author with ID: author_id
            return f"Update author with ID: {author_id}"
        elif request.method == 'DELETE':
            # Delete author with ID: author_id
            return f"Delete user with ID: {author_id}"

    if __name__ == '__main__':
        app.run()
    ```

- 코드 설명
    - 코드는 '/', '/authors', '/authors/<int:author_id>'의 세 가지 경로를 정의
        - 각 경로에는 특정 HTTP 메서드가 연결되어 있음

    - '/' 경로
        - GET 요청만 허용
        - index() 함수는 이 경로에 대한 GET 요청을 처리
            - GET 요청을 통해 액세스할 때 "This is the home page"라는 문자열을 반환
            - 명시적으로 명시되지 않는 한 모든 메서드에 대해 GET가 기본이므로 GET 메서드 매개변수를 생략할 수 있음

    - '/authors' 경로
        - GET 요청과 POST 요청을 모두 허용
        - authors() 함수
            - 요청 메서드가 GET인 경우: "Get all authors"라는 문자열을 반환
            - 요청 메서드가 POST인 경우: 필요한 작업을 수행하여 새 저자를 생성하고 "Create a new author"라는 문자열을 반환

    - '/authors/<int:author_id>' 경로
        - GET, PUT, DELETE 요청을 허용
        - author() 함수
            - 요청 메서드가 GET인 경우: 지정된 ID의 저자를 검색하고 응답 문자열을 반환
            - 요청 메서드가 PUT인 경우: 지정된 ID의 저자를 업데이트하고 응답 문자열을 반환
            - 요청 메서드가 DELETE인 경우: 지정된 ID의 저자를 삭제하고 응답 문자열을 반환

- 특정 HTTP 메서드를 사용하여 경로를 정의함으로써, 다양한 유형의 요청을 처리하고 관련된 리소스에 대해 적절한 작업을 수행하는 RESTful API 또는 웹 애플리케이션을 만들 수 있음

### 2.4 리디렉션(Redirection) 및 오류

- 라우팅과 오류 처리
    - Flask에서 라우팅의 개념을 살펴볼 때 오류를 처리하는 방법도 이해해야 함
    - 사용자는 흔히 잘못된 URL이나 불완전한 정보를 제공하므로 애플리케이션에서 오류가 발생할 수 있음
    - 따라서 애플리케이션이 완료되려면 유익한 메시지를 제공하거나 사용자를 리디렉션하여 오류를 깔끔하게 처리해야 함
    - Flask는 리디렉션 및 오류 처리를 위한 중단 기능 제공함

- 리디렉션 기능
    - 사용자를 새 URL로 안내하는 데 사용
        - 성공적인 양식 제출, 인증 또는 애플리케이션의 다른 섹션으로 사용자를 안내하고자 할 때와 같은 경우에 사용됨
    - URL을 인수 또는 경로 이름으로 사용함
        - 기본적으로 상태 코드 302를 반환하지만
        - 자체 사용자 지정 상태 코드를 정의할 수 있음

- 오류를 처리하기 위해 중단 함수
    - 요청 처리를 중단하고 HTTP 오류 응답을 반환하는 데 사용
        - 애플리케이션의 예외적인 경우나 오류를 처리하고 적절한 HTTP 상태 코드 및 오류 페이지로 응답할 수 있음
    - 무단 액세스, 무효 요청 및 기타 유형의 오류와 같은 다양한 오류를 처리할 수 있음
        - 오류에 적합한 HTTP 상태 코드를 선택하여 클라이언트가 무엇이 잘못되었는지에 대한 정보를 얻을 수 있도록 함

- 예시

    ```python
    #//file: "app.py"

    from flask import Flask, redirect, render_template, request, abort

    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            # Perform login authentication
            username = request.form['username']
            password = request.form['password']

            # Check if the credentials are valid
            if username == 'admin' and password == 'password':
                # Redirect to the user's dashboard on successful login
                return redirect('/dashboard')
            else:
                # Abort the request with a 401 Unauthorized status code
                abort(401)

        return render_template('login.html')

    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')

    @app.errorhandler(401)
    def unauthorized_error(error):
        return render_template('unauthorized.html'), 401

    if __name__ == '__main__':
        app.run()
    ```

    - 코드 설명
        - /login 경로
            - GET 요청과 POST 요청을 모두 처리
            - POST 요청이 수신되면 
                - 제공된 자격 증명으로 인증을 수행
                - 자격 증명이 유효한 경우
                    - 리디렉션 기능을 사용하여 사용자가 /dashboard 경로로 리디렉션
                - 인증에 실패할 경우(예: 사용자가 잘못된 로그인 정보를 제공하는 경우), 
                    - 중단 함수를 사용하여 401 무단 상태 코드로 중단
                    - 그런 다음 사용자는 unautiful_error 오류 처리기에 지정된 오류 페이지로 이동하여 unautifulated.html 템플릿을 렌더링
                    - 401 상태 코드를 반환

- 리디렉션 및 중단 기능을 함께 활용하면 
    - 인증 실패를 효과적으로 처리하고 
    - 사용자를 적절한 페이지로 리디렉션하거나 
    - 관련 오류 메시지를 표시하여 
    - 안전하고 사용자 친화적인 로그인 경험을 보장할 수 있음

### 2.5 Flask에서 URL 라우팅 모범 사례

- URL을 만들 때 애플리케이션이 모범 사례를 따르는 것이 중요함

    - URL을 정리하고 읽기 쉽게 만들 것
    - 관련 경로를 그룹화 할 것
        - 이렇게 하면 코드베이스를 더 쉽게 탐색하고 관리할 수 있습니다.

    - 변수를 활용할 것
        - URL 패턴의 변수를 사용하면 애플리케이션이 사용자의 동적 요청을 처리할 수 있음
        - 이를 위해 <variable_name>과 같은 규칙을 활용할 수 있음
        - 변수를 변환기와 결합하여 URL의 다양한 종류의 데이터를 처리할 수 있음

    - 오류 메시지 지우기
        - 경로에서 오류를 처리할 때 사용자에게 명확하고 유익한 오류 메시지를 제공할 것
        - 이는 사용자가 오류가 발생한 이유를 이해하고 적절한 조치를 취하는 데 도움이 됨

    - url_for 함수를 사용할 것
        - 애플리케이션 전체에서 URL이 자동으로 생성되고 적절하게 구조화되며 일관성 있게 유지되므로 하드 코딩할 필요가 없음

- 이러한 모범 사례를 따르면 플라스크 애플리케이션에서 잘 조직되고 유지 관리 가능한 URL 라우팅을 생성할 수 있음
- 이는 더 깨끗한 코드, 향상된 가독성, 더 나은 오류 처리, 그리고 더 유연하고 구조화된 URL로 이어짐