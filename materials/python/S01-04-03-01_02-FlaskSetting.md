---
layout: page
title:  "Flask 설치 및 환경설정"
date:   2025-03-01 10:00:00 +0900
permalink: /material/S01-04-03-01_02-FlaskSetting
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

- Flask는 파이썬으로 구현되는 개발환경이므로 가상환경의 사용을 권장함
- 파이썬에 대한 설치 및 환경 설정은 [파이썬 가상환경 설정](/material/S01-01-02-01_01-VirtualEnvironment)을 참고할 것
- 개발작업은 /Workspaces/ 등과 같은 별도의 폴더에서 진행하는 것을 권장함

## 1. 가상환경 생성

- 가상환경의 이름은 프로젝트에 맞게 설정함
    - 본 페이지에서는 flaskweb이라는 이름을 사용함

        ```bash
        python -m venv flaskweb
        ```

## 2. 가상환경 활성화

- Linux / MAC의 경우

    ```bash
    cd flaskweb
    source ./bin/activate
    ```

- Windows의 경우

    ```bash
    cd flaskweb
    ./Scripts/activate
    ```

- 프로젝트명이 앞에 나타나면 활성화 성공

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-01_02-001.png' width="700"/></p>

- 비활성화는 `deactivate` 명령어 사용

    ```bash
    deactivate
    ```

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-01_02-002.png' width="700"/></p>

## 3. Flask 웹 프레임워크 설치

- Flask 외에도 필요한 라이브러리/패키지들을 설치할 수 있음

    ```bash
    pip install flask
    ```

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-01_02-003.png' width="800"/></p>

## 4. 샘플코드 작성

- 일단 여기서는 내용의 이해는 건너뛰고 코드만 보고 넘어가도 됨

    ```python
    #//file: "hello.py"
from flask import Flask

    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return 'Hello, World'
    ```

## 5. Flask 실행

1. 기본 환경변수 설정
    - Linux / MAC의 경우

        ```bash
        export FLASK_APP=hello.py
        export FLASK_ENV=development
        ```

    - Windows의 경우

        ```sh
        set FLASK_APP=hello.py
        set FLASK_ENV=development
        ```

2. Flask 실행

    ```bash
    flask run
    ```

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-01_02-004.png' width="800"/></p>

## 6. 웹 브라우저에서 작동 확인

- http://127.0.0.1:5000/ 접속
- `Hello, World` 출력 확인

    <p style="text-align: center;"><img src='/materials/images/python/S01-04-03-01_02-005.png' width="700" style="border: 1px solid lightgray"/></p>
