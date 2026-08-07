---
layout: page
title:  "FastAPI 설치 및 환경설정"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S01-04-04-01_02-FastApiSetting
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 가상 환경(Virtual Environment) 생성

> 가장 먼저 수행해야 할 필수 단계
{: .common-quote}

- **작업 내용:** `python -m venv fastapiweb` 후 가상환경 활성화(source ./bin/activate, ./Script/activate)
- **이유 (중요)**
    - **의존성 격리**
        - FastAPI는 Pydantic, Starlette 등 특정 버전의 라이브러리에 의존함
        - 시스템 자체의 파이썬에 설치할 경우, 다른 AI 프로젝트(PyTorch, TensorFlow 등)와 라이브러리 버전 충돌 발생 가능
    - **깔끔한 배포**
        - `pip freeze`를 통해 해당 프로젝트에 필요한 패키지만 정확히 추출하여 `requirements.txt`를 관리할 수 있음

## 2. FastAPI 및 ASGI 서버 설치

> FastAPI 자체와 이를 구동할 서버 설치
{: .common-quote}

- **작업 내용:** `pip install fastapi uvicorn[standard]`
- **작업 이유**
    - **uvicorn[standard] 설치 이유**
        - FastAPI는 스스로 웹 서버 역할을 하지 못하는 **프레임워크**일 뿐
        - 실제 HTTP 요청을 비동기로 처리해 줄 **ASGI(Asynchronous Server Gateway Interface) 서버**인 `uvicorn`이 반드시 필요함
    - `[standard]` 옵션을 붙이면
        - `uvloop`과 같은 고성능 라이브러리가 함께 설치됨
        - Node.js나 Go에 필적하는 속도를 내는 토대가 됨

## 3. 기본 애플리케이션 코드 작성 (`main.py`)

> 간단한 엔드포인트 작성
{: .common-quote}

- **작업 내용:** `FastAPI()` 인스턴스 생성 및 `@app.get("/")` 데코레이터를 이용한 비동기 함수(`async def`) 정의
- **작업 이유**
    - **async def 사용**
        - FastAPI의 최대 장점인 **비동기 처리**를 활용하기 위함
        - DB 조회나 외부 API 호출(예: LLM API) 시 I/O 바운드 대기 시간을 효율적으로 관리하려면
            - 동기(`def`)가 아닌 비동기(`async def`) 함수로 작성하는 것이 표준

```python
#//file: "main.py"
from fastapi import FastAPI

# 1. FastAPI 인스턴스 생성
# 이 객체가 전체 API 서비스의 중심이 됨
app = FastAPI()

# 2. 경로 데코레이터와 비동기 함수 정의
# 클라이언트가 "/" 경로로 GET 요청을 보냈을 때 실행됨
@app.get("/")
async def read_root():
    """
    가장 기본적인 비동기 엔드포인트
    비동기(async)를 사용하여 서버의 처리 효율을 극대화함.
    """
    return {"Hello": "FastAPI", "Message": "성공적으로 서버가 실행되었습니다."}
```

## 4. 서버 실행 및 워커 설정

> 작성한 코드 실행
{: .common-quote}

- **작업 내용:** `uvicorn main:app --reload`
- **작업 이유**
    - **--reload 옵션**
        - 개발 환경 전용 옵션
        - 코드가 수정될 때마다 서버를 자동으로 재시작하여 생산성을 높여줌
        - 단, 운영 환경에서는 성능 저하 방지를 위해 절대 사용하지 않음
- **확인**
    - 웹 브라우저에서 `http://127.0.0.1:8000` 접속
    - {"Hello": "FastAPI", "Message": "성공적으로 서버가 실행되었습니다."} 가 출력되었는지 확인<br><br>

<div class="insert-image" style="text-align: center;">
    <img style="width: 950px;" src="/materials/python/images/S01-04-04-01_02-001.png"><br><br>
    <img style="width: 700px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-002.png">
</div>

## 5. 자동 문서화 확인

> 브라우저에서 `/docs` 경로로 접속하여 자동 문서화 확인
{: .common-quote}

- **작업 내용:** `http://127.0.0.1:8000/docs` 접속 및 Swagger UI 확인
- **작업 이유**
    - 별도의 설정 없이 작성된 **Python 타입 힌트**를 기반으로 인터랙티브한 API 명세서가 생성되었는지 검증하는 단계<br><br>

<div class="insert-image" style="text-align: center;">
    <img style="width: 700px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-003.png"><br><br>
    <img style="width: 700px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-004.png">
</div>

## 6. (선택적 보완) 환경 변수 관리 (`python-dotenv`)

> 실무 프로젝트라면 보안을 위해 반드시 추가하는 단계
{: .common-quote}

- **작업 내용:** `pip install python-dotenv` 설치 및 `.env` 파일 관리.
    - 1단계: .env 파일 작성 (비밀 금고 만들기)
        - 프로젝트 최상위 폴더(main.py가 있는 곳)에 .env라는 이름의 파일을 만들고 내용을 입력함
        - 주의: = 앞뒤로 공백이 없어야 하며, 문자열에 따옴표를 쓰지 않아도 됨

        ```bash
        #//file: ".env"
        DATABASE_URL=postgresql://user:password@localhost:5432/mydatabase
        OPENAI_API_KEY=sk-your-secret-key-12345
        DEBUG_MODE=True
        ```

    - 2단계: 파이썬 코드에서 확인 (main.py)
        - python-dotenv를 사용하여 파일에 적힌 내용을 시스템 환경 변수로 로드한 뒤, os 모듈로 읽어옴

        ```python
        #//file: "main.py"
import os
        from fastapi import FastAPI
        from dotenv import load_dotenv

        # .env 파일의 내용을 환경 변수로 로드합니다.
        load_dotenv()

        app = FastAPI()

        # os.getenv를 통해 값을 가져옵니다.
        DB_URL = os.getenv("DATABASE_URL")
        API_KEY = os.getenv("OPENAI_API_KEY")

        @app.get("/env-check")
        async def check_env():
            # 실제 실무에서는 보안상 API_KEY 전체를 노출하면 안 되지만, 
            # 확인을 위해 앞글자만 출력해봅니다.
            return {
                "database_url": DB_URL,
                "api_key_prefix": API_KEY[:5] if API_KEY else "Not Found"
            }
        ```
<div class="insert-image" style="text-align: center;">
    <img style="width: 800px; border: solid lightgray 1px;" src="/materials/python/images/S01-04-04-01_02-005.png">
</div>

- **작업 이유**
    - **보안**
        - DB 접속 정보, API Key(OpenAI Key 등)와 같은 민감한 정보를 소스 코드에 하드코딩하지 않고 환경 변수로 분리하기 위함
        - 개인 사업체 운영이나 기업 시스템 개발 경험상 보안의 엄밀함을 위해 필수적인 습관


## 요약 가이드

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">단계</th>
        <th style="width: 400px;">명령어 / 작업</th>
        <th style="width: 400px;">핵심 이유</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader" style="text-align: left;">1. 가상환경</td>
            <td class="td-left"><span style="color:darkred; font-size: 1.3em; font-weight: bold;">python -m venv venv</span></td>
            <td class="td-left">라이브러리 버전 충돌 방지 및 의존성 격리</td>
        </tr>
        <tr>
            <td class="td-rowheader" style="text-align: left;">2. 설치</td>
            <td class="td-left"><span style="color:darkred; font-size: 1.3em; font-weight: bold;">pip install fastapi uvicorn[standard]</span></td>
            <td class="td-left">프레임워크와 고성능 비동기 엔진(ASGI) 확보</td>
        </tr>
        <tr>
            <td class="td-rowheader" style="text-align: left;">3. 코드 작성</td>
            <td class="td-left"><span style="color:darkred; font-size: 1.3em; font-weight: bold;">async def</span> 기반 API 정의</td>
            <td class="td-left">논블로킹 I/O 처리를 통한 고성능 보장</td>
        </tr>
        <tr>
            <td class="td-rowheader" style="text-align: left;">4. 실행</td>
            <td class="td-left"><span style="color:darkred; font-size: 1.3em; font-weight: bold;">uvicorn main:app --reload</span></td>
            <td class="td-left">코드 변경 사항 실시간 반영 (개발 편의성)</td>
        </tr>
        <tr>
            <td class="td-rowheader" style="text-align: left;">5. 검증</td>
            <td class="td-left"><span style="color:darkred; font-size: 1.3em; font-weight: bold;">/docs</span> 접속 확인</td>
            <td class="td-left">타입 힌트 기반 자동 문서화 및 테스트 환경 확보</td>
        </tr>
    </tbody>
</table>
</div>
