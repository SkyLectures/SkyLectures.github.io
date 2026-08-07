---
layout: page
title:  "DALL-E API 연동"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-03_01-UsingDalleApi
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


- **DALL·E API**(OpenAI): 텍스트 프롬프트를 기반으로 이미지를 생성하거나 편집할 수 있게 해주는 기능을 지원하는 API


## 1. DALL·E API란?

- 텍스트를 이미지로 바꿔주는 **Text-to-Image 모델**
- OpenAI API를 통해 `images/generations` 엔드포인트로 호출함
- DALL·E 3부터는 **보다 정교하고 고품질의 이미지** 생성이 가능하며, `gpt-4` API와 통합되고 있음


## 2. DALL·E API 키 설정 방법

1. [OpenAI 플랫폼](https://platform.openai.com/api-keys)에서 로그인 후
2. `Create new secret key` 클릭하여 키 생성
3. 해당 키를 복사하고 환경 변수에 저장

## 3. DALL·E 이미지 생성 요청 형식

### 3.1 기본 요청 구조 (Text-to-Image)

- **REST API**
    ```bash
    POST https://api.openai.com/v1/images/generations
    ```

- **Headers:**

    ```http
    Authorization: Bearer YOUR_OPENAI_API_KEY
    Content-Type: application/json
    ```

- **Body (JSON):**
    - 예시

        ```json
        {
        "model": "dall-e-3",
        "prompt": "a futuristic city skyline at sunset",
        "n": 1,
        "size": "1024x1024",
        "quality": "standard"
        }
        ```

### 3.2 예제

- **Curl 예제**

    ```bash
    curl https://api.openai.com/v1/images/generations \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
    -d '{
        "model": "dall-e-3",
        "prompt": "a futuristic city skyline at sunset",
        "n": 1,
        "size": "1024x1024",
        "quality": "standard"
    }'
    ```

- **Python requests 라이브러리 사용 예제**

    ```python
    import requests

    # OpenAI API Key
    API_KEY = "your_openai_api_key"

    # 요청 헤더
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 요청 바디
    payload = {
        "model": "dall-e-3",
        "prompt": "a futuristic city skyline at sunset",
        "n": 1,
        "size": "1024x1024",
        "quality": "standard"
    }

    # 이미지 생성 요청
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers=headers,
        json=payload
    )

    # 응답 처리
    if response.status_code == 200:
        image_url = response.json()['data'][0]['url']
        print("이미지 URL:", image_url)
        
        # 이미지 다운로드 (선택 사항)
        img_data = requests.get(image_url).content
        with open("generated_image.png", "wb") as f:
            f.write(img_data)
            print("이미지가 'generated_image.png'로 저장되었습니다.")
    else:
        print("오류:", response.status_code, response.text)
    ```

- **OpenAI Python SDK v1.x의 최신 스타일 기준 예제**

    ```python
    from openai import OpenAI

    # OpenAI 클라이언트 초기화 (환경 변수 사용 권장)
    client = OpenAI()

    # 이미지 생성 요청
    response = client.images.generate(
        model="dall-e-3",
        prompt="a futuristic city skyline at sunset",
        n=1,
        size="1024x1024",
        quality="standard"
    )

    # 이미지 URL 추출
    image_url = response.data[0].url
    print("이미지 URL:", image_url)

    # 이미지 다운로드 (선택 사항)
    import requests

    img_data = requests.get(image_url).content
    with open("generated_image.png", "wb") as f:
        f.write(img_data)
        print("이미지가 'generated_image.png'로 저장되었습니다.")
    ```

- requests 방식과 OpenAI SDK 방식 비교

| 항목         | `requests` 방식   | `openai` SDK 방식 (추천)         |
| ------------ | ----------------- | -------------------------------- |
| HTTP 직접 처리 | 수동 헤더/URL 설정 필요 | 자동 처리 (함수 기반) |
| 에러 처리      | 수동 처리 필요        | 예외 발생, `.http_status`로 확인 가능 |
| 코드 길이      | 상대적으로 길고 반복적    | 간결하고 읽기 쉬움 |
| 유지보수       | API 변경에 취약      | SDK 자동 업데이트 반영 |


## 4. Flask를 이용한 DALL·E 이미지 생성 예제

### 4.1 기본 예제
- Flask 설치

    ```bash
    pip install flask
    ```

- 소스코드

    ```python
    #//file: "app.py"

    from flask import Flask, request, render_template_string
    from openai import OpenAI
    import os

    app = Flask(__name__)

    # OpenAI 클라이언트 초기화 (환경 변수 또는 직접 입력)
    client = OpenAI()

    # HTML 템플릿
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DALL·E 이미지 생성기</title>
    </head>
    <body>
        <h1>DALL·E 이미지 생성</h1>
        <form method="POST">
            <label>프롬프트:</label><br>
            <input type="text" name="prompt" style="width: 300px;" required><br><br>
            <button type="submit">이미지 생성</button>
        </form>
        {% if image_url %}
            <h2>결과</h2>
            <img src="{{ image_url }}" alt="Generated Image" style="max-width: 512px;">
        {% endif %}
    </body>
    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def generate_image():
        image_url = None
        if request.method == "POST":
            prompt = request.form["prompt"]
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard"
                )
                image_url = response.data[0].url
            except Exception as e:
                image_url = None
                print("❌ 오류:", str(e))

        return render_template_string(HTML_TEMPLATE, image_url=image_url)

    if __name__ == "__main__":
        app.run(debug=True)
    ```

- 실행 방법
    1. app.py 실행

        ```bash
        python app.py
        ```

    2. 브라우저에서 http://localhost:5000 접속
    3. 텍스트 프롬프트 입력 → 이미지 생성 및 표시

- 특징

| 기능                          | 설명                             |
| ----------------------------- | -------------------------------- |
| 최신 OpenAI SDK (`v1.x`) 사용 | `OpenAI` 객체 기반 호출          |
| 프롬프트 입력 폼              | 사용자 입력 처리                 |
| 이미지 웹 출력                | `img` 태그로 생성된 이미지 표시  |
| 환경 변수 지원                | 보안상의 이유로 API 키 하드코딩 대신 환경변수 권장 |


### 4.2 템플릿 적용 형태 예제

- 프로젝트 구조

    ```bash
    dalle_flask_app/
    ├── app.py                  # Flask 애플리케이션 진입점
    ├── templates/
    │   └── index.html          # HTML 템플릿 (Jinja2)
    ├── static/
    │   └── style.css           # (선택) 스타일 시트
    └── requirements.txt        # 필요한 패키지 목록
    ```

- **app.py**

    ```python
    #//file: "app.py"
    from flask import Flask, render_template, request
    from openai import OpenAI
    import os

    client = OpenAI()

    # Flask 앱 초기화
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def generate_image():
        image_url = None
        prompt = None

        if request.method == "POST":
            prompt = request.form["prompt"]
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard"
                )
                image_url = response.data[0].url
            except Exception as e:
                print("오류:", e)

        return render_template("index.html", image_url=image_url, prompt=prompt)

    if __name__ == "__main__":
        app.run(debug=True)
    ```

- **templates/index.html**

    ```html
    <!--//file: "templates/index.html"-->
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>DALL·E 이미지 생성기</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            input[type="text"] { width: 400px; padding: 10px; }
            button { padding: 10px 20px; }
            img { max-width: 512px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>DALL·E 이미지 생성기</h1>
        <form method="POST">
            <label>프롬프트 입력:</label><br>
            <input type="text" name="prompt" value="{{ prompt or '' }}" required><br><br>
            <button type="submit">이미지 생성</button>
        </form>

        {% if image_url %}
            <h2>생성된 이미지:</h2>
            <img src="{{ image_url }}" alt="Generated Image">
        {% endif %}
    </body>
    </html>
    ```

- 실행 방법
    1. app.py 실행

        ```bash
        python app.py
        ```

    2. 브라우저에서 http://localhost:5000 접속
    3. 텍스트 프롬프트 입력 → 이미지 생성 및 표시
