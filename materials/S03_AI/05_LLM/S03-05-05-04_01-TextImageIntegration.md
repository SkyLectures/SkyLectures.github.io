---
layout: page
title:  "텍스트와 이미지 통합 기능 구현"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-04_01-TextImageIntegration
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Django 기반 예제

- 예제 설명
    - Django 기반으로 구성한 프로젝트
    - 내용
        - 사용자가 프롬프트를 입력하면
        - GPT-4 Turbo로부터 텍스트 설명을 받고
        - DALL·E 3로부터 이미지를 생성하며
        - 둘을 웹 페이지에 동시에 출력하는 예제

- 프로젝트 구조

    ```bash
    openai_webapp/
    ├── mysite/
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── dallegpt/
    │   ├── templates/
    │   │   └── dallegpt/
    │   │       └── index.html
    │   ├── views.py
    │   ├── urls.py
    │   └── __init__.py
    ├── manage.py
    └── requirements.txt
    ```

- 라이브러리

    ```bash
    pip install django
    ```

- 예제 코드
    - Django 설정

        ```python
        #//file: mysite/settings.py

        # settings.py (일부 발췌)
        import os
        from dotenv import load_dotenv
        load_dotenv()

        # templates 경로 설정
        TEMPLATES[0]['DIRS'] = [BASE_DIR / "dallegpt" / "templates"]

        # secret key 등 기존 설정 유지
        ```

    - URL 설정

        ```python
        #//file: mysite/urls.py

        from django.contrib import admin
        from django.urls import path, include

        urlpatterns = [
            path("admin/", admin.site.urls),
            path("", include("dallegpt.urls")),  # 앱 경로 추가
        ]
        ```

        ```python
        #//file: dallegpt/urls.py

        from django.urls import path
        from . import views

        urlpatterns = [
            path("", views.index, name="index"),
        ]
        ```
        
    - 뷰 함수

        ```python
        #//file: dallegpt/views.py

        from django.shortcuts import render
        from openai import OpenAI
        import os

        client = OpenAI()

        def index(request):
            gpt_text = None
            image_url = None
            prompt = ""

            if request.method == "POST":
                prompt = request.POST.get("prompt", "")
                try:
                    # 1. GPT로 텍스트 설명 생성
                    chat_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "당신은 이미지 설명을 생성하는 친절한 도우미입니다."},
                            {"role": "user", "content": f"'{prompt}'에 대한 설명을 작성해줘."}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    gpt_text = chat_response.choices[0].message.content

                    # 2. DALL·E로 이미지 생성
                    image_response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        n=1,
                        size="1024x1024",
                        quality="standard"
                    )
                    image_url = image_response.data[0].url

                except Exception as e:
                    gpt_text = f"오류 발생: {str(e)}"

            return render(request, "dallegpt/index.html", {
                "prompt": prompt,
                "gpt_text": gpt_text,
                "image_url": image_url
            })
        ```

    -  템플릿 (dallegpt/templates/dallegpt/index.html)

        ```html
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>GPT + DALL·E 이미지 생성기</title>
            <style>
                body { font-family: sans-serif; margin: 40px; }
                textarea, input { width: 400px; padding: 8px; }
                img { max-width: 512px; display: block; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>AI 설명 & 이미지 생성기</h1>
            <form method="POST">
                { % csrf_token %}
                <label>프롬프트 입력:</label><br>
                <input type="text" name="prompt" value="{ { prompt }}" required><br><br>
                <button type="submit">생성하기</button>
            </form>

            { % if gpt_text %}
                <h2>GPT 설명 결과:</h2>
                <p>{ { gpt_text }}</p>
            { % endif %}

            { % if image_url %}
                <h2>생성된 이미지:</h2>
                <img src="{ { image_url }}" alt="AI 생성 이미지">
            { % endif %}
        </body>
        </html>
        ```

- 실행 방법
    - 서버 실행

        ```bash
        # 프로젝트 생성 및 앱 등록 후
        python manage.py migrate
        python manage.py runserver
        ```

    - 브라우저 테스트
        - http://localhost:8000에 접속 후 프롬프트를 입력해서 테스트


## 2. Flask 기반 예제

- 예제 설명
    - Flask 기반으로 구성한 프로젝트
    - 내용
        - 사용자가 프롬프트를 입력하면
        - GPT-4 Turbo로부터 텍스트 설명을 받고
        - DALL·E 3로부터 이미지를 생성하며
        - 둘을 웹 페이지에 동시에 출력하는 예제

- 전체 구조 흐름

    ```text
    [사용자 입력 프롬프트]
        ↓
    [Flask] → [GPT 호출] → 요약/설명 텍스트 생성
        → [DALL·E 호출] → 이미지 생성 (URL 반환)
        ↓
    [HTML 페이지에 결과 표시 (텍스트 + 이미지)]
    ```

- 프로젝트 구조

    ```bash
    flask_dalle_gpt/
    ├── app.py                     # Flask 앱 실행 스크립트
    ├── templates/
    │   └── index.html             # 웹 템플릿
    ├── static/
    │   └── style.css              # (선택) CSS 스타일
    └── requirements.txt           # 필요 패키지 목록
    ```

- 라이브러리

    ```bash
    pip install flask
    ```

- 예제 코드

- **app.py (Flask 서버)**

    ```python
    from flask import Flask, render_template, request
    from openai import OpenAI
    import os

    app = Flask(__name__)
    client = OpenAI()

    @app.route("/", methods=["GET", "POST"])
    def index():
        prompt = ""
        gpt_text = None
        image_url = None

        if request.method == "POST":
            prompt = request.form["prompt"]

            try:
                # 1. GPT-4 Turbo로 텍스트 생성
                chat_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "당신은 이미지 설명을 잘 해주는 도우미입니다."},
                        {"role": "user", "content": f"'{prompt}'에 대한 설명을 써줘."}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                gpt_text = chat_response.choices[0].message.content

                # 2. DALL·E 3로 이미지 생성
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = image_response.data[0].url

            except Exception as e:
                gpt_text = f"오류 발생: {str(e)}"

        return render_template("index.html", prompt=prompt, gpt_text=gpt_text, image_url=image_url)

    if __name__ == "__main__":
        app.run(debug=True)
    ```

- **templates/index.html**

    ```html
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>GPT + DALL·E 생성기</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            input[type="text"] { width: 400px; padding: 10px; }
            button { padding: 10px 20px; }
            img { max-width: 512px; margin-top: 20px; }
            textarea { width: 500px; height: 150px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>GPT + DALL·E 생성기</h1>
        <form method="POST">
            <label for="prompt">프롬프트:</label><br>
            <input type="text" id="prompt" name="prompt" value="{ { prompt or '' }}" required>
            <button type="submit">생성</button>
        </form>

        { % if gpt_text %}
            <h2>GPT 설명 결과:</h2>
            <textarea readonly>{ { gpt_text }}</textarea>
        { % endif %}

        { % if image_url %}
            <h2>생성된 이미지:</h2>
            <img src="{ { image_url }}" alt="Generated Image">
        { % endif %}
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

## 3. 확장 아이디어

| 기능                                  | 설명                                                      |
|---------------------------------------|-----------------------------------------------------------|
| 자동 재생성 버튼                      | 텍스트/이미지를 다시 생성                                 |
| 이미지 저장                           | 이미지 다운로드 링크 제공                                 |
| GPT로 프롬프트 튜닝                   | 사용자가 입력한 프롬프트를 GPT가 자동 수정 후 이미지 생성 |
| React 또는 Vue.js 프론트엔드 연동     | API 결과를 프론트로 실시간 전달                           |
| 이미지 편집 기능 (DALL·E inpainting) | 이미지 일부 영역을 선택해 재생성 가능                     |

## 4. 요약

| 항목         | 설명                                        |
|--------------|---------------------------------------------|
| GPT 활용     | 텍스트 설명, 요약, 감성 부여                |
| DALL·E 활용 | 이미지 생성 (512x512, 1024x1024 등)         |
| Flask        | 사용자 입력 수집 → API 호출 → 결과 렌더링 |
| 결과         | 텍스트 + 이미지 통합 UI 제공                |
