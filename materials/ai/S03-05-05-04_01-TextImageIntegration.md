---
layout: page
title:  "텍스트와 이미지 통합 기능 구현"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-04_01-TextImageIntegration
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 기능 구성

### 1.1 처리 작업

**텍스트 프롬프트 1회 입력 →**
1. GPT가 설명 텍스트 생성
2. DALL·E가 해당 프롬프트 기반 이미지 생성
3. 사용자에게 **텍스트 + 이미지** 동시 출력


### 1.2 사용 기술 스택

- Python
- Flask (백엔드 및 웹서버)
- OpenAI API (GPT-3.5 또는 GPT-4 + DALL·E)
- HTML (Jinja 템플릿 엔진)


### 1.3 전체 구조 흐름

```
[사용자 입력 프롬프트]
       ↓
[Flask] → [GPT 호출] → 요약/설명 텍스트 생성
       → [DALL·E 호출] → 이미지 생성 (URL 반환)
       ↓
[HTML 페이지에 결과 표시 (텍스트 + 이미지)]
```


## 2. 예제 구현

### 2.1 프로젝트 구조

```
text-image-app/
├── app.py
├── templates/
│   └── index.html
├── .env
└── requirements.txt
```

### 2.2 코드 구성

- **requirements.txt**

    ```txt
    flask
    openai
    python-dotenv
    ```

- **.env**

    ```
    OPENAI_API_KEY=your_api_key_here
    ```

- **app.py (Flask 서버)**

    ```python
    from flask import Flask, render_template, request
    import openai
    import os
    from dotenv import load_dotenv

    # 환경 변수 로드
    load_dotenv()

    app = Flask(__name__)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    @app.route("/", methods=["GET", "POST"])
    def index():
        gpt_response = None
        image_url = None

        if request.method == "POST":
            user_prompt = request.form["prompt"]

            try:
                # 1. GPT 응답 생성
                chat_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert visual storyteller."},
                        {"role": "user", "content": f"'{user_prompt}'를 설명하는 짧은 글을 만들어줘."}
                    ],
                    temperature=0.7
                )
                gpt_response = chat_response['choices'][0]['message']['content']

                # 2. DALL·E 이미지 생성
                image_response = openai.Image.create(
                    prompt=user_prompt,
                    n=1,
                    size="512x512"
                )
                image_url = image_response['data'][0]['url']

            except Exception as e:
                gpt_response = f"오류 발생: {e}"
                image_url = None

        return render_template("index.html", gpt_response=gpt_response, image_url=image_url)

    if __name__ == "__main__":
        app.run(debug=True)
    ```

- **templates/index.html**

    ```html
    <!DOCTYPE html>
    <html lang="ko">
    <head>
    <meta charset="UTF-8">
    <title>텍스트 + 이미지 생성기</title>
    </head>
    <body>
    <h1>텍스트 + 이미지 생성기</h1>
    <form method="POST">
        <label>프롬프트를 입력하세요:</label><br>
        <input type="text" name="prompt" size="60" required>
        <button type="submit">생성</button>
    </form>

    { % if gpt_response %}
        <h2>GPT가 생성한 텍스트:</h2>
        <p>{ { gpt_response }}</p>
    { % endif %}

    { % if image_url %}
        <h2>생성된 이미지:</h2>
        <img src="{ { image_url }}" alt="Generated Image" width="512">
    { % endif %}
    </body>
    </html>
    ```

## 3. 사용 예시

1. 앱 실행

```bash
python app.py
```

2. 브라우저 접속

```
http://localhost:5000
```

3. 프롬프트 입력 예:

```
a cat astronaut floating in space
```

4. 결과:

- GPT는 해당 내용을 간략히 설명한 텍스트 생성
- DALL·E는 우주에 떠 있는 고양이 우주인 이미지를 생성해 표시


## 4. 확장 아이디어

| 기능                                  | 설명                                                      |
|---------------------------------------|-----------------------------------------------------------|
| 자동 재생성 버튼                      | 텍스트/이미지를 다시 생성                                 |
| 이미지 저장                           | 이미지 다운로드 링크 제공                                 |
| GPT로 프롬프트 튜닝                   | 사용자가 입력한 프롬프트를 GPT가 자동 수정 후 이미지 생성 |
| React 또는 Vue.js 프론트엔드 연동     | API 결과를 프론트로 실시간 전달                           |
| 이미지 편집 기능 (DALL·E inpainting) | 이미지 일부 영역을 선택해 재생성 가능                     |


## 5. 요약

| 항목         | 설명                                        |
|--------------|---------------------------------------------|
| GPT 활용     | 텍스트 설명, 요약, 감성 부여                |
| DALL·E 활용 | 이미지 생성 (512x512, 1024x1024 등)         |
| Flask        | 사용자 입력 수집 → API 호출 → 결과 렌더링 |
| 결과         | 텍스트 + 이미지 통합 UI 제공                |
