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

1. [OpenAI 플랫폼](https://platform.openai.com/account/api-keys)에서 로그인 후
2. `Create new secret key` 클릭하여 키 생성
3. 해당 키를 복사하고 `.env` 파일 또는 환경 변수에 저장

    ```env
    OPENAI_API_KEY=your_api_key_here
    ```

## 3. DALL·E 이미지 생성 요청 형식

### 3.1 기본 요청 구조 (Text-to-Image)

```http
POST https://api.openai.com/v1/images/generations
```

- **Headers:**

    ```http
    Authorization: Bearer YOUR_API_KEY
    Content-Type: application/json
    ```

- **Body (예시):**

    ```json
    {
        "prompt": "a futuristic city with flying cars",
        "n": 1,
        "size": "1024x1024"
    }
    ```


## 4. Flask를 이용한 DALL·E 이미지 생성 예제

### 4.1 프로젝트 구조

```
dalle-flask-app/
├── app.py
├── templates/
│   └── index.html
├── .env
└── requirements.txt
```

### 4.2 프로젝트 코드

- **requirements.txt**

    ```txt
    flask
    openai
    python-dotenv
    ```

- **.env**

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

- **app.py**

```python
#//file: "app.py"
from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']
        except Exception as e:
            image_url = f"Error: {e}"

    return render_template("index.html", image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
```

- **templates/index.html**

```html
<!--//file: "templates/index.html"-->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DALL·E Image Generator</title>
</head>
<body>
    <h1>DALL·E 이미지 생성기</h1>
    <form method="POST">
        <label>프롬프트를 입력하세요:</label><br>
        <input type="text" name="prompt" size="60">
        <button type="submit">이미지 생성</button>
    </form>

    { % if image_url %}
        <h3>생성된 이미지:</h3>
        { % if image_url.startswith('Error') %}
            <p style="color: red">{ { image_url }}</p>
        { % else %}
            <img src="{ { image_url }}" alt="Generated Image" width="512">
        { % endif %}
    { % endif %}
</body>
</html>
```

### 4.3 실행 방법

1. 라이브러리 설치
```bash
pip install -r requirements.txt
```

2. 서버 실행
```bash
python app.py
```

3. 브라우저에서 접속
```
http://localhost:5000/
```

4. 텍스트 프롬프트를 입력하면 DALL·E가 이미지를 생성하고 보여줌

---

## 5. 요약

| 항목 | 설명 |
|------|------|
| **API 키** | OpenAI 대시보드에서 생성 후 `.env`에 저장 |
| **요청 방식** | POST `/v1/images/generations` |
| **입력 파라미터** | `prompt`, `n`, `size` |
| **출력** | 이미지 URL 반환 (웹에서 사용 가능) |
| **Flask 통합** | 사용자 입력 → 이미지 생성 → 결과 웹에 출력 |
