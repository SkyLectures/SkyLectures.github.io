---
layout: page
title:  "GPT API 연동"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-01_01-UsingGptApi
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. OpenAI API 개요

- OpenAI API란?
    - OpenAI에서 제공하는 클라우드 기반 서비스
    - 사용자가 LLM을 활용하여 다양한 자연어 처리 작업을 수행할 수 있도록 지원함
    - 복잡한 머신러닝 모델을 직접 구축하지 않고도 텍스트 생성, 번역, 요약, 질문 응답 등 다양한 기능을 구현할 수 있음

- 주요 특징
    - 다양한 모델 제공:
        - gpt-4/gpt-4o: 최신 언어 모델로 고급 자연어 이해 및 생성 능력을 보유
        - gpt-3.5/gpt-3.5-turbo: 빠른 응답과 효율적인 성능 제공
        - Codex: 코드 생성에 특화된 모델
        - DALL-E: 텍스트를 기반으로 이미지 생성
        - 기타(새로운 버전의 발표에 따라 계속 추가됨)

    - 다양한 활용 사례:
        - 대화형 챗봇 개발
        - 콘텐츠 생성(스토리, 마케팅 카피 등)
        - 언어 번역 및 문서 요약
        - 코드 작성과 디버깅

    - 멀티턴 대화 지원:
        - 이전 대화 내용을 기억하지 못하는 API의 특성을 극복하기 위해 사용자는 이전 대화 내용을 입력으로 전달해야 함

    - 비용 효율성:
        - 사용량에 따라 종량제 요금 부과(GPT-4는 1,000 토큰당 약 $003, GPT-35는 $002)
        - 운영정책에 따라 계속 바뀌는 중

## 2. OpenAI API 키 발급 및 설정

1. OpenAI 계정 생성
    1. [https://platform.openai.com/signup](https://platform.openai.com/signup) 에서 계정 생성
    2. 로그인 후 대시보드로 이동

2. API 키 발급
    1. [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) 접속
        - 직접 찾아가려면: OpenAI 사이트 → 우측 상단 → Dashboard → 좌측 사이드바 중단 → API Keys 클릭
    2. "Create new secret key" 버튼 클릭 → 생성된 키를 복사해서 저장
    - **주의 사항**
        - 생성한 API 키는 단 한 곳에서만 사용이 가능함
        - 다른 서비스에 사용하려면 새로 발급받아야 함
        - 생성한 API 키는 다시 확인이 되지 않으므로 창을 닫기 전에 복사, 저장해 둘것
        - 타인에게 절대로 노출 시키지 말것 → <span style="color: red">**쓴 만큼 돈 나감**</span>

3. 환경 변수로 저장 (예시: `.env` 파일)

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## 3. API 요청 방법

- OpenAI의 GPT API는 HTTPS 요청 기반의 **RESTful API**임
- `chat/completions` 엔드포인트를 통해 GPT를 사용할 수 있음

### 3.1 기본 API 요청 구조

- 요청 URL

    ```
    https://api.openai.com/v1/chat/completions
    ```

- HTTP 헤더

    ```http
    Authorization: Bearer YOUR_API_KEY
    Content-Type: application/json
    ```

- HTTP Body (JSON)

    ```json
    {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "GPT-3.5 Turbo API 사용하는 방법 알려줘"}
        ],
        "temperature": 0.7
    }
    ```

- 예시 (Python, `requests` 라이브러리 사용)

    ```python
    import requests
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "GPT-3.5 Turbo API 사용하는 방법 알려줘"}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    ```


## 4. GPT-3.5 Turbo API 활용을 위한 기본 지식

### 4.1 메시지 구조

- `system`: AI의 행동 지침 설정 (예: "친절한 도우미처럼 행동하세요.")
- `user`: 사용자가 보낸 메시지
- `assistant`: 이전 응답을 기록할 때 사용 (대화 맥락 유지)

### 4.2 주요 파라미터 설명

| 파라미터 | 설명 |
|----------|------|
| `model` | 사용할 모델 이름 (`gpt-3.5-turbo`) |
| `messages` | 대화 기록 (role, content 구성) |
| `temperature` | 창의성 조절 (0.0~2.0, 낮을수록 정답형) |
| `max_tokens` | 응답 최대 길이 설정 (단어가 아닌 토큰 기준) |
| `top_p` | `temperature` 대신 확률 분포 사용 제어 (일반적으론 둘 중 하나만 조정) |
| `stream` | `true`로 설정하면 스트리밍 응답 (서버 푸시 방식) |
| `n` | 응답 개수 설정 (기본 1개) |

### 4.3 모델별 토큰 한도

| 모델명           | 요청+응답 합산 최대 토큰 수 |
|------------------|--------------------------|
| `gpt-3.5-turbo`  | 16,385 tokens            |

### 4.4 토큰 개념
- **1,000 tokens ≒ 750~800단어**
- 예시: "Hello, how are you?" → 약 5 tokens

### 4.5 요금 구조 (2024년 기준)
- **입력 (prompt):** $0.0015 / 1,000 tokens
- **출력 (completion):** $0.002 / 1,000 tokens

> 요금은 변경될 수 있으므로 [공식 가격 페이지](https://openai.com/pricing) 참고


### 4.6 기타 참고 사항

- 하루 사용량 한도 존재 (무료 요금제의 경우 낮음)
- 에러 발생 시:
  - `401`: API 키 오류
  - `429`: 속도 제한 초과
  - `500`: 서버 내부 오류 (재시도 필요)

## 5. Python용 OpenAI 공식 라이브러리 (SDK) 사용법

### 5.1 설치

```bash
pip install openai
```

### 5.2 기본 사용 예제

```python
import openai

openai.api_key = "your_openai_api_key_here"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Python에서 GPT-3.5를 어떻게 사용하나요?"}
    ],
    temperature=0.7,
)

print(response['choices'][0]['message']['content'])
```

### 5.3 환경 변수로 API 키 숨기기 (`.env` 활용 예)
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```
