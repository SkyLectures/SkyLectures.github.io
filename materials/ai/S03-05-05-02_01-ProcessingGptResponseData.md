---
layout: page
title:  "GPT 응답 데이터 처리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S03-05-05-02_01-ProcessingGptResponseData
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. GPT API 응답 처리 전체 프로세스

```
[사용자 입력] → [API 요청 생성] → [OpenAI GPT 응답 수신(JSON)] → [JSON 파싱] → [데이터 가공] → [사용자에게 결과 표시]
```


## 2. GPT API 응답 구조 (기본 JSON 포맷)

- GPT API는 `ChatCompletion.create()` 요청 시 다음과 같은 JSON 응답을 반환함

```json
{
    "id": "chatcmpl-xxxxxx",
    "object": "chat.completion",
    "created": 1681234567,
    "model": "gpt-3.5-turbo-0301",
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "total_tokens": 50
    },
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "안녕하세요! 무엇을 도와드릴까요?"
            },
            "finish_reason": "stop"
        }
    ]
}
```


## 3. 예제 코드

### 3.1 GPT의 응답을 받아서 가공 및 출력하기 (JSON 응답 파싱)

```python
import openai

openai.api_key = "your_api_key_here"

# 1. API 요청
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "파이썬으로 JSON 파싱하는 예제를 알려줘"}
    ],
    temperature=0.7
)

# 2. 응답 JSON 구조 확인
response_json = response.to_dict_recursive()  # dict로 변환

# 3. 응답 내용 추출
gpt_reply = response_json['choices'][0]['message']['content']

# 4. 사용자에게 가공된 결과 표시
print("GPT 응답 결과:")
print(gpt_reply)
```

### 3.2 GPT 응답을 UI에 표시하기 (간단한 CLI 앱 형태)

```python
def gpt_chat(prompt_text):
    import openai
    openai.api_key = "your_api_key_here"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.5
    )

    # JSON에서 응답 텍스트 추출
    reply = response['choices'][0]['message']['content']
    return reply.strip()

# 사용자 입력 받고 결과 보여주기
if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    result = gpt_chat(user_input)
    print("\nGPT의 응답:")
    print(result)
```

### 3.3 응답 데이터 가공하기

- 예를 들어, GPT가 표 형식 또는 항목 목록으로 응답한 경우, 이를 가공하여 보기 좋게 출력할 수 있음

```python
# 예시 응답: "1. 파이썬\n2. 자바스크립트\n3. 자바"

def parse_list_response(response_text):
    items = response_text.split("\n")
    clean_items = [item.strip().lstrip("1234567890. ").strip() for item in items if item.strip()]
    return clean_items

# 사용
gpt_output = "1. 파이썬\n2. 자바스크립트\n3. 자바"
parsed = parse_list_response(gpt_output)

print("추천 언어 목록:")
for i, item in enumerate(parsed, start=1):
    print(f"{i}. {item}")
```


## 4. GPT 스트리밍 응답 처리 방법

스트리밍(streaming): GPT 응답을 **조각 단위로 실시간으로 받아올 수 있는 기능* (예: 실시간 채팅)

### 4.1 Python SDK 기반 스트리밍 예제

```python
import openai

openai.api_key = "your_openai_api_key_here"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "이 답변을 스트리밍으로 출력해줘"}
    ],
    stream=True  # 스트리밍 활성화
)

for chunk in response:
    if 'choices' in chunk:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            print(delta['content'], end='', flush=True)
```

- `stream=True`: 스트리밍 응답을 활성화
- `delta`: 새로 추가되는 텍스트 조각
- 결과가 천천히 출력되며 자연스럽게 "타이핑되듯" 보임

### 4.2 스트리밍 주의사항
- OpenAI 라이브러리는 기본적으로 SSE (Server-Sent Events)를 사용
- 스트리밍은 응답이 완전히 끝나기 전까지 일부 파라미터를 반환하지 않음 (예: 토큰 수)


- 참고: 응답 파싱 팁

    - OpenAI 응답에서 원하는 텍스트는 다음 위치에 있음

        ```python
        response['choices'][0]['message']['content']
        ```

    - 스트리밍 응답은 텍스트를 조각조각 보내므로 위 방식이 아니라 각 조각에서 `delta['content']` 를 누적해서 출력해야 함


## 요약: 핵심 포인트

| 단계 | 설명 |
|------|------|
| 1. API 호출 | `openai.ChatCompletion.create()` 사용 |
| 2. 응답 파싱 | `response['choices'][0]['message']['content']` 추출 |
| 3. JSON 활용 | dict로 변환 후 필요한 데이터만 필터링 |
| 4. 가공 | 응답 포맷에 맞춰 정제 및 출력 |
| 5. 표시 | CLI, 웹, 앱 등 사용자 인터페이스에 맞게 렌더링 |
