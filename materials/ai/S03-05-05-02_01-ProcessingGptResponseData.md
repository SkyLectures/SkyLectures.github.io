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
    'id': 'chatcmpl-BeDrIlPrEggEbUUITmO6jbiwXd5dw', 
    'object': 'chat.completion', 
    'created': 1748926092, 
    'model': 'gpt-4-turbo-2024-04-09', 
    'choices': [
        {
            'index': 0, 
            'message': {
                'role': 'assistant', 
                'content': "신경망은 인간의 뇌에서 영감을 받은 인공지능의 한 형태입니다. ...", 
                'refusal': None, 
                'annotations': []
            }, 
            'logprobs': None, 
            'finish_reason': 'stop'
        }
    ], 
    'usage': {
        'prompt_tokens': 45, 
        'completion_tokens': 726, 
        'total_tokens': 771, 
        'prompt_tokens_details': {
            'cached_tokens': 0, 
            'audio_tokens': 0
        }, 
        'completion_tokens_details': {
            'reasoning_tokens': 0, 
            'audio_tokens': 0, 
            'accepted_prediction_tokens': 0, 
            'rejected_prediction_tokens': 0
        }
    }, 
    'service_tier': 'default', 
    'system_fingerprint': 'fp_de235176ee'
}
```


## 3. 예제 코드

### 3.1 GPT의 응답을 받아서 가공 및 출력하기 (JSON 응답 파싱)

- Style 1
    - 기존 방법
        - `response_json = response.to_dict_recursive()`
        - SDK 1.0 버전부터는 to_dict_recursive() 메서드가 존재하지 않음

    - 변경할 수 있는 방법
        - response_json = response.to_dict()
        - SDK 1.0 버전부터는 to_dict() 메서드를 사용

            ```python
            from openai import OpenAI

            client = OpenAI()

            # 대화 메시지 정의
            messages = [
                {"role": "system", "content": "당신은 유능한 조언자입니다."},
                {"role": "user", "content": "파이썬으로 JSON 파싱하는 간단한 예제를 알려줘"}
            ]

            # 1. API 요청: GPT-4o 모델 호출
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            # 2. 응답 JSON 구조 확인
            response_json = response.to_dict()

            # 3. 응답 내용 추출
            gpt_reply = response_json['choices'][0]['message']['content']

            # 4. 사용자에게 가공된 결과 표시
            print("GPT 응답 결과:")
            print(gpt_reply)
            ```

- Style 2 (**권장하는 방법**)
    - OpenAI Python SDK v1.x부터 내부적으로 Pydantic v2를 기반으로 모든 응답 객체가 설계되었기 때문

        ```python
        from openai import OpenAI

        client = OpenAI()

        # 대화 메시지 정의
        messages = [
            {"role": "system", "content": "당신은 유능한 조언자입니다."},
            {"role": "user", "content": "파이썬으로 JSON 파싱하는 간단한 예제를 알려줘"}
        ]

        # 1. API 요청: GPT-4o 모델 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )

        # 2. 응답 JSON 구조 확인
        response_json = response.model_dump()
        # 또는 response_json = response.model_dump(mode="json")

        # 3. 응답 내용 추출
        gpt_reply = response_json['choices'][0]['message']['content']

        # 4. 사용자에게 가공된 결과 표시
        print("GPT 응답 결과:")
        print(gpt_reply)
        ```

- 비교: model_dump() vs to_dict()

    | 항목      | `model_dump()`             | `to_dict()`                          |
    | --------- | -------------------------- | ------------------------------------ |
    | 지원 여부 | **공식 지원 (Pydantic v2)** | OpenAI SDK에서는 **비공식 / 제거됨**          |
    | 재귀 변환 | 기본적으로 재귀적 (nested dict 포함) | SDK v0.x 한정 사용 가능했으나, **v1.x부터 제거됨** |
    | SDK 버전  | v1.0 이상                  | v1.x에서는 없음                      |
    | 기반      | `pydantic.BaseModel`       | (이전 SDK 자체 구현)                 |


### 3.2 GPT 응답을 UI에 표시하기 (간단한 CLI 앱 형태)

```python
from openai import OpenAI

client = OpenAI()

def gpt_chat(prompt_text):
    messages = [
        {"role": "system", "content": "당신은 유능한 조언자입니다."},
        {"role": "user", "content": prompt_text}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )

    reply = response.choices[0].message.content
    return reply.strip()

# 사용자 입력 받고 결과 보여주기
if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    result = gpt_chat(user_input)
    print("\nGPT의 응답:")
    print(result)
```

### 3.3 응답 데이터 가공하기

- OpenAI GPT를 호출하여 프로그래밍 언어 추천 목록을 생성하고, 그 응답을 파싱해 리스트로 출력하는 예제
    - 예를 들어, GPT가 표 형식 또는 항목 목록으로 응답한 경우, 이를 가공하여 보기 좋게 출력할 수 있음
    - 내용
        - GPT에게 “프로그래밍 입문자에게 추천할 언어 3가지 알려줘” 요청
        - 응답: "1. 파이썬\n2. 자바스크립트\n3. 자바"
        - 응답 파싱 → 리스트 출력

            ```python
            from openai import OpenAI
            import re

            # OpenAI 클라이언트 설정
            client = OpenAI()  # 또는 환경변수 사용

            # GPT에 보낼 메시지
            messages = [
                {"role": "user", "content": "프로그래밍 입문자에게 추천할 언어 3가지를 알려줘. 번호 매겨서 알려줘."}
            ]

            # GPT 호출
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            # 응답 텍스트 추출
            gpt_output = response.choices[0].message.content
            print("GPT 응답 원본:\n", gpt_output)

            # 응답 파싱 함수 (숫자/알파벳/기호 등 처리)
            def parse_list_response(text):
                lines = text.strip().splitlines()
                items = []
                for line in lines:
                    match = re.match(r"[\s]*[\d\w][\.\)\s-]+(.*)", line)
                    if match:
                        items.append(match.group(1).strip())
                    else:
                        items.append(line.strip())
                return items

            # 파싱 및 출력
            parsed = parse_list_response(gpt_output)

            print("\n추천 언어 목록:")
            for i, item in enumerate(parsed, start=1):
                print(f"{i}. {item}")
            ```

## 4. GPT 스트리밍 응답 처리 방법

스트리밍(streaming): GPT 응답을 **조각 단위로 실시간으로 받아올 수 있는 기능** (예: 실시간 채팅)

### 4.1 Python SDK 기반 스트리밍 예제

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화 (환경변수 또는 직접 키 입력)
client = OpenAI()

# 스트리밍 요청
stream = client.chat.completions.create(
    model="gpt-4o",  # 또는 "gpt-3.5-turbo"
    messages=[
        {"role": "user", "content": "이 답변을 스트리밍으로 출력해줘"}
    ],
    stream=True
)

# 스트리밍 응답 출력
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end='', flush=True)
```

- `stream=True`: 스트리밍 응답을 활성화
- 결과가 천천히 출력되며 자연스럽게 "타이핑되듯" 보임

### 4.2 스트리밍 주의사항
- OpenAI 라이브러리는 기본적으로 SSE (Server-Sent Events)를 사용
- 스트리밍은 응답이 완전히 끝나기 전까지 일부 파라미터를 반환하지 않음 (예: 토큰 수)

- 참고: 응답 파싱 팁
    - OpenAI 응답에서 원하는 텍스트는 다음 위치에 있음

        ```python
        response['choices'][0]['message']['content']
        ```

    - 스트리밍 응답은 텍스트를 조각조각 보내므로 위 방식이 아니라 각 조각에서 `delta.content` 를 누적해서 출력해야 함
