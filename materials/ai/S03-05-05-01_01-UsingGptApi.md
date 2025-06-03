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
        - ~~gpt-3.5/gpt-3.5-turbo: 빠른 응답과 효율적인 성능 제공~~
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
        - 사용량에 따라 종량제 요금 부과(GPT-4는 1,000 토큰당 약 $0.03, GPT-35는 $0.02)
        - 운영정책에 따라 계속 바뀌는 중

## 2. OpenAI API 키 발급 (2025.06.01 기준, 변경된 내용 적용됨)

1. OpenAI 계정 생성
    1. [https://auth.openai.com/create-account](https://auth.openai.com/create-account) 에서 계정 생성
    2. 로그인 후 대시보드로 이동

2. API 키 발급
    1. [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) 접속
        - 직접 찾아가려면: OpenAI 사이트 → 우측 상단 → Dashboard → 좌측 사이드바 중단 → API Keys 클릭
    2. "Create new secret key" 버튼 클릭 → 생성된 키를 복사해서 저장
        - <span style="color: #AA0000">**주의 사항**</span>
            - 생성한 API 키는 단 한 곳에서만 사용이 가능함
            - 다른 서비스에 사용하려면 새로 발급받아야 함
            - 생성한 API 키는 다시 확인이 되지 않으므로 창을 닫기 전에 복사, 저장해 둘것
            - 타인에게 절대로 노출 시키지 말것 → <span style="color: red">**쓴 만큼 돈 나감**</span>




## 3. 최신 코드 스타일에서의 API 키 설정 방법 (2025.06.01 기준)

### 3.1 코드 내에서 명시적으로 설정

```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key-here")
```

### 3.2 환경변수로 설정(공식 권장 방식)

```bash
export OPENAI_API_KEY=your-api-key-here  # Linux/macOS
set OPENAI_API_KEY=your-api-key-here     # Windows CMD
$env:OPENAI_API_KEY="your-api-key-here"  # Windows PowerShell
```

```python
from openai import OpenAI
client = OpenAI()  # 환경변수에서 자동으로 API 키를 가져옴
```

### 3.3 환경변수 삭제

#### 3.3.1 Linux/macOS
- 일시적으로 설정한 환경변수 삭제(현재 세션에서만 유지된 것)

    1. 삭제

        ```bash
        unset OPENAI_API_KEY
        ```

    2. 확인: 아무것도 출력되지 않으면 삭제된 것

        ```bash
        echo $OPENAI_API_KEY
        ```

- 영구적으로 설정한 환경변수 삭제(.bashrc, .zshrc, .bash_profile에 저장된 것)
    1. 홈 디렉토리에서 `.bashrc`, `.zshrc`, `.profile` 등 열기:

        ```bash
        nano ~/.bashrc       # 또는 ~/.zshrc, ~/.profile
        ```

    2. 다음과 같은 라인 제거:

        ```bash
        export OPENAI_API_KEY="your-api-key"
        ```

    3. 파일 저장 후 셸 적용:

        ```bash
        source ~/.bashrc     # 또는 ~/.zshrc
        ```

#### 3.3.2 Windows PowerShell
- 일시적으로 설정한 환경변수 삭제(현재 세션에서만 유지된 것)
    - 삭제

        ```powershell
        $Remove-Item Env:OPENAI_API_KEY
        ```

    - 확인: 아무것도 출력되지 않으면 삭제된 것

        ```powershell
        $Env:OPEN_API_KEY
        ```

- 영구적으로 설정한 환경변수 삭제(레지스트리에 저장된 것)
    - 사용자 계정 기준

        ```powershell
        [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $null, "User")
        ```

    - 시스템 전체 기준(관리자 권한 필요)

        ```powershell
        [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $null, "Machine")
        ```

#### 3.3.3 Windows CMD

- 일시적으로 설정한 환경변수 삭제(현재 세션에서만 유지된 것)
    - 삭제

        ```bat
        set OPENAI_API_KEY=
        ```

    - 확인: 아무것도 출력되지 않으면 삭제된 것

        ```bat
        echo %OPENAI_API_KEY%
        ```

- 영구적으로 설정한 환경변수 삭제(레지스트리에 저장된 것)
    - CMD 자체로는 불가능
    - PowerShell 또는 시스템 환경 변수 설정 창에서 삭제
        1. `Win + R` → `sysdm.cpl`
        2. `고급`탭 → `환경 변수(N)...`
        3. 사용자 또는 시스템 변수에서 `OPEN_API_KEY`를 찾아서 삭제


## 4. API 요청 방법

- OpenAI의 GPT API는 HTTPS 요청 기반의 **REST API**임
- `chat/completions` 엔드포인트를 통해 GPT를 사용할 수 있음

### 4.1 기본 API 요청 구조

- 요청 URL

    ```text
    https://api.openai.com/v1/chat/completions
    ```

- HTTP 헤더

    ```text
    Authorization: Bearer YOUR_API_KEY
    Content-Type: application/json
    ```

- HTTP Body (JSON)

    ```json
    {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": "당신은 유능한 조수입니다."},
            {"role": "user", "content": "신경망이 어떻게 작동하는지 간단한 용어로 설명하세요."}
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
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": "당신은 유능한 조언자입니다."},
            {"role": "user", "content": "신경망이 어떻게 작동하는지 간단한 용어로 설명하세요."}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    ```

### 4.2 최신 스타일의 코드 구조

- 현재, OpenAI 공식 문서의 Quickstart 예제는 최신 SDK 구조를 따라 변경된 것이 제공되고 있음
    - 객체 기반 방식으로 구현된 **openai.OpenAI()** 사용
    - 최근에 openai 라이브러리의 버전이 바뀌면서 클래스 기반 사용법으로 전환되었기 때문
    - 앞에서 제공된 예제는 공식적인 방식을 따르고 있으므로 제대로 작동하고 있으나 점차 최신의 코드로 변경되고 있음

- 예제 코드:

    ```python
    from openai import OpenAI

    # 환경변수 사용 시 아래 코드로 충분
    client = OpenAI()

    # 명시적 API 키 전달 (보안에 유의)
    # client = OpenAI(api_key="YOUR-API-KEY-HERE")

    # 대화 메시지 정의
    messages = [
        {"role": "system", "content": "당신은 유능한 조언자입니다."},
        {"role": "user", "content": "신경망이 어떻게 작동하는지 간단한 용어로 설명하세요."}
    ]

    # GPT-4o 모델 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )

    # 출력
    print(response.choices[0].message.content)
    ```

## 5. GPT-4 Turbo API 활용을 위한 기본 지식

### 5.1 메시지 구조

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Tell me a joke." },
  { "role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!" }
]
```

### 5.2 각 메시지의 역할(role)

| 역할 (`role`)         | 설명                                                               |
| ------------------- | ---------------------------------------------------------------- |
| `system`            | 모델의 **성격, 지침, 행위 스타일**을 정의함<br>예: “You are a medical assistant.” |
| `user`              | 사용자의 질문 또는 명령 (실제 입력 내용)                                         |
| `assistant`         | 모델의 이전 응답 (선택적, 과거 문맥 제공용)                                       |
| `tool` / `function` | 도구 호출 시 사용 (함수 호출 기능 활용 시)                                       |
| `tool_call`         | 모델이 함수 호출을 요청할 때 자동 생성됨 (Function calling에서 사용)                  |


### 5.3 메모리 / 문맥 관리

- **전체 메시지 리스트가 모델의 “기억”**
- 모델은 메시지를 처음부터 끝까지 읽고 그에 따라 응답
- 따라서 길어진 대화를 유지하려면 전체 히스토리를 계속 보내야 함(자동 저장 없음)
- 토큰 제한에 주의해야 함 (gpt-4o는 최대 128k 토큰까지 가능)
    - 대략적인 토큰 개념
        - 1,000 tokens ≒ 750~800단어
        - 예시: "Hello, how are you?" → 약 5 tokens

### 5.4 주요 파라미터 설명
- OpenAI GPT-4 Turbo API (예: gpt-4o, gpt-4-turbo)의 주요 파라미터

| 파라미터      | 설명                                                     | 예시                            |
| ------------- | -------------------------------------------------------- | ------------------------------- |
| `model`       | 사용할 모델 ID                                           | `"gpt-4o"`, `"gpt-4-turbo"`     |
| `messages`    | 대화 메시지 리스트 (역할 + 내용 구조)                    | `[{"role": "user", "content": "Hello!"}]` |
| `temperature` | 생성 다양성 (랜덤성), 0~2<br>낮을수록 일관된 답변       | `0.7` (기본값), `0` (결정적), `1.5` (창의적) |
| `top_p`       | Top-P 샘플링(nucleus sampling) 사용 비율 (0~1)<br>보통 `temperature`와 함께 사용 | `1.0`   |
| `max_tokens`  | 응답의 최대 토큰 수                                      | `512`, `2048` 등   |
| `stop`        | 생성 중단할 토큰들 (1~4개 문자열)                       | `["\n", "User:"]`  |
| `stream`      | 응답을 스트리밍 방식으로 받을지 여부                     | `True` or `False`  |

### 5.5 요금 구조 (2024년 기준)
- **입력 (prompt):** $0.0015 / 1,000 tokens
- **출력 (completion):** $0.002 / 1,000 tokens

> 요금은 변경될 수 있으므로 [공식 가격 페이지](https://openai.com/pricing) 참고

### 5.6 기타 참고 사항

- 하루 사용량 한도 존재 (무료 요금제의 경우 낮음)
- 에러 발생 시:
  - `401`: API 키 오류
  - `429`: 속도 제한 초과
  - `500`: 서버 내부 오류 (재시도 필요)
