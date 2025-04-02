---
layout: page
title:  "ChatGPT API / Ollama 사용 실습"
date:   2025-04-01 09:00:00 +0900
permalink: /materials/S03-05-02-00_LLM_Applications
categories: materials
---

## **1. OpenAI API(ChatGPT API)**

### **1.1 OpenAI API 개요**

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

### **1.2 OpenAI API 사용 방법**

#### **1.2.1 Key 생성 방법**

- OpenAI Platform의 API Key 조회화면에 접속
    - https://platform.openai.com/account/apikeys
- 왼족 메뉴에서 User > API Keys 선택
- 'Create new secret key' 버튼 클릭 → Secret Key 생성창 Open → 'Creaet' 키 눌러서 Secret Key 생성

#### **1.2.2 주의 사항**

- 생성한 API 키는 단 한 곳에서만 사용이 가능함
- 다른 서비스에 사용하려면 새로 발급받아야 함
- 생성한 API 키는 다시 확인이 되지 않으므로 창을 닫기 전에 복사, 저장해 둘것
- 타인에게 절대로 노출 시키지 말것 → <span style="color: red">**쓴 만큼 돈 나감**</span>

#### **1.2.3 API Key 사용하기**

```bash
pip install openai
```

```python
import os
import openai
import streamlit as st

openai.organization = "org-LX1aoE2SdFIlz345GlxDZ6vV"
openai.api_key = os.getenv('OPENAI_API_KEY')
```

- openai.organization은 https://platform.openai.com/account/org-settings 의 왼쪽 메뉴 중 Settings 에서 확인할 수 있음
- openai.api_key 값은 OPENAI_API_KEY 환경변수에 저장된 값을 사용함
    - API Key 발급 후 반드시 저장할 것(다시 볼 수 없음)
    - 직접 입력해도 상관없음

#### **1.2.4 실행 예제**

```python
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # 모델 출력에 대한 무작위성의 정도
    )
    return response.choices[0].message["content"]
```
```python
def my_get_completion(text):
    ans = get_completion(f"""
    ```{text}```
    위 문장을 분석하여 긍정적인 확률, 부정적인 확률을 float 형태로 답변 형식에 맞춰서 대답해줘
    답변 형식
    pos_pect:
    neg_pect:
    """)
    return [float(ans.split(": ")[-1]) for ans in ans.split("\n")]
```
```python
st.markdown("# 감성 분석 사이트")
text = st.text_input('분석 대상 텍스트를 입력하세요', '')
result = my_get_completion(text)
st.success(f'{result[0]}, {result[1]}', icon="✅")
```
<br>
<br>

---

<br>
<br>

## **2. Ollama**

### **2.1 Ollama 개요**

- Ollama란?
    - 대규모 언어 모델(Large Language Model, LLM)을 로컬 환경에서 실행할 수 있도록 설계된 오픈소스 플랫폼
    - 개인 사용자부터 개발자까지 누구나 쉽게 LLM을 활용할 수 있도록 설계된 강력한 도구
    - 클라우드 기반 API에 의존하지 않고 자신의 PC에서 직접 LLM을 실행할 수 있음

- 주요 특징
    - 다양한 모델 지원
        - Llama 2, Llama 3, Mistral, CodeLlama 등 여러 LLM 사용 가능
        - 사용자 필요에 따라 모델을 선택하고 커스터마이징 가능
    
    - 로컬 실행
        - macOS, Windows, Linux 등 다양한 운영체제에서 실행 가능
        - Docker를 통한 배포 지원
    
    - 사용자 정의 기능
        - 모델의 프롬프트와 설정을 사용자 맞춤으로 조정 가능
        - REST API를 통해 외부 애플리케이션과 통합 가능
    
    - 간단한 설치 및 사용
        - 명령어 한 줄로 모델 실행 가능 
            - ollama run <모델명>
        - 필요한 모델은 자동으로 다운로드 및 설치됨
        
- 설치 방법
    - Windows: 현재 프리뷰 버전 제공
    - Linux: 스크립트를 통해 설치 (curl 명령어 사용)
    - macOS: Homebrew 또는 다운로드 페이지에서 설치
    - Docker: 공식 이미지를 사용하여 컨테이너 환경에서 실행
    
- 활용 사례
    - 텍스트 생성, 번역, 질의응답, 요약 등 다양한 작업
    - 코드 생성 및 디버깅에 특화된 CodeLlama 같은 모델 활용 가능
    - 멀티모달 입력(텍스트와 이미지) 지원
    
### **2.1 Ollama 사용해 보기**

#### **2.1.1 가상환경 만들기**

```bash
python -m venv ollama
cd ollama
source ./bin/activate
```

#### **2.1.2 Ollama 설치 및 실행**

- 설치
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- 설치된 모델 확인
```bash
ollama list
```

- 모델 선택하여 실행
    - 예제에서는 openchat 모델 사용
```bash
ollama run openchat
```

#### **2.1.3 실행 예제**

- 예제 1
```python
import ollama
response = ollama.chat(model='openchat', messages=[
  {
    'role': 'user',
    'content': '하늘은 왜 푸른지 설명해줘.',
  },
])
print(response['message']['content'])
```

- 예제 2
```python
import requests

def query_ollama(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    response_data = response.json()
    print(response_data)
    # return response_data['response']

if __name__ == "__main__":
    model = "openchat"
    prompt = "하늘은 왜 푸르게 보일까?"
    result = query_ollama(model, prompt)
    # print(result)
```

- 예제 3
```python
import requests
import argparse

def query_ollama(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    response_data = response.json()
    print(response_data)
    # return response_data['response']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input prompt.")
    parser.add_argument("prompt", type=str)
    model = "openchat"
    args = parser.parse_args()
    # prompt = "하늘은 왜 푸르게 보일까?"
    result = query_ollama(model, args.prompt)
    # print(result)
```