---
layout: page
title:  "LangChain: Ollama 사용해 보기"
date:   2025-04-01 09:00:00 +0900
permalink: /materials/S03-05-03-01_01-LangChainOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 가상환경 만들기

```bash
python -m venv ollama
cd ollama
source ./bin/activate
```

## 2. Ollama 설치 및 실행

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

## 3. 실행 예제

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
    response = ollama.chat(model="openchat", messages=[
        {
            "role": "user",
            "content": "하늘이 왜 푸른지에 대하여 설명해줘. " +
            "앞 문장을 분석하여 긍정적인 내용과 부정적인 내용을 float 형태로 답변 형식에 맞춰서 대답해줘.",
        }
    ])
    print(response['message']['content'])
    ```

- 예제 3

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

- 예제 4

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