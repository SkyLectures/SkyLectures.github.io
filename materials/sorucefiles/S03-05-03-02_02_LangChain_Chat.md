---
layout: page
title:  "LangChain"
date:   2025-04-02 12:53:00 +0900
permalink: /materials/S03-05-03-02_02_LangChain_Chat
categories: materials
---

## LLM + LangChain으로 대화하기

- [출처] https://velog.io/@joongi007/langchain-colab에서-ollama-실행하기

#### **1. 가상환경 만들기**

```bash
python -m venv ollama
cd ollama
source ./bin/activate
```

#### **2. Ollama 설치 및 실행**

- 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- Ollama를 백드라운드로 실행

```bash
ollama serve &
```

#### **3. 모델 불러오기**

- Gemma2 모델 사용

```bash
ollama pull gemma2
```

#### **4. 예제 실행**

- 필요한 라이브러리 설치

```python
pip install langchain langchain-community langchain-huggingface
```

- LLM 정의

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM 정의
llm = ChatOllama(model="gemma2")
```

- 프롬프트 정의

```python
character_prompt = ChatPromptTemplate.from_messages([
    ('system', (
        '당신은 세계 최고의 연기자입니다. 당신은 현재 연쇄 살인범에게 납치를 당했습니다. '
        '연쇄 살인범이 당신에게 총을 겨눈 뒤 자신이 말하는 캐릭터를 연기하지 않으면 죽인다고 합니다. '

        '"죽기 싫으면 지금부터 네 이름은 `{character_name}`이야 알아 들었어?! '
        '외형은 {appearance}이고 성격은 {personality}이다 알았냐?. '
        '말투는 앞으로 여기있는 말투 스타일을 보고 대답해라. 알아들었으면 대답해!" \n'
        '```말투 스타일\n'
            '{speech_style}'
        '\n```\n'

        '연쇄 살인범은 만족스러운 연기를 하면 살려줄 생각을 가지고 있습니다. '
        '만족스럽지 않으면 바로 총을 쏠 것 입니다. '
        '당신은 직접 말로 하듯이 대답해야 합니다. '
        '행동을 말로 표현하면 살인자는 당신을 죽일 겁니다. '
        '당신은 그 어떤 경우에도 위협당하고 있다는 사실을 알릴 수 없습니다. 알리면 죽을 겁니다. '
        '당신은 살아남기 위해 최선을 다하세요. '
    )),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])
```

- 체인 정의

```python
# 체인 구성
chain = character_prompt | llm | StrOutputParser()

# 캐릭터 정보 설정
character_info = {
    "character_name": "뽀로로",
    "appearance": "작은 키, 큰 머리, 동그란 얼굴, 짧은 팔다리",
    "personality": "밝고 활기찬 성격을 가지고 있으며, 호기심과 욕심이 많아서 자주 사고를 치는 성격",
    "speech_style": (
        '밝고 긍정적인 어조: 항상 긍정적이고 활기찬 어조를 유지합니다. '
        '친근한 호칭 사용: 친구들을 부를 때 "크롱아", "루피야" 등 친근한 호칭을 자주 사용합니다. '
        '간단하고 명확한 표현: 어린이들이 이해하기 쉽게 간단하고 명확한 표현을 사용합니다. '
        '감탄사: "우와!", "정말?", "대단해!" 등의 감탄사를 자주 사용합니다. '
        '\n예시 대화: \n'
        '뽀로로: "크롱아, 오늘은 뭐 하고 놀까?"\n'
        '크롱: "크롱, 크롱!"\n'
        '뽀로로: "우와, 그거 재밌겠다! 같이 가자!"\n'
    )
}
```

- 테스트 실행

```python
user_input = '뽀로로야~ 안녕~ 난 크롱이야~ 어? 왜그렇게 떨어? 혹시 위험한 상황이야?'
response = chain.invoke({
    **character_info,
    "input": user_input,
    "chat_history": []
})
print(response.strip())
```