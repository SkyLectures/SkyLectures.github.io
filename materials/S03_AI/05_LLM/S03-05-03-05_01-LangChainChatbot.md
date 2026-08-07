---
layout: page
title:  "LLM + LangChain으로 대화하기 실습"
date:   2025-04-02 12:53:00 +0900
permalink: /materials/S03-05-03-05_01-LangChainChatbot
categories: materials
---



> - [소스코드 출처] [https://velog.io/@joongi007/langchain-colab에서-ollama-실행하기](https://velog.io/@joongi007/langchain-colab에서-ollama-실행하기){: target="_blank"}
{: .common-quote}


## 1. 인프라 및 가상환경 구축

### [방법 A] 호스트 직접 설치 및 실행

```bash
# 1. 가상환경 생성 및 활성화
python -m venv ollama
cd ollama
source ./bin/activate  # Windows: .\Scripts\activate

# 2. Ollama 설치 및 백그라운드 엔진 가동
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

- `ollama serve &` 명령어는 로컬 호스트(`http://localhost:11434`)에 Ollama 백엔드 데몬을 리스닝 상태로 띄움
- 외부 파이썬 스크립트나 LangChain 프레임워크는 이 엔드포인트와 REST API로 통신하며 로컬 LLM에 추론을 요청하게 됨


### [방법 B] Docker 컨테이너 기반 가동 (추천)

- MFA(스마트팩토리) 인프라나 클라우드 이식성을 고려할 때,
- GPU 가속을 컨테이너 레이어에 격리하여 배포하는 것이 아키텍처 관리 측면에서 유리함

```bash
# NVIDIA GPU 가속을 포함하여 Ollama 컨테이너 구동
docker run -d \
    --gpus=all \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama-service \
    ollama/ollama
```

- **아키텍처적 장점:**
    - `--gpus=all` 옵션을 통해 호스트의 하드웨어 GPU 자원을 컨테이너 내부 `llama.cpp` 가속 엔진으로 직접 매핑
    - `-v` 옵션으로 대용량 LLM 가중치 파일(Model Weights)을 볼륨에 영속화 🡲 컨테이너 재시작 시의 재다운로드 오버헤드를 방지



## 2. LLM 모델 준비 (Gemma 4)

- 컨테이너 환경 또는 로컬 터미널에서 구글의 차세대 오픈소스 모델인 **Gemma 4**를 다운로드

```bash
# 호스트 직접 설치인 경우
ollama pull gemma4

# Docker 환경인 경우 컨테이너 내부 내부 명령어로 실행
docker exec -it ollama-service ollama pull gemma4
```

- Ollama 레지스트리에서 Gemma 4 모델 가중치를 풀링
- 인퍼런스(추론) 효율을 극대화하기 위해 원본 가중치의 정밀도를 낮춘 **4-bit 양자화(Quantization)** 런타임 파일이 기본 탑재됨
- 로컬 워크스테이션 환경에서도 VRAM 소비량을 최소화하며 고성능 추론이 가능함



## 3. LangChain 애플리케이션 구현

1. **라이브러리 설치**

    - 현재 LangChain 생태계의 패키지 분할(Decoupling) 아키텍처에 따라,
    - `langchain-community` 대신 Ollama 공급자 전용 전용 공식 파트너 패키지인 `langchain-ollama`를 필수로 설치해야 함

        ```bash
        pip install langchain langchain-core langchain-ollama
        ```


2. **파이썬 예제 코드**

```python
#//file: "llmchatbot.py"
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==========================================
# 1. Model I/O: 모델 추상화 레이어 정의
# ==========================================
# 주류 파트너 패키지(langchain-ollama)의 ChatOllama 클래스를 명시적으로 사용합니다.
# 내부적으로 대화형 메시지 스펙(System/Human/AI)과 도구 호출 인터페이스를 완벽히 지원합니다.
llm = ChatOllama(
    model="gemma4",
    temperature=0.7  # 연기 장르의 창의적인 답변 생성을 위해 온도를 적절히 조정
)

# ==========================================
# 2. Prompts: 페르소나 및 다중 턴 컨텍스트 설계
# ==========================================
# ChatPromptTemplate의 메시지 구조화를 통해 LLM의 탈옥(Jailbreak)을 방지하고 역할을 강제합니다.
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
    # MessagesPlaceholder는 추후 대화 기록(Memory) 객체가 연속적으로 주입될 수 있는 동적 버퍼 공간입니다.
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

# ==========================================
# 3. Chain: LCEL기반 파이프라인 결합 (RunnableSequence)
# ==========================================
# 파이프(|) 연산자를 통해 컴포넌트를 선언적으로 엮어 컴파일합니다.
# StrOutputParser를 후행에 결합하여 LLM 응답 객체에서 메타데이터를 정제한 순수 문자열만 추출합니다.
chain = character_prompt | llm | StrOutputParser()

# ==========================================
# 4. 데이터 세팅 및 런타임 실행 (Execution)
# ==========================================
# 프롬프트 플레이스홀더에 바인딩될 비즈니스 딕셔너리 데이터
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

user_input = '뽀로로야~ 안녕~ 난 크롱이야~ 어? 왜그렇게 떨어? 혹시 위험한 상황이야?'

# 파이썬 Unpacking 문법(**)을 통해 프롬프트 템플릿에 캐릭터 특성을 한 번에 바인딩하여 호출합니다.
# 현재 단발성 테스트이므로 chat_history 빈 리스트를 명시적으로 인젝션합니다.
response = chain.invoke({
    **character_info,
    "input": user_input,
    "chat_history": []
})

print("==== LLM Response ====")
print(response.strip())
```