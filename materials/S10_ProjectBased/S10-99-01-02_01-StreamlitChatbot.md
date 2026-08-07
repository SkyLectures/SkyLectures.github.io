---
layout: page
title:  "Streamlit 기반의 챗봇 인터페이스 개발"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S10-99-01-02_01-StreamlitChatbot
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - Streamlit은 데이터 분석가나 AI 개발자가 복잡한 프론트엔드 지식 없이도 파이썬 코드만으로 빠르고 직관적인 웹 애플리케이션을 구축할 수 있게 해주는 강력한 프레임워크
> - 특히, 내장된 챗봇 UI 컴포넌트들을 활용하면 Agentic Loop나 로컬 LLM 테스트를 위한 인터페이스를 매우 효과적으로 구현할 수 있음
{: .common-quote}


## 1. 관련 기술 및 핵심 개념

- Gemma 4와 같은 로컬 온프레미스 AI를 Streamlit 웹 대시보드로 제어하기 위해 반드시 이해해야 하는 3가지 핵심 메커니즘

- **세션 상태 (Session State - `st.session_state`)**
    - **동작 원리:**
        - Streamlit은 사용자가 상호작용(버튼 클릭, 텍스트 입력 등)을 할 때마다 스크립트 전체를 위에서부터 아래로 다시 실행(Rerun)
        - 일반적인 파이썬 변수는 이 시점에 모두 초기화되어 사라짐
    - **적용:**
        - 이를 방지하기 위해 브라우저 세션 메모리 영역인 `st.session_state`(딕셔너리 객체)를 메모장 삼아 대화 기록 배열(`messages`)을 저장
        - 매 Rerun 시점에 이 배열을 역추적해 화면을 재건축함으로써 대화의 맥락이 단절되지 않도록 보존

- **채팅 요소 (Chat Elements - `st.chat_message`, `st.chat_input`)**
    - **`st.chat_input`:**
        - 화면 하단에 스티키(Sticky) 형태로 밀착 고정되는 카카오톡 스타일의 모던 채팅 입력 바(입력 창)를 생성
        - 사용자가 입력한 순간 조건문(`if prompt := ...`)을 트리거
    - **`st.chat_message`:**
        - 메신저 프로필 레이아웃 컨테이너
        - `"user"`와 `"assistant"` 파라미터 매핑에 따라
        - 사용자(User)와 봇(Assistant)의 메시지를 시각적으로 구분하여 화면에 렌더링 🡪 가독성 높은 UI를 완성함
            - 아이콘(사람/로봇)과 대화 말풍선 정렬 방향을 자동으로 분기함
            

- **실시간 추론 스트리밍 (Real-time Streaming - `st.write_stream` & Generator)**
    - **동작 원리:**
        - LLM이 토큰을 생성할 때마다 실시간으로 화면에 출력하는 효과를 줌
        - 로컬 LLM이 무거운 문장을 완전히 끝맺을 때까지 빈 화면으로 대기하는 병목(Latency)을 해결
    - **적용:**
        - OpenAI SDK의 `stream=True` 응답 청크(Chunk)를 한 글자씩 가로채는 파이썬 **제너레이터(Generator)** 함수(`response_generator()`) 선언
            - 이를 Streamlit의 최신 표준 API인 `st.write_stream()`에 다이렉트로 전달
        - 이는 사용자 경험(UX)을 크게 향상시키며 파이썬의 제너레이터(Generator) 패턴과 함께 사용됨
            - 사용자는 대기 시간 없이 실시간으로 답변이 출력되는 타자기 효과(UX)를 직접 경험할 수 있음


## 2. 인터페이스 설계 및 구조

- 효율적인 인프라 통제와 몰입도 높은 대화 경험을 선사하기 위해
- 화면을 **이원화(Two-column)**하여 레이아웃을 엄격히 통제함

- **사이드바 (Sidebar - 제어 및 인프라 관제 영역)**
    - 챗봇의 동작을 제어하는 설정 영역
        - 메인 대화창의 시각적 노이즈를 완전히 배제
        - AI 런타임을 정밀 튜닝하는 하드웨어 및 모델 설정

    - **주요 역할:**
        - **인프라 주소 제어:**
            - Docker 컨테이너의 포트 및 API 호환 포인트를 통제하는 `Ollama 엔드포인트` 지정
        - **LLM 모델 전환:**
            - 로컬 디스크 및 VRAM 상에 상주 중인 `gemma4` 등 구동 모델명 타이핑 지정
        - **성향 및 창의성 튜닝:**
            - 시스템의 페르소나를 규정하는 `System Prompt` 텍스트 영역
            - AI의 상상력을 제어하는 수치형 슬라이더 `Temperature` 🡪 하이퍼파라미터 조절
        - **컨테이너 청소:**
            - 축적된 메모리 버퍼를 완전히 비워내고 리소스를 초기화하는 `대화 기록 전체 초기화` 버튼(최신 호환 규격 적용) 배치

- **메인 화면 (Main Area - 대화 및 인터랙션 시각화 영역)**
    - 실제 사용자가 로컬 AI 관제 에이전트와 질문을 주고받는 핵심 타임라인
    - **주요 역할:**
        - **상단:**
            - 플랫폼의 정체성을 보여주는 굵직한 Title과 가이드 캡션 명시
                - 인사말 또는 제목 등
        - **중간:**
            - `st.session_state` 메모리에 보관되어 있던 누적 대화 기록이
            - 매 Rerun마다 시간 순서대로 부드럽게 리드로잉(렌더링)되는 영역
        - **하단:**
            - 고정형 채팅 입력창(`st.chat_input`)을 배치하여 유저의 행동 반경을 명확하게 제한


## 3. 모듈별 예제 코드 및 설명

- **모듈 1: 초기화 및 세션 상태 관리**
    - 앱이 처음 실행될 때 대화 기록을 저장할 리스트를 `st.session_state`에 초기화

    - **역할:**
        - 챗봇의 메모리 역할을 하는 **대화 기록 저장소** 🡪 웹 브라우저 메모리 영역에 최초 1회 생성하고 관리함
    - **상세 설명:**
        - Streamlit은 사용자가 글자를 입력할 때마다 전체 코드가 다시 실행됨 🡪 일반 변수에 대화 내용을 담으면 매번 빈 값으로 초기화됨
        - 이를 방지하기 위해 
            - 사용자가 새로고침을 하거나
            - 세션을 종료하기 전까지 데이터가 유지되는 `st.session_state` 활용
        - 코드에서는
            - `'messages'`라는 키가 리스트 형태로 존재하는지 체크
            - 없을 때만 시스템의 첫 가이드 멘트(`"안녕하세요!... gemma4..."`)를 배열에 저장


    ```python
    def init_session_state():
        """챗봇의 대화 기록 세션 저장소 초기화"""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "안녕하세요! 로컬 대형 언어 모델(LLM) **Gemma 4** 기반 관제 에이전트입니다. 무엇을 도와드릴까요?"
                }
            ]
    ```

- **모듈 2: 사이드바 및 전역 환경 설정 UI**
    - Agent 동작이나 모델 변경을 테스트할 수 있도록 제어판을 구성

    - **역할:**
        - 웹 화면 좌측에 접이식 메뉴창을 열어
        - **AI 인프라의 주소, 구동할 모델, 그리고 AI의 성향을 결정하는 파라미터**를 한곳에 격리하여 설정
    - **상세 설명:**
        - 메인 대화창의 시각적 복잡도를 낮추기 위해 `with st.sidebar:` 문법을 통해 UI 영역을 분리
        - 사용자는 Docker 환경에 맞춰 `gemma4` 모델 이름이나 API 주소를 언제든 실시간으로 바꿀 수 있음
            - 다수의 모델이 이미 작동 중일 경우, SelectBox 등을 이용할 수도 있음
        - 수집된 설정값들(`ollama_url`, `model_choice`, `temperature` 등)은 튜플 형태로 메인 루프에 반환(`return`)되어 백엔드 호출 시 전달
        - 누적된 대화 배열을 비워버리고 화면을 강제 갱신하는 `🔄 대화 기록 전체 초기화` 버튼을 배치하여 자원 제어권을 부여

    ```python
    def setup_sidebar():
        """좌측 사이드바 영역: 인프라 주소 및 LLM 하이퍼파라미터 제어"""
        with st.sidebar:
            st.header("⚙️ 런타임 환경 설정")
            st.caption("로컬 인프라 및 에이전트의 성향을 통제합니다.")
            st.markdown("---")
            
            # Docker 컨테이너 엔드포인트 기본값 지정
            ollama_url = st.text_input(
                "Ollama 엔드포인트 주소", 
                value="http://localhost:11434/v1",
                help="Docker로 서빙 중인 Ollama의 OpenAI 호환 API 주소입니다."
            )
            
            # 컨테이너 내부에 성공적으로 적재한 gemma4 지정
            model_choice = st.text_input(
                "구동 모델 태그", 
                value="gemma4",
                help="Ollama 인프라 내에 pull 완료된 모델명을 입력하세요."
            )
            
            system_prompt = st.text_area(
                "시스템 페르소나 (System Prompt)", 
                value="당신은 스마트팩토리 및 AI 기술을 친절하고 명쾌하게 설명하는 유능한 전문 수석 엔지니어입니다.", 
                height=120
            )
            
            with st.expander("추론 파라미터 제어"):
                temperature = st.slider("창의성 (Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                max_tokens = st.number_input("최대 토큰 수 (Max Tokens)", min_value=100, max_value=4000, value=2000)
                
            st.markdown("---")
            
            # [2026 표준 반영]: 기존 경고 발생 옵션인 use_container_width=True 제거 -> width='stretch' 사용
            if st.button("🔄 대화 기록 전체 초기화", width="stretch"):
                st.session_state.messages = []
                st.rerun()
                
            return ollama_url, model_choice, temperature, max_tokens, system_prompt
    ```

- **모듈 3: 대화 기록 렌더링 파트**
    - 저장된 대화 목록을 순회하며 메인 화면에 뿌려줌

    - **역할:**
        - 스크립트가 Rerun될 때마다 **과거에 주고받았던 모든 대화 내역을 화면에 순서대로 복원하여 다시 그려주는(Rendering)** 역할
    - **상세 설명:**
        - 사용자가 새로운 질문을 던지는 순간 코드가 처음부터 다시 돌기 때문에,
            - 과거 대화들이 눈앞에서 사라지지 않도록 `for`문을 돌려 화면을 재작성
        - `st.session_state.messages` 배열에 쌓여있던 딕셔너리 정보들을 하나씩 꺼내어,
            - 역할(`role`)이
                - `"user"`면 사용자 아이콘과 함께 우측에,
                - `"assistant"`면 로봇 아이콘과 함께 좌측에
            - 마크다운 서식으로 배치

    ```python
    def display_chat_history():
        """스크립트가 Rerun될 때마다 기존 세션 기록을 순차적으로 복원하여 드로잉"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    ```

- **모듈 4: 사용자 입력 및 실시간 봇 응답 스트리밍**
    - 사용자 입력을 받고, 가상의 LLM 응답을 제어레이터로 만들어 스트리밍 출력

    - **역할:**
        - 백엔드-프론트엔드 연결의 핵심 모듈
            - 사용자가 입력한 신규 메시지를 받아 화면에 뿌리는 동시에 세션에 저장
            - 로컬 Ollama(Gemma 4) API와 실시간 통신
            - 답변을 한 글자씩 타자기 효과로 렌더링
    - **상세 설명:**
        - **맥락(Context) 전송:**
            - 단순히 유저의 마지막 질문 하나만 보내는 것이 아니라,
            - 모듈 1과 3에서 관리되던 과거 대화 이력 전체(`api_messages`)를 패키징하여 Ollama에 전달
            - 이로 인해 AI가 직전 대화의 문맥을 기억하고 답변할 수 있음
        - **실시간 추론 스트리밍:**
            - `stream=True` 옵션을 통해
                - Ollama가 문장을 다 만들 때까지 기다리지 않고 쪼개진 글자 조각(Chunk)을 던지도록 유도
        - **`st.write_stream`과 제너레이터:**
            - 대화 루프 완성(아래의 내용은 한 턴의 과정)
                - 내부 함수인 `response_generator()`가
                - 실시간으로 들어오는 글자 조각들을 `yield`로 가로채어 양보하면,
                - Streamlit의 최신 표준 함수인 `st.write_stream`이 이를 받아
                - 브라우저에 부드러운 타자기 애니메이션을 연출
                - 출력이 완료되면 최종 텍스트 문자열을 반환받아 세션 상태에 저장함

    ```python
    def handle_user_input(ollama_url, model_choice, temperature, max_tokens, system_prompt):
        """사용자 메시지 접수 및 Ollama API 백엔드 호출 처리"""
        if prompt := st.chat_input("에이전트에게 공정 현황 또는 기술 질문을 입력하세요..."):
            
            # 1) 유저 질문 즉시 드로잉 및 세션 배열 적재
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 2) 어시스턴트 답변 스트리밍 공간 확보
            with st.chat_message("assistant"):
                # OpenAI 호환 로컬 클라이언트 선언
                client = OpenAI(base_url=ollama_url, api_key="ollama")
                
                try:
                    # 대화 히스토리 어레이 포맷으로 전환 (RAG 확장성 고려)
                    api_messages = [{"role": "system", "content": system_prompt}]
                    for msg in st.session_state.messages:
                        api_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Ollama 인프라에 스트림 형태 컴플리션 요청
                    stream_response = client.chat.completions.create(
                        model=model_choice,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True  # 실시간 글자 출력을 위해 스트림 활성화 필수
                    )
                    
                    # 가치가 다른 2026 표준 스트림 렌더링 제너레이터 함수 정의
                    def response_generator():
                        for chunk in stream_response:
                            content_chunk = chunk.choices[0].delta.content
                            if content_chunk:
                                yield content_chunk
                    
                    # st.write_stream을 활용하여 타자기 버퍼 효과 연출 및 데이터 변수 추출
                    full_response = st.write_stream(response_generator())
                    
                    # 3) 정상 출력된 최종 문자열을 세션 상태에 잠금 보존
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"⚠️ Ollama 인프라 연동 에러: {e}")
                    st.info("Docker 컨테이너 상태와 터미널에서 `ollama list` 내역에 모델이 상주해 있는지 대조하세요.")
    ```


## 4. 통합 전체 소스 코드 (app.py)


- **전체 실행 시퀀스 (오케스트레이션)**
    - 사용자가 채팅창에 질문을 입력하고 엔터를 치면,
    - `main()` 함수에 의해 아래 순서로 단 몇 밀리초(ms) 만에 일련의 과정이 진행됨

        <div class="insert-image" style="text-align: left;">
            <img src="/materials/project/images/S10-99-01-02_01-001_StreamlitChatbot.png" style="width: 80%;">
        </div>

        - 코드의 가독성을 높이며
        - 로컬에 축적된 텍스트 데이터셋을 조회하는 **RAG(검색 증강 생성) 파이프라인 컴포넌트**나 자율 추론형 **Agentic Loop** 기술을
        - 중간에 이식하려는 목적에 적합하도록 정형화된 아키텍처 패턴

```python
#//file: "app.py"

import streamlit as st
from openai import OpenAI  # Ollama의 OpenAI 호환 API 인터페이스 활용
import time

# ==========================================
# 0. 페이지 기본 설정 및 모던 UI 스타일링
# ==========================================
st.set_page_config(
    page_title="Gemma 4 AI 관제 챗봇", 
    page_icon="💬", 
    layout="wide"
)

# 2026 디자이너 지침을 반영한 깔끔한 UI 여백 처리
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .report-box { padding: 15px; border-radius: 8px; background-color: white; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 초기화 및 세션 상태 관리 (모듈 1)
# ==========================================
def init_session_state():
    """챗봇의 대화 기록 세션 저장소 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "안녕하세요! 로컬 대형 언어 모델(LLM) **Gemma 4** 기반 관제 에이전트입니다. 무엇을 도와드릴까요?"
            }
        ]

# ==========================================
# 2. 사이드바 및 전역 환경 설정 UI (모듈 2)
# ==========================================
def setup_sidebar():
    """좌측 사이드바 영역: 인프라 주소 및 LLM 하이퍼파라미터 제어"""
    with st.sidebar:
        st.header("⚙️ 런타임 환경 설정")
        st.caption("로컬 인프라 및 에이전트의 성향을 통제합니다.")
        st.markdown("---")
        
        # Docker 컨테이너 엔드포인트 기본값 지정
        ollama_url = st.text_input(
            "Ollama 엔드포인트 주소", 
            value="http://localhost:11434/v1",
            help="Docker로 서빙 중인 Ollama의 OpenAI 호환 API 주소입니다."
        )
        
        # 컨테이너 내부에 성공적으로 적재한 gemma4 지정
        model_choice = st.text_input(
            "구동 모델 태그", 
            value="gemma4",
            help="Ollama 인프라 내에 pull 완료된 모델명을 입력하세요."
        )
        
        system_prompt = st.text_area(
            "시스템 페르소나 (System Prompt)", 
            value="당신은 스마트팩토리 및 AI 기술을 친절하고 명쾌하게 설명하는 유능한 전문 수석 엔지니어입니다.", 
            height=120
        )
        
        with st.expander("추론 파라미터 제어"):
            temperature = st.slider("창의성 (Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            max_tokens = st.number_input("최대 토큰 수 (Max Tokens)", min_value=100, max_value=4000, value=2000)
            
        st.markdown("---")
        
        # [2026 표준 반영]: 기존 경고 발생 옵션인 use_container_width=True 제거 -> width='stretch' 사용
        if st.button("🔄 대화 기록 전체 초기화", width="stretch"):
            st.session_state.messages = []
            st.rerun()
            
        return ollama_url, model_choice, temperature, max_tokens, system_prompt

# ==========================================
# 3. 대화 기록 렌더링 파트 (모듈 3)
# ==========================================
def display_chat_history():
    """스크립트가 Rerun될 때마다 기존 세션 기록을 순차적으로 복원하여 드로잉"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ==========================================
# 4. 사용자 입력 및 실시간 봇 응답 스트리밍 (모듈 4)
# ==========================================
def handle_user_input(ollama_url, model_choice, temperature, max_tokens, system_prompt):
    """사용자 메시지 접수 및 Ollama API 백엔드 호출 처리"""
    if prompt := st.chat_input("에이전트에게 공정 현황 또는 기술 질문을 입력하세요..."):
        
        # 1) 유저 질문 즉시 드로잉 및 세션 배열 적재
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2) 어시스턴트 답변 스트리밍 공간 확보
        with st.chat_message("assistant"):
            # OpenAI 호환 로컬 클라이언트 선언
            client = OpenAI(base_url=ollama_url, api_key="ollama")
            
            try:
                # 대화 히스토리 어레이 포맷으로 전환 (RAG 확장성 고려)
                api_messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Ollama 인프라에 스트림 형태 컴플리션 요청
                stream_response = client.chat.completions.create(
                    model=model_choice,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True  # 실시간 글자 출력을 위해 스트림 활성화 필수
                )
                
                # 가치가 다른 2026 표준 스트림 렌더링 제너레이터 함수 정의
                def response_generator():
                    for chunk in stream_response:
                        content_chunk = chunk.choices[0].delta.content
                        if content_chunk:
                            yield content_chunk
                
                # st.write_stream을 활용하여 타자기 버퍼 효과 연출 및 데이터 변수 추출
                full_response = st.write_stream(response_generator())
                
                # 3) 정상 출력된 최종 문자열을 세션 상태에 잠금 보존
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"⚠️ Ollama 인프라 연동 에러: {e}")
                st.info("Docker 컨테이너 상태와 터미널에서 `ollama list` 내역에 모델이 상주해 있는지 대조하세요.")

# ==========================================
# 5. 엔트리 포인트 오케스트레이션 (Main)
# ==========================================
def main():
    st.title("💬 프라이빗 AI 챗봇 관제 에이전트")
    st.caption("보안 유출 걱정 없는 로컬 온프레미스 인프라 환경 프로토타이핑 대시보드")
    
    # 순서 보장을 위한 오케스트레이션 제어
    init_session_state()
    ollama_url, model, temp, tokens, sys_prompt = setup_sidebar()
    display_chat_history()
    handle_user_input(ollama_url, model, temp, tokens, sys_prompt)

if __name__ == "__main__":
    main()
```

<br>

- **실행 화면**

    <div class="insert-image" style="text-align: left; border: 1px solid lightgray;">
        <img src="/materials/project/images/S10-99-01-02_01-002_StreamlitChatbot.png" style="width: 100%;">
    </div>
    <div class="insert-image" style="text-align: left; border: 1px solid lightgray;">
        <img src="/materials/project/images/S10-99-01-02_01-003_StreamlitChatbot.png" style="width: 100%;">
    </div>
