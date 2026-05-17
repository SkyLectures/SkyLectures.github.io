---
layout: page
title:  "Streamlit 기반의 챗봇 인터페이스 개발"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S10-99-01-02_01-StreamlitChatbot
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



Streamlit은 데이터 분석가나 AI 개발자가 복잡한 프론트엔드 지식 없이도 파이썬 코드만으로 빠르고 직관적인 웹 애플리케이션을 구축할 수 있게 해주는 강력한 프레임워크입니다. 특히, 내장된 챗봇 UI 컴포넌트들을 활용하면 Agentic Loop나 로컬 LLM 테스트를 위한 인터페이스를 매우 효과적으로 구현할 수 있습니다.

요청하신 Streamlit 기반 챗봇 인터페이스의 핵심 개념부터 모듈별 구현, 그리고 전체 통합 소스코드까지 체계적으로 정리해 드립니다.

---

## 1. 관련 기술 및 핵심 개념

성공적인 챗봇 UI를 구축하기 위해 Streamlit에서 제공하는 다음 세 가지 핵심 개념을 이해하는 것이 중요합니다.

* **세션 상태 (Session State - `st.session_state`):** Streamlit은 사용자가 상호작용(버튼 클릭, 텍스트 입력 등)을 할 때마다 코드를 처음부터 다시 실행합니다. 따라서 이전 대화 내용을 유지하려면 `st.session_state`라는 딕셔너리 객체에 메시지 기록을 저장하여 상태를 보존해야 합니다.
* **채팅 요소 (Chat Elements - `st.chat_message`, `st.chat_input`):**
* `st.chat_input`: 화면 하단에 고정되는 채팅 입력창을 제공합니다.
* `st.chat_message`: 사용자(User)와 봇(Assistant)의 메시지를 시각적으로 구분하여 화면에 렌더링하는 컨테이너입니다.


* **스트리밍 출력 (Streaming Output - `st.write_stream`):** LLM이 토큰을 생성할 때마다 실시간으로 화면에 출력하는 효과를 줍니다. 이는 사용자 경험(UX)을 크게 향상시키며 파이썬의 제너레이터(Generator) 패턴과 함께 사용됩니다.

## 2. 인터페이스 설계 및 구조

효율적인 테스트와 사용을 위해 화면을 두 가지 주요 영역으로 나눕니다.

* **사이드바 (Sidebar):** 챗봇의 동작을 제어하는 설정 영역입니다.
* **역할:** 시스템 프롬프트 설정, 모델 선택(예: Local vLLM, API 등), Temperature 등 하이퍼파라미터 조절.


* **메인 화면 (Main Area):** 실제 대화가 이루어지는 영역입니다.
* **역할:** 상단에는 인사말 또는 제목, 중간에는 누적된 대화 기록(`st.session_state` 기반 렌더링), 하단에는 채팅 입력창 배치.



---

## 3. 모듈별 예제 코드 및 설명

전체 구조를 이해하기 쉽도록 4개의 논리적 모듈로 나누어 설명합니다.

### 모듈 1: 초기화 및 상태 관리

앱이 처음 실행될 때 대화 기록을 저장할 리스트를 `st.session_state`에 초기화합니다.

```python
import streamlit as st
import time

# 1. 초기화 및 상태 관리
def init_session_state():
    # 'messages' 키가 없으면 빈 리스트로 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 초기 인사말 추가 (선택 사항)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "안녕하세요! 무엇을 도와드릴까요?"
        })

```

### 모듈 2: 사이드바 및 환경 설정 UI

Agent 동작이나 모델 변경을 테스트할 수 있도록 제어판을 구성합니다.

```python
# 2. 사이드바 구성
def setup_sidebar():
    with st.sidebar:
        st.header("⚙️ 챗봇 설정")
        # 모델 선택
        model_choice = st.selectbox(
            "모델 선택",
            ["Qwen-Coder-Local", "GPT-4-API", "Ollama-Llama3"]
        )
        # 하이퍼파라미터 제어
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # 설정값 초기화 버튼
        if st.button("대화 기록 지우기"):
            st.session_state.messages = []
            st.rerun() # 화면 새로고침
            
        return model_choice, temperature

```

### 모듈 3: 대화 기록 렌더링

저장된 대화 목록을 순회하며 메인 화면에 뿌려줍니다.

```python
# 3. 대화 기록 출력
def display_chat_history():
    for message in st.session_state.messages:
        # role(user/assistant)에 따라 다른 아이콘과 스타일로 렌더링됨
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

```

### 모듈 4: 사용자 입력 및 봇 응답 스트리밍 처리

사용자 입력을 받고, 가상의 LLM 응답을 제어레이터로 만들어 스트리밍 출력합니다.

```python
# 4. 사용자 입력 및 봇 응답 처리
def handle_user_input():
    # 사용자가 메시지를 입력하고 엔터를 치면 prompt에 값이 할당됨
    if prompt := st.chat_input("메시지를 입력하세요..."):
        
        # 1) 사용자 메시지 화면에 출력 및 상태 저장
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2) 봇 응답 처리
        with st.chat_message("assistant"):
            # 실제 LLM 연동 시에는 API 스트리밍 응답 객체를 사용합니다.
            # 여기서는 제너레이터를 사용하여 스트리밍 효과를 흉내 냅니다.
            def generate_mock_response():
                mock_text = f"'{prompt}'에 대한 가상의 스트리밍 응답입니다. 실제 구현 시에는 이 부분을 로컬 모델 추론 코드나 API 호출로 대체하시면 됩니다."
                for word in mock_text.split():
                    yield word + " "
                    time.sleep(0.05) # 토큰 생성 지연 효과
            
            # 스트리밍 출력 및 전체 응답 텍스트 반환
            full_response = st.write_stream(generate_mock_response())
            
        # 3) 봇 메시지 상태 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})

```

---

## 4. 통합 전체 소스 코드 (app.py)

위의 모듈들을 하나로 합친 실행 가능한 전체 코드입니다. 이 코드를 `app.py`로 저장하고 터미널에서 `streamlit run app.py`를 실행하면 웹 브라우저에서 인터페이스를 확인할 수 있습니다.

```python
import streamlit as st
import time

# 페이지 기본 설정
st.set_page_config(page_title="AI Chat Interface", page_icon="💬", layout="wide")

def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 시스템 테스트를 위한 챗봇 인터페이스입니다."}]

def setup_sidebar():
    """좌측 사이드바 설정 영역"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        model_choice = st.selectbox("엔진 선택", ["Local VLLM (Qwen)", "Ollama", "External API"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        system_prompt = st.text_area("System Prompt", "당신은 유능한 코딩 어시스턴트입니다.", height=150)
        
        st.markdown("---")
        if st.button("🔄 세션 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        return model_choice, temperature, system_prompt

def display_chat_history():
    """기존 대화 기록 렌더링"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(model_choice, temperature, system_prompt):
    """채팅 입력 및 스트리밍 응답 처리"""
    if prompt := st.chat_input("질문이나 코드를 입력해주세요..."):
        # 1. 사용자 입력 처리
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Assistant 응답 처리
        with st.chat_message("assistant"):
            # TODO: 실제 LLM 로직 연동 포인트
            # 예: response = local_llm.stream(prompt, temp=temperature, system_prompt=system_prompt)
            
            def mock_stream():
                response_text = f"""
**[설정 정보]**
* 모델: `{model_choice}`
* 온도: `{temperature}`

'{prompt}'에 대해 처리 중입니다. 이 부분은 VSCode Continue 확장과 같이 **Local GPU (Ollama/vLLM)** 환경의 응답 스트림으로 쉽게 교체할 수 있도록 설계되었습니다. Agentic Loop의 중간 추론 과정을 출력하는 용도로도 활용 가능합니다.
"""
                for chunk in response_text.split(' '):
                    yield chunk + ' '
                    time.sleep(0.03)

            # write_stream이 제너레이터를 받아 화면에 실시간으로 그리고, 최종 문자열을 반환
            response = st.write_stream(mock_stream())
            
        # 3. 응답 기록 저장
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.title("💬 Streamlit AI Agent Interface")
    st.caption("LLM 및 Agentic Workflow 테스트를 위한 대화형 프로토타입")
    
    init_session_state()
    model_choice, temp, sys_prompt = setup_sidebar()
    display_chat_history()
    handle_user_input(model_choice, temp, sys_prompt)

if __name__ == "__main__":
    main()

```

위의 뼈대 코드(TODO 주석 부분)에 로컬 LLM을 연동하시면 훌륭한 개인용 "Stock-Ops" 대시보드나 코딩 어시스턴트의 프론트엔드로 즉시 확장이 가능합니다.

현재 구상 중이신 환경에서는 백엔드로 로컬의 Ollama나 vLLM 중 어떤 방식을 우선적으로 연동하여 테스트해 보실 계획이신가요?