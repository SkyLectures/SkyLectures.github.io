---
layout: page
title:  "Streamit를 활용한 LLM 연동 어플리케이션"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S10-99-01-01_01-StreamLlmAppGuideline
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 프로세스 & 가이드라인

> - Streamlit은 Python 환경에서 LLM 모델의 결과물을 시각화하고 사용자 인터페이스를 구축하는 데 가장 효율적인 도구
>   - 그러나 꼭 Streamlit을 사용해야 하는 것은 아님 🡲 자신에게 맞는 것, 시간 내에 적용해 볼 수 있는 것으로 적당히 선택할 것
{: .common-quote}


### 1.1 프로젝트 준비 및 환경 구성 (Setup)

> - 일반적인 환경 설정을 AI 개발 환경으로 자연스럽게 전환하는 단계
{: .common-quote}

- **일반 요령**
    - 가상환경(`venv`, `conda`) 구축
    - 필수 라이브러리(`streamlit`, `python-dotenv`) 설치

- **특화 요령**
    - API Key 보안 관리.
        - `.env` 파일을 통해 보안을 유지
        - Streamlit Cloud 배포 시 `Secrets` 설정 고려

- **프로세스:**
    1.  `openai`, `langchain`, `streamlit` 라이브러리 설치
    2.  `secrets.toml` 또는 `.env`를 통한 환경 변수 관리 로직 구현


### 1.2 데이터 및 로직 설계 (Core Logic)

> - 일반적인 데이터 CRUD를 넘어 LLM의 지식 베이스를 구축하는 단계
{: .common-quote}

- **일반 요령**
    - 데이터 소스(PDF, CSV, 웹페이지 등)를 선정하고 이를 읽어오는 기능 구현

- **특화 요령**
    - **RAG(검색 증강 생성) 파이프라인** 설계
        - 데이터를 텍스트로 추출
        - 의미 단위로 쪼개어(`Chunking`)
        - 벡터 DB에 저장하는 과정을 설계

- **가이드라인**
    - 단순히 모델에 묻는 것이 아니라,
    - "우리 프로젝트만의 데이터"를 AI가 참고하게 하여
    - 환각 현상을 줄이는 구조를 만들어 볼 것


### 1.3 UI/UX 구현 (Frontend with Streamlit)
> - Streamlit의 특성을 활용해 AI 대화형 경험을 극대화하는 단계
{: .common-quote}

- **일반 요령**
    - 사이드바(Sidebar), 입력창(Text Input), 버튼(Button) 등 기본 컴포넌트 배치

* **특화 요령**
    - **채팅 인터페이스(Chat UI)** 구현
        - `st.chat_message`와 `st.chat_input`을 사용하여 친숙한 대화형 경험을 제공
        - **Session State**를 활용해 대화 기록(Chat History)을 유지

* **가이드라인**
    - AI의 답변이 생성되는 동안 사용자에게 '생각 중'임을 알리는 `st.spinner`나 `st.status`를 적절히 배치하여 UX를 개선할 것


### 1.4 불확실성 제어 및 최적화 (Troubleshooting)

> - 앞서 보편화된 내용으로 정리했던 '연결 포인트'를 실제 코드로 구현하는 핵심 단계
{: .common-quote}

* **일반 요령**
    - 에러 핸들링(Try-Except) 및 예외 상황 처리

* **특화 요령**
    - **토큰 절약**
        - 대화 기록이 길어질 경우 오래된 기록을 요약하거나 삭제하는 로직 추가
    * **스트리밍 구현**
        - 답변이 한 번에 나오지 않고 한 글자씩 출력되게 하여
        - 사용자가 체감하는 대기 시간을 단축

* **가이드라인**
    - API 호출 실패 시 사용자에게 친절한 안내 메시지를 띄우는 재시도(Retry) 로직을 포함할 것


### 1.5 배포 및 포트폴리오 자산화 (Deployment)

> - 단순한 코드가 아닌 '서비스'로 완성하는 단계
{: .common-quote}

* **일반 요령**
    - GitHub에 코드를 업로드하고
    - Streamlit Cloud를 통해 웹에 배포

* **특화 요령**
    - `README.md`에 프로젝트의 **프롬프트 전략**과 **시스템 아키텍처**를 포함
    
* **가이드라인**
    - **Demo 영상**
        - 서비스 실행 과정을 GIF나 짧은 영상으로 기록
    * **회고록 작성**
        - "왜 특정 모델을 선택했는가?", "환각 현상을 어떻게 제어했는가?"에 대한 기술 블로그 포스팅 링크를 첨부해도 좋음


> 💡 프로젝트 성공을 위한 '체크리스트'
>
> 1. [ &nbsp; ] API 호출 시 발생할 수 있는 네트워크 오류를 처리했는가?
> 2. [ &nbsp; ] 대화가 길어져도 컨텍스트가 유지되도록 `Session State`를 관리했는가?
> 3. [ &nbsp; ] 사용자가 입력한 데이터의 보안(Personal Info 등)을 고려했는가?
> 4. [ &nbsp; ] (RAG 사용 시) 문서의 맥락을 가장 잘 파악할 수 있는 Chunk Size를 찾기 위해 테스트했는가?
{: .summary-quote}




## 2. 미니 프로젝트 주제

- **주제: 지능형 멀티모달 뉴스 분석 및 브리핑 에이전트**
- **[외부 데이터 활용(RAG 맛보기) + 멀티 에이전트 역할 분담 + 자동화된 문서 생성]**

- **핵심 요소**
    - **Context Window 활용 (입력의 다양화):**
        - 단순히 텍스트를 복사-붙여넣기 하는 게 아니라,
        - **PDF 파일 업로드 기능**을 추가하여 실제 업무용 도구의 느낌을 줌

    - **Role-playing Prompting (에이전트 설계):**
        - 사용자가 선택한 페르소나(예: 냉철한 투자 분석가 vs 따뜻한 뉴스 앵커)에 따라
        - 동일한 뉴스도 다른 톤으로 분석하게 함

    - **Functionality (결과물 저장):**
        - 분석 결과를 화면에 보여주는 데서 끝나지 않고,
        - **PDF 파일로 다운로드** 받거나 이미지와 결합된 **카드 뉴스 형태**로 시각화


- **윈도우 환경에서 Streamlit을 설치, 실행**

    - **1단계: 파이썬(Python) 확인**
        - 윈도우 명령 프롬프트(CMD) 또는 PowerShell을 열고 파이썬이 설치되어 있는지 확인

        ```bash
        python --version
        ```

        > - 만약 설치되어 있지 않다면 [python.org](https://www.python.org/)에서 최신 버전을 설치하고, 
        > - 설치 시 **"Add Python to PATH"** 옵션을 반드시 체크해야 함


    - **2단계: 프로젝트 폴더 생성 및 가상 환경 설정**
        - 프로젝트별로 라이브러리가 꼬이지 않도록 가상 환경을 만드는 것을 권장

        1.  **폴더 생성:** `mkdir my-ai-app` 후 `cd my-ai-app`
        2.  **가상 환경 생성:** 
            ```bash
            python -m venv venv
            ```
        3.  **가상 환경 활성화:**
            - **CMD:** `venv\Scripts\activate`
            - **PowerShell:** `.\venv\Scripts\Activate.ps1`
            - **주의:** 
                - PowerShell에서 보안 오류가 뜬다면 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 명령어를 한 번 실행


    - **3단계: Streamlit 설치**
        - 활성화된 가상 환경 상태(`(venv)`가 표시됨)에서 설치를 진행

        ```bash
        pip install streamlit
        ```
        
        - 설치가 끝난 후 다음 명령어로 데모 화면이 뜨는지 확인

        ```bash
        streamlit hello
        ```
        *브라우저가 자동으로 열리며 풍선이 날아다니는 화면이 나오면 성공!*


    - **4단계: 내 첫 앱 만들고 실행하기**

        1.  **코드 작성:** 메모장이나 VS Code를 열어 `app.py` 파일을 만듦

            ```python
            import streamlit as st
            st.title("윈도우에서 실행하는 AI 비서")
            st.write("반갑습니다! Streamlit이 정상 작동 중입니다.")
            ```

        2.  **실행:** 터미널에서 아래 명령어를 입력

            ```bash
            streamlit run app.py
            ```

- **윈도우 사용자들을 위한 팁**

    - **포트 번호:**
        - 기본값은 `8501`
        - 브라우저 주소창에 `localhost:8501`을 입력하여 접속
        - 파일 수정 후 반영
            - `app.py` 코드를 수정하고 저장(`Ctrl + S`)하면,
            - 브라우저 우측 상단에 **'Always rerun'** 옵션이 뜸
            
    - **종료 방법:**
        - 터미널 창에서 `Ctrl + C`를 누르면 서버가 중단됨
        - 이를 클릭하면 코드 수정 시 실시간으로 웹 화면이 갱신됨



- **실습 진행**

1. **환경 설정**

- 준비 사항 (Windows 기준)
    - 폴더 생성: C:\ai_project 폴더 생성
    - 파일 생성: 해당 폴더에 app.py 파일 생성 (아래 코드 복사)
    - 필수 라이브러리 설치: 터미널(CMD)에서 실행

    ```Bash
    pip install streamlit openai PyPDF2
    ```

2. **UI 설계 및 파일 업로드**
    - **Streamlit Layout:** 사이드바(설정), 메인 화면(결과) 분리
    - **File Uploader:** 뉴스 기사 텍스트뿐만 아니라 `.txt` 또는 `.pdf` 파일을 업로드하는 기능 구현

        ```python
        uploaded_file = st.file_uploader("분석할 문서를 업로드하세요", type=['txt', 'pdf'])
        ```

3. **지능형 분석 로직 구현**
    - **Persona Selector:**
        - `st.selectbox`로 분석 전문가의 성격 선택
    - **Structured Output:**
        - 결과를 단순히 출력하지 않고 '요약', '인사이트', '비판적 시각' 등으로 구조화
        - JSON 형태로 받거나 프롬프트로 강제함

4. **멀티모달 시각화**
    - **DALL-E 3 연동:**
        - 분석 내용 중 가장 핵심적인 장면을 프롬프트로 자동 추출하여 삽화 생성
    - **Data Visualization:**
        - (초급자용) 뉴스 내 언급된 키워드 빈도를 간단한 차트로 표시

5. **서비스 패키징**
    - **Download Button:**
        - 생성된 요약문과 이미지를 하나로 합친 리포트 만들기
    - **LLM 파라미터 조절:** 
        - `Temperature` 슬라이더를 통해 AI의 '창의성' 수치를 조절하며 결과 변화 관찰



6. **OpenAI API 응용서비스: 지능형 뉴스 에이전트 풀 소스**

```python
import streamlit as st
from openai import OpenAI
import PyPDF2
import io

# ==========================================
# 1. 초기 설정 및 UI 레이아웃
# ==========================================
st.set_page_config(
    page_title="AI 뉴스 분석 에이전트",
    page_icon="🤖",
    layout="wide"
)

# 커스텀 CSS로 UI 디자인 개선
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .report-box { padding: 20px; border-radius: 10px; background-color: white; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# 사이드바 구성
with st.sidebar:
    st.header("🌟 Control Panel")
    api_key = st.text_input("OpenAI API Key", type="password", help="sk-... 형식의 키를 입력하세요.")
    
    st.divider()
    
    model_choice = st.selectbox("LLM 엔진", ["gpt-4o-mini", "gpt-4o"], index=0)
    persona = st.radio(
        "분석 전문가 선택",
        ["금융 투자 전문가", "IT 기술 전략가", "사회/정치 평론가"],
        index=0
    )
    
    with st.expander("세부 설정"):
        temperature = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.7)
        max_tokens = st.number_input("최대 토큰 수", 500, 4000, 2000)

st.title("🗞️ 지능형 멀티모달 뉴스 분석 에이전트")
st.info("뉴스 텍스트나 PDF 파일을 업로드하면 AI 전문가가 분석 리포트와 삽화를 생성합니다.")

# ==========================================
# 2. 핵심 유틸리티 함수
# ==========================================
def get_text_from_file(uploaded_file):
    """업로드된 파일에서 텍스트를 추출하는 함수"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            return text
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

# ==========================================
# 3. 메인 로직 서비스 파트
# ==========================================
uploaded_file = st.file_uploader("분석할 파일을 업로드하세요 (TXT, PDF)", type=["txt", "pdf"])

if uploaded_file:
    content = get_text_from_file(uploaded_file)
    
    if content:
        with st.expander("업로드된 본문 미리보기"):
            st.write(content[:1000] + "..." if len(content) > 1000 else content)

        if st.button("전문가 분석 및 삽화 생성 시작"):
            if not api_key:
                st.error("사이드바에 OpenAI API Key를 입력해 주세요.")
            else:
                client = OpenAI(api_key=api_key)
                
                # 화면 분할
                col1, col2 = st.columns([6, 4])
                
                try:
                    # [Step 1] LLM 분석 수행
                    with st.spinner(f'{persona}의 관점으로 분석 중...'):
                        sys_msg = f"""
                        당신은 {persona}입니다. 입력된 뉴스 내용을 바탕으로 리포트를 작성하세요.
                        출력 형식:
                        1. 요약: 핵심 내용 3줄 요약
                        2. 분석: 전문가적 시각에서의 상세 분석 (300자 내외)
                        3. 삽화 묘사: 이 뉴스를 상징하는 DALL-E 3용 영문 프롬프트 (단문)
                        """
                        
                        chat_response = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": sys_msg},
                                {"role": "user", "content": content[:8000]} # 컨텍스트 제한 고려
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        analysis_result = chat_response.choices[0].message.content
                        
                        # 프롬프트 추출 (마지막 줄 근처에서 가져오기)
                        lines = analysis_result.split('\n')
                        img_prompt = lines[-1].replace("삽화 묘사:", "").strip()
                        if len(img_prompt) < 10: # 제대로 안 뽑혔을 경우 대비
                            img_prompt = f"An artistic professional illustration about {persona} analyzing news."

                    # [Step 2] 결과 출력
                    with col1:
                        st.success("✅ 분석 완료")
                        st.markdown(f"### 📝 {persona} 리포트")
                        st.markdown(f"<div class='report-box'>{analysis_result}</div>", unsafe_allow_html=True)
                        st.download_button("결과 다운로드", analysis_result, file_name="ai_news_report.txt")

                    with col2:
                        st.subheader("🎨 뉴스 삽화 생성")
                        with st.spinner('DALL-E 3 이미지 생성 중...'):
                            img_response = client.images.generate(
                                model="dall-e-3",
                                prompt=img_prompt,
                                size="1024x1024",
                                n=1
                            )
                            st.image(img_response.data[0].url, caption="Generated by DALL-E 3")
                            st.info(f"**AI Prompt:** {img_prompt}")

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("파일의 내용을 읽을 수 없습니다.")
else:
    st.info("파일을 업로드하면 분석이 시작됩니다.")
```

1. **PyPDF2 라이브러리:**
    - PDF 분석을 위해 꼭 필요 (`pip install PyPDF2` 안내 필수)

2. **API Key 보안:**
    - 강의 시 화면 공유 중에 본인의 API Key가 노출되지 않도록 주의

3. **에러 핸들링:**
    - `try-except` 구문. 키가 잘못되거나 모델 권한이 없을 때 브라우저에 에러가 예쁘게 표시됨

4. **토큰 관리:**
    - `content[:8000]` 처럼 본문 길이를 제한하는 로직을 넣음
    - 초급자들이 너무 긴 파일을 넣어 API 비용이 폭발하거나 에러가 나는 것을 방지하는 최소한의 장치

- 윈도우 터미널에서 `streamlit run app.py`를 실행