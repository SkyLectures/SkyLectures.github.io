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

### 1.1 프로젝트 준비 및 환경 구성

> - 일반적인 환경 설정을 AI 개발 환경으로 자연스럽게 전환하는 단계
{: .common-quote}

- **의존성 격리의 필수성:** 
    - LLM 생태계(LangChain, LlamaIndex 등)는 업데이트 주기가 매우 빠르고 파괴적 변경(Breaking Changes)이 잦음
    - 가상환경을 쓰지 않고 글로벌 환경에 설치하면, 다른 프로젝트의 라이브러리 버전이 꼬여 기존에 잘 되던 코드까지 망가질 수 있음
    - requirements.txt 또는 poetry를 통해 라이브러리 버전을 고정(LangChain==0.3.x 등)하는 가이드가 선행되어야 함
    - 주요 이슈: 의존성 충돌
        - Streamlit 자체의 Pydantic 버전과 최신 LLM SDK의 Pydantic 버전이 충돌하는 경우가 많음
        - 가상환경 초기 빌드 시 버전을 면밀히 맞추어야 함

- **하이브리드 보안 관리:**
    - 로컬 개발 환경(app.py와 같은 위치)에서는 .env 사용
    - 배포 후 Streamlit Cloud 환경 관리자에서는 secrets.toml 규격에 맞게 Key-Value를 입력
    - st.secrets["OPENAI_API_KEY"]로 통합 호출하는 크로스 플랫폼 코드 작성이 필요함
    - 주요 이슈: API Key 유출 사고
        - .gitignore에 .env나 .streamlit/secrets.toml을 등록하지 않고 GitHub Public 저장소에 Push하여
        - 몇 분 만에 수백 달러의 API 비용이 청구되는 사고가 빈번함
        - 배포 전 반드시 유출 방지 체크리스트를 확인해야 함

- **필수 라이브러리 설치**
    - `pip install streamlit openai langchain python-dotenv`
        - 환경에 따라 langchain-openai 같은 특화 패키지가 추가로 필요할 수 있음
        - streamlit은 프론트엔드 UI를 담당
        - python-dotenv는 보안/환경 파일(.env)을 파이썬 코드가 읽을 수 있도록 중계하는 역할(툴킷의 개념)
    - 환경 변수 관리 로직 구현 (Cross-Platform 코드 구조)
        - 개발자가 로컬에서 테스트할 때와 클라우드에 배포했을 때 코드를 수정하지 않고 둘 다 호환되도록 만드는 예외 처리 코드 필요
        - Streamlit은 내장 기능으로 st.secrets를 지원하므로, 이를 조합한 표준 코드를 프로세스 가이드에 예시로 제공하는 것이 좋음

> - **일반 요령**
>   - 가상환경(`venv`, `conda`) 구축
>   - 필수 라이브러리(`streamlit`, `python-dotenv`) 설치
> - **특화 요령**
>   - API Key 보안 관리.
>       - `.env` 파일을 통해 보안을 유지
>       - Streamlit Cloud 배포 시 `Secrets` 설정 고려
> - **프로세스:**
>   1.  `openai`, `langchain`, `streamlit` 라이브러리 설치
>   2.  `secrets.toml` 또는 `.env`를 통한 환경 변수 관리 로직 구현
{: .common-quote}

- **코드 예시**

    ```python
    import os
    import streamlit as st
    from dotenv import load_dotenv

    # 1. 로컬 환경인 경우 .env 파일을 메모리에 로드 (클라우드 배포 환경이면 무시됨)
    load_dotenv()

    # 2. 두 환경을 모두 지원하는 하이브리드 키 호출 방식
    # st.secrets에 키가 있으면 그걸 쓰고, 없으면 로컬 시스템 환경 변수(os.environ)에서 가져옵니다.
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        st.error("API Key가 설정되지 않았습니다. 로컬의 .env 또는 배포 환경의 Secrets를 확인하세요.")
        st.stop() # 키가 없으면 하단 LLM 로직이 실행되지 않도록 강제 중단
    ```

> - 1.1 단계의 핵심은 **'협업과 배포를 고려한 격리(가상환경)'**와 **'과금 사고 방지를 위한 차단(.env + .gitignore)'**
> - 개발 편의성 때문에 보안을 타협하는 순간 프로젝트 자산에 치명적인 타격이 올 수 있음
> - 이 단계의 프로세스는 체크리스트 형태로 엄격히 준수하도록 가이드하는 것이 좋음
{: .summary-quote}


### 1.2 데이터 및 로직 설계

> - 일반적인 데이터 CRUD를 넘어 LLM의 지식 베이스를 구축하는 단계
> - LLM 애플리케이션의 성능과 지식의 깊이를 결정하는 가장 핵심적인 구간
> - 단순히 API만 호출하는 수준을 넘어, **'우리 기업/프로젝트만의 데이터'를 안전하고 정확하게 AI에게 학습(참고)시키는 RAG 아키텍처의 중추**
{: .common-quote}

- **데이터 소스(PDF, CSV, 웹페이지 등) 선정 및 텍스트 파싱**
    - LLM은 원본 파일(PDF, Excel 등)의 형태를 직접 읽을 수 없음
    - 파일 안에서 오직 '순수한 텍스트 문자열'만 깨끗하게 발라내야 벡터 DB에 넣을 수 있음
    - **파일 포맷에 따라 사용하는 파이썬 라이브러리가 완전히 다름**
        - **PDF:** `PyPDF`, `pdfplumber`, `PyMuPDF` 
            - PDF나 웹페이지는 겉보기와 달리 보이지 않는 공백, 표(Table), 깨진 텍스트가 많음<br>
                🡪 이를 정제하는 전처리(Cleaning) 로직이 RAG 성능의 80%를 결정
            - 데이터의 상태가 나쁘면(예: PDF 스캔본이라 글자가 이미지로 되어 있는 경우) 텍스트 추출이 불가능
                - OCR(`Tesseract`, `EasyOCR`) 도입이 추가로 필요할 수 있음
            - 표나 레이아웃이 복잡하면 `pdfplumber`가 효과적
        - **CSV/Excel:** `Pandas` (`pd.read_csv`, `pd.read_excel`)
        - **웹페이지:** `BeautifulSoup`, `Playwright` 또는 LangChain의 `WebBaseLoader`

- **RAG 파이프라인의 3단계**
    - RAG(검색 증강 생성)의 핵심: 수많은 데이터 중 질문과 가장 관련된 조각을 신속하게 찾아내는 것
    - **3단계 가이드라인**
        1. 텍스트 추출 및 전처리 (Extraction & Cleaning)
            - 단순히 텍스트를 긁어오는 것이 아니라
            - 의미 없는 공백, 줄바꿈 부호(`\n`), PDF 상하단의 페이지 번호나 머리말(Header/Footer) 같은 불순물을 제거하는 코드 처리가 필요
                - 불필요한 데이터까지 벡터 DB에 인덱싱되어 검색 품질을 떨어뜨리는 이슈가 발생
            - `Garbage In, Garbage Out`

        2. 의미 단위로 쪼개기 (`Chunking`)
            - 100페이지짜리 책 한 권을 통째로 LLM에 넣으면 비용도 폭증하고 문맥을 놓치기 쉬움
            - 적절한 크기(예: 500자~1000자)의 '조각(Chunk)'으로 분할해야 함
            - **청킹 전략:**
                - 무조건 글자 수로 쪼개는 것(`CharacterTextSplitter`)보다 문맥을 유지하는 토큰 기준 쪼개기(`RecursiveCharacterTextSplitter`)나 구조를 인식하는 마크다운 기준 쪼개기를 권장
                - **CharacterTextSplitter:**
                    - 단순히 글자 수 기준으로 쪼개기
                    - 문장이 중간에 뚝 끊길 위험이 큼
                - **RecursiveCharacterTextSplitter (추천):**
                    - 문단( `\n\n`), 문장(`.`), 공백(` `) 순서로 중요도를 보며 문맥이 최대한 깨지지 않게 유연하게 쪼개기
                    - 보통 500토큰 크기에 50~100토큰 정도의 오버랩(Overlap, 앞뒤 조각이 겹치는 구간)을 두어 문맥의 연속성을 유지함
            - 유저가 질문했을 때 관련 문서(Chunk)를 제대로 찾아오지 못해, LLM이 엉뚱한 데이터를 기반으로 답변(잘못된 RAG)을 내놓는 현상 발생<br>
                🡪 유사도 임계치(`score_threshold`) 조절 필수

        3. 벡터 DB에 저장 (Embedding & Vector Store)
            - **작동 메커니즘:**
                - 쪼개진 텍스트 조각들을 고차원 숫자 배열인 **임베딩(Embedding)** 벡터로 변환
                    - OpenAI의 `text-embedding-3-small` 등 활용
                - '벡터 데이터베이스'에 적재
                    - 변환된 벡터를 저장하고 빠르게 유사도 검색을 수행할 수 있음
            - **데이터베이스 추천 크기:**
                - 로컬 및 실습 환경, 프로토타입 단계에서는 메모리 기반인 가벼운 `Chroma`, `FAISS`를 활용
                - 상용화 및 엔터프라이즈 환경에서는 `Pinecone`, `Milvus`, `Qdrant` 같은 클라우드/서버형 DB 등 전문 벡터 서버를 사용

- **환각 제어 구조**
    - 일반적인 GPT 모델은 인터넷의 공개 데이터만 학습했기 때문에 사내 보안 문서나 최신 프로젝트 정보에 대해 물으면 환각 현상을 일으킴
    - **작동 프로세스 명확화:**
        1. 유저가 질문을 던짐 
        2. 질문을 임베딩하여 벡터 DB에서 가장 유사한 데이터 조각 3~4개를 먼저 검색 
        3. LLM에게 "너의 상식으로 답하지 말고, 아래 제시된 [참고 문서]의 내용에만 기반해서 답변해줘"라고 시스템 프롬프트를 제어
    - **핵심 가이드:**
        - 프롬프트 템플릿 설계 시 아래와 같은 제약 조건을 명시하도록 유도해야 환각률을 낮출 수 있음
            > "만약 제공된 참고 문서에 유저의 질문에 대한 답이 없다면, 억지로 지어내지 말고 '관련 정보를 찾을 수 없습니다'라고 솔직하게 답변해 주세요."


> - **일반 요령**
>   - 데이터 소스(PDF, CSV, 웹페이지 등)를 선정하고 이를 읽어오는 기능 구현
> - **특화 요령**
>   - **RAG(검색 증강 생성) 파이프라인** 설계
>       - 데이터를 텍스트로 추출
>       - 의미 단위로 쪼개어(`Chunking`)
>       - 벡터 DB에 저장하는 과정을 설계
> - **가이드라인**
>   - 단순히 모델에 묻는 것이 아니라,
>   - "우리 프로젝트만의 데이터"를 AI가 참고하게 하여 환각 현상을 줄이는 구조를 만들어 볼 것
{: .common-quote}


> - 1.2 단계의 본질은 **'LLM에게 오픈북 시험을 볼 수 있도록 정교한 요약 정리본(Vector DB)을 만들어 주는 과정'**
> - 데이터의 질(Quality), 쪼개는 크기(Chunk Size), 문맥 오버랩(Overlap) 세 가지 요소가 RAG 어플리케이션의 최종 성능을 좌우하는 핵심 파라미터임
{: .summary-quote}


### 1.3 UI/UX 구현
> - Streamlit의 특성을 활용해 AI 대화형 경험을 극대화하는 단계
> - 백엔드에서 만든 RAG 로직과 AI 모델을 사용자가 직접 보고 느끼는 '얼굴'을 만드는 단계
> - Streamlit은 프로토타이핑에 최적화되어 있지만, 특유의 작동 메커니즘을 이해하지 못하면 화면이 엉망이 되기 쉬움
{: .common-quote}

- **사이드바(Sidebar), 입력창, 버튼 등 기본 컴포넌트 배치**
    - **구조적 분리(Context Separation):**
        - 대시보드 설계 시 가장 흔히 하는 실수가 메인 화면에 모든 설정 위젯을 쏟아놓는 것
        - 사용자의 주의력을 분산시키지 않기 위한 규칙이 필요함
            - **`st.sidebar`**:
                - LLM 파라미터 제어(모델 종류 선택, `temperature` 조절, 최대 토큰 제한)
                - 시스템 리셋 버튼 등
                - **앱의 전역 설정**을 배치
            - **메인 화면**:
                - 오직 사용자의 질문과 AI의 대화 흐름, 결과 시각화에만 집중하도록 설계

    - **앱의 전역 설정**
        - 웹 애플리케이션 화면 전체나 백엔드 로직 전체에 영향을 주는 제어 위젯들을 한곳으로 모아 관리하는 것
            - AI 답변 온도나 모델 종류처럼 프로그램 전체의 뼈대를 건드리는 중요한 위젯들은 사용자가 대화하는 메인 창에 두지 말고,
            - `st.sidebar` 같은 별도의 환경 설정 영역으로 격리하여 관리하라는 의미
        - 스마트폰의 '설정 앱'이나 웹사이트의 **'환경 설정' 메뉴** 같은 공간 만들기 등

        - **왜 메인 화면이 아닌 다른 곳에 배치할까? (시각적 격리)**
            - 사용자가 AI와 대화하는 메인 화면에 온갖 설정 버튼들이 널려 있으면 화면이 복잡하고 지저분해짐
                - 그래서 Streamlit에서는 접거나 펼칠 수 있는 왼쪽 사이드바 공간(`with st.sidebar:`)을 제공
            - 사이드바에 '전역 설정'들을 몰아서 배치하는 것이 UI/UX 디자인의 표준 패턴

        - **'전역 설정'에 들어가는 요소들은 무엇일까?**
            - '앱 전체의 행동 양식을 결정하는 값'들이 전역 설정에 해당
            - **사용할 AI 모델 선택:**
                - `GPT-4o`, `Claude 3.5 Sonnet`, `Llama 3` 등 어떤 인공지능 모델을 쓸지 결정 (셀렉트박스)
            - **LLM 하이퍼파라미터 조절:**
                - AI의 창의성/무작위성을 조절하는 `Temperature`(온도)
                - 답변의 최대 길이를 제한하는 `Max Tokens` 조절 (슬라이더) 등
            - **데이터 소스 변경:**
                - RAG 파이프라인에서 참조할 지식 베이스를 전환 (라디오 버튼)
                    - 예: '2026년 상반기 보고서.pdf' 또는 '2026년 하반기 보고서.pdf'
            - **시스템 초기화:**
                - 지금까지의 대화 기록을 모두 지우고 처음부터 다시 시작하는 `Clear Chat History` 버튼 등

        - **코드 예시:** 메인 대화창과 설정을 시각적·기능적으로 분리

            ```python
            import streamlit as st

            # [1] 왼쪽 사이드바 영역에 '전역 설정'들을 배치
            with st.sidebar:
                st.header("⚙️ LLM 전역 설정")
                
                # 1. 모델 선택 (앱 전체의 LLM 객체 생성에 영향을 줌)
                selected_model = st.selectbox("사용할 AI 모델", ["GPT-4o", "GPT-3.5-turbo"])
                
                # 2. 온도 설정 (AI의 답변 성향을 전역적으로 통제)
                model_temp = st.slider("답변의 창의성 (Temperature)", 0.0, 1.0, 0.7)
                
                st.divider()
                # 3. 전역 상태 리셋 버튼
                if st.button("🔄 대화 기록 초기화"):
                    st.session_state.messages = []
                    st.rerun()

            # [2] 오른쪽 메인 화면 영역에는 오직 대화창 UI만 배치
            st.title("🤖 나만의 AI 비서")
            st.caption(f"현재 {selected_model} 모델(온도: {model_temp})로 구동 중입니다.")

            # 이후 메인 대화 로직 진행...
            ```

    - **적절한 컴포넌트 매칭:**
        - 사용자가 수치를 직접 입력하게 할 때는 `st.number_input`보다 `st.slider`를 제공
            - 입력 오류(예: 0~1 사이여야 하는 온도를 50으로 입력하는 행위)를 원천 차단하는 디자인 패턴 등 권장

- **채팅 UI 및 상태 관리**
    - Streamlit에서 챗봇 인터페이스를 만들 때 가장 기술적 이해도가 집중되는 구간

    1. **`st.chat_message`와 `st.chat_input`을 통한 친숙한 대화형 경험**
        - **작동 방식:**
            - Streamlit은 챗봇 전용 빌트인 위젯을 제공함
                - `st.chat_input()`:
                    - 화면 최하단에 카카오톡이나 ChatGPT처럼 입력창 고정
                - `with st.chat_message("user"):` 또는 `with st.chat_message("assistant"):`
                    - 블록을 열어 내부 콘텐트를 채우면, 아바타 아이콘과 함께 좌우 배치 폼이 자동으로 잡힘

        - **비주얼 커스텀:**
            - 기본 아바타 외에 회사 로고나 커스텀 이모지(`🏭`, `🤖`)를 아바타 매개변수에 전달하는 등
            - 브랜드 아이덴티티를 살리는 것이 좋음

    2. 세션 상태(`st.session_state`)를 활용한 대화 기록(Chat History) 유지
        - **가장 중요한 렌더링 루프 이해:**
            - Streamlit은 사용자가 입력을 마치고 엔터를 누르는 순간, 스크립트 처음부터 끝까지 다시 실행(Rerun)됨
                - 만약 일반 파이썬 리스트(`chat_history = []`)에 대화 내용을 저장해 두었다면,
                - Rerun이 일어나는 순간 이 변수가 다시 빈 리스트로 리셋되어
                - **직전의 대화가 눈앞에서 증발**

        - **핵심 메커니즘 가이드:**
            - 메모리가 초기화되지 않는 안전 구역인 `st.session_state`에 대화 배열을 최초 1회 생성
            - 스크립트가 새로 실행될 때마다 기존 배열에 저장된 대화 로그를 화면에 `for`문으로 다시 렌더링해 주는 루프 코드 적용

    - **가이드라인에 필수 포함할 Chat UI 표준 루프**

        ```python
        # 1. 세션 상태에 대화 기록 저장소 최초 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 2. 페이지가 Rerun될 때마다 과거 대화 기록을 화면에 순서대로 다시 그림
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 3. 신규 입력 처리
        if user_query := st.chat_input("질문을 입력하세요:"):
            # 유저 메시지 노출 및 세션 저장
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # (이후 이곳에서 LLM 호출 및 답변 처리 진행)
        ```

- **UX 개선 요소**
    - **`st.spinner`와 `st.status`를 활용한 대기 시간 통제**
        - **인지적 대기 시간(Cognitive Wait Time) 줄이기:**
            - LLM이 질문을 해석하고 답변을 뼈대부터 만들어내는 데는 보통 2~5초 이상 소요됨
            - 아무런 표시가 없으면 사용자는 앱이 멈춘 줄 알고 새로고침을 연타하거나 이탈하게 됨
        - **컴포넌트 선택 기준:**
            - **`st.spinner("메시지...")`**:
                - 답변이 만들어지는 동안 로딩 애니메이션 원이 회전하며 앱이 "작업 중"임을 직관적으로 알려주기
            - **`st.status("메시지...")` (추천)**:
                - RAG 파이프라인처럼 내부 동작이 다단계로 일어날 때 효과적
                    - "1. 문서 검색 중...", "2. 프롬프트 생성 중...", "3. LLM 추론 중..." 처럼
                    - 내부 태스크의 진행 상황을 단계별로 확장형 접이식 박스로 보여주어 신뢰감을 극대화

> - **일반 요령**
>   - 사이드바(Sidebar), 입력창(Text Input), 버튼(Button) 등 기본 컴포넌트 배치
> - **특화 요령**
>   - **채팅 인터페이스(Chat UI)** 구현
>       - `st.chat_message`와 `st.chat_input`을 사용하여 친숙한 대화형 경험을 제공
>       - **Session State**를 활용해 대화 기록(Chat History)을 유지
> - **가이드라인**
>   - AI의 답변이 생성되는 동안 사용자에게 '생각 중'임을 알리는 `st.spinner`나 `st.status`를 적절히 배치하여 UX를 개선할 것
{: .common-quote}

> - 1.3 단계의 핵심은 **'파괴적인 Rerun 메커니즘 속에서도 굳건하게 살아남는 대화 화면을 설계하는 것'**
> - `st.session_state`를 활용한 대화 기록 복원 루프와 유저의 불안감을 해소하는 실시간 로딩 인터랙션(`st.status`) 처리가 결합되어야
> - 비로소 상용 서비스 수준의 부드러운 AI 애플리케이션 UX가 완성됨
{: .summary-quote}


### 1.4 불확실성 제어 및 최적화

> - 앞서 보편화된 내용으로 정리했던 '연결 포인트'를 실제 코드로 구현하는 핵심 단계
> - 장난감 수준의 토이 프로젝트와 실제 상용화 가능한 **'프로덕션급 서비스'를 가르는 가장 결정적인 구간**
>   - LLM 기반 앱은 일반적인 웹 앱과 달리 비용(토큰), 네트워크 지연(Latency), 외부 API 불안정성이라는 고유의 리스크를 안고 있기 때문
{: .common-quote}

- **에러 핸들링(`try-except`) 및 예외 상황 처리**
    - LLM 연동 앱은 내 코드가 완벽해도 에러가 발생할 수 있음
        - OpenAI나 Anthropic 같은 외부 서버가 순간적으로 과부하 상태에 빠지거나 점검에 들어가면 내 앱이 통째로 멈출 수 있음

    - **대응 방법**
        - 단순히 `except Exception:`으로 뭉뚱그리는 것이 아니라, 발생 가능한 **LLM 전용 예외 클래스**를 명확히 지정하여 대응
            - 예시
                - `openai.RateLimitError`: 분당 호출 한도 초과 시 🡪 "현재 요청이 많으니 10초 후 다시 시도해 주세요" 안내
                - `openai.AuthenticationError`: API Key 만료나 결제 오류 시 🡪 "시스템 관리자에게 문의하세요" 안내
    - **핵심 가이드:**
        - 에러가 발생했을 때 파이썬 터미널에만 로그를 남기면 사용자는 서비스가 중지된 것으로 인식
            - 반드시 `st.error()` 위젯을 통해 화면에 에러 상황을 표기하도록 함

- **비용 및 속도 최적화**
    - **토큰 절약 (Context Window 관리)**
        - LLM은 대화가 길어질수록 과거 내역을 전부 Context에 다시 입력해야 함
        - 대화가 20턴, 30턴 넘어가면 에러(`ContextWindowExceeded`) 발생
            - 질문 한 번 할 때마다 수만 토큰이 소비되어 **비용이 기하급수적으로 폭증**
            - 모델의 최대 토큰 한도를 초과
        
        - **메모리 다이어트 전략:**
            - **슬라이딩 윈도우 (Sliding Window): `ConversationBufferWindowMemory`**
                - 최신 대화 기록 N개(예: 최근 5개의 대화만)만 남기고 그 이전의 오래된 대화는 과감히 삭제하는 방식
                - 구현이 매우 쉽고 확실하게 토큰을 절약할 수 있음
            - **요약 메모리 (Summarization):**
                - 이전 대화 내용을 또 다른 가벼운 LLM을 시켜 한두 문장으로 압축 요약하게 한 뒤, 그 요약본을 항상 컨텍스트 상단에 얹어두는 방식
                - 맥락은 유지하면서 토큰을 크게 줄일 수 있음

    - **스트리밍(Streaming) 구현**
        - LLM이 문장을 완전히 다 완성한 뒤 화면에 출력(블로킹 방식)하게 하면,
            - 답변이 길 경우 사용자는 5초~10초 동안 빈 화면만 보며 기다려야 함
        - 한 글자씩 실시간으로 뿌려주는 스트리밍을 구현하면 **체감 대기 시간(Time to First Token)이 0.5초 미만으로 단축**되어 UX가 극적으로 좋아짐
            - LangChain의 `astream`이나 OpenAI SDK의 `stream=True` 옵션을 켬
            - Streamlit의 `st.empty()` 공간에 한 글자씩 채워 넣는(`st.write_stream` 활용) 방식 연결
        - **Streamlit 2026 표준 인터페이스:**
            - 파이썬 OpenAI SDK나 LangChain의 `stream()` 메서드는 제너레이터(Generator) 형태로 데이터를 분할하여 반환
            - Streamlit에서는 최신 내장 함수인 `st.write_stream()`을 지원
                - 제너레이터를 이 함수에 전달하기만 하면
                - 복잡한 비동기 loop를 짤 필요 없이 ChatGPT처럼 타자기 효과가 자동으로 화면에 렌더링됨

    - **주요 이슈 (Critical Issues)**
        - **API 타임아웃 및 레이트 리밋(Rate Limit):**
            - LLM 서버의 응답이 늦어지거나 분당 호출 제한에 걸려 앱이 멈추는 현상
            - 백오프(Exponential Backoff) 기반의 재시도 로직이나 예외 처리창(`st.error`) 표기가 필수
        - **비동기 처리 미비로 인한 UI 고사:**
            - 스트리밍이 돌거나 답변을 생성하는 동안 브라우저가 일시적으로 먹통이 되어 사용자가 새로고침을 누르게 만드는 UX 병목이 발생

- **API 호출 실패 시 재시도(Retry) 로직 포함**
    - 네트워크 순시 단절이나 LLM 서버의 일시적인 혼잡은 **1~2초 뒤에 다시 요청하면 성공하는 경우가 대부분**
    - 첫 실패에 바로 에러 화면을 띄우는 대신, 프로그램이 배후에서 자동으로 재시도하게 만들면 서비스의 안정성이 수십 배 올라감
    - **대응 방법**
        - 실패하자마자 바로 재시도하는 것이 아니라, 
            - 1초 쉼 🡪 재시도 실패 🡪 2초 쉼 🡪 재시도 실패 🡪 4초 쉼과 같이 대기 시간을 지수적으로 늘려가며 재시도하는 지수 백오프(Exponential Backoff) 알고리즘이 실무 표준
        - 오픈소스 라이브러리인 `tenacity` 활용
            - LLM 호출 함수 위에 데코레이터(`@retry`) 하나를 얹는 것만으로 이 복잡한 재시도 처리를 단 한 줄에 끝낼 수 있음

> - **일반 요령**
>   - 에러 핸들링(Try-Except) 및 예외 상황 처리
> - **특화 요령**
>   - **토큰 절약**
>       - 대화 기록이 길어질 경우 오래된 기록을 요약하거나 삭제하는 로직 추가
>   - **스트리밍 구현**
>       - 답변이 한 번에 나오지 않고 한 글자씩 출력되게 하여
>       - 사용자가 체감하는 대기 시간을 단축
> - **가이드라인**
>   - API 호출 실패 시 사용자에게 친절한 안내 메시지를 띄우는 재시도(Retry) 로직을 포함할 것
{: .common-quote}

> - 1.4 단계의 본질은 **'통제 불가능한 외부 변수(비용, 인프라 장애)를 예측 가능한 범위 내로 묶어두는 방어적 코딩 기법'**
> - 이 최적화 단계를 거쳐야 비로소 현업 실무자들이 안심하고 상용 배포를 승인할 수 있는 탄탄한 인프라 구조가 완성됨
{: .summary-quote}


### 1.5 배포 및 포트폴리오 자산화

> - 단순한 코드가 아닌 '서비스'로 완성하는 단계
> - 코딩이 끝났다고 프로젝트가 끝난 것이 아님
> - 특히 AI 프로젝트는 배포 환경의 제약과 프롬프트 자산화라는 독특한 영역이 존재
{: .common-quote}

- **인프라 배포**
    - **GitHub 업로드 및 Streamlit Cloud 웹 배포**
        - 전 세계 누구나 URL 링크만 클릭하면 내 AI 서비스를 즉시 써볼 수 있도록 환경을 열어주는 작업
            - "제 로컬 컴퓨터에서는 잘 돌아가는데요"라는 말은 협업이나 포트폴리오에서 통하지 않음
    - **배포 프로세스:**
        1. GitHub에 코드를 올릴 때 오픈 소스 패키지 명세서인 `requirements.txt`가 루트 폴더에 반드시 포함되어야 Streamlit Cloud 서버가 의존성을 자동으로 설치함
        2. 1.1단계에서 다룬 환경 변수(`.env`)는 절대 GitHub에 올리면 안 됨
            - Streamlit Cloud 대시보드의 `Advanced settings 🡪 Secrets` 영역에 해당 API Key를 똑같이 기입해 주는 과정 필수
    - **주요 이슈 (Critical Issues)**
        - **배포 환경의 리소스 제한:**
            - Streamlit Cloud 무료 버전은 컨테이너당 메모리 제한(보통 1GB 내외)이 엄격함
            - 무거운 로컬 벡터 DB(Chroma 등)를 앱 실행 시점에 통째로 로드하면 서버 빌드 시점에 `OOM(Out of Memory)` 에러로 배포가 중단될 수 있음
            - 임베딩 데이터는 경량화하거나 클라우드 DB를 쓰는 등의 인프라 고려가 필요함
        - **콜드 스타트(Cold Start):**
            - 앱을 오랫동안 쓰지 않으면 컨테이너가 절전 모드로 들어감
            - 첫 방문자가 접속할 때 모델 로딩 등의 캐싱 처리가 되어 있지 않으면 초기 구동 시간이 지나치게 길어짐
            - `@st.cache_resource`를 통한 철저한 자원 최적화가 필요

            > - 콜드 스타트 문제
            >   - 인프라/클라우드 관점(본문 내용 참고)
            >   - AI/데이터 관점
            >       - AI 모델 또는 추천 모델 등에서 초기 운영 시 누적된 데이터가 없어서 어느 시점까지는 제대로 된 결과가 나오지 않는 현상

- **README 자산화**
    - 인사담당자나 기술 평가자가 프로젝트의 소스코드를 처음부터 끝까지 다 읽어볼 확률은 극히 낮음
    - 이들의 시선을 사로잡는 것은 잘 정리된 `README.md` 문서

    - **시스템 아키텍처 다이어그램 포함**
        - 내 앱이 데이터를 어떻게 처리하는지 한눈에 보여주는 도식화
            - 예: 사용자 입력 🡪 Streamlit (UI) 🡪 임베딩 모델 🡪 벡터 데이터베이스 (Chroma/Pinecone) 🡪 컨텍스트 추출 🡪 LLM (OpenAI) 🡪 스트리밍 답변 반환
        - 전체 데이터 흐름을 Mermaid(텍스트 기반 다이어그램 툴)나 Draw.io 같은 도구로 시각화하여 첨부
            - 단순 코더가 아닌 'AI 인프라 및 아키텍처 설계 능력을 갖춘 엔지니어'라는 인상을 강하게 줄 수 있음

    - **프롬프트 전략 명시**
        - LLM의 답변 품질을 높이기 위해 어떤 프롬프트 엔지니어링 기법을 적용했는지 기록하는 것
        - AI에게 페르소나를 부여한 *System Prompt*, 답변의 예시를 미리 학습시킨 *Few-shot Prompt*, 혹은 논리적 단계를 거쳐 생각하게 만든 *CoT(Chain-of-Thought)* 전략 등을 명시하여, AI 모델을 정교하게 제어하기 위해 고민한 흔적을 증명

- **UX 증명 및 기술 회고**
    - **데모(Demo) 영상 및 GIF 기록**
        - 외부 API 장애나 크레딧 만료 등으로 인해 내가 배포한 웹사이트가 일시적으로 접속이 안 되거나 느려질 수 있음
            - 평가자가 링크를 클릭해 보기 귀찮을 수도 있음
        - README 상단에 10~20초짜리 깔끔한 구동 GIF(또는 유튜브 링크)가 있으면 **접속하지 않고도 서비스의 완성도를 100% 즉시 확인**시킬 수 있음
            - ScreenToGif, ShareX 같은 무료 도구를 이용해 핵심 인터랙션(질문 입력 후 스트리밍 답변 출력, RAG 데이터 작동 과정) 위주로 짧고 굵게 캡처

    - **기술 회고록 작성 (Engineering Log)**
        - 단순히 "프로젝트를 완성했다"로 끝내지 않고,
        - 개발 과정에서 마주한 **기술적 트레이드오프(Trade-off)와 해결 과정**을 기록하는 진짜 포트폴리오 자산화 단계

        - **핵심 질문 가이드:**
            - 비용과 성능 문제:
                - 왜 가격이 비싼 GPT-4o 대신 상대적으로 저렴한 GPT-4o-mini 모델을 채택했고, 성능 격차를 프롬프트로 어떻게 극복했는가?
            - 환각 통제 문제:
                - RAG 파이프라인 구축 시 검색 유사도 점수(`score_threshold`)를 몇으로 설정했을 때 환각 현상이 가장 적었으며, 데이터 청크 오버랩 크기는 어떻게 최적화했는가?

> - **일반 요령**
>   - GitHub에 코드를 업로드하고
>   - Streamlit Cloud를 통해 웹에 배포
> - **특화 요령**
>   - `README.md`에 프로젝트의 **프롬프트 전략**과 **시스템 아키텍처**를 포함
> - **가이드라인**
>   - **Demo 영상**
>       - 서비스 실행 과정을 GIF나 짧은 영상으로 기록
>   * **회고록 작성**
>       - "왜 특정 모델을 선택했는가?", "환각 현상을 어떻게 제어했는가?"에 대한 기술 블로그 포스팅 링크를 첨부해도 좋음
{: .common-quote}

> - 1.5 배포 단계는 **'내가 만든 결과물의 가치를 세상에 증명하고 반영구적인 내 기술 자산으로 잠금(Lock-in)하는 과정'**
> - 작동하는 실시간 웹 URL, 시각화된 아키텍처 도표, 그리고 문제 해결 과정을 담은 기술 회고록이라는 3박자가 갖춰질 때,
> - 프로젝트는 이력서에서 가장 빛나는 무기가 될 것
{: .summary-quote}


> **참고: 프로젝트 성공을 위한 '체크리스트'의 예시**
> 1. [ &nbsp; ] API 호출 시 발생할 수 있는 네트워크 오류를 처리했는가?
> 2. [ &nbsp; ] 대화가 길어져도 컨텍스트가 유지되도록 `Session State`를 관리했는가?
> 3. [ &nbsp; ] 사용자가 입력한 데이터의 보안(Personal Info 등)을 고려했는가?
> 4. [ &nbsp; ] (RAG 사용 시) 문서의 맥락을 가장 잘 파악할 수 있는 Chunk Size를 찾기 위해 테스트했는가?


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
#//file: "project.py"

import streamlit as st
import pandas as pd
from openai import OpenAI  # Ollama는 OpenAI 호환이므로 OpenAI 인터페이스 활용가능
import PyPDF2
import io

# ==========================================
# 1. 초기 설정 및 UI 레이아웃
# ==========================================
st.set_page_config(
    page_title="로컬 AI 뉴스 분석 에이전트",
    page_icon="🤖",
    layout="wide"
)

# 2026년 UI 가이드라인에 맞춘 모던 스타일 개선
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button { width: 100%; border-radius: 6px; height: 3.2em; background-color: #2b82f6; color: white; font-weight: bold; }
    .report-box { padding: 24px; border-radius: 12px; background-color: white; border: 1px solid #e2e8f0; font-size: 16px; line-height: 1.7; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# 사이드바 구성 (전역 설정 분리)
with st.sidebar:
    st.header("⚙️ Ollama 서버 제어")
    
    # API Key 입력을 빼고, 로컬 URL과 모델 지정을 직관적으로 변경
    ollama_url = st.text_input(
        "Ollama 엔드포인트", 
        value="http://localhost:11434/v1", 
        help="로컬 API 주소입니다. 기본값은 11434 포트의 v1 호환 주소입니다."
    )
    
    model_choice = st.text_input(
        "구동할 로컬 모델명", 
        value="llama3", 
        help="컴퓨터에 다운로드된 모델명(예: llama3, mistral, qwen2.5)을 입력하세요."
    )
    
    st.divider()
    
    persona = st.radio(
        "분석 전문가 페르소나",
        ["금융 투자 전문가", "IT 기술 전략가", "사회/정치 평론가"],
        index=0
    )
    
    with st.expander("LLM 하이퍼파라미터"):
        temperature = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.5)  # 분석을 위해 조금 더 정교하게 조정
        max_tokens = st.number_input("최대 생성 토큰 수", 500, 4000, 2000)

st.title("🗞️ 지능형 로컬 뉴스 분석 관제 플랫폼")
st.markdown("로컬 AI를 활용하여 텍스트 및 PDF 뉴스를 보안 유출 걱정 없이 **비용 0원**으로 정밀 분석합니다.")

# ==========================================
# 2. 핵심 데이터 추출 유틸리티
# ==========================================
def get_text_from_file(uploaded_file):
    """업로드된 파일 객체에서 텍스트 스트링을 안정적으로 추출"""
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
        st.error(f"파일 텍스트 추출 중 치명적 오류 발생: {e}")
        return None

# ==========================================
# 3. 데이터 입력 및 비즈니스 로직
# ==========================================
uploaded_file = st.file_uploader("분석용 리포트/뉴스 원본 업로드 (TXT, PDF 지원)", type=["txt", "pdf"])

if uploaded_file:
    content = get_text_from_file(uploaded_file)
    
    if content:
        with st.expander("📥 파싱된 본문 마스터 데이터 미리보기 (최대 1,000자)"):
            st.text_area(label="Raw Text", value=content[:1000] + "..." if len(content) > 1000 else content, height=150, disabled=True)

        if st.button("🚨 로컬 에이전트 분석 분석 및 요약 기동"):
            # 로컬 Ollama 엔드포인트로 클라이언트 브릿지 생성
            client = OpenAI(base_url=ollama_url, api_key="ollama")
            
            try:
                st.subheader(f"📝 {persona} 분석 리포트")
                
                # 실시간 스트리밍 답변을 렌더링할 빈 컨테이너 우선 확보
                report_placeholder = st.empty()
                
                sys_msg = f"""
                당신은 냉철하고 지능적인 {persona}입니다. 입력된 뉴스 원문을 분석하여 전문 리포트를 작성하세요.
                반드시 한국어로 정중하고 격식 있는 어조로 답변해야 합니다.
                
                출력 포맷 가이드라인:
                ## 1. 핵심 요약 (3줄 요약)
                - [요약 내용 1]
                - [요약 내용 2]
                - [요약 내용 3]
                
                ## 2. 전문가 심층 분석
                (원문의 맥락을 관통하는 통찰을 300자 내외의 마크다운 서식으로 작성하세요)
                """
                
                with st.spinner("로컬 대형 언어 모델(LLM) 추론 엔진 가동 중..."):
                    # [2026년 최신 표준 패턴] stream=True 활성화로 덩어리 응답 유도
                    stream_response = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": content[:8000]}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # [2026 최신 표준 함수] st.write_stream은 제너레이터를 받아 
                    # 화면에 실시간 타자기 효과를 구현하고 최종 완성 텍스트를 반환합니다.
                    with report_placeholder.container():
                        # report-box 스타일 내부에서 실시간으로 글자가 타이핑되도록 매핑
                        # 간단한 전처리를 거친 스트림 제너레이터 함수 정의
                        def chunk_generator():
                            for chunk in stream_response:
                                content_chunk = chunk.choices[0].delta.content
                                if content_chunk:
                                    yield content_chunk
                        
                        final_output = st.write_stream(chunk_generator())
                
                st.success("✅ 분석 프로세스가 성공적으로 완료되었습니다.")
                
                # [2026 가이드 보완] 다운로드 버튼 구조 업그레이드
                # 파일 내보내기 인터랙션 추가 (전체 너비 스트레치)
                st.download_button(
                    label="📥 분석 리포트 다운로드 (CSV/TXT)",
                    data=final_output,
                    file_name=f"Local_AI_{persona}_Report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"엔진 호출 오류: {e}")
                st.info("💡 **문제 해결 팁:** 터미널 창에 `ollama run [모델명]`을 입력하여 로컬 인프라가 살아있는지, 혹은 모델명이 정확한지 대조해 보세요.")
    else:
        st.warning("파일 포맷은 일치하나 내부 텍스트 스트링을 검출하지 못했습니다.")
else:
    st.info("상단의 파일 업로더에 문서가 감지되면 인공지능 분석 파이프라인이 즉시 활성화됩니다.")
```

1. **PyPDF2 라이브러리:**
    - PDF 분석을 위해 꼭 필요 (`pip install PyPDF2` 안내 필수)

2. **에러 핸들링:**
    - `try-except` 구문. 키가 잘못되거나 모델 권한이 없을 때 브라우저에 에러가 예쁘게 표시됨

3. **토큰 관리:**
    - `content[:8000]` 처럼 본문 길이를 제한하는 로직을 넣음
    - 초급자들이 너무 긴 파일을 넣어 API 비용이 폭발하거나 에러가 나는 것을 방지하는 최소한의 장치

- 윈도우 터미널에서 `streamlit run project.py`를 실행


<br>

> - **참고**
>   - **PC 성능으로 인해 로컬 LLM 가동이 어려운 경우**
>       - LLM 모델의 실제 파일 크기가 2GB 이하 정도의 크기인 경우, GPU가 없어도 8~16GB의 RAM에서 쾌적하게 사용할 수 있음
>           - **Qwen 2.5:3B (1.9GB)**
>               - docker exec -it ollama-container ollama run qwen2.5:3b
>           - **Qwen 3.5:2B (2.7GB)**
>               - docker exec -it ollama-container ollama run qwen3.5:2b
>           - **Naver Hyper CLOVAX SEED 0.5B**
>               - docker exec -it ollama-container ollama run hf.co/Mungert/HyperCLOVAX-SEED-Text-Instruct-0.5B-GGUF:Q4_K_M
>               - 명칭이 너무 길어서 불편한 경우 다음과 같이 복사하여 사용할 수 있음
>                   - 형식: docker exec -it <컨테이너명> ollama cp <기존의_긴_이름> <새롭고_짧은_이름>
>                       - docker exec -it ollama-container ollama cp hf.co/Mungert/HyperCLOVAX-SEED-Text-Instruct-0.5B-GGUF:Q4_K_M clova-seed:0.5b
>                   - 원본은 삭제해도 됨
>                       - docker exec -it ollama-container ollama rm hf.co/Mungert/HyperCLOVAX-SEED-Text-Instruct-0.5B-GGUF:Q4_K_M