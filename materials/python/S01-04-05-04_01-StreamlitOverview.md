---
layout: page
title:  "Streamit의 개요와 기초 사용법"
date:   2026-07-01 10:00:00 +0900
permalink: /materials/S01-04-05-04_01-StreamlitOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. Streamlit 개요

- **Streamlit이란?**
    - 데이터 사이언스, 머신러닝, AI 분야의 연구자나 SW 개발자가 복잡한 프론트엔드(HTML/CSS/JavaScript) 지식 없이도<br>
        **Python 코드 몇 줄만으로 빠르고 대화형인 웹 애플리케이션을 구축할 수 있게 해주는 오픈소스 Python 라이브러리**
    - 특히 AI 모델 데모, 데이터 시각화 대시보드, 프로토타입 시스템을 빠르게 프로덕션 수준으로 구현할 때 압도적인 생산성을 자랑함
    - 과거에는 데이터 분석 결과나 AI 모델을 웹으로 시각화하기 위해서 Django, Flask, FastAPI 같은 백엔드 프레임워크와 React, Vue 같은 프론트엔드 프레임워크를 연동해야 했으나,
    - Streamlit은 이 과정을 대폭 단순화하여 데이터 제품의 프로토타이핑 시간을 극적으로 단축시킴

- **Streamlit의 주요 특징과 장점**
    - Streamlit이 기존의 일반적인 웹 프레임워크와 차별화되는 가장 큰 특징은 **데이터 흐름(Data Flow) 방식의 아키텍처**를 가진다는 점

    - **Pure Python:**
        - 오직 Python만으로 UI 컴포넌트와 비즈니스 로직을 모두 작성

    - **UI 디자인 자동화 (No Front-end code):**
        - 개발자가 버튼, 차트, 사이드바의 HTML/CSS 위치를 직접 지정하지 않아도,
        - Streamlit이 내부적으로 디자인 가이드라인에 맞추어
        - 유려하고 반응형인 UI 레이아웃을 자동으로 배치함

    - **스크립트 전체 재실행 (Rerun from top to bottom)**
        - Streamlit 앱은 웹 페이지에서 사용자가 버튼을 클릭하거나, 슬라이더를 움직이는 등 위젯과 상호작용(Interaction)할 때마다
            - Python 소스 코드 전체가 위에서 아래로 처음부터 끝까지 다시 실행(Rerun)됨

        - **장점:**
            - 복잡한 상태 관리(State Management)나 콜백 함수를 촘촘히 설계하지 않아도 되므로,
            - 코드가 직관적이고 단순한 선형적 흐름을 유지함
        - **단점:**
            - 위젯을 하나만 움직여도 무거운 머신러닝 모델을 다시 로드하거나 대용량 데이터를 다시 읽어오는 비효율이 발생할 수 있음
                - 이를 해결하기 위해 '캐싱' 기술이 필수적으로 사용됨

    - **캐싱 (Caching)을 통한 성능 최적화**
        - 전체 재실행으로 인한 병목 현상을 막기 위해 Streamlit은 강력한 캐싱 메커니즘을 제공함
        - 특정 함수에 캐싱 데코레이터를 지정하면,
            - 함수의 입력 매개변수가 바뀔 때만 함수를 재실행하고
            - 그렇지 않으면 이전 실행 결과를 메모리에서 즉시 꺼내와 웹 페이지에 렌더링

        - **`@st.cache_data`:**
            - 파일 로드, 데이터프레임 변환, 일반 API 연산 등 '데이터 자체'를 캐싱할 때 사용
        - **`@st.cache_resource`:**
            - 머신러닝/DL 모델 객체 로드, DB 커넥션 생성 등 공유 리소스나 네트워크 연결을 유지해야 할 때 사용

    - **풍부한 위젯 지원:**
        - `st.slider`, `st.button`, `st.selectbox`, `st.text_input` 등
        - 사용자의 입력을 받아 변수로 바로 할당할 수 있는
        - 다양한 위젯을 한 줄의 코드로 구현

    - **시각화 라이브러리와의 높은 호환성:**
        - Matplotlib, Seaborn, Plotly, Bokeh, Altair 등 Python 생태계의 주요 데이터 시각화 라이브러리로 그린 차트를
        - `st.plotly_chart()`와 같은 전용 함수를 통해
        - 웹상에 대화형(Interactive) 상태로 손쉽게 띄울 수 있음

    - **실시간 변경 반영 (Live Reloading):**
        - 개발 환경에서 `.py` 소스 코드를 수정하고 저장하면,
        - 실행 중인 웹 브라우저가 이를 감지하여
        - 변경사항을 실시간으로 화면에 업데이트함


- **웹 프레임워크 간 비교**

    | 비교 항목 | 전통적 웹 (Django/Flask) | Plotly Dash | **Streamlit** |
    | --- | --- | --- | --- |
    | **주 사용 언어** | Python + HTML/CSS/JS | Python (대부분) | **Pure Python** |
    | **UI 제어력** | 매우 높음 (완전 커스텀) | 높음 (HTML/CSS 컴포넌트 제어) | **중간 (정해진 템플릿/레이아웃)** |
    | **학습 곡선** | 가파름 (Front/Back 지식 필요) | 완만함 (React 개념 일부 필요) | **매우 낮음 (Python 기본만 알면 가능)** |
    | **개발 속도** | 느림 (설계 및 구축 시간 소요) | 중간 | **매우 빠름 (몇 시간 내 프로토타입 완성)** |
    | **주요 목적** | 대규모 프로덕션 서비스 웹앱 | 복잡하고 정밀한 데이터 대시보드 | **AI 데모, 빠른 데이터 시각화 앱** |

<br>

> - **요약 및 활용처**
>   - Streamlit은 "가장 최소한의 코딩으로, 가장 빠르게 아이디어를 웹 앱으로 검증하는 도구"
>   - 다음과 같은 시나리오에서 현업 생산성을 극대화하는 데 자주 활용됨
>       - AI/ML 모델의 추론 결과를 클라이언트나 비개발자 팀원에게 시각적으로 보여주는 **데모(Demo) 페이지**
>       - 기업 내부에서 수집되는 원시 데이터(Raw Data)나 스마트팩토리 센서 데이터를 실시간으로 모니터링하는 **사내 대시보드**
>       - 정식 웹 서비스를 론칭하기 전, 핵심 기능만 빠르게 검증하는 **MVP(Minimum Viable Product, 최소 기능 제품) 개발**
{: .summary-quote}


## 2. 개발 환경 설정 및 실행 방법

- **설치하기**

    ```bash
    pip install streamlit
    ```

- **실행하기**
    - 스크립트를 실행할 때는 `python` 명령어가 아닌 `streamlit run` 명령어를 사용

        ```bash
        streamlit run app.py
        ```

    - 실행 시 로컬 서버(`http://localhost:8501`)가 켜지며 브라우저에서 실시간으로 결과를 확인할 수 있음
    - 코드를 수정하고 저장하면 브라우저에 실시간 반영(Live Reloading)됨


## 3. 기초 사용 방법 및 주요 API 익히기

- **텍스트 및 제목 작성**

```python
import streamlit as st

# 웹 페이지 제목 및 헤더
st.title("Streamlit 기초 가이드")
st.header("1. 텍스트 요소 출력")
st.subheader("서브 헤더 영역입니다.")

# 일반 텍스트 및 마크다운 지원
st.text("일반 텍스트 출력 함수입니다.")
st.markdown("마크다운을 지원하여 **굵은 글씨**나 *이탤릭체*, `code` 표현이 가능합니다.")
```

- **데이터 및 테이블 출력**
    - Streamlit은 판다스 데이터프레임(DataFrame)을 매우 깔끔한 대화형 테이블로 시각화함

    ```python
    import pandas as pd
    import numpy as np

    st.header("2. 데이터 및 테이블")

    # 샘플 데이터 생성
    df = pd.DataFrame(
        np.random.randn(10, 3),
        columns=['A', 'B', 'C']
    )

    # 대화형 테이블 (정렬, 검색 가능)
    st.dataframe(df)

    # 정적 테이블 (단순 노출용)
    st.table(df)
    ```

- **다양한 사용자 입력 위젯**
    - 사용자의 입력을 받아 변수에 할당하는 과정이 매우 직관적

    ```python
    st.header("3. 입력 위젯 활용")

    # 버튼
    if st.button("클릭해 보세요"):
        st.write("버튼이 클릭되었습니다!")

    # 체크박스
    show_data = st.checkbox("데이터 보기 선택")
    if show_data:
        st.write("체크박스가 활성화되었습니다.")

    # 라디오 버튼 & 셀렉트 박스
    choice = st.radio("가장 좋아하는 언어는?", ["Python", "Java", "C++"])
    st.write(f"선택한 언어: {choice}")

    option = st.selectbox("기기를 선택하세요", ["스마트폰", "태블릿", "노트북"])
    st.write(f"선택한 기기: {option}")

    # 슬라이더 및 텍스트 입력
    age = st.slider("나이를 선택하세요", 0, 100, 25)
    st.write(f"선택된 나이: {age}")

    user_input = st.text_input("이름을 입력하세요", "홍길동")
    st.write(f"입력된 이름: {user_input}")
    ```

- **레이아웃 구성 (사이드바 및 컬럼)**
    - 화면을 분할하거나 사이드바를 제어하는 것도 간단한 컨텍스트 매니저(`with`)문으로 가능

    ```python
    st.header("4. 레이아웃 분할")

    # 사이드바 구성
    with st.sidebar:
        st.header("사이드바 메뉴")
        sidebar_input = st.text_input("사이드바 입력창")
        st.write(f"입력값: {sidebar_input}")

    # 컬럼 나누기 (좌우 배치)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Left Column")
        st.write("여기는 왼쪽 영역입니다.")

    with col2:
        st.subheader("Right Column")
        st.write("여기는 오른쪽 영역입니다.")
    ```

- **성능 최적화를 위한 캐싱 (Caching)**
    - Streamlit은 위젯이 변경될 때마다 전체 스크립트를 재실행함
    - 만약 대용량 ERP 데이터를 로드하거나 복잡한 AI 모델을 추론하는 로직이 있다면, 매번 재실행할 때 심각한 병목이 발생
    - 이를 방지하기 위해 데코레이터를 제공함

        ```python
        @st.cache_data
        def load_large_data():
            # 이 함수는 처음 한 번만 실행되고, 이후에는 캐싱된 데이터를 바로 반환합니다.
            df = pd.read_csv("large_dataset.csv")
            return df

        @st.cache_resource
        def load_ai_model():
            # 머신러닝 모델 객체나 데이터베이스 연결 등은 cache_resource를 사용합니다.
            model = load_my_llm_model()
            return model
        ```

        - **`@st.cache_data`**: 데이터프레임, API 응답, 텍스트 등 일반적인 '데이터'를 캐싱할 때 사용
        - **`@st.cache_resource`**: ML 모델, DB 커넥션 등 한 번 로드 후 재사용해야 하는 '서버 리소스'를 캐싱할 때 사용

