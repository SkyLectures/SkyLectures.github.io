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
    - Streamlit은 이 과정을 대폭 단순화하여 데이터 제품의 프로토타이핑 시간을 극적으로 단축시킴<br><br>

    <div class="insert-image">
        <img src="/materials/python/images/S01-04-05-04_01-001_StreamlitOverview.png" style="width: 90%;">
    </div>

- **Streamlit의 주요 특징과 장점**
    - Streamlit이 기존의 일반적인 웹 프레임워크와 차별화되는 가장 큰 특징은 **데이터 흐름(Data Flow) 방식의 아키텍처**를 가진다는 점

    - **Pure Python:**
        - 오직 Python만으로 UI 컴포넌트와 비즈니스 로직을 모두 작성

    - **UI 디자인 자동화 (No Front-end code):**
        - 개발자가 버튼, 차트, 사이드바의 HTML/CSS 위치를 직접 지정할 필요가 없음
        - Streamlit이 내부적으로 디자인 가이드라인에 맞추어 유려하고 반응형인 UI 레이아웃을 자동으로 배치함

    - **스크립트 전체 재실행 (Rerun from top to bottom)**
        - Streamlit 앱은 웹 페이지에서 사용자가 버튼을 클릭하거나, 슬라이더를 움직이는 등 위젯과 상호작용(Interaction)할 때마다
            - Python 소스 코드 전체가 위에서 아래로 처음부터 끝까지 다시 실행(Rerun)됨

        - **장점:**
            - 복잡한 상태 관리(State Management)나 콜백 함수를 촘촘히 설계하지 않아도 됨
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

<br>

- **웹 프레임워크 간 비교**

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">비교 항목</th>
        <th style="width: 250px;">전통적 웹 (Django/Flask)</th>
        <th style="width: 250px;">Plotly Dash</th>
        <th style="width: 300px;">Streamlit</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">주 사용 언어</td>
            <td>Python + HTML/CSS/JS</td>
            <td>Python (대부분)</td>
            <td>Pure Python</td>
        </tr>
        <tr>
            <td class="td-rowheader">UI 제어력</td>
            <td>매우 높음 (완전 커스텀)</td>
            <td>높음 (HTML/CSS 컴포넌트 제어)</td>
            <td>중간 (정해진 템플릿/레이아웃)</td>
        </tr>
        <tr>
            <td class="td-rowheader">학습 곡선</td>
            <td>가파름 (Front/Back 지식 필요)</td>
            <td>완만함 (React 개념 일부 필요)</td>
            <td>매우 낮음 (Python 기본만 알면 가능)</td>
        </tr>
        <tr>
            <td class="td-rowheader">개발 속도</td>
            <td>느림 (설계 및 구축 시간 소요)</td>
            <td>중간</td>
            <td>매우 빠름 (몇 시간 내 프로토타입 완성)</td>
        </tr>
        <tr>
            <td class="td-rowheader">주요 목적</td>
            <td>대규모 프로덕션 서비스 웹앱</td>
            <td>복잡하고 정밀한 데이터 대시보드</td>
            <td>AI 데모, 빠른 데이터 시각화 앱</td>
        </tr>
    </tbody>
</table>
</div>

> - Plotly Dash
>   - Python, R, Julia 환경에서 데이터 분석 결과를 완벽하게 제어할 수 있는 복잡하고 정밀한 분석용 대화형 웹 애플리케이션(대시보드) 제작 프레임워크
>   - Flask(백엔드), Plotly.js(시각화), React.js(프론트엔드)를 기반으로 구축됨
>   - 대규모 엔터프라이즈 환경에서 세밀한 UI 스타일링과 정교한 상태 관리가 필요한 대시보드를 개발할 때 주로 활용됨
{: .common-quote}

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

### 3.1 기초적인 표현 사용하기

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

<br>

- **다양한 사용자 입력 위젯**
    - 사용자의 입력을 받아 변수에 할당하는 과정이 직관적

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

<br>

- **레이아웃 구성 (사이드바 및 컬럼)**
    - 화면 분할이나 사이드바 제어 등 레이아웃의 구성도 간단한 컨텍스트 매니저(`with`)문으로 가능

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

### 3.2 데이터 및 테이블 출력

- 판다스 데이터프레임(DataFrame)을 깔끔한 대화형 테이블로 시각화할 수 있음

    ```bash
    pip install pandas numpy
    ```

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


### 3.3 그래프 그리기

- **지원하는 그래프 종류**
    - 자체적으로 내장된 간단한 차트 기능
        - `st.line_chart`, `st.bar_chart`, `st.area_chart` 등
    - Python의 대표적인 시각화 라이브러리인 **Plotly, Matplotlib, Seaborn** 등으로 그린 정교한 그래프 완벽 지원


- **대화형 그래프 실습 예제 코드**
    - 대화형 시각화에 가장 널리 쓰이는 `Plotly`와 Streamlit 내장 차트를 골고루 활용

    ```bash
    pip install pandas numpy plotly
    ```

    ```python
    #//file: "graph.py"

    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # 1. 페이지 레이아웃 설정
    st.set_page_config(page_title="Streamlit 그래프 실습", layout="wide")

    st.title("Streamlit 핵심 그래프 시각화 실습")
    st.markdown("가장 빈번하게 사용되는 5가지 기본 그래프를 직접 확인하고 코드를 실습해보세요.")

    # 2. 실습용 샘플 데이터 생성 (날짜별 제품 판매 및 고객 데이터)
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=30)
    chart_data = pd.DataFrame({
        '날짜': dates,
        'A제품_판매량': np.random.randint(50, 100, size=30),
        'B제품_판매량': np.random.randint(40, 90, size=30),
        '고객_만족도': np.random.uniform(3.0, 5.0, size=30),
        '방문_고객수': np.random.randint(100, 500, size=30),
        '연령대': np.random.choice(['20대', '30대', '40대', '50대'], size=30)
    })

    # 3. Streamlit의 탭(Tabs) 기능을 이용해 그래프 분할 배치
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "선 그래프 (Line)", 
        "막대 그래프 (Bar)", 
        "영역 그래프 (Area)", 
        "산점도 (Scatter)", 
        "히스토그램 (Histogram)"
    ])

    # ------------------------------------------------------------------
    # Tab 1: 선 그래프 (시간 추이, 트렌드 분석용)
    # ------------------------------------------------------------------
    with tab1:
        st.header("1. 선 그래프 (Line Chart)")
        st.caption("시계열 데이터나 연속적인 값의 변화 트렌드를 파악할 때 가장 유용합니다.")
        
        # Streamlit 내장 라인 차트 사용
        # 날짜를 인덱스로 지정하여 다중 컬럼 선 그래프 그리기
        line_df = chart_data.set_index('날짜')[['A제품_판매량', 'B제품_판매량']]
        st.line_chart(line_df)
        
        st.info("💡 **Tip:** Streamlit 내장 `st.line_chart`는 마우스 오버 시 툴팁을 기본 지원하며, 드래그로 확대가 가능합니다.")

    # ------------------------------------------------------------------
    # Tab 2: 막대 그래프 (항목 간 크기 비교용)
    # ------------------------------------------------------------------
    with tab2:
        st.header("2. 막대 그래프 (Bar Chart)")
        st.caption("서로 다른 범주(Category) 간의 수치나 양을 직관적으로 비교할 때 사용합니다.")
        
        # Streamlit 내장 바 차트 사용
        bar_df = chart_data.set_index('날짜')[['A제품_판매량']]
        st.bar_chart(bar_df)

    # ------------------------------------------------------------------
    # Tab 3: 영역 그래프 (누적 합계 및 전체 비중 확인용)
    # ------------------------------------------------------------------
    with tab3:
        st.header("3. 영역 그래프 (Area Chart)")
        st.caption("선 그래프 아래의 면적을 채워, 시간 경과에 따른 수치 변화뿐만 아니라 전체적인 '부피/총량'을 시각화합니다.")
        
        # Streamlit 내장 영역 차트 사용
        area_df = chart_data.set_index('날짜')[['A제품_판매량', 'B제품_판매량']]
        st.area_chart(area_df)

    # ------------------------------------------------------------------
    # Tab 4: 산점도 (두 변수 간의 상관관계 확인용)
    # ------------------------------------------------------------------
    with tab4:
        st.header("4. 산점도 (Scatter Plot)")
        st.caption("두 개의 연속형 변수 간의 분포와 상관관계(예: 방문 고객수와 판매량의 관계)를 파악할 때 강력합니다.")
        
        # Plotly Express를 활용한 대화형 산점도
        fig_scatter = px.scatter(
            chart_data, 
            x='방문_고객수', 
            y='A제품_판매량', 
            color='연령대',          # 색상으로 범주 구별
            size='고객_만족도',       # 만족도에 따라 점의 크기 조절
            hover_data=['날짜'],
            title="방문 고객수 대비 A제품 판매량 (점 크기 = 고객 만족도)"
        )
        # Plotly 객체를 Streamlit에 렌더링
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 5: 히스토그램 (데이터의 분포 및 빈도 분석용)
    # ------------------------------------------------------------------
    with tab5:
        st.header("5. 히스토그램 (Histogram)")
        st.caption("데이터가 특정 구간에 얼마나 밀집해 있는지 빈도(Frequency) 분포를 파악할 때 사용합니다.")
        
        # Plotly Express를 활용한 히스토그램
        fig_hist = px.histogram(
            chart_data, 
            x='방문_고객수', 
            nbins=10,                # 구간(Bin) 개수 설정
            color='연령대',          # 연령대별 누적 분포 확인
            title="방문 고객수 구간별 데이터 빈도수"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ------------------------------------------------------------------
    # 하단 원본 데이터 확인 영역
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("실사용 샘플 데이터셋 보기"):
        st.dataframe(chart_data)
    ```

    - **실습 팁 및 컴포넌트 안내**
        - **`st.tabs` 활용:**
            - 화면이 아래로 너무 길어지면 가독성이 떨어짐
            - 그래프 종류별로 탭을 나누어 깔끔한 UI 유지

        - **내장 차트 vs 외부 라이브러리:**
            - **내장 차트(`st.line_chart` 등):**
                - 가볍고 빠르게 트렌드를 밀어 넣을 때 좋음
                - 별도의 파라미터 튜닝 없이 대용량도 부드럽게 표현됨

            - **Plotly (`st.plotly_chart`):**
                - 다차원 분석(색상 변수 지정, 마우스 오버 커스텀 툴팁, 점 크기 제어 등)이 필요할 때 정교한 시각화를 제공

        - **`use_container_width=True`:**
            - Plotly 그래프를 넣을 때 이 옵션을 주면
            - 브라우저나 컬럼 화면 크기에 맞춰 그래프 너비가 유연하게 자동 반응형으로 조절됨



### 3.4 데코레이터와 캐싱

- **Python 데코레이터(Decorator)란?**
    - Python 언어가 제공하는 독특한 문법하는 포장지"입니다.
    - 기존 함수를 수정하지 않고도, 함수의 앞뒤에 새로운 기능을 추가(장식)하는 등 그 함수의 기능을 확장하거나 추가적인 행동을 덧붙일 수 있도록 도와주는 일종의 포장지(Wrapper)
    - 문법적으로는 **함수 정의 위에 `@데코레이터_이름`** 형태로 표기함
    - 주로 로깅, 권한 확인, 실행 시간 측정 등 여러 함수에서 **공통적으로 반복되는 로직(횡단 관심사)을 분리하여 재사용**할 때 사용됨

- **데코레이터의 사용 이유**
    - 만약 여러 함수가 실행될 때마다 걸린 시간을 측정해야 한다면,
        - 모든 함수 안에 `time.time()` 코드를 일일이 집어넣어야 함 🡪 코드가 중복되고 지저분해짐
        - 이때 데코레이터를 하나 만들어 함수 위에 `@` 기호로 얹어주면 🡪 공통 기능을 깔끔하게 주입할 수 있음

    - **동작 원리:**
        - 데코레이터는 대상 함수를 매개변수로 받아서
        - 내부의 '새로운 포장 함수'로 감싼 뒤,
        - 그 포장 함수를 리턴

    - **핵심 이점:**
        - 핵심 비즈니스 로직과 부가 기능(로그 출력, 권한 체크, 캐싱 등)을 완벽히 분리
        - **코드의 가독성과 재사용성** 극대화

- **캐싱(Caching)이란?**
    - 한 번 계산하거나 가져온 무거운 데이터 결과물을 메모리에 똑똑하게 기억(저장)해 두는 기술
    - 데코레이터라는 리모컨을 통해 Streamlit에서 구동되는 핵심 시스템

- **캐싱(Caching)의 메커니즘**
    - Streamlit은 위젯이 변경될 때마다 전체 스크립트를 재실행함
        - 화면에 작은 변화(위젯 조작 등)만 생겨도 스크립트 전체를 위에서 아래로 처음부터 끝까지 다시 실행(`Rerun`)하는 특성을 가짐
    - 만약 대용량 ERP 데이터를 로드하거나 복잡한 AI 모델을 추론하는 로직이 있다면
        - 캐싱 기능이 없다면 매번 재실행할 때마다 수 GB짜리 파일이나 AI 모델을 새로 읽어와야 하므로
        - 화면이 멈추어 버리는 심각한 병목이 발생
    - 이런 문제를 해결하기 위하여 캐싱 기능을 제공하고, 데코레이터를 이용해 캐싱 기능을 사용함

- **Streamlit에서의 데코레이터 사용 개념**
    - Streamlit 환경에서 데코레이터는 주로 '성능 최적화(캐싱)'를 위해 필수적으로 사용됨
    - Streamlit은
        - 사용자가 웹 페이지의 위젯(버튼, 슬라이더 등)을 조작할 때마다
        - 소스 코드 전체를 위에서 아래로 처음부터 다시 실행(Rerun)하는 구조를 가짐
        - 만약 코드 내에 수 GB짜리 AI 모델을 로드하거나 무거운 데이터베이스 조회를 수행하는 함수가 있다면,
        - 버튼 하나 누를 때마다 수십 초씩 화면이 멈추게 됨

    - Streamlit이 제공하는 데코레이터를 함수 위에 붙이면 다음과 같이 동작함
        - **최초 실행:**
            - 함수를 실제로 실행하고, 그 결과물(Return 값)을 메모리 공간에 캐싱(저장)
        - **이후 재실행 (Rerun):**
            - 함수의 입력 파라미터가 변경되지 않았다면,
            - 함수를 다시 실행하지 않고 **메모리에 저장된 결과물을 즉시 반환**하여 화면에 출력

- **Streamlit 캐싱의 분류**
    - Streamlit은 저장하려는 대상의 성격에 따라 캐시 메모리를 관리하는 방식을 두 가지 데코레이터로 분리함<br><br>

    - **데이터 캐싱 (`@st.cache_data`)**
        - **역할:**
            - 호출할 때마다 매번 완전히 동일한 값(정적 데이터)을 반환하는 함수에 사용
        - **주요 대상:**
            - 데이터프레임(`pd.DataFrame`)
            - CSV/Excel 로드 결과
            - REST API로 받아온 JSON 텍스트 등
        - **특징:**
            - 캐시 내부적으로 데이터가 복사되어 안전하게 보관되므로,
            - 함수 밖에서 데이터프레임을 일부 수정하더라도 원본 캐시가 오염되지 않음

    - **리소스 캐싱 (`@st.cache_resource`)**
        - **역할:**
            - 한 번 연결하거나 로드하면 앱이 켜져 있는 동안 내내 형태와 상태를 유지하며 재사용해야 하는 객체에 사용
        - **주요 대상:**
            - 머신러닝/딥러닝 모델 객체(`TensorFlow`, `PyTorch`, `Transformers`)
            - 데이터베이스 연결 객체(`DB Connection`)
            - 네트워크 소켓 등
        - **특징:**
            - 데이터를 복사하는 것이 아니라 '객체의 참조 주소(메모리 위치)'를 그대로 공유함
            - 무겁고 복잡한 글로벌 자원을 효율적으로 돌려 쓸 수 있음

- **캐싱 데코레이터의 내부 동작 4단계**
    - Streamlit에서 함수 위에 `@st.cache_data` 같은 캐싱 데코레이터를 붙이면, 시스템은 다음과 같은 프로세스로 작동함

    ```
    [함수 호출 발견] 
        🡫
    1. 함수 이름, 입력 파라미터(값), 함수 코드를 해시(Hash)로 변환하여 '고유 키(Key)' 생성
        🡫
    2. 이 고유 키가 메모리(캐시 저장소)에 이미 존재하는지 검사
        ├─🡪 [YES (존재함)] 🡪 3. 함수를 실행하지 않고, 저장된 결과물(Value)을 즉시 가로채서 반환 (0초 소요)
        └─🡪 [NO (처음 봄)] 🡪 4. 함수를 실제로 수행한 뒤, 그 결과물을 메모리에 저장하고 반환 (지연 소요)
    ```

- **실습 예제**

    ```bash
    pip install pandas numpy
    ```

    ```python
    #//file: "caching.py"

    import streamlit as st
    import pandas as pd
    import numpy as np
    import time

    # ------------------------------------------------------------------
    # [개념 1] @st.cache_data : 파일이나 데이터프레임 로드용
    # ------------------------------------------------------------------
    @st.cache_data
    def load_heavy_csv():
        time.sleep(4) # 대용량 파일 읽기 시뮬레이션 (4초 소요)
        
        # 2024~2026년도에 걸친 1,000,000행의 대용량 가상 데이터 생성
        np.random.seed(42)
        df = pd.DataFrame({
            "연도": np.random.choice([2024, 2025, 2026], size=1000000),
            "매출액(원)": np.random.randint(10000, 500000, size=1000000),
            "공장ID": np.random.choice(["A", "B", "C", "D"], size=1000000)
        })
        return df

    # ------------------------------------------------------------------
    # [개념 2] @st.cache_resource : AI 모델이나 글로벌 객체 로드용
    # ------------------------------------------------------------------
    class DummyAIModel:
        def predict(self, text):
            return f"익명 가동 [{text}] 처리 완료"

    @st.cache_resource
    def load_ai_model():
        # 모델 로딩 스왑 시간을 시뮬레이션하기 위해 3초 대기
        time.sleep(3)
        
        # AI 모델 객체 생성 후 반환
        return DummyAIModel()


    # ==================================================================
    # 메인 앱 화면 구성 및 함수 호출
    # ==================================================================
    st.title("Streamlit 데코레이터 실습")
    st.write("아래 버튼을 눌러 스크립트를 재실행해 보세요. 캐싱된 함수는 다시 실행되지 않습니다.")

    # --------------------------------------
    # 1. 데이터 캐싱 함수 호출
    # --------------------------------------
    st.subheader("1. 데이터 캐싱 (@st.cache_data)")
    with st.spinner("100만 건의 마스터 데이터를 메모리에 로드 중... (최초 1회만 4초 소요)"):
        # 이 함수는 연도와 관계없이 무조건 처음 한 번만 실행되어 메모리에 고정됩니다.
        total_df = load_heavy_csv()

    # 사용자의 선택 위젯 (캐싱 함수 밖에 위치)
    selected_year = st.selectbox("조회할 연도를 선택하세요:", [2024, 2025, 2026])

    # 필터링은 캐싱된 데이터를 대상으로 메모리 상에서 수행되므로 '0초' 만에 끝납니다.
    start_time = time.time()
    filtered_df = total_df[total_df["연도"] == selected_year]
    st.write(f"{selected_year}년도 데이터 수량: {len(filtered_df):,} 건")
    st.write(f"필터링 소요 시간: {time.time() - start_time:.4f}초")
    st.dataframe(filtered_df.head(10))

    # --------------------------------------
    # 2. 리소스 캐싱 함수 호출
    # --------------------------------------
    st.subheader("2. 리소스 캐싱 (@st.cache_resource)")
    with st.spinner("AI 핵심 모델 객체 로딩 중... (최초 1회만 3초 소요)"):
        model_object = load_ai_model()

    # 앱에서 모델 객체 활용
    user_input = st.text_input("분석할 문장을 입력하세요:", "공장 설비 A동 정상")
    if st.button("AI 분석 실행"):
        # 버튼을 누르면 전체 페이지가 재실행되지만, 
        # 위에서 정의한 두 함수는 스킵되므로 지연 시간 없이 즉시 결과가 출력됩니다.
        result = model_object.predict(user_input)
        st.success(f"모델 결과: {result}")
    ```

> - **요약**
>   - **데코레이터**는 함수를 포장하여 기능을 확장하는 **Python의 문법적 수단**이며,
>   - **캐싱**은 재실행에 따른 병목을 막기 위해 연산 결과를 메모리에 저장해 두는 **컴퓨터 과학의 기능**
>   - Streamlit은 이 두 가지를 결합하여, 개발자가 복잡한 상태 관리 코드를 짤 필요 없이 단 한 줄(`@st.cache_data`)만으로 강력한 앱 최적화를 달성할 수 있도록 설계됨
{: .summary-quote}

### 3.5 세션 상태 관리 (`st.session_state`)

- **세션:**
    - 브라우저가 켜져 있는 동안 데이터를 메모리에 지워지지 않게 고정해두는 **'글로벌 저장소'**

- **왜 필요할까?:**
    - Streamlit은 화면이 바뀔 때마다 스크립트 전체를 다시 실행(`Rerun`)
    - 이 때문에 일반 Python 변수는 페이지가 재실행되면 값이 초기화(리셋)되어 버림
    - 다음과 같은 경우에 필수
        - 로그인 상태를 유지하고 싶을 때
        - 버튼을 누를 때마다 카운트 값을 1씩 증가시키고 싶을 때
        - 여러 페이지 간에 데이터를 공유하고 싶을 때

- **사용 예시**

    ```python
    # 일반 변수는 버튼 누르면 매번 0으로 리셋되지만, session_state는 유지됨
    if 'count' not in st.session_state:
        st.session_state.count = 0

    if st.button("카운트 증가"):
        st.session_state.count += 1

    st.write(f"현재 카운트: {st.session_state.count}")
    ```


### 3.6 멀티 페이지 구성 (Multi-page Apps)

- 처음에는 `app.py` 한 파일에 코드를 다 짜지만, 기능이 많아지면 스크립트가 수천 줄 이상으로 늘어나 유지보수가 점점 불가능해짐
- 대시보드에 '메인 화면', '상세 분석', '설정 페이지'처럼 메뉴를 나누는 기능이 필요함<br><br>

- **구현 방식:**
    - 프로젝트 폴더 안에 `pages/`라는 이름의 하위 폴더 생성
    - `pages/` 폴더 안에 `Dashboard.py`, `Setting.py` 같은 파일 넣기
    - Streamlit이 알아서 사이드바에 유려한 다중 페이지 메뉴를 렌더링해 줌


### 3.7 사용자 경험(UX)을 위한 미디어 및 로딩 처리

- 기본 위젯 외에 사용자가 앱을 쓸 때 "와, 잘 만들었다"고 느끼게 하는 마감 요소들<br><br>

- **미디어 출력:**
    - 이미지(`st.image`), 오디오(`st.audio`), 비디오(`st.video`) 처리 및 AI 모델의 결과물인 오디오/영상 파일 시각화

- **파일 업로드/다운로드:**
    - `st.file_uploader`
        - 사용자가 자신의 CSV 파일이나 이미지를 웹 앱에 올려서 AI 모델로 분석하게 함

    - `st.download_button`
        - 분석 결과를 엑셀로 내려받게 해줌

- **상태 메시지 애니메이션:**
    - 연산이 진행 중임을 보여주는 공용 스피너(`st.spinner`)와 성공/경고 알림 창(`st.success`, `st.error`)


> - **요약**
>   1. **기초 UI & 위젯:** 텍스트, 입력 위젯, 레이아웃(사이드바, 컬럼)
>   2. **데이터 & 시각화:** Pandas 데이터 출력, 내장 차트 및 Plotly/Matplotlib 연동
>   3. **성능 최적화:** 데코레이터를 활용한 데이터/리소스 캐싱
>   4. **상태 및 구조 (추가 필요):** `st.session_state`(세션 제어) 및 `pages/`(멀티 페이지 구조)
>   5. **인터랙션 완성 (추가 필요):** 파일 업로드/다운로드 및 미디어 처리
{: .summary-quote}

# 4. 종합 예제

- Streamlit의 구동 패러다임(위에서 아래로 스크립트가 매번 통째로 재실행되는 방식)을 이해하고,<br>이를 통제하기 위한 핵심 기능들로 구성된 **실무형 템플릿**
- Streamlit의 "위젯 조작 🡪 페이지 전체 Rerun(재실행)"이라는 특성을 완벽히 통제하는 예제
- 자주 바뀌는 값(위젯 입력, 세션 상태, 필터링 결과)은 일반 스크립트 흐름에 맡겨 유연하게 대처하고,<br>바뀌지 않거나 무거운 자원(마스터 데이터, AI 모델)은 캐싱 데코레이터(`@`)로 묶어 메모리에 고정함으로써<br>**성능과 유연성을 모두 잡은 기초 실습 예제**

```bash
pip install pandas numpy matplotlib plotly
```

```python
#//file: "app.py"

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------------------------------------------------------
# [사전 작업] Matplotlib 한글 폰트 문제 해결
# ------------------------------------------------------------------
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf') # 한글 폰트 추가
plt.rcParams['font.family'] = "NanumBarunGothic" # 폰트 지정
plt.rc("axes", unicode_minus = False) # 마이너스 부호 충돌 문제 해결

# ------------------------------------------------------------------
# [개념 1] 페이지 기본 설정 & 리소스 캐싱 (@st.cache_resource)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="통합 스마트팩토리 관제 시스템",
    page_icon="🏭",
    layout="wide"
)

@st.cache_resource
def load_heavy_ai_model():
    """무거운 리소스를 시스템 메모리에 딱 한 번만 로드하는 리소스 캐싱"""
    time.sleep(1.0)  # 가상의 모델 로딩 시간
    # 이상 감지 스코어를 계산하는 가상 알고리즘 함수 객체 반환
    return lambda df, thresh: df[df < thresh].dropna(how='all')

detect_anomaly_model = load_heavy_ai_model()


# ------------------------------------------------------------------
# [개념 2] 데이터 캐싱 (@st.cache_data)
# ------------------------------------------------------------------
@st.cache_data
def get_factory_data():
    """가상의 공장 설비 24시간 로그 데이터 마스터 세트 생성"""
    np.random.seed(42)
    chart_data = pd.DataFrame(
        np.random.randint(70, 100, size=(24, 3)),
        columns=['1호기(프레스)', '2호기(조립)', '3호기(포장)']
    )
    return chart_data

df_master = get_factory_data()


# ------------------------------------------------------------------
# [개념 3] 세션 상태 관리 (st.session_state)
# ------------------------------------------------------------------
# Rerun(재실행)되어도 지워지지 않는 글로벌 진단 이력 저장소 초기화
if "history_logs" not in st.session_state:
    st.session_state.history_logs = []


# ------------------------------------------------------------------
# [개념 4] 레이아웃 구성: 사이드바 (Sidebar) & 인터랙션 (파일 업로드)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 관제 및 데이터 제어")
    
    # 기초 위젯: 셀렉트박스 및 라디오
    factory_line = st.selectbox("모니터링 라인 선택", ["제 1 공장 (시흥)", "제 2 공장 (창원)"])
    refresh_rate = st.radio("데이터 관제 주기", ["실시간 (1초)", "5분", "10분"], index=1)
    
    st.divider()
    
    # 인터랙션: 파일 업로드 기능
    st.subheader("📥 외부 공장 데이터 병합")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하여 데이터를 교체하세요.", type=["csv"])
    if uploaded_file is not None:
        try:
            df_master = pd.read_csv(uploaded_file)
            st.success("외부 데이터 동기화 완료!")
        except Exception as e:
            st.error(f"파일 로드 실패: {e}")
            
    st.divider()
    show_raw_data = st.checkbox("하단에 원본 로그 데이터 표시", value=False)
    
    st.divider()
    st.info("💡 **[멀티 페이지 구조 안내]**\n실무 환경에서는 프로젝트 폴더 내 `pages/` 폴더를 생성하고 `1_Dashboard.py`, `2_Settings.py` 파일을 넣어 다중 메뉴를 빌드할 수 있습니다.")


# ------------------------------------------------------------------
# [개념 5] 기초 UI: 텍스트 및 상태 안내 요소
# ------------------------------------------------------------------
st.title("🏭 실시간 스마트팩토리 통합 관제 대시보드")
st.markdown(f"**{factory_line}**의 주요 생산 설비 가동률과 데이터 흐름을 추적합니다.")

with st.spinner("마스터 데이터 동기화 중..."):
    time.sleep(0.3)
st.success("시스템 정상 가동 중")


# ------------------------------------------------------------------
# [개념 6] 레이아웃 구성: 컬럼 분할 (Columns) & KPI 출력
# ------------------------------------------------------------------
st.subheader("📊 핵심 가동 지표 (KPI)")
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.metric(label="전체 평균 가동률", value="91.4 %", delta="1.2 %")
with col_kpi2:
    st.metric(label="실시간 공정 불량률", value="0.42 %", delta="-0.08 %", delta_color="inverse")
with col_kpi3:
    st.metric(label="공장 내부 대기 온도", value="23.5 °C", delta="0.5 °C")

st.divider()


# ------------------------------------------------------------------
# [개념 7] 데이터 & 시각화 (내장 차트 및 Plotly/Matplotlib 다각화)
# ------------------------------------------------------------------
st.subheader("📈 설비별 흐름 분석 및 멀티 시각화")

# 기초 위젯: 슬라이더 범위 필터링
time_range = st.slider("분석 시간대 범위 설정 (24시간 제어)", 0, len(df_master)-1, (0, len(df_master)-1))
start_h, end_h = time_range
filtered_data = df_master.iloc[start_h:end_h+1]

# 탭을 활용하여 3가지 시각화 도구를 비교 구현
tab_native, tab_plotly, tab_matplotlib = st.tabs(["🔹 Streamlit 내장 차트", "🔸 Plotly 대화형 차트", "🎨 Matplotlib 정적 차트"])

with tab_native:
    st.caption("가볍고 빠르게 실시간 트렌드를 표현할 때 유용합니다.")
    st.line_chart(filtered_data)

with tab_plotly:
    st.caption("마우스 오버 및 범례 필터링이 가능한 엔터프라이즈급 대화형 그래프입니다.")
    # Plotly 데이터 재구조화 후 선 그래프 생성
    fig_plotly = px.line(filtered_data.reset_index(), x='index', y=filtered_data.columns, 
                         labels={'index': '시간 (Hour)', 'value': '가동률 (%)'}, title="시계열 설비 가동 추이")
    st.plotly_chart(fig_plotly, use_container_width=True)

with tab_matplotlib:
    st.caption("커스텀 레이아웃 및 논문/보고서용 정밀 출력이 장점입니다.")
    # Matplotlib을 이용한 피겨 생성
    fig_plt, ax = plt.subplots(figsize=(10, 3.5))
    for col in filtered_data.columns:
        ax.plot(filtered_data.index, filtered_data[col], marker='o', label=col)
    ax.set_title("설비 가동률 상세 추이 (Matplotlib)")
    ax.set_xlabel("시간 (Hour)")
    ax.set_ylabel("가동률 (%)")
    ax.legend()
    st.pyplot(fig_plt)

st.divider()


# ------------------------------------------------------------------
# [개념 8] 세션 상태 활용 이상 진단 & 파일 다운로드 인터랙션
# ------------------------------------------------------------------
st.subheader("🛠️ 캐싱 모델 기반 설비 이상 상태 모의 진단")

col_input, col_result = st.columns([1, 2])

with col_input:
    worker_name = st.text_input("점검자 성명 명기", "홍길동")
    threshold_val = st.number_input("이상 판정 가동률 임계치 (%)", min_value=50, max_value=100, value=75)
    run_analysis = st.button("🚨 AI 이상 감지 알고리즘 기동")

with col_result:
    if run_analysis:
        # 캐싱된 리소스 모델(@st.cache_resource)을 호출하여 즉시 결과 가로채기
        under_performing = detect_anomaly_model(filtered_data, threshold_val)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        if not under_performing.empty:
            status_msg = f"⚠️ [이상 발견] 임계치 {threshold_val}% 미달 설비 검출"
            st.error(status_msg)
            st.dataframe(under_performing)
        else:
            status_msg = f"✅ [정상] 모든 설비 안전 가동률 충족"
            st.success(status_msg)
            
        # [세션 상태 반영] 전체 페이지 재실행(Rerun)에 상관없이 데이터를 누적 배열에 추가
        st.session_state.history_logs.append({
            "점검일시": timestamp,
            "점검자": worker_name,
            "설정 임계치": threshold_val,
            "진단 결과": status_msg
        })

# 세션 상태에 이력이 존재할 경우 누적 기록 출력 및 다운로드 기능 구현
if st.session_state.history_logs:
    st.subheader("📋 누적 점검 이력 기록 데이터 (Session State 작동)")
    df_logs = pd.DataFrame(st.session_state.history_logs)
    st.dataframe(df_logs, use_container_width=True)
    
    # 인터랙션: 파일 다운로드 연동
    csv_bytes = df_logs.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 누적 점검 이력 데이터 다운로드 (CSV)",
        data=csv_bytes,
        file_name="factory_diagnostic_history.csv",
        mime="text/csv"
    )

st.divider()


# ------------------------------------------------------------------
# [개념 9] 멀티미디어(Media) 처리 및 조건부 원본 데이터 출력
# ------------------------------------------------------------------
st.subheader("📹 공장 내부 실시간 CCTV 및 오디오 브리핑 안내")
col_vid, col_aud = st.columns(2)

with col_vid:
    st.caption("CCTV 스트리밍 연동 가이드 미디어(가상)")
    # 가상 데모 비디오 소스
    st.video("https://www.youtube.com/watch?v=nit5PAbHZfM")

with col_aud:
    st.caption("일일 시스템 공정 현황 음성 브리핑 요약 가이드(가상)")
    # 가상의 안내 음성을 위한 비어있는 샘플 오디오 컴포넌트 재생
    st.audio("./media/Conversation_Sample_Kr.mp3")

if show_raw_data:
    st.divider()
    st.subheader("📋 분석용 원본 로그 데이터프레임 (Raw Data)")
    st.dataframe(df_master, use_container_width=True)
```

- **코드 설명**

    1. **초기 설정 및 이원화된 캐싱 (Caching)**

        ```python
        st.set_page_config(
            page_title="통합 스마트팩토리 관제 시스템",
            page_icon="🏭",
            layout="wide"
        )

        @st.cache_resource
        def load_heavy_ai_model():
            time.sleep(1.0)  
            return lambda df, thresh: df[df < thresh].dropna(how='all')

        detect_anomaly_model = load_heavy_ai_model()

        @st.cache_data
        def get_factory_data():
            np.random.seed(42)
            chart_data = pd.DataFrame(
                np.random.randint(70, 100, size=(24, 3)),
                columns=['1호기(프레스)', '2호기(조립)', '3호기(포장)']
            )
            return chart_data

        df_master = get_factory_data()
        ```

        - **`st.set_page_config`**:
            - 웹 앱 브라우저 탭에 표시될 타이틀, 아이콘, 그리고 레이아웃 너비(`wide`: 화면을 넓게 씀)를 지정하는 환경 설정 함수
            - 스크립트 최상단에 위치해야 함

        - **`@st.cache_resource`**:
            - 모델, DB 커넥션처럼 **한 번 생성되면 상태를 유지하며 앱 전체에서 공유해야 하는 자원**을 보존함
            - 예제에서는 이상 감지 함수(익명 함수 객체)를 메모리에 상주시켜 재실행 시 1초의 대기 시간(부하 시뮬레이션)을 스킵

        - **`@st.cache_data`**:
            - Pandas 데이터프레임과 같이 **순수한 데이터 결과물**을 메모리에 보관함
            - 사용자가 다른 위젯을 조작하더라도 무작위 데이터가 매번 새로 생성되지 않고 기존에 고정된 마스터 데이터를 즉시 반환


    2. **상태 보존 (Session State)**

        ```python
        if "history_logs" not in st.session_state:
            st.session_state.history_logs = []
        ```

        - **`st.session_state`**:
            - Streamlit에서 가장 중요한 개념
            - 웹 페이지에서 버튼을 클릭하거나 슬라이더를 움직이면 코드 전체가 처음부터 재실행(`Rerun`)되면서 일반 Python 변수(`logs = []` 등)는 모두 초기화됨
            - 이 재실행 메커니즘 속에서 **데이터를 유실하지 않고 브라우저 세션 동안 계속 유지·누적**하기 위해
            - `st.session_state`라는 전역 딕셔너리 공간을 활용함
            - 여기서는 진단 이력을 적재할 리스트를 최초 1회만 안전하게 초기화


    3. **사이드바 레이아웃 및 파일 업로드 (I/O)**

        ```python
        with st.sidebar:
            st.header("⚙️ 관제 및 데이터 제어")
            factory_line = st.selectbox("모니터링 라인 선택", ["제 1 공장 (시흥)", "제 2 공장 (창원)"])
            refresh_rate = st.radio("데이터 관제 주기", ["실시간 (1초)", "5분", "10분"], index=1)
            
            st.divider()
            uploaded_file = st.file_uploader("CSV 파일을 업로드하여 데이터를 교체하세요.", type=["csv"])
            if uploaded_file is not None:
                try:
                    df_master = pd.read_csv(uploaded_file)
                    st.success("외부 데이터 동기화 완료!")
                except Exception as e:
                    st.error(f"파일 로드 실패: {e}")
        ```

        - **`with st.sidebar`**:
            - 메인 화면 왼쪽에 고정되는 제어판 영역 구성

        - **기초 위젯 활용**:
            - `st.selectbox`와 `st.radio`를 통해 사용자의 선택 값을 `factory_line`, `refresh_rate` 변수에 즉시 담아 메인 로직으로 전달
        
        - **`st.file_uploader`**:
            - 웹 브라우저로 로컬 파일을 드래그 앤 드롭할 수 있는 인터랙션 컴포넌트
            - 사용자가 CSV를 업로드하는 순간(`is not None`), 판다스가 이를 읽어
                - 상단에서 캐싱했던 가상 데이터 `df_master`를 **새로운 실제 데이터셋으로 덮어쓰기(Overwrite)** 처리하여
                - 대시보드 전체를 동적으로 교체함


    4. **메인 KPI 및 복합 시각화 (Tabs)**

        ```python
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric(label="전체 평균 가동률", value="91.4 %", delta="1.2 %")
        # ... (생략) ...

        tab_native, tab_plotly, tab_matplotlib = st.tabs(["🔹 Streamlit 내장 차트", "🔸 Plotly 대화형 차트", "🎨 Matplotlib 정적 차트"])
        ```

        - **`st.columns(3)`**:
            - 화면의 가로 너비를 3분할하는 레이아웃 기법
            - 각 컬럼 공간 안에 대시보드 전용 UI인 `st.metric`을 배치
            - 주요 지표의 현재 수치와 전일 대비 증감율(`delta`)을 깔끔하게 표현

        - **`st.tabs`**:
            - 한정된 화면 공간을 효율적으로 쓰기 위해 탭(Tab) 인터페이스를 제공

        - **`st.line_chart` (내장)**: 
            - 가볍고 빠른 반응 속도

        - **`st.plotly_chart` (외부)**:
            - 마우스 조작(확대/축소, 툴팁 확인)이 가능한 엔터프라이즈급 시각화

        - **`st.pyplot` (외부)**:
            - 데이터 분석가들에게 친숙한 전통적 Matplotlib 피겨(`fig`) 출력


    5. **실시간 데이터 필터링 및 비즈니스 로직 연동**

        ```python
        time_range = st.slider("분석 시간대 범위 설정 (24시간 제어)", 0, len(df_master)-1, (0, len(df_master)-1))
        start_h, end_h = time_range
        filtered_data = df_master.iloc[start_h:end_h+1]
        ```

        - 사용자가 시간 조절 슬라이더(`st.slider`)를 움직이면 `time_range` 튜플 값(예: `(4, 18)`)이 실시간으로 반환됨
        - 이 값을 판다스의 행 슬라이싱 메서드인 `.iloc[start_h:end_h+1]`에 곧바로 대입
        - 사용자의 상호작용이 **데이터 필터링 코드로 다이렉트 연동**되는 Streamlit의 직관적인 데이터 흐름을 잘 보여주는 부분


    6. **데이터 저장을 위한 다운로드 인터랙션**

        ```python
        if run_analysis:
            under_performing = detect_anomaly_model(filtered_data, threshold_val)
            # ... (상태창 출력 생략) ...
            st.session_state.history_logs.append({ ... })

        if st.session_state.history_logs:
            df_logs = pd.DataFrame(st.session_state.history_logs)
            st.dataframe(df_logs, use_container_width=True)
            
            csv_bytes = df_logs.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="다운로드", data=csv_bytes, ...)
        ```

        - 사용자가 `AI 이상 감지 알고리즘 기동` 버튼을 누르면 캐싱된 모델을 통해 분석이 수행됨
        - 그 결과 정보 딕셔너리가 전역 공간인 `st.session_state.history_logs` 배열에 `.append()`로 추가됨
        - 리스트에 로그가 쌓이면 판다스 데이터프레임(`df_logs`)으로 변환하여 실시간 테이블 화면에 갱신

        - **`st.download_button`**:
            - 브라우저 메모리에 쌓인 `df_logs` 객체를
            - 텍스트 CSV 데이터 바이트(`utf-8-sig` 인코딩으로 한글 깨짐 방지)로 즉시 가공하여,
            - 사용자가 버튼을 누르면 로컬 PC 파일로 내려받을 수 있도록 내보내기 인터랙션을 완성함


    7. **멀티미디어 (Media) 처리**

        ```python
        st.video("https://www.youtube.com/watch?v=B2iAodbp0fQ")
        st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        ```

        - 텍스트, 데이터프레임 외에 공장 모니터링에 필요한 CCTV 스트리밍, 현장 브리핑 가이드를 모사하기 위해 `st.video` 및 `st.audio` 도입
        - 외부 URL 혹은 로컬 파일 경로를 집어넣는 것만으로 미디어 플레이어를 웹 화면에 깔끔하게 빌드해 줌

> - Streamlit은 Python 환경에서 LLM 모델의 결과물을 시각화하고 사용자 인터페이스를 구축하는 데 가장 효율적인 도구
>   - 꼭 Streamlit을 사용해야 하는 것은 아님 🡲 자신에게 맞는 것, 시간 내에 적용해 볼 수 있는 것으로 적당히 선택할 것
{: .summary-quote}

> - **기타 유사한 도구들**
>   - **Gradio (그라디오)**
>       - AI/머신러닝 모델의 데모 및 프로토타이핑에 가장 특화된 도구
>       - Hugging Face 생태계와 완벽히 통합되어 있어 AI 모델 웹 데모를 만들 때 가장 많이 사용됨
>       - Streamlit보다 코드가 더 직관적이며 입력(Input) 컴포넌트와 출력(Output) 컴포넌트를 매핑하는 구조가 매우 단순함
>       - 장점: 이미지/음성/텍스트 입출력(예: 챗봇 UI, 이미지 생성기 등)을 단 몇 줄 만으로 구현 가능
>       - 단점: 정밀한 데이터 대시보드나 복잡한 레이아웃을 유연하게 짜기에는 화면 제어력이 떨어짐
>   - **Shiny for Python (파이썬용 샤이니)**
>       - R 생태계에서 대시보드 끝판왕이었던 Shiny의 강력한 반응형(Reactive) 엔진을 Python에 이식한 도구
>       - Streamlit처럼 코드 전체를 무식하게 다시 실행하지 않고, 데이터의 의존성 그래프를 파악해 값이 바뀐 컴포넌트만 정밀하게 업데이트함
>       - 장점: 대규모 데이터나 복잡하게 얽힌 다중 위젯 환경에서도 성능 저하(병목) 없이 압도적으로 빠르고 안정적
>       - 단점: Streamlit에 비해 함수형 반응형 프로그래밍(@reactive.calc 등) 개념을 이해해야 하므로 초기 학습 곡선이 다소 높음
>   - 그 외에도 Reflex (리플렉스), marimo (마리모), Plotly Dash (대시) 등 다양한 도구가 있음
{: .common-quote}