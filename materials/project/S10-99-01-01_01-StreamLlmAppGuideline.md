---
layout: page
title:  "Streamit를 활용한 LLM 연동 어플리케이션"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S10-99-01-01_01-StreamLlmAppGuideline
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 프로세스 & 가이드라인

> - Streamlit은 Python 환경에서 LLM 모델의 결과물을 시각화하고 사용자 인터페이스를 구축하는 데 가장 효율적인 도구
>   - 그러나 꼭 Streamlit을 사용해야 하는 것은 아님 🡲 자신에게 맞는 것, 시간 내에 적용해 볼 수 있는 것으로 적당히 선택할 것
{: .common-quote}


### 1. 프로젝트 준비 및 환경 구성 (Setup)

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


### 2. 데이터 및 로직 설계 (Core Logic)

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


### 3. UI/UX 구현 (Frontend with Streamlit)
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


### 4. 불확실성 제어 및 최적화 (Troubleshooting)

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


### 5. 배포 및 포트폴리오 자산화 (Deployment)

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