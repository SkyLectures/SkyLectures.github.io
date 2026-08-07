---
layout: page
title:  "GPT와 Copilot 실습환경"
date:   2025-12-14 10:00:00 +0900
permalink: /materials/S11-01-01-02_01-GptCopilotEnv
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. GPT 사용을 위한 기본 절차

> - 대부분의 GPT 기반 서비스(예: ChatGPT)는 다음과 같은 과정을 거쳐 사용함
{: .common-quote}

### 1.1 서비스 접속 및 로그인

- **웹 기반 서비스 선택**
    - OpenAI의 ChatGPT(chat.openai.com) 또는 Microsoft Copilot(copilot.microsoft.com)과 같은 웹 기반 AI 챗봇 서비스에 접속
        - Microsoft Copilot은 다른 LLM도 사용하지만 그 기반은 GPT임
    - URL: [ChatGPT](https://chatgpt.com/){: target="_blank"}
        
        <div class="insert-image" style="text-align: center;">
            <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-001.png">
        </div>

- **계정 생성 또는 로그인**
    - 서비스를 이용하려면 일반적으로 계정(구글, 마이크로소프트 계정 등으로 연동 가능)을 생성하거나 기존 계정으로 로그인함
    - 로그인하지 않아도 사용할 수 있으나 사용 이력이 저장되지 않음


### 1.2 프롬프트(Prompt) 입력

- **질문 또는 지시**
    - 채팅창에 텍스트로 GPT에게 원하는 것을 요청
    - 이를 '프롬프트(Prompt)'라고 부름
        - 예시: "자율주행 기술의 최신 동향 5가지에 대해 설명해 줘."
        - 예시: "다음 문단을 더 간결하고 전문적인 문체로 다듬어 줘: [문단 내용 붙여넣기]"
        
        <div class="insert-image" style="text-align: center;">
            <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-002.png">
        </div>


### 1.3 GPT의 응답 확인

- **텍스트 생성**
    - GPT는 입력된 프롬프트를 이해하고, 학습된 지식과 패턴을 바탕으로 응답 텍스트를 생성하여 채팅창에 보여줌
    - 이 과정은 보통 몇 초 이내로 빠르게 진행됨


### 1.4 추가 질문 및 개선 요청 (선택 사항)

- **대화 이어가기**
    - GPT는 이전 대화의 맥락을 기억(세션 내에서)하고 있으므로, 이전 응답을 바탕으로 추가적인 질문을 하거나 개선을 요청할 수 있음
        - 예시: "방금 설명한 5가지 동향 중, '센서 융합 기술'에 대해 더 자세히 알려줘."
        - 예시: "요약된 문단을 '강한 설득력'을 가진 문체로 다시 작성해 줘."

        <div class="insert-image" style="text-align: center;">
            <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-003.png">
        </div>

- **정확성 검증**
    - GPT의 응답은 항상 정확하지 않을 수 있으므로, 특히 중요한 정보(숫자, 사실, 출처 등)는 반드시 사용자가 직접 검증해야 함

<br>

> - 이 네 가지 단계를 통해 GPT와 상호작용하며 원하는 정보를 얻거나 텍스트를 생성하는 작업을 수행할 수 있음
> - 마치 사람과 대화하듯이 자연어로 소통하는 것이 핵심
{: .summary-quote}

## 2. GPT 유료 구독

>- 핵심은 **'더 강력한 AI 성능과 안정성을 통해 업무의 복잡성을 처리하고, 생산성을 극대화하며, 자동화의 범위를 확장한다'**는 것
{: .common-quote}

### 2.1 GPT 유료 구독의 필요성

- **고품질 & 복잡한 작업 처리 (GPT-4 접근)**
    - 무료 버전의 GPT-3.5보다 훨씬 **강력하고 지능적인 GPT-4 모델**에 접근할 수 있음
    - 이는 복잡한 보고서 초안 작성, 심층적인 데이터 분석 및 통찰력 도출, 다단계 의사결정 시뮬레이션, 정교한 비즈니스 문서 생성 등 **고품질과 정확성이 요구되는 업무**를 자동화하고 고도화하는 데 필수적

- **업무 연속성 및 생산성 극대화**
    - 사용자가 몰릴 때 **접근 제한이나 느려짐 없이** 안정적이고 빠른 응답 속도를 보장
    - 이는 자동화된 워크플로우가 멈추지 않고 원활하게 작동하여 **업무 연속성과 생산성을 유지**하는 데 중요

- **확장된 데이터 처리 및 자동화 기능 (고급 데이터 분석, 플러그인/GPTs)**
    - **고급 데이터 분석(코드 인터프리터)** 기능을 통해 파일을 직접 업로드하여 데이터 분석, 시각화, 통계 처리 등을 수행
    - **수동 분석의 한계를 넘어선 자동화된 인사이트** 제공
    - **플러그인(Plug-ins) 또는 GPTs**를 활용하여 외부 서비스(웹 검색, 문서 관리, 스케줄링 도구 등)와 연동할 수 있음
    - 이는 Power Automate와 같은 도구가 없더라도 GPT 자체적으로 **더 넓은 범위의 자동화 워크플로우를 구축**하는 기반이 됨

- **긴 컨텍스트 처리 능력**
    - **더 긴 컨텍스트 윈도우**를 제공하여 방대한 문서를 한 번에 요약하거나, 복잡한 대화를 끊김 없이 이어가며 심층적인 작업 지시를 내리는 것이 가능
    - 이는 긴 보고서 작성이나 복잡한 프로젝트 관리와 같은 업무를 AI와 효과적으로 협업하는 데 결정적인 이점을 제공함

> - 결론적으로, GPT 유료 구독(예: ChatGPT Plus)은 **단순한 기능 확장 이상의 '업무 방식 혁신'**을 위한 투자이며
> - AI 기반 자동화 및 고도화를 통해 **전략적인 사고와 창의성에 집중할 수 있는 환경**을 구축하는 핵심적인 요소
> - GPT 유료 구독은 단순히 'AI를 쓴다'는 것을 넘어, **업무 프로세스의 지능적 자동화와 콘텐츠의 품질 고도화**를 위한 필수적인 투자임
> <br><br>
> - 그러나 업무에 대한 적용이 무료 서비스만으로 충분한 정도의 활용도에 그치는 경우, 무리해서 유료 구독을 할 필요는 없음
{: .summary-quote}

### 2.2 GPT 유료 구독 비용

<div class="insert-image" style="text-align: center;">
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-004.png"><br><br>
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-005.png">
</div>


## 3. Copilot 사용을 위한 기본 절차

> - Copilot은 웹 인터페이스에서 사용하는 Copilot과 Microsoft 365 앱 내에서 작동하는 Copilot으로 구분할 수 있음
{: .common-quote}

웹 기반 Copilot은 일반적인 AI 챗봇처럼 작동함

### 3.1 웹 기반 Copilot 사용

- 웹 기반 Copilot은 일반적인 AI 챗봇처럼 작동함

1. **접속**
    - 웹 브라우저(Microsoft Edge 권장)를 열고 [`copilot.microsoft.com`](https://copilot.microsoft.com){: target="_blank"}으로 접속
2. **로그인**
    - Microsoft 계정으로 로그인 (일부 기능 및 상업적 데이터 보호를 위해 필요함)

    <div class="insert-image" style="text-align: center;">
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-006.png">
    </div>

3. **프롬프트 입력**
    - 화면 하단의 채팅 입력창에 질문이나 요청 사항을 자연어로 입력
    - **파일 활용**
        - 필요시 입력창 위에 있는 **클립 아이콘**을 클릭하여
        - PDF, Word, 이미지 파일 등을 업로드하고 해당 파일의 내용을 기반으로 질문할 수 있음
    - **웹 검색**
        - Copilot은 기본적으로 웹 검색(Bing) 기능을 활용하여 답변을 생성함

4. **응답 확인**
    - Copilot이 입력된 프롬프트와 웹 검색 결과, 또는 업로드된 파일 내용을 바탕으로 응답 텍스트를 생성하여 보여줌

    <div class="insert-image" style="text-align: center;">
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-007.png">
    </div>

5. **대화 이어가기 및 검토**
    - GPT와 마찬가지로, 이전 대화의 맥락을 바탕으로 추가 질문을 하거나, 응답 내용을 검토하고 개선을 요청할 수 있음
    - 생성된 내용 중 중요한 정보는 반드시 직접 검증해야 함

    <div class="insert-image" style="text-align: center;">
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-008.png">
    </div>


### 3.2 Microsoft 365 앱 내의 Copilot 사용

- Microsoft 365 앱과의 직접적인 연동 없이, 범용적인 AI 챗봇 기능과 웹 검색, 파일 분석 기능 활용을 대상으로 설명

1. **앱 실행**
    - Word, Excel, PowerPoint, Outlook, Teams 등 Copilot이 통합된 Microsoft 365 애플리케이션 실행

2. **Copilot 호출**
    - **아이콘 클릭**: 앱 내 리본 메뉴 또는 작업 영역에 있는 Copilot 아이콘 클릭
    - **단축키**: `Alt + I` (Windows) 또는 `Option + I` (Mac)를 눌러 채팅창 열기

    <div class="insert-image" style="text-align: center;">
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-009.png">
    </div>

3. **프롬프트 입력**
    - Copilot 채팅창에 원하는 작업을 자연어로 요청
    - 이때 현재 문서나 열려 있는 파일의 컨텍스트를 활용하거나 `/` 기호를 사용하여 다른 파일을 참조하도록 지시할 수 있음

4. **응답 확인**
    - Copilot이 프롬프트를 이해하고
    - 해당 앱 내에서 텍스트 생성, 데이터 분석, 슬라이드 초안 작성 등 요청된 작업을 수행한 결과(텍스트, 차트, 슬라이드 등)를 제시

    <div class="insert-image" style="text-align: center;">
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-010.png"><br><br>
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-011.png"><br><br>
        <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-012.png">
    </div>

5. **검토 및 적용**
    - Copilot이 제시한 결과물을 검토하고
    - 필요한 경우 직접 수정하거나 추가 프롬프트로 보완 및 개선을 요청함

### 3.3 유료 Copilot 사용

- Word, Excel, PowerPoint, Outlook, Teams 등 Microsoft 365 애플리케이션 내에 직접 통합되어 작동하는 Copilot은 현재 Microsoft 365 Copilot 라이선스가 있는 유료 구독자만 사용할 수 있음

- **주요 특징 및 기능**
    - **작업 중인 앱의 컨텍스트를 실시간으로 인지**
        - Word 문서의 특정 문단, Excel 시트의 데이터 범위, Outlook 메일 내용 등을 Copilot이 직접 읽고 이해하여 작업을 수행
    - **Microsoft Graph 데이터 활용**
        - 사용자의 이메일, 캘린더, 미팅 기록, 채팅, OneDrive/SharePoint에 저장된 문서 등 Microsoft 365 전반의 '나의(조직의) 데이터'를 참고하여 개인화되고 관련성 높은 응답을 생성
        - 예: "오늘 미팅 요약해 줘", "이 보고서 내용을 참조해서 메일 써 줘."
    - **앱 내 직접 조작 및 삽입**
        - Word 문서 내에 텍스트를 바로 삽입하거나, PowerPoint 슬라이드를 만들고, Excel 데이터를 직접 분석하여 차트를 생성하는 등 앱 기능을 직접 조작
    - **워크플로우 통합**
        - Power Automate와 연동하여 더욱 복잡한 자동화 워크플로우를 시작하는 트리거 역할을 할 수 있음

- **무료 사용자(웹 기반 Copilot)의 활용 범위**
    - 웹 검색 기반 정보 탐색 및 요약
    - 붙여넣기 텍스트 또는 직접 업로드한 파일 (PDF, Word 문서 등)의 요약 및 정보 추출
    - 다양한 텍스트 콘텐츠(초안, 아이디어, 이메일 등) 생성 및 재작성
    - 질의응답, 브레인스토밍
    - 이미지 생성 (DALL-E 3 연동)
    <br><br>
    - <span style="color: darkred;">**워드 문서 내에서 자동으로 내용을 파악해서 특정 위치에 글을 써넣거나, 엑셀 데이터를 직접 조작해서 차트를 만들거나, 아웃룩에서 받은 메일에 답장 초안을 작성하는 등의 '앱 내 통합 작업'은 무료 Copilot에서는 불가능**</span>
    - 대신 프롬프트에 직접 데이터를 입력하거나, 파일을 업로드하여 처리하는 작업은 가능함
    - 어느 정도까지의 작업은 무료 사용으로도 충분히 가능하지만 Microsoft 365 앱과 직접 연동되어 조작, 생성하는 강력한 기능은 유료 구독을 필요로 함
    - 필요한 업무의 내용을 확인하여 구독 여부를 결정할 것


### 3.4 Copilot 유료 구독 비용

<div class="insert-image" style="text-align: center;">
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-013.png"><br><br>
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-014.png"><br><br>
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-015.png">
</div>
