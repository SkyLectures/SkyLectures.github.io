---
layout: page
title:  "Copilot 환경 설정"
date:   2025-12-14 10:00:00 +0900
permalink: /materials/S11-01-02-02_01-CopilotEnvForOffice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Copilot 사용을 위한 기본 절차

> - Copilot은 웹 인터페이스에서 사용하는 Copilot과 Microsoft 365 앱 내에서 작동하는 Copilot으로 구분할 수 있음
{: .common-quote}

웹 기반 Copilot은 일반적인 AI 챗봇처럼 작동함

### 1.1 웹 기반 Copilot 사용

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


### 1.2 Microsoft 365 앱 내의 Copilot 사용

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

## 2. 유료 Copilot 사용

- Word, Excel, PowerPoint, Outlook, Teams 등 Microsoft 365 애플리케이션 내에 직접 통합되어 작동하는 Copilot은 현재 Microsoft 365 Copilot 라이선스가 있는 유료 구독자만 사용할 수 있음
- Copilot은 Microsoft 365(구 Office 365) 구독 환경에서만 지원됨
    - 영구 설치형(예: Office 2019, Office 2021)에서는 Copilot을 제대로 사용할 수 없음

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


## 3. Copilot 유료 구독 비용

<div class="insert-image" style="text-align: center;">
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-013.png"><br><br>
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-014.png"><br><br>
    <img style="width: 980px; border: 1px solid gray;" src="/materials/biz/images/S11-01-01-02_01-015.png">
</div>

