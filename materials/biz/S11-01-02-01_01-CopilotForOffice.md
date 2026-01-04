---
layout: page
title:  "Copilot 개요"
date:   2025-12-14 10:00:00 +0900
permalink: /materials/S11-01-02-01_01-CopilotForOffice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## **1. Copilot 개요**

### 1.1 Copilot이란 무엇인가?

- **AI 기반의 지능형 비서**
    - Microsoft에서 개발한 AI 기반 생산성 도구 (MS는 이를 AI 동반자(Copilot)라고 지칭함)
        - 사용자가 업무를 더 효율적으로 처리하도록(업무와 일상에서 생산성을 높이고 창의성을 확장하도록) 돕는 인공지능 비서

- **생성형 AI 활용**
    - GPT와 같은 다양한 최신 대규모 언어 모델(LLM, Large Language Model)을 기반으로 생성형 AI 기술 활용
    - 사용자의 자연어 명령을 이해하고 문맥에 맞는 적절한 결과물을 생성하며
    - 단순한 답변을 넘어 사용자의 맥락을 이해하고, 맥락 기반의 제안과 분석 및 맞춤형 지원을 제공

- **지원 환경**
    - 멀티 플랫폼 지원: Windows, Mac, Web, iOS, Android, Edge 등
    - 다양한 앱 환경 지원: Windows, Microsoft 365, Edge, Teams 등
    - 업무의 내용에 따른 다양한 모드 지원: Smart Mode(GPT-5 기반), Study Mode, Think Deeper, Deep Research 등
    - 멀티모달 기능 지원: 텍스트, 이미지, 데이터, 음성 모두 처리 가능
    
- **확장된 역할**
    - 기존의 AI 챗봇이 단순히 질문에 답하거나 콘텐츠를 생성하는 데 그쳤다면, Copilot은 그 이상의 역할을 수행함
        - 통합성
            - Word, Excel, PowerPoint, Outlook, Teams 등 Microsoft 365 애플리케이션 통합됨
            - 해당 앱의 기능과 사용자의 데이터를 활용함
        - 지능형 비서
            - 사용자의 명령을 이해하고, 문맥을 파악하며, 관련 정보를 찾아 최적의 결과물을 제공함
            - 단순히 정보 제공을 넘어 아이디어를 제안하고, 초안을 작성하고, 데이터를 분석하는 등 실질적인 업무를 수행함
        - 생산성 증대
            - 반복적인 업무를 자동화하고, 정보 검색 시간을 단축하며, 창의적인 작업의 영감을 제공하여
            - 전반적인 업무 생산성을 혁신적으로 향상시킴
        - 자연어 인터페이스
            - 사용자가 자연어로 명령(프롬프트)을 내리면 Copilot이 이를 이해하고 작업을 처리함
            - 복잡한 코딩이나 명령어 학습이 필요 없음

    - 할 수 있는 일
<div class="info-table">
    <table>
        <thead>
            <th style="width: 200px;">구분</th>
            <th style="width: 600px;">내용</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">콘텐츠 생성</td>
                <td style="text-align: left;">문서 초안 작성, 이메일 답장 초안, 프레젠테이션 슬라이드 구성, 블로그 글 작성 등</td>
            </tr>
            <tr>
                <td class="td-rowheader">정보 요약</td>
                <td style="text-align: left;">긴 문서나 회의록, 보고서에서 핵심 내용 추출 및 요약</td>
            </tr>
            <tr>
                <td class="td-rowheader">데이터 분석</td>
                <td style="text-align: left;">엑셀 데이터 분석, 패턴 파악, 차트 및 그래프 생성</td>
            </tr>
            <tr>
                <td class="td-rowheader">협업 지원</td>
                <td style="text-align: left;">Teams 회의 요약, 주요 액션 아이템 도출, 일정 관리</td>
            </tr>
            <tr>
                <td class="td-rowheader">검색 및 탐색</td>
                <td style="text-align: left;">웹 검색, 최신 정보 확인, 문서·이메일·파일 탐색</td>
            </tr>
            <tr>
                <td class="td-rowheader">개인화 기능</td>
                <td style="text-align: left;">사용자의 선호와 맥락을 기억해 맞춤형 지원 제공</td>
            </tr>
        </tbody>
    </table>
</div>


### 1.2 Copilot의 핵심 연동 원리

- **프롬프트(Prompt) 기반**
    - 사용자가 자연어로 지시(프롬프트)를 입력 ➜ Copilot은 이를 의도(Intent)와 맥락(Context) 단위로 분석
    - 단순히 텍스트를 이해하는 수준을 넘어, 사용자의 업무 환경(앱, 문서, 대화 맥락)을 고려하여 적절한 작업을 결정

- **Microsoft Graph 연동(Contextual Data 활용)**
    - 단순히 텍스트를 LLM에 넘기는 것이 아니라, 먼저 사용자의 의도와 필요한 데이터 소스를 파악한 뒤 Graph와 연동함
    - 사용자의 Microsoft 365 데이터(이메일, 문서, 캘린더, 채팅 등)와 연결된 Microsoft Graph의 정보 활용

    >- Microsoft Graph
    >    - Microsoft 365, Azure AD 등 다양한 Microsoft 클라우드 서비스의 데이터를 하나의 [API 엔드포인트](https://graph.microsoft.com/){: target="blank"}로 통합 제공하는 플랫폼
    >    - 이메일, 일정, 파일, Teams 채팅, 사용자 프로필 등 조직 내 데이터를 안전하게 조회·활용 가능
    >        - 이를 통해 개발자는 조직 내 데이터를 손쉽게 연결하고 활용할 수 있음
    {: .common-quote}

    - 예시
        - 프롬프트 입력: "지난주 팀 회의록 요약해 줘"<br> ➜ Copilot은 Microsoft Graph를 통해 Teams 회의록 데이터를 가져오고<br> ➜ LLM이 이를 요약하여 사용자에게 제공

- **LLM (대규모 언어 모델) 활용**
    - 지능형 처리
        - Microsoft Graph에서 수집된 문맥 정보 + 사용자의 프롬프트
        ➜ OpenAI의 GPT-4와 같은 최신 '대규모 언어 모델(LLM)'로 전송
        ➜ LLM은 이 정보를 분석하여 사용자의 의도를 파악
        ➜ 최적의 응답이나 작업을 위한 계획을 수립
        - Graph + LLM 결합 구조
            - Graph는 데이터 공급자, LLM은 언어·맥락 처리자 역할
            - 두 시스템이 결합되어 Copilot의 지능형 응답이 완성됨        

    - 추론 및 이해
        - LLM은 광범위한 데이터를 학습하여 언어를 이해하고 생성하는 능력이 탁월함
        - 이를 통해 사용자의 요청이 단순히 키워드의 나열이 아닌 복합적인 의미를 가진 자연어로 처리됨
        - 단순 요약·생성뿐 아니라, 맥락 기반 추론을 통해 업무에 필요한 액션 아이템, 데이터 분석, 문서 초안 등을 제안



- **보안 및 개인 정보 보호**
    - 모든 과정은 Microsoft 365의 엔터프라이즈급 보안 및 규정 준수 체계를 따름
        - Copilot은 기업 환경에서 사용되므로, 데이터 보호와 규정 준수가 핵심 원리 중 하나임
    - 사용자 데이터는 AI 학습에 활용되지 않으며, 개인 정보 보호 원칙을 철저히 준수
    - 데이터 접근은 사용자의 권한과 조직 정책에 따라 제한되며, Copilot은 허용된 범위 내에서만 정보를 활용

<div class="insert-image" style="text-align: center;">
    <img style="width: 600px;" src="/materials/biz/images/S11-01-02-01_01-001.png">
</div>

## 2. Microsoft 365 기반 Copilot의 기능

- Microsoft 365 Copilot은 Word, Excel, PowerPoint, Outlook, Teams 등 다양한 애플리케이션에 녹아들어 있음

<div class="info-table">
    <table>
        <thead>
            <th style="width: 110px;">애플리케이션</th>
            <th style="width: 190px;">기능</th>
            <th style="width: 590px;">내용 및 프롬프트 예시</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader" rowspan="3">Word</td>
                <td>문서 초안 작성</td>
                <td style="text-align: left;">"이 프로젝트의 목표와 진행 상황에 대한 1페이지 보고서 초안을 작성해 줘."</td>
            </tr>
            <tr>
                <td>내용 요약 및 재작성</td>
                <td style="text-align: left;">"이 문서를 500자 이내로 요약하고, 비전문가도 이해하기 쉽게 다시 써 줘."</td>
            </tr>
            <tr>
                <td>아이디어 브레인스토밍</td>
                <td style="text-align: left;">"새로운 마케팅 전략에 대한 아이디어 5가지 제안해 줘."</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">Excel</td>
                <td>데이터 분석 및 인사이트 도출</td>
                <td style="text-align: left;">"이 데이터에서 가장 매출이 높은 상위 5개 제품을 찾아주고, 매출 증감 추이를 그래프로 보여줘."</td>
            </tr>
            <tr>
                <td>수식 생성</td>
                <td style="text-align: left;">"이 열의 평균을 구하는 수식을 작성해 줘."</td>
            </tr>
            <tr>
                <td>데이터 시각화 제안</td>
                <td style="text-align: left;">"이 데이터를 가장 잘 나타낼 수 있는 차트 유형을 제안하고 만들어 줘."</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">PowerPoint</td>
                <td>프레젠테이션 초안 생성</td>
                <td style="text-align: left;">"월별 실적 보고서에 대한 5장짜리 프레젠테이션 초안을 만들어 줘."</td>
            </tr>
            <tr>
                <td>슬라이드 내용 보강</td>
                <td style="text-align: left;">"이 슬라이드에 있는 주요 내용을 간결한 불릿 포인트로 정리하고, 관련 이미지를 추천해 줘."</td>
            </tr>
            <tr>
                <td>텍스트 다듬기</td>
                <td style="text-align: left;">"이 슬라이드의 텍스트를 더 전문적인 용어로 다듬어 줘."</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">Outlook</td>
                <td>이메일 초안 작성</td>
                <td style="text-align: left;">"고객의 문의 사항에 대한 친절한 답변 이메일을 작성해 줘."</td>
            </tr>
            <tr>
                <td>긴 이메일 요약</td>
                <td style="text-align: left;">"이 긴 이메일 스레드에서 핵심 내용과 필요한 액션 아이템만 요약해 줘."</td>
            </tr>
            <tr>
                <td>캘린더 제안</td>
                <td style="text-align: left;">이메일 내용을 분석하여 미팅 일정을 제안함</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">Teams</td>
                <td>회의 요약</td>
                <td style="text-align: left;">회의 내용을 실시간으로 요약하고, 주요 의사결정 및 액션 아이템을 정리함</td>
            </tr>
            <tr>
                <td>빠른 정보 찾기</td>
                <td style="text-align: left;">채팅 기록에서 특정 정보를 빠르게 찾아줌</td>
            </tr>
        </tbody>
    </table>
</div>


