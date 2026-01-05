---
layout: page
title:  "Copilot 개요"
date:   2025-12-14 10:00:00 +0900
permalink: /materials/S11-01-02-01_01-CopilotForOffice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. LLM 개요

- 대량의 텍스트 데이터를 학습하여 사람의 언어를 이해하고 생성하는 거대한 인공지능 모델
- 단순히 단어를 나열하는 것을 넘어, 문맥을 이해하고 자연스러운 대화, 요약, 번역, 창작 등 복잡한 언어 작업을 수행함

- **이론적 배경**
    - **트랜스포머(Transformer) 아키텍처**
        - 2017년 Google에서 발표한 기술로 LLM의 성능을 비약적으로 발전시킨 핵심 요소
        - AI가 문장 내의 각 단어 간의 관계와 중요도를 파악하여 더 정확한 문맥 이해를 가능하게 함

        <div class="insert-image" style="text-align: center;">
            <img style="width: 400px;" src="/materials/biz/images/S11-01-01-01_01-001.png"><br>
            <div>그림출처: <a href="https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf" target="_blank">Attention Is All You Need(Google, 2017)</a></div>
        </div>

    - **사전 학습(Pre-training)**
        - LLM은 인터넷상의 수많은 책, 문서, 웹사이트 등 방대한 텍스트 데이터를 미리 학습함
        - 이 과정을 통해 세상의 지식과 언어 패턴, 문법 등을 습득
            - 마치 인턴 시절, 수많은 보고서와 자료를 읽으면서 회사 업무의 흐름과 용어를 익히는 것과 같음

    - **생성(Generative)**
        - 단순히 정해진 답변을 주는 것이 아니라
        - 학습된 지식을 바탕으로 사용자의 질문에 대한 '새로운' 텍스트를 창의적으로 생성함
            - 익힌 지식을 바탕으로 새로운 아이디어를 내고 보고서를 작성하는 것과 같음


## 2. Copilot 개요

- **AI 기반의 지능형 비서**
    - Microsoft에서 개발한 AI 기반 생산성 도구 (MS는 이를 AI 동반자(Copilot)라고 지칭함)
        - 사용자가 업무를 더 효율적으로 처리하도록(업무와 일상에서 생산성을 높이고 창의성을 확장하도록) 돕는 인공지능 비서
        - Microsoft 365 애플리케이션(Word, Excel, PowerPoint, Outlook, Teams 등)에 **직접 통합되어 있음**
        - Copilot: 부조종사처럼, 사용자의 업무를 보조하고 생산성을 높이는 역할을 한다는 의미를 담은 명명
        
- **생성형 AI 활용**
    - GPT 등 최신 LLM의 강력한 능력을 기반으로 함
    - 사용자의 자연어 명령을 이해하고 문맥에 맞는 적절한 결과물을 생성하며,
    - 사용자가 작업하는 앱의 **실시간 컨텍스트와 조직의 데이터**를 활용하여,
    - 단순한 답변을 넘어 사용자의 맥락을 이해하고,
    - 맥락 기반의 제안과 분석 및 맞춤형 지원을 제공함

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

- **할 수 있는 일**
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

## 3. Copilot의 핵심 연동 원리

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
    <img style="width: 500px;" src="/materials/biz/images/S11-01-02-01_01-001.png">
</div>


## 4. Copilot의 핵심 기능과 비즈니스 활용

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
                <td class="td-rowheader" rowspan="4">Word</td>
                <td>문서 초안 작성</td>
                <td style="text-align: left;">
                    - 문서 초안을 텍스트 프롬프트, 또는 다른 Word/Excel/PowerPoint 파일의 내용을 기반으로 작성<br>
                    - 예시: "이 프로젝트의 목표와 진행 상황에 대한 1페이지 보고서 초안을 작성해 줘."
                </td>
            </tr>
            <tr>
                <td>내용 요약 및 재작성</td>
                <td style="text-align: left;">
                    - 긴 문서를 요약하거나, 선택한 텍스트를 다른 톤앤매너/길이로 재작성<br>
                    - 예시: "이 문서를 500자 이내로 요약하고, 비전문가도 이해하기 쉽게 다시 써 줘."
                </td>
            </tr>
            <tr>
                <td>정보 추출</td>
                <td style="text-align: left;">
                    - 다른 문서를 참조하여 특정 정보를 현재 문서에 삽입<br>
                    - 예시: "경쟁사분석.docx" 파일을 참조하여 '기술적 우위' 섹션 초안을 작성해 줘
                </td>
            </tr>
            <tr>
                <td>아이디어 브레인스토밍</td>
                <td style="text-align: left;">
                    - 예시: "새로운 마케팅 전략에 대한 아이디어 5가지 제안해 줘."
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="2">Excel</td>
                <td>데이터 분석 및 인사이트 도출</td>
                <td style="text-align: left;">
                    - 자연어로 질문하면 스프레드시트 데이터를 분석하고, 패턴을 찾거나, 수식을 제안하는 등 데이터에서 중요한 인사이트를 요약하여 제시<br>
                    - 예시: "데이터에서 가장 매출이 높은 상위 5개 제품을 찾아서, 매출 증감 추이를 그래프로 보여줘."
                </td>
            </tr>
            <tr>
                <td>데이터 시각화 제안</td>
                <td style="text-align: left;">
                    - 분석 결과를 기반으로 차트나 그래프 생성<br>
                    - 예시: "이 스프레드시트의 '판매 데이터' 시트에서 지난 3개월간 제품별 판매량 추이를 분석하고, 막대 차트로 시각화해 줘"
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="4">PowerPoint</td>
                <td>프레젠테이션 초안 생성</td>
                <td style="text-align: left;">
                    - Word 문서나 텍스트를 기반으로 프레젠테이션 슬라이드 초안을 자동 생성<br>
                    - 예시: "월별 실적 보고서에 대한 5장짜리 프레젠테이션 초안을 만들어 줘."
                </td>
            </tr>
            <tr>
                <td>슬라이드 내용 보강</td>
                <td style="text-align: left;">
                    - 특정 슬라이드의 내용을 요약하거나 다른 문서/데이터를 참조하여 보충함<br>
                    - 예시: "이 슬라이드에 있는 주요 내용을 간결한 불릿 포인트로 정리하고, 관련 이미지를 추천해 줘."
                </td>
            </tr>
            <tr>
                <td>텍스트 다듬기</td>
                <td style="text-align: left;">
                    - 슬라이드의 표현을 더 매끄럽게 하거나 특정 목적에 맞는 표현으로 조절함<br>
                    - 예시: "이 슬라이드의 텍스트를 더 전문적인 용어로 다듬어 줘."
                </td>
            </tr>
            <tr>
                <td>디자인 아이디어</td>
                <td style="text-align: left;">
                    - 시각적 레이아웃이나 이미지 배치에 대한 아이디어 제안<br>
                    - 예시: "'프로젝트_기획서.docx' 파일을 바탕으로 7개 슬라이드의 발표 자료 초안을 만들어 줘"
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">Outlook</td>
                <td>이메일 초안 작성</td>
                <td style="text-align: left;">
                    - 받은 메일 내용을 바탕으로 회신 이메일 초안을 작성하거나, 특정 정보를 포함한 새 이메일을 작성함<br>
                    - 예시: "고객의 문의 사항에 대한 친절한 답변 이메일을 작성해 줘."
                </td>
            </tr>
            <tr>
                <td>긴 이메일 요약</td>
                <td style="text-align: left;">
                    - 복잡한 이메일 대화 내용을 간결하게 요약하여 빠르게 핵심을 파악할 수 있도록 지원<br>
                    - 예시: "이 긴 이메일 스레드에서 핵심 내용과 필요한 액션 아이템만 요약해 줘."
                </td>
            </tr>
            <tr>
                <td>캘린더 제안</td>
                <td style="text-align: left;">
                    - 이메일 내용을 분석하여 미팅 일정을 제안함
                    - 예시: "이메일의 내용과 현재 등록되어 있는 일정을 참고하여 미팅 일정을 제시해 줘."
                </td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">Teams</td>
                <td>회의 요약</td>
                <td style="text-align: left;">
                    - 회의 내용(녹화/전사된 내용)을 실시간으로 요약하고, 핵심 결정 사항, 논의 내용, 액션 아이템을 정리<br>
                    - 예시: "오늘 미팅에서 결정된 핵심 사항과 담당자별 액션 아이템을 정리해 줘"
                </td>
            </tr>
            <tr>
                <td>채팅 내용 요약 및<br>빠른 정보 찾기</td>
                <td style="text-align: left;">
                    - 긴 채팅 스레드의 주요 내용 요약<br>
                    - 채팅 기록에서 특정 정보를 빠르게 찾아줌
                </td>
            </tr>
            <tr>
                <td>아이디어 생성</td>
                <td style="text-align: left;">
                    - 미팅 중에 특정 주제에 대한 아이디어 생성
                </td>
            </tr>
        </tbody>
    </table>
</div>

## 5. Copilot의 강점과 한계

- **강점**
    - **실시간 컨텍스트 이해**
        - M365 앱 내에서 작업 중인 문맥과 조직 데이터를 이해하여 고도로 개인화되고 관련성 높은 답변 생성
    - **원활한 워크플로우 통합**
        - 앱을 전환할 필요 없이, 작업 흐름 내에서 직접 AI 기능을 활용
    - **생산성 극대화**
        - 단순 반복 작업을 줄이고
        - 창의적이고 전략적인 업무에 집중할 수 있도록 지원
    - **보안 및 규정 준수**
        - 조직의 데이터 거버넌스 및 보안 규정을 준수하며 작동

- **한계**
    - **M365 생태계 종속**
        - Microsoft 365 환경 밖의 앱이나 서비스와는 직접적인 연동이 제한될 수 있음
    - **여전히 인간의 검토 필수**
        - 생성된 내용의 정확성, 비즈니스 목표 부합 여부 등을 인간이 최종적으로 검토하고 승인해야 함
    - **프라이버시/보안 우려**
        - 내부 데이터를 활용하는 만큼, 데이터 사용 방식에 대한 사용자들의 프라이버시 및 보안 우려가 존재함


## 6. AI/LLM이 업무에 가져올 변화

- **LLM 기반 스마트 워크의 비전**
    - **업무의 '자동화'를 넘어 '고도화'로**
        - 단순히 반복 작업을 줄이는 것을 넘어,
        - AI가 분석과 콘텐츠 생성을 통해 업무의 품질 자체를 향상시킴

    - **창의성 및 전략적 사고에 집중**
        - 시간 소모적인 초안 작성, 정보 검색 등에서 해방되어,
        - 인간 고유의 창의력, 비판적 사고, 전략 수립에 더 많은 시간을 할애할 수 있음

    - **새로운 업무 방식의 창출**
        - AI와 협업하는 새로운 방식의 업무가 탄생하고,
        - 이는 개인의 역량 강화뿐만 아니라 조직 전체의 생산성 증대로 이어짐



> - **AI는 조력자이자 확장된 나**
>   - AI/LLM은 우리 업무를 빼앗아 가는 것이 아니라, 우리의 역량을 수십 배로 확장시켜 주는 **강력한 '조력자(Copilot)'**
>   - 이 도구들을 이해하고 능숙하게 활용하는 능력이 미래 업무 환경의 핵심 경쟁력이 될 것
{:.summary-quote}

