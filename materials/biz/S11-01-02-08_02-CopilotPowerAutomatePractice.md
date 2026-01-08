---
layout: page
title:  "Copilot + Power Automation + GPT 통합 자동화 실습"
date:   2025-12-14 10:00:00 +0900
permalink: /materials/S11-01-02-08_02-CopilotPowerAutomatePractice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - '라즈베리파이 기반 엣지 AI 자율주행 세그멘테이션 모델 상용화 추진' 프로젝트를 가정
{: .common-quote}

## 1. Word 보고서 내용 심층 분석 및 요약 + Teams 알림

- **목표**
    - Word로 작성된 주간/월간 기술 보고서의 핵심 내용을 GPT가 심층 분석하여 비즈니스적 시사점을 도출하고,
    - 이를 간결하게 요약하여 관련 팀에 Teams로 알림

- **구성 요소**
    - Word Copilot, Power Automate, GPT

- **전체 프로세스**
    1. **[Word Copilot 트리거]**
        - 주간 기술 보고서(`주간_기술보고서_20251230.docx`)를 Word에서 작성 후,
        - Copilot에게 "이 보고서의 기술적 내용에서 비즈니스적 시사점을 도출하고 핵심 요약 5개를 생성하여 팀즈로 공유해 줘."라고 요청
        
    2. **[Power Automate 시작]**
        - Word Copilot이 이 보고서의 전체 텍스트와 요청(`비즈니스적 시사점 도출 및 핵심 요약 5개 생성`)을 Power Automate 플로우로 전달
        - 플로우는 '특정 키워드가 포함된 Word 문서가 OneDrive에 업데이트될 때' 등으로 트리거될 수도 있음

    3. **[Power Automate ➜ GPT]**
        - Power Automate는 Word 문서의 텍스트와 요청을 GPT API로 전송
        - 이때 Power Automate는 다음과 같은 프롬프트를 GPT에 구성하여 보낼 수 있음

        ```
        #Role: 기술 분석가 및 비즈니스 전략가

        아래 [Word 보고서 텍스트]를 분석하여 다음을 수행해 줘:
        1. 이 기술 보고서 내용에서 도출할 수 있는 '비즈니스적 시사점' 3가지를 명확하고 간결하게 설명해 줘.
           (예: 신규 시장 기회, 경쟁 우위, 잠재적 리스크)
        2. 보고서 전체 내용을 '고위 경영진'이 빠르게 파악할 수 있도록 '핵심 요약' 5개 불릿 포인트로 정리해 줘.

        [Word 보고서 텍스트]: [Power Automate가 Word에서 추출한 보고서 텍스트]
        ```

    4. **[GPT ➜ Power Automate]**
        - GPT는 분석 결과(비즈니스 시사점 3가지, 핵심 요약 5가지)를 Power Automate에 반환
        
    5. **[Power Automate 액션]**
        - Power Automate는 GPT 결과물을 받아서 다음과 같은 액션을 수행
            - `Outlook 커넥터`: 비즈니스 시사점과 핵심 요약을 정리하여 관련 경영진에게 이메일을 발송
            - `Teams 커넥터`: 프로젝트 팀 채널에 핵심 요약을 포스팅하고, 이메일 링크를 공유
            - (옵션) `Word 커넥터`: GPT가 생성한 비즈니스 시사점을 보고서의 별도 섹션(`비즈니스 시사점 분석`)에 삽입하여 보고서를 업데이트
            
- **명령 시 주의사항**
    - Word Copilot에 명확히 '비즈니스적 시사점'과 '요약'을 요청해야 함
    - Power Automate에서는 Word 텍스트를 GPT 프롬프트에 효율적으로 삽입하고,
    - GPT 응답을 Teams 메시지 형식에 맞게 파싱하는 로직을 잘 설계해야 함


## 2. Excel 데이터 기반 심층 시장 분석 + PowerPoint 제안서 슬라이드 초안 생성

- **목표**
    - Excel에 있는 판매/시장 데이터와 경쟁사 분석 데이터를 GPT가 분석하여 시장 트렌드를 도출하고,
    - 이를 바탕으로 PowerPoint 제안서의 초안 슬라이드를 자동으로 생성

- **구성 요소**
    - Excel Copilot, Power Automate, GPT, PowerPoint

- **전체 프로세스**
    1. **[Excel Copilot 트리거]**
        - [`시장_판매_데이터_2025.xlsx`](./docs/시장_판매_데이터_2025.xlsx) 파일에서 Excel Copilot에게 다음과 같이 요청
            - "이 스프레드시트의 '제품 판매' 시트와 '경쟁사' 시트를 분석하여, 라즈베리파이 엣지 AI 시장의 최근 6개월 동향과 우리 제품의 시장 점유율에 대한 상세 분석을 수행하고, 이를 바탕으로 '시장 분석 및 전략 제안' PowerPoint 슬라이드 초안을 만들어 줘."
        
    2. **[Power Automate 시작]**
        - Excel Copilot은 '제품 판매' 및 '경쟁사' 시트의 데이터를 Power Automate 플로우로 전달
        
    3. **[Power Automate ➜ GPT]**
        - Power Automate는 Excel에서 추출한 데이터를 가공(예: JSON 형식으로 변환)하여 GPT API로 전송

        ```
        #Role: 시장 분석 전문가 및 전략 컨설턴트

        아래 제공된 [시장 판매 데이터]와 [경쟁사 분석 데이터]를 기반으로 다음을 수행해 줘:
        1. 라즈베리파이 엣지 AI 시장의 최근 6개월간 주요동향 3가지를 도출하고, 각각에 대한 간략한 설명을 추가해줘.
        2. 우리 제품 '라즈베리파이 엣지 AI 세그멘테이션 솔루션'의 현재 시장점유율(가정: 5%) 및 잠재성장률을 분석
           하고, 주요 경쟁사 대비 강점 3가지와 약점 2가지를 설명해 줘.
        3. 이 분석을 바탕으로, '시장분석 및 전략제안' PowerPoint 슬라이드 3개(제목 및 핵심내용)의 초안을 제시해줘.
           각 슬라이드의 제목과 주요 내용 3~4개의 불릿 포인트를 포함해 줘.

        [시장 판매 데이터]: [Power Automate가 Excel에서 추출한 데이터]
        [경쟁사 분석 데이터]: [Power Automate가 Excel에서 추출한 데이터]
        ```

    4. **[GPT ➜ Power Automate]**
        - GPT는 시장 동향, 자사 제품 분석, 그리고 3개 슬라이드의 제목과 내용으로 구성된 JSON 형식의 결과를 Power Automate에 반환
        
    5. **[Power Automate 액션]**
        - Power Automate는 GPT 결과를 받아서
            - `PowerPoint 커넥터`
                - GPT가 제안한 슬라이드 초안을 바탕으로 새로운 PowerPoint 프레젠테이션(`시장_분석_전략_제안_2025_초안.pptx`)을 생성하거나,
                - 기존 프레젠테이션에 슬라이드를 추가

            - (옵션) `Email 커넥터`
                - PowerPoint 초안이 완성되었음을 알리는 이메일을 담당자에게 발송
                
- **명령 시 주의사항**
    - Excel 데이터의 품질과 명확한 헤더가 중요함
    - GPT 프롬프트에서 어떤 시트를 분석할지, 어떤 관점(시장 동향, 점유율, 강약점)으로 분석할지 명확히 해야 함
    - PowerPoint 커넥터 사용 시, 슬라이드 레이아웃이나 서식은 템플릿을 활용하여 Power Automate에서 지정할 수 있음


## 3. Outlook 메일 내용 기반 고객 요청 분석 ➜ GPT 심층 분석 ➜ Jira 티켓 생성

- **목표**
    - 특정 고객으로부터 받은 기술 문의 메일 내용을 GPT가 분석하여 핵심 문제와 예상 해결책을 도출하고,
    - 이를 바탕으로 Jira(또는 다른 프로젝트 관리 도구)에 기술 지원 티켓을 자동 생성

- **구성 요소**
    - Outlook Copilot, Power Automate, GPT, Jira (또는 다른 웹 서비스)

- **전체 프로세스**
    1. **[Outlook Copilot 트리거]**
        - Outlook에서 특정 고객 (`@고객사`)으로부터 '라즈베리파이 모델 관련 문의' 메일을 수신
        - Outlook Copilot에게 "메일내용을 분석하여 핵심기술 문제와 가능한 해결책을 도출하고, 이를 Jira에 기술지원티켓으로 생성해줘."라고 요청

    2. **[Power Automate 시작]**
        - Outlook Copilot (또는 '특정 제목/발신자의 메일 수신'을 트리거로 하는 Power Automate 플로우)은 해당 이메일의 전문(제목, 본문, 첨부파일 유무 등)을 Power Automate 플로우로 전달

    3. **[Power Automate ➜ GPT]**
        - Power Automate는 이메일 본문을 GPT API로 전송

        ```
        #Role: 기술 지원 전문가

        아래 [고객 문의 이메일 본문]을 분석하여 다음 정보를 추출하고, 잠재적인 해결책을 제안해 줘:
        1. 고객이 겪고 있는 '핵심 기술 문제'를 1~2문장으로 명확히 요약해 줘.
        2. 이 문제의 '심각도'를 'High', 'Medium', 'Low' 중 하나로 평가해 줘.
        3. 문제 해결을 위한 '예상 해결책' 3가지(또는 그 이하)를 간략히 제시해 줘. 
        4. '문제 관련 키워드' 3~5개를 추출해 줘.

        [고객 문의 이메일 본문]: [Power Automate가 Outlook에서 추출한 이메일 본문]
        ```

    4. **[GPT ➜ Power Automate]**
        - GPT는 분석된 문제 요약, 심각도, 예상 해결책, 키워드 등의 구조화된 데이터를 Power Automate에 반환
        
    5. **[Power Automate 액션]**
        - Power Automate는 GPT 결과물을 받아서:
            - `Jira 커넥터`
                - GPT가 분석한 내용을 바탕으로 Jira에 새로운 '이슈(티켓)'를 생성
                    - Jira 이슈 타입: '기술 지원'
                    - 제목: GPT의 핵심 문제 요약
                    - 설명: GPT의 상세 요약 + 해결책 제안
                    - 담당자: 자동 할당
                    - 라벨: GPT의 키워드

            - `Outlook 커넥터`
                - 고객에게 "문의가 접수되었으며, 곧 담당자가 연락할 것"이라는 자동 응답 메일을 보냄

            - `Teams 커넥터`
                - 담당 팀원에게 Jira 티켓이 생성되었음을 알리는 메시지를 보냄

- **명령 시 주의사항*
    - 이메일 본문에서 고객의 의도를 정확히 파악하는 것이 중요함
    - GPT 프롬프트에서 어떤 정보를 추출하고 싶은지 매우 구체적으로 요청해야 함
    - Jira 커넥터 사용 시, 이슈 생성에 필요한 필수 필드(프로젝트, 이슈 타입 등)를 Power Automate에서 정확히 매핑해야 함
    - 심각도는 GPT가 제안하지만, 실제 고객 지원 정책에 맞춰 수동 조정이 필요할 수 있음

> - **참고**
> - [Microsoft Learn: 자동화 센터의 Copilot 사용-Power Automate](https://learn.microsoft.com/ko-kr/power-automate/automation-center-copilot){: target="_blank"}
> - [Microsoft Learn: Power Automate에서 Copilot으로 자동화 적용](https://learn.microsoft.com/ko-kr/power-automate/copilot-overview){: target="_blank"}
> - [Microsoft: 자동화 하고 싶은 업무를 직접 입력해 보세요!](https://www.youtube.com/watch?v=lep1mj9FRzw){: target="_blank"}