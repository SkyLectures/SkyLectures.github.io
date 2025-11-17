---
layout: page
title:  "자율주행 레벨(Level)의 이해"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S10-01-02-01_01-AutonomousDrivingLevels
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 자율주행 레벨 기준

- 자율주행 레벨(Level)은 자동화 항목, 운전 주시, 시스템 오류 대응 등의 행위 주체와 자동화 구간 등에 따라 6단계로 구성됨
    - 국제표준 J3016, SAE(Society of Automotive Engineers, 미국자동차공학회) 2019 제정

<div style="text-align:center; width: 80%;"><h4>자율 주행차의 레벨(Level, 단계)</h4></div>
<div class="info-table" style="width: 80%;">
    <table>
        <caption>(자료출처: 정보통신기획평가원, ICT R&D 기술로드맵 2025 보고서 - 4. 인공지능·SW·자율주행자동차)</caption>
        <thead>
            <th style="width: 60px;">단계</th>
            <th style="width: 150px;">특징</th>
            <th>자율주행차 기능 및 내용</th>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2" class="td-rowheader">0단계</td>
                <td rowspan="2">비자동화<br>No Automation</td>
                <td class="td-left">운전자는 차의 속도와 방향을 통제하며, 시스템은 주행에 전혀 영향을 주지 않는 단계</td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 없음(경고 등)<br>
                    * 운전 주시 : 항시 필수<br>
                    * 자동화 구간 : 없음
                </td>
            </tr>
            <tr>
                <td rowspan="2" class="td-rowheader">1단계</td>
                <td rowspan="2">운전자 지원<br>Driver Assistance</td>
                <td class="td-left">
                    운전자는 차의 속도와 방향을 통제하며, 시스템은 특정 주행모드에서 일시 강제운전으로 개입하여 조향 또는 감가속 중 하나의 기능을 수행 (예: 전방충돌방지 보조, 후측방 충돌경고, 차선유지보조, 속도조절기능 등)
                </td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 조향 or 속도<br>
                    * 운전 주시 : 항시 필수<br>
                    * 비상대응 : 운전자<br>
                    * 자동화 구간 : 특정구간
                </td>
            </tr>
            <tr>
                <td rowspan="2" class="td-rowheader">2단계</td>
                <td rowspan="2">부분 자동화<br>Partial Automation</td>
                <td class="td-left">
                    운전자는 주행환경을 항시 주시하고 적극적 차량 조작에 개입, 운전자가 비상대응에 책임을 지며, 시스템은 정해진 조건에서 속도와 방향을 복합적으로 자동화함
                </td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 조향 & 속도<br>
                    * 운전 주시: 항시 필수<br>
                    * 자동화 구간 : 특정구간
                </td>
            </tr>
            <tr>
                <td rowspan="2" class="td-rowheader">3단계</td>
                <td rowspan="2">조건부 자동화<br>Conditional Automation</td>
                <td class="td-left">
                    특정 주행모드에서 시스템이 조향 및 감가속을 수행하는 고속도로 주행보조이며,운전자는 주행 환경의 주시와 비상 대응에 책임
                </td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 없음(경고 등)<br>
                    * 운전 주시 : 항시 필수<br>
                    * 자동화 구간 : 없음
                </td>
            </tr>
            <tr>
                <td rowspan="2" class="td-rowheader">4단계</td>
                <td rowspan="2">고도 자동화<br>High Automation</td>
                <td class="td-left">
                    운전자는 정해진 조건 하에서 전혀 개입하지 않으며, 시스템이 적극적 운전조작과 주행환경의 주시 및 비상시의 대처 등을 수행
                </td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 조향 & 속도<br>
                    * 운전 주시 : 작동 구간 내 불필요<br>
                    * 자동화 구간 : 특정구간
                </td>
            </tr>
            <tr>
                <td rowspan="2" class="td-rowheader">5단계</td>
                <td rowspan="2">완전 자동화<br>Full Automation</td>
                <td class="td-left">모든 도로 조건과 환경에서 시스템이 항상 주행 담당</td>
            </tr>
            <tr>
                <td class="td-left">
                    * 자동화 항목 : 조향 & 속도<br>
                    * 운전 주시 : 전 구간 불필요<br>
                    * 자동화 구간 : 전 구간
                </td>
            </tr>
        </tbody>
    </table>
</div>

## 2. 자율주행 기술의 현재와 전망

- 2025년 기준으로 자율주행 시장에서 가장 널리 보급된 단계는 **레벨 2(부분 자동화)**
    - 자율주행 기술 상용 적용 상황
        - 레벨 1: 크루즈 컨트롤(속도 유지), 차선 유지 보조 시스템 등
        - 레벨 2: 테슬라의 오토파일럿, GM의 슈퍼 크루즈, 현대차의 HDA 등
        - 크루즈 컨트롤의 경우, 조향/속도 중에서 한 가지를 시스템이 조절하면 레벨 1, 두 가지 모두 조절하면 레벨 2

    - 자율주행 기술 실험용 적용 상황
        - 레벨 3: 일부 고급 차량에서 제한적으로 도입 중
        - 레벨 4: 일부 로보택시 서비스와 셔틀 서비스가 제한된 지역에서 테스트 중
        - 레벨 5: 개발 및 연구 단계

- **레벨 2가 주류로 자리잡은 이유**
  - 법적 제약이 적음 (운전자가 여전히 운전에 관여)
  - 뛰어난 가성비 (고도화된 시스템보다 비용이 낮으면서도 실질적인 운전 보조 효과 제공)
  - 소비자의 신뢰와 익숙함 (점진적으로 자율주행에 익숙해질 수 있는 단계)

- 업계에서는 2030년 전후로 레벨 4~5 차량이 본격적으로 시장에 진입할 것으로 예상

<br><br>

> - 참고
>   - **2025 경주 APEC, 한국형 자율주행 셔틀버스, 경주 APEC 손님 맞는다.**<br>
>       - 출처: [대한민국정책브리핑-정책포커스](https://www.korea.kr/news/policyFocusView.do?pkgId=49500822&newsId=148952839&pWise=main&pWiseMain=F2){: target="blank"}
>       <br><br>
>       - 2025 경주 APEC에서 운행 중인 자율주행 셔틀버스는 국내 최초로 **레벨 4 자율주행 기술을 적용**한 셔틀버스로,<br>
>       APEC 기간 동안 경주 보문단지 일대를 순환하는 노선에서 운행되었습니다.
>       - 특히 B형 셔틀버스는 운전석이 없는 형태로 설계되어 있으며, 안전요원 1명을 포함해 총 11명이 탑승할 수 있습니다.
>       - 또한 이 자율주행 버스는 96%라는 높은 국산화율을 달성했다는 점에서도 주목받고 있습니다.
>       <div class="insert-image" style="text-align: center;">
>           <img style="width: 400px;" src="/materials/project/images/S10-01-02-01_01-001.jpg">
>       </div>
{: .common-quote}