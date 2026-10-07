---
layout: page
title: (워크숍) 사업재편 AI 과제발굴
subtitle: 
permalink: /materials/S06-05-01-05_01-IdentifyingAiProjects
description: 신규 사업라인/기존 공정 페인포인트 도출 및 난이도 & ROI 기반 과제 우선순위 매트릭스 작성
categories:
    - materials
tags:
    - 제조산업
    - Manufacturing
    - AX
    - AI Transformation
related_posts:
    - /materials/S06-05-01-01_01-AxInsight
    - /materials/S06-05-01-02_01-AiRoadmapForBusinessRestructuring
    - /materials/S06-05-01-03_01-ManufacturingAiCaseStudy1
    - /materials/S06-05-01-03_02-ManufacturingAiCaseStudy2
    - /materials/S06-05-01-04_01-MaximizingWorkProductivity
qna: false
---
* toc
{:toc}




> - **실습 진행**
>   - [0~10분] 개별 문제 도출 및 Part 1 스크리닝 작성
>   - [10~20분] 조별 상호 리뷰(동료 평가)를 거쳐 Quick-Win 1개 확정
>   - [20~30분] Part 2 과제 정의서 1-Pager 집중 작성
>   - [30~50분] 2~3개 조 대표 과제 발표 및 피드백<br><br>
> - **주의사항**
>   - 다음과 같은 실수를 하지 않도록 주의할 것
>       - 어떤 데이터를 모을지 안 정하고 AI부터 찾기
>       - 기대효과를 '품질 향상' 같은 추상적 문구로 적기
{: .yellow-quote}


## 1. 과제 스크리닝 및 평가 매트릭스

- 막연한 아이디어를 나열하지 않도록, 비즈니스 효과(Impact)와 실행 가능성(Feasibility)을 각각 3가지 하위 지표(각 5점 만점, 총 15점)로 객관화하여 채점

### 1.1 평가 기준 가이드 (채점표)

<div class="info-table">
    <table>
        <thead>
            <th style="width: 100px;">구분</th>
            <th style="width: 0px;"> </th>
            <th style="width: 140px;">평가 항목</th>
            <th style="width: 240px;">1점 (낮음/어려움)</th>
            <th style="width: 240px;">3점 (보통)</th>
            <th style="width: 260px;">5점 (높음/수월)</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader" rowspan="3">비즈니스 효과<br>(Impact)</td>
                <td> </td>
                <td class="td-left"><b>① 재무적 가치 (ROI)</b></td>
                <td class="td-left">단순 편의성 개선, 비용 절감 미미</td>
                <td class="td-left">연간 수천만 원 단위 원가 절감/생산성 향상</td>
                <td class="td-left">억 단위 원가 절감, 신규 매출 창출 직결</td>
            </tr>
            <tr>
                <td> </td>
                <td class="td-left"><b>② 사업재편 정합성</b></td>
                <td class="td-left">기존 단순 반복 업무 유지</td>
                <td class="td-left">기존 공정/라인 일부 개선</td>
                <td class="td-left">신규 사업 라인 가동, 친환경/고부가 전환 핵심</td>
            </tr>
            <tr>
                <td> </td>
                <td class="td-left"><b>③ 확산/파급력</b></td>
                <td class="td-left">특정 설비 1대에만 국한</td>
                <td class="td-left">자사 전 공정 또는 유사 라인 수평전개 가능</td>
                <td class="td-left">협력사/고객사 밸류체인 전반으로 확장 가능</td>
            </tr>
            <tr>
                <td class="td-rowheader" rowspan="3">실행 가능성<br>(Feasibility)</td>
                <td> </td>
                <td class="td-left"><b>① 데이터 가용성</b></td>
                <td class="td-left">데이터 미수집, 수기 관리 (수집 체계 필요)</td>
                <td class="td-left">센서/DB는 있으나 라벨링·정제 부족</td>
                <td class="td-left">정형 센서/이미지 DB 축적 완료, 라벨링 용이</td>
            </tr>
            <tr>
                <td> </td>
                <td class="td-left"><b>② 기술/도입 난이도</b></td>
                <td class="td-left">세계 최초 시도 수준, R&D 필수</td>
                <td class="td-left">상용 솔루션/오픈소스 모델 커스텀 필요</td>
                <td class="td-left">검증된 상용 AI 솔루션 즉시 도입 가능</td>
            </tr>
            <tr>
                <td> </td>
                <td class="td-left"><b>③ 현장 수용성/규제</b></td>
                <td class="td-left">작업자 거부감 극심, 안전/인증 규제 복잡</td>
                <td class="td-left">현장 협조 필요, 표준 운영 절차(SOP) 수정</td>
                <td class="td-left">현장 작업자 니즈 큼, 규제 리스크 없음</td>
            </tr>
        </tbody>
    </table>
</div>

<br>

### 1.2 과제 후보군 스크리닝 시트 (A4 앞면)

```
팀/작성자:                                대상 라인/제품군:
```

<div class="info-table">
    <table>
        <thead>
            <th style="width: 40px;">No</th>
            <th style="width: 240px;">후보 과제명</th>
            <th style="width: 260px;">해결하려는 문제 (Pain Point)</th>
            <th style="width: 180px;">적용 AI 유형<br>(Vision/ML/LLM/Agent)</th>
            <th style="width: 100px;">비즈니스 효과<br>(합계 15점)</th>
            <th style="width: 100px;">실행 가능성<br>(합계 15점)</th>
            <th style="width: 100px;">최종 분류<br>(아래 2x2 참조)</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader"><i>(예시)</i></td>
                <td class="td-left"><i>신규 배터리팩 케이스 용접 비전 검사</i></td>
                <td class="td-left"><i>미세 기공 불량 수작업 전수 검사 한계</i></td>
                <td><i>Vision DL</i></td>
                <td><i>13점</i></td>
                <td><i>12점</i></td>
                <td><i>Quick-Win</i></td>
            </tr>
            <tr><td class="td-rowheader">1</td><td></td><td></td><td></td><td>점</td><td>점</td><td></td></tr>
            <tr><td class="td-rowheader">2</td><td></td><td></td><td></td><td>점</td><td>점</td><td></td></tr>
            <tr><td class="td-rowheader">2</td><td></td><td></td><td></td><td>점</td><td>점</td><td></td></tr>
        </tbody>
    </table>
</div>

<br>

### 1.3 과제 포지셔닝 2×2 매트릭스

<div class="insert-image" style="width: 600px;">
    <img src="/materials/S06_SmartFactory/images/S06-05-05-01_01-001_PositioningMatrix.png">
    <span class="caption">(Source: Sky Lectures / AiDALab)</span>
</div>

<br>

## 2. Quick-Win 1-Page 과제 정의서

- 위 매트릭스에서 **[Quick-Win]** 사분면에 선정된 최우선 과제 1개를 구체화하는 템플릿


- **[사업재편 AX 과제 정의서]**

    <div class="info-table">
        <table>
            <thead><th colspan="4" width="1100px">과제 기본 정보</th></thead>
            <tbody>
                <tr>
                    <td class="td-rowheader" width="120px">과제명</td>
                    <td class="td-left" colspan="3">(예: 친환경 모터 코어 가공 라인 설비 이상 감지 및 예지보전 AI)</td>
                </tr>
                <tr>
                    <td class="td-rowheader" width="120px">추진 부서</td>
                    <td class="td-left">생산기술팀 / DX추진TF</td>
                    <td class="td-rowheader" width="120px">사업재편 목표</td>
                    <td class="td-left">[ ] 신사업 라인 구축    [v] 기존 라인 고도화    [ ] 규제/인증 대응</td>
                </tr>
                <tr>
                    <td class="td-rowheader">예상 추진기간</td>
                    <td class="td-left">PoC (3개월) → 현장 양산 적용 (3개월)</td>
                    <td class="td-rowheader">적용 AI 기술</td>
                    <td class="td-left">[ ] Vision   [v] Tabular/시계열 ML   [ ] LLM/RAG   [ ] Agent</td>
                </tr>
            </tbody>
        </table>
    </div>


- **배경 및 문제 정의 (Why)**
    - **현장의 고질적 문제 (As-Is):**
        - *예: 가공 설비 스핀들 베어링 마모로 불시 라인 정지(연 4회 발생). 1회 정지당 복구 12시간 소요 및 납기 지연 리스크 발생.*

    - **문제의 근본 원인:**
        - *예: 숙련공의 청각/촉각 기반 간헐적 점검에 의존하여 초기 진동 이상 징후 감지 불가.*


- **AI 적용 방안 및 목표 (What & How)**
    - **AI 도입 후 모습 (To-Be):**
        - *예: 메인 모터에 진동/전류 센서를 장착하고, 시계열 머신러닝 이상 감지 모델이 고장 징후를 최소 72시간 전 경보.*

    - **필요 데이터 현황:**
        - 기존 보유 데이터: *설비 가동 이력(MES 로그 2년치)*
        - 신규 수집 필요 데이터: *스핀들 모터 고주파 진동 데이터(센서 신규 설치 필요)*


- **정량적 기대효과 (Impact / ROI)**
    - **정량 지표:**
        - 불시 다운타임(Downtime) 시간: 기존 연간 48시간 → **연간 6시간 이하 (87.5% 감축)**
        - 연간 비용 절감: **약 1.2억 원 절감** *(시간당 라인 중단 손실액 250만 원 기준)*

    - **정성 지표:**
        - 신규 거래처 대상 스마트 품질 관리 역량 입증 (사업재편 계획 승인 시 가점 확보)


- **리스크 및 장애요인 극복 방안 (Risk & Solution)**
    - **발생 가능 리스크:** *현장 가공 칩/절삭유로 인한 센서 내구성 저하 및 오경보(False Alarm) 발생 가능성*
    - **대응 방안:** *방수/방진(IP67) 규격 산업용 무선 센서 채택 및 초기 1개월간 임계값 튜닝 기간 운영*

- **다음 단계 액션 플랜 (Next Step)**
    - **D+1주:** 설비팀 협의 후 센서 부착 위치 및 PLC 데이터 추출 가능 여부 현장 실사
    - **D+2주:** 외부 AI 전문 기업(솔루션사) 2곳 미팅 및 레퍼런스 확인
    - **D+4주:** 경영진 보고용 PoC(개념검증) 예산안 확정



## 3. 참고자료

- **Quick-Win 1-Page 과제 정의서**를 작성할 때 기준점으로 삼을 수 있는 주요 업종별 대표 예시 3종

    - **[예시 1] 자동차 부품 업종 (공정 전환 & 설비 보전)**

        <div class="info-table">
            <table>
                <thead><th colspan="4" width="1100px">과제 기본 정보</th></thead>
                <tbody>
                    <tr>
                        <td class="td-rowheader" width="120px">과제명</td>
                        <td class="td-left" colspan="3">전기차 감속기 기어 치형 가공 라인 설비 진동 예지보전 AI</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader" width="120px">추진 부서</td>
                        <td class="td-left">생산기술팀 / 설비보전TF</td>
                        <td class="td-rowheader" width="120px">사업재편 목표</td>
                        <td class="td-left">[ ] 신사업 라인 구축    [v] 기존 라인 고도화    [ ] 규제/인증 대응</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader">예상 추진기간</td>
                        <td class="td-left">PoC (2.5개월) 🡪 라인 양산 적용 (2개월)</td>
                        <td class="td-rowheader">적용 AI 기술</td>
                        <td class="td-left">[ ] Vision   [v] Tabular/시계열 ML   [ ] LLM/RAG   [ ] Agent</td>
                    </tr>
                </tbody>
            </table>
        </div>


        1. **배경 및 문제 정의 (Why)**
            - **현장의 고질적 문제 (As-Is):**
                - 기존 내연기관 수동변속기 부품에서 고정밀 EV 감속기 기어로 사업 전환 중
                - 기어 연삭(Grinding) 설비의 스핀들 베어링 마모 시 미세 진동으로 제품 치형 오차가 발생하며, 연간 5회 불시 설비 정지 발생(회당 복구 8시간 소요)

            - **문제의 근본 원인:**
                - 가공 정밀도가 기존 대비 3배 이상 요구되나, 베어링 이상 여부를 작업자의 청각/촉각 점검 및 정기 3개월 교체 주기에만 의존


        2. **AI 적용 방안 및 목표 (What & How)**
            - **AI 도입 후 모습 (To-Be):**
                - 연삭 설비 스핀들 하우징에 3축 가속도 진동 센서 및 CT 전류 센서를 부착
                - 시계열 머신러닝 이상 감지 알고리즘이 정상 파형 이탈을 감지하여 고장 최소 72시간 전 경보 및 부품 교체 지침 알림

            - **필요 데이터 현황:**
                - 기존 보유 데이터: 설비 보전 일지(Excel 3년치), MES 가동/비가동 로그
                - 신규 수집 데이터: 스핀들 3축 진동 가속도 데이터(초당 2,000Hz), 모터 부하 전류값


        3. **정량적 기대효과 (Impact / ROI)**
            - **정량 지표:**
                - 불시 다운타임(Downtime): 연간 40시간 🡪 **연간 4시간 이하 (90% 감축)**
                - 연간 비용 절감: **약 1억 800만 원** (비가동 손실 시간당 250만 원 $\times$ 36시간 + 스크랩 불량 절감 1,800만 원)

            - **정성 지표:**
                - 전방 완성차(OEM) 감사 시 스마트 예지보전 체계 입증으로 품질 신뢰도 가점 확보


        4. **리스크 및 장애요인 극복 방안 (Risk & Solution)**
            - **발생 가능 리스크:** 가공 시 튀는 절삭유 및 미세 쇠가루(Chip)로 인한 센서 통신 단선 위험
            - **대응 방안:** IP68 방수·방진 규격 유선 방폭 센서 및 스테인리스 차폐 케이블 배관 시공


        5. **다음 단계 액션 플랜 (Next Step)**
            - **D+1주:** 1호기 설비 스핀들 외벽 센서 장착 위치 선정 및 배선 간섭 실사
            - **D+3주:** IoT 게이트웨이 설치 및 정상 가공 진동 베이스라인 데이터 2주간 연속 수집
            - **D+6주:** 이상 감지 1차 모델 학습 및 임계값 튜닝

    <br>

    - **[예시 2] 화학/소재 업종 (배합 & 수율 최적화)**

        <div class="info-table">
            <table>
                <thead><th colspan="4" width="1100px">과제 기본 정보</th></thead>
                <tbody>
                    <tr>
                        <td class="td-rowheader" width="120px">과제명</td>
                        <td class="td-left" colspan="3">2차전지 분리막용 고기능성 코팅액 배합 파라미터 최적화 AI</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader" width="120px">추진 부서</td>
                        <td class="td-left">배합공정팀 / R&D연구소</td>
                        <td class="td-rowheader" width="120px">사업재편 목표</td>
                        <td class="td-left">[v] 신사업 라인 구축    [ ] 기존 라인 고도화    [ ] 규제/인증 대응</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader">예상 추진기간</td>
                        <td class="td-left">PoC (3개월) 🡪 공정 제어 연동 (3개월)</td>
                        <td class="td-rowheader">적용 AI 기술</td>
                        <td class="td-left">[ ] Vision   [v] Tabular/시계열 ML   [ ] LLM/RAG   [ ] Agent</td>
                    </tr>
                </tbody>
            </table>
        </div>


        1. **배경 및 문제 정의 (Why)**
            - **현장의 고질적 문제 (As-Is):**
                - 범용 산업용 필름에서 2차전지용 세라믹 코팅 분리막으로 신사업 진출
                - 코팅액 교반(혼합) 시 외부 온·습도 변화에 따라 점도 편차가 발생해 도포 두께 불량 및 뭉침 결함 발생 (초기 불량률 11.8%)

            - **문제의 근본 원인:**
                - 배합 시간, 교반기 RPM, 투입 온도 등 10여 가지 공정 변수의 상관관계를 제어하지 못해 연구원의 수작업 샘플링 테스트(회당 40분 소요)에 의존


        2. **AI 적용 방안 및 목표 (What & How)**
            - **AI 도입 후 모습 (To-Be):**
                - 반응기 내부 점도·온도 센서와 외기 온·습도 데이터를 실시간 수집
                - 회귀 머신러닝(XGBoost/Random Forest) 모델이 목표 점도를 달성하기 위한 최적의 '교반 속도 및 가열 프로파일'을 실시간 제시(가이드 피드포워드)

            - **필요 데이터 현황:**
                - 기존 보유 데이터: 연구소 파일럿 테스트 배합 데이터 150건
                - 신규 수집 데이터: 양산 배합조 실시간 온도/압력/교반 모터 토크 데이터


        3. **정량적 기대효과 (Impact / ROI)**
            - **정량 지표:**
            - 코팅액 점도 불량률: 기존 11.8% 🡪 **2.5% 이하 (78.8% 감축)**
            - 신사업 수율 안정화 기간: 예상 6개월 🡪 **2개월로 4개월 단축** (원자재 스크랩 비용 연 2.4억 원 절감)

            - **정성 지표:**
            - 연구원의 현장 샘플 채취 공수 절감으로 신소재 후속 배합 개발에 집중 가능


        4. **리스크 및 장애요인 극복 방안 (Risk & Solution)**
            - **발생 가능 리스크:** 화학 원료 로트(Lot)별 미세 물성 차이로 인한 AI 예측 오차 발생
            - **대응 방안:** 원자재 입고 검사 시 측정된 기본 물성값(수분율, 입도)을 AI 입력 변수로 함께 포함


        5. **다음 단계 액션 플랜 (Next Step)**
            - **D+1주:** 배합조 PLC에서 교반 토크 및 온도 시계열 데이터 추출 포트 개방
            - **D+4주:** 파일럿 데이터 기반 1차 예측 모델 학습 및 정확도 검증 ($R^2 > 0.85$ 목표)
            - **D+8주:** 양산 1호 배합조 현장 디스플레이에 추천값 파일럿 표출 시작

    <br>

    - **[예시 3] 전자/반도체 조립 업종 (외관 품질 검사)**

        <div class="info-table">
            <table>
                <thead><th colspan="4" width="1100px">과제 기본 정보</th></thead>
                <tbody>
                    <tr>
                        <td class="td-rowheader" width="120px">과제명</td>
                        <td class="td-left" colspan="3">전장용 차량 카메라 모듈 와이어 본딩 및 하우징 딥러닝 비전 검사</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader" width="120px">추진 부서</td>
                        <td class="td-left">품질보증팀 / 생산기술팀</td>
                        <td class="td-rowheader" width="120px">사업재편 목표</td>
                        <td class="td-left">[v] 신사업 라인 구축    [v] 기존 라인 고도화    [ ] 규제/인증 대응</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader">예상 추진기간</td>
                        <td class="td-left">PoC (2개월) 🡪 인라인 인스펙터 구축 (2개월)</td>
                        <td class="td-rowheader">적용 AI 기술</td>
                        <td class="td-left">[v] Vision   [ ] Tabular/시계열 ML   [ ] LLM/RAG   [ ] Agent</td>
                    </tr>
                </tbody>
            </table>
        </div>


        1. **배경 및 문제 정의 (Why)**
            - **현장의 고질적 문제 (As-Is):**
                - 모바일용 카메라 조립에서 자율주행 ADAS용 전장 카메라 모듈로 생산 라인 전환
                - 금속 하우징 미세 스크래치 및 25um 두께 금선(Wire) 접합 불량을 육안으로 전수 검사하느라 검사원 6명이 투입되나, 피로도에 따른 오검수·미검출 누락 발생

            - **문제의 근본 원인:**
                - 금속 하우징의 빛 반사와 와이어 굴곡 특성 때문에 기존 룰 기반 비전 장비는 정상품을 불량으로 잡는 과검률(False Positive)이 18%에 달해 무용지물


        2. **AI 적용 방안 및 목표 (What & How)**
            - **AI 도입 후 모습 (To-Be):**
                - 돔 조명과 편광 필터를 장착한 1,200만 화소 고속 카메라 검사 부스를 컨베이어 말단에 설치
                - 딥러닝 세그멘테이션(Segmentation) 모델을 인라인 엣지 PC에 탑재해 택트타임(개당 1.2초) 내에 미세 결함을 99% 이상 실시간 판정

            - **필요 데이터 현황:**
                - 기존 보유 데이터: 과거 육안 검사원이 촬영한 불량 사진 400여 장
                - 신규 수집 데이터: 다양한 각도/조명 조건에서의 양품 이미지 3,000장 및 불량 유형별 이미지 증강(Data Augmentation)


        3. **정량적 기대효과 (Impact / ROI)**
            - **정량 지표:**
                - 육안 검사 과검률: 기존 18.0% 🡪 **1.5% 이하로 급감**
                - 검사 공정 인건비 절감: 검사 인력 6명 중 4명을 출하 검사 및 패키징 공정으로 전환 배치 (**연간 약 1.8억 원 공수 절감**)

            - **정성 지표:**
                - 자율주행 부품 전수 검사 로그(이미지) 100% 디지털 아카이빙으로 글로벌 완성차 품질 감사 대응


        4. **리스크 및 장애요인 극복 방안 (Risk & Solution)**
            - **발생 가능 리스크:** 조립 라인 주변 진동으로 인한 렌즈 초점 흔들림(Blur) 및 광학계 미세 오차
            - **대응 방안:** 검사 챔버 하부에 에어 스프링 방진 마운트 시공 및 고속 전자 셔터 속도 확보


        5. **다음 단계 액션 플랜 (Next Step)**
            - **D+1주:** 불량 유형(와이어 단선, 들뜸, 하우징 찍힘) 표준 결함집(Defect Catalog) 정의
            - **D+3주:** 광학 벤더사 미팅 및 최적 조명/렌즈 조합 샘플 테스트(PoC 이미지 500장 취득)
            - **D+6주:** 인라인 검사기 프로토타입 설치 및 라벨링 데이터 기반 1차 판정 정확도 평가