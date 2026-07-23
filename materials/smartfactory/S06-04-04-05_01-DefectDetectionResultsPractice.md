---
layout: page
title:  "[실습] 불량 검출 결과 해석 및 개선 포인트 도출"
date:   2026-07-22 15:00:00 +0900
permalink: /materials/S06-04-04-05_01-DefectDetectionResultsPractice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - AI가 도출한 평가 지표(Confusion Matrix)와 시각화 자료(Heatmap)를 보고
> - 현장 관리자가 어떤 의사결정(Threshold 조율, 조명 변경, 데이터 재학습)을 내릴 것인가?
{: .common-quote}


## 1. 실습 목표

- AI 비전 검사 소프트웨어가 출력한 평가 지표와 Heatmap(Grad-CAM)을 읽어내고
- **과검(Over-kill)과 미검(Under-kill)을 줄이기 위한 현장 조치점 및 AI 재학습 전략** 도출


## 2. 실습 환경 및 제공 Dataset 시나리오

- **실제 비전 AI 솔루션 화면을 모사한 2가지 분석 리포트(Dashboard) 데이터**

    - 데이터 1: 혼동 행렬(Confusion Matrix) 및 임계값(Threshold) 변동 데이터
        - AI 모델이 $1,000$장의 테스트 이미지(양품 $900$장, 불량 $100$장)를 검사한 결과 데이터표
        - **용어의 현장 매핑:**
            - **미검 (False Negative, FN):**
                - 실제 불량(NG)을 AI가 양품(OK)으로 놓침 🡪 **[치명적]** 🡪 고객사 유출, 클레임, 신뢰도 추락
            - **과검 (False Positive, FP):**
                - 실제 양품(OK)을 AI가 불량(NG)으로 오판함 🡪 **[비용 발생]** 🡪 버리지 않고 작업자가 수동 재검사(Re-inspection)해야 하므로 공수 증가


## 3. 실습 1: 지표 해석 및 최적 임계값 조율

> - 임계값(Threshold): AI가 '이 제품은 불량일 확률이 몇 % 이상일 때 NG로 팝업을 띄울 것인가?'를 결정하는 값
{: .common-quote}

- **실습용 데이터표**

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">구분</th>
        <th style="width: 250px;">임계값 0.7 (높음)<br>(까다로운 불량 기준)</th>
        <th style="width: 250px;">임계값 0.5 (기본)<br>(표준 기준)</th>
        <th style="width: 250px;">임계값 0.3 (낮음)<br>(보수적/안전 기준)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">미검 (FN, 불량 유출)</td>
            <td>12건 (불량 12개 놓침)</td>
            <td>3건 (불량 3개 놓침)</td>
            <td>0건 (불량 100% 잡아냄)</td>
        </tr>
        <tr>
            <td class="td-rowheader">과검 (FP, 양품 오인)</td>
            <td>5건 (재검사 적음)</td>
            <td>25건 (재검사 소량)</td>
            <td>85건 (양품 85개를 NG 처리)</td>
        </tr>
        <tr>
            <td class="td-rowheader">정밀도 (Precision)</td>
            <td>88.0%</td>
            <td>79.5%</td>
            <td>54.1%</td>
        </tr>
        <tr>
            <td class="td-rowheader">재현율 (Recall)</td>
            <td>88.0%</td>
            <td>97.0%</td>
            <td>100.0%</td>
        </tr>
    </tbody>
</table>
</div>

- **실습 과제 및 도출 답안**
    - **질문:**
        - 우리 공장이 [자동차 안전 부품]을 만드는 공장이라면 Threshold를 몇으로 설정해야 하는가?
        - 반대로 [저가 일회용 용기]를 만드는 공장이라면?

    - **현장 가이드라인 및 해설:**
        - **자동차/반도체 핵심 부품 공장 ➔ `Threshold = 0.3` 선택:**
            - 미검(고객사 유출)은 회사 문을 닫게 만들 만큼 치명적
            - 과검이 85건 발생하여 작업자가 손으로 재검사하는 한이 있더라도
            - 미검 0% (재현율 100%)를 확보하는 0.3을 선택하고,
            - 과검은 후속 조치로 줄여나가야 함

        - **저가 대량 생산 공장 ➔ `Threshold = 0.5 ~ 0.6` 선택:**
            - 단가가 매우 낮고 재검사 인력이 부족한 경우,
            - 약간의 미검 위험을 감수하더라도
            - 과검 발생으로 인한 수동 재검사 병목을 막는 것이 경제적


## 4. 실습 2: XAI의 Grad-CAM Heatmap 해석

> - AI가 단순히 'NG'라고 판정했을 때, 현장 작업자는 '왜 NG냐?'라고 반문함
> - **Grad-CAM(Heatmap): AI가 이미지의 어느 영역(픽셀)을 집중해서 보고 NG 판단을 내렸는지 붉은색 열지도**로 시각화해 주는 XAI 기술
{: .common-quote}

<br>

- **실습용 Heatmap 이미지 케이스 분석 (3가지 현장 시나리오)**

    - **케이스 A: 조명/그림자 노이즈 오판**
        - **AI 출력 이미지:**
            - 제품의 가장자리 꺾이는 부위에 **붉은색 Heatmap**이 집중되며 `NG (확률 88%)` 출력
        - **실제 현장 상태:**
            - 제품 표면은 깨끗함
            - 단지 제품 테두리에 '설비 내부 구조물의 그림자'가 짙게 드리워져 있음
        - **도출해야 할 개선 조치점:**
            - **[광학계 개선]**
                - 조명의 각도를 변경하거나, 측면 백라이트(Backlight)를 추가하여 **테두리 그림자 제거**

    - **케이스 B: 비해해성 오염(기름때) 과검**
        - **AI 출력 이미지:**
            - 제품 중앙의 원형 얼룩 영역에 **붉은색 Heatmap**이 켜지며 `NG (확률 92%)` 출력
        - **실제 현장 상태:**
            - 가공 과정에서 묻은 세척 가능한 단순 방청유(기름방울)임
            - 실제 크랙이나 스크래치가 아님
        - **도출해야 할 개선 조치점:**
            - **[데이터 라벨링 개선]**
                - 해당 기름방울 이미지를 수집하여 `양품(OK) - 방청유 허용` 클래스로 재라벨링 후
                - **AI 모델 재학습(Re-training)** 진행

    - **케이스 C: 정상적인 미세 스크래치 감지 (성공)**
        - **AI 출력 이미지:**
            - 제품 표면의 0.2mm 미세 수직선 부위에 핀포인트로 **붉은색 Heatmap** 형성되며 `NG (확률 99%)` 출력
        - **실제 현장 상태:**
            - 금형 마모로 인해 발생한 실제 스크래치 불량 맞음
        - **도출해야 할 개선 조치점:**
            - **[설비 제어 Interlock]**
            - 정상 작동 확인
            - 해당 Lot 배출 에어 블로워 작동 및 금형 정비 알람 발송


> - **[보고서 양식] 비전 AI 검사 결과 분석 및 개선 대책**
{: .common-quote}

1. **설정 임계값(Threshold) 및 선정 사유:**
    - 선택한 Threshold: `[ 0.3 ]`
    - 사유: 
        -미검률을 0%로 낮추어 고객사 클레임을 방지하기 위함
        - 이로 인해 증가한 과검(85건)은 하단 조치를 통해 절반 이하로 감축 목표

2. **Heatmap 분석을 통한 현장 원인 및 개선점 (Action Plan):**

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">분석 케이스</th>
        <th style="width: 360px;">AI 오판 원인 (Heatmap 분석)</th>
        <th style="width: 360px;">현장 개선 조치 (Action Plan)</th>
        <th style="width: 150px;">담당 부서</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">사례 1</td>
            <td>테두리 그림자를 불량으로 오인</td>
            <td>암실 내부 링 조명 각도 45˚ 🡪 60˚ 재조정</td>
            <td>설비보전팀</td>
        </tr>
        <tr>
            <td class="td-rowheader">사례 2</td>
            <td>방청유 기름방울을 스크래치로 오인</td>
            <td>방청유 적용 이미지 50장 추가 수집 후 AI 모델 재학습</td>
            <td>품질보증팀/AI팀</td>
        </tr>
        <tr>
            <td class="td-rowheader">사례 3</td>
            <td>정상적인 미세 스크래치 선명히 감지</td>
            <td>현행 유지 및 해당 불량 패턴 MLOps 데이터베이스 등록</td>
            <td>생산팀</td>
        </tr>
    </tbody>
</table>
</div>

