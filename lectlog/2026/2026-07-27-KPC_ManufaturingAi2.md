---
layout: page
title:  "제조 산업과 AI 활용 과정 2차(21H)"
date:   2026-04-02 10:50:00 +0900
permalink: /lectlog/2026-07-27-KPC_ManufaturingAi2
categories: lectlog
---
* toc
{:toc .large-only .toc-sticky:true}


<h2>● 강의 개요</h2>

> - **교육분야:** 스마트팩토리 + AI
> - **교육목적**
>   - **현장의 "숙련도"를 "데이터"로 체득하고 인사이트까지 얻는 과정을 이해한다.**
>   - 스마트 팩토리와 제조 데이터의 구조적 이해
>   - 이미지 데이터를 활용한 비전 AI 검사 기술 습득
>   - 노코드 AI를 이용한 공정 관리 및 불량 검출 모델 구현
>   - 설비 예지 정비(PdM) 및 이상 탐지 체계 구축 역량 강화
>   - 가상 공정 시뮬레이션 및 유연 생산 시스템의 이해
> - **교육대상:** 해당 분야 관심있는 재직자 혹은 구직자
>   - 제조 현장에 AI 도입을 검토 중인 현업 실무자
>   - 데이터 기반의 공정 최적화를 원하는 생산 관리자
>   - 설비 유지보수 및 예지 정비 시스템 구축 담당자
>   - 스마트 팩토리 전환 프로젝트 리더 및 기획자
>   - 제조 분야 AI 알고리즘 적용에 관심 있는 엔지니어
> - **강의기간:** 2026.07.27~07.29 (09:00 ~ 17:00) (21H)
> - **운영방식:** 오프라인 대면 강의
{: .common-quote}


<h2>● 강의 내용</h2>

- **1일차**
    - 스마트 제조와 AI 개요 이해
        - [스마트 팩토리 개념 및 제조 패러다임 변화](/materials/S06-01-01-01_01-SmartFactoryOverview)
        - [제조 산업 내 AI 활용 사례 및 도입 효과 분석](/materials/S06-01-01-02_01-ParadigmShift)
        - [현장 적용 관점에서의 AI 도입 포인트 정리](/materials/S06-01-01-03_01-KeyPointsForAiAdoption)

    - 제조 데이터 수집, 정제 및 분석 기초
        - [제조 데이터 유형 및 특성](/materials/S06-04-01-01_01-ManufacturingDataFeature)
        - [제조 데이터 구조 이해](/materials/S06-04-01-02_01-ManufacturingDataOverview)
        - [제조 데이터 수집 구조](/materials/S06-04-01-03_01-ManufacturingDataCollectAndAnalyze)
            - PLC-센서-클라우드 기반 데이터 수집 구조
        - [데이터 전처리 및 정제 방법 (결측치, 이상치 처리)](/materials/S02-02-02-01_01-DataPreprocess)
        - [데이터 통합 및 품질 관리 방법](/materials/S02-02-02-03_01-DataIntegrationQuality)
        - [EDA(탐색적 데이터 분석) 기반 패턴 분석](/materials/S02-02-03-01_01-DataAnalysis)
        - [(실습) 공정 데이터 기반 인사이트 도출](/materials/S02-02-03-03_01-DataAnalysisPractice)

- **2일차**
    - 비전 AI 기반 품질 검사 (3H)
        - [비전 AI 기반 검사 시스템 구조](/materials/S06-04-04-01_01-VisionAiBasedInspectionSystem)
        - [이미지 데이터 수집 및 라벨링 프로세스](/materials/S06-04-04-02_01-ImageDataCollectionLabeling)
        - [불량 검출 모델 개념 및 적용 방식](/materials/S06-04-04-03_01-DefectDetectionModels)
        - [품질 검사 자동화 사례 분석](/materials/S06-04-04-04_01-QualityInspectionAutomationCases)
        - [(실습) 불량 검출 결과 해석 및 개선 포인트 도출](/materials/S06-04-04-05_01-DefectDetectionResultsPractice)

    - 공정 최적화를 위한 AI 모델 적용 (4H)
        - [노코드/로우코드 기반 AI 활용 구조 이해](/materials/S06-04-05-01_01-NoCodeLowCodeBasedAi)
        - [공정 데이터 기반 예측 모델 개념](/materials/S06-04-05-02_01-DataBasedPredictionModels)
        - [예측 결과를 공정 개선에 연결하는 방법](/materials/S06-04-05-03_01-ConnectPredictionResultsToProcessImprovement)
        - [AI 모델 운영 흐름(MLOps) 기초 이해](/materials/S06-04-05-04_01-AiModelOperationalFlow)
        - [(실습) 공정 개선 시나리오 설계](/materials/S06-04-05-05_01-DesigningProcessImprovementScenarios)
        - [PyTorch 기반 DNN 예제](https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/ai/notebooks/S03-02-03-02_01-OpenCv.ipynb){: target="_blank"}

- **3일차**
    - 설비 데이터 분석과 스마트 운영 (3H)
        - [설비 센서 데이터 기반 이상 탐지](/materials/S06-04-06-01_01-AnomalyDetectionEquipmentSensorData)
        - [예지보전(Predicted Maintenance) 개념](/materials/S06-04-06-02_01-PredictedMaintenance)
        - [KPI 기반 공정 모니터링 및 대시보드 설계](/materials/S06-04-06-03_01-KpiBasedProcessMonitoring)
        - [실시간 데이터 기반 의사결정 구조](/materials/S06-04-06-04_01-RealTimeDecisionMaking)
        - [(실습) 사례 분석: 현장 적용 운영 전략](/materials/S06-04-06-05_01-FieldApplicationOperationalStrategies)

    - 스마트 공장 확장과 미래 제조 전략 (4H)
        - [디지털 트윈 및 가상 공장 개념과 활용](/materials/S06-04-07-01_01-DigitalTwinVirtualFactory)
        - [물류 자동화 및 제조 로봇(AVR, 협동로봇) 이해](/materials/S06-04-07-02_01-LogisticsAutomationManufacturingRobots)
        - [유연 생산 시스템(FMS) 및 스마트 생산 전략](/materials/S06-04-07-03_01-FlexibleManufacturingSystems)
        - [제조 AI 도입 로드맵 및 조직 변화](/materials/S06-04-07-04_01-ManufacturingAiAdoptionRoadmap)
        - [미래 제조 트렌드 및 산업 전망](/materials/S06-04-07-05_01-FutureManufacturingTrends)

<br>

> - 참고
>   - 실습은 주로 Orange 등의 무료도구를 사용함
>   - 실습 데이터는 가상의 데이터를 사용함
>   - 실습의 목적은 실질적인 기술을 익히는 것이 아니라 관리자와 실무자로서 데이터의 흐름을 이해하고 의사결정에 활용하는 법을 중심으로 함