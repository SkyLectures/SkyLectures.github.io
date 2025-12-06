---
layout: page
title:  "모빌리티 AI 산업 프로젝트 과정(80H)"
date:   2025-11-03 13:00:00 +0900
permalink: /lectlog/2025-11-17-IPA_Mobility_AI_Project
categories: lectlog
---
* toc
{:toc .large-only .toc-sticky:true}


<h2>● 강의 개요</h2>

> - **강의주제:** 모빌리티 AI 산업 프로젝트 과정
> - **강의기간:** 2025.11.17 ~ 2025.12.12
> - **강의시간:** 18:30 ~ 22:30
> - **강의시수:** 80H
> - **교재경로:** https://skylectures.github.io/lectlog/2025-11-17-IPA_Mobility_AI_Project

> - **학습목표**
>   - 모빌리티 산업에서 사용되는 AI기술에 대하여 이해하고 이를 실제 개발 및 적용하는 프로젝트를 통해 실제 활용가능한 역량을 확보
> - **SW융합요소**
>   - 프로그래밍 언어: Python
>   - 데이터 처리: Pandas, NumPy, 데이터 전처리
>   - 시각화 도구: Matplotlib / OpenCV
>   - 딥러닝 프레임워크 및 모델: TensorFlow / PyTorch, CNN
>   - 컴퓨터 비전: OpenCV, 객체 탐지(Object Detection) 등
>   - 기기 제어: Raspberry Pi
>   - 프로젝트 응용: 모빌리티 데이터 분석 및 AI 모델 개발
{: .common-quote}


<h2>● 강의 내용</h2>

### 1. 모빌리티 AI의 이해
- [모빌리티 AI 개요](/materials/S10-01-01-01_01-MobilityAiOverview)
- [AI 개요](/materials/S03-01-01-01_01-AiOverview)
- [딥러닝 개요](/materials/S03-03-01-01_01-DeepLearningOverview)

### 2. 파이썬 기초

#### 2.1 파이썬 개요
- [파이썬 소개](/materials/S01-01-01-01_01-PythonOverview)
- [개발 환경 설정](/materials/S01-01-02-01_01-VirtualEnvironment)
- [파이썬 기본 문법](/materials/S01-01-03-01_01-PythonBasic)

#### 2.2 파이썬 라이브러리
- [파이썬 표준 라이브러리](/materials/S01-01-04-01_01-PythonLibrary)
- [NumPy를 활용한 데이터 연산](/materials/S01-01-04-02_01-Numpy)
- [Pandas 데이터 처리, 데이터 프레임 구조 이해](/materials/S01-01-04-03_01-Pandas)
- [Matplotlib을 이용한 데이터 시각화 기초](/materials/S01-01-04-04_01-Matplotlib)    

### 3. 자율주행 기술과 영상 처리의 이해

#### 3.1 자율주행 기술의 원리 및 요소
- [자율주행 레벨(Level)의 이해](/materials/S10-01-02-01_01-AutonomousDrivingLevels)
- [인지-판단-제어 프로세스 소개](/materials/S10-01-02-01_02-CognitionJudgmentControlProcess)
- [자율주행 센서의 종류와 역할](/materials/S10-01-02-01_03-AutonomousDrivingSensors)

#### 3.2 영상 처리 및 컴퓨터 비전 기초
- [디지털 이미지의 구조](/materials/S03-02-03-01_01-DigitalImageStructure)
- [OpenCV 기초](/materials/S03-02-03-02_01-OpenCv)
- [영상 전처리 및 필터링](/materials/S03-02-03-03_01-ImagePreprocessingFiltering)
- [엣지 검출 및 기하학적 변환](/materials/S03-02-03-04_01-EdgeDetectionTransform)
- [특징점 검출 및 추적 기초](/materials/S03-02-03-05_01-FeatureDtectionTracking)
- [CNN 모델의 구조와 동작 원리](/materials/S03-03-02-02_01-CnnModel)
                
### 4. 자율주행 시스템 구축

#### 4.1 자율주행 키트 조립 및 환경 설정
- [자율주행 키트 조립](/materials/S10-01-02-02_01-AssemblingKit)
- [라즈베리파이 OS 설치 및 개발 환경 설정](/materials/S05-03-01-01_01-RaspberryPiSetup)
- [라즈베리파이 개요](/materials/S05-03-02-01_01-RaspberryPiOverview)
- [라즈베리파이 제어 기초](/materials/S10-01-02-03_01-RaspberryPiControl)
- [자동차 무선조종 기능 구현](/materials/S10-01-02-03_02-RaspberryPiRemoteControl)
- [카메라를 활용한 자율주행 자동차 구현(OpenCV)](/materials/S10-01-02-03_03-AutoDrivingUsingCameraOpenCv)
- [라즈베리파이-카메라 실시간 영상 처리](/materials/S10-01-03-06_01-RealtimeImageProcessing)
            
#### 4.2 딥러닝 기반 자율주행 인지
- [딥러닝 기반 객체 탐지](/materials/S10-01-04-04_01-DeepLearningBasedObjectDetection)
- [딥러닝 기반 차선 인식](/materials/S10-01-04-05_01-DeepLearningBasedLaneRecognition)
- [도로 표지판 및 신호등 인식](/materials/S10-01-04-06_01-RoadSignTrafficLightRecognition)
- [자율주행 인지 모델 구현](/materials/S10-01-04-07_01-AutonomousDrivingCognitiveModelImplementation)
                
#### 4.3 자율주행 판단 및 제어 시스템 구축
- [인지 결과 기반 판단 전략](/materials/S10-01-05-01_01-DecisionStrategy)
- [자율주행 제어의 기초 및 PID 제어](/materials/S10-01-05-02_01-AutonomousDrivingControlPidControl)
- [라즈베리파이 기반 차량 제어](/materials/S10-01-05-03_01-RaspberryPiBasedVehicleControl)

### 5. 실전 프로젝트
- [영상 인식을 통한 자율주행 제어 구현](/materials/S10-01-05-04_01-AutonomousDrivingControlImplementation)
