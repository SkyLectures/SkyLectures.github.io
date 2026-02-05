---
layout: page
title: 강의 자료 (Lecture Materials)
slug: materials
description: >
  강의에 사용된 강의자료들을 Markdown 형식으로 제공합니다. 지속적으로 갱신/수정됩니다.
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Python

### 1.1 파이썬 문법 / 기본 라이브러리
- **파이썬 기본 문법**
    - [파이썬 개요](/materials/S01-01-01-01_01-PythonOverview)
    - [파이썬 개발 환경 설정](/materials/S01-01-02-01_01-VirtualEnvironment)
    - [파이썬 기본 문법](/materials/S01-01-03-01_01-PythonBasic)

- **파이썬 문법 상세**
    - [값의 처리와 변수](/materials/S01-02-01-01_01-Variables)
    - [연산자](/materials/S01-02-02-01_01-Operatros)
    - [자료형](/materials/S01-02-03-01_01-DataTypes)
    - [Sequence 자료형](/materials/S01-02-04-01_01-SequenceDataTypes)
    - [Mapping & Set 자료형](/materials/S01-02-05-01_01-MappingSetDataTypes)
    - [제어문](/materials/S01-02-06-01_01-ControlStatements)
    - [예외처리](/materials/S01-02-07-01_01-Exceptions)
    - [함수](/materials/S01-02-08-01_01-Functions)
    - [내장함수](/materials/S01-02-09-01_01-BuiltInFunctions)
    - [클래스](/materials/S01-02-10-01_01-Classes)
    - [모듈과 패키지](/materials/S01-02-11-01_01-Modules)
    - [입출력](/materials/S01-02-12-01_01-InputOutput)
    - [파일처리](/materials/S01-02-13-01_01-FileHandling)
    - [정규표현식](/materials/S01-02-14-01_01-RegularExpressions)
    - [Under Bar의 이해](/materials/S01-02-15-01_01-UnderBar)

- **파이썬 라이브러리**
    - [파이썬 표준 라이브러리](/materials/S01-01-04-01_01-PythonLibrary)
    - [Numpy](/materials/S01-01-04-02_01-Numpy)
    - [Pandas](/materials/S01-01-04-03_01-Pandas)
    - [Matplotlib](/materials/S01-01-04-04_01-Matplotlib)
    - [Seaborn](/materials/S01-01-04-05_01-Seaborn)

- **고성능 파이썬**
    - 준비중...

### 1.2 파이썬 응용

- **웹 프레임워크 / 웹 서비스 개발**
    - **웹 서비스 기초**
        - [웹서버 개발의 이해](/materials/S01-04-01-01_01-WebServerOverview)
        - [Simple 웹서버 개발 예제](/materials/S01-04-01-02_01-SimpleWebServer)
        - [REST API 개요](/materials/S01-04-01-03_01-RestApiOverview)

    - **Django 웹 프레임워크**
        - [Django 웹프레임워크 개요](/materials/S01-04-02-01_01-DjangoOverview)
        - [Django 기반 서비스 기본 흐름](/materials/S01-04-02-01_02-DjangoBasedServiceProcess)
        - [사진 목록 보기](/materials/S01-04-02-02_01-PictureList)
        - [Todo List 웹서비스](/materials/S01-04-02-02_02-TodoList)

    - **Django REST Framework(DRF)**
        - [Django REST Framework(DRF) 개요](/materials/S01-04-02-03_01-DrfOverview)
        - [도서정보 API](/materials/S01-04-02-03_02-BookInfoApi)
        - [Todo List API](/materials/S01-04-02-03_03-TodoListApi)
        - [게시판 사용자 관리](/materials/S01-04-02-04_01-DrfBbsUsers)
        - [게시판 글 관리](/materials/S01-04-02-04_02-DrfBbsPosts)

    - **DRF 활용 실습:** Pystagram 만들기 (코드출처: 이한영의 Django 입문)
        - [환경구축,기본 정보 설정](/materials/S01-04-02-05_01-DrfPystagramEnvironment)
        - [로그인,로그아웃 기능 구현](/materials/S01-04-02-05_02-DrfPystagramBasicInfo)
        - [회원가입 기능 구현](/materials/S01-04-02-05_03-DrfPystagramLoginLogout)
        - [글 관리 기능 구현](/materials/S01-04-02-05_04-DrfPystagramMemberRegister)
        - [부가 기능 구현](/materials/S01-04-02-05_05-DrfPystagramPost)
        - [글 상세 관리 기능 구현](/materials/S01-04-02-05_06-DrfPystagramAdditionalFunctions)

    - **Flask 웹 프레임워크**
        - [Flask 웹프레임워크 개요](/materials/S01-04-03-01_01-FlaskOverview)
        - [Flask 설치 및 환경설정](/materials/S01-04-03-01_02-FlaskSetting)
        - [Flask 기반 웹서비스 기본 흐름](/materials/S01-04-03-02_01-FlaskBasedServiceProcess)
        - [Flask의 라우팅 및 URL 설계](/materials/S01-04-03-02_02-FlaskRoutingUrlDesign)
        - [템플릿 엔진(Jinja2) 활용](/materials/S01-04-03-02_03-TemplateEngineJinja2)
        - [정적 파일 관리](/materials/S01-04-03-02_04-StaticFiles)
        - [동적 콘텐츠를 위한 Jinja2 활용](/materials/S01-04-03-02_05-DynamicContentsJinja2)
        - [사용자 입력처리(Form 및 API)](/materials/S01-04-03-02_06-UserInputs)
        - [GET / POST 메서드 활용](/materials/S01-04-03-02_07-GetPostMethods)    

    - **FastAPI 웹 프레임워크 (준비 중)**

- **파이썬 GUI**
    - [그래픽 유저 인터페이스(GUI)](/materials/S01-05-01-01_01-GuiOverview)
    - [GUI 예제 실습](/materials/S01-05-02-01_01-GuiExamples)

- **파이썬 게임 (준비 중)**

---

## [2. Data Science](/materials/02_DataScience)
- 데이터 과학 개요
- 데이터 분석 및 모델링
- 데이터 분석 결과 해석
- 데이터 분석 모델 평가
- 데이터베이스 기초
- 데이터베이스 활용
- 벡터 데이터베이스

---

## [3. ArtificiaL Intelligence(AI)](/materials/03_AI)
### 3.1 AI 개요
- [AI 개요](/materials/S03-01-01-01_01-AiOverview)
- [AI 시스템의 개발 공정](/materials/S03-01-02-01_01-AiDevelopmentProcess)

### 3.2 머신러닝(ML)
- 머신러닝 개요
- 머신러닝 모델
    - 기본모델
        - [선형회귀(Linear Regression)](/materials/S03-02-02-01_01-LinearRegression)
        - [로지스틱 회귀(Logistic Regression)](/materials/S03-02-02-01_02-LogisticRegression)
        - [K-최근접이웃(K-Nearest Neighbors, KNN)](/materials/S03-02-02-01_03-KNearestNeighbors)
        - [의사결정나무(Decision Tree)](/materials/S03-02-02-01_04-DecisionTree)
        - [서포트 벡터 머신(Support Vector Machine, SVM)](/materials/S03-02-02-01_05-SupportVectorMachine)

    - 심화모델(앙상블 모델)
        - [앙상블 러닝 개요(Bagging, Boosting)](/materials/S03-02-02-02_01-EnsembleLearningOverview)
        - [랜덤 포레스트(Random Forest)](/materials/S03-02-02-02_02-RandomForest)
        - [그래디언트 부스팅 머신(Gradient Boosting Machine, GBM)](/materials/S03-02-02-02_03-GradientBoostingMachine)
        - [XGBoost(Extream Gradient Boost)](/materials/S03-02-02-02_04-ExtreamGradientBoost)
        - [LightGBM(Light Gradient Boosting Machine)](/materials/S03-02-02-02_05-LightGbm)
        - [CatBoost](/materials/S03-02-02-02_06-CatBoost)

    - [인공신경망(Artificial Neural Network, ANN)](/materials/S03-02-02-02_07-ArtificialNeuralNetwork)

    - OpenCV 기반 영상처리
        - [디지털 이미지의 구조](/materials/S03-02-03-01_01-DigitalImageStructure)
        - [OpenCV 기초](/materials/S03-02-03-02_01-OpenCv)
        - [영상 전처리 및 필터링](/materials/S03-02-03-03_01-ImagePreprocessingFiltering)
        - [엣지 검출 및 기하학적 변환](/materials/S03-02-03-04_01-EdgeDetectionTransform)
        - [특징점 검출 및 추적 기초](/materials/S03-02-03-05_01-FeatureDtectionTracking)

### 3.3 딥러닝(DL) 
- [딥러닝 개요](/materials/S03-03-01-01_01-DeepLearningOverview)
- 딥러닝 모델
    - 기본 모델
        - DNN
            - [DNN 모델](/materials/S03-03-02-01_01-DnnModel)
            - [DNN 모델 실습](/materials/S03-03-02-01_02-DnnPractice)
    - 응용 모델
        - CNN
            - [CNN 모델](/materials/S03-03-02-02_01-CnnModel)
            - [CNN 모델 실습](/materials/S03-03-02-02_02-CnnPractice)
        - RNN
            - [RNN 모델](/materials/S03-03-02-03_01-RnnModel)
            - [RNN 모델 실습](/materials/S03-03-02-03_02-RnnPractice)
            - [LSTM 모델](/materials/S03-03-02-03_03-LstmModel)
            - [LSTM 모델 실습](/materials/S03-03-02-03_04-LstmPractice)
        - 통합실습
            - [DNN + CNN 모델 실습](/materials/S03-03-02-04_01-DnnCnnPractice)
    - 확장 기술
        - 전이학습(Transfer Learning)
            - [전이학습 개요](/materials/S03-03-03-01_01-TransferLearningOverview)
            - [전이학습 실습](/materials/S03-03-03-01_02-TransferLearningPractice)
        - 미세 조정(Fine-tuning)
            - [미세조정 개요](/materials/S03-03-03-02_01-FineTuningOverview)
            - [미세조정 실습](/materials/S03-03-03-02_02-FineTuningPractice)

### 3.4 자연어 처리(NLP) 

### 3.5 대형 언어 모델(LLM)
- [LLM 개요](/materials/S03-05-01-00_LLM_Overview)
- [생성형 AI 개요](/materials/S03-06-01-01_01-GenAiOverview)
- [생성형 AI 작동 원리](/materials/S03-06-02-01_01-GenAiPrinciple)
- [생성형 AI 활용사례](/materials/S03-06-03-01_01-GenAiUseCases)

- [LLM 활용(ChatGPT API / Ollama 사용 실습)](/materials/S03-05-02-00_LLM_Applications)
    - ChatGPT OpenAPI 활용
    - Ollama 활용
        - [Ollama 개요](/materials/S03-05-06-01_01-OllamaOverview)
        - [Ollama 기초(설치 및 실행)](/materials/S03-05-06-02_01-OllamaBasic)
        - [Ollama 기반 데이터 분석](/materials/S03-05-06-03_01-OllamaDataAnalysis)
- LangChain
    - [LangChain 개요](/materials/S03-05-03-01_LangChain_Overview)
    - [LLM + LangChain으로 데이터 처리하기](/materials/S03-05-03-02_01_LangChain_Pandas)
    - [LLM + LangChain으로 대화하기](/materials/S03-05-03-02_02_LangChain_Chat)
    - [LangChain 기반 LLM의 WebService 활용](/materials/S03-05-03-03_LangChain_Web_Service)

- 프롬프트 엔지니어링(Prompt Engineering)
    - [프롬프트 설계 기초와 최적화](/materials/S03-05-04-01_01-PromptDesignBasic)
    - [LLM을 활용한 데이터 생성 및 전처리](/materials/S03-05-04-02_01-DataGenerationPreprocessing)
    - [시나리오 기반 프롬프트 작성](/materials/S03-05-04-03_01-ScenarioBasedPrompts)
    - [텍스트 생성-심화](/materials/S03-05-04-04_01-AdvTextGeneration)
    - [데이터 분석 및 요약-심화](/materials/S03-05-04-05_01-AdvDataAnalysis)
    - [사용자 맞춤형 프롬프트 설계](/materials/S03-05-04-06_01-AdvCustomPrompts)
    - [GPT 응용 서비스 설계](/materials/S03-05-04-07_01-AdvGptService)

### 3.6 생성형 AI

### 3.7 강화학습(RL)
- [강화학습 개요](/materials/S03-07-01-01_01-RlOverview)
- [강화학습 모델](/materials/S03-07-02-01_01-RlModel)
- [강화학습 실습](/materials/S03-07-03-01_01-RlPractice)

### 3.8 자동 음성인식


### 3.9 AI 윤리
- [AI 윤리 개요](/materials/S03-09-01-01_01_AiEthicsOverview)
- [AI 윤리와 전통 윤리](/materials/S03-09-02-01_01_AiEthicsTraditionalEthics)
- [인공지능의 사회적 문제](/materials/S03-09-03-01_01_AiSocialProblems)
- [인공지능과 규제](/materials/S03-09-04-01_01_AiRegulations)


### 3.10 AI SW 테스트

### 3.11 Physical AI
- [Physical AI 개요](/materials/S03-11-01-01_01-PhysicalAiOverview)
- Physical AI의 작동 원리
- On Device AI와 Edge AI
- 응용
    - 모빌리티 AI
    - 자율제조와 스마트팩토리
    - 휴머노이드와 서비스 AI
- 실습

---

## [4. Cloud Platform](/materials/04_CloudPlatform)
- 클라우드 시스템 개요
- 클라우드 플랫폼
    - AWS / Azure / Google Cloud Platform

---

## [5. Single Board Computer(SBC)](/materials/05_SBC)
- SBC 개요

- 아두이노(Arduino)
- 라즈베리파이(Raspberry Pi)
    - [라즈베리파이 OS 설치 및 개발 환경 설정](/materials/S05-03-01-01_01-RaspberryPiSetup)
    - [라즈베리파이 개요](/materials/S05-03-02-01_01-RaspberryPiOverview)
    - [라즈베리파이 제어 기초](/materials/S05-03-03-01_01-RaspberryPiControlBasic)
- NVIDIA 젯슨 나노(Jetson Nano)

---

## [6. Smart Factory](/materials/06_SmartFactory)

### 6.1 스마트팩토리 개요
- 스마트팩토리 개요
    - [스마트팩토리란 무엇인가?](/materials/S06-01-01-01_01-SmartFactoryOverview)
    - [제조산업의 패러다임 변화](/materials/S06-01-01-02_01-ParadigmShiftInTheManufacturingIndustry)
- [스마트팩토리의 구축절차](/materials/S06-01-02-01_01-SmartFactoryConstructionProcedures)
- [스마트팩토리의 구성 요소 및 기술](/materials/S06-01-03-01_01-SmartFactoryComponents)
- [스마트팩토리 구축 성공사례](/materials/S06-01-04-01_01-SmartFactorySuccessStories)

### 6.2 스마트팩토리 구성 시스템
- 제조 프로세스
- MES(제조실행시스템)
- MRP(자재소요계획)
- PLM(제품 수명 주기 관리)
- ERP(전사적 자원 관리)
- SCM(공급망 관리)
- 물류 시스템

### 6.3 품질경영시스템(ISO 9001)
- [스마트팩토리와 DX 경영의 필요성](/materials/S06-03-01-01_01-DxManagement)
- [품질경영시스템(ISO 9001)의 개요](/materials/S06-03-02-01_01-Iso9001Overview)
- [스마트팩토리와 ISO 9001 연계](/materials/S06-03-03-01_01-SmartFactoryXIso9001)
- [스마트 품질경영 프로세스 설계](/materials/S06-03-04-01_01-DxManagementProcess)
- [스마트 품질경영 문서화의 중요성](/materials/S06-03-05-01_01-DxManagementDocumentation)



---

## [7. Git & Github](/materials/07_Github)
- Git & Github 개요
- Git & Github 사용 기초
- Git 고급 사용 
- Githbu Action 기반 CI/CD

---

## [8. Linux](/materials/08_Linux)
- 리눅스 개요
- 리눅스 기초 이해
- 리눅스 심화
- 리눅스 기반 개발
- 리눅스 명령어
    - 파일 및 디렉토리 관리
    - 사용자 및 권한 관리
    - 패키지 관리
    - 프로세스 관리
    - 네트워크 관리
    - 텍스트 처리 및 검색

---

## 9. [Project 기획 및 관리](/materials/09_ProjectManagement)
- 프로젝트 기획 및 설계
- 프로젝트 구현
- 프로젝트 테스트 및 디버깅
- 깃허브를 통한 코드 리뷰 및 최적화
- 프로젝트 발표 준비 및 최종 업데이트

---

## 10. Project-based

### 10.1 인공지능 기반 자율주행 자동차
- 모빌리티 AI의 이해
    - [모빌리티 AI 개요](/materials/S10-01-01-01_01-MobilityAiOverview)

- 자율주행 기술의 원리 및 요소
    - [자율주행 레벨(Level)의 이해](/materials/S10-01-02-01_01-AutonomousDrivingLevels)
    - [인지-판단-제어 프로세스 소개](/materials/S10-01-02-01_02-CognitionJudgmentControlProcess)
    - [자율주행 센서의 종류와 역할](/materials/S10-01-02-01_03-AutonomousDrivingSensors)

- 자율주행 구현
    - [자율주행 키트 조립](/materials/S10-01-02-02_01-AssemblingKit)
    - [라즈베리파이 제어 기초](/materials/S10-01-02-03_01-RaspberryPiControl)
    - [자동차 무선조종 기능 구현](/materials/S10-01-02-03_02-RaspberryPiRemoteControl)
    - [카메라를 활용한 자율주행 자동차 구현(OpenCV)](/materials/S10-01-02-03_03-AutoDrivingUsingCameraOpenCv)
    - [라즈베리파이-카메라 실시간 영상 처리](/materials/S10-01-03-06_01-RealtimeImageProcessing)
    - [딥러닝 기반 객체 탐지](S10-01-04-04_01-DeepLearningBasedObjectDetection)
    - [딥러닝 기반 차선 인식](/materials/S10-01-04-05_01-DeepLearningBasedLaneRecognition)
    - [도로 표지판 및 신호등 인식](/materials/S10-01-04-06_01-RoadSignTrafficLightRecognition)
    - [자율주행 인지 모델 구현](/materials/S10-01-04-07_01-AutonomousDrivingCognitiveModelImplementation)
    - [인지 결과 기반 판단 전략](/materials/S10-01-05-01_01-DecisionStrategy)
    - [자율주행 제어의 기초 및 PID 제어](/materials/S10-01-05-02_01-AutonomousDrivingControlPidControl)
    - [라즈베리파이 기반 차량 제어](/materials/S10-01-05-03_01-RaspberryPiBasedVehicleControl)
    - [영상 인식을 통한 자율주행 제어 구현](/materials/S10-01-05-04_01-AutonomousDrivingControlImplementation)
                
- 음성인식 기반 제어
    - [음성 명령어 기반 차량 제어](/materials/S10-01-06-03_01-VoiceCommandBasedVehicleControl)
    - [LLM 기반 차량 제어](/materials/S10-01-06-04_01-LlmBasedVehicleControl)


### 10.2 AI/LLM Agent 개발

- AI 및 LLM 기초 이해 (10H)
    - AI, 머신러닝, 딥러닝 개념 이해 (3H)
        - 인공지능(AI)의 개요: 역사와 미래
        - 머신러닝과 딥러닝의 기본 원리 및 차이점
        - 데이터의 중요성: 양질의 데이터와 모델 성능
        - 파이썬 개발 환경 설정 (아나콘다, VS Code 등)
        - 실습: 파이썬 기초 문법 복습 및 개발 환경 확인 (간단한 코드 실행)

    - 자연어 처리(NLP)와 LLM (3H)
        - 자연어 처리(NLP)란 무엇인가? 기본 개념 소개
        - 텍스트 데이터의 처리: 토큰화, 임베딩 등
        - 순환 신경망(RNN), 트랜스포머(Transformer) 아키텍처 개요 (개념 위주 설명)
        - 거대 언어 모델(LLM)의 등장 배경 및 파급 효과
        - 실습: Hugging Face transformers 라이브러리 설치 및 간단한 텍스트 처리 예제

    - 주요 LLM 소개 및 활용 맛보기 (4H)
        - ChatGPT, GPT-4, Llama, Gemini 등 주요 LLM 특징 비교
        - LLM의 응용 분야 (챗봇, 요약, 번역, 코드 생성 등)
        - API를 통한 LLM과의 상호작용 (OpenAI API 키 발급 및 기본 사용법)
        - 실습: OpenAI Playground(또는 유사 환경)에서 다양한 프롬프트로 LLM 체험, 간단한 API 호출 예제 실습

- LLM 에이전트의 핵심 구성 요소 (10H)
    - 프롬프트 엔지니어링의 기본 (4H)
        - 프롬프트 엔지니어링이란? 왜 중요한가?
        - 좋은 프롬프트 작성 원칙 (명확성, 구체성, 제약 조건 등)
        - In-context Learning (few-shot, one-shot) 기법
        - -of-Thought (CoT) 및 Tree-of-Thought (ToT) (개념 소개)
        - 실습: 다양한 프롬프트 기법을 적용하여 LLM 응답 품질 개선 실습

    - 에이전트 아키텍처 이해 (3H)
        - LLM 기반 에이전트의 개념 및 필요성
        - 기본적인 에이전트 구조: Planner, Memory, Tool Use (개념 위주)
        - RAG(Retrieval-Augmented Generation)의 개념과 중요성
        - 실습: 에이전트가 처리할 작업 정의 및 흐름도 설계 (페이퍼 프로토타이핑)

    - Tool 사용 및 외부 연동의 기초 (3H)
        - LLM이 외부 도구를 사용하는 이유
        - 간단한 도구(함수) 정의 및 LLM에게 도구 사용 지시
        - Function Calling의 기본 원리 (OpenAI Function Calling 위주 설명)
        - 실습: LLM이 날씨 API 호출 또는 계산기 함수를 사용하도록 유도하는 간단한 예제

- LLM 에이전트 개발 라이브러리 및 프로젝트 (10H)
    - LangChain 소개 및 기초 사용법 (4H)
        - LangChain이란 무엇인가? 주요 구성 요소 (Models, Prompts, Parsers)
        - Chain의 개념: 순차적 프롬프트 연결
        - LLM 모델 연동 및 환경 설정
        - 실습: LangChain을 활용하여 간단한 챗봇 구현 (질의응답 체인)

    - LangChain Tools 및 Agent 활용 (3H)
        - LangChain에서 Tool을 정의하고 사용하는 방법
        - Agent의 개념과 Agent Executor
        - ReAct(Reasoning and Acting) 프레임워크 이해
        - 실습: LangChain Agent가 정의된 Tool (예: 검색 엔진)을 활용하도록 구현

    - RAG(Retrieval-Augmented Generation)의 이해와 구현 (3H)
        - RAG의 필요성: LLM의 한계 극복 (환각 현상, 최신 정보 부족)
        - 문서 로더(Document Loader), 텍스트 스플리터(Text Splitter), 임베딩(Embeddings)
        - 벡터 스토어(Vector Store)의 개념 및 활용 (ChromaDB 또는 FAISS)
        - 실습: 간단한 로컬 문서 기반 Q&A RAG 시스템 구축

- 미니 프로젝트 및 고급 개념 (10H)
    - 미니 프로젝트: 나만의 LLM 기반 문서 분석 에이전트 개발 (5H)
        - 프로젝트 개요: 
            - 특정 문서(예: 회사 보고서, 블로그 게시물)를 업로드하면
            - 내용을 분석하고 요약, 특정 질문에 답변하며
            - 필요 시 외부 검색 도구를 사용하는 에이전트 개발
        - 구현 요소: LangChain의 RAG 기능 활용, Tool (검색 API) 연동, 프롬프트 엔지니어링
        - 실습: 프로젝트 요구사항 정의, 단계별 구현 및 디버깅

    - 고급 에이전트 개념 및 확장 (3H)
        - 에이전트의 Memory 관리: ConversationBufferMemory, ConversationSummaryMemory
        - Long-term Memory와 Short-term Memory (개념 위주)
        - LangChain 외 에이전트 프레임워크 소개 (CrewAI, AutoGen 등)
        - 실습: 기존 미니 프로젝트에 대화 기록(Memory) 추가 및 관리 기능 구현

    - 배포 및 추가 학습 가이드 (2H)
        - 개발한 에이전트를 간단하게 배포하는 방법 (Streamlit 또는 Flask 연동 개념 소개)
        - LLM 에이전트 개발 시 윤리적 고려 사항 및 보안
        - 최신 LLM 연구 동향 및 향후 학습 로드맵
        - Q&A 및 수료 소감 나누기



[&nbsp;](/materials/99_Test)
