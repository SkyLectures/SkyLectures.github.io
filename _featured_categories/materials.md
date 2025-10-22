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

### 3.4 강화학습(RL)
- [강화학습 개요](/materials/S03-07-01-01_01-RlOverview)
- [강화학습 모델](/materials/S03-07-02-01_01-RlModel)
- [강화학습 실습](/materials/S03-07-03-01_01-RlPractice)

### 3.5 자연어 처리(NLP) 

### 3.6 대형 언어 모델(LLM) + 생성형 AI(GenAI)
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

### 3.7 AI 윤리
- [AI 윤리 개요](/materials/S03-09-01-01_01_AiEthicsOverview)
- [AI 윤리와 전통 윤리](/materials/S03-09-02-01_01_AiEthicsTraditionalEthics)
- [인공지능의 사회적 문제](/materials/S03-09-03-01_01_AiSocialProblems)
- [인공지능과 규제](/materials/S03-09-04-01_01_AiRegulations)

---

## [4. Cloud Platform](/materials/04_CloudPlatform)
- 클라우드 시스템 개요
- 클라우드 플랫폼
    - AWS / Azure / Google Cloud Platform

---

## [5. Single Board Computer(SBC)](/materials/05_SBC)
- 아두이노(Arduino)
- 라즈베리파이(Raspberry Pi)
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


[&nbsp;](/materials/99_Test)
