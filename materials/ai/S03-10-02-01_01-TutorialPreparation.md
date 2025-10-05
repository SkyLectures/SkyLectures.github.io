---
layout: page
title:  "튜토리얼 환경 준비: 파이썬 환경 설정"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-10-02-01_01-TutorialPreparation
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

> **참고자료**
> - [파이썬 개요](/materials/S01-01-01-01_01-PythonOverview)
{: .expert-quote}

## 1. 실행환경 설정

### 1.1 파이썬 설치

- **설치파일 다운로드**
    - Python 공식사이트에서 각 OS에 맞는 설치파일 다운로드
    - [https://www.python.org/downloads/](https://www.python.org/downloads/)

    <div class="insert-image">
        <img src="/materials/python/images/S01-01-02-01_01-002.png">
    </div>
    <div class="insert-image">
        <img src="/materials/python/images/S01-01-02-01_01-003.png">
    </div>


- **설치파일 실행(windows 기준)**

    <div class="insert-image">
        <img src="/materials/python/images/S01-01-02-01_01-004.png">
    </div>

    - **최신 버전의 Windows의 경우**
        - ‘Windows Terminal’ / ‘PowerShell’ / ‘명령 프롬프트’에서 python 실행

        <div class="insert-image">
            <img src="/materials/python/images/S01-01-02-01_01-005.png">
        </div>

        - 파이썬 버전을 선택하려면

        <div class="insert-image">
            <img src="/materials/python/images/S01-01-02-01_01-006.png">
        </div>

### 1.2 가상환경 설정

- **가상환경을 사용하는 이유**

> - 파이썬 프로젝트는..
>     - 프로젝트의 특징에 따른 다양한 환경을 가짐
>     - 각 프로젝트마다 다른 버전의 파이썬과 모듈을 사용하는 경우가 많음
>     - 그러나 오픈소스 프로젝트의 특징에 의해 각 라이브러리, 모듈, 버전에 따라 개발자 및 개발 정책이 상이한 경우가 많음
>     - 이 때문에 많은 파이썬 라이브러리 및 모듈은 <span style="color: red;">버전별 호환성 문제가 존재함</span>
> 
>     <div class="insert-image">
>         <img src="/materials/python/images/S01-01-02-01_01-001.png" style="width: 600px;">
>     </div>
> 
>     - 이러한 이유로  <span style="color: red;">**파이썬 프로젝트는 가상환경에서의 개발을 권장함**</span>
{: .expert-quote}

- **파이썬 가상환경 직접 구축하기**

    - **가상환경을 지원하는 도구**
        - VirtualEnv: 구버전의 파이썬에서부터 많이 사용되어 온 도구
        - Venv: 파이썬 3.4 부터 기본적으로 포함된 도구 (권장)
        - Anaconda: 최근 가장 인기있는 파이썬의 배포 패키지

    - **가상환경 설정하기 (Windows 기준)**

        1. 작업용 폴더(디렉토리) 생성

            - 작업용 폴더(디렉토리)의 생성은 필수사항이 아닌 권장사항임
            - mkdir 명령을 통해 작업용 폴더(디렉토리)를 생성함 (mkdir: make directory)
                - 윈도우 탐색기에서 생성해도 무방함
                - 생성한 폴더 안에서 작업을 진행함 (권장사항)

                    <div class="insert-image">
                        <img src="/materials/python/images/S01-01-02-01_01-007.png">
                    </div>

        2. 가상환경 설정

            - python -m venv [생성하고자 하는 가상환경의 이름]

                <div class="insert-image">
                    <img src="/materials/python/images/S01-01-02-01_01-008.png">
                </div>

        3. 가상환경 활성화

            - .\Script\activate 명령으로 가상환경 활성화
                - 프롬프트의 앞에 **(가상환경이름)**이 표시되면 활성화 성공임

                    <div class="insert-image">
                        <img src="/materials/python/images/S01-01-02-01_01-009.png">
                    </div>

            - Linux/Mac의 경우 : **$ source ./bin/activate** 명령으로 활성화
                - 해당 가상환경의 폴더에 들어와 있는 경우에는 source ./bin/activate를 사용하고
                - 그렇지 않은 경우에는 source 가상환경이름/bin/activate를 사용함

            - 가상환경 활성화 오류 발생 시 해결 방안
                - Windows 환경에서 PowerShell을 사용하는 경우 다음과 같은 오류가 자주 발생함

                    <div class="insert-image">
                        <img src="/materials/python/images/S01-01-02-01_01-010.png">
                    </div>

                    - 대부분의 경우 권한부족으로 인한 문제임
                    - PowerShell을 관리자 권한으로 실행한 후

                        <div class="insert-image">
                            <img src="/materials/python/images/S01-01-02-01_01-011.png">
                        </div>

                    - **Set-ExecutionPolicy RemoteSigned** 명령 실행
                        - 변경 여부 확인에서 **Y (또는 A)** 선택

                            <div class="insert-image">
                                <img src="/materials/python/images/S01-01-02-01_01-012.png">
                            </div>

- **가상환경 구축 명령어 정리**

    <div class="insert-image">
        <img src="/materials/python/images/S01-01-02-01_01-013.png">
    </div>

    - Linux / MAC의 경우

        ```bash
        cd workspace
        python -m venv myenv
        cd fab
        source ./bin/activate

        pip install numpy pandas matplotlib jupyter
        jupyter notebook

        deactivate
        ```

    - Windows의 경우

        ```bash
        cd workspace
        python -m venv myenv
        cd fab
        ./Scripts/activate

        pip install numpy pandas matplotlib jupyter
        jupyter notebook

        deactivate
        ```

<br><br>

> - 가상환경의 이름을 '프로젝트 명'으로 할 것인가, 'venv' 등의 통일된 이름으로 할 것인가는 개발자/팀의 정책에 따를 것
>   - '프로젝트 명'으로 하는 경우
>       - 다양한 프로젝트를 수행하는 경우에 프로젝트 별로 인식하기 편리함
>       - 생성되는 폴더 명이 프로젝트 명이 되고 그 안에 모든 파일들이 설치됨
>       - 프로젝트의 환경 파일들이 포함된 각 폴더들이 앞으로 작성할 소스코드와 같은 위치에 있게 됨
>       - 파일이 아니라 폴더이므로 파일끼리 섞이거나 할 위험이 있는 것은 아님
>   - 'venv'로 하는 경우
>       - 프로젝트 폴더를 별도의 이름으로 만든 후, 프로젝트 폴더 내부에서 'venv'로 가상환경을 만듦
>       - 프로젝트 폴더 자체가 별도로 만들어졌기 때문에 모든 프로젝트 환경에서 동일한 설정을 사용할 수 있음
>       - 앞으로 작성할 소스코드 외에는 'venv' 폴더 안에 필요한 모든 파일들이 설치됨
>       - 다만 폴더의 레벨이 한 단계 더 깊어짐
{: .expert-quote}


## 2. Google Colaboratory(Colab)

### 2.1 가상환경 직접 구축 시의 문제점

- **일반 프로젝트의 경우**
    - 혼자 작업할 때
        - 그다지 문제는 없음

    - 여러 팀원이 협업할 때
        - 각자의 시스템 사양이 다른 경우 개발 환경이 차이가 날 수 있음
        - 동일한 가상환경을 구축한다면 문제가 없어야 하지만 실제로는 가끔 문제가 발생함
            - 예: 동일한 가상환경이지만 CPU/RAM 등의 차이에 의한 결과물의 성능 차이 발생. 특히 동기화 등의 작업 시 발생 가능성이 높아짐

- **학습/교육의 경우**
    - 학습자의 환경이 서로 다를 경우, 예제 코드의 작동 여부 및 결과가 다르게 나타날 가능성이있음 

- **데이터 분석 / AI 관련 프로젝트의 경우**
    - 가장 큰 문제는 비용
    - 딥러닝용(또는 데이터 분석용) PC/서버를 직접 구성하려면 대규모의 GPU가 필요함
        - NVIDIA H199 80GB(고급사양) 가격: 약 4,565만원(2025년 3월 기준)
        - NVIDIA RTX 5090(개인용) 가격: 약 705만원(2025년 3월 기준)
        - NVIDIA RTX 4090 Ti 24G(개인용) 가격: 약 475만원(2025년 3월 기준)

        <span style="color: red;">**→ 이러한 이유로 Google에서 제공하는 Colab 활용 권장**</span>

### 2.2 Colab의 특징

- **Colab의 지원 환경**
    - 파이썬 / R 지원
    - Jupyter Notebook과 유사한 클라우드 기반 개발 환경 제공
    - 브라우저 기반의 개발환경 제공 → 스마트폰에서도 사용 가능
    - GPU / TPU 지원

- **Colab과 다른 유사 서비스와의 차별성**
    - 유사한 다른 서비스의 경우: 1일 무료 사용량을 초과하면 자동으로 과금됨 
    - Colab의 경우: 1일 무료 사용량을 초과하면 당일의 GPU 사용이 제한될 뿐 과금되지 않음

### 2.3 Colab 사용하기

- Gmail 계정 생성(무료)
    - 타 메일 계정을 사용해도 괜찮지만 서비스 활용 시 제한이 있을 경우가 있음

- Google Drive(G-Drive) 확인
    - Colab 서비스는 무료인 대신 12시간이 지나면 메모리에서 작업내용이 삭제됨
    - 작업 내용, 데이터 파일 등을 Google Drive와 연동하여 사용함으로써 해결 가능
    - 무료 용량: 최대 15GB

- Colab 환경 설정
    - G-Drive 화면에서 마우스 우클릭 메뉴에서 Colaboratory 선택

        <div class="insert-image">
            <img src="/materials/python/images/S01-01-02-01_01-019.png" style="border: 1px solid gray;">
        </div>

    - Colaboratory 메뉴가 보이지 않는 경우
        1. https://colab.research.google.com 접속

        2. 우측 하단 “새 노트“ 선택하여 Note 생성

            <div class="insert-image">
                <img src="/materials/python/images/S01-01-02-01_01-014.png">
            </div>

        3. 원하는 파일명 지정 후 작업 시작

            <div class="insert-image">
                <img src="/materials/python/images/S01-01-02-01_01-015.png">
            </div>

            - 작업 내용은 자동 저장되며, 파일 메뉴에서 직접 저장도 가능

                <div class="insert-image">
                    <img src="/materials/python/images/S01-01-02-01_01-016.png">
                </div>

        4. 저장 후 자동으로 생성된 Colab Notebooks 폴더로 돌아가서 작업 파일 저장 확인 가능

            <div class="insert-image">
                <img src="/materials/python/images/S01-01-02-01_01-017.png">
            </div>

            <div class="insert-image">
                <img src="/materials/python/images/S01-01-02-01_01-018.png">
            </div>

    - 작업 진행

        <div class="insert-image">
            <img src="/materials/python/images/S01-01-02-01_01-020.png">
        </div>

## 3. 외부 라이브러리 및 테스트 도구 설치

> - 사용 교재: 인공지능 소프트웨어 품질 보증을 위한 테스트 기법 (제이펍)
>   - 단점: 구 버전의 Tensorflow 등을 사용하고 있으므로 현재 시점에서 실무에 구현할 때에는 부적절한 부분이 있을 수 있음
>   - 수정 및 검증이 가능한 범위에서 수정하여 진행함(버전, 코드, 활용 등)
{: .expert-quote}

### 3.1 외부 라이브러리

- **TensorFlow**
    - Google에서 개발한 오픈소스 머신러닝 플랫폼
    - 딥러닝 모델을 포함한 다양한 인공지능 모델의 구축, 학습, 배포에 사용됨
    - 복잡한 계산 과정을 데이터 흐름 그래프(data flow graph)로 표현하여 효율적인 연산을 지원함
    - 서버, 엣지 디바이스, 브라우저 등 다양한 환경에 모델을 배포할 수 있는 강력한 기능을 제공함
    - 교재에서는 1.12.3 버전을 사용하고 있음
    - 설치 명령: pip install tensorflow==1.12.3

- **Keras**
    - 딥러닝 모델 구축을 간편하게 해주는 고수준 API
    - 특히 TensorFlow의 상위 계층 라이브러리로 많이 사용
    - 직관적인 코드로 빠르게 모델을 개발할 수 있도록 지원함
    - 복잡한 딥러닝 모델을 쉽게 만들 수 있는 장점을 가짐
    - 단, 매우 복잡한 프로젝트에서는 직접적인 백엔드 프레임워크(예: TensorFlow)를 다루는 것이 더 유연할 수 있음
    - 교재에서는 2.2.5 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install keras==2.2.5</span>

- **NumPy**
    - 파이썬에서 다차원 배열(ndarray) 객체를 효과적으로 처리하고, 고성능 수치 계산을 가능하게 하는 핵심 라이브러리
    - 대규모 행렬 및 벡터 연산에 최적화되어 있음
    - 머신러닝 모델의 학습 과정에서 발생하는 수많은 수치 계산의 기반이 됨
    - 교재에서는 1.16.6 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install numpy==1.16.6</span>

- **Pandas**
    - 데이터 분석 및 조작을 위한 강력한 라이브러리
    - 'DataFrame'이라는 테이블 형태의 자료 구조를 중심으로 데이터의 읽기, 정리, 필터링, 통계분석 등을 효율적으로 수행하기 위하여 사용됨
    - 데이터 전처리 단계에서 필수적이며, AI 모델에 입력할 데이터를 가공하는 데 매우 유용합니다.
    - 교재에서는 1.1.2 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install pandas==1.1.2</span>

- **Matplotlib**
    - 파이썬에서 데이터를 시각화하기 위해 가장 널리 사용되는 라이브러리
    - 간단한 선 그래프부터 복잡한 3D 플롯까지 다양한 형태의 그래프를 생성
    - 데이터의 패턴, 추세, 이상치 등의 시각적 분석과 모델의 성능 평가를 지원함
    - 교재에서는 3.3.2 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install matplotlib==3.3.2</span>

- **Scikit-learn**
    - 머신러닝에 필요한 다양한 알고리즘과 유틸리티를 제공하는 범용 라이브러리
    - 회귀, 분류, 군집화 등의 주요 머신러닝 작업을 위한 표준 도구들을 포함
    - AI 모델의 정밀도, 재현율, F1-스코어 등 성능 지표 계산에도 활용
    - 다양한 머신러닝 기법을 쉽게 적용하고 평가할 수 있게 지원함
    - 교재에서는 0.23.2 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install scikit-learn==0.23.2</span>

- **XGBoost**
    - 'eXtreme Gradient Boosting' 알고리즘을 구현한 라이브러리
    - 분류와 회귀 문제에서 뛰어난 성능을 보이는 트리 기반 앙상블 모델을 정의하고 학습하는 데 사용
    - 캐글(Kaggle)과 같은 데이터 과학 경연에서 자주 우승을 차지할 만큼 강력한 성능을 자랑함
    - 교재에서는 1.2.0 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install xgboost==1.2.0</span>

- **Z3-solver**
    - 'SMT(Satisfiability Modulo Theories) Solver' 도구 중 하나
    - 논리식의 만족성(satisfiability)을 평가하는 데 사용됨
    - 소프트웨어 검증, 형식 추론, 최적화 문제 등 복잡한 논리 기반 문제 해결에 활용됨
    - 교재에서는 4.8.6 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install z3-solver==4.8.6</span>

- **Numba**
    - 파이썬 코드를 CPU나 GPU에서 실행될 수 있는 고성능 기계어 코드로 변환하여 실행 속도를 고속화하기 위한 Just-In-Time(JIT) 컴파일러 라이브러리
    - 특히 반복문이나 수치 연산이 많은 파이썬 코드를 최적화하는 데 효과적
    - 교재에서는 0.51.2 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install numba==0.51.2</span>

- **MMdnn**
    - 다양한 딥러닝 프레임워크(예: TensorFlow, PyTorch, Keras)로 만들어진 모델의 형식을 서로 변환하거나 배포하기 위한 라이브러리
    - 'Model Migration and Deployment for Deep Neural Network'의 약자
    - 서로 다른 프레임워크 간의 모델 호환성을 높여줌
    - 교재에서는 0.3.1 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install mmdnn==0.3.1</span>

- **H5py**
    - HDF5(Hierarchical Data Format) 형식의 파일을 다루기 위한 파이썬 인터페이스 라이브러리
    - AI 모델의 가중치, 학습 데이터, 모델 구조 등 대용량 데이터를 파일에 저장하거나 읽어들이는 데 주로 사용
    - 효율적인 데이터 관리를 가능하게 함
    - 교재에서는 2.8.0 버전을 사용하고 있음
    - 설치 명령: <span style="color: green;">pip install h5py==2.8.0</span>

> - **일괄설치 명령**<br>
>   - pip install tensorflow==1.12.3 keras==2.2.5 numpy==1.16.6 pandas==1.1.2 matplotlib==3.3.2 scikit-learn==0.23.2 xgboost==1.2.0 z3-solver==4.8.6 numba==0.51.2 mmdnn==0.3.1 h5py==2.8.0
{: .expert-quote}
