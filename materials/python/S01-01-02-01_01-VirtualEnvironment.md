---
layout: page
title:  "파이썬 가상환경 설정"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-01-02-01_01-VirtualEnvironment
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 가상환경 설정 (로컬 시스템)

### 1.1 가상환경을 사용하는 이유

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

### 1.2 파이썬 가상환경 직접 구축하기

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
                    <img src="/materials/images/python/S01-01-02-01_01-007.png">
                </div>

    2. 가상환경 설정

        - python -m venv [생성하고자 하는 가상환경의 이름]

            <div class="insert-image">
                <img src="/materials/images/python/S01-01-02-01_01-008.png">
            </div>

    3. 가상환경 활성화

        - .\Script\activate 명령으로 가상환경 활성화
            - 프롬프트의 앞에 **(가상환경이름)**이 표시되면 활성화 성공임

                <div class="insert-image">
                    <img src="/materials/images/python/S01-01-02-01_01-009.png">
                </div>

        - Linux/Mac의 경우 : **$ source ./bin/activate** 명령으로 활성화
            - 해당 가상환경의 폴더에 들어와 있는 경우에는 source ./bin/activate를 사용하고
            - 그렇지 않은 경우에는 source 가상환경이름/bin/activate를 사용함

        - 가상환경 활성화 오류 발생 시 해결 방안
            - Windows 환경에서 PowerShell을 사용하는 경우 다음과 같은 오류가 자주 발생함

                <div class="insert-image">
                    <img src="/materials/images/python/S01-01-02-01_01-010.png">
                </div>

                - 대부분의 경우 권한부족으로 인한 문제임
                - PowerShell을 관리자 권한으로 실행한 후

                    <div class="insert-image">
                        <img src="/materials/images/python/S01-01-02-01_01-011.png">
                    </div>

                - **Set-ExecutionPolicy RemoteSigned** 명령 실행
                    - 변경 여부 확인에서 **Y (또는 A)** 선택

                        <div class="insert-image">
                            <img src="/materials/images/python/S01-01-02-01_01-012.png">
                        </div>

- **가상환경 구축 명령어 정리**

    <div class="insert-image">
        <img src="/materials/images/python/S01-01-02-01_01-013.png">
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
            <img src="/materials/images/python/S01-01-02-01_01-019.png" style="border: 1px solid gray;">
        </div>

    - Colaboratory 메뉴가 보이지 않는 경우
        1. https://colab.research.google.com 접속

        2. 우측 하단 “새 노트“ 선택하여 Note 생성

            <div class="insert-image">
                <img src="/materials/images/python/S01-01-02-01_01-014.png">
            </div>

        3. 원하는 파일명 지정 후 작업 시작

            <div class="insert-image">
                <img src="/materials/images/python/S01-01-02-01_01-015.png">
            </div>

            - 작업 내용은 자동 저장되며, 파일 메뉴에서 직접 저장도 가능

                <div class="insert-image">
                    <img src="/materials/images/python/S01-01-02-01_01-016.png">
                </div>

        4. 저장 후 자동으로 생성된 Colab Notebooks 폴더로 돌아가서 작업 파일 저장 확인 가능

            <div class="insert-image">
                <img src="/materials/images/python/S01-01-02-01_01-017.png">
            </div>

            <div class="insert-image">
                <img src="/materials/images/python/S01-01-02-01_01-018.png">
            </div>

    - 작업 진행

        <div class="insert-image">
            <img src="/materials/images/python/S01-01-02-01_01-020.png">
        </div>
