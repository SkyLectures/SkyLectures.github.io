---
layout: page
title:  "도커의 워크플로우 (Docker Lifecycle)"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-01-01_03-DockerWorkflow
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - 도커는 애플리케이션을 개발하고 배포하는 과정을 **Build(구축) 🡲 Ship(전송) 🡲 Run(실행)** 이라는 3단계로 표준화함
>   - 도커의 가장 핵심적인 표준 워크플로우(Lifecycle)
> - 이는 전통적인 배포 방식의 복잡성을 제거하고 소프트웨어 공급망을 단순화하는 핵심 프로세스
{: .summary-quote}

<div class="insert-image">
    <img src="/materials/devtools/images/S13-01-01-01_02-004_Docker_Workflow.png" style="width: 90%;"><br>
</div>


## 1. Build (구축)

### 1.1 내용 설명

- 개발자가 애플리케이션의 실행 환경을 독립된 하나의 '도커 이미지'로 정적 패키징하는 단계 🡲 환경을 코드로 정의하고 이미지화
    - **Dockerfile 작성 (레시피 정의):**
        - 베이스 OS 선택부터 소스 코드 복사, 의존성 패키지 설치, 환경 변수 정의, 컨테이너 기동 명령(`CMD`/`ENTRYPOINT`)까지의 과정을 텍스트 스크립트로 작성
        - 인프라 구성을 명문화함으로써 '인프라의 코드화(IaC, Infrastructure as Code)'를 실현

    - **`docker build` 명령 수행 (이미지 굽기):**
        - 도커 클라이언트가 빌드 명령을 내리면 🡲 도커 데몬이 `Dockerfile`의 명령을 한 줄씩 해석하면서 독립된 레이어(Layer)들을 쌓아 올림
        - 최종적으로 애플리케이션 실행에 필요한 모든 파일 시스템이 압축된 '읽기 전용(Read-only) 이미지'가 호스트 로컬 저장소에 생성됨

- **실무적 관점**
    - 캐시(Cache) 효율을 극대화하도록 `Dockerfile` 명령어 순서를 배치
    - 멀티 스테이지 빌드(Multi-stage Build)를 도입
    - 이미지의 용량을 최소화하고 보안 취약점을 줄이는 최적화 작업이 수행됨

### 1.2 실습

- **예제:** "Python Flask 웹 서버"를 띄우는 예제
    - 내 컴퓨터에 파이썬이 설치되어 있지 않아도 웹 서버를 도커 컨테이너로 띄울 수 있게 됨

    1. **실습 환경 준비 (파일 2개 만들기)**

        1. 작업할 빈 폴더를 하나 만들고, 그 안에 아래의 파일 2개를 생성하기

            ```python
            #//file: "app.py"
            from flask import Flask
            app = Flask(__name__)

            @app.route('/')
            def hello():
                return "<h2>Docker Build 실습 성공!</h2><p>컨테이너에서 실행 중인 웹 서버입니다.</p>"

            if __name__ == '__main__':
                # 0.0.0.0으로 설정해야 컨테이너 외부(호스트)에서 접속이 가능합니다.
                app.run(host='0.0.0.0', port=5000)
            ```

        2. `Dockerfile` (설계도 파일) 만들기
            - 확장자(txt 등) 없이 파일명을 정확히 `Dockerfile`로 만들어야 함

            ```dockerfile
            #//file: "Dockerfile"
            # 1. 베이스 이미지 지정 (파이썬이 내장된 가벼운 리눅스 이미지)
            FROM python:3.9-slim

            # 2. 컨테이너 내부의 작업 디렉토리 설정
            WORKDIR /app

            # 3. 필요한 패키지(Flask) 설치
            RUN pip install flask

            # 4. 현재 폴더의 소스 코드(app.py)를 컨테이너 안으로 복사
            COPY app.py .

            # 5. 컨테이너가 사용할 포트 명시 (안내용)
            EXPOSE 5000

            # 6. 컨테이너가 켜질 때 실행할 명령어
            CMD ["python", "app.py"]
            ```

    2. **터미널 명령어로 빌드 및 실행하기**
        - 파일이 있는 폴더에서 터미널(CMD, 파워쉘, 또는 Git Bash 등)을 열고 아래 명령어를 순서대로 입력

        - **1 단계: 이미지 빌드 (`docker build`)**
            - `Dockerfile`을 읽어서 `my-flask-app`이라는 이름의 이미지를 만듦

            ```bash
            docker build -t my-flask-app:1.0 .
            ```

            > - **⚠️ 주의:** 
            >   - 맨 끝에 있는 **점(`.`)**을 반드시 찍어야 함!
            >   - 이는 '현재 디렉토리'를 빌드 컨텍스트로 사용하겠다는 의미임
            >   - 첫 빌드 시에는 파이썬 이미지를 다운로드하고 패키지를 설치하느라 시간이 조금 걸릴 수 있음
            {: .common-quote}

        - **2 단계: 빌드된 이미지 확인**
            - 이미지가 성공적으로 구워졌는지 목록 확인

            ```bash
            docker images
            ```

            - 목록에 `my-flask-app`과 태그 `1.0`이 보인다면 빌드 성공

        - **3 단계: 컨테이너 실행 (`docker run`)**
            - 포트 포워딩(`-p`) 옵션을 사용해 이미지를 컨테이너로 가동

            ```bash
            docker run -d -p 80:5000 --name flask-server my-flask-app:1.0
            ```

            - **`-d`**: 백그라운드(데몬) 모드로 실행 🡲 터미널 창을 계속 사용할 수 있게 함
            - **`-p 80:5000`**: 내 컴퓨터(호스트)의 8080 포트로 접속 🡲 컨테이너의 5000 포트로 전달

        - **결과 확인하기**
            - 인터넷 브라우저 주소창에 **`http://localhost:8080`** 입력
            - 화면에 "Docker Build 실습 성공!"이라는 한글 메시지가 뜨면 성공


        - **실습 종류 후, 컨테이너를 안전하게 끄고 삭제하는 방법**

            ```bash
            # 1. 실행 중인 컨테이너 정지
            docker stop flask-server

            # 2. 정지된 컨테이너 삭제
            docker rm flask-server
            ```

> - **학습 포인트 복습**
>   - 만약 `app.py`에서 문구를 수정하고 브라우저에서 새로고침을 하면 반영이 될까?
>       - <span style="color: darkred;">**안됨**</span>
>       - 이미지는 '불변(Immutable)'이기 때문에, 코드를 수정했다면 **`docker build` 단계를 다시 수행해서 새 이미지를 구워야 함
>       - 단, 이때 다시 빌드를 하면
>           - 도커의 **레이어 캐싱** 기능 덕분에 파이썬 이미지 다운로드나 Flask 설치 과정은 순식간에 지나가고
>           - 변경된 소스 코드 레이어만 빠르게 새로 빌드되는 것을 확인할 수 있음
{: .common-quote}


## 2. Ship (전송/배포)

### 2.1 내용 설명

- 로컬 환경에서 검증을 마친 불변의 이미지를 원격 저장소에 업로드, 어디서나 다운로드할 수 있도록 유통하는 단계 🡲 표준화된 이미지의 중앙 저장 및 공유
    - **`docker tag`를 통한 버전 관리:**
        - 생성된 이미지를 `docker push`를 통해 레지스트리(Docker Hub 등)에 업로드
        - 업로드 전, 이미지에 레지스트리 주소와 프로젝트명, 고유한 버전(Tag)을 부여
            - 예: `[myregistry.com/myapp:v1.0.0](https://myregistry.com/myapp:v1.0.0)`
        - 태그 관리를 통해 상용 환경에 배포된 버전과 이전 버전들을 명확히 구분하고 관리할 수 있음

    - **`docker push` 명령 수행 (레지스트리 업로드):**
        - 로컬에 저장된 이미지 레이어들을 네트워크를 통해 도커 레지스트리(Docker Registry)로 전송
        - 도커는 레이어 아키텍처를 따름 🡲 이미 레지스트리에 존재하는 레이어는 제외하고 **새롭게 변경된 레이어만 전송**
            - 대역폭과 시간을 크게 절약

- **실무적 관점**
    - 전 세계 공용 저장소인 **Docker Hub**를 사용할 수 있음
    - 또는 기업의 보안 정책에 따라 사내 폐쇄망에 사설 레지스트리(Private Registry)나 AWS ECR, Github Packages 등을 구축하여 배포 파이프라인(CI/CD)과 연동

### 2.2 실습

- **[실습 1] Docker Hub를 이용한 이미지 배포(Ship) 및 클린 삭제**
    - 전 세계 공용 저장소인 Docker Hub에 내가 만든 불변의 이미지를 업로드하고, 안전하게 원격 저장소를 삭제하는 전체 파이프라인의 학습

    - **Prerequisites (사전 준비)**
        - [Docker Hub 공식 홈페이지](https://hub.docker.com/) 회원가입 완료
        - 로컬 환경에 `my-flask-app:1.0` 이미지 빌드 완료 상태

    - **1 단계: 터미널에서 Docker Hub 인증(로그인)하기**
        - Windows PowerShell 또는 CMD를 열고 도커 엔진에게 도커 허브 계정 권한 부여

            ```powershell
            docker login
            ```

            - 명령어를 치면 `Username:`과 `Password:`를 입력하라고 나옴 🡲 가입정보를 입력하여 `Login Succeeded` 메시지를 확인

    <br>

    - **2 단계: 이미지에 원격 저장소 주소(Tag) 부여하기**
        - 도커 허브는 `[도커허브ID]/[이미지명]:[태그]` 구조를 주소로 인식함
        - 기존 로컬 이미지에 배송지 주소를 붙여주는 작업

            ```powershell
            # 규칙: docker tag [기존이미지명:태그] [내도커허브ID]/[원하는레포지토리명:태그]
            docker tag my-flask-app:1.0 도커허브ID/flask-test:1.0
            ```

    <br>

    - **3 단계: Docker Hub로 이미지 업로드 (Push)**
        - 네트워크를 통해 인터넷 저장소로 이미지 전송 🡲 레이어별로 쪼개져 올라가는 속도 확인

            ```powershell
            docker push 도커허브ID/flask-test:1.0
            ```

    <br>

    - **4 단계: 로컬 이미지 완전 청소 후 원격 배포 검증 (Run)**
        - 내 컴퓨터에 있는 이미지를 강제로 지운 뒤,
        - 도커 허브 주소만 입력하여 자동으로 다운로드 및 실행이 되는지 배포 환경을 최종 검증

            ```powershell
            # 1. 로컬에 띄워져 있던 컨테이너와 이미지 완전 삭제
            docker rmi 도커허브ID/flask-test:1.0
            docker rmi my-flask-app:1.0

            # 2. 내 컴퓨터엔 이미지가 없지만, 도커 허브 주소를 호출하면 Pull과 동시에 실행됨
            docker run -d -p 80:5000 --name flask-server 도커허브ID/flask-test:1.0
            ```

            - 웹 브라우저를 열고 `http://localhost`에 접속하여 "Docker Build 실습 성공!"이 뜨는지 확인

    <br>

    - **5 단계: [공간 절약] Docker Hub 원격 저장소 클린 삭제**
        - 실습이 끝났으므로 도커 허브 서버의 자원을 반납하기 위해 웹 대시보드에서 프로젝트를 영구 삭제

        1. [Docker Hub](https://hub.docker.com/) 웹사이트 로그인 후 상단 **[Repositories]** 메뉴 클릭
        2. 방금 올린 `flask-test` 레포지토리 클릭
        3. 상단 탭 메뉴 맨 우측의 **[Settings]** 클릭
        4. 화면 맨 아래(Danger Zone)로 스크롤을 내려 빨간색 **[Delete repository]** 버튼 클릭
        5. 확인 창에 레포지토리 이름(`flask-test`)을 입력하고 최종 삭제 진행

    <br>

    - **6 단계: 실습 종료 후 깨끗이 지우기**
        - 도커 허브에서 새로 다운로드받아 가동 중인 '컨테이너(프로세스)' 강제 정지 및 삭제

            ```powershell
            docker rm -f flask-server
            ```

        - 다운로드받으면서 내 로컬 하드디스크에 다시 쌓인 '원격 이미지 파일' 완전 삭제

            ```powershell
            docker rmi 도커허브ID/flask-test:1.0
            ```

        - [최종 확인] 내 PC에 가동 중인 컨테이너나 남은 이미지가 없는지 눈으로 검증

            ```powershell
            docker ps -a
            docker images
            ```            

    <br>

    > - **참고**
    >   - 도커 허브에 올릴 때 `docker login` 절차가 필수
    >   - 웹에서 영구 삭제가 가능함
    {: .common-quote}

<br>

- **[실습 2] 로컬 사설 레지스트리(Private Registry) 구축 및 배포**
    - 기업 보안 정책상 외부망(도커 허브)을 쓸 수 없을 때, 기업 폐쇄망 내부에 '컨테이너 형태의 사설 저장소'를 구축하고 유통하는 실무 기법 학습

    - **사전준비**
        - 개발 환경에서 Flask 애플리케이션 이미지 빌드

            ```powershell
            docker build -t my-flask-app:1.0 .
            docker images
            ```

    <br>

    - **1 단계: 로컬 사설 레지스트리 서버 가동하기 (5000번 포트)**
        - 도커 공인 진영이 제공하는 오픈소스 저장소 엔진(`registry:2`)을 내 컴퓨터의 `5000`번 포트에 백그라운드로 실행

            ```powershell
            docker run -d -p 5000:5000 --name local-registry registry:2
            ```

            - **인프라 구조:** 내 컴퓨터 내부(`localhost:5000`)에 완벽한 독립형 이미지 사설 저장소가 오픈되었음

    <br>

    - **2 단계: 사설 레지스트리용 주소(Tag) 부여하기**
        - 사설 저장소에 이미지를 밀어 넣으려면 이미지 이름 앞에 저장소의 네트워크 주소(`localhost:5000`)를 지정해야 함

            ```powershell
            # 규칙: docker tag [기존이미지명:태그] localhost:5000/[이미지명:태그]
            docker tag my-flask-app:1.0 localhost:5000/flask-local:1.0
            ```

    <br>

    - **3 단계: 로컬 사설 레지스트리에 이미지 업로드 (Push)**
        - 인터넷망을 타지 않고, 로컬 가상 네트워크 인터페이스를 통해 방금 만든 `local-registry` 컨테이너 안으로 이미지를 업로드
        - 별도의 로그인 인증 없이 즉시 초고속으로 전송

            ```powershell
            docker push localhost:5000/flask-local:1.0
            ```

    <br>

    - **4 단계: 사설 배포 검증을 위한 로컬 이미지 '완전 청소' (Clean Slate)**
        - 훈련 환경을 초기화하기 위해 기존 실습 컨테이너와 이미지 조각 삭제
            - 오리지널과 사설 주소 이미지 둘 다 지워야 로컬 하드가 완벽히 비워짐
        - 로컬 시스템에 만든 사설 저장소 주소로 서비스 배포

            ```powershell
            # 1. 기존 가동 중인 실습 서버 및 태그 이미지 삭제
            docker rmi localhost:5000/flask-local:1.0

            # 2.100% 사설 Pull 화면 연출을 위해 필수!
            docker rmi my-flask-app:1.0
            ```

    <br>

    - **5 단계: 사설 저장소(local-registry)로부터 이미지를 가져와 80포트로 서비스 구동**
    
            ```powershell
            docker run -d -p 80:5000 --name my-flask-server localhost:5000/flask-local:1.0
            ```

            - 웹 브라우저 창에 `http://localhost`를 입력하여 정상적으로 웹서버가 구동되는지 확인합니다.

    <br>

    - **6 단계: 사설 인프라 및 자원 깨끗이 지우기**
        - 실습이 종료되었으므로 
        - 내 컴퓨터의 하드디스크 용량과 포트를 확보하기 위해 사설 레지스트리 엔진과 저장된 데이터 레이어 전체를 지움

            ```powershell
            # 1. 사설 저장소에서 다운로드받아 가동 중인 웹 서버 컨테이너 강제 종료 및 삭제
            docker rm -f my-flask-server

            # 2. 내 컴퓨터를 저장소 서버로 만들었던 레지스트리 엔진 삭제 
            # (-v 옵션으로 레지스트리 컨테이너 내부 묵직하게 쌓였던 저장 데이터 조각까지 완전 청소)
            docker rm -f -v local-registry

            # 3. 테스트 구동하면서 내 하드디스크에 다시 다운로드되었던 사설 이미지 잔재 최종 소거
            docker rmi localhost:5000/flask-local:1.0

            # 4. [최종 상태 확인] 내 PC가 실습 전 상태로 완전히 깨끗해졌는지 목록 확인
            docker ps -a
            docker images     
            ```

    <br>

> - **참고**
>   - `registry:2`라는 도커 이미지를 사용하여 "인프라(저장소)조차도 컨테이너로 띄워서 해결한다"는 도커의 철학을 이해
>   - 마지막 청소 단계에서 `-v` 옵션을 주어야 레지스트리 컨테이너 안에 쌓인 Flask 이미지 파일들까지 디스크에서 깔끔하게 삭제됨
>   - `registry:2` 도커 이미지까지 지우려면 다음을 실행해 줌
>       - docker rmi registry:2
{: .common-quote}


## 3. Run (실행/운영)

- 운영 서버, 테스트 서버, 혹은 다른 개발자의 PC에서 레지스트리에 저장된 이미지를 가져와 격리된 프로세스(컨테이너)로 구동하는 단계 🡲 이미지를 실체화하여 서비스 가동
    - **`docker pull` 명령 수행 (이미지 다운로드):**
        - 도커 호스트(데몬)가 지정된 레지스트리로부터 이미지를 다운로드
        - 로컬에 이미 동일한 레이어가 있다면 다운로드를 건너뛰고 없는 레이어만 수신

    - **`docker run` 명령 수행 (컨테이너 인스턴스화):**
        - 내려받은 정적 이미지 위에 쓰기 가능 레이어(Writable Layer)를 한 층 얹어 동적인 컨테이너 프로세스로 실행
        - 인프라 자원을 할당
            - 호스트와 컨테이너 간의 가상 통신로를 여는 **포트 포워딩(`-p`)**
            - 데이터 영속성을 위한 **볼륨 매핑(`-v`)**
            - 실행에 필요한 **환경 변수(`-e`)** 등을 함께 주입

- **실무적 관점:**
    - 개발 PC에서 테스트를 마친 이미지와 운영 서버에서 구동하는 이미지가 완벽히 동일 🡲 환경 차이로 인한 장애가 발생하지 않음
    - 서비스 스케일 아웃(Scale-out)이 필요할 때 `docker run` 명령을 반복하는 것만으로 단 몇 초 만에 동일한 서버를 수십 대 수준으로 확장할 수 있음

<br>

> - **요약 : Build 🡲 Ship 🡲 Run이 주는 궁극의 이점**
>   - 이 라이프사이클은 단순히 단계의 나열이 아니라, "한 번 빌드하면 어디서나 실행된다(Build Once, Run Anywhere)"는 철학을 완성하는 주기
>   - 개발자는 **Build** 단계에서 환경 설정을 완결 짓고, 배포 담당자는 **Ship** 단계에서 유통을 표준화하며,
>   - 운영자는 **Run** 단계에서 환경 설정에 대한 고민 없이 서비스 안정성에만 집중할 수 있게 됨
{: .summary-quote}

## 4. 도커 사용하기

- 도커의 핵심은 "어디서나 동일하게 실행되는 소프트웨어 패키징"

> - **핵심 명령어**
>   - **docker pull [이미지명]**: 레지스트리에서 이미지 다운로드
>   - **docker run [옵션] [이미지명]**: 이미지를 기반으로 컨테이너 생성 및 실행
>   - **docker ps**: 실행 중인 컨테이너 목록 확인
>   - **docker exec -it [컨테이너ID] /bin/bash**: 실행 중인 컨테이너 내부에 접속
{: .common-quote}