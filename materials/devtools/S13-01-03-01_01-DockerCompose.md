---
layout: page
title:  "Docker Compose 이해"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-03-01_01-DockerCompose
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - `Docker Compose`는 **여러 개의 컨테이너로 구성된 애플리케이션을 정의하고 실행하기 위한 도구**
> - 현대적인 웹 서비스는 보통 웹 서버, API 서버, 데이터베이스, 캐시 서버 등이 함께 유기적으로 돌아가야 하는데, 
> - 이를 일일이 `docker run`으로 실행하는 불편함을 해결해 줌
{: .common-quote}


## 1. Docker Compose의 핵심 철학

- 도커 컴포즈의
  - **핵심 철학: 정의서 한 장으로 인프라 구축**
  - **핵심 요소: `docker-compose.yml`**이라는 YAML 형식의 설정 파일
    - 이 파일에 애플리케이션에 필요한 서비스, 네트워크, 볼륨 등을 **선언적**으로 기술함

- **주요 이점**
  - **다중 컨테이너 관리:** 여러 컨테이너를 하나의 명령으로 시작, 중지, 재생성할 수 있음
  - **환경 격리:** 각 프로젝트별로 독립된 가상 네트워크를 생성하여 컨테이너 간 간섭 방지
  - **서비스 가속화:** 변경된 컨테이너만 다시 생성하므로 설정 변경 시 매우 빠름
  - **변수 활용:** **`.env`** 파일을 통해 환경별(개발, 테스트, 운영) 설정을 쉽게 바꿀 수 있음


## 2. docker-compose.yml의 구조 이해

- 파일은 크게 세 가지 주요 섹션(services, networks, volumes)으로 나뉨

  ```yaml
  version: '3.8' # 파일 규격 버전 (최근에는 생략 가능)

  services:      # 1. 실행할 컨테이너들의 정의
    web-app:
      build: .   # 현재 디렉토리의 Dockerfile을 빌드
      ports:
        - "8080:80"
      depends_on:
        - db     # db 서비스가 먼저 실행된 후 시작됨
      environment:
        - DB_HOST=db

    db:
      image: postgres:15
      volumes:
        - db-data:/var/lib/postgresql/data # 2. 데이터 영속성을 위한 볼륨

  networks:      # 3. 컨테이너 간 통신을 위한 네트워크 (자동 생성됨)
    default:

  volumes:       # 물리적인 저장소 공간 정의
    db-data:
  ```


## 3. 핵심 사용법 및 워크플로우

- 컴포즈를 사용하는 과정은 매우 직관적

1. **서비스 실행 (`up`)**
  - 설정된 모든 서비스를 백그라운드(`-d`)에서 실행
  - 이미지가 없다면 빌드하거나 내려받고, 네트워크와 볼륨도 자동으로 생성

    ```bash
    docker-compose up -d
    ```

2. **상태 확인 및 로그 (`ps`, `logs`)**

    ```bash
    docker-compose ps        # 실행 중인 서비스 목록 확인
    docker-compose logs -f   # 전체 혹은 특정 서비스의 로그를 실시간 확인
    ```

3. **서비스 중지 및 삭제 (`down`)**
  - 실행 중인 컨테이너를 멈추고 삭제
  - 생성된 가상 네트워크도 함께 삭제되어 시스템을 깨끗하게 유지 (단, 볼륨은 명시하지 않으면 보존됨)

    ```bash
    docker-compose down
    ```


## 4. 다중 컨테이너 관리

- 실제 운영 환경의 애플리케이션은 단 하나의 컨테이너로만 작동하는 경우가 거의 없음
  - 예시: 일반적인 웹 서비스라면 웹 애플리케이션(Spring Boot), 관계형 데이터베이스(MySQL), 인메모리 캐시(Redis)가 서로 유기적으로 연결되어 하나의 시스템을 이룸

- 이러한 다중 컨테이너 환경을 효율적으로 통합 관리하기 위해 Docker Compose(도커 컴포즈)가 탄생함

### 4.1 단일 컨테이너 실행의 한계와 컴포즈의 필요성

- 만약 Docker Compose 없이 세 개의 컨테이너(Spring Boot, MySQL, Redis)를 띄우려면 다음과 같은 불편함과 한계가 발생함

  1. **복잡하고 긴 명령어**
    - 각 컨테이너를 실행할 때마다
    - 네트워크, 볼륨, 환경 변수, 포트 포워딩 옵션이 주렁주렁 달린 긴 `docker run` 명령어를 터미널에 일일이 입력해야 함

  2. **실행 순서(의존성) 관리의 어려움**
    - 데이터베이스(MySQL)가 완전히 켜지기 전에 웹 애플리케이션(Spring Boot)이 먼저 실행되면,
    - DB 연결 실패 오류가 발생하며 앱이 종료됨 🡲 즉, 사람이 직접 순서를 맞추어 켜야 함

  3. **네트워크 연결의 번거로움**
    - 컨테이너들이 서로 통신할 수 있도록 가상 네트워크를 수동으로 만들고(`docker network create`),
    - 실행 시마다 매번 지정해 주어야 함

- Docker Compose는 이 모든 설정을 `docker-compose.yml`이라는 하나의 파일에 코드로 정의(선언)하고,
- 명령어 한 줄로 전체 시스템을 제어할 수 있게 해줌


### 4.2 `docker-compose.yml` 작성 예시

- **Spring Boot + MySQL + Redis**를 한 번에 묶어서 관리하는 표준적인 도커 컴포즈 설정 파일 예시

  ```yaml
  version: '3.8'

  services:
    # 1. 웹 애플리케이션 서비스
    web-app:
      build: .                 # 현재 디렉토리의 Dockerfile을 기반으로 이미지 빌드
      ports:
        - "8080:8080"          # 호스트 8080 포트를 컨테이너 8080 포트와 연결
      environment:
        - SPRING_DATASOURCE_URL=jdbc:mysql://mysql-db:3306/mydb
        - SPRING_REDIS_HOST=redis-cache
      depends_on:              # 의존성 정의: DB와 캐시가 먼저 실행된 후 웹 앱이 실행됨
        - mysql-db
        - redis-cache

    # 2. 데이터베이스 서비스
    mysql-db:
      image: mysql:8.0         # Docker Hub에서 공식 이미지 다운로드
      environment:
        - MYSQL_DATABASE=mydb
        - MYSQL_ROOT_PASSWORD=secretpass
      volumes:
        - mysql-data:/var/lib/mysql # 데이터 영속성을 위한 볼륨 매핑

    # 3. 캐시 서비스
    redis-cache:
      image: redis:7.0-alpine
      ports:
        - "6379:6379"

  # 영구 저장 공간(볼륨) 정의
  volumes:
    mysql-data:
  ```

### 4.3 Docker Compose의 핵심 메커니즘과 동작 원리

- **`docker-compose.yml`** 파일이 실행될 때 🡲 시스템 내부에서는 다음의 기능들이 자동으로 작동

1. **서비스 이름 기반의 내부 통신 (Service Discovery)**

  - 도커 컴포즈는 파일을 실행하는 순간
    - 정의된 서비스들을 위한 전용 격리 네트워크(Default Bridge Network)를 자동으로 생성
    - 이 망 안에서 컨테이너들은 서로의 IP 주소를 알 필요 없이, **YAML 파일에 적힌 서비스 이름(예: `mysql-db`, `redis-cache`)을 호스트네임(도메인)처럼 사용하여 통신** 할 수 있음
      - 상단 예시 코드의 `jdbc:mysql://mysql-db:3306` 구문이 바로 이 원리를 활용한 것

2. **`depends_on`을 통한 실행 순서 보장**

  - **`depends_on`** 속성을 명시하면
  - 도커 컴포즈가 알아서 선행 컨테이너(`mysql-db`, `redis-cache`)를 먼저 구동한 뒤, 후행 컨테이너(`web-app`)를 시작함
    - 사람이 직접 실행 타이밍을 계산할 필요가 없어짐

3. **인프라의 통합 제어 및 생명 주기 관리**

  - 단 두 개의 명령어만으로 전체 인프라의 생명 주기를 완벽하게 제어할 수 있음
    - **통합 실행:** **`docker-compose up -d`**
      - **동작**
        - 이미지 빌드, 다운로드, 네트워크/볼륨 생성, 모든 컨테이너 백그라운드 생성을 단 한 번에 처리

    - **통합 종료:** **`docker-compose down`**
      - **동작**
        - 구동 중인 모든 컨테이너를 안전하게 정지 및 삭제
        - 가상 네트워크까지 깔끔하게 제거하여 시스템 리소스를 확보

## 5. 실습

- **Docker Compose를 이용한 멀티 컨테이너 배포 및 관리**
  - **[Flask 웹 서버(80번 포트)] + [Redis 메모리 데이터베이스]** 조합의 멀티 컨테이너 아키텍처를 이용한 예제
    - 브라우저를 새로고침할 때마다 Redis에 방문 횟수(Count)가 누적되는 직관적인 시스템
  - 여러 개의 컨테이너(웹 서버와 데이터베이스)로 구성된 복잡한 애플리케이션 스택을 단 하나의 설정 파일(`docker-compose.yml`)로 정의하고, 명령어 한 줄로 일괄 가동 및 완전 청소하는 실무 기법을 학습


- **1 단계: 실습 전용 디렉터리 생성 및 이동**
  - 실습 파일들이 엉키지 않도록 새 폴더를 만들고 진입 (Windows PowerShell 기준)

    ```powershell
    mkdir flask-compose-demo
    cd flask-compose-demo
    ```

- **2 단계: 애플리케이션 소스 코드 (`app.py`) 작성**
  - Redis 데이터베이스와 연동하여 방문자 수를 1씩 증가시키는 Flask 웹 서버 코드
  - 폴더 내에 `app.py` 파일을 생성하고 아래 코드를 저장

    ```python
    import time
    import redis
    from flask import Flask

    app = Flask(__name__)

    # Redis 컨테이너 서비스 이름으로 연결
    cache = redis.Redis(host='redis-db', port=6379)

    def get_hit_count():
        retries = 5
        while True:
            try:
                return cache.incr('hits')
            except redis.exceptions.ConnectionError as exc:
                if retries == 0:
                    raise exc
                retries -= 1
                time.sleep(0.5)

    # [@app.index_string] 라인을 삭제하고 바로 라우팅 데코레이터만 남깁니다.
    @app.route('/')
    def hello():
        count = get_hit_count()
        return f'<h1>Docker Compose 실습 성공!</h1><p>방문자 수: {count}번째 보았습니다.</p>\n'

    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000)
    ```

- **3 단계:  의존성 파일 (`requirements.txt`) 작성**
  - Flask 앱 실행에 필요한 파이썬 패키지 목록
  - `requirements.txt` 파일을 만들고 아래 내용을 작성

    ```text
    flask
    redis
    ```

- **4 단계:  웹 서버용 `Dockerfile` 작성**
  - 파이썬 앱을 이미지화하기 위한 설계도 🡲 `Dockerfile` 파일 생성

    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /code
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY app.py .
    EXPOSE 5000
    CMD ["python", "app.py"]
    ```

- **5 단계:  Docker Compose 오케스트레이션 설정 (`docker-compose.yml`) 작성**
  - 웹 서버 빌드 규칙과 Redis DB 인프라를 한 장으로 정의하는 마스터 설계도
  - `docker-compose.yml` 파일을 생성하고 아래 내용을 입력 (YAML 파일이므로 들여쓰기에 주의)

    ```yaml
    services:
      web-server:
        build: .
        ports:
          - "8080:5000"
        depends_on:
          - redis-db

      redis-db:
        image: "redis:alpine"
    ```

- **6 단계:  명령어 단 한 줄로 전체 스택 빌드 및 가동 (`up`)**
  - 이제 터미널에서 여러 명령어를 내릴 필요 없이, 다음 명령 한 줄이면 
    - 도커 컴포즈가 `Dockerfile`을 읽어 웹 이미지를 빌드하고,
    - Redis 이미지를 다운로드하여
    - 내부 가상 네트워크로 묶어 동시에 띄워줌
    - 필요 시, sudo apt install docker-compose를 통해 docker-compose를 설치할 것

      ```powershell
      docker-compose up -d
      ```

      - **`-d` 옵션:** 백그라운드(데몬) 모드로 실행하여 터미널을 자유롭게 사용할 수 있게 함

- **7 단계:  실행 상태 및 브라우저 검증**
  - 가동 중인 컴포즈 서비스 목록을 확인하고 웹페이지에 접속

    ```powershell
    # 컴포즈로 가동 중인 컨테이너 레이아웃 확인
    docker-compose ps
    ```

    - **웹 브라우저 검증:** 주소창에 포트 번호 없이 **`http://localhost`** 를 입력하고 이동
    - 새로고침(`F5`)을 누를 때마다 "방문자 수: X번째 보았습니다."의 숫자가 1씩 끊김 없이 올라가면 멀티 컨테이너 연동 배포에 완벽하게 성공한 것


- **8 단계:  실습 종료 후 흔적 없는 "최종 클린 청소" (`down`)**
  - 실습이 끝났으므로 내 컴퓨터의 자원을 태초의 상태로 되돌리기 위해 청소 진행
  - 도커 컴포즈의 가장 큰 장점은 **지울 때도 단 한 줄로 연관된 모든 컨테이너와 가상 네트워크 인터페이스까지 한꺼번에 안전하게 폭파**할 수 있다는 점

    ```powershell
    # 1. 가동 중인 모든 컨테이너 정지 및 프로세스 소거, 가상 네트워크 삭제
    docker-compose down

    # 2. [완벽주의적 청소] 하드디스크에 생성 및 다운로드된 이미지 설계도 원본까지 완전 소거
    docker rmi flask-compose-demo-web-server:latest
    docker rmi redis:alpine

    # 3. [최종 확인] 내 PC가 실습 전 상태로 깨끗해졌는지 목록 검증
    docker ps -a
    docker images
    ```

> - **핵심 포인트**
>   - **컨테이너 간의 서비스 디스커버리 (Service Discovery):**
>     - `app.py` 코드 내부에서 Redis 주소를 하드코딩된 IP(`172.17.0.2` 등)로 적지 않고, `docker-compose.yml`에 선언한 서비스 이름인 `redis-db`로 적어도 알아서 찾아감
>       - 도커 컴포즈가 내부 가상 DNS 서버를 만들어 이름을 IP로 매핑해 주기 때문<br><br>
>   - **`up`과 `down`의 라이프사이클 관리:**
>     - 일일이 컨테이너를 하나씩 `stop`, `rm` 하던 방식에서 탈피하여,
>     - `docker-compose up`과 `docker-compose down`이라는 거대한 수명 주기로 인프라 전체를 핸들링하는 편리함을 제공
{: .common-quote}


## 6. 컴포즈 활용 팁 (Best Practices)

- **서비스 이름으로 통신하기 (Service Discovery)**
  - 컴포즈로 띄운 컨테이너들은 동일한 기본 네트워크에 속함
  - 이때 컨테이너의 IP 주소를 알 필요 없이, **YAML 파일에 정의한 서비스 이름**을 호스트 이름처럼 사용하여 통신할 수 있음
    - 예: 웹 앱 설정 파일에서 DB 접속 주소를 `localhost`가 아닌 `db`로 설정하면 도커가 알아서 연결해 줌

- **`depends_on`의 한계 이해**
  - `depends_on`은 컨테이너의 **실행 순서**만 보장함
  - DB 컨테이너가 켜졌다고 해서 그 안의 DB 서비스가 즉시 쿼리를 받을 준비가 된 것은 아닐 수 있음
  - 이를 해결하기 위해 애플리케이션 코드 내에서 재시도(Retry) 로직을 넣거나 `wait-for-it` 같은 스크립트를 사용하기도 함

- **볼륨 매핑을 통한 실시간 개발**
  - 개발 환경에서는 소스 코드를 수정할 때마다 이미지를 다시 빌드하는 것이 번거울 수 있음
  - `volumes` 설정을 통해 호스트의 소스 폴더를 컨테이너에 동기화하면, 코드 수정 즉시 컨테이너 안에 반영되어 매우 효율적

<br>

> - **요약**
>   - Docker Compose는 "여러 컨테이너를 묶어서 하나의 패키지처럼 관리하는, 복잡한 다중 컨테이너 인프라의 명세서이자 자동 제어기"
>   - 단일 컨테이너 단계를 넘어 컴포즈를 다룰 줄 알게 되면, 로컬 개발 환경 구축부터 실제 운영 서버 배포의 전 과정을 코드로 규격화하여 관리할 수 있게 됨<br><br>
>   1. **Dockerfile**로 개별 컴포넌트(클래스)를 만들고, 
>   2. **docker-compose.yml**로 그들의 관계(시스템 구성)를 정의하며, 
>   3. **docker-compose up**으로 전체 시스템(애플리케이션)을 구동하는 흐름을 익히면 실무 역량이 한 단계 더 높아질 것
{: .summary-quote}
