---
layout: page
title:  "MongoDB 개요 및 설치, 환경설정"
date:   2026-05-01 09:00:00 +0900
permalink: /materials/S02-03-06-03_01-MongoDbOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. MongoDB 개요

### 1.1 정의 및 개요

- **정의**
    - 현대 애플리케이션 개발 환경에 맞춰 설계된 대표적인 오픈소스 **NoSQL(Not-Only SQL)** 데이터베이스

- **특징**
    - 전통적인 관계형 데이터베이스(RDBMS)가 데이터를 엄격한 테이블(행과 열) 구조로 관리하는 반면,
    - MongoDB는 데이터를 자유로운 형태의 도큐먼트(Document) 형식으로 저장함
    - MongoDB의 명칭은 '어마어마하게 큰'이라는 뜻의 'Humongous'에서 유래
        - 이름에 걸맞게 대용량 데이터 처리와 뛰어난 확장성을 자랑함

- **탄생 배경 및 의의**
    - 과거에는 정형화된 텍스트 데이터가 주를 이루었지만,
    - 스마트폰과 클라우드의 발전<br>
        🡲 로그 파일, 소셜 미디어 피드, 이미지/영상 메타데이터 등<br>
         구조가 일정하지 않은 비정형/반정형 데이터가 폭발적으로 증가함
    - MongoDB는 이러한 대규모 비정형 데이터를
        - 스키마 변경에 따른 부담(서비스 중단, 마이그레이션 공수 등) 없이 유연하게 적재하고,
        - 초당 수만 건의 읽기/쓰기 작업을 안정적으로 처리하기 위해 고안됨
    - 현재는 전 세계 개발자들이 가장 선호하는 대표적인 NoSQL 솔루션으로 자리 잡고 있음

### 1.2 도큐먼트 지향 아키텍처

- 데이터를 JSON(JavaScript Object Notation)과 유사한 구조인 BSON(Binary JSON) 포맷의 도큐먼트로 저장함

- **BSON 포맷**
    - JSON의 인간 친화적인 형태를 유지하면서도, 컴퓨터가 빠르게 읽고 쓸 수 있도록 이진(Binary) 형태로 인코딩된 포맷
    - 타임스탬프나 바이너리 데이터 같은 추가적인 데이터 타입을 지원함

- **유연한 임베디드 구조**
    - RDBMS에서는 복잡한 관계를 표현하기 위해 여러 테이블을 만들고 JOIN을 수행해야 함
    - MongoDB는 하나의 도큐먼트 내부에 배열(Array)이나 하위 도큐먼트(Sub-document)를 중첩하여 관련 데이터를 한 번에 저장할 수 있음
        - 데이터 모델
            - RDBMS의 '테이블(Table)' 🡲 **'컬렉션(Collection)'**에 대응
            - RDBMS의 '행(Row)' 🡲 MongoDB의 **'도큐먼트(Document)'**에 해당
        - 스키마리스(Schema-less)
            - 고정된 스키마가 없음 🡲 같은 컬렉션 내에서도 각 도큐먼트가 서로 다른 필드 구조를 가질 수 있음

<div class="insert-image">
    <img src="/materials/datascience/images/S02-03-06-03_01-001.png" style="width: 90%;">
</div>

### 1.3 주요 특징

- **동적 스키마 (Schema-less)**
    - RDBMS처럼 미리 데이터 구조(칼럼 타입, 길이 등)를 정의하지 않고 데이터를 바로 삽입할 수 있음
    - 같은 컬렉션 안에 있더라도 각 도큐먼트는 서로 다른 필드를 가질 수 있음<br> 
        🡲 비즈니스 요구사항이나 데이터 포맷이 수시로 바뀌는 환경에 최적화
    - 데이터 요구사항이 변경되어도 복잡한 `ALTER TABLE` 과정 없이 즉시 필드를 추가하거나 수정할 수 있음

- **쉬운 수평 확장 (Horizontal Scalability)**
    - 데이터베이스 자체에서 강력하게 지원하는 **샤딩(Sharding)** 기술을 통해 데이터를 여러 서버에 분산 저장<br> 
        🡲 대용량 트래픽과 데이터를 효율적으로 처리
        - 샤딩(Sharding): 데이터의 양이 늘어나면 비싼 고성능 서버로 교체하는 대신, 저렴한 일반 서버를 여러 대 추가하여 성능을 높이는 기술

- **고가용성과 레플리카 셋 (Replica Set)**
    - 데이터를 자동으로 복제하여 여러 서버에 분산 저장하는 **레플리카 셋** 기능을 제공
    - 주 서버(Primary)가 다운되면 백업 서버(Secondary) 중 하나가 자동으로 주 서버로 승격되어 서비스가 중단 없이 유지됨

- **인덱싱과 강력한 쿼리 언어**
    - SQL 못지않게 복잡한 데이터 분석과 통계를 처리할 수 있는 집계 프레임워크(Aggregation Framework) 내장
    - 도큐먼트의 내장된 필드나 배열 속성에도 인덱스를 걸 수 있어 빠른 검색이 가능함
    - 지형 정보에 대한 쿼리를 지원하기 위한 인덱스인 위치 기반 쿼리 지원

### 1.4 장점과 단점

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">구분</th>
        <th style="width: 400px;">장점 (Pros)</th>
        <th style="width: 450px;">단점 (Cons)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">유연성</td>
            <td class="td-left">데이터 모델이 복잡하거나 자주 변하는 프로젝트에 최적화됨</td>
            <td class="td-left">데이터 구조가 너무 자유로워 관리가 소홀하면 데이터 일관성이 깨질 수 있음</td>
        </tr>
        <tr>
            <td class="td-rowheader">성능</td>
            <td class="td-left">복잡한 JOIN 연산 없이 한 번의 쿼리로 관련 데이터를 모두 가져올 수 있어 읽기 속도가 빠름</td>
            <td class="td-left">메모리(RAM) 사용량이 상대적으로 많으며, 데이터 중복 저장으로 인해 저장 공간 소비가 큼</td>
        </tr>
        <tr>
            <td class="td-rowheader">확장성</td>
            <td class="td-left">서버를 추가하는 것만으로 성능을 확장하기 용이함(Scale-out)</td>
            <td class="td-left">다중 문서 트랜잭션(Transaction) 지원이 강화되었으나, RDBMS에 비해서는 여전히 제약이 있음</td>
        </tr>
        <tr>
            <td class="td-rowheader">개발 속도</td>
            <td class="td-left">객체 지향 프로그래밍 코드와 데이터 구조(JSON)가 유사해 개발 생산성이 높음</td>
            <td class="td-left">복잡한 JOIN이 빈번하게 필요한 정형 데이터 처리에는 부적합할 수 있음</td>
        </tr>
    </tbody>    
</table>
</div>


### 1.5 주요 활용 사례

- MongoDB는 정형화되지 않은 대량의 데이터를 빠르게 처리해야 하는 분야에서 빛을 발함
    - **빅데이터 및 실시간 분석**
        - 로그 데이터, 소셜 미디어 피드, 센서 데이터 등 다양한 형태의 데이터를 수집하고 분석할 때 사용됨
    - **콘텐츠 관리 시스템(CMS)**
        - 기사, 비디오, 이미지 등 각기 다른 속성을 가진 콘텐츠를 하나의 시스템에서 유연하게 관리할 수 있음
    - **전자상거래 및 카탈로그**
        - 제품마다 옵션(색상, 사이즈, 사양 등)이 제각각인 상품 정보를 저장하는 데 유리함
    - **개인화 서비스**
        - 사용자별 프로필, 행동 패턴, 설정 값 등 개별화된 데이터를 저장하고 조회하는 데 효율적임
    - **모바일 및 게임**
        - 빠른 업데이트가 필요하고 트래픽 변동이 심한 게임 서버의 유저 정보 및 상태 저장소로 애용됨

<br>

> - MongoDB는 **"빠른 개발 속도"**와 **"대규모 확장성"**이 필요한 현대적 서비스에 최적화된 도구
> - 데이터의 구조가 명확하고 금융권처럼 엄격한 트랜잭션 일관성이 최우선이라면 RDBMS가 유리하겠지만,
> - 변화무쌍한 비즈니스 환경에서 데이터를 민첩하게 다루고 싶다면 MongoDB는 최고의 선택지 중 하나가 될 것
{: .common-quote}


## 2. MongoDB의 설치

- MongoDB Community Edition 설치를 기준으로 함

### 2.1 Windows 환경에서 설치

- 윈도우 환경에서는 GUI 설치 마법사(.msi)를 제공하므로 비교적 직관적으로 설치할 수 있음

- **단계 1: 설치 파일 다운로드**
    1. MongoDB 공식 다운로드 센터(MongoDB Community Server Download)에 접속
    2. **Version**은 최신 안정 버전을 선택, **Platform**은 `Windows`, **Package**는 `msi`를 선택한 후 **Download**를 클릭

- **단계 2: 설치 마법사 실행 및 설정**
    1. 다운로드한 `.msi` 파일을 실행
    2. **Setup Type** 단계에서 **Complete**를 선택
    3. **Service Configuration** 화면이 나오면 다음 설정을 확인
        - **Run service as Network Service user**
            - 기본값으로 두고 다음으로 넘어감 🡲 MongoDB를 윈도우 백그라운드 서비스로 등록하여 부팅 시 자동 실행되게 함
        - **Service Name**
            - **`MongoDB`**
        - **Data Directory**
            - 데이터가 저장될 경로
            - 기본값: `C:\Program Files\MongoDB\Server\[버전]\data\`
        - **Log Directory**
            - 로그가 저장될 경로
            - 기본값: `C:\Program Files\MongoDB\Server\[버전]\log\`

    4. **Install MongoDB Compass** 체크박스 🡲 체크 (MongoDB 전용 GUI 관리 도구)
    5. **Install**을 눌러서 설치 완료

- **단계 3: 환경 변수(Path) 설정 (선택사항이지만 권장)**
    - 명령 프롬프트(CMD)나 파워셸 어디서나 `mongosh`(MongoDB 쉘) 명령어를 실행할 수 있도록 설정

    1. `제어판` > `시스템 및 보안` > `시스템` > `고급 시스템 설정`으로 이동
    2. **환경 변수** 버튼 클릭
    3. '시스템 변수' 목록에서 `Path`를 찾아 선택하고 **편집** 클릭
    4. **새로 만들기**를 클릭하고 MongoDB 바이너리 폴더 경로 추가
        - 예시 경로: `C:\Program Files\MongoDB\Server\[설치버전]\bin`
    5. 확인을 눌러 모든 창 닫기


### 2.2 Linux (Ubuntu) 환경에서 설치

- 공식 APT 저장소(Repository)를 추가하여 패키지 관리자로 설치하는 것이 가장 안전하고 업데이트하기 좋음
- Ubuntu 22.04 LTS / 24.04 LTS 기준 설명

- **단계 1: 공개 키 가져오기 및 저장소 등록**
    1. 터미널을 열고 패키지의 무결성을 검증하기 위한 MongoDB 공식 GPG 키를 다운로드

        ```bash
        # 1. 필수 패키지 설치
        sudo apt-get install gnupg curl

        # 2. MongoDB 공개 GPG 키 가져오기
        curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        sudo gpg --o /usr/share/keyrings/mongodb-server-7.0.gpg \
        --dearmor
        ```

        - 주의: 사용하는 MongoDB 버전에 따라 URL의 `server-7.0` 부분을 해당 버전으로 변경해야 할 수 있음

    2. Ubuntu 버전에 맞는 리스트 파일을 생성하여 저장소 등록

        ```bash
        # Ubuntu 저장소 추가
        echo "deb [ png-keyring=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.drop/mongodb-org-7.0.list
        ```

- **단계 2: MongoDB 패키지 설치**
    - 저장소를 새로고침하고 MongoDB 메타 패키지 설치

        ```bash
        # 패키지 데이터베이스 업데이트
        sudo apt-get update

        # MongoDB Community Edition 전체 설치
        sudo apt-get install -y mongodb-org
        ```

- **단계 3: MongoDB 서비스 시작 및 상태 확인**
    - 리눅스에서는 설치 후 서비스를 수동으로 시작해 주어야 함 (`systemd` 관리 기준)

        ```bash
        # 1. MongoDB 서비스 시작
        sudo systemctl start mongod

        # 2. 부팅 시 자동 시작 등록
        sudo systemctl enable mongod

        # 3. 서비스 실행 상태 확인
        sudo systemctl status mongod
        ```

    - 상태를 확인했을 때 `active (running)` 상태가 표시되면 정상적으로 구동된 것


### 2.3 설치 완료 후 접속 테스트 (공통)

- 설치가 완료되었다면 MongoDB 쉘을 통해 데이터베이스에 정상적으로 연결되는지 확인
    1. 터미널(리눅스) 또는 명령 프롬프트(윈도우) 열기
    2. **`mongosh`** 명령어를 입력하고 엔터 (과거 버전의 경우 `mongo`)
    3. 성공적으로 접속되면 아래와 같은 쉘 프롬프트가 나타남

        ```javascript
        test> _
        ```

    4. 정상 작동 여부 테스트

        ```javascript
        // 현재 데이터베이스 목록 확인
        show dbs

        // 테스트 데이터 삽입
        db.testCollection.insertOne({ name: "MongoDB 설치 완료!" })

        // 삽입한 데이터 확인
        db.testCollection.find()
        ```

## 3. MongoDB 설치 후, 환경 설정

### 3.1 환경 설정 기초 정보

- MongoDB 설치 후, 외부 접속 허용, 보안 강화, 성능 최적화를 위해서 환경 설정 파일을 올바르게 수정해야 함
    - 환경 설정 파일 위치 (`mongod.conf`)

- MongoDB의 모든 핵심 설정은 `mongod.conf` 파일에서 관리함
    - **Windows:** `C:\Program Files\MongoDB\Server\[버전]\bin\mongod.cfg` (확장자가 `.cfg`)
    - **Linux (Ubuntu/CentOS):** `/etc/mongod.conf`

> ⚠️ **주의:**
>   - 설정 파일은 **YAML 포맷**을 사용하므로, 들여쓰기(띄어쓰기 공백 2칸)가 엄격하게 적용됨
>   - 탭(Tab) 문자를 사용하면 에러가 발생하여 서비스가 켜지지 않으니 주의
{: .common-quote}


### 3.2 주요 설정 항목 및 수정 방법

1. **네트워크 설정 (외부 접속 허용)**
    - 기본적으로 MongoDB는 보안을 위해 로컬 호스트(`127.0.0.1`)에서만 접속이 가능하도록 묶여 있음
    - 외부 WAS 서버나 GUI 툴(Compass)에서 접속하려면 이 부분을 수정해야 함

        ```yaml
        net:
        port: 27017             # 기본 포트 (보안을 위해 다른 포트로 변경 권장)
        bindIp: 127.0.0.1       # 로컬 접속만 허용 (기본값)
        ```

    - **[수정 후]**
        - 외부의 모든 IP에서 접속을 허용하려면 다음과 같이 변경
        - 또는 특정 서버의 IP를 쉼표로 구분하여 기록

            ```yaml
            net:
            port: 27017
            bindIp: 0.0.0.0         # 모든 IP로부터의 접속을 허용
            ```

2. **보안 설정 (인증 기능 활성화)**
    - `admin` 데이터베이스에 관리자 계정 생성
    - `bindIp`를 `0.0.0.0`으로 열었다면, **반드시 계정 인증 기능을 켜야 함**
    - 그렇지 않으면 전 세계 누구나 내 데이터베이스에 접근할 수 있게 됨

        ```yaml
        security:
        authorization: enabled  # 인증 기능 활성화 (기본값은 disabled 또는 주석 처리됨)
        ```

3. **데이터 및 로그 저장 경로 (Storage & SystemLog)**
    - 데이터가 쌓이는 물리적 위치와 시스템 로그 파일의 경로 지정
    - 용량이 큰 별도의 디스크 마운트 경로로 변경할 때 자주 수정함

        ```yaml
        storage:
        dbPath: /var/lib/mongodb       # 데이터 저장 경로 (리눅스 기준)
        journal:
            enabled: true

        systemLog:
        destination: file
        logPath: /var/log/mongodb/mongod.log    # 로그 저장 경로
        logAppend: true                         # 기존 로그 뒤에 이어서 기록
        ```

### 3.3 설정 적용 단계

- 인증 기능과 외부 접속을 안전하게 활성화하는 가장 표준적인 작업 순서

- **단계 1: 관리자 계정 생성 (인증을 켜기 전 먼저 수행)**
    1. 터미널이나 CMD에서 `mongosh`로 접속
    2. 최고 관리자 계정 생성

        ```javascript
        use admin

        db.createUser({
        user: "siteAdmin",
        pwd: "StrongPassword123!", // 강력한 비밀번호 설정
        roles: [ { role: "root", db: "admin" } ]
        })
        ```

- **단계 2: 설정 파일 수정**
    - 설정 파일(`mongod.conf` 또는 `mongod.cfg`)을 열어 네트워크와 보안 설정을 수정하고 저장

        ```yaml
        net:
        port: 27017
        bindIp: 0.0.0.0

        security:
        authorization: enabled
        ```

- **단계 3: MongoDB 서비스 재시작**
    - 수정한 내용을 반영하기 위해 서비스를 다시 시작
        - **Windows:** `작업 관리자` > `서비스` 탭 > `MongoDB` 우클릭 후 **재시작**
        - **Linux (Ubuntu):** 

            ```bash
            sudo systemctl restart mongod
            ```

- **단계 4: 인증 접속 테스트**
    - 서비스가 재시작된 후에는 기존 방식(`mongosh`)으로 접속하면 권한이 없어 아무 작업도 할 수 없음
    - 생성한 계정 정보를 명시하여 접속해야 함

        ```bash
        # 로컬에서 인증 접속
        mongosh -u siteAdmin -p --authenticationDatabase admin

        # 외부 서버에서 접속할 때 (아이피와 포트 명시)
        mongosh "mongodb://siteAdmin:StrongPassword123!@192.168.0.10:27017/admin"
        ```

### 3.4 리눅스 환경 추가 최적화 (선택 사항)

- 대규모 데이터를 다루는 프로덕션 리눅스 서버라면
    - OS 레벨에서 제한하는 자원 한계 값을 늘려주어야 대량의 커넥션을 매끄럽게 처리할 수 있음
        - **ulimit 설정:**
            - MongoDB는 파일과 커넥션을 대량으로 열기 때문에 `max open files` 제한을 늘려야 함
            - `/etc/security/limits.conf` 수정

                ```text
                mongod soft nofile 64000
                mongod hard nofile 64000
                ```

    - **THP(Transparent Huge Pages) 비활성화**
        - MongoDB 공식 문서에서는 메모리 관리 효율을 위해 운영체제의 THP 기능을 끄는 것을 강력히 권장함


### 3.5 방화벽(Firewall) 설정 방법

- 외부 접속(`bindIp: 0.0.0.0`)을 허용한 후, 특정 신뢰할 수 있는 서버(예: WAS 서버)만 데이터베이스에 접근할 수 있도록 제한
    - MongoDB의 `bindIp`를 `0.0.0.0`으로 변경했다면, 전 세계 모든 IP에 포트가 노출된 상태
    - 인증(`authorization: enabled`)과 더불어 **네트워크 레벨에서 방화벽으로 2차 잠금장치**를 해야 함

<br>

- **Linux 환경 방화벽 설정 (UFW / CentOS Firewalld)**
    - 리눅스 서버에서는 기본 방화벽 도구를 사용하여 **WAS(웹 애플리케이션 서버)의 IP만 27017 포트로 접근할 수 있도록** 허용

        - **Ubuntu 환경 (UFW 사용)**

            ```bash
            # 1. 방화벽 활성화 (기본적으로 모든 들어오는 연결 차단)
            sudo ufw enable

            # 2. SSH 접속 허용 (작업 중인 관리자 튕김 방지 필수!)
            sudo ufw allow 22/tcp

            # 3. [핵심] 특정 WAS 서버 IP(예: 192.168.1.50)만 MongoDB 포트(27017) 접근 허용
            sudo ufw allow from 192.168.1.50 to any port 27017 proto tcp

            # 4. 방화벽 설정 상태 및 규칙 적용 확인
            sudo ufw status verbose
            ```

        - **RedHat/CentOS 환경 (Firewalld 사용)**

            ```bash
            # 1. 방화벽 서비스 시작 및 활성화
            sudo systemctl start firewalld
            sudo systemctl enable firewalld

            # 2. 특정 IP(예: 192.168.1.50)가 27017 포트로 접근할 수 있는 규칙 추가
            sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="192.168.1.50" port port="27017" protocol="tcp" accept'

            # 3. 방화벽 규칙 재로드 (반드시 실행해야 적용됨)
            sudo firewall-cmd --reload

            # 4. 허용된 규칙 목록 확인
            sudo firewall-cmd --list-all
            ```

<br>

- **Windows 환경 방화벽 설정 (고급 보안이 설정된 Windows 방화벽)**
    - 윈도우 서버 환경에서는 GUI 또는 파워셸(PowerShell)을 통해 설정할 수 있음

    - **PowerShell 명령어로 한 번에 설정하기 (관리자 권한 실행)**

        ```powershell
        # 1. 기본적으로 들어오는 모든 27017 포트 차단 규칙 생성
        New-NetFirewallRule -DisplayName "Block MongoDB Public" -Direction Inbound -LocalPort 27017 -Protocol TCP -Action Block

        # 2. 특정 WAS 서버 IP(예: 192.168.1.50)만 통과시키는 허용 규칙 생성
        New-NetFirewallRule -DisplayName "Allow WAS to MongoDB" -Direction Inbound -LocalPort 27017 -Protocol TCP -RemoteAddress 192.168.1.50 -Action Allow
        ```

<br>

- **클라우드 환경 방화벽 설정 (AWS Security Group)**
    - 만약 AWS, NCP 같은 클라우드 환경에서 작업한다면, OS 내부 방화벽보다 클라우드 인프라 방화벽(보안 그룹)을 설정하는 것이 표준<br><br>
    - **인바운드 규칙(Inbound Rules) 설정 변경:**

        <div class="info-table">
        <table>
            <thead>
                <th style="width: 150px;">유형</th>
                <th style="width: 100px;">프로토콜</th>
                <th style="width: 100px;">포트 범위</th>
                <th style="width: 350px;">소스</th>
                <th style="width: 250px;">설명</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">사용자 지정 TCP</td>
                    <td>TCP</td>
                    <td>27017</td>
                    <td><span style="color: darkred;">192.168.1.50/32</span> (단일 WAS IP)</td>
                    <td>WAS 서버만 허용</td>
                </tr>
                <tr>
                    <td class="td-rowheader">사용자 지정 TCP</td>
                    <td>TCP</td>
                    <td>27017</td>
                    <td><span style="color: darkred;">sg-0123456789abcdef0</span> (WAS 보안그룹)</td>
                    <td>WAS의 보안그룹 통째로 허용</td>
                </tr>
            </tbody>    
        </table>
        </div>

        > 💡 **활용 팁** 
        > - 실무에서는 IP가 유동적으로 변할 수 있음
        >   - AWS 환경이라면 오른쪽 '소스' 항목에 IP 대신 **WAS 서버들이 속해있는 보안 그룹(Security Group) ID를 직접 지정**하는 방식이 훨씬 안전함
        {: .common-quote}


### 3.6 최종 연결 확인 테스트 (WAS 서버 시점)

- 방화벽 설정이 끝난 후, WAS 서버(192.168.1.50)로 이동하여 MongoDB 서버(예: 192.168.1.10)로 통신이 성공하는지 검증

    ```bash
    # WAS 서버 터미널에서 실행
    nc -zv 192.168.1.10 27017
    # 또는
    telnet 192.168.1.10 27017
    ```

    - **성공 시:**
        - `Connection to 192.168.1.10 27017 port [tcp/*] succeeded!` 메세지가 뜸

    - **실패 시 (허용되지 않은 다른 IP에서 시도 시):**
        - `Timeout` 혹은 `Connection refused`가 발생하며 접근이 차단됨