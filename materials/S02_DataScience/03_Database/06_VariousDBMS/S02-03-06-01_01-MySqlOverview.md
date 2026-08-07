---
layout: page
title:  "MySQL 개요 및 설치, 환경설정"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S02-03-06-01_01-MySqlOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. MySQL 개요

> - **웹의 표준 RDBMS:** 전 세계적으로 가장 인기 있는 오픈 소스  **관계형 데이터베이스 관리 시스템(RDBMS)**
> - **LAMP 스택의 주역:**
>   - Linux, Apache, MySQL, PHP/Python/Perl로 이어지는 현대 웹 아키텍처(LAMP 스택)의 핵심 구성 요소
>   - 페이스북, 유튜브, 위키피디아 등 수많은 대형 서비스의 기초 자산이 되고 있음
> - **강력한 성능과 쉬운 사용성:**
>   - 빠르고 신뢰할 수 있으며 가벼워, 소규모 스타트업부터 대규모 기업까지 폭넓게 선택하는 최적의 도구
>   - 다중 스레드, 다중 사용자 형식을 지원하며, 빠르고 유연하며 사용하기 쉬워 웹 애플리케이션 개발의 필수 도구로 자리 잡음
{: .common-quote}

- **주요 특징**

<div class="info-table">
<table>
    <thead>
        <th style="width: 170px;">구분</th>
        <th style="width: 200px;">주요 학습 항목</th>
        <th style="width: 580px;">상세 설명</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">정형화된 구조</td>
            <td>계층 구조 및 정규화</td>
            <td class="td-left">
                - 데이터베이스 - 테이블 - 행(Row) - 열(Column)로 이어지는 엄격한 구조를 가짐<br>
                - ERD 설계를 통해 데이터 중복을 최소화(정규화)하여 저장
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">강력한 스토리지 엔진</td>
            <td>플러그형 아키텍처</td>
            <td class="td-left">
                - 사용 목적에 따라 스토리지 엔진을 유연하게 교체할 수 있음<br>
                    - 예: 데이터 무결성을 위한 InnoDB, 빠른 읽기를 위한 MyISAM
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">데이터 무결성</td>
            <td>트랜잭션 및 ACID</td>
            <td class="td-left">
                - <b>InnoDB 엔진</b>을 기본으로 사용하여 트랜잭션(ACID 원칙)을 완벽하게 지원<br>
                - 대용량 데이터 처리 시에도 데이터 오류를 방지하고 신뢰성을 보장
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">오픈 소스 및 커뮤니티</td>
            <td>무료 및 빠른 문제 해결</td>
            <td class="td-left">
                - 커뮤니티 에디션은 누구나 무료로 사용할 수 있음<br>
                - 방대한 사용자 커뮤니티를 통해 문제 해결과 학습이 매우 빠름
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">호환성 및 확장성</td>
            <td>다양한 플랫폼 및 리플리케이션</td>
            <td class="td-left">
                - Windows, Linux, macOS 등 대부분의 플랫폼에서 구동됨<br>
                - <b>데이터베이스 복제(Replication)</b> 기능을 통해 대량의 트래픽을 분산 처리<br>
                - 고가용성을 확보할 수 있음
            </td>
        </tr>
    </tbody>
</table>
</div>


## 2. MySQL 설치 가이드

- 운영체제별 설치 방식이 상이함 🡲 사용 중인 환경에 맞춰 진행할 것

### 2.1 Windows 환경

1.  **설치 파일 다운로드:**
    - [MySQL 공식 홈페이지](https://dev.mysql.com/downloads/installer/)에서 'MySQL Installer for Windows'를 다운로드

2.  **설치 타입 선택:**
    - 일반적인 개발 용도 🡲 **'Developer Default'**
    - 필요한 항목만 설치 🡲 **'Custom'**

3.  **Check Requirements:**
    - 필요한 Visual Studio Redistributable 등의 패키지가 없다면 설치를 진행

4.  **Configuration (중요):**
    - **Type and Networking:** 'Development Computer'로 설정 (포트 번호는 기본값인 **3306**).
    - **Authentication Method:** 보안 강화를 위해 'Use Strong Password Encryption'을 권장
    - **Accounts and Roles:** **Root 계정의 비밀번호**를 설정 (반드시 기억할 것)

### 2.2 Linux (Ubuntu 기준) 환경

- 터미널을 통해 빠르고 간편하게 설치할 수 있음
    - Mac OS도 유사함

    ```bash
    # 패키지 목록 업데이트
    sudo apt update

    # MySQL 서버 설치
    sudo apt install mysql-server

    # 설치 확인 및 서비스 상태 체크
    sudo systemctl status mysql
    ```

    <div class="insert-image">
        <img src="/materials/datascience/images/S02-03-06-01_01-001.png" style="width: 90%;">
    </div>

## 3. 환경 설정 및 보안 최적화

- 설치 완료 후, 실제 사용을 위한 필수 설정 단계
- 리눅스를 기준으로 설명 (타 OS는 내용을 참고해서 설정)

### 3.1 초기 보안 설정 (Linux 필수)

- 설치 직후에는 비밀번호가 없거나 보안이 취약할 수 있음
- `sudo mysql_secure_installation`
    - MySQL 설치 직후 데이터베이스의 보안을 강화하기 위해 반드시 수행해야 하는 초기 보안 설정 스크립트
    - 비밀번호 복잡도 설정, 익명 사용자 제거, 원격 접속 차단 여부 등을 차례로 선택

    ```bash
    sudo mysql_secure_installation
    ```

- **단계별 설정 가이드 및 권장 사항**

    - **[1단계] VALIDATE PASSWORD COMPONENT (비밀번호 복잡도 검사 기능)**
        - **질문:** `Would you like to setup VALIDATE PASSWORD component?`
        - **선택 권장:** **Y (Yes)**
        - **이유:**
            - 외부 해킹 공격(무차별 대입 공격 등)으로부터 DB를 보호하기 위해 패스워드 정책을 활성화하는 것이 좋음
        - **추가 선택 (정책 수준):** `0 (Low)`, `1 (MEDIUM)`, `2 (STRONG)` 중 선택
        - **로컬 개발 환경:** `0 (Low)` 권장 (8자 이상이기만 하면 되므로 편하게 개발 가능)
        - **운영/배포 환경:** `1 (Medium)` 이상 권장 (숫자, 대소문자, 특수문자 포함 필수)

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-002.png" style="width: 90%;">
        </div>

    - **[2단계] Root 비밀번호 설정**
        - **질문:** `New password:` / `Re-enter new password:`
        - **설정:** 관리자(root) 계정으로 사용할 강력한 비밀번호를 입력
        - **주의:** 1단계에서 설정한 복잡도 기준에 맞지 않으면 통과되지 않고 다시 입력해야 함

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-003.png" style="width: 90%;">
        </div>

    - **[3단계] 익명 사용자 제거 (Anonymous Users)**
        - **질문:** `Remove anonymous users?`
        - **선택 권장:** **Y (Yes)**
        - **이유:**
            - MySQL은 기본적으로 로그인 없이 접속할 수 있는 익명 계정이 생성되어 있을 수 있음
            - 보안상 매우 위험하므로 무조건 제거해야 함

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-004.png" style="width: 90%;">
        </div>

    - **[4단계] Root 계정의 원격 접속 차단 (Disallow root login remotely)**
        - **질문:** `Disallow root login remotely?`
        - **선택 권장:** **Y (Yes)**
        - **이유:**
            - 최고 권한을 가진 `root` 계정이 외부 네트워크(원격)에서 바로 접속할 수 있게 두는 것은 해커들의 표적이 되기 쉬움
        - **실무 팁:**
            - 원격 접속은 일반 사용자 계정을 따로 생성하여 권한을 부여
            - `root`는 오직 서버 내부(`localhost`)에서만 접근하도록 차단하는 것이 표준

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-005.png" style="width: 90%;">
        </div>

    - **[5단계] 테스트 데이터베이스 제거 (Test Database)**
        - **질문:** `Remove test database and access to it?`
        - **선택 권장:** **Y (Yes)**
        - **이유:**
            - 누구나 접근 가능한 기본 `test` 데이터베이스와 관련 권한을 제거 🡲 잠재적인 보안 취약점을 없앰

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-006.png" style="width: 90%;">
        </div>

    - **[6단계] 권한 테이블 테이블 반영 (Reload Privilege Tables)**
        - **질문:** `Reload privilege tables now?`
        - **선택 권장:** **Y (Yes)**
        - **이유:**
            - 지금까지 설정한 모든 보안 규칙(비밀번호 변경, 계정 및 DB 삭제 등)을 MySQL 서버에 즉시 적용(Flush)

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-007.png" style="width: 90%;">
        </div>

- **한눈에 보는 요약 테이블**

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">단계</th>
        <th style="width: 350px;">주요 설정 내용</th>
        <th style="width: 200px;">권장 선택</th>
        <th style="width: 400px;">실무적 의의</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">1</td>
            <td class="td-left">패스워드 복잡도 검사 활성화</td>
            <td><span style="color: darkred;"><b>Y</b></span></td>
            <td class="td-left">취약한 비밀번호 사용 강제 차단</td>
        </tr>
        <tr>
            <td class="td-rowheader">2</td>
            <td class="td-left">Root 비밀번호 설정</td>
            <td><span style="color: darkred;"><b>입력</b></span></td>
            <td class="td-left">관리자 계정 보안 강화</td>
        </tr>
        <tr>
            <td class="td-rowheader">3</td>
            <td class="td-left">익명 사용자(Anonymous) 제거</td>
            <td><span style="color: darkred;"><b>Y</b></span></td>
            <td class="td-left">인증되지 않은 사용자의 접근 원천 차단</td>
        </tr>
        <tr>
            <td class="td-rowheader">4</td>
            <td class="td-left">Root 계정 원격 로그인 차단</td>
            <td><span style="color: darkred;"><b>Y</b></span></td>
            <td class="td-left">외부 해킹 공격 표적 방지 (로컬 접속만 허용)</td>
        </tr>
        <tr>
            <td class="td-rowheader">5</td>
            <td class="td-left">기본 Test DB 삭제</td>
            <td><span style="color: darkred;"><b>Y</b></span></td>
            <td class="td-left">불필요한 기본 자산 및 권한 정리</td>
        </tr>
        <tr>
            <td class="td-rowheader">6</td>
            <td class="td-left">변경된 권한 적용 (Reload)</td>
            <td><span style="color: darkred;"><b>Y</b></span></td>
            <td class="td-left">설정을 재시작 없이 즉시 반영</td>
        </tr>
    </tbody>
</table>
</div>



### 3.2 시스템 환경 변수 설정

- **Windows 기준**
    - 터미널(CMD/PowerShell)에서 `mysql` 명령어를 바로 사용하기 위해 필요함
    - **`내 PC 우클릭 > 속성 > 고급 시스템 설정 > 환경 변수`**
    - `System Variables` 중 **Path** 항목을 편집하여 MySQL 설치 폴더의 `bin` 경로를 추가
        - 예: `C:\Program Files\MySQL\MySQL Server 8.0\bin`

- **Linux 기준**
    - 터미널 어디서나 `mysql` 명령어를 바로 실행하기 위해 환경 변수(PATH) 설정을 진행
    - GUI 대신 **텍스트 기반의 설정 파일(`.bashrc` 또는 `.zshrc`)을 수정**하는 방식으로 등록

    1. **MySQL 실행 파일(`bin`) 경로 확인**
        - 보통 패키지 매니저(`apt`, `yum`)로 설치하면 자동으로 등록됨
        - 수동 설치했거나 경로를 명시해야 할 때 기본 경로는 다음과 같음
            - **일반적인 리눅스 기본 경로:** `/usr/bin/mysql` 또는 `/usr/local/mysql/bin`
            - 참고: 터미널에 `which mysql`을 입력하면 현재 설치된 MySQL의 실행 파일 경로를 확인할 수 있음

    2. **환경 변수 등록 및 반영 단계**
        - Linux에서는 사용자가 사용하는 쉘(Shell) 종류에 따라 설정 파일이 다름
        - 가장 보편적인 **Bash 쉘**을 기준으로 설명<br><br>
            - **[1단계] 설정 파일 열기**
                - 사용자의 홈 디렉토리에 있는 `.bashrc` 파일을 텍스트 에디터(nano 또는 vi)로 엶

                    ```bash
                    nano ~/.bashrc
                    ```
                - 만약 Mac 환경과 유사한 Zsh 쉘을 사용한다면 `nano ~/.zshrc`를 입력

            - **[2단계] PATH 경로 추가**
                - 파일의 **가장 최하단**으로 이동하여 아래 코드를 추가

                    ```bash
                    # MySQL PATH 설정
                    export PATH=$PATH:/usr/bin/mysql
                    ```

                    - 기존 윈도우의 Path 뒤에 세미콜론(`;`)을 붙이듯, 리눅스는 콜론(`:`)으로 경로를 구분함
                    - `$PATH`: 기존에 등록되어 있던 다른 환경 변수 경로들을 그대로 유지한다는 의미
                    - `:/usr/bin/mysql`: 새로 추가할 MySQL의 `bin` 폴더 경로

            - **[3단계] 변경 사항 적용 (★중요)**
                - 설정 파일을 저장하고 나온 뒤,
                - 터미널을 껐다 켜지 않고 **즉시 환경 변수를 반영**하기 위해 아래 명령어를 실행
                    - Windows에서 확인 버튼을 누르는 것과 같음

                    ```bash
                    source ~/.bashrc
                    ```

- **한눈에 보는 Windows vs Linux 비교 테이블**

<div class="info-table">
<table>
    <thead>
        <th style="width: 150px;">항목</th>
        <th style="width: 400px;">Windows (윈도우)</th>
        <th style="width: 400px;">Linux (리눅스)</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">설정 방식</td>
            <td class="td-left">GUI (고급 시스템 설정 🡲 환경 변수)</td>
            <td class="td-left">CLI (터미널에서 설정 파일 편집)</td>
        </tr>
        <tr>
            <td class="td-rowheader">대상 파일/위치</td>
            <td class="td-left">시스템 속성의 `Path` 변수</td>
            <td class="td-left">홈 디렉토리의 `~/.bashrc` 또는 `~/.zshrc`</td>
        </tr>
        <tr>
            <td class="td-rowheader">경로 구분자</td>
            <td class="td-left">세미콜론 (`;`)</td>
            <td class="td-left">콜론 (`:`)</td>
        </tr>
        <tr>
            <td class="td-rowheader">적용 명령어</td>
            <td class="td-left">(보통 터미널 재시작 필요)</td>
            <td class="td-left">`source ~/.bashrc` (즉시 반영)</td>
        </tr>
        <tr>
            <td class="td-rowheader">확인 명령어</td>
            <td class="td-left">`echo %PATH%`</td>
            <td class="td-left">`echo $PATH`</td>
        </tr>
    </tbody>    
</table>
</div>

> - **활용 팁**
>   - 리눅스 환경에 익숙하지 않은 경우,
>       - `source` 명령어를 빼먹어서 "환경 변수를 추가했는데도 `command not found`가 떠요!"라고 질문하는 경우가 많음
>       - 환경 변수 수정 후에는 **반드시 `source` 명령어로 동기화를 하거나 터미널 세션을 재연결**해야 함
{: .common-quote}


### 3.3 원격 접속 허용 설정

- 기본적으로 MySQL은 로컬 접속만 허용함 
- 외부 개발 도구(DBeaver, Workbench 등)나 응용 프로그램 서버에서 Linux에 설치된 MySQL에 접근하려면 🡲 아래의 3단계 설정을 순서대로 수행<br><br>

- **[1단계] MySQL 설정 파일 수정 (IP 바인딩 해제)**
    - 터미널에서 텍스트 에디터(`vi` 또는 `nano`)를 사용하여 MySQL 환경 설정 파일열기
    - 설정 파일 변경은 시스템 권한이 필요하므로 반드시 `sudo` 명령어를 앞에 붙여야 함

    - **설정 파일 열기 명령어:**

        ```bash
        sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
        ```

        - CentOS/RHEL 계열의 경우 주로 `/etc/my.cnf` 또는 `/etc/my.cnf.d/` 경로에 위치함

    - **수정 사항:**
        - 파일 내부에서 `bind-address` 항목을 찾아 아래와 같이 변경

            ```text
            # 기존 설정 (로컬 접속만 허용)
            bind-address = 127.0.0.1

            # 변경 후 설정 (모든 IP로부터의 원격 접속 허용)
            bind-address = 0.0.0.0
            ```

            - 만약 `mysqlx-bind-address = 127.0.0.1` 설정이 있다면 역시 `0.0.0.0`으로 수정할 것을 권장함

    - **의미:**
        - `127.0.0.1`은 서버 자신만 접속할 수 있는 루프백 IP
        - 이를 `0.0.0.0`으로 변경하면 서버가 가진 모든 네트워크 인터페이스를 열어 외부 접속을 받아들이겠다는 의미


- **[2단계] Linux 방화벽 개방 및 MySQL 서비스 재시작**
    - 설정 파일을 바꿨더라도 리눅스 자체 방화벽이 차단하고 있다면 외부에서 접속할 수 없음
    - 또한, 설정 파일의 변경 사항을 적용하기 위해 MySQL 서비스를 재시작해야 함

        - **MySQL 서비스 재시작:**

            ```bash
            sudo systemctl restart mysql
            ```

        - **리눅스 방화벽(UFW)에서 MySQL 포트(3306) 개방:**
        
            ```bash
            sudo ufw allow 3306/tcp
            sudo ufw reload
            ```
        - CentOS의 경우: `sudo firewall-cmd --zone=public --add-port=3306/tcp --permanent` 후 reload


- **[3단계] 외부 접속용 사용자 생성 및 권한 부여**
    - 설정과 방화벽이 해결되었다면, 
    - MySQL 내부로 접속(`mysql -u root -p`)하여 어떤 IP에 대해서도(`%`) 접속할 수 있는 전용 계정을 만들고 권한을 부여

        ```sql
        -- 1. 모든 IP('%')에서 접속 가능한 사용자 생성
        CREATE USER 'username'@'%' IDENTIFIED BY 'password';

        -- 2. 해당 사용자에게 모든 데이터베이스(*.*)에 대한 권한 부여
        GRANT ALL PRIVILEGES ON *.* TO 'username'@'%' WITH GRANT OPTION;

        -- 3. 변경된 권한 설정 즉시 반영
        FLUSH PRIVILEGES;
        ```

        > - **`mysql -u root -p`를 실행했을때, 비밀번호를 물어보고 비밀번호가 틀렸다고 한다면?**
        >   - **`sudo mysql -u root -p`**를 실행할 것
        >   - MySQL 8.0부터는 고정된 '기본 초기 비밀번호(예: 1234, 없음 등)'가 존재하지 않음
        {: .common-quote}

        <div class="insert-image">
            <img src="/materials/datascience/images/S02-03-06-01_01-008.png" style="width: 90%;">
        </div>


> - **활용 팁**
>   - 외부 접속 오류가 날 때 원인을 단계별로 찾는 체크리스트를 장표에 넣어주면 좋음
>       1. `bind-address`를 안 바꿨는가? 🡲 **MySQL 설정 문제**
>       2. `systemctl restart`를 안 했는가? 🡲 **서비스 미반영 문제**
>       3. 리눅스 `3306` 포트를 안 열었는가? 🡲 **방화벽 문제**
>       4. 계정 생성 시 `@'localhost'`로 만들었는가? 🡲 **MySQL 권한 문제**
{: .summary-quote}


## 4. 접속 테스트 및 확인

- 설정이 모두 끝났다면 터미널에서 접속을 시도하여 정상 작동을 확인

    ```bash
    # root 계정으로 접속
    mysql -u root -p
    ```

- 비밀번호 입력 후 `mysql>` 프롬프트가 나타나면 성공

> - **관리 도구 추천**
>   - 텍스트 기반의 CLI가 불편하다면 아래와 같은 GUI 도구를 함께 사용
>       - **MySQL Workbench:** 공식 관리 도구. 모델링부터 쿼리 작성까지 지원
>       - **DBeaver:** 다양한 DB를 한꺼번에 관리할 수 있는 오픈 소스 도구
>       - **HeidiSQL:** 가볍고 빠른 Windows 전용 관리 도구
{: .common-quote}