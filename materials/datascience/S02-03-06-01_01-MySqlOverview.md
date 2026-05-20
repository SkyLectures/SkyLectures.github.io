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

- 전 세계적으로 가장 널리 사용되는 오픈 소스 **관계형 데이터베이스 관리 시스템(RDBMS)**
- 다중 스레드, 다중 사용자 형식을 지원하며, 빠르고 유연하며 사용하기 쉬워 웹 애플리케이션 개발의 필수 도구로 자리 잡음<br><br>

- **주요 특징**
    * **오픈 소스:**
        - 커뮤니티 에디션은 무료로 사용 가능
        - 거대한 사용자 커뮤니티를 통해 문제 해결이 빠름
    * **계층 구조:**
        - 데이터베이스 - 테이블 - 행(Row) - 열(Column)로 이어지는 정형화된 구조
    * **성능 및 신뢰성:**
        - 대용량 데이터 처리에 최적화
        - 트랜잭션 지원(InnoDB 엔진)을 통해 데이터의 무결성 보장
    * **다양한 운영체제 지원:**
        - Windows, Linux, macOS 등 대부분의 플랫폼에서 구동


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


## 3. 환경 설정 및 보안 최적화

- 설치 완료 후, 실제 사용을 위한 필수 설정 단계

### 3.1 초기 보안 설정 (Linux 필수)

- 설치 직후에는 비밀번호가 없거나 보안이 취약할 수 있음

```bash
sudo mysql_secure_installation
```

- 비밀번호 복잡도 설정, 익명 사용자 제거, 원격 접속 차단 여부 등을 차례로 선택

### 3.2 시스템 환경 변수 설정 (Windows)

- 터미널(CMD/PowerShell)에서 `mysql` 명령어를 바로 사용하기 위해 필요함

- **`내 PC 우클릭 > 속성 > 고급 시스템 설정 > 환경 변수`**
- `System Variables` 중 **Path** 항목을 편집하여 MySQL 설치 폴더의 `bin` 경로를 추가
    - 예: `C:\Program Files\MySQL\MySQL Server 8.0\bin`


### 3.3 원격 접속 허용 설정

- 기본적으로 MySQL은 로컬 접속만 허용함 🡲 외부에서 접속이 필요한 경우 설정 파일을 수정해야 함<br><br>

- **파일 경로:**
    - `/etc/mysql/mysql.conf.d/mysqld.cnf` (Linux 기준)

- **수정 사항:**
    - `bind-address = 127.0.0.1` 부분을 `0.0.0.0`으로 변경하거나 주석 처리

- **권한 부여:**
    - MySQL 접속 후 아래 쿼리를 실행

    ```sql
    CREATE USER 'username'@'%' IDENTIFIED BY 'password';
    GRANT ALL PRIVILEGES ON *.* TO 'username'@'%' WITH GRANT OPTION;
    FLUSH PRIVILEGES;
    ```


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
```
{: .common-quote}