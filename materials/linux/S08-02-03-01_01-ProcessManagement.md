---
layout: page
title:  "리눅스 명령어: 프로세스 관리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-03-01_01-ProcessManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. ps
- 현재 실행 중인 프로세스 목록 표시<br><br>
    - `ps aux`: 모든 사용자의 자세한 프로세스 정보 표시
    - `ps -ef`: 시스템에서 실행 중인 모든 프로세스의 전체 정보 표시

    ```bash
    ps
    ps aux | less
    ps -ef | grep myapp
    ```

## 2. top 또는 htop
- 시스템의 실시간 프로세스 활동과 자원 사용률(CPU, 메모리 등) 표시<br><br>
    - `htop`: 더 사용자 친화적인 인터페이스 제공, 설치가 필요할 수 있음
        - `sudo apt install htop` 또는 `sudo yum install htop`

    ```bash
    top
    htop
    ```

## 3. kill
- 실행 중인 프로세스에 시그널 전송. 일반적으로 프로세스 종료에 사용<br><br>
    - `kill <PID>`: 지정된 프로세스 ID(PID)에 TERM(종료) 시그널 전송
    - `kill -9 <PID>` 또는 `kill -KILL <PID>`: 지정된 프로세스 강제 종료 (SIGKILL 시그널)<br> → <span style="color: red;">**데이터 손실의 위험**이 있으므로 **최후의 수단**으로 사용할 것!</span>

    ```bash
    kill 1234
    kill -9 5678
    ```

## 4. pgrep
- 지정된 패턴과 일치하는 프로세스의 PID 검색

    ```bash
    pgrep nginx
    pgrep -u myuser java
    ```

## 5. pkill
- 이름 또는 다른 속성을 이용하여 프로세스 종료

    ```bash
    pkill -f myapp # 'myapp'을 포함하는 모든 프로세스 종료
    pkill -u otheruser firefox # 'otheruser'가 실행한 'firefox' 프로세스 종료
    ```

## 6. bg
- 백그라운드로 중단된 작업 다시 실행

## 7. fg
- 백그라운드 작업을 포그라운드로 가져옴

## 8. jobs
- 현재 백그라운드 작업 목록 표시
