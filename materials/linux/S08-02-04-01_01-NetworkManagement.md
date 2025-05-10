---
layout: page
title:  "리눅스 명령어: 네트워크 관리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-04-01_01-NetworkManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. ip addr 또는 ifconfig
- 네트워크 인터페이스 정보 확인
- `ifconfig`는 더 이상 기본 명령어가 아닐 수 있으며, `net-tools` 패키지를 설치해야 할 수도 있음

    ```bash
    ip addr show
    ifconfig
    ```

## 2. netstat
- 네트워크 연결, 라우팅 테이블, 인터페이스 통계 등 표시<br><br>
    - `netstat -tuln`: 현재 열려 있는 TCP 및 UDP 포트 목록 표시
    - `netstat -rn`: 라우팅 테이블 표시
    
    ```bash
    netstat -tuln
    netstat -rn
    ```

## 3. ss
- netstat`의 최신 버전, 더 많은 정보를 효율적으로 보여줌<br><br>
    - `ss -tuln`: 현재 열려 있는 TCP 및 UDP 포트 목록 표시
    - `ss -rn`: 라우팅 테이블 표시

    ```bash
    ss -tuln
    ss -rn
    ```

## 4. ping
- 특정 호스트에 네트워크 연결이 가능한지 확인

    ```bash
    ping google.com
    ping 192.168.1.1
    ```

## 5. traceroute 또는 tracepath
- 특정 호스트까지의 네트워크 경로 추적

    ```bash
    traceroute google.com
    tracepath google.com
    ```

## 6. ssh
- Secure Shell의 약자, 원격 서버에 안전하게 접속

    ```bash
    ssh <사용자_이름>@<호스트_IP_또는_도메인>
    ssh -i <개인키_파일> <사용자_이름>@<호스트_IP_또는_도메인>
    ```

## 7. scp
- Secure Copy의 약자, 로컬과 원격 서버 간에 파일을 안전하게 복사<br><br>
    - 로컬 -> 원격: `scp <로컬_파일> <원격_사용자>@<원격_호스트>:<원격_경로>`
    - 원격 -> 로컬: `scp <원격_사용자>@<원격_호스트>:<원격_파일> <로컬_경로>`
    - 디렉토리 복사 시 `-r` 옵션 사용

    ```bash
    scp mylocal.txt user@192.168.1.10:/home/user/
    scp -r mydir user@example.com:/opt/backup/
    scp user@remote.server:/var/log/app.log /tmp/
    ```

## 8. firewall-cmd (CentOS/RHEL) 또는 ufw (Ubuntu)
- 방화벽 설정을 관리하는 명령어
- GCP에서는 네트워크 방화벽 규칙을 콘솔에서 설정하는 것이 일반적이지만, 인스턴스 내부 방화벽을 관리해야 할 경우 사용될 수 있음