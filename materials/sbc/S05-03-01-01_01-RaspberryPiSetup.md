---
layout: page
title:  "라즈베리파이 OS 설치 및 개발 환경 설정"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S05-03-01-01_01-RaspberryPiSetup
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 라즈베리파이 OS 설치

- **먼저 Raspberry Pi Imager를 설치해야 하므로 본인이 사용하는 PC의 OS용을 다운로드 하여 설치**
    - [Raspberry Pi Imager 다운로드 (https://www.raspberrypi.com/software/)](https://www.raspberrypi.com/software/){: target="blank"}

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-001.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-002.png">
    </div>

- **Raspberry Pi Imager 설치**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-003.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-004.png">
    </div>

- **SD Card/Micro SD Card를 PC에 연결한 후 Raspberry Pi Imager 실행**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-005.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-006.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-007.png">
    </div>

- **OS와 SD Card를 선택한 후 톱니바퀴(설정) 버튼을 누름**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-008.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-009.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-010.png">
    </div>

- **SSH 사용 설정을 하고 내용 입력, 저장**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-011.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-012.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-013.png">
    </div>

- **쓰기를 선택하여 MicroSD에 저장**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-014.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-015.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-016.png"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-017.png">
    </div>

- **MicroSD카드를 라즈베리파이에 꽂은 후 전원 스위치를 켜고 부팅**

    <div class="insert-image" style="text-align: center;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-018.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-019.jpg">
    </div>


## 2. SSH를 이용한 원격 접속

- **SSH (Secure Shell)**
    - 보안이 지원되지 않는 네트워크를 통해 네트워크 서비스를 안전하게 운영하기 위한 암호화 네트워크 프로토콜
    - 원격으로 다른 시스템에 로그인 할 수 있는 대표적인 프로그램
    - 다중 접속을 허용하는 리눅스에서는 하나의 서버에서 여러 클라이언트에 접속하여 관리해야 하는 경우가 많은데 이런 경우에 유용하게 사용됨
    - 많이 사용되는 패키지: OpenSSH

- **SSH의 특징**
    - 암호화된 패킷을 전송함
    - 클라이언트와 서버라는 관계로 연결됨
    - sftp를 지원함

        ```bash
        sftp://[계정명]@[ip주소][파일경로]
        ```

    - 패스워드 없이 로그인이 가능함
    - scp(원격복사 기능)를 사용할 수 있음

        ```bash
        scp [복사될 파일명] [원격지 ip주소]:[파일]
        ```

- **SSH를 이용한 보안 원격 연결 설정 이전에 확인해야 할 조건들**
    - 원격 서버의 전원이 켜져 있고 안정적인 네트워크 연결이 있어야 함
    - 서버의 IP 주소가 필요함
    - 해당 IP를 통해서 원격 서버에 액세스할 수 있어야 함(ping 명령으로 테스트)
    - SSH 서버 및 클라이언트 도구는 각 서버 및 클라이언트 OS에 설치하여야 함
    - 원격 서버의 사용자 이름과 비밀번호가 필요함
    - 방화벽이 연결을 차단해서는 안됨(미리 열어줄 필요가 있음)

- **Wi-Fi 연결 후 종료(또는 시스템 업데이트)**
    - 비밀번호 설정 후 **→** Wi-Fi 연결 **→** 시스템 업데이트 **→** 종료

        <div class="insert-image" style="text-align: center;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-020.jpg"><br><br>
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-021.jpg"><br><br>
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-022.jpg">
        </div>

- **리눅스(Ubuntu 기준) 환경 설정**
    - OpenSSH 서버 설치

        ```bash
        sudo apt update		# 패키지 데이터베이스 업데이트
        sudo apt upgrade		# 설치된 패키지 업그레이드
        sudo apt install openssh-server	# OpenSSH 서버 설치
        ```

    - SSH 서버 구성

        ```bash
        sudo nano /etc/ssh/ssh_config	# 환경설정 파일 열기
        ```

        - ssh_config 파일을 위와 같이 수정함

            ```bash
            #   IdentityFile ~/.ssh/id_rsa
            #   IdentityFile ~/.ssh/id_dsa
            #   IdentityFile ~/.ssh/id_ecdsa
            #   IdentityFile ~/.ssh/id_ed25519
            Port 22
            MaxAuthTries 4
            #   Protocol 2
            #   Ciphers aes128-ctr,aes192-ctr,aes256-ctr,aes128-cbc,3bes-cbc
            #   MACs hmac-md5,hmac-sha1,umac-64@openssh.com
            #   EscapeChar ~
            #   Tunnel no
            #   TunnelDevice any:any
            #   PermitLocalCommand no
            #   VisualHostKey no
            #   ProxyCommand ssh –q –w %h:%p gateway.example.com
            #   RekeyLimit 1G 1h
                SendEnv LANG LC_*
                HashKnownHosts yes
                GSSAPIAuthentication yes
            ```

    - SSH 서비스 상태 확인

        ```bash
        sudo service ssh status	# SSH 서비스 활성화 상태 확인 → SSH 서비스가 실행 중인지 확인함
        sudo service ssh start	# SSH 서비스를 시작할 때
        sudo service ssh restart	# SSH 서비스를 재시작할 때
        sudo service ssh stop		# SSH 서비스를 중지할 때
        ```

- **클라이언트에서 리눅스 SSH에 접속**
    - 터미널 프로그램을 사용하여 SSH 서버에 접속함
        - Windows의 경우 PowerShell 을 사용하거나 적당한 터미널 프로그램을 설치하여 사용할 수 있음

    - 접속 명령

        ```bash
        ssh [원격 서버] [포트번호]
        ```

        또는

        ```bash
        ssh [사용자 이름]@[원격 서버] [포트번호]
        ```

        <div class="insert-image" style="text-align: center;">
            <img style="width: 700px;" src="/materials/sbc/images/S05-03-01-01_01-023.png">
        </div>

- **원격 접속 서버(Ubuntu 기준) 의 GUI환경을 사용하고 싶다면..**
    - 다양한 프로그램을 사용하여 원격 접속 서버의 GUI환경을 사용할 수 있음
        - RealVNC (VNC Viewer)
        - TightVNC
        - Xrdp 등
    - 어떤 프로그램을 사용해도 무방함. 자기에게 편한 것을 선택하여 사용

    - VNC Viewer를 이용한 예시 (원격서버는 Ubuntu 기준)
        - [VNC Viewer 다운로드 후 설치](https://www.realvnc.com/en/connect/download/viewer/)
        - VNC Viewer 실행 **→** IP 입력 **→** 접속

            <div class="insert-image" style="text-align: center;">
                <img style="width: 800px;" src="/materials/sbc/images/S05-03-01-01_01-024.jpg"><br><br>
                <img style="width: 800px;" src="/materials/sbc/images/S05-03-01-01_01-025.jpg">
            </div>

## 3. 원격 접속으로 라즈베리파이 사용하기

- **라즈베리파이를 사용하여 개발하는 방법**
    - 첫 번째 방법
        - 그냥 컴퓨터처럼 모니터, 키보드, 마우스를 연결하여 사용함
        - 자율주행 자동차의 경우 줄줄이 달고 다니기 힘듦
    - 두 번째 방법
        - 원격 GUI 환경으로 접속하여 사용함
        - 네트워크 환경이 필요함
        - 작업 시 속도가 조금 느림(실제 사용 시에는 문제 없음)

- **PuTTY 프로그램을 이용하여 SSH 프로토콜로 원격 접속**
    - Windows에서 주로 사용하는 방법
        - Linux, Mac에서는 자체 내장된 Terminal을 사용할 수 있음
        - Windows에도 Terminal이 제공되었으나 현재는 PowerShell과 거의 통합된 상태

    - [PuTTY 다운로드 및 설치 (https://apps.microsoft.com/detail/xpfnzksklbp7rj?hl=ko-KR&gl=US)](https://apps.microsoft.com/detail/xpfnzksklbp7rj?hl=ko-KR&gl=US){: target="blank"}

    - 원격 접속
        - 할당받은 IP 사용
        - 기본 설정된 ID/PW: pi / raspberry

            <div class="insert-image" style="text-align: center;">
                <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-026.png">
            </div>
