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

    <div class="insert-image" style="text-align: left;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-001.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-002.png">
    </div>

- **Raspberry Pi Imager 설치**

    <div class="insert-image" style="text-align: left;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-003.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-004.png">
    </div>

- **SD Card/Micro SD Card를 PC에 연결한 후 Raspberry Pi Imager 실행**
    - 실행화면
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-005.png"><br><br>
        </div>
    - 장치선택
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-006.png"><br><br>
        </div>
    - 운영체제 선택
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-007.png"><br><br>
        </div>
    - 저장소 선택
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-008.png"><br><br>
        </div>
    - 다음 버튼
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-009.png">
        </div>

- **OS 커스터마이징 설정**
    - OS 커스터마이징 설정 확인 화면
        - 해당 설정은 OS 설치가 완료된 후에도 할 수 있으므로 건너뛰어도 무방함
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-010.png">
        </div>
    - OS 커스터마이징 설정(자신의 환경에 맞게 설정함)
        - ID / PW 기본 값: <b><span style="color: green;">pi / raspberry</span></b>
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px; border: 1px solid gray;" src="/materials/sbc/images/S05-03-01-01_01-011.png">
        </div>
    - OS 커스터마이징 적용여부 확인 화면
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-012.png">
        </div>
    - 이미지 쓰기 전 경고 화면
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-013.png">
        </div>

- **MicroSD카드를 라즈베리파이에 꽂은 후 전원 스위치를 켜고 부팅**

    <div class="insert-image" style="text-align: left;">
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-014.jpg"><br><br>
        <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-015.jpg">
    </div>


## 2. SSH / VNC를 이용한 원격 접속

### 2.1 SSH

- **SSH (Secure Shell) 개요**
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


### 2.2 VNC

- **VNC (Virtual Network Computing, 가상 네트워크 컴퓨팅)**
    - 네트워크를 통해 다른 컴퓨터의 그래픽 데스크톱(GUI) 환경을 원격으로 제어할 수 있게 해주는 '가상 화면 공유 및 제어' 시스템
        - 라즈베리파이의 데스크톱 화면을 PC에서 보고 조작할 수 있도록 해주는 시스템

- **핵심 기능**
    - 원격 제어
        - 마치 직접 모니터, 키보드, 마우스를 연결한 것처럼 멀리 떨어진 컴퓨터의 화면을 보고 조작할 수 있습니다.
    - 클라이언트-서버 모델
        - VNC Server
            - 원격으로 제어될 컴퓨터(예: 라즈베리파이)에 설치됨
            - 설치된 컴퓨터의 화면을 캡처하여 클라이언트로 전송하고, 클라이언트로부터 받은 키보드/마우스 입력을 해당 컴퓨터에 전달함
        - VNC Viewer (클라이언트)
            - 원격으로 제어하는 컴퓨터(예: 사용자의 PC)에 설치됨
            - 서버가 보내는 화면 데이터를 받아 표시하고, 사용자의 입력(키보드/마우스)을 서버로 전송함
    - 크로스 플랫폼 호환성
        - 리눅스, 윈도우즈, macOS 등 다양한 운영체제와 장치(PC, 스마트폰, 태블릿 등) 간에 원격 접속 가능

- **주요 목적**
    - 원격 작업 및 유지 보수
    - 헤드리스(모니터 없는) 시스템 관리
    - 문제 해결 지원
    - 라즈베리파이를 원격으로 사용해야 할 때 유용함

- **VNC의 장점**
    - 그래픽 환경 지원
        - CLI(명령줄 인터페이스)뿐만 아니라 GUI 기반의 모든 작업을 원격으로 수행할 수 있음
    - 쉬운 접근성
        - 기본적인 네트워크 지식만 있으면 쉽게 설정하고 사용할 수 있음

- **고려 사항**
    - 보안
        - VNC 연결 시 반드시 적절한 보안 조치(비밀번호 설정, VPN 사용 등)를 적용해야 함
    - 성능
        - 네트워크 환경에 따라 화면 전송 속도나 반응 속도가 저하될 수 있음

### 2.3 환경 설정

- **SSH / VNC 허용 설정**
    - Menu(좌상단 라즈베리파이 아이콘) - Preference - Control Center
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-016.png">
        </div>
    - SSH, VNC 허용
        <div class="insert-image" style="text-align: left;">
            <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-017.png">
        </div>

- **OpenSSH 서버 설치**

    ```bash
    sudo apt update		# 패키지 데이터베이스 업데이트
    sudo apt upgrade		# 설치된 패키지 업그레이드
    sudo apt install openssh-server	# OpenSSH 서버 설치
    ```

- **SSH 서버 구성**

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

- **SSH 서비스 상태 확인**

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
            <img style="width: 700px;" src="/materials/sbc/images/S05-03-01-01_01-018.png">
        </div>

### 2.4 원격 접속으로 라즈베리파이 사용하기

- **라즈베리파이를 사용하여 개발하는 방법**
    - 첫 번째 방법
        - 그냥 컴퓨터처럼 모니터, 키보드, 마우스를 연결하여 사용함
        - 자율주행 자동차의 경우 줄줄이 달고 다니기 힘듦
    - 두 번째 방법
        - **SSH를 이용**하여 원격 접속하거나 **원격 GUI 도구(VNC 등)를 이용**하여 원격 접속
        - 네트워크 환경이 필요함
        - 작업 시 속도가 조금 느림(실제 사용 시에는 문제 없음)

- **SSH 프로토콜로 원격 접속**
    - PuTTY 프로그램 이용
        - Windows에서 주로 사용하는 방법
            - Linux, Mac에서는 자체 내장된 Terminal을 사용할 수 있음
            - Windows에도 Terminal이 제공되었으나 현재는 PowerShell과 거의 통합된 상태

    - [PuTTY 다운로드 및 설치 (https://apps.microsoft.com/detail/xpfnzksklbp7rj?hl=ko-KR&gl=US)](https://apps.microsoft.com/detail/xpfnzksklbp7rj?hl=ko-KR&gl=US){: target="blank"}

    - 원격 접속
        - 할당받은 IP 사용
        - 기본 설정된 ID/PW: pi / raspberry

            <div class="insert-image" style="text-align: center;">
                <img style="width: 600px;" src="/materials/sbc/images/S05-03-01-01_01-019.png">
            </div>

- **VNC를 이용하여 원격 접속**
    - 다양한 프로그램을 사용하여 원격 접속 서버의 GUI환경을 사용할 수 있음
        - RealVNC (VNC Viewer), NoMachine, TightVNC, Xrdp 등

    - 어떤 프로그램을 사용해도 무방함. 자기에게 편한 것을 선택하여 사용

    - RealVNC 사용의 예
        - VNC 서버
            - 일반적으로 라즈베리파이에 기본적으로 설치되어 있음
            - 설치되어 있지 않다면 RealVNC 사이트에서 라즈베리파이용 서버를 다운로드하여 설치
                - [RealVNC 라즈베리파이용 서버 다운로드](https://www.realvnc.com/en/connect/download/vnc/?lai_vid=wARqGjMagFP9n&lai_sr=0-4&lai_sl=l){: target="blank"}
                - ARM64 버전을 선택함
            - 설치하면 자동으로 실행되어 작동함
            - 상태 확인
                - 라즈베리파이 버전에 따라 vncserver-virtuald 또는 vncserver-x11-stub 중 선택
                - 라즈베리파이5의 경우 vncserver-virtuald

                ```bash
                sudo systemctl status vncserver-virtuald.service
                ```

            - 만약 비활성화 상태라면 수동 실행
            
                ```bash
                sudo systemctl start vncserver-virtuald.service
                ```

            - 부팅 시마다 자동 실행하도록 설정
            
                ```bash
                sudo systemctl enable vncserver-virtuald.service
                ```


        - VNC Viewer(우분투 기준)
            - 각자 시스템에 맞게 선택할 것
            - 우분투 계열의 경우, DEB X64 버전을 선택함
                - [VNC Viewer 다운로드](https://www.realvnc.com/en/connect/download/viewer/?lai_vid=wARqGjMagFP9n&lai_sr=5-9&lai_sl=l){: target="blank"}
            - 설치 후, VNC Viewer 실행 ➜ IP 입력 ➜ 접속

                <div class="insert-image" style="text-align: center;">
                    <img style="width: 800px;" src="/materials/sbc/images/S05-03-01-01_01-020.png"><br><br>
                    <img style="width: 800px;" src="/materials/sbc/images/S05-03-01-01_01-021.png">
                </div>
