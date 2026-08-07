---
layout: page
title:  "GCP 활용을 위한 기초 리눅스 명령어"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S04-02-04-01_01-GcpLinuxCommands
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

- GCP(Google Cloud Platform)를 효율적으로 활용하기 위해 익혀두면 좋은 기본적인 리눅스 명령어와 사용법 정리
- GCP의 Compute Engine 인스턴스는 대부분 Linux 환경으로 운영되므로 이러한 명령어들을 숙지하고 있으면 인스턴스 관리, 파일 관리, 네트워크 설정 등 다양한 작업을 터미널에서 편리하게 수행할 수 있음

## 1. 파일 및 디렉토리 관리

- <span style="color: green;">**`pwd`**</span>: 현재 작업 디렉토리(Present Working Directory)의 경로 출력

    ```bash
    pwd
    ```

- <span style="color: green;">**`ls`**</span>: 현재 디렉토리의 파일 및 하위 디렉토리 목록 표시
    - `ls -l`: 파일의 상세 정보(권한, 소유자, 크기, 수정 시간 등)를 함께 출력
    - `ls -a`: 숨김 파일(이름이 `.`으로 시작하는 파일)까지 모두 표시
    - `ls -h`: 파일 크기를 사람이 읽기 쉬운 단위(예: KB, MB, GB)로 표시
    - `ls -r`: 역순으로 정렬하여 표시
    - `ls -t`: 수정 시간 순으로 정렬하여 표시

    ```bash
    ls
    ls -l
    ls -a
    ls -lh
    ls -rt
    ```

- <span style="color: green;">**`cd`**</span>: Change Directory의 약자, 디렉토리를 변경
    - `cd <디렉토리_이름>`: 지정된 디렉토리로 이동
    - `cd ..`: 상위 디렉토리로 이동
    - `cd ~` 또는 `cd`: 현재 사용자의 홈 디렉토리로 이동
    - `cd -`: 이전 작업 디렉토리로 이동

    ```bash
    cd myproject
    cd ..
    cd
    cd -
    ```

- <span style="color: green;">**`mkdir`**</span>: Make Directory의 약자, 새로운 디렉토리 생성
    - `mkdir <디렉토리_이름>`: 지정된 이름의 디렉토리 생성
    - `mkdir -p <경로/디렉토리_이름>`: 지정된 경로에 없는 상위 디렉토리까지 모두 생성

    ```bash
    mkdir new_directory
    mkdir -p /home/user/nested/directory
    ```

- <span style="color: green;">**`rmdir`**</span>: Remove Directory의 약자, 비어 있는 디렉토리 삭제

    ```bash
    rmdir empty_directory
    ```

- <span style="color: green;">**`rm`**</span>: Remove의 약자, 파일 또는 디렉토리 삭제
    - `rm <파일_이름>`: 지정된 파일 삭제
    - `rm -r <디렉토리_이름>`: 지정된 디렉토리와 그 안의 모든 파일 및 하위 디렉토리 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)
    - `rm -f <파일_이름>`: 강제로 파일 삭제 (삭제 여부를 묻지 않음)
    - `rm -rf <디렉토리_이름>`: 강제로 디렉토리와 그 안의 모든 내용 삭제 (<span style="color: red;">**매우 주의**해서 사용할 것!</span>)

    ```bash
    rm myfile.txt
    rm -r old_directory
    rm -f important.log # 정말 필요한 경우에만 사용할 것!
    ```

- <span style="color: green;">**`cp`**</span>: Copy의 약자, 파일 또는 디렉토리 복사
    - `cp <원본_파일> <대상_파일>`: 원본 파일을 대상 파일 이름으로 복사
    - `cp <원본_파일> <대상_디렉토리>/`: 원본 파일을 대상 디렉토리 안으로 복사
    - `cp -r <원본_디렉토리> <대상_디렉토리>/`: 원본 디렉토리와 그 안의 모든 내용을 대상 디렉토리 안으로 복사

    ```bash
    cp original.txt backup.txt
    cp report.pdf /home/user/documents/
    cp -r source_folder /opt/backup/
    ```

- <span style="color: green;">**`mv`**</span>: Move의 약자, 파일 또는 디렉토리의 위치를 변경하거나 이름 변경
    - `mv <원본_파일> <대상_파일>`: 원본 파일의 이름을 대상 파일 이름으로 변경
    - `mv <원본_파일> <대상_디렉토리>/`: 원본 파일을 대상 디렉토리로 이동
    - `mv <원본_디렉토리> <새로운_디렉토리_이름>`: 원본 디렉토리의 이름을 새로운 디렉토리 이름으로 변경
    - `mv <원본_디렉토리> <대상_디렉토리>/`: 원본 디렉토리를 대상 디렉토리 안으로 이동

    ```bash
    mv old_name.txt new_name.txt
    mv data.csv /tmp/
    mv project_a project_b
    mv logs /var/archive/
    ```

- <span style="color: green;">**`cat`**</span>: 파일의 내용을 화면에 출력

    ```bash
    cat logfile.txt
    ```

- <span style="color: green;">**`less`**</span>: 파일 내용을 페이지 단위로 넘겨가며 볼 수 있음. 긴 파일 내용을 확인할 때 유용함 (Spacebar로 다음 페이지, `b`로 이전 페이지, `q`로 종료)

    ```bash
    less very_long_file.txt
    ```

- <span style="color: green;">**`head`**</span>: 파일의 처음 몇 줄(기본적으로 10줄)을 표시. `-n <숫자>` 옵션으로 줄 수를 지정 가능

    ```bash
    head config.ini
    head -n 20 error.log
    ```

- <span style="color: green;">**`tail`**</span>: 파일의 마지막 몇 줄을 표시. 실시간 로그를 확인할 때 유용함. `-f` 옵션 사용 시, 파일 내용이 추가될 때마다 계속해서 출력

    ```bash
    tail access.log
    tail -n 5 status.log
    tail -f application.log
    ```

- <span style="color: green;">**`touch`**</span>: 빈 파일을 생성하거나 파일의 접근 및 수정 시간을 업데이트

    ```bash
    touch new_file.txt
    touch existing_file.txt # 파일의 타임스탬프 업데이트
    ```

## 2. 사용자 및 권한 관리

- <span style="color: green;">**`sudo`**</span>: Superuser Do의 약자, 관리자 권한으로 명령어 실행. 대부분의 시스템 관리 작업에 필요함

    ```bash
    sudo apt update
    ```

- <span style="color: green;">**`chmod`**</span>: 파일 또는 디렉토리의 권한(mode) 변경
    - 숫자 모드 (예: 755, 644) 또는 심볼릭 모드 (예: `u+rwx`, `g+rx`, `o-w`)를 사용함
    - `u`: 사용자(owner), `g`: 그룹(group), `o`: 기타(others), `a`: 모두(all)
    - `+`: 권한 추가, `-`: 권한 제거, `=`: 권한 설정
    - `r`: 읽기, `w`: 쓰기, `x`: 실행

    ```bash
    chmod 755 script.sh # 사용자에게 읽기, 쓰기, 실행 권한; 그룹과 기타 사용자에게 읽기, 실행 권한 부여
    chmod u+x script.sh # 사용자에게 실행 권한 추가
    chmod g-w data.txt # 그룹에게 쓰기 권한 제거
    ```

- <span style="color: green;">**`chown`**</span>: 파일 또는 디렉토리의 소유자(owner)와 그룹(group) 변경
    - `chown <새로운_소유자> <파일_또는_디렉토리>`
    - `chown <새로운_소유자>:<새로운_그룹> <파일_또는_디렉토리>`

    ```bash
    sudo chown user1 myfile.txt
    sudo chown root:admin mydirectory
    ```

- <span style="color: green;">**`useradd`**</span>: 새로운 사용자 추가 (일반적으로 `sudo`와 함께 사용)

    ```bash
    sudo useradd newuser
    ```

- <span style="color: green;">**`passwd`**</span>: 사용자 계정의 비밀번호를 설정하거나 변경

    ```bash
    passwd # 현재 사용자 비밀번호 변경
    sudo passwd newuser # 'newuser'의 비밀번호 변경
    ```

- <span style="color: green;">**`userdel`**</span>: 사용자 삭제. `-r` 옵션 사용 시 사용자의 홈 디렉토리와 메일 스풀도 함께 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)

    ```bash
    sudo userdel olduser
    sudo userdel -r olduser
    ```

- <span style="color: green;">**`groupadd`**</span>: 새로운 그룹 추가

    ```bash
    sudo groupadd newgroup
    ```

- <span style="color: green;">**`groupdel`**</span>: 그룹 삭제

    ```bash
    sudo groupdel oldgroup
    ```

- <span style="color: green;">**`usermod`**</span>: 사용자 계정의 속성 수정 (그룹 변경, 로그인 셸 변경 등)
    - `usermod -aG <그룹_이름> <사용자_이름>`: 사용자를 추가 그룹에 추가 (`-a`는 append, `-G`는 groups)

    ```bash
    sudo usermod -aG developers myuser
    ```


## 3. 프로세스 관리

- <span style="color: green;">**`ps`**</span>: 현재 실행 중인 프로세스 목록 표시
    - `ps aux`: 모든 사용자의 자세한 프로세스 정보 표시
    - `ps -ef`: 시스템에서 실행 중인 모든 프로세스의 전체 정보 표시

    ```bash
    ps
    ps aux | less
    ps -ef | grep myapp
    ```

- <span style="color: green;">**`top`**</span> 또는 <span style="color: green;">**`htop`**</span>: 시스템의 실시간 프로세스 활동과 자원 사용률(CPU, 메모리 등) 표시
    - `htop`: 더 사용자 친화적인 인터페이스 제공, 설치가 필요할 수 있음 (`sudo apt install htop` 또는 `sudo yum install htop`)

    ```bash
    top
    htop
    ```

- <span style="color: green;">**`kill`**</span>: 실행 중인 프로세스에 시그널 전송. 일반적으로 프로세스 종료에 사용
    - `kill <PID>`: 지정된 프로세스 ID(PID)에 TERM(종료) 시그널 전송
    - `kill -9 <PID>` 또는 `kill -KILL <PID>`: 지정된 프로세스 강제 종료 (SIGKILL 시그널) → <span style="color: red;">**데이터 손실의 위험**이 있으므로 **최후의 수단**으로 사용할 것!</span>

    ```bash
    kill 1234
    kill -9 5678
    ```

- <span style="color: green;">**`pgrep`**</span>: 지정된 패턴과 일치하는 프로세스의 PID 검색

    ```bash
    pgrep nginx
    pgrep -u myuser java
    ```

- <span style="color: green;">**`pkill`**</span>: 이름 또는 다른 속성을 이용하여 프로세스 종료

    ```bash
    pkill -f myapp # 'myapp'을 포함하는 모든 프로세스 종료
    pkill -u otheruser firefox # 'otheruser'가 실행한 'firefox' 프로세스 종료
    ```

- <span style="color: green;">**`bg`**</span>: 백그라운드로 중단된 작업 다시 실행
- <span style="color: green;">**`fg`**</span>: 백그라운드 작업을 포그라운드로 가져옴
- <span style="color: green;">**`jobs`**</span>: 현재 백그라운드 작업 목록 표시

## 4. 네트워크 관리

- <span style="color: green;">**`ip addr`**</span> 또는 <span style="color: green;">**`ifconfig`**</span>: 네트워크 인터페이스 정보 확인
    - `ifconfig`는 더 이상 기본 명령어가 아닐 수 있으며, `net-tools` 패키지를 설치해야 할 수도 있음

    ```bash
    ip addr show
    ifconfig
    ```

- <span style="color: green;">**`netstat`**</span>: 네트워크 연결, 라우팅 테이블, 인터페이스 통계 등 표시
    - `netstat -tuln`: 현재 열려 있는 TCP 및 UDP 포트 목록 표시
    - `netstat -rn`: 라우팅 테이블 표시
    
    ```bash
    netstat -tuln
    netstat -rn
    ```

- <span style="color: green;">**`ss`**</span>: `netstat`의 최신 버전, 더 많은 정보를 효율적으로 보여줌
    - `ss -tuln`: 현재 열려 있는 TCP 및 UDP 포트 목록 표시
    - `ss -rn`: 라우팅 테이블 표시

    ```bash
    ss -tuln
    ss -rn
    ```

- <span style="color: green;">**`ping`**</span>: 특정 호스트에 네트워크 연결이 가능한지 확인

    ```bash
    ping google.com
    ping 192.168.1.1
    ```

- <span style="color: green;">**`traceroute`**</span> 또는 <span style="color: green;">**`tracepath`**</span>: 특정 호스트까지의 네트워크 경로 추적

    ```bash
    traceroute google.com
    tracepath google.com
    ```

- <span style="color: green;">**`ssh`**</span>: Secure Shell의 약자, 원격 서버에 안전하게 접속

    ```bash
    ssh <사용자_이름>@<호스트_IP_또는_도메인>
    ssh -i <개인키_파일> <사용자_이름>@<호스트_IP_또는_도메인>
    ```

- <span style="color: green;">**`scp`**</span>: Secure Copy의 약자, 로컬과 원격 서버 간에 파일을 안전하게 복사
    - 로컬 -> 원격: `scp <로컬_파일> <원격_사용자>@<원격_호스트>:<원격_경로>`
    - 원격 -> 로컬: `scp <원격_사용자>@<원격_호스트>:<원격_파일> <로컬_경로>`
    - 디렉토리 복사 시 `-r` 옵션 사용

    ```bash
    scp mylocal.txt user@192.168.1.10:/home/user/
    scp -r mydir user@example.com:/opt/backup/
    scp user@remote.server:/var/log/app.log /tmp/
    ```

- <span style="color: green;">**`firewall-cmd`**</span> (CentOS/RHEL) 또는 <span style="color: green;">**`ufw`**</span> (Ubuntu): 방화벽 설정을 관리하는 명령어
    - GCP에서는 네트워크 방화벽 규칙을 콘솔에서 설정하는 것이 일반적이지만, 인스턴스 내부 방화벽을 관리해야 할 경우 사용될 수 있음

## 5. 패키지 관리 (Linux 배포판에 따라 다름)

- <span style="color: green;">**Debian/Ubuntu 계열 (`apt`)**</span>:
    - `sudo apt update`: 패키지 목록 업데이트
    - `sudo apt upgrade`: 설치된 모든 패키지를 최신 버전으로 업그레이드
    - `sudo apt install <패키지_이름>`: 새로운 패키지 설치
    - `sudo apt remove <패키지_이름>`: 패키지 제거 (설정 파일은 남김)
    - `sudo apt purge <패키지_이름>`: 패키지와 설정 파일을 모두 제거
    - `sudo apt search <검색어>`: 패키지 목록에서 검색어를 포함하는 패키지를 검색
    - `sudo apt show <패키지_이름>`: 특정 패키지의 상세 정보 표시

- <span style="color: green;">**CentOS/RHEL 계열 (`yum` 또는 `dnf`)**</span>:
    - `sudo yum update` 또는 `sudo dnf update`: 패키지 목록을 업데이트하고 설치된 패키지를 업그레이드
    - `sudo yum install <패키지_이름>` 또는 `sudo dnf install <패키지_이름>`: 새로운 패키지 설치
    - `sudo yum remove <패키지_이름>` 또는 `sudo dnf remove <패키지_이름>`: 패키지 제거
    - `sudo yum search <검색어>` 또는 `sudo dnf search <검색어>`: 패키지 목록에서 검색어를 포함하는 패키지를 검색
    - `sudo yum info <패키지_이름>` 또는 `sudo dnf info <패키지_이름>`: 특정 패키지의 상세 정보 표시

## 6. 텍스트 처리 및 검색

- <span style="color: green;">**`grep`**</span>: 주어진 패턴과 일치하는 텍스트를 파일에서 검색
    - `grep <패턴> <파일>`: 파일에서 패턴을 검색
    - `grep -i <패턴> <파일>`: 대소문자를 구분하지 않고 검색합
    - `grep -r <패턴> <디렉토리>`: 디렉토리 및 하위 디렉토리의 모든 파일에서 패턴을 검색
    - `grep -n <패턴> <파일>`: 패턴이 발견된 줄 번호를 함께 출력
    - `grep -v <패턴> <파일>`: 패턴과 일치하지 않는 줄만 출력

    ```bash
    grep "error" logfile.txt
    grep -i "warning" access.log
    grep -r "config" /etc/
    ```

- <span style="color: green;">**`sed`**</span>: 스트림 편집기(Stream EDitor), 텍스트 스트림에 대한 기본적인 텍스트 변환 수행. 파일의 내용을 변경하거나, 찾아서 바꾸는 작업 등에 사용

    ```bash
    sed 's/old_string/new_string/g' input.txt > output.txt # 'input.txt'에서 'old_string'을 'new_string'으로 모두 바꾸어 'output.txt'에 저장
    sed -i 's/original/modified/g