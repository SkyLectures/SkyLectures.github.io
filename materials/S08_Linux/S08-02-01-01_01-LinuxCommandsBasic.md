---
layout: page
title:  "리눅스 기초 명령어"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-01-01_01-LinuxCommandsBasic
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

> - 단순히 명령어를 외우는 것이 아니라, **'데이터가 흐르는 통로'**로서의 리눅스 구조를 이해하는 데 집중
{: .summary-quote}

## 1. 입문용 기초 커맨드

### 1.1 파일 및 디렉토리 관리

- **pwd**
    - 현재 작업 디렉토리(Present Working Directory)의 경로 출력

        ```bash
        pwd
        ```

- **ls**
    - 현재 디렉토리의 파일 및 하위 디렉토리 목록 표시<br><br>
        - **`ls -l`** : 파일의 상세 정보(권한, 소유자, 크기, 수정 시간 등)를 함께 출력
        - **`ls -a`** : 숨김 파일(이름이 **`.`**으로 시작하는 파일)까지 모두 표시
        - **`ls -h`** : 파일 크기를 사람이 읽기 쉬운 단위(예: KB, MB, GB)로 표시
        - **`ls -r`** : 역순으로 정렬하여 표시
        - **`ls -t`** : 수정 시간 순으로 정렬하여 표시

        ```bash
        ls
        ls -l
        ls -a
        ls -lh
        ls -rt
        ```

- **cd**
    - Change Directory의 약자, 디렉토리를 변경<br><br>
        - **`cd <디렉토리_이름>`** : 지정된 디렉토리로 이동
        - **`cd ..`** : 상위 디렉토리로 이동
        - **`cd ~`** 또는 **`cd`** : 현재 사용자의 홈 디렉토리로 이동
        - **`cd -`** : 이전 작업 디렉토리로 이동

        ```bash
        cd myproject
        cd ..
        cd
        cd -
        ```

- **mkdir**
    - Make Directory의 약자, 새로운 디렉토리 생성<br><br>
        - **`mkdir <디렉토리_이름>`** : 지정된 이름의 디렉토리 생성
        - **`mkdir -p <경로/디렉토리_이름>`** : 지정된 경로에 없는 상위 디렉토리까지 모두 생성

        ```bash
        mkdir new_directory
        mkdir -p /home/user/nested/directory
        ```

- **rmdir**
    - Remove Directory의 약자, 비어 있는 디렉토리 삭제

        ```bash
        rmdir empty_directory
        ```

- **rm**
    - Remove의 약자, 파일 또는 디렉토리 삭제<br><br>
        - **`rm <파일_이름>`** : 지정된 파일 삭제
        - **`rm -r <디렉토리_이름>`** : 지정된 디렉토리와 그 안의 모든 파일 및 하위 디렉토리 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)
        - **`rm -f <파일_이름>`** : 강제로 파일 삭제 (삭제 여부를 묻지 않음)
        - **`rm -rf <디렉토리_이름>`** : 강제로 디렉토리와 그 안의 모든 내용 삭제 (<span style="color: red;">**매우 주의**해서 사용할 것!</span>)

        ```bash
        rm myfile.txt
        rm -r old_directory
        rm -f important.log # 정말 필요한 경우에만 사용할 것!
        ```

- **cp**
    - Copy의 약자, 파일 또는 디렉토리 복사<br><br>
        - **`cp <원본_파일> <대상_파일>`** : 원본 파일을 대상 파일 이름으로 복사
        - **`cp <원본_파일> <대상_디렉토리>/`** : 원본 파일을 대상 디렉토리 안으로 복사
        - **`cp -r <원본_디렉토리> <대상_디렉토리>/`** : 원본 디렉토리와 그 안의 모든 내용을 대상 디렉토리 안으로 복사

        ```bash
        cp original.txt backup.txt
        cp report.pdf /home/user/documents/
        cp -r source_folder /opt/backup/
        ```

- **mv**
    - Move의 약자, 파일 또는 디렉토리의 위치를 변경하거나 이름 변경<br><br>
        - **`mv <원본_파일> <대상_파일>`** : 원본 파일의 이름을 대상 파일 이름으로 변경
        - **`mv <원본_파일> <대상_디렉토리>/`** : 원본 파일을 대상 디렉토리로 이동
        - **`mv <원본_디렉토리> <새로운_디렉토리_이름>`** : 원본 디렉토리의 이름을 새로운 디렉토리 이름으로 변경
        - **`mv <원본_디렉토리> <대상_디렉토리>/`** : 원본 디렉토리를 대상 디렉토리 안으로 이동

        ```bash
        mv old_name.txt new_name.txt
        mv data.csv /tmp/
        mv project_a project_b
        mv logs /var/archive/
        ```

- **cat**
    - 파일의 내용을 화면에 출력

        ```bash
        cat logfile.txt
        ```

- **less**
    - 파일 내용을 페이지 단위로 넘겨가며 볼 수 있음
    - 긴 파일 내용을 확인할 때 유용함 (Spacebar로 다음 페이지, `b`로 이전 페이지, `q`로 종료)

        ```bash
        less very_long_file.txt
        ```

- **head**
    - 파일의 처음 몇 줄(기본적으로 10줄)을 표시
    - `-n <숫자>` 옵션으로 줄 수를 지정 가능

        ```bash
        head config.ini
        head -n 20 error.log
        ```

- **tail**
    - 파일의 마지막 몇 줄을 표시
    - 실시간 로그를 확인할 때 유용함
    - `-f` 옵션 사용 시, 파일 내용이 추가될 때마다 계속해서 출력

        ```bash
        tail access.log
        tail -n 5 status.log
        tail -f application.log
        ```

- **touch**
    - 빈 파일을 생성하거나 파일의 접근 및 수정 시간을 업데이트

        ```bash
        touch new_file.txt
        touch existing_file.txt # 파일의 타임스탬프 업데이트
        ```

### 1.2 사용자 및 권한 관리

- **sudo**
    - Superuser Do의 약자, 관리자 권한으로 명령어 실행
    - 대부분의 시스템 관리 작업에 필요함

        ```bash
        sudo apt update
        ```

- **chmod**
    - 파일 또는 디렉토리의 권한(mode) 변경<br><br>
        - 숫자 모드 (예: 755, 644) 또는 심볼릭 모드 (예: **`u+rwx`**, **`g+rx`**, **`o-w`**)를 사용함
        - **`u`** : 사용자(owner), **`g`** : 그룹(group), **`o`** : 기타(others), **`a`** : 모두(all)
        - **`+`** : 권한 추가, **`-`** : 권한 제거,** `=`** : 권한 설정
        - **`r`** : 읽기, **`w`** : 쓰기, **`x`** : 실행

        ```bash
        chmod 755 script.sh # 사용자에게 읽기, 쓰기, 실행 권한; 그룹과 기타 사용자에게 읽기, 실행 권한 부여
        chmod u+x script.sh # 사용자에게 실행 권한 추가
        chmod g-w data.txt # 그룹에게 쓰기 권한 제거
        ```

- **chown**
    - 파일 또는 디렉토리의 소유자(owner)와 그룹(group) 변경<br><br>
        - **`chown <새로운_소유자> <파일_또는_디렉토리>`**
        - **`chown <새로운_소유자>:<새로운_그룹> <파일_또는_디렉토리>`**

        ```bash
        sudo chown user1 myfile.txt
        sudo chown root:admin mydirectory
        ```

- **useradd**
    - 새로운 사용자 추가 (일반적으로 **`sudo`**와 함께 사용)

        ```bash
        sudo useradd newuser
        ```

- **passwd**
    - 사용자 계정의 비밀번호를 설정하거나 변경

        ```bash
        passwd # 현재 사용자 비밀번호 변경
        sudo passwd newuser # 'newuser'의 비밀번호 변경
        ```

- **userdel**
    - 사용자 삭제
    - **`-r`** 옵션 사용 시 사용자의 홈 디렉토리와 메일 스풀도 함께 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)

        ```bash
        sudo userdel olduser
        sudo userdel -r olduser
        ```

- **groupadd**
    - 새로운 그룹 추가

        ```bash
        sudo groupadd newgroup
        ```

- **groupdel**
    - 그룹 삭제

        ```bash
        sudo groupdel oldgroup
        ```

- **usermod**
    - 사용자 계정의 속성 수정 (그룹 변경, 로그인 셸 변경 등)<br><br>
        - **`usermod -aG <그룹_이름> <사용자_이름>`** : 사용자를 추가 그룹에 추가 (`-a`는 append, `-G`는 groups)

        ```bash
        sudo usermod -aG developers myuser
        ```


### 1.3 프로세스 관리

- **ps**
    - 현재 실행 중인 프로세스 목록 표시<br><br>
        - **`ps aux`** : 모든 사용자의 자세한 프로세스 정보 표시
        - **`ps -ef`** : 시스템에서 실행 중인 모든 프로세스의 전체 정보 표시

        ```bash
        ps
        ps aux | less
        ps -ef | grep myapp
        ```

- **top 또는 htop**
    - 시스템의 실시간 프로세스 활동과 자원 사용률(CPU, 메모리 등) 표시<br><br>
        - **`htop`** : 더 사용자 친화적인 인터페이스 제공, 설치가 필요할 수 있음
            - **`sudo apt install htop`** 또는 **`sudo yum install htop`**

        ```bash
        top
        htop
        ```

- **kill**
    - 실행 중인 프로세스에 시그널 전송. 일반적으로 프로세스 종료에 사용<br><br>
        - **`kill <PID>`** : 지정된 프로세스 ID(PID)에 TERM(종료) 시그널 전송
        - **`kill -9 <PID>`** 또는 **`kill -KILL <PID>`** : 지정된 프로세스 강제 종료 (SIGKILL 시그널)<br> → <span style="color: red;">**데이터 손실의 위험**이 있으므로 **최후의 수단**으로 사용할 것!</span>

        ```bash
        kill 1234
        kill -9 5678
        ```

- **pgrep**
    - 지정된 패턴과 일치하는 프로세스의 PID 검색

        ```bash
        pgrep nginx
        pgrep -u myuser java
        ```

- **pkill**
    - 이름 또는 다른 속성을 이용하여 프로세스 종료

        ```bash
        pkill -f myapp # 'myapp'을 포함하는 모든 프로세스 종료
        pkill -u otheruser firefox # 'otheruser'가 실행한 'firefox' 프로세스 종료
        ```

- **bg**
    - 백그라운드로 중단된 작업 다시 실행

- **fg**
    - 백그라운드 작업을 포그라운드로 가져옴

- **jobs**
    - 현재 백그라운드 작업 목록 표시


### 1.4 네트워크 관리

- **ip addr 또는 ifconfig**
    - 네트워크 인터페이스 정보 확인
    - **`ifconfig`**는 더 이상 기본 명령어가 아닐 수 있으며, **`net-tools`** 패키지를 설치해야 할 수도 있음

        ```bash
        ip addr show
        ifconfig
        ```

- **netstat**
    - 네트워크 연결, 라우팅 테이블, 인터페이스 통계 등 표시<br><br>
        - **`netstat -tuln`** : 현재 열려 있는 TCP 및 UDP 포트 목록 표시
        - **`netstat -rn`** : 라우팅 테이블 표시
        
        ```bash
        netstat -tuln
        netstat -rn
        ```

- **ss**
    - netstat`의 최신 버전, 더 많은 정보를 효율적으로 보여줌<br><br>
        - **`ss -tuln`** : 현재 열려 있는 TCP 및 UDP 포트 목록 표시
        - **`ss -rn`** : 라우팅 테이블 표시

        ```bash
        ss -tuln
        ss -rn
        ```

- **ping**
    - 특정 호스트에 네트워크 연결이 가능한지 확인

        ```bash
        ping google.com
        ping 192.168.1.1
        ```

- **traceroute 또는 tracepath**
    - 특정 호스트까지의 네트워크 경로 추적

        ```bash
        traceroute google.com
        tracepath google.com
        ```

- **ssh**
    - Secure Shell의 약자, 원격 서버에 안전하게 접속

        ```bash
        ssh <사용자_이름>@<호스트_IP_또는_도메인>
        ssh -i <개인키_파일> <사용자_이름>@<호스트_IP_또는_도메인>
        ```

- **scp**
    - Secure Copy의 약자, 로컬과 원격 서버 간에 파일을 안전하게 복사<br><br>
        - 로컬 -> 원격: **`scp <로컬_파일> <원격_사용자>@<원격_호스트>:<원격_경로>`**
        - 원격 -> 로컬: **`scp <원격_사용자>@<원격_호스트>:<원격_파일> <로컬_경로>`**
        - 디렉토리 복사 시 **`-r`** 옵션 사용

        ```bash
        scp mylocal.txt user@192.168.1.10:/home/user/
        scp -r mydir user@example.com:/opt/backup/
        scp user@remote.server:/var/log/app.log /tmp/
        ```

- **firewall-cmd (CentOS/RHEL) 또는 ufw (Ubuntu)**
    - 방화벽 설정을 관리하는 명령어
    - GCP에서는 네트워크 방화벽 규칙을 콘솔에서 설정하는 것이 일반적이지만, 인스턴스 내부 방화벽을 관리해야 할 경우 사용될 수 있음


### 1.5 패키지 관리

- **Debian/Ubuntu 계열 (apt)**
    - **`sudo apt update`** : 패키지 목록 업데이트
    - **`sudo apt upgrade`** : 설치된 모든 패키지를 최신 버전으로 업그레이드
    - **`sudo apt install <패키지_이름>`** : 새로운 패키지 설치
    - **`sudo apt remove <패키지_이름>`** : 패키지 제거 (설정 파일은 남김)
    - **`sudo apt purge <패키지_이름>`** : 패키지와 설정 파일을 모두 제거
    - **`sudo apt search <검색어>`** : 패키지 목록에서 검색어를 포함하는 패키지를 검색
    - **`sudo apt show <패키지_이름>`** : 특정 패키지의 상세 정보 표시

- **CentOS/RHEL 계열 (yum 또는 dnf)**
    - **`sudo yum update` / `sudo dnf update`** : 패키지 목록을 업데이트하고 설치된 패키지를 업그레이드
    - **`sudo yum install <패키지_이름>` / `sudo dnf install <패키지_이름>`** : 새로운 패키지 설치
    - **`sudo yum remove <패키지_이름>` / `sudo dnf remove <패키지_이름>`** : 패키지 제거
    - **`sudo yum search <검색어>` / `sudo dnf search <검색어>`** : 패키지 목록에서 검색어를 포함하는 패키지를 검색
    - **`sudo yum info <패키지_이름>` / `sudo dnf info <패키지_이름>`** : 특정 패키지의 상세 정보 표시

### 1.6 텍스트 처리 및 검색

- **grep**
    - 주어진 패턴과 일치하는 텍스트를 파일에서 검색<br><br>
        - **`grep <패턴> <파일>`** : 파일에서 패턴을 검색
        - **`grep -i <패턴> <파일>`** : 대소문자를 구분하지 않고 검색합
        - **`grep -r <패턴> <디렉토리>`** : 디렉토리 및 하위 디렉토리의 모든 파일에서 패턴을 검색
        - **`grep -n <패턴> <파일>`** : 패턴이 발견된 줄 번호를 함께 출력
        - **`grep -v <패턴> <파일>`** : 패턴과 일치하지 않는 줄만 출력

        ```bash
        grep "error" logfile.txt
        grep -i "warning" access.log
        grep -r "config" /etc/
        ```

- **sed**
    - 스트림 편집기(Stream EDitor)
    - 텍스트 스트림에 대한 기본적인 텍스트 변환 수행
    - 파일의 내용을 변경하거나, 찾아서 바꾸는 작업 등에 사용

        ```bash
        sed 's/old_string/new_string/g' input.txt > output.txt # 'input.txt'에서 'old_string'을 'new_string'으로 모두 바꾸어 'output.txt'에 저장
        sed -i 's/original/modified/g
        ```

## 2. 추가로 알아두어야 할 실무 커맨드

### 2.1 고급 텍스트 처리 및 데이터 가공

- **awk**
    - 패턴 탐색 및 처리 언어
    - 텍스트 파일에서 특정 열(column)만 추출하거나 계산할 때 사용
        - 예: 로그 파일의 3번째 열인 '응답 시간'의 평균 구하기
    - 텍스트 파일을 행(Row)과 열(Column) 단위로 인식하여 데이터를 추출하거나 통계 계산을 수행
    - 로그 분석 시 특정 필드만 골라내거나 합계를 구할 때 매우 강력함

        ```bash
        awk '{print $3}' access.log           # 로그 파일의 3번째 열(필드)만 출력
        awk '$3 >= 400 {print $0}' access.log # 3번째 열이 400 이상인 행(에러 로그 등) 전체 출력
        ```

* **find**
    - **파일 검색 도구**
    - 이름, 크기, 수정 날짜, 권한 등 다양한 조건을 기반으로 파일을 검색
    - 단순히 파일을 찾는 것을 넘어, 특정 조건(날짜, 크기 등)에 맞는 파일을 찾아 명령어를 일괄 실행
    - 검색된 파일에 대해 특정 명령을 즉시 실행(**`-exec`**)할 수 있어 자동화에 필수적임

        ```bash
        find /var/log -name "*.log" -mtime +30  # /var/log에서 30일 이상 된 로그 파일 찾기
        find . -type f -size +100M -exec ls -lh {} \; # 100MB가 넘는 파일을 찾아 상세 정보 출력
        ```

- **xargs**
    - 표준 입력 전달 도구
    - 앞선 명령의 결과를 다음 명령의 인자(Argument)로 변환하여 전달할 때 사용
        - **`find`**와 조합하여 특정 로그들을 한꺼번에 삭제하는 등
    - 파일 목록을 받아 일괄 삭제하거나 압축할 때 주로 사용

        ```bash
        find . -name "*.tmp" | xargs rm      # .tmp 파일을 모두 찾아 한꺼번에 삭제
        cat servers.txt | xargs -I {} ssh {} # 파일에 적힌 서버 목록에 순차적으로 접속 시도
        ```

* **diff / patch:**
    - **파일 차이 비교 및 업데이트 적용**
    - **`diff`**는 두 파일의 내용 차이를 보여주고, **`patch`**는 그 차이점(diff 파일)을 원본에 적용하여 업데이트함
    - 형상 관리(Git)의 기초 원리이며, 서버 설정 파일의 변경 사항을 배포할 때 유용함

        ```bash
        diff config.old config.new > update.patch # 두 파일의 차이점을 패치 파일로 저장
        patch config.old < update.patch           # 패치 파일을 사용하여 원본 파일을 업데이트
        ```

- **sort**
    - **텍스트 정렬 도구**
    - 텍스트 파일의 내용을 행 단위로 알파벳순이나 숫자순으로 정렬
    - 데이터 엔지니어가 흩어져 있는 로그를 시간순이나 크기순으로 나열할 때 필수적임

        ```bash
        sort data.txt              # 파일을 알파벳 순서대로 정렬
        sort -n -r numbers.txt     # 숫자를(n) 역순으로(r) 정렬
        ```

- **uniq**
    - **중복 행 제거 도구**
    - 연속되는 중복 행을 하나로 합치거나, 중복된 횟수를 출력
    - 반드시 **`sort`**와 함께 사용해야 하며(정렬되어 있어야 중복을 찾음), "가장 많이 접속한 IP Top 10" 같은 통계를 낼 때 사용함

        ```bash
        sort access.log | uniq -c  # 로그에서 각 행이 몇 번씩 나타났는지 카운트(c)
        sort data.txt | uniq -u    # 중복되지 않는 유일한 행만 출력
        ```


### 2.2 디스크 및 시스템 자원 분석 (시스템 진단)

- 데이터 엔지니어는 대용량 데이터를 다루므로 디스크 용량 관리가 매우 중요함

- **df -h / du -sh**
    - 디스크 공간 확인(Disk Free / Disk Usage)
    - df: 전체 디스크 용량과 남은 공간을 사람이 보기 편한 단위로 확인
    - du: 특정 디렉토리나 파일이 차지하는 용량을 확인
        - 어떤 녀석이 용량을 다 먹고 있는지 찾을 때 필수
    - 데이터 엔지니어링 중 디스크 풀(Full) 장애 예방을 위한 필수 명령어

        ```bash
        df -h               # 사람이 보기 편한 단위(GB, MB)로 전체 디스크 용량 확인
        du -sh /var/log     # /var/log 디렉토리의 전체 크기 요약 출력
        ```

- **free -m / -g**
    - **메모리 사용량 확인**
    - 시스템의 물리적 메모리(RAM)와 스왑(Swap) 메모리의 전체, 사용 중, 여유량을 확인
    - **`-m`**(MB), **`-g`**(GB) 옵션을 통해 원하는 단위로 출력하여 메모리 부족 여부를 판단
        
        ```bash
        free -m    # 메모리 상태를 MB 단위로 확인
        free -h    # 사용자가 읽기 편한 최적의 단위(Human-readable)로 확인
        ```

- **lsof**
    - 열린 파일 확인(LiSt Open Files)
    - 어떤 프로세스가 어떤 파일이나 네트워크 포트를 점유하고 있는지 확인
    - 포트 충돌 해결이나 사용 중인 파일 삭제가 안 될 때 원인 파악에 사용
    
        ```bash
        lsof -i :8080               # 8080번 포트를 사용 중인 프로세스 정보 확인
        lsof /var/log/myapp.log     # 특정 로그 파일을 잡고 있는 프로그램 확인
        ```

- **uptime**
    - **시스템 가동 시간 및 부하 확인**
    - 서버가 마지막으로 켜진 후 경과된 시간, 현재 접속자 수, 시스템 평균 부하(Load Average)를 한 줄로 출력
    - "서버가 왜 이렇게 느리지?" 싶을 때 가장 먼저 입력해서 전반적인 부하 상황을 파악하는 용도
    
        ```bash
        uptime          # 가동 시간과 1분, 5분, 15분간의 평균 부하 확인
        uptime -p       # 예쁘게(pretty) 가동 시간만 출력 (예: up 2 weeks, 3 days)
        ```


### 2.3 네트워크 트러블슈팅 및 외부 통신

- API를 연동하거나 원격 서버의 데이터를 가져올 때 필요함

- **nslookup / dig**
    - **DNS(도메인 네임 시스템) 질의 도구**
    - 도메인 이름(DNS)이 IP로 잘 변환되는지 확인
    - 특정 도메인이 어떤 IP 주소로 연결되어 있는지, 혹은 그 반대의 경우를 확인
    - 네트워크 연결 문제 발생 시 도메인 설정(DNS) 문제인지 서버 문제인지 판별할 때 사용
    
        ```bash
        nslookup www.google.com  # 도메인에 연결된 IP 주소 확인
        dig www.google.com       # DNS 레코드 정보를 더 상세하고 기술적으로 확인
        ```

- **curl / wget**
    - **데이터 전송 도구 (Client URL)**
    - 웹 서버에 요청을 보내거나 파일을 다운로드함 (API 테스트 시 **`curl`**은 필수 항목)
    - HTTP, HTTPS, FTP 등 다양한 프로토콜을 이용해 서버와 데이터를 주고받음
    - REST API 테스트, 웹 페이지 응답 확인, 파일 다운로드 시 사용
    
        ```bash
        curl -X GET https://api.example.com/data # API 서버에 GET 요청 보내기
        curl -I https://www.google.com           # 웹사이트의 헤더(상태 코드 등) 정보만 확인
        ```

- **nc (netcat)**
    - **네트워크의 맥가이버 칼**
    - 임의의 TCP/UDP 연결을 생성하거나 데이터를 주고받음
    - 서버의 특정 포트가 열려있는지, 방화벽에 막혀 있는지, 네트워크 통신이 가능한지 체크(테스트)할 때 가장 많이 사용
    
        ```bash
        nc -zv 192.168.0.10 3306    # 원격지의 3306(DB) 포트가 열려 있는지 확인
        ```


### 2.4 파일 아카이브 및 압축 (데이터 전송)**

- 서버 간 데이터를 옮길 때 묶음 처리는 기본

- **tar**
    - **파일 아카이브 도구 (Tape ARchive)**
    - 여러 파일을 하나로 묶거나(Archive, **`-cvf`**), 묶인 파일을 다시 풂(압축 해제, **`-xvf`**)
    - 대부분의 리눅스 배포용 소프트웨어나 로그 백업 시 사용
    
        ```bash
        tar -cvzf backup.tar.gz /home/user   # /home/user 디렉토리를 압축해서 backup.tar.gz 생성
        tar -xvzf backup.tar.gz              # 압축된 파일을 현재 디렉토리에 해제
        ```

- **gzip / gunzip**
    - **단일 파일 압축 및 해제**
    - 리눅스에서 가장 표준적으로 사용되는 압축 방식 중 하나
    - **`tar`**와 조합되어 **`.tar.gz`** 형태로 가장 많이 쓰이며, 대용량 로그 파일 하나를 빠르게 압축할 때 유리함
    
        ```bash
        gzip access.log    # access.log를 압축하여 access.log.gz 생성 (원본은 사라짐)
        gunzip access.log.gz # 압축을 해제하여 원본 파일 복원
        ```

- **zip / unzip**
    - **범용 압축 및 해제**
    - 윈도우 등 타 OS와 파일을 주고받을 때 가장 호환성이 좋은 방식
    - 여러 파일을 하나로 묶으면서 동시에 압축하며, 리눅스 서버에서 자료를 전달받을 때 자주 사용됨
    
        ```bash
        zip backup.zip file1.txt file2.txt # 여러 파일을 하나의 zip으로 압축
        unzip data.zip -d ./target_dir     # 특정 디렉토리에 압축 해제
        ```

<br><br>

> - **분야별 종합 체크리스트 (수정 제안)**
>
>   <div class="info-table">
>   <table>
>       <tbody>
>           <tr>
>               <td class="td-rowheader" style="width: 200px;">파일/디스크</td>
>               <td style="width: 300px;">`find`, `du`, `df`</td>
>               <td style="width: 400px;">용량 부족 및 파일 위치 추적은 데이터 엔지니어의 숙명</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">고급 텍스트 처리</td>
>               <td>`awk`, `xargs`, `sort`, `uniq`</td>
>               <td>로그 데이터 가공 및 통계 추출의 핵심 도구</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">네트워크</td>
>               <td>`curl`, `nc`, `nslookup`</td>
>               <td>마이크로서비스(MSA) 환경에서 통신 장애 진단 필수</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">시스템 정보</td>
>               <td> `free`, `lsof`, `uptime`</td>
>               <td>시스템 부하 원인 파악 및 메모리 부족 진단</td>
>           </tr>
>           <tr>
>               <td class="td-rowheader">압축/전송</td>
>               <td>`tar`, `gzip`</td>
>               <td>로그 백업 및 데이터 전송을 위한 아카이브 관리</td>
>           </tr>
>       </tbody>
>   </table>
>   </div>
>
{: .common-quote}

<br><br>

> - 위 명령어들은 단독으로 쓰일 때보다 **파이프(`|`)**로 연결될 때 진정한 위력을 발휘함
>   - 예시
>       - **`du -ah | sort -rh | head -n 5`** 🡲 "현재 폴더에서 가장 용량을 많이 차지하는 파일 5개 찾기"
>       - **`sort`** + **`uniq`** 🡲 "에러 로그 중 가장 빈번한 에러 5개만 뽑아줘" (데이터 엔지니어 단골 과제)
>       - **`uptime`** 🡲 "지금 서버 터지기 직전인가요?" (운영자 필수 체크)<br><br>
> - **`find`**는 현업에서 **`xargs`**와 짝꿍처럼 쓰임
> - **`free`**에서 **`available`** 항목이 실제 앱이 사용 가능한 메모리이므로 **`free`** 수치보다 더 중요하게 보아야 함
> - **`dig`**는 데이터 엔지니어가 분산 환경(Hadoop, Spark 등)에서 노드 간 통신이 안 될 때 호스트네임 해석 문제를 찾기 위해 매우 자주 사용하는 도구임
{: .expert-quote}