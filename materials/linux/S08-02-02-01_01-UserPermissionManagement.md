---
layout: page
title:  "리눅스 명령어: 사용자 및 권한 관리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-02-01_01-UserPermissionManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. sudo
- Superuser Do의 약자, 관리자 권한으로 명령어 실행
- 대부분의 시스템 관리 작업에 필요함

    ```bash
    sudo apt update
    ```

## 2. chmod
- 파일 또는 디렉토리의 권한(mode) 변경<br><br>
    - 숫자 모드 (예: 755, 644) 또는 심볼릭 모드 (예: `u+rwx`, `g+rx`, `o-w`)를 사용함
    - `u`: 사용자(owner), `g`: 그룹(group), `o`: 기타(others), `a`: 모두(all)
    - `+`: 권한 추가, `-`: 권한 제거, `=`: 권한 설정
    - `r`: 읽기, `w`: 쓰기, `x`: 실행

    ```bash
    chmod 755 script.sh # 사용자에게 읽기, 쓰기, 실행 권한; 그룹과 기타 사용자에게 읽기, 실행 권한 부여
    chmod u+x script.sh # 사용자에게 실행 권한 추가
    chmod g-w data.txt # 그룹에게 쓰기 권한 제거
    ```

## 3. chown
- 파일 또는 디렉토리의 소유자(owner)와 그룹(group) 변경<br><br>
    - `chown <새로운_소유자> <파일_또는_디렉토리>`
    - `chown <새로운_소유자>:<새로운_그룹> <파일_또는_디렉토리>`

    ```bash
    sudo chown user1 myfile.txt
    sudo chown root:admin mydirectory
    ```

## 4. useradd
- 새로운 사용자 추가 (일반적으로 `sudo`와 함께 사용)

    ```bash
    sudo useradd newuser
    ```

## 5. passwd
- 사용자 계정의 비밀번호를 설정하거나 변경

    ```bash
    passwd # 현재 사용자 비밀번호 변경
    sudo passwd newuser # 'newuser'의 비밀번호 변경
    ```

## 6. userdel
- 사용자 삭제
- `-r` 옵션 사용 시 사용자의 홈 디렉토리와 메일 스풀도 함께 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)

    ```bash
    sudo userdel olduser
    sudo userdel -r olduser
    ```

## 7. groupadd
- 새로운 그룹 추가

    ```bash
    sudo groupadd newgroup
    ```

## 8. groupdel
- 그룹 삭제

    ```bash
    sudo groupdel oldgroup
    ```

## 9. usermod
- 사용자 계정의 속성 수정 (그룹 변경, 로그인 셸 변경 등)<br><br>
    - `usermod -aG <그룹_이름> <사용자_이름>`: 사용자를 추가 그룹에 추가 (`-a`는 append, `-G`는 groups)

    ```bash
    sudo usermod -aG developers myuser
    ```
