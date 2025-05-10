---
layout: page
title:  "리눅스 명령어: 파일 및 디렉토리 관리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-01-01_01-FileDirectoryManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. pwd
- 현재 작업 디렉토리(Present Working Directory)의 경로 출력

    ```bash
    pwd
    ```

## 2. ls
- 현재 디렉토리의 파일 및 하위 디렉토리 목록 표시<br><br>
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

## 3. cd
- Change Directory의 약자, 디렉토리를 변경<br><br>
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

## 4. mkdir
- Make Directory의 약자, 새로운 디렉토리 생성<br><br>
    - `mkdir <디렉토리_이름>`: 지정된 이름의 디렉토리 생성
    - `mkdir -p <경로/디렉토리_이름>`: 지정된 경로에 없는 상위 디렉토리까지 모두 생성

    ```bash
    mkdir new_directory
    mkdir -p /home/user/nested/directory
    ```

## 5. rmdir
- Remove Directory의 약자, 비어 있는 디렉토리 삭제

    ```bash
    rmdir empty_directory
    ```

## 6. rm
- Remove의 약자, 파일 또는 디렉토리 삭제<br><br>
    - `rm <파일_이름>`: 지정된 파일 삭제
    - `rm -r <디렉토리_이름>`: 지정된 디렉토리와 그 안의 모든 파일 및 하위 디렉토리 삭제 (<span style="color: red;">주의해서 사용할 것!</span>)
    - `rm -f <파일_이름>`: 강제로 파일 삭제 (삭제 여부를 묻지 않음)
    - `rm -rf <디렉토리_이름>`: 강제로 디렉토리와 그 안의 모든 내용 삭제 (<span style="color: red;">**매우 주의**해서 사용할 것!</span>)

    ```bash
    rm myfile.txt
    rm -r old_directory
    rm -f important.log # 정말 필요한 경우에만 사용할 것!
    ```

## 7. cp
- Copy의 약자, 파일 또는 디렉토리 복사<br><br>
    - `cp <원본_파일> <대상_파일>`: 원본 파일을 대상 파일 이름으로 복사
    - `cp <원본_파일> <대상_디렉토리>/`: 원본 파일을 대상 디렉토리 안으로 복사
    - `cp -r <원본_디렉토리> <대상_디렉토리>/`: 원본 디렉토리와 그 안의 모든 내용을 대상 디렉토리 안으로 복사

    ```bash
    cp original.txt backup.txt
    cp report.pdf /home/user/documents/
    cp -r source_folder /opt/backup/
    ```

## 8. mv
- Move의 약자, 파일 또는 디렉토리의 위치를 변경하거나 이름 변경<br><br>
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

## 9. cat
- 파일의 내용을 화면에 출력

    ```bash
    cat logfile.txt
    ```

## 10. less
- 파일 내용을 페이지 단위로 넘겨가며 볼 수 있음
- 긴 파일 내용을 확인할 때 유용함 (Spacebar로 다음 페이지, `b`로 이전 페이지, `q`로 종료)

    ```bash
    less very_long_file.txt
    ```

## 11. head
- 파일의 처음 몇 줄(기본적으로 10줄)을 표시
- `-n <숫자>` 옵션으로 줄 수를 지정 가능

    ```bash
    head config.ini
    head -n 20 error.log
    ```

## 12. tail
- 파일의 마지막 몇 줄을 표시
- 실시간 로그를 확인할 때 유용함
- `-f` 옵션 사용 시, 파일 내용이 추가될 때마다 계속해서 출력

    ```bash
    tail access.log
    tail -n 5 status.log
    tail -f application.log
    ```

## 13. touch
- 빈 파일을 생성하거나 파일의 접근 및 수정 시간을 업데이트

    ```bash
    touch new_file.txt
    touch existing_file.txt # 파일의 타임스탬프 업데이트
    ```