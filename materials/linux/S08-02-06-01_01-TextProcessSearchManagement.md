---
layout: page
title:  "리눅스 명령어: 텍스트 처리 및 검색"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-06-01_01-TextProcessSearchManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. grep
- 주어진 패턴과 일치하는 텍스트를 파일에서 검색<br><br>
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

## 2. sed
- 스트림 편집기(Stream EDitor)
- 텍스트 스트림에 대한 기본적인 텍스트 변환 수행
- 파일의 내용을 변경하거나, 찾아서 바꾸는 작업 등에 사용

    ```bash
    sed 's/old_string/new_string/g' input.txt > output.txt # 'input.txt'에서 'old_string'을 'new_string'으로 모두 바꾸어 'output.txt'에 저장
    sed -i 's/original/modified/g