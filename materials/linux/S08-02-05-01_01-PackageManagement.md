---
layout: page
title:  "리눅스 명령어: 패키지 관리"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-02-05-01_01-PackageManagement
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Debian/Ubuntu 계열 (apt)
- **sudo apt update**
    - 패키지 목록 업데이트

- **sudo apt upgrade**
    - 설치된 모든 패키지를 최신 버전으로 업그레이드

- **sudo apt install < 패키지_이름>**
    - 새로운 패키지 설치

- **sudo apt remove < 패키지_이름>**
    - 패키지 제거 (설정 파일은 남김)

- **sudo apt purge < 패키지_이름>**
    - 패키지와 설정 파일을 모두 제거

- **sudo apt search < 검색어>**
    - 패키지 목록에서 검색어를 포함하는 패키지를 검색

- **sudo apt show < 패키지_이름>**
    - 특정 패키지의 상세 정보 표시

## 2. CentOS/RHEL 계열 (yum 또는 dnf)
- **sudo yum update**<br>**sudo dnf update**
    - 패키지 목록을 업데이트하고 설치된 패키지를 업그레이드

- **sudo yum install < 패키지_이름>**<br>**sudo dnf install < 패키지_이름>**
    - 새로운 패키지 설치

- **sudo yum remove < 패키지_이름>**<br>**sudo dnf remove < 패키지_이름>**
    - 패키지 제거

- **sudo yum search < 검색어>**<br>**sudo dnf search < 검색어>**
    - 패키지 목록에서 검색어를 포함하는 패키지를 검색

- **sudo yum info < 패키지_이름>**<br>**sudo dnf info < 패키지_이름>**
    - 특정 패키지의 상세 정보 표시