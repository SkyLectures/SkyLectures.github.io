---
layout: page
title:  "Git 기본 명령어 실습"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S07-03-01-01_01-GitBasicCommands
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. Git/Github 소스 관리 기본 흐름

<img src='/materials/images/git/S07-03-01-01_01-001.png' width="800"/><br>

- **지역 저장소 → Github(원격 저장소)**
    1. 지역 저장소에 새 프로젝트 생성
    2. <span style="color: red;">**git init**</span> 명령어로 해당 프로젝트를 Git 지역 저장소로 지정
    3. 파일 수정
    4. <span style="color: red;">**git add**</span> 명령어로 수정한 파일을 스테이징 영역으로 이동
    5. <span style="color: red;">**git commit**</span> 명령어로 지역 저장소에 저장
    6. <span style="color: red;">**git push**</span> 명령어로 지역 저장소에서 발생한 변경 내역을 원격 저장소에 반영

- **Github(원격 저장소) → 지역 저장소**
    1. Github에 올려진 프로젝트 전체를 <span style="color: red;">**git clone**</span> 명령어로 다운로드
    2. Github에 올려진 프로젝트에서 변경 사항만을 <span style="color: red;">**git pull**</span> 명령어로 다운로드 
        - 다른 사람이 수정한 내용을 나의 지역 저장소에 통합

## 2. Git 기본 명령어

### 2.1 git init

- 기능: Git 초기화(=지역 저장소 생성)
- 명령어 일람
    - 초기화 수행 시
        - git init<br>
        <img src='/materials/images/git/S07-03-01-01_01-002.png' width="600"/><br>
    - 초기화 취소 시
        - rm -rf .git
        - 그냥 파일 탐색기에서 삭제해도 무방함

### 2.2 git config

- 기능: 사용자 정보 등록
- 명령어 일람
    - 현재의 깃 지역 저장소에만 해당하는 사용자 정보를 등록할 경우
        - git config user.name "사용자 이름"
        - git config user.email "이메일 주소"
    - 모든 프로젝트에 적용될 사용자 정보를 등록할 경우
        - git config --global user.name "사용자 이름"
        - git config --global user.email "이메일 주소"
    > 참고:<br>사용자 이름과 이메일 주소는 각자의 GitHub 계정의 Username, Email과 동일해야 함

    <img src='/materials/images/git/S07-03-01-01_01-003.png' width="600"/><br>

    > **cat 명령어(Linux)**
    > - "concatenate"의 약어
    > - 파일의 내용을 터미널에 출력하거나 파일을 합치는 데 사용되는 명령어
    > - Linux 명령어지만 Windows의 PowerShell에서도 사용 가능함

### 2.3 git status

- 기능: 현재 프로젝트의 파일 상태 확인하기
- 명령어 일람
    - git status
        - 실습
            - 파일을 하나 만들고<br>
                <img src='/materials/images/git/S07-03-01-01_01-004.png' width="600"/><br>

            - git status로 확인<br>
                <img src='/materials/images/git/S07-03-01-01_01-005.png' width="600"/><br>

### 2.4 git add

- 기능: 커밋에 포함될 파일 등록
- 명령어 일람
    - git add {파일 명}<br>
        <img src='/materials/images/git/S07-03-01-01_01-006.png' width="600"/><br>

### 2.5 git commit

- 기능: 커밋 생성/수정
- 명령어 일람
    - 기본 사용법(커밋 생성)
        - git commit -m "메시지"
            - 변경된 파일들을 묶어서 관련된 설명을 추가한 후 커밋을 수행함<br>
                <img src='/materials/images/git/S07-03-01-01_01-007.png' width="600"/><br>
                > - 커밋에 등록할 메시지를 보여주고 1개의 파일 변경이 있었는데 추가된 파일이 1개라는 출력을 보여줌
                > - 더이상 변경된 파일이 등록되어 있지 않으면 커밋할 내용이 없다는 메시지를 출력함

    - 커밋 메시지를 상세하게 작성해야 할 경우(에디터를 통해 작성 가능)
        - git commit
            - 실습
                - 파일을 하나 만들고<br>
                    <img src='/materials/images/git/S07-03-01-01_01-008.png' width="600"/><br>

                - 커밋 목록에 파일을 추가함<br>
                    <img src='/materials/images/git/S07-03-01-01_01-009.png' width="600"/><br>

                - 커밋 실행<br>
                    <img src='/materials/images/git/S07-03-01-01_01-010.png' width="800"/><br>

                    - 에디터에서 내용을 작성하고 저장 후 에디터를 닫으면<br>
                    <img src='/materials/images/git/S07-03-01-01_01-011.png' width="600"/><br>

                    - 커밋이 완료됨<br>
                    <img src='/materials/images/git/S07-03-01-01_01-012.png' width="600"/><br>

                - 만약 아래와 같이 만약 이런 식으로 에디터가 실행되지 않는다면<br>
                    <img src='/materials/images/git/S07-03-01-01_01-013.png' width="600"/><br>

                    - 환경 설정을 적용함<br>
                        <img src='/materials/images/git/S07-03-01-01_01-014.png' width="700"/><br>

                        > - 에디터 별 환경 설정 방법
                        >    - VIM을 전용 에디터로 사용하는 경우
                        >        - git config --global core.editor "vim"
                        >    - SubLime을 전용 에디터로 사용하는 경우
                        >        - git config --global core.editor "subl --wait"
                        >    - Atom을 전용 에디터로 사용하는 경우
                        >        - git config --global core.editor "atom --wait"
                        >    - Visual Studio Code를 전용 에디터로 사용하는 경우
                        >        - git config --global core.editor "code --wait"

    - 기존 커밋을 수정해야 할 경우
        - git commit --amend<br>
            <img src='/materials/images/git/S07-03-01-01_01-015.png' width="800"/><br>

            <img src='/materials/images/git/S07-03-01-01_01-016.png' width="800"/><br>
            - 마지막 커밋 에디터 화면을 보여줌
            - 수정 후 저장하면 마지막 커밋 메시지가 수정됨

        - git commit --amend -m "수정 메시지"<br>
            <img src='/materials/images/git/S07-03-01-01_01-017.png' width="600"/><br>
            - git commit --amend 명령과 동일하지만 터미널 환경에서 수정함

    - 기존 커밋의 사용자를 수정해야 할 경우
        - git commit --amend --author "username < email>"<br>
            <img src='/materials/images/git/S07-03-01-01_01-018.png' width="800"/><br>


## 3. Github 작업 내용(Web과 연동)  

### 3.1 Git 원격 저장소 설정

- 먼저 GitHub 페이지에 가서 새로운 저장소(Create a new repository) 생성 작업을 수행함<br>

    <img src='/materials/images/git/S07-03-01-01_01-019.png' width="600"/><br>

    <img src='/materials/images/git/S07-03-01-01_01-020.png' width="600"/><br>

    <img src='/materials/images/git/S07-03-01-01_01-021.png' width="600"/><br>

### 3.2 git remote add

- 기능: 원격 저장소 주소를 Git 지역 저장소에 등록
- 명령어 일람
    - git remote add origin {복사한 원격 저장소 주소}

        <img src='/materials/images/git/S07-03-01-01_01-022.png' width="600"/><br>

### 3.3 git push

- 기능: 지역 저장소에 있는 커밋을 원격 저장소에 등록
- 명령어 일람
    - git push {원격 저장소(식별자)} {브랜치}

        <img src='/materials/images/git/S07-03-01-01_01-023.png' width="600"/><br>

        > - **권한 오류가 발생함**
        >   - 2021년 08월 13일부터 기존의 패스워드를 통한 인증 방식이 중단됨
        >   - GitHub에서 제공하는 개인용 Access Token을 발급받아서 해당 Token을 패스워드로 사용해야 함
        >   - <span style="color: red;">[Setting] → [Developer settings] → [Personal access tokens]에서 발급</span>

### 3.4 Personal Access Token 발급 받기

- [Setting] → [Developer settings] → [Personal access tokens]

    <img src='/materials/images/git/S07-03-01-01_01-024.png' width="300"/>
    &nbsp;<span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-025.png' width="300"/><br>

    <span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-026.png' width="300"/>
    &nbsp;<span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-027.png' width="300"/><br>

    <span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-028.png' width="500"/><br>

    <span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-029.png' width="700"/><br>

    <span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-030.png' width="700"/><br>

    <span style="font-size: 50px; font-width: bold; color: orange;">→</span>&nbsp;
    <img src='/materials/images/git/S07-03-01-01_01-031.png' width="700"/>


### 3.5 자격 증명 저장하기

- Windows의 경우
    - 제어판 > 자격 증명 관리자 > Windows 자격 증명 > 일반 자격 증명 추가

        <img src='/materials/images/git/S07-03-01-01_01-032.png' width="800"/><br><br>    
        <img src='/materials/images/git/S07-03-01-01_01-033.png' width="600"/><br><br>
        <img src='/materials/images/git/S07-03-01-01_01-034.png' width="600"/><br><br>
        <img src='/materials/images/git/S07-03-01-01_01-035.png' width="600"/><br><br>

### 3.6 git push

- 명령어 일람
    - git push origin main

        <img src='/materials/images/git/S07-03-01-01_01-036.png' width="800"/><br><br>
        <img src='/materials/images/git/S07-03-01-01_01-037.png' width="400"/><br><br>
        <img src='/materials/images/git/S07-03-01-01_01-038.png' width="600"/><br><br>

### 3.7 git clone

- 기능: 원격 저장소 복제
- 명령어 일람
    - git clone "원격 저장소 주소" "새로운 저장소 이름"

        <img src='/materials/images/git/S07-03-01-01_01-039.png' width="800"/><br><br>
        <img src='/materials/images/git/S07-03-01-01_01-040.png' width="600"/><br><br>

### 3.8 git log

- 기능: git 로그 조회
- 명령어 일람
    - 기본 사용법
        - git log
            - Git 작업 중 커밋이 수행된 로그(기록)을 보여줌
            - 메시지가 수정된 내용이나, 사용자 정보가 수정된 내용이 표시됨

            <img src='/materials/images/git/S07-03-01-01_01-041.png' width="600"/><br><br>

    - 커밋 로그를 파일 단위에서 변경 내용을 확인하려면
        - git log -p

            <img src='/materials/images/git/S07-03-01-01_01-042.png' width="600"/><br><br>

        - git log --patch

            <img src='/materials/images/git/S07-03-01-01_01-043.png' width="600"/><br><br>

    - 최근 몇 개의 커밋 로그를 보려면
        - git log -{숫자}

            <img src='/materials/images/git/S07-03-01-01_01-044.png' width="600"/><br><br>

    - 조건을 결합해서 사용할 수도 있음
        - git log -p -{숫자}

            <img src='/materials/images/git/S07-03-01-01_01-045.png' width="600"/><br><br>

    - 각 커밋의 통계 정보를 보려면
        - git log --stat

            <img src='/materials/images/git/S07-03-01-01_01-046.png' width="600"/><br><br>

    - 커밋 로그를 보여주는 형식 지정하기
        - git log --pretty={option}

            <img src='/materials/images/git/S07-03-01-01_01-047.png' width="800"/><br><br>
            <img src='/materials/images/git/S07-03-01-01_01-048.png' width="600"/><br><br>

        - --pretty 옵션에서 사용할 수 있는 출력 형식

            | 형식 | 설명               |
            | ---- | ------------------ |
            | %H   | 커밋 해시          |
            | %h   | 짧은 커밋 해시     |
            | %T   | 트리 해시          |
            | %t   | 짧은 트리 해시     |
            | %P   | 부모 해시          |
            | %p   | 짧은 부모 해시     |
            | %s   | 커밋 요약          |
            | %an  | 저자 이름          |
            | %ae  | 저자 이메일        |
            | %ar  | 저자 상대적 시각   |
            | %cn  | 커미터 이름        |
            | %ce  | 커미터 이메일      |
            | %cr  | 커미터 상대적 시각 |

## Git 기본 명령 요약

| 명령어              | 기능                                              | 명령 형식                                                                                                    |
| ------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| git init            | Git 초기화(=지역 저장소 생성)                     | git init                                                                                                     |
| git config          | 프로젝트 별 지역 사용자 등록                      | git config user.name "사용자 이름"<br>git config user.email "이메일 주소"                                    |
| git config --global | 지역 환경의 전체 프로젝트를 위한<br>사용자 등록   | git config --global user.name "사용자 이름"<br>git config --global user.email "이메일 주소"                  |
| git remote add      | 원격 저장소 주소를 지역 저장소에 등록             | git remote add "원격 저장소 주소"                                                                            |
| git add             | 커밋에 포함될 파일 등록                           | git add "파일 명"                                                                                            |
| git status          | 현재 프로젝트의 파일 상태 확인                    | git status                                                                                                   |
| git commit          | 새로운 커밋 생성<br>└(메시지 등록)               | git commit<br>git commit -m "메시지"                                                                         |
| git commit --amend  | 커밋 수정<br>└(메시지 등록)<br>커밋 사용자 수정  | git commit --amend<br>git commit --amend -m "수정 메시지"<br>git commit --amend --author "username < email>" |
| git push            | 지역 저장소에 있는 커밋을<br>원격 저장소에 등록   | git push<br>git push {원격 저장소 식별자} {브랜치}                                                           |
| git clone           | 원격 저장소 복제                                  | git clone {원격 저장소 주소} {새로운 저장소 이름}                                                            |
| git log             | git 로그 조회(=커밋 내역 확인)<br>└(파일 단위)<br>커밋 내역을 가시적/그래프 표현으로 확인   | git log<br>git log -p / git log --patch<br>git log --pretty=oneline --graph         |