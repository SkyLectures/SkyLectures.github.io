---
layout: page
title:  "Github Branch 관리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S07-04-01-01_01-GithubBranch
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Git & GitHub 협업 핵심 개념

### 1.1 Git vs GitHub

| 항목    | Git                                      | GitHub                                      |
|---------|------------------------------------------|---------------------------------------------|
| 역할    | 로컬에서 코드의 버전을 관리하는 도구     | Git 저장소를 온라인에서 관리하고 공유       |
| 특징    | 분산형 버전 관리                         | 중앙 저장소 제공, PR, Issue, Wiki 기능 제공 |

### 1.2 브랜치란?

- 브랜치: 기존 코드 흐름에서 독립적으로 작업할 수 있도록 만든 새로운 경로
    - 실험, 기능 추가, 수정 작업 등을 안전하게 할 수 있음
    - 다른 팀원에게 영향 없이 개발 가능

#### 1.2.1 예시 명령어

```bash
git branch feature/login       # 브랜치 생성
git checkout feature/login     # 브랜치 전환
git checkout -b feature/login  # 생성 + 전환
```


## 2. 팀 협업을 위한 브랜치 전략

### 2.1 브랜치 전략이 필요한 이유

- 여러 명이 동시에 작업하면 충돌 가능성 높음
- 작업 분리 없이 하나의 브랜치에서만 작업하면 **혼란 + 코드 꼬임**
- 전략을 도입하면 업무 분담, 병합, 릴리즈 시점 관리가 쉬워짐

### 2.2 단순화된 Git Flow 전략

```
main       ← 배포용 안정된 브랜치
│
├── develop ← 기능 브랜치를 병합하는 테스트 브랜치
│   ├── feature/login
│   ├── feature/signup
│   └── bugfix/error-popup
```

### 2.3 브랜치별 용도 설명

| 브랜치 종류 | 용도 설명                                              |
|-------------|--------------------------------------------------------|
| main        | 최종 배포 브랜치. CI/CD 배포 연결 가능                 |
| develop     | 전체 기능 통합 브랜치. 항상 최신 개발 코드 유지        |
| feature/*   | 새로운 기능 개발용 브랜치 (예: 로그인, 게시글 추가 등) |
| bugfix/*    | 버그 수정용 브랜치                                     |
| hotfix/*    | 급하게 수정해서 배포해야 할 치명적 오류용 브랜치       |



## 3. 실습: 팀 협업 브랜치 실습 프로젝트

### 3.1 프로젝트 초기화

```bash
git clone https://github.com/your-team/project.git
cd project
git checkout -b develop origin/develop
```

- 저장소를 클론 후 `develop` 브랜치로 이동
- `develop` 브랜치는 기능 개발 브랜치들의 부모 브랜치 역할


### 3.2 실습 1: 기능 브랜치 생성 및 작업

**시나리오:**  
로그인 기능 개발을 위해 `feature/login` 브랜치 생성 후 커밋

```bash
git checkout -b feature/login develop  # 브랜치 생성 및 이동

# 코드 작성
echo "console.log('Login feature');" > login.js

# Git에 추가 후 커밋
git add login.js
git commit -m "✨ Add login feature"

# 원격 저장소에 푸시
git push -u origin feature/login
```

- 브랜치는 항상 `develop`에서 파생하도록 함
- `✨`는 커밋 이모지 예시로, 커밋 메시지 규칙 도입 시 사용 가능함

---

### 3.3 실습 2: Pull Request (PR) 생성 및 병합

1. GitHub 웹에서 `feature/login` → `develop` PR 생성
2. 팀원이 리뷰 후 `Approve`
3. 병합(Merge) 후 `feature/login` 브랜치 삭제

```bash
git branch -d feature/login                  # 로컬 브랜치 삭제
git push origin --delete feature/login      # 원격 브랜치 삭제
```

- PR은 코드 리뷰 및 변경 이력 확인에 필수
- 팀마다 병합 방식(Merge, Squash, Rebase 등) 결정

---

### 3.4 실습 3: 충돌 해결 실습

- 시나리오
    - 두 개의 브랜치에서 같은 파일 `main.js`를 수정하여 충돌 발생

```bash
# feature/signup 브랜치에서 main.js 수정
git checkout -b feature/signup develop
echo "console.log('Signup feature');" > main.js
git add .
git commit -m "signup 기능 추가"
git push origin feature/signup

# feature/login 브랜치에서 동일 파일 수정
git checkout feature/login
echo "console.log('Login feature');" > main.js
git add .
git commit -m "login 기능 추가"

# feature/login에서 signup 브랜치 병합 시도
git merge feature/signup
```

- 충돌 발생 시 Git이 알려줌
- 직접 파일 열어 충돌 부분 수정
- 수정 후 다음 명령어 실행:

```bash
git add main.js
git commit -m "🛠️ merge conflict 해결"
```


## 4. 팀 브랜치 네이밍 규칙 제안

| 유형      | 형식 예시              | 설명                       |
|-----------|------------------------|----------------------------|
| 기능      | `feature/login`        | 기능 추가                  |
| 버그 수정 | `bugfix/modal-close`   | 버그 수정                  |
| 핫픽스    | `hotfix/payment-error` | 배포 후 긴급 수정 사항     |
| 실험용    | `experiment/ai-test`   | 실험성 개발, POC 등        |

- 네이밍은 **일관성**이 가장 중요
- Jira, Notion과 연결되면 `feature/JIRA-1234-login`처럼 확장 가능

---

## 5. GitHub Flow 전략 (간단 버전)

### 5.1 특징

- **develop 없이 main에 직접 기능 브랜치를 병합**
- 배포 자동화가 잘 된 경우 사용
- 스타트업/소규모 프로젝트에 적합

```
main
├── feature/header → PR → main
```

### 5.2 차이점

| 항목        | Git Flow                | GitHub Flow            |
|-------------|-------------------------|------------------------|
| 브랜치 구조 | main + develop + 기타   | main + 기능 브랜치     |
| 복잡성      | 다소 복잡               | 간단                   |
| 추천 상황   | 대규모 협업/기업 팀     | 개인/소규모 팀         |



## 6. 보너스 실습: 커밋 시각화 보기

```bash
git log --oneline --graph --all --decorate
```

- 브랜치 병합 히스토리를 트리 형태로 보여줌
- 협업 시 브랜치 병합 흐름을 시각적으로 파악 가능


## 7. 실습 체크리스트

| 실습 항목                | 설명                         | 확인 여부  |
|--------------------------|------------------------------|------------|
| 브랜치 생성 및 전환      | 기능 브랜치 만들고 이동      | ☐         |
| 커밋 및 푸시             | 변경 사항 저장소에 업로드    | ☐         |
| PR 생성 및 코드 리뷰     | 팀원 간 리뷰로 품질 향상     | ☐         |
| 충돌 발생 및 해결        | 충돌이 발생한 파일 수동 수정 | ☐         |
| 병합 완료 후 브랜치 삭제 | 정리 단계                    | ☐         |


## 8. 마무리 및 Q&A

### 8.1 핵심 정리

- **Branch 전략은 협업의 질을 좌우함**
- 기능/버그 브랜치를 잘 나누고 PR 리뷰를 적극 활용
- 충돌 해결 능력도 실무에 중요한 협업 역량

| 항목        | 요약 설명                        |
|-------------|----------------------------------|
| 브랜치 전략 | 개발 흐름 정리의 기준점          |
| 기능 브랜치 | 독립적 개발 및 병합 간소화       |
| GitHub Flow | 간단한 프로젝트에 적합           |
| Git Flow    | 조직적이고 규모 있는 협업에 적합 |

## 부록: Git 명령어 요약

| 작업 항목           | 명령어                                            |
|---------------------|---------------------------------------------------|
| 브랜치 목록 확인    | `git branch`                                      |
| 브랜치 생성 및 이동 | `git checkout -b 브랜치명`                        |
| 브랜치 병합         | `git merge 브랜치명`                              |
| 브랜치 삭제         | `git branch -d 브랜치명`                          |
| 원격 브랜치 삭제    | `git push origin --delete 브랜치명`               |
| 충돌 확인           | `git status`                                      |
| 충돌 해결 후 커밋   | `git add .` → `git commit -m "resolve conflict"` |

