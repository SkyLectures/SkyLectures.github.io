---
layout: page
title:  "Github 풀 리퀘스트와 코드 리뷰"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S07-05-01-01_01-GithubPullRequestCodeReview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Pull Request(PR) 개요

### 1. Pull Request란?

- 브랜치에서 작업한 내용을 다른 브랜치(main/develop 등)에 병합해달라고 요청하는 절차

### 1.2 왜 필요한가?

| 이유           | 설명                                     |
|----------------|------------------------------------------|
| 코드 품질 유지 | 리뷰를 통해 실수를 줄이고 품질 향상 가능 |
| 협업 효율      | 변경 내용을 공유하고 의견 교환           |
| 기록 유지      | 모든 변경사항은 PR로 기록되고 추적 가능  |


### 1.3 Pull Request 기본 흐름

1. 브랜치 생성
2. 기능 개발
3. 커밋 및 푸시
4. GitHub에서 PR 생성
5. 코드 리뷰 및 승인
6. 병합(Merge)


### 1.4 실습: PR 생성부터 병합까지

- 예제: `feature/login` 브랜치를 `develop`에 병합

    1. 브랜치 생성 및 기능 작업

        ```bash
        git checkout -b feature/login develop
        echo "console.log('Login page');" > login.js
        git add .
        git commit -m "✨ Add login page"
        git push -u origin feature/login
        ```

    2. GitHub에서 Pull Request 생성

        1. GitHub 저장소 방문
        2. “Compare & pull request” 버튼 클릭
        3. **base: develop ← compare: feature/login** 설정
        4. 제목 및 설명 작성 후 “Create Pull Request”


## 2. 코드 리뷰

### 2.1 리뷰어가 확인하는 항목

| 항목        | 설명                               |
|-------------|------------------------------------|
| 코드 스타일 | 팀 코딩 스타일 가이드 준수 여부    |
| 기능 정확성 | 의도한 기능이 정확히 구현되었는가? |
| 보안 문제   | 민감 정보 노출 여부 등             |
| 리팩토링    | 더 나은 코드 구조를 제안할 수 있음 |
| 테스트      | 관련 테스트 코드가 있는가?         |

### 2.2 GitHub에서 리뷰어가 할 수 있는 행동

| 기능            | 설명                      |
|-----------------|---------------------------|
| Comment         | 일반적인 의견 추가        |
| Suggestion      | 코드 수정을 제안하는 기능 |
| Approve         | 승인(병합 가능)           |
| Request changes | 변경 요청 (병합 불가)     |

#### 2.2.1 예시: 코드에 코멘트 달기

```diff
- console.log("Login page");
+ console.log("Login page loaded");
```
> 리뷰어: "불필요한 콘솔 로그는 배포 전에 제거해주세요!"


## 3. Pull Request 병합

### 3.1 GitHub 웹 UI에서 3가지 병합 방식

| 병합 방식        | 설명                                             |
|------------------|--------------------------------------------------|
| Merge commit     | 기본 병합 방식. 히스토리를 남김                  |
| Squash and merge | 커밋을 하나로 합쳐 병합 (히스토리 간결)          |
| Rebase and merge | 커밋 히스토리를 이어붙여 병합 (리베이스 사용 시) |

> 팀 내에서 어떤 병합 방식 사용할지 미리 정해두는 것이 좋음


### 3.2 PR 병합 후 정리 작업

- 로컬 브랜치 삭제

```bash
git checkout develop
git pull origin develop         # 최신 develop 반영
git branch -d feature/login     # 로컬 브랜치 삭제
git push origin --delete feature/login  # 원격 브랜치 삭제
```


## 4. 실습 체크리스트

| 항목                | 설명                           | 완료 여부 |
|---------------------|--------------------------------|-----------|
| 기능 브랜치 생성    | `feature/기능명` 브랜치로 작업 | ☐        |
| 커밋 및 푸시        | 변경 내용을 GitHub에 푸시      | ☐        |
| PR 생성             | GitHub에서 Pull Request 작성   | ☐        |
| 코드 리뷰           | 리뷰어가 피드백 제공           | ☐        |
| 병합 및 브랜치 삭제 | 병합 후 브랜치 삭제            | ☐        |


## 5. PR 작성 시 유용한 템플릿 예시

```md
### 작업 내용
- 로그인 페이지 UI 구현
- 이메일/비밀번호 입력 폼 추가

### 체크리스트
- [x] UI 구현 완료
- [ ] 입력 유효성 검사 추가 예정

### 스크린샷
(이미지 첨부)

### 관련 이슈
- #14
```

> GitHub 저장소에 `.github/PULL_REQUEST_TEMPLATE.md` 파일을 만들면 자동 적용됨



## 6. 좋은 코드 리뷰 문화 만들기

### 6.1 리뷰어의 자세

- 비판보다 **피드백 중심**
- 이해 안 되는 부분은 질문
- 제안은 **대안 제시**까지

### 6.2 작성자의 자세

- 리뷰는 **성장을 위한 기회**
- 방어적 태도보단 열린 자세
- 리뷰 반영은 신속하게

## 7. 마무리 요약

| 개념         | 요약 설명                             |
|--------------|---------------------------------------|
| Pull Request | 브랜치 작업을 다른 브랜치에 병합 요청 |
| 코드 리뷰    | 팀원이 기능, 품질, 보안 등을 체크     |
| 병합 방식    | merge, squash, rebase 중 선택         |
| 리뷰 문화    | 상호 존중, 적극적인 피드백이 핵심     |


## 부록

- GitHub UI 핵심 메뉴 정리

| 항목          | 위치           | 설명                        |
|---------------|----------------|-----------------------------|
| Pull requests | 상단 메뉴      | PR 리스트 확인              |
| Reviewers     | PR 오른쪽 패널 | 리뷰어 지정 가능            |
| Conversation  | PR 내부 탭     | 전체 리뷰 코멘트            |
| Files changed | PR 내부 탭     | 변경된 코드 비교            |
| Merge 버튼    | PR 하단        | 병합 방식 선택 후 병합 가능 |

- 추가 학습 자료

    - GitHub 공식 문서: [PR Docs](https://docs.github.com/en/pull-requests)
    - GitHub Actions로 PR 자동 검사 구현
    - GitHub Codespaces로 리뷰 환경 통합
