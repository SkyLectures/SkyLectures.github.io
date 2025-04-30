---
layout: page
title:  "Github Action"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S07-06-01-01_01-GithubAction
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. GitHub Actions를 활용한 CI/CD 파이프라인 구성 가이드

### 1.1 CI/CD란?

| 구분                                  | 설명                                          | 수행 목적      |
|---------------------------------------|-----------------------------------------------|----------------|
| CI (Continuous Integration)           | 코드 변경 시 자동 빌드 및 테스트 수행         | 코드 품질 보장 |
| CD (Continuous Deployment / Delivery) | 변경된 코드 자동 배포 (Staging or Production) | 자동화된 배포  |


### 1.2 GitHub Actions란?

- GitHub에서 제공하는 **자동화 워크플로우 엔진**
- 이벤트 기반으로 작업을 자동 실행함

- **주요 개념**

| 항목     | 설명                                         |
|----------|----------------------------------------------|
| Workflow | 자동화 작업 단위 (YAML 파일로 정의)          |
| Job      | 여러 Step으로 이루어진 실행 단위             |
| Step     | 실제 명령 실행 단위 (쉘 명령 또는 액션 사용) |
| Action   | 재사용 가능한 작업 모듈                      |
| Runner   | 실제 실행 환경 (Ubuntu, macOS, Windows 등)   |



### 1.3 폴더 구조

```plaintext
.github/
└── workflows/
    └── ci-cd.yml     <-- 워크플로우 정의 파일
```


### 1.4 기본 예제: Node.js 프로젝트 CI/CD 파이프라인

- `ci-cd.yml` 전체 예제

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: 소스 체크아웃
        uses: actions/checkout@v3

      - name: Node.js 설치
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: 의존성 설치
        run: npm install

      - name: 테스트 실행
        run: npm test

  deploy:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - name: 소스 체크아웃
        uses: actions/checkout@v3

      - name: 배포 스크립트 실행
        run: |
          echo "🚀 배포 중..."
          ./deploy.sh  # 배포 스크립트 실행
```

- **주요 구성 요소 상세 설명**

    - `on:` (트리거)

        - 어떤 이벤트가 발생할 때 워크플로우가 실행될지 정의함

            ```yaml
            on:
            push:
                branches: [ main ]
            pull_request:
                branches: [ main ]
            ```

        - main 브랜치에 push 또는 PR 생성 시 실행됨

    - Job: build-and-test
        - Job은 Ubuntu 환경에서 실행됨

            ```yaml
            jobs:
            build-and-test:
                runs-on: ubuntu-latest
            ```

        - Step 설명

            ```yaml
            - uses: actions/checkout@v3
            ```
            - 현재 커밋된 코드를 워크플로우에 가져옴

                ```yaml
                - uses: actions/setup-node@v3
                with:
                    node-version: '18'
                ```

            - Node.js 18 환경을 설정함

                ```yaml
                - run: npm install
                ```

            - 의존성 설치

                ```yaml
                - run: npm test
                ```

            - 테스트 실행

        - Job: deploy

            - 앞선 Job이 성공해야 실행됨 (`needs:` 사용)
            - main 브랜치에서 `push` 이벤트일 때만 실행

                ```yaml
                if: github.ref == 'refs/heads/main' && github.event_name == 'push'
                ```

            - `./deploy.sh` 파일을 실행하여 배포 자동화

### 1.5 실제 배포 예시 (예: Vercel, Firebase, S3 등)

- 예: Vercel CLI로 배포하기

    ```yaml
    - name: Vercel CLI 설치
    run: npm i -g vercel

    - name: Vercel 배포
    run: vercel --token=${{ secrets.VERCEL_TOKEN }} --prod
    ```

> 🔐 `VERCEL_TOKEN`은 GitHub Repository의 **Settings > Secrets**에서 등록함


### 1.6 Secrets 설정 방법

1. GitHub 저장소 > Settings > Secrets > Actions
2. `New repository secret` 클릭
3. 예: `VERCEL_TOKEN`, `AWS_ACCESS_KEY_ID`, `DEPLOY_KEY` 등 추가


### 1.7 테스트 예제 프로젝트용 Workflow

- 예: Python 프로젝트용 CI

    ```yaml
    name: Python CI

    on: [push]

    jobs:
    test:
        runs-on: ubuntu-latest

        steps:
        - uses: actions/checkout@v3

        - name: Python 설치
            uses: actions/setup-python@v4
            with:
            python-version: '3.10'

        - name: 의존성 설치
            run: pip install -r requirements.txt

        - name: 테스트 실행
            run: pytest
    ```


### 1.8 배포 자동화 고도화 아이디어

| 항목                 | 설명                                    |
|----------------------|-----------------------------------------|
| Preview 배포         | PR마다 서브 도메인에 배포하여 미리보기  |
| Slack 알림           | 배포 완료/실패 시 팀 채널에 알림        |
| Test Coverage 리포트 | CI 과정 중 테스트 커버리지 확인         |
| Linting & Formatting | `eslint`, `black` 등을 CI에서 자동 실행 |



### 1.9 워크플로우 모니터링 방법

- GitHub > Actions 탭에서 각 워크플로우 확인
- 실패 시 로그를 확인하고 재실행 가능
- 각 Step의 실행 시간, 결과 상태도 추적 가능


### 마무리 요약

| 항목           | 설명                                    |
|----------------|-----------------------------------------|
| GitHub Actions | GitHub 내에서 제공하는 자동화 도구      |
| CI/CD          | 테스트 및 배포를 자동으로 수행          |
| Workflow       | `.github/workflows/*.yml` 파일로 작성   |
| Job & Step     | 실행 작업을 구체적으로 분리 가능        |
| Secrets        | 민감한 정보는 암호화된 환경 변수로 관리 |


### 추가 자료

- [GitHub Actions 공식 문서](https://docs.github.com/en/actions)
- [Awesome GitHub Actions](https://github.com/sdras/awesome-actions)
- [Actions 마켓플레이스](https://github.com/marketplace?type=actions)
