---
layout: page
title:  "Dockerfile 작성 및 이미지 최적화"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-02-01_01-Dockerfile
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


`Dockerfile`은 도커 이미지를 만들기 위한 **자동화된 스크립트**입니다. 단순히 앱을 실행하는 것을 넘어, 운영 환경에서 효율적으로 동작하는 이미지를 만들기 위해서는 작성 요령과 최적화 기법을 깊이 있게 이해해야 합니다.

---

## 1. Dockerfile의 주요 명령어와 역할

Dockerfile은 아래에서 위로 쌓이는 **레이어(Layer)** 구조를 형성합니다.

*   **`FROM`**: 모든 Dockerfile의 시작입니다. 기반이 될 **베이스 이미지**를 지정합니다.
*   **`WORKDIR`**: 컨테이너 내부에서 명령어가 실행될 **작업 디렉토리**를 설정합니다. (없으면 자동 생성)
*   **`COPY` / `ADD`**: 호스트 머신의 파일을 컨테이너 내부로 복사합니다. (`ADD`는 URL 다운로드나 압축 해제 기능이 추가로 포함됨)
*   **`RUN`**: 이미지를 빌드하는 동안 실행할 명령어입니다. (패키지 설치, 파일 권한 변경 등)
*   **`CMD` / `ENTRYPOINT`**: 컨테이너가 시작될 때 실행할 명령입니다.
    *   `CMD`: 인자값이 변경될 수 있는 기본 명령
    *   `ENTRYPOINT`: 컨테이너의 주 목적으로 고정된 실행 명령
*   **`EXPOSE`**: 컨테이너가 사용할 포트를 명시합니다. (실제 포트 포워딩은 `docker run -p`에서 수행)
*   **`ENV`**: 환경 변수를 설정합니다.



---

## 2. 이미지 최적화의 필요성

이미지가 무거우면 **빌드 속도가 느려지고**, 네트워크 전송 시 **대역폭을 낭비**하며, 무엇보다 배포 시 **컨테이너 기동 속도**에 악영향을 줍니다. 또한, 불필요한 도구가 포함되면 보안 취약점(Attack Surface)이 늘어납니다.

---

## 3. 핵심 최적화 전략 (Best Practices)

### ① 경량 베이스 이미지 선택 (Alpine & Slim)
일반적인 OS 이미지(Ubuntu 등)는 수백 MB를 차지합니다. 이를 최소화된 이미지로 교체하세요.
*   **Alpine:** 5MB 내외의 초경량 리눅스 배포판입니다. 보안이 강력하고 가볍지만, 일부 C 라이브러리 호환성(musl vs glibc)을 체크해야 합니다.
*   **Slim:** 필요한 런타임(Python, Java 등)만 포함하고 불필요한 패키지를 제거한 버전입니다.

### ② 레이어 수 최소화 (Layer Flattening)
도커는 `RUN`, `COPY`, `ADD` 명령마다 새로운 레이어를 생성합니다. 비슷한 작업은 `&&`로 묶어 하나의 레이어로 합치는 것이 좋습니다.

```dockerfile
# 나쁜 예 (레이어 3개 생성)
RUN apt-get update
RUN apt-get install -y python3
RUN rm -rf /var/lib/apt/lists/*

# 좋은 예 (레이어 1개 생성)
RUN apt-get update && apt-get install -y \
    python3 \
    && rm -rf /var/lib/apt/lists/*
```

### ③ 멀티 스테이지 빌드 (Multi-stage Build) - **가장 중요**
빌드 시에만 필요한 도구(컴파일러, SDK 등)와 실행 시에 필요한 파일(바이너리, 라이브러리)을 분리하는 기법입니다.

```dockerfile
# Stage 1: Build (무거운 이미지를 사용하여 컴파일)
FROM node:18 AS builder
WORKDIR /app
COPY . .
RUN npm install && npm run build

# Stage 2: Production (가벼운 이미지를 사용하여 결과물만 복사)
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```
이렇게 하면 최종 이미지에는 소스 코드나 빌드 도구가 포함되지 않아 용량이 극적으로 줄어듭니다.



### ④ 캐시 효율성 극대화 (Layer Caching)
도커는 빌드 시 변경되지 않은 레이어를 재사용합니다. **변경이 잦은 파일(소스 코드)**은 Dockerfile의 뒷부분에 배치하고, **변경이 적은 파일(종속성 설정)**은 앞부분에 배치하세요.

```dockerfile
COPY package.json .  # 라이브러리 목록을 먼저 복사
RUN npm install      # 변경이 없는 한 캐시된 레이어 사용
COPY . .             # 소스 코드는 자주 변하므로 마지막에 복사
```

### ⑤ `.dockerignore` 파일 활용
Git의 `.gitignore`처럼 이미지 빌드에 불필요한 파일(`.git`, `node_modules`, 로그 파일 등)이 컨테이너 내부로 복사되지 않도록 제외합니다.

---

## 4. 요약: 효율적인 Dockerfile 체크리스트

1.  **가장 가벼운 베이스 이미지**를 썼는가?
2.  **멀티 스테이지 빌드**를 활용했는가?
3.  **캐시를 고려하여 명령어 순서**를 배치했는가?
4.  불필요한 **임시 파일(캐시 삭제 등)**을 `RUN` 명령 끝에 지웠는가?
5.  **.dockerignore**로 불필요한 파일을 걸러냈는가?

이러한 최적화 과정을 거치면 수 GB에 달하던 이미지가 수십 MB 수준으로 줄어드는 마법을 경험하실 수 있습니다. 추가로 특정 언어(Python, Java 등) 환경에 맞춘 최적화가 궁금하시면 말씀해 주세요.