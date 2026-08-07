---
layout: page
title:  "Dockerfile 작성 및 이미지 최적화"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S13-01-02-01_01-Dockerfile
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - **`Dockerfile`**은 도커 이미지를 만들기 위한 **자동화된 스크립트**
> - 단순히 앱을 실행하는 것을 넘어, 운영 환경에서 효율적으로 동작하는 이미지를 만들기 위해서는 작성 요령과 최적화 기법을 깊이 있게 이해해야 함.
{: .summary-quote}


## 1. Dockerfile의 주요 명령어와 역할

- Dockerfile은 아래에서 위로 쌓이는 **레이어(Layer)** 구조를 형성함

> - **`FROM`**: 모든 Dockerfile의 시작. 기반이 될 **베이스 이미지**를 지정함
> - **`WORKDIR`**: 컨테이너 내부에서 명령어가 실행될 **작업 디렉토리**를 설정 (없으면 자동 생성)
> - **`COPY` / `ADD`**: 호스트 머신의 파일을 컨테이너 내부로 복사 (`ADD`는 URL 다운로드나 압축 해제 기능이 추가로 포함됨)
> - **`RUN`**: 이미지를 빌드하는 동안 실행할 명령어 (패키지 설치, 파일 권한 변경 등)
> - **`CMD` / `ENTRYPOINT`**: 컨테이너가 시작될 때 실행할 명령
>   - `CMD`: 인자값이 변경될 수 있는 기본 명령
>   - `ENTRYPOINT`: 컨테이너의 주 목적으로 고정된 실행 명령
> - **`EXPOSE`**: 컨테이너가 사용할 포트 명시 (실제 포트 포워딩은 `docker run -p`에서 수행)
> - **`ENV`**: 환경 변수 설정
{: .common-quote}


## 2. 이미지 최적화의 필요성

- 이미지가 무거우면
    - **빌드 속도가 느려지고**,
    - 네트워크 전송 시 **대역폭을 낭비**하며,
    - 무엇보다 배포 시 **컨테이너 기동 속도**에 악영향을 줌
- 또한, 불필요한 도구가 포함되면 보안 취약점(Attack Surface)이 늘어남
- **`Dockerfile`** 작성의 성패는 "얼마나 캐시를 잘 활용하는가"와 "최종 이미지를 얼마나 가볍게 만드는가"에 달려 있음


## 3. 핵심 최적화 전략

> - 최적화 과정을 거치면 수 GB에 달하던 이미지가 수십 MB 수준으로 줄어드는 경험을 할 수 있음
> - **효율적인 빌드를 위한 Dockerfile 최적화 전략**
>   - 도커 이미지를 빌드할 때 효율성을 극대화하기 위한 두 가지 핵심 기법
>       - **레이어 캐싱(Layer Caching) 최적화**
>       - **멀티 스테이지 빌드(Multi-stage Build)**
>   - 실무에서는 빌드 속도와 이미지 용량이 배포 효율성을 결정짓는 핵심 지표가 됨
{: .common-quote}

1. **경량 베이스 이미지 선택 (Alpine & Slim)**
    - 일반적인 OS 이미지(Ubuntu 등)는 수백 MB를 차지 🡲 최소화된 이미지로 교체
    - 예시
        - **Alpine**
            - 5MB 내외의 초경량 리눅스 배포판
            - 보안이 강력하고 가볍지만, 일부 C 라이브러리 호환성(musl vs glibc)을 체크해야 함
        - **Slim**
            - 필요한 런타임(Python, Java 등)만 포함하고 불필요한 패키지를 제거한 버전

2. **레이어 수 최소화 (Layer Flattening)**
    - 도커는 `RUN`, `COPY`, `ADD` 명령마다 새로운 레이어를 생성함 🡲 비슷한 작업은 `&&`로 묶어 하나의 레이어로 합치는 것이 좋음
        - 나쁜 예(레이어 3개 생성)

            ```dockerfile
            RUN apt-get update
            RUN apt-get install -y python3
            RUN rm -rf /var/lib/apt/lists/*
            ```
        - 좋은 예 (레이어 1개 생성)

            ```dockerfile
            RUN apt-get update && apt-get install -y \
                python3 \
                && rm -rf /var/lib/apt/lists/*
            ```

3. **멀티 스테이지 빌드 (Multi-stage Build)** <span style="color: darkred;"> 🡲 **가장 중요**</span>
    - 빌드 시에만 필요한 도구(컴파일러, SDK 등)와 실행 시에 필요한 파일(바이너리, 라이브러리)을 분리하는 기법
        - 단일 `Dockerfile` 내에서 `FROM` 구문을 여러 번 사용하여 "빌드 단계(Build Stage)"와 "실행 단계(Run Stage)"를 완전히 분리함
            - 애플리케이션을 빌드할 때는 컴파일러, SDK, 빌드 툴 등 많은 도구가 필요하지만, 
            - **실제 서비스를 실행할 때는 소스 코드가 컴파일된 결과물(바이너리나 배포용 파일)만 있으면 됨**
    - 최종 이미지의 용량을 수 GB에서 수십 MB 단위로 줄일 수 있음
    - 불필요한 도구가 제거되어 보안성 극대화

    - **멀티 스테이지 빌드 적용 예시**
        - **Node.JS 기준**

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

        - **Python 기준**

            ```dockerfile
            # ---------------------------------------------------------
            # Stage 1: Build Stage (의존성 패키지 빌드 및 컴파일을 위한 환경)
            # ---------------------------------------------------------
            # C 확장 모듈이나 무거운 라이브러리 설치를 위해 빌드 도구(gcc 등)가 포함된 베이스 사용
            FROM python:3.9 AS builder
            WORKDIR /app

            # 캐시 효율을 위해 의존성 정의 파일 먼저 복사
            COPY requirements.txt .

            # --user 옵션을 사용하여 패키지를 로컬 사용자 디렉토리(.local)에 설치
            # 빌드 도구가 필요한 패키지들이 여기서 컴파일 및 빌드됩니다.
            RUN pip install --no-cache-dir --user -r requirements.txt

            # 소스 코드 복사 (컴파일 언어가 아니므로 이 단계에서는 소스만 준비)
            COPY . .

            # ---------------------------------------------------------
            # Stage 2: Run Stage (실제 서비스를 구동하기 위한 초경량 환경)
            # ---------------------------------------------------------
            # 실행 시에는 컴파일러가 필요 없으므로 매우 가벼운 -slim 이미지를 사용
            FROM python:3.9-slim
            WORKDIR /app

            # 중요: Stage 1(builder)의 결과물인 .local(설치된 패키지들) 폴더만 쏙 빼와서 복사함
            COPY --from=builder /root/.local /root/.local
            # 실제 애플리케이션 소스 코드 복사
            COPY --from=builder /app /app

            # 복사해온 패키지들을 파이썬 런타임이 인식할 수 있도록 환경 변수(PATH) 설정
            ENV PATH=/root/.local/bin:$PATH

            EXPOSE 5000

            # 최종 실행 명령 (최종 이미지에는 무거운 빌드 툴체인이 포함되지 않음)
            CMD ["python", "app.py"]
            ```

    - 이렇게 하면 최종 이미지에는 소스 코드나 빌드 도구가 포함되지 않아 용량이 극적으로 줄어듦<br><br>

    - **멀티 스테이지 빌드 전후 비교 결과**

        <div class="info-table">
        <table>
            <thead>
                <th style="width: 200px;">빌드 방식</th>
                <th style="width: 300px;">포함되는 구성 요소</th>
                <th style="width: 200px;">최종 이미지 용량</th>
                <th style="width: 200px;">보안성 (공격 표면)</th>
            </thead>
            <tbody>
                <tr>
                    <td class="td-rowheader">단일 스테이지 빌드</td>
                    <td>Go SDK + 소스 코드 + 빌드 툴 + 실행 파일</td>
                    <td>약 800 MB</td>
                    <td>위험 (컴파일 도구 노출)</td>
                </tr>
                <tr>
                    <td class="td-rowheader">멀티 스테이지 빌드</td>
                    <td>초경량 Linux(Alpine) + 실행 파일</td>
                    <td>약 20 MB</td>
                    <td>안전 (필수 파일만 존재)</td>
                </tr>
            </tbody>
        </table>
        </div>    

4. **레이어 캐싱(Layer Caching) 효율 극대화**
    
    - 도커는 이미지 빌드 시, `Dockerfile`의 명령어가 변경되지 않았다면 기존에 빌드된 레이어를 재사용(Caching) 함
        - **단 하나의 레이어라도 변경되면, 그 이후에 오는 모든 레이어는 캐시가 깨져(Cache Busting) 처음부터 다시 빌드** 됨
        - 따라서 명령어의 순서를 **"변하지 않는 파일 🡲 자주 변하는 파일"** 순으로 배치해야 빌드 속도를 획기적으로 줄일 수 있음

    - **변경이 잦은 파일(소스 코드)**은 Dockerfile의 뒷부분에, **변경이 적은 파일(종속성 설정)**은 앞부분에 배치할 것

        ```dockerfile
        COPY package.json .  # 라이브러리 목록을 먼저 복사
        RUN npm install      # 변경이 없는 한 캐시된 레이어 사용
        COPY . .             # 소스 코드는 자주 변하므로 마지막에 복사
        ```
        - 나쁜 예시 (캐시 효율 저하)

            ```dockerfile
            FROM node:18-alpine
            WORKDIR /app

            # 소스 코드와 설정 파일을 한 번에 복사
            COPY . .

            # 소스 코드가 1줄만 바뀌어도 아래의 무거운 패키지 설치과정이 매번 새로 실행됨
            RUN npm install

            CMD ["npm", "start"]

            ```

        - 좋은 예시 (캐시 최적화)

            ```dockerfile
            FROM node:18-alpine
            WORKDIR /app

            # 1. 자주 변하지 않는 의존성 정의 파일만 먼저 복사
            COPY package.json package-lock.json ./

            # 2. 패키지 설치 (package.json이 바뀌지 않았다면 이 무거운 작업은 캐시 처리됨)
            RUN npm install

            # 3. 자주 변하는 소스 코드는 가장 마지막에 복사
            COPY . .

            CMD ["npm", "start"]
            ```

5. **`.dockerignore` 파일 활용**
    - Git의 `.gitignore`처럼 이미지 빌드에 불필요한 파일(`.git`, `node_modules`, 로그 파일 등)이 컨테이너 내부로 복사되지 않도록 제외할 것

<br>

> - **요약**
>   - **캐시 최적화:** 소스 코드(`COPY . .`)보다 패키지 설치(`npm install`, `pip install` 등) 명령을 무조건 앞서 배치
>   - **용량 최적화:** 컴파일이 필요한 언어(Go, Java, C++)나 고도화된 프론트엔드 빌드(React, Vue) 환경에서는 **멀티 스테이지 빌드**가 필수<br><br>
> - **효율적인 Dockerfile 체크리스트**
>   1. **가장 가벼운 베이스 이미지**를 썼는가?
>   2. **멀티 스테이지 빌드**를 활용했는가?
>   3. **캐시를 고려하여 명령어 순서**를 배치했는가?
>   4. 불필요한 **임시 파일(캐시 삭제 등)**을 `RUN` 명령 끝에 지웠는가?
>   5. **.dockerignore**로 불필요한 파일을 걸러냈는가?
{: .summary-quote}
