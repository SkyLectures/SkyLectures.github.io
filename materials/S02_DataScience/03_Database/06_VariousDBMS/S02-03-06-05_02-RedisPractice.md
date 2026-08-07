---
layout: page
title:  "Redis 활용 및 실습"
date:   2026-05-31 09:00:00 +0900
permalink: /materials/S02-03-06-05_02-RedisPractice
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. Redis 접속 및 기본 조작법

- **사전 준비**
    - **가상환경 생성 및 활성화**

    ```bash
    python -m venv redis
    cd redis
    source ./bin/activate
    ```

    - 필수 라이브러리 설치

    ```bash
    pip install redis
    ```

    - **샘플데이터 작성**

        ```python
        import redis

        # Redis 연결 (decode_responses=True 설정으로 문자열로 깔끔하게 처리)
        rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        print("안전한 실습을 위해 기존 데이터를 먼저 청소합니다...")
        rd.flushdb()

        print("\n[1/4] Strings(단순 키-값) 샘플 데이터 주입 중...")
        rd.set("user:101:name", "김철수")
        rd.set("user:101:email", "chulsoo@example.com")
        rd.set("server:status", "ONLINE")
        rd.set("page:views", 42) # 숫자형태의 데이터

        print("[2/4] Lists(순서가 있는 줄 세우기) 샘플 데이터 주입 중...")
        # 'recent:search'라는 검색어 기록 리스트에 순서대로 넣기
        rd.rpush("recent:search", "레디스 설치")
        rd.rpush("recent:search", "파이썬 기초")
        rd.rpush("recent:search", "도커 데스크탑")

        print("[3/4] Sets(중복 없는 주머니) 샘플 데이터 주입 중...")
        # 'user:101:tags'라는 주머니에 관심사 태그 넣기
        rd.sadd("user:101:tags", "프로그래밍")
        rd.sadd("user:101:tags", "데이터베이스")
        rd.sadd("user:101:tags", "프로그래밍") # 일부러 중복 데이터 주입

        print("[4/4] Sorted Sets(점수별 정렬 집합) 샘플 데이터 주입 중...")
        # 'game:leaderboard'에 유저별 점수와 함께 주입
        rd.zadd("game:leaderboard", {"Player_A": 2500, "Player_B": 4200, "Player_C": 1800})

        print("\n모든 실습용 샘플 데이터가 성공적으로 채워졌습니다!")
        print("이제 터미널(redis-cli)로 가셔서 아래 가이드를 보며 실습을 시작하세요!")
        ```


- **`redis-cli`를 입력해 Redis 작업 시작**

    ```bash
    redis-cli
    ```

- **서버 연결 확인하기 (`PING`)**
    - 사용법: `PING`
    - 설명
        - 서버가 살아있는지 노크하는 명령어
        - 정상적이라면 `PONG`이라는 답변이 돌아옴

- **데이터 확인 및 관리를 위한 필수 명령어**

    - **현재 저장된 모든 키 확인 (`KEYS`)**
        - 사용법: `KEYS *`
        - 주의
            - 실습 환경에서 유용
            - 상용 서버(Production)에서 데이터가 수천만 건일 때 이 명령어를 치면 🡲 싱글 스레드 특성상 서버가 일시 마비됨
                - 학습 단계를 넘어서면 `SCAN` 명령어를 쓰는 습관을 들여야 함

    - **안전하게 키 목록 나누어 조회 (`SCAN`)**
        - 사용법: `SCAN cursor [MATCH pattern] [COUNT count]`
        - 설명: 
            - 데이터베이스 전체를 멈추지 않고, 커서(Cursor)를 이용해 키 목록을 조금씩 쪼개어(안전하게) 검색함
            - 첫 시작은 `SCAN 0`으로 하며, 결과로 나오는 다음 커서 번호를 가지고 연속해서 호출하는 방식임
        - 중요:
            - 대량의 데이터가 쌓인 상용 서버에서 전체를 긁어오는 `KEYS *`를 치면 서버가 마비되므로,
            - 실무에서는 반드시 이 `SCAN` 명령어를 사용하는 것이 글로벌 표준 보안 규칙임

    - **키 존재 여부 확인 (`EXISTS`)**
        - 사용법: `EXISTS key`
        - 설명:
            - 특정 키(방)가 현재 데이터베이스에 존재하는지 체크함
            - 키가 존재하면 `1`, 존재하지 않으면 `0`을 반환함
            - 무작정 데이터를 읽거나 쓰기 전에 안전하게 존재 여부를 검사할 때 사용함

    - **데이터 전체 삭제 (`FLUSHDB`)**
        - 사용법: `FLUSHDB`
        - 설명
            - 현재 데이터베이스의 모든 데이터를 깨끗하게 지움
            - 작업/실습을 처음부터 다시 깔끔하게 시작하고 싶을 때 사용함

    - **특정 데이터만 골라서 삭제 (`DEL`)**
        - 사용법: `DEL key`
        - 설명: 
            - `FLUSHDB`처럼 전체를 다 지우지 않고, 지정한 특정 키(방) 하나만 조각내어 삭제함
            - 삭제에 성공하면 `1`, 없는 키를 지우려고 하면 `0`을 반환함
            - 로그아웃한 특정 유저의 세션 데이터를 지우거나, 유효기간이 끝난 임시 정보를 수동으로 파괴할 때 사용함

    - **남은 만료 시간 확인 (`TTL`)**
        - 사용법: `TTL key`
        - 설명: 
            - `SETEX` 등으로 설정한 데이터의 시한부 생명(만료 시간)이 현재 몇 초 남아있는지 실시간으로 알려줌
            - 만료 시간이 지정되지 않은 일반 데이터는 `-1`, 이미 만료되어 사라진 키는 `-2`를 반환함
            - 캐시 데이터나 인증 토큰이 언제 파기되는지 백엔드 로직을 디버깅하고 모니터링할 때 필수적으로 사용함

<br>

> - **Redis에서 키(Key)가 가지는 본질적인 의의**
>   - 데이터의 '유일한 주소'이자 '절대적 이정표'
>       - Redis는 데이터를 찾을 때, 데이터의 내부 상태를 보고 찾지 않음
>       - 오직 데이터가 저장될 때 부여된 '방 번호(Key)'만 보고 찾아감
>       - Redis의 키는 메모리 공간에 존재하는 수많은 데이터 주머니 중 원하는 것을 단 한 번에 찾아가기 위한 유일무이한 내비게이션 주소를 의미
>   - 스키마(Schema)를 대체하는 구조화 도구
>       - Redis에는 표(Table)나 폴더 구조가 없음
>       - 대신 개발자들은 키 이름에 콜론(:) 규칙을 넣어서 데이터의 구조를 표현함
>           - 예시:
>               - user:1001:profile 🡲 "1001번 유저의 프로필 데이터구나"
>               - shop:best_sellers:top5 🡲 "쇼핑몰의 베스트셀러 상위 5개 데이터구나"
>       - 이처럼 키 이름 자체가 데이터의 정체성과 분류(카테고리)를 결정하는 역할을 함
{: .common-quote}

<br>

> - **왜 모든 주요 내용이 '키의 검색'을 중심으로 흘러갈까?**
>   - **Redis의 내부 아키텍처: 거대한 해시 맵 (Hash Map)**
>       - Redis가 초당 수십만 건의 데이터를 마이크로초 단위로 처리할 수 있는 비결은 내부적으로 해시 테이블(Hash Table) 구조를 사용하기 때문
>       - 해시 테이블은 키를 입력하면 그 즉시 데이터가 있는 메모리 주소를 출력함
>       - 데이터가 10개든, 1억 개가 있든 관계없이 키를 통해 데이터를 찾아가는 시간 복잡도는 항상 $$O(1)$$(상수 시간)
>       - 따라서 Redis의 압도적인 속도를 온전히 활용하려면, 반드시 "키를 정확히 알고 지목하여 검색하는 방식"으로 접근해야함
>   - **"값(Value)"을 기준으로 한 검색이 불가능에 가깝기 때문**
>       - MySQL에서는 인덱스(Index)를 타지 않더라도 WHERE content LIKE '%비트코인%' 같은 쿼리를 날려 값 내부를 뒤질 수 있음
>           - 성능은 느려지더라도 검색은 가능
>       - Redis는 기본적으로 값(Value)의 내용물을 필터링하여 검색하는 기능이 매우 제한적
>           - 1,000만 명의 유저 세션이 Strings 구조로 들어가 있다면,
>               - "이름이 홍길동인 유저의 세션 키를 찾아줘"라는 명령을 내릴 수 없음
>               - 값을 다 확인하려면 Redis 서버 전체를 멈추고 1,000만 개를 다 열어봐야 하기 때문
>       - 결국 원하는 데이터를 찾으려면 설계 단계에서부터
>           - "나는 이 데이터를 user:name:홍길동이라는 키로 저장해두고, 나중에 이 키로만 찾겠다"라는 규칙을 정해야 함
>       - 데이터의 탐색과 제어가 100% 키 중심으로 돌아갈 수밖에 없는 구조적 원인
>   - **싱글 스레드를 보호하기 위한 규칙**
>       - Redis는 싱글 스레드 기반 🡲 개발자가 키 관리를 허술하게 하여 "어떤 키들이 있는지 기억이 안 나네? 일단 다 뒤져봐야지" 하고 데이터 주머니를 헤집는 순간 서버가 다운됨
>       - 내가 원하는 키가 어디에 있는지, 어떤 패턴(예: user:*)으로 묶여 있는지 명확히 인지하고 제어하는 것이 Redis 운영의 전부
>       - 따라서 모든 교재와 예제가 키 검색과 관리를 최우선으로 가르치는 것
{: .expert-quote}

## 2. Redis 활용 기초

### 2.1 기본 데이터 타입
    
> - Redis는 단순 캐시(Memcached 등)와 달리, '값(Value)' 자리에 단순 텍스트뿐만 아니라 다양한 형태의 주머니(자료구조)를 넣을 수 있음
{: .common-quote}

- **Strings (가장 단순한 상자)**
    - 가장 기본이 되는 타입
    - 하나의 Key에 하나의 Value(텍스트, 숫자, 이미지 등)를 1:1로 저장함
    - 주요 용도: 로그인 토큰 저장, 단순 방문자 수 카운팅, 웹페이지 HTML 통째로 캐싱하기

- **Lists (순서가 있는 줄 세우기)**
    - 기차 칸처럼 데이터들이 앞뒤로 연결된 형태(Linked List)
    - 데이터를 맨 앞에 넣거나, 맨 뒤에서 빼는 작업이 매우 빠름
    - 주요 용도: 먼저 들어온 요청을 먼저 처리하는 '큐(Queue)' 시스템, 최근에 본 상품 목록 저장

- **Sets (중복을 허용하지 않는 주머니)**
    - 주머니 안에 데이터들을 무작위로 던져 넣는 형태
    - 가장 큰 특징은 중복을 알아서 제거해 준다는 점과, 집합 연산(교집합, 합집합)이 가능하다는 점
    - 주요 용도: 오늘 우리 사이트에 방문한 '중복 없는 사용자 수(UV)' 카운팅, 친구 추천 기능(서로 공통으로 아는 친구 교집합 구하기)

- **Sorted Sets (순서대로 정렬되는 줄 세우기)**
    - Sets와 똑같이 중복을 허용하지 않으며, 데이터를 넣을 때 '점수(Score)'라는 꼬리표를 함께 붙여서 넣음
    - Redis는 이 점수를 기준으로 데이터를 항상 정렬된 상태로 유지함
    - 주요 용도: 게임 실시간 스코어보드, 포털 사이트 실시간 인기 검색어 순위

### 2.2 핵심 메커니즘

> - 인메모리 저장소인 Redis에게 가장 무서운 적은 "메모리 고갈(Out Of Memory)"
> - RAM은 용량이 제한되어 있고 비용이 비싸기 때문에 방치하면 금방 가득 차서 서버가 뻗어버림
{: .common-quote}

- **TTL (Time To Live)**
    - 메모리 고갈 문제를 방지하기 위해 Redis는 데이터마다 '시한부 생명(만료 시간)'을 부여하는 기능(TTL)을 제공
    - "이 데이터는 지금부터 딱 10분만 살아있어라"라고 설정 🡲 Redis가 시간을 체크하고 있다가 10분이 지나는 순간 메모리에서 삭제
    - 활용
        - '임시 데이터'를 관리할 때 별도의 삭제 로직을 짤 필요가 없음 🡲 혁신적으로 편리함
            - 3분간만 유효한 휴대폰 인증번호, 로그인 후 30분간만 유지되는 사용자 세션 등


### 2.3 Redis의 기본 작동 원리 실습

- **Redis 클라이언트 실행**

    ```bash
    redis-cli --raw
    ```

    - 문자열의 문자코드가 맞지 않을 경우, 우리가 읽기 어려운 코드로 출력됨
        - 아스키(ASCII) 범위를 벗어나는 문자(한글, 일어, 이모지 등)를 만나면 사람이 읽을 수 없게 변환하여 출력함
    - `--raw` 옵션을 주면 원본 그대로의 데이터를 출력함

- **CLI 대화형 실습 스크립트**
    1. **String 기본 다루기 (`SET`과 `GET`)**
        - 가장 단순한 "상자(Key)"에 "물건(Value)"을 넣고 꺼내는 과정

        ```bash
        # 1. 'user:name'이라는 상자에 '홍길동'을 저장해줘
        127.0.0.1:6379> SET user:name "홍길동"
        (응답) OK

        # 2. 'user:name' 상자에 뭐가 들어있는지 꺼내와줘
        127.0.0.1:6379> GET user:name
        (응답) "홍길동"

        # 3. 없는 상자를 열려고 하면 어떻게 될까?
        127.0.0.1:6379> GET user:age
        (응답) (nil)  <-- 'nil'은 아무것도 없다는 뜻(Null)
        ```

    2. **안전한 계산기 기능 사용하기 (`INCR`)**
        - 숫자를 1씩 더하는 원자적(Atomic) 카운터
        - 값이 없으면 0에서부터 시작함

        ```bash
        # 1. 'page:views'라는 키의 숫자를 1 올려줘
        127.0.0.1:6379> INCR page:views
        (응답) (integer) 1  <-- 방이 없었기 때문에 1을 만들고 결과를 알려줌

        # 2. 한 번 더 실행
        127.0.0.1:6379> INCR page:views
        (응답) (integer) 2  <-- 기존 1에 1이 더해져 2가 됨

        # 3. 진짜 잘 저장되었는지 조회해보기
        127.0.0.1:6379> GET page:views
        (응답) "2"

        # [실패 테스트] 글자가 들어있는 상자에 더하기를 시도하면 어떻게 될까?
        127.0.0.1:6379> INCR user:name
        (응답) (error) ERR value is not an integer or out of range
        # (해석: 에러! 홍길동이라는 글자는 숫자가 아니라서 더할 수 없음)
        ```

    3. **데이터에 시한부 생명 부여하기 (`SETEX`)**
        - 메모리 고갈을 막기 위해 지정한 시간(초)이 지나면 자동으로 사라지게 만듦

        ```bash
        # 1. 'auth:code' 상자에 '1234'를 넣고, 딱 '5초'만 살려둬! (SETEX = SET + EXpire)
        127.0.0.1:6379> SETEX auth:code 5 "1234"
        (응답) OK

        # 2. 넣자마자 빛의 속도로 바로 조회해보기 (5초가 지나기 전에!)
        127.0.0.1:6379> GET auth:code
        (응답) "1234"  <-- 아직 살아있음

        # 3. 마음속으로 5초를 세어본 뒤(하나, 둘, 셋, 넷, 다섯...) 다시 조회해보기
        127.0.0.1:6379> GET auth:code
        (응답) (nil)  <-- 데이터가 자동 소멸됨
        ```

> - **Key 이름의 콜론(`:`) 규칙**
>   - Redis에서는 상자 이름(Key)을 지을 때 `user:name`, `page:views` 처럼 콜론(`:`)을 자주 사용
>   - 관계형 DB의 테이블명과 컬럼명을 흉내 내어 사람이 읽기 쉽게 폴더 구조처럼 구조화하는 Redis 진영의 관례(Best Practice)
{: .common-quote}


## 3. 기능별 독립 예제

> - **학습 행동 가이드**
>   1. 터미널에서 `redis-cli` 실행
>   2. 위의 명령어들(`SET`, `GET`, `INCR`, `ZADD` 등)을 임의의 값으로 2~3번씩 직접 타이핑해 보며 값이 어떻게 변하는지 눈으로 확인
>   3. "아, 명령어가 이렇게 움직이는구나!" 감이 오면, **기능별 개별 파이썬 예제 코드**를 하나씩 실행해 보면서 파이썬 라이브러리가 레디스를 제어하는 방식을 매칭
>   4. 마지막으로 **통합 아키텍처 예제**를 실행하여 거대한 거래소 흐름을 한눈에 파악
{: .common-quote}

> - **제안 예제의 장점**
>   1. **시각적 피드백:**
>       - RDBMS를 쓸 때 0.5초 걸리던 화면 조회
>       - Redis를 거치자마자 0.00초로 바뀌는 것을 눈으로 직접 보며 인메모리의 위력을 쉽게 체감
>   2. **복잡한 인프라 지식 불필요:**
>       - 원래 분산 환경의 동시성 제어나 실시간 정렬, 메시지 큐를 구현하려면 매우 복잡한 백엔드 코드가 필요함
>       - Redis는 **명령어 단 한 줄**로 이 기능을 제공
>       - 즉, "복잡한 내부 로직은 Redis가 다 해줄 테니, 초급자는 가져다 쓰기만 하세요"의 관점이라 초급자에게 더 친숙함
{: .common-quote}


### 3.1 Look-Aside 캐시 레이어

- **주요 주제:**
    - 수시로 변하는 가상자산 시세 및 마켓 정보를 빠르게 제공하기 위한 **Look-Aside 캐싱**
        - String & TTL 기반의 'Look-Aside 캐시 레이어'
        - 인메모리 데이터 캐싱 및 자동 만료를 통한 디스크 DB 보호

- **의의:**
    - 변동성이 있는 외부 데이터나 무거운 DB 조회 결과를 메모리에 임시 저장하여 API 응답 속도 극대화

- **알아야 할 기술: Strings & TTL**
    - **가장 단순한 상자(Key)에 물건(Value)을 넣고 빼는 구조**
        - **`SET key value` / `GET key**`
        - `SET user:name "Kim"` 🡲 `user:name`이라는 방에 "Kim"을 저장
        - `GET user:name` 🡲 "Kim"을 꺼내옴

    - **시간 제한 걸기 (`SETEX`)**
        - `SETEX token:101 10 "secret"` 🡲 `token:101`에 "secret"을 저장하되, 10초 뒤에 자동으로 삭제(Expire)하라는 의미
        - 캐시 시스템의 유효 기간을 설정할 때 필수적으로 사용됨

- **체감 난이도**
    - **단계 1: 캐시 레이어 (`GET`, `SETEX`) 🡲 [초급]**
    - **설명:** 
        - 변수에 값을 넣고(`SET`), 가져오는(`GET`) 수준의 가장 기초적인 문법
        - 여기에 '몇 초 뒤에 자동으로 사라지게 해라'라는 시간 설정(`TTL`) 하나만 추가된 형태라 초급자도 직관적으로 이해할 수 있음

```python
#//file: "look_aside.py"
import redis
import json
import time

# 1. 독립된 Redis 연결 및 데이터 초기화
rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
rd.delete("crypto:market:BTC") # 이전 실습 데이터 오염 방지

def get_heavy_db_data():
    """가상의 디스크 기반 DB 조회 (0.5초 대기)"""
    time.sleep(0.5) 
    return {"ticker": "BTC", "price": 95000000, "volume_24h": 12500}

def get_market_data_with_cache(ticker):
    cache_key = f"crypto:market:{ticker}"
    
    # Cache Hit 확인
    cached_data = rd.get(cache_key)
    if cached_data:
        return json.loads(cached_data), "Cache Hit! (Redis)"
    
    # Cache Miss 시 데이터 생성 및 3초 동안 캐싱
    db_data = get_heavy_db_data()
    rd.setex(cache_key, 3, json.dumps(db_data))
    return db_data, "Cache Miss... (DB 조회 후 캐싱)"

# --- 실행 및 검증 ---
print("--- [예제 1] 캐시 테스트 시작 ---")
# 1회차: 캐시에 없으므로 0.5초 지연 발생
start = time.time()
data, status = get_market_data_with_cache("BTC")
print(f"1차 요청: {status} | 소요시간: {time.time() - start:.4f}초")

# 2회차: 캐시에 존재하므로 즉시 반환
start = time.time()
data, status = get_market_data_with_cache("BTC")
print(f"2차 요청: {status} | 소요시간: {time.time() - start:.4f}초")
```


### 3.2 API Rate Limiter

- **주요 주제:**
    - 초당 수만 건의 매수/매도 요청 속에서 시스템을 보호하기 위한 **IP별 API Rate Limiter**
        - Atomic Counter & TTL 기반의 'API Rate Limiter'
        - 싱글 스레드 원자적 연산을 이용한 분산 환경 동시성 제어 및 처리율 제한

- **의의:**
    - 동시성 문제를 해결하는 안전한 계산기 기능
    - 특정 사용자의 과도한 API 호출을 차단하여 인프라 시스템 전체의 붕괴를 방지

- **알아야 할 기술: Atomic Counter**    
    - **숫자 1씩 더하기 (`INCR`)**
        - `INCR visitor:count` 🡲 `visitor:count`라는 키의 값을 1 증가시킴
            - 만약 키가 없었다면 0에서 1로 만들고 시작
        - **왜 중요할까?**
            - 일반 데이터베이스는 
                - "현재 값 조회 🡲 프로그램에서 1 더하기 🡲 다시 저장"이라는 3단계를 거침
                - 동시에 100명이 몰리면 계산이 꼬임
            - Redis의 `INCR`은
                - 그 자체로 완벽하게 쪼갤 수 없는 단 하나의 연산(Atomic)으로 처리
                - 절대 계산이 틀리지 않음

    - **기존 키에 만료 시간만 부여하기 (`EXPIRE`)**
        - `EXPIRE visitor:count 5` 🡲 이미 존재하는 키가 5초 뒤에 사라지도록 타이머를 맞춤

- **체감 난이도**
    - **단계 2: API 속도 제한기 (`INCR`, `EXPIRE`) 🡲 [초급]**
    - **설명:**
        - 숫자를 1씩 더하는 컴퓨터의 가장 기본적인 카운터(Counter) 기능
        - "숫자가 3보다 크면 차단한다"라는 단순한 `if` 조건문 분기이기 때문에 코딩 기초 지식만 있으면 쉽게 따라올 수 있음

```python
#//file: "api_rate_limit.py"
import redis
import time

rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
user_id = "attacker_ip_127_0_0_1"
rd.delete(f"user:rate:{user_id}") # 데이터 초기화

def is_api_allowed(uid):
    limit_key = f"user:rate:{uid}"
    max_allowed = 2      # 5초당 최대 2회 허용
    window_time = 5      # 5초의 유효 시간윈도우
    
    # INCR 명령어는 원자적으로 증가하며 격리성(Locking) 문제를 해결함
    current_count = rd.incr(limit_key)
    
    if current_count == 1:
        rd.expire(limit_key, window_time)
        
    if current_count > max_allowed:
        return False, f"거부 (호출 횟수: {current_count}/{max_allowed})"
    return True, f"허용 (호출 횟수: {current_count}/{max_allowed})"

# --- 실행 및 검증 ---
print("\n--- [예제 2] API 처리율 제한 테스트 시작 ---")
for i in range(1, 5):
    allowed, message = is_api_allowed(user_id)
    print(f"{i}번째 API 호출 요청 -> {message}")
    time.sleep(0.2)
```


### 3.3 실시간 거래량 랭킹 보드

- **주요 주제:**
    - 고성능 메모리 연산을 활용한 거래소 거래대금 기준 **실시간 거래량 랭킹 보드(Leaderboard)**
        - Sorted Set 기반의 '실시간 거래량 랭킹 보드'
        - 가중치(Score)를 내장한 구조화 데이터 셋 활용 ($$O(\log N)$$ 정렬)

- **의의:**
    - 데이터를 저장할 때 '점수'를 꼬리표로 달아주면 Redis가 실시간으로 자동 정렬해 주는 기능
    - 실시간으로 발생하는 대규모 점수 변경 레이스를 디스크 정렬(`ORDER BY`) 없이 실시간 중계

- **알아야 할 기술: Sorted Set (ZSET)**
    - **데이터 정렬해서 넣기 (`ZADD`)**
        - `ZADD g:rank 100 "User_A"` 🡲 `g:rank`라는 랭킹 보드에 "User_A"를 100점과 함께 넣기

    - **점수 실시간으로 누적하기 (`ZINCRBY`)**
        - `ZINCRBY g:rank 50 "User_A"` 🡲 "User_A"의 기존 점수에 50점을 더해 총 150점으로 만들고, 순위를 즉시 재배치

    - **높은 점수 순으로 조회하기 (`ZREVRANGE`)**
        - `ZREVRANGE g:rank 0 2 WITHSCORES` 🡲 랭킹 보드에서 가장 점수가 높은 1등(0등)부터 3등(2등)까지 점수와 함께 보여달라는 명령
        - RDBMS의 `ORDER BY DESC LIMIT 3`을 메모리 상에서 초고속으로 수행하는 것과 같음

- **체감 난이도**
    - **단계 3: 실시간 랭킹 (`ZADD`, `ZINCRBY`) 🡲 [초급~중급]**
    - **설명:**
        - 데이터를 넣을 때 점수(Score)를 같이 넣으면, Redis가 알아서 내부적으로 순위를 정렬해 주는 기능
        - 복잡한 정렬 알고리즘을 코딩하는 것이 아니라 "Redis야, 이 사람한테 500점 더해줘"라는 명령어 한 줄만 사용
            - 사용법 자체는 초급 수준

```python
#//file: "leader_board.py"
import redis

rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
rank_key = "crypto:leaderboard:volume"
rd.delete(rank_key) # 데이터 초기화

# 1. 고래 유저 샘플 데이터 직접 준비 및 주입
initial_data = {"Whale_Alpha": 1500, "Whale_Bravo": 4200, "Whale_Charlie": 2800}
rd.zadd(rank_key, initial_data)

print("\n--- [예제 3] 실시간 랭킹 시스템 시작 ---")
print("현재 실시간 거래대금 순위:")
for rank, (user, score) in enumerate(rd.zrevrange(rank_key, 0, -1, withscores=True), start=1):
    print(f"{rank}위: {user} ({score} BTC)")

# 2. 실시간 데이터 변동 발생 (Whale_Alpha가 대규모 매수 체결)
print("\n[실시간 이벤트] Whale_Alpha 유저가 3000 BTC를 추가 거래했습니다.")
rd.zincrby(rank_key, 3000, "Whale_Alpha")

print("\n변동된 실시간 거래대금 순위 (Top 2):")
for rank, (user, score) in enumerate(rd.zrevrange(rank_key, 0, 1, withscores=True), start=1):
    print(f"{rank}위: {user} ({score} BTC)")
```


### 3.4 분산 이벤트 메시지 브로커

- **주요 주제:**
    - 체결된 주문 내역을 다른 마이크로서비스(알림, 정산 등)로 안전하게 토스하는 **Pub/Sub 이벤트 브로커**
        - Pub/Sub 기반의 '분산 이벤트 메시지 브로커'
        - 발행-구독 모델을 통한 마이크로서비스 간 비동기 결합도 완화

- **의의:**
    - 라디오 방송국(Publisher)과 청취자(Subscriber)의 관계를 만들어주는 기능
    - 핵심 로직(주문)과 부가 로직(알림/정산)을 분리하여 시스템 확장성과 내결함성을 높임

- **알아야 할 기술: Pub/Sub**
    - **방송 청취하기 (`SUBSCRIBE`)**
        - `SUBSCRIBE news:chat` 🡲 `news:chat`이라는 채널(주파수)을 귀 기울여 듣기 시작
        - 이 명령을 치면 터미널은 다음 메시지가 올 때까지 대기 상태에 빠짐

    - **방송 송출하기 (`PUBLISH`)**
        - (다른 터미널 창을 열고) `PUBLISH news:chat "Hello World"` 🡲 `news:chat` 채널을 듣고 있는 모든 사람에게 "Hello World"라는 메시지를 동시에 전송

- **체감 난이도**
    - **단계 4: 메시지 브로커 (`PUBLISH`, `SUBSCRIBE`) 🡲 [중급 입문]**
    - **설명:**
        - 유튜브 채널을 구독하고, 알림을 받는 개념과 동일
        - 개념적으로는 '분산 아키텍처'라는 단어가 들어가서 어려워 보이지만, 실습 코드는 "이 채널로 메시지 던져줘", "그 채널에 메시지 왔나 확인해줘"가 전부라 개념 이해용으로 아주 좋음

```python
#//file: "pub_sub.py"
import redis
import json

rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
channel = "order:events"

print("\n--- [예제 4] 분산 이벤트 메시지 브로커 시작 ---")
# 1. 구독자(Subscriber) 설정 및 채널 리스닝 시뮬레이션
sub_client = rd.pubsub()
sub_client.subscribe(channel)

# 2. 발행자(Publisher)가 주문 체결 이벤트 샘플 데이터 발행
event_payload = {"order_no": "TXT-001", "ticker": "ETH", "amount": 10, "status": "EXECUTED"}
print(f"발행처(주문엔진) -> 이벤트를 채널 [{channel}]에 전송합니다.")
rd.publish(channel, json.dumps(event_payload))

# 3. 구독처가 메시지를 수신하여 비동기 처리 수행
msg = sub_client.get_message(ignore_subscribe_messages=True)
if msg:
    data = json.loads(msg['data'])
    print(f"수신처(알림서비스) -> [비동기 알림 처리완료] 주문번호 {data['order_no']}의 {data['ticker']} 거래 내역 통지 완료.")
```

- 실습 방법
    1. 새로운 터미널 창에서 `redis-cli --raw'를 실행

        ``` bash
        redis-cli --raw
        ```
    
    2. 수신 대기

        ```bash
        127.0.0.1:6379> SUBSCRIBE order:events
        ```

    3. 예제 코드 실행

        ```bash
        python pub_sub.py
        ```

    4. 수신 대기 창에서 메시지 수신 상태 확인


## 4. 통합 예제(Integration)

- **주제**
    - 가상자산 거래 주문 파이썬 파이프라인 API
    - 위의 4가지 독립 컴포넌트를 하나의 유기적 아키텍처로 엮기
    - 사용자가 API를 호출하여 시세를 조회하고, 제한 속도 내에서 주문을 체결한 뒤, 랭킹에 반영되고 알림이 발송되는 전체 라이프사이클

- **핵심 학습 포인트**
    - 하나의 단일 백엔드 API 라이프사이클 안에서, 
    - Redis가 서로 다른 데이터 구조(String, Sorted Set)와 통신 메커니즘(Pub/Sub)을 적재적소에 배치하여
    - RDBMS 및 애플리케이션의 메모리 부하를 복합적으로 방어하고
    - 설계 완성도를 높이는 방식의 전반적인 이해


```python
import redis
import json
import time

class CryptoExchangeEngine:
    def __init__(self):
        # 전체 통합 관리용 레디스 객체 생성
        self.rd = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.rd.flushdb() # 시스템 초기 데이터 셋을 위해 초기화
        
        # 통합 시스템 초기 샘플 데이터 셋업
        self.rd.zadd("integrated:leaderboard", {"Trader_A": 100, "Trader_B": 50})
        self.pubsub = self.rd.pubsub()
        self.pubsub.subscribe("integrated:notifications")
        print("가상자산 통합 거래소 엔진 아키텍처가 활성화되었습니다.\n")

    def run_pipeline(self, user_id, ticker, order_amount):
        print(f"유저 [{user_id}]의 요청 프로세스 시작 ===")
        
        # [통합 단계 1] Rate Limiter 작동
        limit_key = f"integrated:rate:{user_id}"
        if self.rd.incr(limit_key) > 2: # 2회 초과 시 차단
            print(f"[Rate Limiter] {user_id} 접근 제한: 과도한 API 요청으로 주문이 거부되었습니다.\n")
            return False
        self.rd.expire(limit_key, 10) # 10초 윈도우 생성
        print("[Rate Limiter] 통과: 정상적인 접근 유저입니다.")

        # ⚡ [통합 단계 2] 시세 캐시 레이어 작동 (Look-Aside)
        cache_key = f"integrated:market:{ticker}"
        cached_ticker = self.rd.get(cache_key)
        
        if cached_ticker:
            market_info = json.loads(cached_ticker)
            print(f"[Cache Layer] Cache Hit! 실시간 {ticker} 시세 반영 -> {market_info['price']}원")
        else:
            # Cache Miss인 경우 가상 DB 생성 후 캐싱 데이터 등록 (유효시간 5초)
            market_info = {"ticker": ticker, "price": 95000000 if ticker=="BTC" else 4000000}
            self.rd.setex(cache_key, 5, json.dumps(market_info))
            print(f"[Cache Layer] Cache Miss! DB 연산 후 {ticker} 시세 세팅 완료.")

        # [통합 단계 3] 주문 처리 및 실시간 랭킹(Sorted Set) 업데이트
        total_value = order_amount * market_info['price']
        # 거래 대금 기준 랭킹 가산 점수 부여
        self.rd.zincrby("integrated:leaderboard", order_amount, user_id)
        print(f"[Order Engine] 주문 완료! 총액: {total_value:,}원 분량 체결.")

        # [통합 단계 4] 이벤트 전파 (Pub/Sub)
        notification_payload = {"user_id": user_id, "msg": f"{ticker} {order_amount}개 주문 정상 체결"}
        self.rd.publish("integrated:notifications", json.dumps(notification_payload))
        
        # 비동기 메시지 수신부 핸들링
        event_msg = self.pubsub.get_message(ignore_subscribe_messages=True)
        if event_msg:
            evt_data = json.loads(event_msg['data'])
            print(f"[Pub/Sub Event] 알림 서버가 이벤트를 수신함 🡲 {evt_data['user_id']}님에게 '{evt_data['msg']}' 알림 발송 완료.")
            
        # 현재 실시간 전체 고래 랭킹 출력
        print("[Current Leaderboard Top 3]")
        for rank, (user, score) in enumerate(self.rd.zrevrange("integrated:leaderboard", 0, 2, withscores=True), start=1):
            print(f"   {rank}위: {user} (누적 {score} 개)")
        print(f"유저 [{user_id}]의 요청 프로세스 종료 ===\n")
        return True

# --- 통합 시스템 시뮬레이션 가동 ---
exchange = CryptoExchangeEngine()

# Scenario A: 정상적인 유저 거래 진행 (Cache Miss 발생)
exchange.run_pipeline("Trader_A", "BTC", 2.5)

# Scenario B: 동일 유저의 반복 거래 (Cache Hit 발생 및 Leaderboard 순위 역전)
time.sleep(1)
exchange.run_pipeline("Trader_A", "BTC", 5.0)

# Scenario C: 디도스/악성 매크로 형태의 급격한 반복 요청 시도 (Rate Limiter에 의한 차단 처리)
exchange.run_pipeline("Trader_A", "BTC", 1.0)
```
