---
layout: page
title:  "Producer / Consumer 애플리케이션 만들기"
date:   2025-07-07 10:00:00 +0900
permalink: /materials/S13-07-03-01_01-KafkaProducerConsumer
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



> - **호스트 PC(가상환경)에서 실행할 두 개의 독립된 파이썬 스크립트**
>   1. 데이터를 끊임없이 밀어 넣는 생산자(Producer)
>   2. 데이터를 실시간으로 가져와 처리하는 소비자(Consumer)
{: .common-quote}


## 1. Producer(생산자) 애플리케이션 작성

- 생산자는 1초마다 가상의 '사용자 행동 로그(JSON 형태)'를 생성하여 Kafka의 `my-official-topic`으로 전송

```python
#//file: "producer.py"
import time
import json
import random
from kafka import KafkaProducer

# 1. KafkaProducer 객체 생성
# 호스트 PC 기준이므로 외부 포트 3개를 모두 지정하여 고가용성을 확보합니다.
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"],
    # 전송할 객체(딕셔너리)를 JSON 문자열로 바꾼 뒤, 바이트(utf-8)로 직렬화(Serialization)합니다.
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    # 메시지가 특정 파티션으로 고정되도록 키(Key)도 바이트로 변환하는 설정을 넣습니다.
    key_serializer=lambda k: str(k).encode('utf-8')
)

topic_name = "my-official-topic"
print(f"Kafka Producer가 시작되었습니다. '{topic_name}' 토픽으로 메시지를 전송합니다.\n")

try:
    event_id = 1
    while True:
        # 가상의 실시간 스트리밍 데이터 생성
        user_id = random.randint(1000, 1010) # 10명의 유저 중 무작위
        actions = ["click", "view", "purchase", "logout"]
        
        payload = {
            "event_id": event_id,
            "user_id": user_id,
            "action": random.choice(actions),
            "timestamp": time.time()
        }
        
        # 메시지 전송 (Key를 user_id로 지정)
        # 동일한 user_id를 가진 이벤트는 항상 '동일한 파티션'으로 들어가 순서가 보장됩니다.
        producer.send(topic_name, key=user_id, value=payload)
        
        print(f"[전송 성공] Event #{event_id} | User: {user_id} | Action: {payload['action']}")
        
        event_id += 1
        time.sleep(1.0) # 1초에 한 번씩 전송

except KeyboardInterrupt:
    print("\n사용자에 의해 Producer가 종료되었습니다.")
finally:
    # 종료 전 버퍼에 남은 메시지를 강제로 밀어 넣고 연결을 닫습니다.
    producer.flush()
    producer.close()
```


## 2. Consumer(소비자) 애플리케이션 작성

- 소비자는 생산자가 보낸 메시지를 실시간으로 솎아내어 화면에 출력함
- **`my-analytics-group`**이라는 고유한 소비자 그룹을 형성함

```python
#//file: "consumer.py"
import json
from kafka import KafkaConsumer

# 1. KafkaConsumer 객체 생성
consumer = KafkaConsumer(
    "my-official-topic", # 구독할 토픽 이름
    bootstrap_servers=["localhost:9092", "localhost:9094", "localhost:9095"],
    group_id="my-analytics-group", # 소비자 그룹 ID (병렬 처리 및 복구의 기준)
    # 처음 접속 시 가장 오래된 메시지부터 읽어옵니다 ('latest'로 바꾸면 켠 순간부터 들어오는 것만 읽음)
    auto_offset_reset="earliest", 
    # 자동으로 오프셋을 커밋하도록 설정 (실무 수동 커밋은 복잡하므로 실습에선 자동 사용)
    enable_auto_commit=True,
    # 받아온 바이트 데이터를 다시 파이썬 딕셔너리(JSON) 객체로 역직렬화(Deserialization)합니다.
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    # 키값도 문자열로 변환하여 읽습니다.
    key_deserializer=lambda x: x.decode('utf-8') if x else None
)

print("Kafka Consumer가 대기 중입니다. 메시지가 들어오면 실시간으로 처리합니다...\n")

try:
    # consumer 객체 자체가 무한 루프 스트림 역할을 합니다 (지속적 쿼리/Pull 방식)
    for message in consumer:
        data = message.value
        partition = message.partition
        offset = message.offset
        key = message.key
        
        print(f"[수신 완료] 파티션: {partition} | 오프셋: {offset} | Key(User): {key}")
        print(f"    --> 내용: Event #{data['event_id']} - 유저 {data['user_id']}가 {data['action']}함.")
        print("-" * 50)

except KeyboardInterrupt:
    print("\n사용자에 의해 Consumer가 종료되었습니다.")
finally:
    # 클러스터에게 나간다고 알리고 세션을 정리합니다. (Rebalancing 유도)
    consumer.close()
```

## 3. 실습 진행 및 핵심 확인 방법 (매우 중요)

- **1단계: 소비자 먼저 켜기**
    - 새로운 터미널 창(가상환경 활성화 상태)을 열고 소비자를 먼저 실행

        ```bash
        python consumer.py
        ```

    - 아직 데이터가 없으므로 아무것도 출력되지 않고 대기 중

- **2단계: 생산자 켜기**
    - 또 다른 터미널 창을 열고 생산자를 실행

        ```bash
        python producer.py
        ```

    - 생산자 창에 `[전송 성공]` 로그가 1초마다 찍히기 시작
    - **동시에** 대기 중이던 소비자 창에서 데이터가 지연 없이 **실시간(`[수신 완료]`)으로 출력**됨

- **3단계: 파티션 분산 눈으로 확인 (이번 실습의 핵심)**
    - 소비자 화면에 찍히는 로그 확인

        ```text
        [수신 완료] 파티션: 2 | 오프셋: 45 | Key(User): 1005
        [수신 완료] 파티션: 0 | 오프셋: 31 | Key(User): 1001
        [수신 완료] 파티션: 2 | 오프셋: 46 | Key(User): 1005
        ```

    - **동일한 유저(Key)는 항상 동일한 파티션 번호로만 들어가는 것**을 볼 수 있음
        - 예: 유저 1005번의 데이터는 오직 파티션 2번으로만 순서대로 쌓임
    - 이를 통해 카프카가 
        - 전체 데이터는 분산 처리하면서도,
        - 특정 유저의 이벤트 순서는 완벽히 보장
    - 하는 스트리밍의 핵심 철학을 실무 코드로 증명
