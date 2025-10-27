---
layout: page
title:  "파이썬 표준 라이브러리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-01-04-01_01-PythonLibrary
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

<div class="colab-link">
    <a href="https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/python/notebooks/S01-01-04-01_01-PythonLibrary.ipynb" target="_blank">Colab에서 실습파일 열기 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

> - 파이썬에는 수많은 표준 라이브러리가 기본적으로 내장되어 있습니다.
> - 그 중에서 자주 사용할만한 몇 개의 라이브러리의 예시를 살펴보겠습니다.
> - 그 외의 필요한 라이브러리들은 파이썬 공식 사이트의 레퍼런스를 참고하시기 바랍니다.<br><br>
>    - [파이썬 언어 레퍼런스](https://docs.python.org/3/reference/index.html){: target="_blank"}
>    - [파이썬 표준 라이브러리 레퍼런스](https://docs.python.org/3/library/index.html){: target="_blank"}
{: .common-quote}

## 1. math 모듈

- 수학 관련 함수들을 제공하는 모듈
- 특히 정밀한 계산을 위한 복잡한 수학 연산 라이브러리 제공
- 제곱근, 삼각함수, 로그함수, 팩토리얼 등의 수학 연산 및 상수 값(pi, e)과 반올림, 내림, 올림 함수 등의 간단한 함수 포함

  ```python
  import math

  print(math.sin(math.radians(45)))
  print(math.sqrt(2))
  print(math.factorial(5))
  ```

## 2. random 모듈

- 난수 생성을 위한 다양한 함수 제공
- 리스트에서 무작위 선택(choice), 범위 내 난수 생성(randint), 리스트 섞기(shuffle) 등 지원
- 통계적 분포(가우시안, 이항 등)에 따른 난수 생성 가능
- 특히 AI 모델의 학습 등을 위한 모델 초기화 등에서 항상 사용됨

  ```python
  import random

  for i in range(5):
    print(random.random())
  ```

  ```python
  for i in range(5):
    print(random.randint(1, 10))
  ```

  ```python
  import random

  # randrange()를 사용해서 0~100 사이의 숫자 생성하기
  print ("Random number from 0~100 is : ",end="")
  print (random.randrange(100))
  ```

  ```python
  # randrange()를 사용해서 50~100 사이의 숫자 생성하기
  print ("Random number from 50~100 is : ",end="")
  print (random.randrange(50,100))
  ```

  ```python
  # randrange()를 사용해서 50~100 사이의 숫자를 5 간격을 기준으로 생성하기
  print ("Random number from 50~100 skip 5 is : ",end="")
  print (random.randrange(50,100,5))
  ```

  ```python
  import random

  # 리스트에서 랜덤으로 값 가져오기
  list1 = [1, 2, 3, 4, 5, 6]
  print(random.choice(list1))
  ```

  ```python
  # 문자열에서 랜덤으로 값 가져오기
  string = "striver"
  print(random.choice(string))
  ```

  ```python
  import random

  # 리스트 정의
  sample_list = ['A', 'B', 'C', 'D', 'E']

  print("Original list : ")
  print(sample_list)
  ```

  ```python
  # 리스트 섞기
  random.shuffle(sample_list)
  print("\nAfter the first shuffle : ")
  print(sample_list)
  ```

  ```python
  # 두 번째 셔플(섞기)
  random.shuffle(sample_list)
  print("\nAfter the second shuffle : ")
  print(sample_list)
  ```

  ```python
  import random

  # 리스트로부터 길이 3의 랜덤 아이템으로 구성된 리스트를 샘플링하여 출력
  list1 = [1, 2, 3, 4, 5, 6]
  print("With list:", random.sample(list1, 3))
  ```

  ```python
  # 문자열로부터 길이 4의 랜덤 아이템으로 구성된 리스트를 샘플링하여 출력
  string = "GeeksforGeeks"
  print("With string:", random.sample(string, 4))
  ```

  ```python
  # 튜플로부터 길이 4의 랜덤 아이템으로 구성된 리스트를 샘플링하여 출력
  tuple1 = ("ankit", "geeks", "computer", "science", "portal", "scientist", "btech")
  print("With tuple:", random.sample(tuple1, 4))
  ```

  ```python
  # set(집합)으로부터 길이 3의 랜덤 아이템으로 구성된 리스트를 샘플링하여 출력
  set1 = {"a", "b", "c", "d", "e"}
  print("With set:", random.sample(set1, 3))

  # '오류 설명' 버튼을 이용해보자.
  ```

  ```python
  import random

  # set(집합)으로부터 길이 3의 랜덤 아이템으로 구성된 리스트를 샘플링하여 출력
  set1 = {"a", "b", "c", "d", "e"}
  # set을 list로 변환하여 sample 함수에 전달합니다.
  print("With set:", random.sample(list(set1), 3))
  ```

  ```python
  import random

  for i in range(5):
    random.seed(1234)
    print(random.randint(1, 1000))
  ```

## 3. sys 모듈

- 파이썬 인터프리터와 상호작용하는 기능 제공
- 명령줄 인자(argv), 표준 입출력 스트림, 종료(exit) 등 제어 가능
- 모듈 경로(path)와 버전 정보 확인 가능

  ```python
  import sys

  print("버전:", sys.version)
  print("플랫폼:", sys.platform)
  ```

  ```python
  if sys.platform == "win32":
    print(sys.getwindowsversion())
  else:
    print("윈도우가 아닙니다.")
  ```

  ```python
  print("바이트 순서:", sys.byteorder)
  print("모듈 경로:", sys.path)
  ```

## 4. os 모듈

- 운영체제와 상호작용하는 기능 제공
- 파일/디렉토리 생성, 삭제, 이름 변경 등의 파일 시스템 작업 지원
- 환경 변수 접근, 프로세스 관리, 경로 조작 기능 포함

  ```python
  import os

  # 기본 정보 출력
  print("현재 운영체제:", os.name)
  print("현재 폴더:", os.getcwd())
  print("현재 폴더의 내부 요소:", os.listdir())
  ```

  ```python
  # 폴더 생성
  os.mkdir("hello")
  ```

  ```python
  # 폴더 삭제
  os.rmdir("hello")
  ```

  ```python
  # 파일 생성
  with open("original.txt", "w") as file:
    file.write("hello")
  ```

  ```python
  # 파일명 변경
  os.rename("original.txt", "new.txt")
  ```

  ```python
  # 파일 삭제
  os.remove("new.txt")
  ```

- 시스템 명령어 실행
  - Unix/Linux 계열은 Windows 계열과 명령어에 대한 결과가 다름
  - Unix에서 반환 값은 두 개의 서로 다른 정보를 포함하는 16비트 숫자이며, 여기에서의 결과 0은 오류가 없음을 의미함
    - 여기에서는 16비트 숫자로, 로우 바이트가 프로세스를 종료한 신호 번호이고 하이 바이트가 종료 상태(신호 번호가 0인 경우)를 의미함

  ```python
  # 시스템 명령어 실행
  os.system("dir")
  ```

## 5. time 모듈

- 시간 관련 함수 제공
- 현재 시간 얻기, 시간 지연(sleep), 시간 측정 등
- 시간 형식 변환과 타임스탬프 조작에 유용함
- 기본적으로 유닉스 시간을 기준으로 함

  ```python
  import time
  print(time.time())
  ```

  ```python
  # 일상 시간 문자열로 변환 가능
  t = time.time()
  print(time.ctime(t))
  ```

  ```python
  # 실행 시간 측정
  start = time.time()

  for a in range(100):
  print(a)

  end = time.time()
  print(end - start)
  ```

## 6. datetime 모듈

- 날짜와 시간을 처리하기 위한 클래스 제공
- 날짜/시간 연산, 형식 변환, 타임존 처리 가능
- time 모듈보다 더 직관적이고 고수준의 인터페이스 제공

  ```python
  import datetime

  # 현재 날짜와 시간 출력
  now = datetime.datetime.now()
  print("현재 날짜와 시간:", now)

  # 날짜 연산하기
  future_date = now + datetime.timedelta(days=30)
  print("30일 후:", future_date)

  # 특정 형식으로 날짜 출력
  formatted_date = now.strftime("%Y년 %m월 %d일 %H시 %M분")
  print("포맷팅된 날짜:", formatted_date)
  ```

## 7. urllib 모듈

- URL 작업을 위한 여러 모듈을 포함하는 패키지
- 웹 리소스 열기, 데이터 검색, HTTP 요청 처리 등 가능
- URL 파싱, 인코딩/디코딩, 요청 헤더 설정 기능 제공

  ```python
  from urllib import request

  target = request.urlopen("https://google.com")
  output = target.read()

  print(output)
  ```

## 8. calendar 모듈

- 달력 관련 함수 제공
- 년/월 달력 출력, 요일 확인, 윤년 판단 등 가능
- 날짜 계산과 달력 형식 지정 기능 포함

  ```python
  import calendar

  print(calendar.calendar(2022))
  print(calendar.month(2022, 9))
  ```

## 9. json 모듈

- JSON 데이터의 인코딩/디코딩 기능 제공
- 파이썬 객체를 JSON 문자열로 변환, 그 반대도 가능
- 웹 API와의 데이터 교환에 매우 유용

  ```python
  import json

  # 파이썬 딕셔너리 생성
  data = {
      "이름": "홍길동",
      "나이": 25,
      "관심분야": ["모빌리티 데이터", "머신러닝", "레트로 게임"],
      "개발자": True
  }

  # 딕셔너리를 JSON 문자열로 변환
  json_str = json.dumps(data, ensure_ascii=False, indent=4)
  print(json_str)

  # JSON 문자열을 파이썬 객체로 변환
  parsed_data = json.loads(json_str)
  print(parsed_data["관심분야"][0])  # 모빌리티 데이터 출력
  ```

## 10. re 모듈

- 정규 표현식을 사용한 문자열 검색과 조작 기능 제공
- 패턴 매칭, 문자열 추출, 치환 등의 고급 텍스트 처리 가능
- 복잡한 문자열 검증 및 파싱 작업에 필수적

  ```python
  import re

  # 이메일 주소 패턴 매칭
  text = "연락처: user@example.com, admin@test.co.kr"
  emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
  print("찾은 이메일:", emails)

  # 문자열 치환하기
  phone = "전화번호: 010-1234-5678"
  masked_phone = re.sub(r'(\d{3})-(\d{4})-(\d{4})', r'\1-\2-****', phone)
  print(masked_phone)  # 전화번호: 010-1234-****
  ```

## 11. collections 모듈

- 기본 자료구조를 확장한 특수 컨테이너 타입 제공
- Counter, defaultdict, OrderedDict, deque 등 효율적인 데이터 구조 포함
- 데이터 처리와 알고리즘 구현에 매우 유용

  ```python
  from collections import Counter, defaultdict, deque

  # Counter 사용 예제
  text = "머신러닝과 딥러닝을 활용한 모빌리티 데이터 분석"
  word_count = Counter(text.split())
  print("단어 빈도수:", word_count)

  # defaultdict 사용 예제
  interests = [('AI', '홍길동'), ('데이터분석', '홍길동'), ('AI', '전우치'), ('레트로게임', '홍길동')]
  interest_dict = defaultdict(list)
  for category, person in interests:
      interest_dict[category].append(person)
  print("관심사별 사람들:", dict(interest_dict))

  # deque 사용 예제 (양방향 큐)
  tasks = deque(["태스크1", "태스크2", "태스크3"])
  tasks.append("태스크4")  # 오른쪽에 추가
  tasks.appendleft("태스크0")  # 왼쪽에 추가
  print("작업 목록:", list(tasks))
  ```