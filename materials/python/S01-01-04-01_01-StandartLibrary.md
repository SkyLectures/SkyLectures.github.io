---
layout: page
title:  "파이썬 기초: 라이브러리-파이썬 표준 라이브러리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-01-04-01_01-StandardLibrary
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

- 파이썬에는 내장되어 있는 수많은 표준 라이브러리가 있습니다. 그 중에서 자주 사용할만한 몇 개의 라이브러리의 예시를 살펴보겠습니다. 그 외의 필요한 라이브러리들은 파이썬 공식 사이트의 레퍼런스를 참고하시기 바랍니다.<br><br>
    - [파이썬 언어 레퍼런스](https://docs.python.org/3/reference/index.html)
    - [파이썬 표준 라이브러리 레퍼런스](https://docs.python.org/3/library/index.html)

## 1. math 모듈

- 정밀한 계산을 위한 복잡한 수학 연산 라이브러리
- 삼각함수, 제곱근 등 연산 예시

    ```python
    import math

    print(math.sin(math.radians(45)))
    print(math.sqrt(2))
    print(math.factorial(5))
    ```

## 2. random 모듈

- 난수 생성 기능. 어떤 수가 나올지 예측할 수 없는 무작위 동작 구현

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
    tuple1 = ("ankit", "geeks", "computer", "science",
                    "portal", "scientist", "btech")
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

- 날짜와 시간 관련 기능 제공
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

## 6. urllib 모듈

- URL을 다루는 라이브러리
    - urlopen() 함수 : URL 주소의 페이지 열기

    ```python
    from urllib import request

    target = request.urlopen("https://google.com")
    output = target.read()

    print(output)
    ```

## 7. calendar 모듈

- 달력 기능
- 인수로 받은 연도의 달력 객체 반환
- month 함수
  - 연도와 달을 인수로 받아 해당 월 달력 객체 반환
- weekday 함수
  - 특정 날짜가 어떤 요일인지 조사

    ```python
    import calendar

    print(calendar.calendar(2022))
    print(calendar.month(2022, 9))
    ```
