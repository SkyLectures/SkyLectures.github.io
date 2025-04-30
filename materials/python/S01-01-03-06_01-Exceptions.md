---
layout: page
title:  "파이썬 기본 문법: 6. 예외 처리"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-01-03-06_01-Exceptions
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 예외 처리(Exception Handling)의 개요

### 1.1 예외 처리(Exception Handling)의 개념
- 프로그램 실행 중에 발생할 수 있는 <span style="color: #00C">예상치 못한 오류(예외, Exception)</span>에 대비하고
- 이러한 오류가 발생했을 때 프로그램이 <span style="color: #00C">비정상적으로 종료되는 것을 방지</span>하며 
- <span style="color: #00C">우아하게(gracefully) 처리</span>하는 메커니즘입니다.<br><br>
- 예외 처리는 <span style="color: #C00">프로그래밍에 있어서 필수적인 부분이며, 견고하고 안정적인 프로그램을 만들기 위해 반드시 고려해야 하는 중요한 개념</span>입니다.

### 1.2 핵심 아이디어

- 예상 가능한 문제 상황 대비 
    - 개발자는 코드를 작성하면서 발생할 가능성이 있는 오류 상황 (예: 파일 없음, 0으로 나누기, 잘못된 입력 등)을 미리 예측하고 대비
- 오류 발생 시 대처 
    - 프로그램 실행 중 예상된 오류가 실제로 발생하면, 
    - 미리 정의해둔 예외 처리 코드가 실행되어 오류를 적절히 처리
- 프로그램의 안정성 확보 
    - 예외 처리를 통해 오류 발생에도 프로그램이 멈추지 않고 계속 실행되거나, 
    - 오류 메시지를 출력하고 안전하게 종료할 수 있도록 함

### 1.3 예외 처리의 중요성

- 프로그램의 안정성 향상
    - 예외 처리 없이 오류가 발생하면 프로그램이 갑자기 종료되어 사용자 경험을 저하시키고 데이터 손실을 초래할 수 있음
- 유연한 오류 대응
    - 각기 다른 유형의 오류에 대해 맞춤형으로 대처할 수 있도록 함
- 디버깅 용이성
    - 예외 발생 시 오류 정보 (오류 유형, 발생 위치 등)를 제공하여 디버깅을 도와줌

### 1.4 일반적인 예외 처리 구조 (Python 기준)

- 아래 예시에서 `try` 블록 안의 코드는 실행되다가 `ZeroDivisionError`라는 예외가 발생하면 실행이 중단되고, `except` 블록의 코드가 실행됩니다. `finally` 블록은 예외 발생 여부와 관계없이 항상 실행됩니다.

    ```python
    try:
        # 오류가 발생할 가능성이 있는 코드
        result = 10 / 0
        print(result)
    except ZeroDivisionError as e:
        # ZeroDivisionError 발생 시 실행될 코드
        print("0으로 나눌 수 없습니다!")
        print("오류 정보:", e)
    finally:
        # 예외 발생 여부와 상관없이 항상 실행될 코드 (선택 사항)
        print("예외 처리 완료")
    ```


## 2. 예외 처리 실습 예제

```python
print('안녕하세요.')
print(param)
```

```python
try:
    print('안녕하세요.')
    print(param)

except:
    print('예외가 발생했습니다.')
```

```python
del param2
```

```python
# param2 = '반갑습니다.'

try:
    print('안녕하세요.')
    print(param2)

except:
    print('예외가 발생했습니다.')

else:
    print('예외가 발생하지 않았습니다.')

finally:
    print('무조건 실행하는 코드')
```

```python
try:
    print('안녕하세요.')
    print(param3)

except:
    print('예외가 발생했습니다.')

finally:
    print('무조건 실행하는 코드')
```

```python
try:
    print('안녕하세요.')
    print(param3)

except Exception as e:
    print(e)
    print('예외가 발생했습니다.')
```

- 아래의 예제는 키보드 인터럽트 예외를 발생시키기 위하여 무한루프를 돌게 됩니다.<br>
    - Colab을 사용 중이시라면 적당한 시점에 셀의 좌측상단에 있는 (▶) 또는 (◼) 버튼을 눌러서 중지시켜주세요.
    - 로컬 시스템의 자체 가상환경을 사용 중이시라면 Ctrl-C를 눌러서 중지시켜주세요.

```python
import time

count = 1
try:
    while True:
        print(count)
        count += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print('사용자에 의해 프로그램이 중단되었습니다.')
```