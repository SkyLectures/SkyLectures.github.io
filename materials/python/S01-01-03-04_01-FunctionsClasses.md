---
layout: page
title:  "파이썬 기본 문법: 4. 함수, 클래스"
date:   2025-03-01 10:00:00 +0900
permalink: /material/S01-01-03-04_01-FunctionsClasses
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. 함수(Function)

### 1.1 함수의 개요

#### 1.1.1 함수의 개념
- 함수(Function)는 특정 작업을 수행하는 코드 블록을 말합니다. 
- 프로그래밍에서의 함수는 마치 수학의 함수처럼, 입력을 받아서 정해진 연산을 수행한 후 결과를 반환하는 역할을 합니다.
- 함수는 프로그래밍에서 코드의 효율적인 관리, 재사용, 구조화 및 유지보수를 위한 핵심적인 도구입니다.

#### 1.1.2 핵심 아이디어

- 코드의 묶음: 여러 줄의 코드를 하나의 의미 있는 단위로 묶어 놓은 것
- 재사용성: 한번 정의된 함수는 필요할 때마다 여러 번 호출하여 동일한 코드를 반복해서 작성할 필요가 없음
- 모듈화 및 가독성 향상: 복잡한 프로그램을 여러 개의 작은 함수로 나누어 작성하면 코드의 구조가 명확해지고 이해하기 쉬워짐
- 유지보수 용이성: 특정 기능에 대한 코드가 함수 안에 격리되어 있어, 해당 기능을 수정해야 할 때 함수 내부만 변경하면 됨

#### 1.1.3 함수의 주요 구성 요소

- 정의(Definition)
    - 함수를 만들고 그 기능을 정의하는 부분
    - 함수 이름, 입력(매개변수), 수행할 코드 블록, 출력(반환 값) 등을 포함
- 호출(Call)
    - 정의된 함수를 실제로 실행하는 행위
    - 함수 이름과 필요한 입력 값(인수)을 전달하여 함수를 실행
- 매개변수(Parameter)
    - 함수를 정의할 때 입력으로 받을 값을 나타내는 변수
- 인수(Argument)
    - 함수를 호출할 때 실제로 전달하는 값
- 반환 값(Return Value)
    - 함수가 작업을 수행한 후 호출한 곳으로 돌려주는 결과 값
    - 반환 값이 없을 수도 있음

#### 1.1.4 비유

- 함수: 요리 레시피
- 매개변수: 레시피에 필요한 재료
- 인수: 실제로 사용하는 재료의 양
- 반환 값: 완성된 요리

#### 1.1.5 함수의 예시 (Python 기준)

```python
def add(a, b):  # 함수 정의: 이름은 'add', 매개변수는 a, b
  result = a + b
  return result  # 결과 값 반환

sum_result = add(5, 3)  # 함수 호출: 인수 5와 3을 전달
print(sum_result)  # 출력: 8
```

## 1.2. 함수 실습 예제

#### 1.2.1 함수의 선언 방법

```python
def 함수이름(인자1, 인자2, ...):
    코드들
    return 결과값
```

#### 1.2.2 함수 선언과 사용

```python
# 함수 선언
def add_number(n1, n2):
    result = n1 + n2
    return result

def add_text(t1, t2):
    print(t1 + t2)

# 함수 사용
answer = add_number(5, 10)
print(answer)

text1 = '안녕하세요~'
text2 = '만나서 반갑습니다.'
add_text(text1, text2)
```

#### 1.2.3 함수 호출 시 인자의 전달 순서 및 전달 인자 지정방법

```python
add_text(text2, text1)
add_text(t2=text2, t1=text1)
```

#### 1.2.4 함수 선언 시 인자의 기본값 설정방법

- 함수 선언 시 인자의 기본값 설정할 때, 기본값이 설정된 인자는 뒤쪽에 위치해야 합니다.

    ```python
    def add_number(n1=100, n2=200):
        result = n1 + n2
        return result

    result = add_number(30)
    print(result)
    ```

    ```python
    # 잘못된 기본값 설정의 예
    def add_number(n1=100, n2):
        result = n1 + n2
        return result
    ```

    ```python
    # 올바른 기본값 설정의 예
    def add_number(n1, n2=100):
        result = n1 + n2
        return result

    result = add_number(30)
    print(result)
    ```

#### 1.2.5 함수의 결과값의 다중 반환

```python
def reverse(x, y, z):
    return z, y, x

a, b, c = reverse(10, 20, 30)
print(a, b, c)
```

```python
print(result[0])
print(result[1])
print(result[2])
```

```python
r1, r2, r3 = reverse(1, 2, 3)
print(r1)
print(r2)
print(r3)
```

## 2. 클래스(Class)

### 2.1 클래스의 개요

#### 2.1.1 클래스의 개념

- 클래스(Class)는 객체(Object)를 만들기 위한 설계도 또는 틀입니다. 객체를 선언하기 위한 자료구조로서의 틀이라고 생각할 수 있습니다.
- 클래스는 객체를 선언하기 위한 틀이므로 그 자체로는 사용할 수 없으며, 인스턴스 객체를 생성하여 사용합니다.
- 현실 세계의 사물이나 개념을 프로그램 내에서 표현하기 위해 사용되며, 데이터(속성)와 기능(메서드)을 하나의 단위로 묶습니다.<br><br>
- 클래스는 객체 지향 프로그래밍의 핵심 개념으로, 데이터와 기능을 묶어 현실 세계를 프로그램 내에 효과적으로 모델링하고, 코드의 재사용성과 유지보수성을 높이는 데 중요한 역할을 합니다.

#### 2.1.2 핵심 아이디어

- 객체 지향 프로그래밍(OOP)의 핵심: 클래스는 객체 지향 프로그래밍의 중요한 구성 요소입니다.
- 데이터와 기능의 캡슐화: 관련된 데이터와 해당 데이터를 조작하는 기능을 하나의 클래스 안에 묶어 관리합니다.
- 코드의 재사용성 및 확장성: 한번 정의된 클래스를 기반으로 여러 개의 객체를 생성할 수 있으며, 상속 등의 기능을 통해 코드를 확장하고 재사용하기 용이합니다.
- 추상화: 복잡한 시스템을 단순화하여 모델링하고 이해하기 쉽게 만들어줍니다.

#### 2.1.3 클래스의 주요 구성 요소

- 속성(Attribute)
    - 클래스가 가지는 데이터 또는 상태를 나타내는 변수
    - 객체의 특징이나 정보를 저장(예: 자동차 클래스의 속성 - 색상, 모델, 속도)
    - 프로그래밍 언어에 따라 멤버(멤버변수, 클래스멤버)라고 표현하기도 함

- 메서드(Method)
    - 클래스 내부에 정의된 함수
    - 객체의 행위 또는 기능을 나타냄
    - 객체의 속성을 조작하거나 특정 작업을 수행함
        - 예: 자동차 클래스의 메서드 → 가속하다, 멈추다, 경적을 울리다

- 생성자(Constructor)
    - 클래스의 객체가 생성될 때 자동으로 호출되어 객체의 초기 상태를 설정하는 특별한 메서드
    - 일반적으로 '__ init__'이라는 이름을 사용함

- 인스턴스(Instance)
    - 클래스를 기반으로 실제로 생성된 객체
    - 클래스는 설계도이고, 인스턴스는 그 설계도에 따라 만들어진 실체라고 생각할 수 있음

#### 2.1.4 비유

- 클래스: 자동차 설계도
- 속성: 설계도에 명시된 차량의 색상, 모델, 엔진 종류 등
- 메서드: 설계도에 명시된 가속 기능, 제동 기능 등
- 객체(인스턴스): 설계도에 따라 실제로 만들어진 특정 자동차 (예: 빨간색 소나타)

#### 2.1.5 클래스의 예시 (Python 기준)

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name  # 속성
        self.breed = breed  # 속성

    def bark(self):  # 메서드
        print("멍멍!")

my_dog = Dog("해피", "푸들")  # Dog 클래스의 인스턴스 생성
print(my_dog.name)   # 객체의 속성에 접근
my_dog.bark()      # 객체의 메서드 호출
```

### 2.2 클래스 실습 예제

#### 2.2.1 클래스의 선언과 사용

```python
class MyClass:
    var = '안녕하세요'

    def sayHello(self):
        print(self.var)
```

```python
obj = MyClass()
print(obj.var)
obj.sayHello()
```

#### 2.2.2 생성자와 소멸자

- 생성자

    ```python
    class MyClass2:
        def __init__(self):
            self.var = "안녕하세요"
            print("MyClass2 인스턴스 객체가 생성되었습니다.")

    obj = MyClass2()
    print(obj.var)
    ```

- 소멸자

    ```python
    class MyClass3:
        def __init__(self):
            self.var = "안녕하세요"
            print("MyClass3 인스턴스 객체가 생성되었습니다.")

        def __del__(self):
            print("MyClass3 인스턴스 객체가 메모리에서 제거됩니다.")

    obj = MyClass3()
    del obj
    ```

#### 2.2.3 클래스의 상속

```python
class Add:
    def add(self, n1, n2):
        return n1 + n2

class Calculator(Add):
    def sub(self, n1, n2):
        return n1 - n2

calc = Calculator()
print(calc.add(1, 2))
print(calc.sub(1, 2))
```

```python
class Add:
    def add(self, n1, n2):
        return n1 + n2

class Multiply:
    def multiply(self, n1, n2):
        return n1 * n2

class Calculator(Add, Multiply):
    def sub(self, n1, n2):
        return n1 - n2

calc = Calculator()
print(calc.add(1, 2))
print(calc.multiply(3, 2))
print(calc.sub(1, 2))
```