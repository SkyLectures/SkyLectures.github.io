---
layout: page
title:  "파이썬 중급: Sequence 자료형"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-02-04-01_01-SequenceDataTypes
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


# **Sequence Data Type**

- 여러 원자로 구성된 자료형
- Sequence 자료형의 특징
    - 다양한 클래스(문자열, List, Tuple, bytes, bytearray 등)가 존재함
    - 순서에 따라 데이터가 저장되며, 이 연속적인 순서에 따라 검색이 가능함
    - 데이터에 순서가 부여되어 있으므로 Index를 이용해서 검색이 가능하며, 슬라이스로 부분 검색도 가능함
    - 동일한 타입의 원소를 가질 수도 있고 List처럼 객체를 원소로 가질 수도 있음
    - 특정 자료형은 한 번 생성되면 원소를 변경할 수 없고, 다른 경우에는 원소들을 변경할 수 있음


- Sequence 자료형의 종류별 특징
    - 문자열(str): UniCode 문자의 순서대로 처리됨
    - bytes, bytearray: ASCII 문자의 순서대로 처리됨
    - List, Tuple: 다양한 객체를 넣을 수 있으므로 다양한 원소를 가진 Collection을 구성함
    


- 데이터 불변성
    - 문자열(str), Tuple, bytes: 한 번 생성되면 원소를 변경할 수 없음
    - List, bytearray: 생성된 후, 원소 추가 및 생성, 변경 가능

- 내부 메서드의 처리
    - 변경 가능한 자료형(List 등)
        - 내부 메서드가 실행되면 내부 원소를 변경함
    - 변경 불가능한 자료형(문자열 등)
        - 내부 메서드가 실행되면 새로운 인스턴스를 생성함
        - 생성자로 생성할 때 기존에 만들어진 인스턴스에 대해서는 새로 만든 것이 아니라 만들어져 있는 것을 반환하는 interning 처리를 수행함

## **1. Sequence 자료형의 특징**

### 1.1 Runtime에 속성 추가 여부

- 파이썬 내장 자료형은 런타임에 속성의 추가가 불가능함
    - Cython엔진이 실행되는 Python의 경우
        - C 언어로 제공되는 공통기능에 대한 일관성을 유지하기 위하여 내장 자료형은 변경할 수 없도록 차단함
    - 클래스나 인스턴스에서 속성이나 메서드를 관리하는 네임스페이스(__ dict __) 속성이 없는 경우
        - 접근해서 갱신, 삭제를 하는 것이 불가능함
        - 내부에 만들어진 속성이나 메서드만 사용 가능
        - 다른 기능과 속성을 추가하고 싶을 때는 클래스를 상속받고 사용자 클래스를 만들어서 추가 속성과 기능의 확장을 이용함

- Sequence 내장 타입의 인스턴스 Namespce 미존재
    - List를 리터럴로 생성하고 이 List 인스턴스에 네임스페이스를 조회하면 예외가 발생함

# 내부 속성에 네임스페이스(__dict__)가 존재하지 않음
l = [1,2,3]
l.__dict__

# 리스트를 상속한 List 클래스를 정의하고 추가적인 속성으로 name을 만든 후
# List 클래스의 인스턴스를 생성하고 네임스페이스를 확인하면 name 속성이 있는 것을 알 수 있음
class List(list):
    def __init__(self, name, value):
        super().__init__(value)
        self.name = name

l = List("리스트", [1, 2, 3])
print(l.__dict__)

# 리스트에 대한 원소들은 부모 클래스 리스트 내부에 생성했으므로 부모 리스트 클래스의 메서드를 그대로 사용해서 처리가 가능함
print(l+l)
print(l.__dict__)

### 1.2 변경 가능 여부: Mutable & Immutable

- 변경 가능(Mutable)과 변경 불가능(Immutable)에 대한 기본 개념 중 변경 가능하다는 말은 객체 내부의 원소들을 추가, 삭제, 변경할 수 있다는 의미이지만 객체 자신을 변경할 수 있다는 말은 아님

#### 1.2.1 문자열은 변경 불가

- 문자열을 리터럴로 정의하고 첫 번째 원소의 값을 변경할 경우, 변경이 불가능하다는 예외가 발생함

s = "창덕"

print(s[0])
s[0] = "성"

- 변경할 수 없다는 뜻은 변경하기 위한 __ setitem __ 메서드가 만들어져 있지 않다는 의미임

str.__setitem__

#### 1.2.2 Tuple은 변경 불가

- Tuple을 리터럴로 정의하고 첫 번째 원소의 값을 변경할 경우, 문자열과 동일하게 원소 변경이 불가능하다는 예외가 발생함

t = ("고","요","한")

print(t[0])
t[0] = "김"

- 변경할 수 없다는 뜻은 Tuple 역시 변경하기 위한 __ setitem __ 메서드가 만들어져 있지 않다는 의미임

tuple.__setitem__

#### 1.2.3 변경 가능한(Mutable) 자료형

l = ["전","민","수"]

print(type(l))
l[0] = "김"

print(l)

list.__setitem__

list.__delitem__

### 1.3 Collection 여부

- Collection이란 다양한 원소를 가진 데이터 구조를 말함(Sequence 자료형은 기본적으로 Collection)
- Collection 여부는 원소 개수, 포함 관계, 반복 가능 여부를 확인할 수 있으면 됨

#### 1.3.1 자료형 내의 원소의 개수 확인: len()

- 파이썬 3 버전에서는 문자열이 UniCode로 되어 있으므로 문자코드 단위, 즉 문자 단위로 길이를 표시함

s = "강대명"
l = ["고","요","한"]

print(len(s))
print(len(l))

#### 1.3.2 반복형을 반복자로 변환: iter()

- Sequence 자료형들은 내부에 원소들이 없거나 연속적으로 들어 있어 반복해서 원소를 읽을 수 있으므로 반복형(Iterable)으로 처리가 가능함
- 반복자를 생성하는 iter() 함수로 호출하려면 내부에 반드시 __ iter __ 스페셜 메서드가 존재해야 함
    - 그러나 기존 버전과의 호환성 유지를 위해 __ getitem __이 구현되어 잇으면 이를 호출해서 반복자로 처리하도록 만들어 줌

s = "강대명"
si = iter(s)
print(si)

for i in si :
    print(i)

l = ["고","요","한"]
li = iter(l)
print(li)

for i in li :
    print(i)

#### 1.3.3 Sequence 자료형 내에 원소 포함여부 확인: in 연산자

s = "강대명"
print("대" in s)

l = ["고","요","한"]
print("한" in l)

### 1.4 Sequence 자료형 내의 메서드 처리 기준

#### 1.4.1 내장 메서드 처리 기준

- 변경 가능한 자료형의 메소드는 내부를 갱신

l = ["고","가","한"]
c = l.sort()

print(c)
print(l)

- 변경 불가능한 경우는 별도의 객체를 만들어서 반환 처리

s = "강대명"
sr = s.replace("명","한")

print(id(s), s)
print(id(sr),sr)

### 1.5 interning 처리

- interning 이란?
    - 기존에 만들어진 변경이 불가능한 Sequence 자료형이 있을 경우
    - 생성자를 통해 인스턴스를 다시 생성하면
    - 새로운 인스턴스를 만드는 것이 아니라 기존에 있는 것을 불러다 반환하는 처리 방식

- 변경불가능한 자료형의  interning 처리

t = ("고","가","한")
ti = tuple(t)
print(t is ti)

s = "달문"
si = str(s)
print(s is si)

l = ["고","요","한"]
li = list(l)
print(l is li)

## **2. 문자열 자료형**

- 파이썬 버전 3부터 문자의 기본 코드가 UniCode로 변경되면서 다양한 언어로 처리가 가능해 짐
- 파이썬 버전 2의 Unicode 클래스가 사라지고 대신 bytes 클래스가 사용되면서 기본적인 byte 처리는 bytes 클래스로 수행하게 됨

### 2.1 문자열 생성

- 리터럴과 생성자로 문자열 생성

sl =

s = str(123)
ss = str("성대현")
sf = str(123.00)
sl = "파이썬"

print(s)
print(ss)
print(sf)
print(sl)

print(type(sl))

### 2.2 문자열 주요 메서드

- 대소문자 처리

sl = "spiderman"

sh = sl.upper()
print(sh)

su = "WONDER WOMEN"
sh = su.lower()
print(sh)

su = "WONDER WOMEN"
st = su.title()
print(st)

sl = "spiderman"
sc = sl.capitalize()
print(sc)

su = "WONDER WOMEN"
scc = su.casefold()
print(scc)

- 문자열의 위치를 조정해서 꾸미기

s = "빅데이터와 인공지능"

sc = s.center(30,"%")
print(sc)
sb = s.center(30)
print(sb)

s = "빅데이터와 인공지능"

sc = s.ljust(30,"%")
print(sc)
sb = s.ljust(30)
print(sb)

s = "빅데이터와 인공지능"

sc = s.rjust(30,"%")
print(sc)
sb = s.rjust(30)
print(sb)

s = "빅데이터와 인공지능"

sc = s.center(30,"파")
print(sc)
sb = s.center(30,"bb")
print(sb)

- 특정 문자 찾기

s = "특정 문자 찾기를 한다. 찾은 문자는"

print(s.find("찾"))
print(s[s.find("찾")])

print(s.rfind("찾"))
print(s[s.rfind("찾")])

s = "특정 문자 찾기를 한다. 찾은 문자는"

print(s.index("찾"))
print(s[s.index("찾")])

print(s.rindex("찾"))
print(s[s.rindex("찾")])

s = "특정 문자 찾기를 한다. 찾은 문자는"

print(s.count("찾"))

- 문자열 패턴 매칭하기

s = dir(str)

print(type(s))

count = 1
for i in s :
    if i.startswith("__") :
        continue
    else :
        print(i, end=" ")

    if count % 6 == 0 :
        print()
    count += 1

s = dir(str)

print(type(s))

count = 1
for i in s :
    if i.endswith("__") :
        print(i, end=" ")
    else :
        continue

    if count % 6 == 0 :
        print()
    count += 1

- 빈 문자열로 분리하고 결합하기

s = "빈 문자열로 분리하고 결합하기"

ss = s.split(" ")
print(ss)

sl = " ".join(ss)
print(sl)

- 개행문자가 있을 경우 문자열 분리하기

import pprint

s ="""A simple object subclass
that provides attribute access
to its namespace,
as well as a meaningful repr."""

ss = s.split("\n")
pprint.pprint(ss)

sl1 = " ".join(ss[:2])
print(sl1)
sl2 = " ".join(ss[2:])
print(sl2)

- 문자열 길이 확인하기

s = "문자열 검색"

print(len(s))

sa = " string indexing"
print(len(sa))

s = "문자열 검색"

print(s[0])
print(s[1])

s = "문자열 검색"

print(s[-1])
print(s[-2])

- 암호화(encode)와 복호화(decode) 기준

s = "성균관대학교"

sb = s.encode("utf-8")
print(sb)
print(sb.decode("utf-8"))

s = "성균관대학교"
print(s[0].encode("utf-16"))
sb = s.encode("utf-16")
print(sb)
print(sb.decode("utf-16"))

s = "성균관대학교"
print(s[0].encode("utf-16le"))
sb = s.encode("utf-16le")
print(sb)
print(sb.decode("utf-16le"))

s = "성균관대학교"
print(s[0].encode("utf-16be"))
sb = s.encode("utf-16be")
print(sb)
print(sb.decode("utf-16be"))

## **3. 바이트 자료형(bytes data type)**

- 파이썬 3 버전에 새로 추가된 자료형으로 컴퓨터가 기본으로 처리하는 바이트 자료형
- 저장되는 형태가 16진수의 Hexa 값으로 관리되고 ASCII 코드인 경우에는 Hexa 값 대신 문자로 보여 줌
- 바이트 자료형도 파이썬의 기본 문자열 자료형인 Unicode와 동일한 메서드를 가지고 처리됨
- 문자열 자료형처럼 변경이 불가능한 구조를 따름

### 3.1 바이트 생성

- 바이트 자료형의 생성은 리터럴 형태로 b를 문자열 앞에 붙여서 표시함
- bytes 생성자를 기반으로 인스턴스도 만들 수 있음

b = b"hello"
print(type(b), b)

s = "성균관대학교"
bs = bytes(s.encode("utf-8"))
print(type(bs))
print(bs)

print(bytes("성균관대학교",encoding="utf-8"))

s = "성균관대학교"
bs = bytes(s.encode("utf-8"))
print(type(bs))
print(bs.decode("utf-8"))

### 3.2 바이트 자료형의 메서드 확인

b = set(dir(bytes))
s = set(dir(str))

bs = b - s
print(bs)

bb = b"Hello"
bh = bb.hex()

print(type(bh), bh)

bfh = bytes.fromhex('B901EF')
print(bfh)

### 3.3 encode/decode 메서드 처리

s = "하늘과 바람과 별과 시"
b = s.encode("utf-8")

print(type(b))
bs = b.decode("utf-8")
print(type(bs))
print(bs)

### 3.4 bytes/str 생성자에서 직접 encode, decode 하기

s = "휀휁휂휃휄"

b = s.encode("utf-8")

bb = bytes(s, "utf-8")

print(b == bb)

print(str(bb, "utf-8"))

## **4. 바이트 어레이 자료형(bytearray data type)**

- 바이트 자료형은 문자열처럼 변경이 불가능하지만 바이트 어레이 자료형은 리스트처럼 변경이 가능한 구조를 지원함
- 변경이 가능하므로 리스트처럼 원소를 변경, 추가, 삭제할 수 있는 메서드를 지원함
- 바이트 자료형처럼 바이트 기준으로 데이터를 관리함

### 4.1 bytes로 생성한 것을 bytearray로 변환

b = b"abcde"
ba = bytearray(b)

print(type(ba))
print(ba)

bs = bytearray("바이트어레이","utf-8")
print(bs)

### 4.2 버퍼 처리하기: bytearray

buffer = bytearray(20)

print(buffer)

b = b"abcde"
buffer[:len(b)] = b
print(buffer)

### 4.3 bytes와 bytearray 메서드의 차이점

- 바이트 어레이는 변경이 가능하므로 스페셜 메서드인 __ setitem __, __ delitem __을 제공함
- 내부 원소를 변경 및 삭제할 수 있는 append, extend, remove, pop 메서드 등이 있으며 역정렬을 위한 reverse 메서드도 제공됨

import pprint

bs = set(dir(bytes))
bb = set(dir(bytearray))

pprint.pprint(bb - bs)

buffer = bytearray(20)

print(buffer)

buffer.insert(0, 31)
print(buffer)
buffer.pop()
print(buffer)
buffer.append(31)
print(buffer)

buffer[:3] = b'abc'
print(buffer)

buffer[0] = b'a'

buffer.insert(0,b'a')

a = bytearray(b"abcd")

a.insert(0,b"a")

## **5. 튜플 자료형(tuple data type)**

- 튜플 자료형은 리스트 자료형과 동일하게 처리되지만 변경이 불가능함
- 따라서 리스트와의 차이점은 내부 원소를 변경하거나 갱신하기 위한 메서드들이 존재하지 않는 것임
- 튜플 자료형은 파이썬 내부적으로 변경이 되지 않으면서 관리되어야 할 기능에서 필요하며, 튜플을 이용해서 데이터를 전달하는 방식으로 많이 사용됨
<br><br>
- 튜플 자료형의 사용 사례
    - 함수의 매개변수에 가변적으로 들어오는 인자를 처리할 때
    - 특정 변수에 여러 데이터를 할당할 때
    - 함수의 반환값 여러 개를 하나로 묶어서 반환이 필요할 때 등

### 5.1 튜플 생성

- 튜플 리터럴 및 생성자로 생성

t = 1,2,3,4
print(t)

t1 = (1,2,3,4)
print(t1)

def func(x,y) :
    return x,y

t2 = func(10,10)
print(type(t2))
print(t2)

t = tuple([1,2,3,4])
print(t)
ts = tuple("광화문")
print(ts)

- 튜플 내의 원소가 하나일 경우 주의사항
    - 단일 원소도 쉼표로 분리할 것
        - 스칼라 값인지 튜플인지 구분에 용이함

o = (1)
t = (1,)

print(type(o), o)
print(type(t), t)

### 5.2 튜플의 메서드

- count 메서드

t = dir(tuple)

for i in t :
    if not i.startswith("__") :
        print(i)

- index 메서드

t = (1,2,3,2,3)

print(t.count(2))
print(t.index(2))

print(t.index(2,3))

### 5.3 튜플 원소로 Mutable 자료형 처리 방법

- 원소가 리스트일 경우

t = (1,2,[1,2])
print(id(t))
t[2][0] = 99
print(t)

print(id(t))

## **6. 리스트 자료형(list data type)**

- 파이썬에서 가장 자주 사용되는 자료형
- 변경가능한 Sequence 자료형이며 다양한 자료형을 내부의 원소로 수용할 수 있음

### 6.1 리스트 생성

l1 = list((1,2,3,))
print(l1)

l2 = []
print(l2)

l3 = [1,2,3]
print(l3)

### 6.2 리스트 복사 처리

- 파이썬에서 복사가 되는 기준
    - 리스트 내에 변경이 가능한 원소가 들어올 경우, 단순히 복사하면 내부까지 전부 복사가 되지 않아 동일한 인스턴스를 공유한 채로 처리될 수 있음
    - 이런 점을 방지하기 위해서는 복사할 때 원소도 전부 다른 인스턴스로 변환이 되어야 다른 로직에서 리스트를 갱신할 때 공유된 리스트로 처리되는 것을 막을 수 있음
- 2가지의 복사 기능
    - 얕은 복사(Shallow Copy): 새로운 객체를 만들지만 내부 원소는 기존 원소를 참조함
        - 리스트 자료형 내의 copy 메서드는 얕은 복사를 처리해서 새로운 리스트 인스턴스를 하나 만드는데 사용함
    - 깊은 복사(Deep Copy): 새로운 객체를 만들고 내부 원소들도 다른 원소로 만듦

- 리스트로 선언된 변수의 별칭 사용

l = [1,2,3,4]

alias = l

print(alias is l)
print(alias ==  l)

- 리스트로 copy 메서드 사용

l = [1,2,3,4]

lc = l.copy()

print(lc == l)
print(lc is l)

- 리스트 내의 원소로 리스트를 가질 경우 copy 메소드 사용

l = [1,2,3,4]

ll = [l,l]
print(ll)

lc = ll.copy()

print(lc)
lc[0][0] = 999

print(l)
print(lc)

- 깊은 복사(deepcopy)를 하는 이유
    - 리스트 내의 copy 메서드는 리스트 자체만을 복사해서 새로운 사본을 만들고 리스트 내의 원소에 대해서는 사본을 만들지 않음
    - 리스트를 복사할 때 원본 리스트와 전혀 다른 리스트를 만들기 위해서는 리스트 내의 리스트 원소들도 다 다른 리스트가 만들어져야 함
    - 이를 처리하기 위하여 copy라는 모듈을 별도로 제공하고, 그 내부의 deepcopy 함수를 사용해서 복사하여 원본과는 전혀 다른 리스트를 생성함

import copy

l = [1,2,3,4]

ll = [l,l]
print(ll)

lc = copy.deepcopy(ll)

print(lc)
lc[0][0] = 999

print(l)
print(lc)

import copy

l = [1,2,3,4]

ll = [l.copy(),l.copy()]
print(ll)

lc = copy.deepcopy(ll)

print(lc)
lc[0][0] = 999

print(l)
print(lc)

### 6.3 리스트 자료형의 메서드

- 리스트의 주요 메서드

import pprint

for i in dir(list) :
    if not i.startswith("__") :
        print(i)

- 리스트의 원소 추가 삭제하기

# 빈 리스트 생성
ll = []
# 리스트의 끝에 원소 추가: 정수
ll.append(1)
print(ll)

# 리스트의 끝에 원소 추가: 리스트
ll.append([2])
print(ll)

# 리스트에 값 삭제 : 삭제된 원소를 리턴값
a = ll.pop()
print(a)
print(ll)

ll.append(2)
print(ll)

#특정 위치(인덱스)를 가지고 삭제
b = ll.pop(0)
print(b)
print(ll)

- 리스트 합치기: extend

l8 = [1,2,3]
l10 = [4,5,6]

# l9에 다른 리스트 추가해서 합치기
l8.extend(l10)
print(l8)
l9 = [1,2,3]
print(l9 + l10)

- 리스트 특정 위치에 삽입 및 전체 삭제

# 특정 위치에 삽입하기
l2 = [1,2,3,4]
l2.insert(0,5)
print(l2)

l2.insert(0,5)
print(l2)

# 특정 값으로 삭제
l2.remove(5)
print(l2)

l2.remove(5)
print(l2)

# 전체 삭제
l2.clear()
print(l2)

- 리스트 내의 동일원소를 확인하고 삭제하기

l4 = [1,2,3,4,5,2,2,2]
# 동일한 원소 갯수 세기
c = l4.count(2)
print(c)

for i in range(c) :
    l4.remove(2)

print(l4)

- 리스트 원소들 정렬하기: sort, reverse

l4 = [1,2,3,4,5,2,2,2]

# 원소들 소팅 : 올림차수
l4.sort()
print(l4)

# 원소들 소팅 : 역순으로 소팅
l4.reverse()
print(l4)

# 소팅 기준을 주고 역순으로 소팅
l4.sort()
print(l4)

l4.sort(reverse=True)
print(l4)

### 6.4 리스트 내의 리스트 초기화 하기

- 리스트로 * 연산자 비교하기

l = [4,5,6]

l2 = l*2

print(l2)
l.extend(l)
print(l)

#### 6.4.1 리스트 초기화 시 주의사항

- 외부에 리스트 원소를 반복해서 처리하고 내부에 넣고 처리

row = [3] * 3
print(row)

li = []
for _ in range(3) :
    li.append(row)

print(li)

li[0][0] = 99

print(li)
print(row)

- 리스트를 for문 안에서 초기화

li = []
for _ in range(3) :
    row = [3] * 3
    print(id(row))
    li.append(row)

print(li)

li[0][0] = 99

print(li)
print(id(row), row)

- 지능형(comprehension) 리스트로 처리하기

li = [ [3]*3 for _ in range(3)]
print(li)

li[0][0] = 99

print(li)

squares = [x**2 for x in range(1, 6)]
print(squares)

even_squares = []
for i in range(1, 11, 2):
    even_squares.append(i**2)

print(even_squares)

even_squares = [x**2 for x in range(1, 11, 2)]
print(even_squares)

list_ = [3 * x for x in range(1, 11)]
print(list_)

list_ = [3 * x for x in range(1, 11) if x % 3 != 0]
print(list_)

list_ = [x * y for x in range(2, 5) for y in range(1, 10)]
print(list_)

list_ = [3 * x if x % 3 != 0 else 1 for x in range(1, 11) ]
print(list_)

## **7. Sequence 자료형 형 변환**

- 파이썬은 별도의 형 변환 기능이 없음
- Sequence 자료형, 숫자 자료형 등 형 변환은 새롭게 인스턴스를 만듦

### 7.1 클래스만 사용한 형 변환

l = [1,2,3,4]

t = tuple(l)
print(t)

s = str(l)
print(repr(s))

ls = list(s)
print(ls)

- 문자열 원소로 구성된 리스트나 튜플을 문자열로 변환

s = "str"

l = list(s)
print(l)

print("".join(l))

- 문자열이나 숫자로 구성된 리스트나 튜플을 문자열로 변환

l = [1,'h','e','l','l']

print(" ".join(l))

l = [1,'h','e','l','l']

print("".join(map(str,l)))