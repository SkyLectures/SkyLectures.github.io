---
layout: page
title:  "파이썬 중급: Mapping & Set 자료형"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-02-05-01_01-MappingSetDataTypes
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

# **Mapping & Set Data Type**

- Sequence 자료형과 같이 여러 개의 원소를 관리하고 처리할 수 있는 Collection 형태의 자료형
- 원고를 검색할 때 Index가 아닌 Key로 접근하고 읽어서 처리함
- Key로 검색해서 읽기 이해서는 유일성을 유지해야 하므로 Key를 생성할 때 Hash 알고리즘을 통해 유일한 값만 구성함
<br><br>
- 구분
    - Mapping Data Type(매핑 자료형)
        - Key와 Value(값)를 쌍으로 관리하는 자료형
        - 종류
            - Dictionary: 원소의 변경, 추가, 삭제가 가능함
    - Set Data Type(집합 자료형)
        - Key만으로 관리하는 자료형
        - 종류
            - Set: 원소의 변경, 추가, 삭제가 가능함
            - Frozenset: 원소의 변경이 불가능함

## **1. Dictionary**

### 1.1 Dictionary의 Key 구성 및 생성 기준

- Key는 유일성을 유지하는 자료형만 지원되며 Value는 모든 값을 지원함
- 항상 Key와 Value의 쌍으로 구성됨
- Key는 Hash 알고리즘을 적용하여 유일한 값으로 생성되며, 변경이 불가능한 자료형(int, float, tuple, str, bytes, frozenset 등)으로 생성됨
    - 변경이 가능한 List, Dictionary 등은 원소들이 변경되어 동일한 형태를 유지할 수 없으므로 Key로 구성할 수 없음
    - Tuple도 완전한 불변성을 유지하지 못하므로 원소의 일부 List가 들어올 경우, Key로 사용할 수 없음

- Dictionary를 생성할 때 우리가 Key-Value를 입력하는데 Hash 알고리즘을 통한 생성이란 것은 무슨 말인가?
    - Dictionary는 Key를 관리하기 위한 Hash라는 구조가 별도로 생성되며, 이를 기반으로 다양한 값을 가진 인스턴스와 1:1 매핑된 구조가 만들어짐
    - 이러한 1:1 형태의 매핑 구조를 유지하기 때문에 Key가 중복되어 관리되지 않음

- Key를 구성하는 Hash의 기본 정보
    - Hash는 모듈 sys 내의 hash_info 속성에서 관리함
        - width: Hash 값에 사용되는 비트의 너비
        - hash_bits: Hash 알고리즘의 내부 출력 크기
        - seed_bits: Hash 알고리즘의 Seed Key 크기
        - inf: 양의 무한대의 Hash 값을 반환
        - nan: Nan에 대한 Hash 값
        - algorithm: Hash 알고리즘의 이름(str, bytes, memoryview)

import sys

print(sys.hash_info.width)
print(sys.hash_info.hash_bits)
print(sys.hash_info.seed_bits)
print(sys.hash_info.inf)
print(sys.hash_info.nan)
print(sys.hash_info.algorithm)

### 1.2 Dictionary 생성하기

- 빈 Dictionary 생성하기

d = {}
print(type(d),d)

dc = dict()
print(type(dc),dc)

- 키워드 인자에 동일한 Key를 중복시킬 경우

d = dict(a=10,b=20,a=20)

# 키워드 인자의 이름을 변경하면 값이 같더라도 키가 다르면 생성 가능
d = dict(a=10, b=20, c=20)
print(d)

- Tuple 원소 List로 동일한 Key를 중복시킬 경우

# Tuple의 구성이 다르기때문에 예외가 발생하지 않음. 대신 키가 같으면 나중에 들어온 것으로 대체됨
l = [('a',1),('b',2),('a',3)]

d = dict(l)
print(d)

- Dictionary의 Key에 대한 처리 방식

d = {'name': 'John', 'age': 30 }
print(type(d), d)

d = {1: 'apple', 2: 'ball'}
print(type(d), d)

d = {'name': 'John', 1: [2, 4, 3]}
print(type(d), d)

- bytes, frozenset을 Key로 사용하는 경우

b = bytes(b'123')
d = {b:1}
print(d)

b = frozenset([1,2,3])
d = {b:1}
print(d)

- 2 Tuple 원소를 가진 List 인자

# using dict()
my_dict = dict({1:'apple', 2:'ball'})
print(my_dict)

# from sequence having each item as a pair
my_dict1 = dict([(1,'apple'), (2,'ball')])
print(my_dict1)

- List나 Tuple을 이용해서 Key만 생성하기

l = [1,2,3,4]
d = {}
d = d.fromkeys(l)
print(d)

t = (1,2,3,4)
d = {}
d = d.fromkeys(t)
print(d)

a = (1,2,[1,2])
d = {}
d.fromkeys(a)

### 1.3 Dictionary에 요소 추가하기

d = {1:1,2:2}
d2 = dict([("a",'apple'), ("b",'ball')])

d.update(d2)
print(d)

d = {1:1,2:2}
d2 = dict([("a",'apple'), (1,'ball')])

d.update(d2)
print(d)

solar1 = ['태양', '수성', '금성', '지구', '화성', '목성', '토성', '천왕성', '해왕성']
solar2 = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
solardict = {}

for i, k in enumerate(solar1):
   val = solar2[i]
   solardict[k] = val

print(solardict)

### 1.4 Dictionary의 특정 요소 값 변경하기

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
names['Aimy'] = 10000
print(names)

### 1.5 Dictionary의 특정 요소 제거하기

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
del names['Sams']
print(names)

my_dict = {}

for i in range(10) :
    my_dict[i] = i
print(my_dict)

d = my_dict.popitem()
print(d)
print(my_dict)

f = my_dict.popitem()
print(f)
print(my_dict)

f1 = my_dict.pop(1)
print(f1)
print(my_dict)

### 1.6 Dictionary의 모든 요소 제거하기

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
names.clear()
print(names)

### 1.7 Dictionary에서 Key만 추출하기

d = dict([('a',1),('b',2)])

keys = d.keys()
print(type(keys))

keys = iter(keys)
print(next(keys))
print(next(keys))
print(next(keys))

l = [('a',1), ('b',2)]
d = dict(l)
print(d.keys())

for i in d.keys() :
    print(i)

print(list(d.keys()))

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
ks = names.keys()
print(ks)

for k in ks:
   print('Key:%s \tValue:%d' %(k, names[k]))

### 1.8 Dictionary에서 값(Value)만 추출하기

l = [('a',1), ('b',2)]
d = dict(l)

print(d.values())
print(list(d.values()))

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
vals = names.values()
print(vals)

vals_list = list(vals)
ret = sum(vals_list)
print('출생아수 총계: %d' %ret)

### 1.9 Dictionary에서 모든 요소 추출하기

l = [('a',1), ('b',2)]
d = dict(l)
print(d.items())

print(list(d.items()))

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
items = names.items()
print(items)

for item in items:
   print(item)

### 1.10 Dictionary 내부 원소를 조회하고 Tuple/Set 반환하기

l = [('a',1), ('b',2)]
d = dict(l)

# 튜플로 데이터 변환
print(tuple(d.items()))
print(tuple(d.keys()))
print(tuple(d.values()))

# set로 데이터 변환
print(set(d.items()))
print(set(d.keys()))
print(set(d.values()))

### 1.11 Dictionary에 특정 Key가 존재하는지 확인하기

l = [('a',1), ('b',2)]
d = dict(l)
print(d['c'])

l = [('a',1), ('b',2)]
d = dict(l)

print(d.get("c", "default value"))

l = [('a',1), ('b',2)]
d = dict(l)

# 조회하고 없으면 세팅하기
d.setdefault("c", "default value")

print(d['c'])
print(d)

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
k = input('이름을 입력하세요: ')

if k in names:
   print('이름이 <%s>인 출생아수는 <%d>명 입니다.' %(k, names[k]))
else:
   print('자료에 <%s>인 이름이 존재하지 않습니다.' %k)

### 1.12 Dictionary 정렬하기

names = {'Mary':10999, 'Sams':2111, 'Aimy':9778, 'Tom':20245, 'Michale':27115, 'Bob':5887, 'Kelly':7855}
ret1 = sorted(names)
print(ret1)

def f1(x):
   return x[0]

def f2(x):
   return x[1]

ret2 = sorted(names.items(), key=f1)
print(ret2)

ret3 = sorted(names.items(), key=f2)
print(ret3)

ret4 = sorted(names.items(), key=f2, reverse=True)
print(ret4)

## **2. Set**

- 일반적인 수학의 집합을 구현한 자료형
    - 집합에서 처리하는 산술식을 메서드로 제공
- 집합은 동일한 값이 여러 번 나올 수 없으므로 원소를 Hash로 처리하여 유일성을 유지함
- Dictionary와 동일하게 {}로 표기하지만 내부에 Value가 없고 Key만 존재함

### 2.1 set 생성하기

- 빈 set은 반드시 set()로 생성

# 빈 set 생성

s = set()
d = {}

# {}는 dict 타입 처리용
# set 일 경우는 반드시 set()으로 빈 set 생성
print(type(s))
print(type(d))

- 리터럴이나 생성자로 set 만들기

l = set([1,2,3,'a','b'])
s = set("abc")
print(l)
print(s)

# 리터럴로 생성
sl = {1,2,3}
print(sl)

### 2.2 set 기본 연산 처리

l = set([1,2,3,'a','b'])
s = set("abc")

#합집합 처리
u = l | s
print(u)

u = l.union(s)
print(u)

#교집합 처리
u = l & s
print(u)
u = l.intersection(s)
print(u)

#차집합
u = l - s
print(u)

u = l.difference(s)
print(u)

# 집합에 대한 대칭 차집합
u = l ^ s
print(u)

u = l.symmetric_difference(s)
print(u)

### 2.3 집합연산을 통해 자기 내부 변경하기

ll = set([1,2,3,'a','b'])
ss = set("abc")

ll.difference_update(ss)
print(ll)

ll = set([1,2,3,'a','b'])
ss = set("abc")
ll.intersection_update(ss)
print(ll)

ll = set([1,2,3,'a','b'])
ss = set("abc")
ll.symmetric_difference_update(ss)
print(ll)

### 2.4 집합 원소 처리

s = set([1,2,3,'a','b'])

# 원소 추가
s.add('c')
print(s)

# 원소 추가
s.update({4,5,})
print(s)

s = set([1,2,3,'a','b'])

s.remove('b')
print(s)

s.remove('c')
print(s)

s = set([1,2,3,'a','b'])
sp = s.pop()
print(sp)
print(s)

# 삭제할 것을 넣어주면 됨
sp = s.discard('a')
print(sp)
print(s)

# remove는 키가 반드시 있어야 하지만 discard는 없어도 됨
sp = s.discard('d')
print(sp)

### 2.5 집합 간의 관계 확인 연산자 및 메소드

s = set([1,2,3,'a','b'])
ss = set([1,2,3])

print(ss < s)
print(ss.issubset(s))

s = set([1,2,3,'a','b'])
ss = set([1,2,3,'a','b'])

print(ss <= s)
print(ss.issubset(s))

## **3. Frozenset**

- 집합 자료형에서는 변경가능한 set과 변경불가능한 frozenset을 제공함
- set은 Dictionary의 키로 사용하지 못하므로 불변형을 제공하고 필요한 경우 frozenset으로 형변환을 해서 처리함

### 3.1 frozenset 생성하기

# 빈 frozenset 생성하기
s = frozenset()

print(s)
print(type(s))

s = frozenset([1,3,4])
l = frozenset([1,2,4])
print(s)
print(l)

### 3.2 set과의 차이

- set은 변경불가의 자료형이므로 집합연산의 결과는 내부를 변경하지 않고 새로운 인스턴스 객체를 만듦
- 집합연산 자체는 set과 동일하며, 집합연산을 하고 난 후에 다른 인스턴스를 생성하므로 원본 frozenset은 갱신되지 않음

s = frozenset([1,3,4])
l = frozenset([1,2,4])

u = s.union(l)
print(u)
u = s.intersection(l)
print(u)
u = s.difference(l)
print(u)

# 변경할 수 없으므로 그대로 유지
print(s)
print(l)

