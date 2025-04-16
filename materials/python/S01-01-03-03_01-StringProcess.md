---
layout: page
title:  "파이썬 기본 문법: 3. 문자열 처리"
date:   2025-03-01 10:00:00 +0900
permalink: /material/python/S01-01-03-03_01-StringProcess
categories: materials
---

문자열을 조작하는 작업은 이후 데이터 분석, AI 학습을 위한 데이터 조작 등에서 많이 활용되므로 다양하게 익혀두는 것이 좋습니다.

## 1. 특정 위치의 문자 얻기

```python
txt1 = 'A tale that was not right'
txt2 = '이 또한 지나가리라'
```

```python
print(txt1[5])
print(txt2[-2])
```

## 2. 지정한 구간의 문자열 얻기

```python
print(txt1[3:7])
print(txt1[:6])
print(txt2[-4:])
```

```python
txt = 'python'

for i in range(len(txt)):
    print(i, ":", txt[:i+1])
```

## 3. 홀수 번째 문자만 추출하기

```python
txt = 'aAbBcCdDeEfFgG'

result = txt[::2]
print(result)
```

## 4. 문자열 거꾸로 만들기

```python
txt = 'abcdefg'

result = txt[::-1]
print(result)
```

## 5. 특정 문자가 있는지 확인하기

```python
msg = '안녕하세요'

if 'a' in msg:
    print('문자열에 a가 포험되어 있음')
else:
    print('문자열에 a가 포함되어 있지 않음')
```

## 6. 숫자인지 알파벳인지 검사하기

### 6.1 문자열이 숫자만으로 구성되었는지 확인하기

```python
txt1 = '010-1234-5678'
txt2 = 'R2D2'
txt3 = '1234'

result1 = txt1.isdigit()
result2 = txt2.isdigit()
result3 = txt3.isdigit()

print(result1)
print(result2)
print(result3)
```

### 6.2 문자열이 알파벳만으로 구성되었는지 확인하기

```python
txt1 = 'A'
txt2 = '안녕'
txt3 = 'Star Craft'
txt4 = '3PO'

result1 = txt1.isalpha()
result2 = txt2.isalpha()
result3 = txt3.isalpha()
result4 = txt3.isalpha()

print(result1)
print(result2)
print(result3)
print(result4)
```

### 6.3 문자열이 알파벳과 숫자로 구성되었는지 확인하기

```python
txt1 = '안녕하세요?'
txt2 = '1. Title-제목을 입력하세요'
txt3 = '3피오R2D2'

result1 = txt1.isalnum()
result2 = txt2.isalnum()
result3 = txt3.isalnum()

print(result1)
print(result2)
print(result3)
```
