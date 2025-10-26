---
layout: page
title:  "파이썬 기본 문법(통합)"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S01-01-03-01_01-PythonBasic
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

> 본 페이지에 한하여 직접 설명하는 식으로 작성되었습니다.<br>
> <span style="color: #C00">프로그래밍을 처음 접하는 사람</span>을 대상으로 설명합니다.<br>
> 천천히 읽으면서 실습을 하고 다음 단계로 넘어가세요.
{: .summary-quote}

> **프로그래밍이란?**<br><br>
> "프로그래밍"이란 것을 어려운 것으로 인식하는 경우가 많습니다.<br>
> 그러나 프로그래밍의 본질은 아주 단순합니다.<br><br>
> 흔히 알려진 것처럼 컴퓨터 프로그래밍이란 "**컴퓨터에게 일을 시키기 위한 명령어 모음을 만드는 일**"이라고 할 수 있습니다.<br>
> 조금 더 정확하게 말하면 "<span style="color: #00C">**컴퓨터에게 일을 시키기 위한 명령어들의** <span style="color: #C00">**순차적인**</span> **모음을 만드는 작업**</span>"이라고 할 수 있습니다.<br><br>
> 그리고 이 작업은 단순하게 보면, <span style="color: #C00">**우리가 가지고 있는 데이터를 우리가 원하는 형태로 가공, 처리하는 과정을 정리하는 작업**</span>일 뿐입니다.<br>
> 우리가 어떤 데이터를 가지고 있고, 그 데이터를 특정 형태로 바꾸기를 원할 때,그 바꾸는 과정을 몇 개의 명령어를 이용하여 규칙에 따라 순서대로 적어주면 끝입니다.<br><br>
>그래서 프로그래밍 작업에서의 코드 구성은 <span style="color: #C00">**데이터 입력 → 데이터 처리 → 데이터 출력**</span>이라는 과정으로 이루어집니다.<br><br>
>
> <div class="insert-image" style="text-align: center;">
>   <img style="width: 600px;" src="/materials/python/images/S01-01-03-01_01-001.png"><br>
>   <div style="text-align: right;width: 790px;">그림출처: 아이다랩(AiDALab)</div>
> </div>
> <div class="contents-source" style="text-align: left;">
>   <a href="https://blog.naver.com/aida-smart/223744963918" target="_blank">(참고) 그럼 왜 그렇게 어렵게 보일까요?</a>
> </div>
{: .expert-quote}

- [<span style="color: #0A0;font-weight: bold;">Colab에서 실습파일 열기</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkyLectures/SkyLectures.github.io/blob/main/materials/ai/notebooks/S03-03-02-04_01-DnnCnn.ipynb)

## 1. 첫번째 파이썬 프로그램

<div class="insert-image" style="text-align: left;">
    <img style="width: 450px;" src="/materials/python/images/S01-01-03-01_01-002.png">
</div>

- 매우 단순해 보이지만 위의 코드는 제대로 작동하는 프로그램입니다.<br><br>

- 위의 내용을 입력과 출력으로 분리하는 작업을 생각해 봅시다<br>
  - 먼저 데이터를 입력을 하려면 그 데이터를 담아둘 통이 필요하겠죠.<br>
  그냥 휙 던지면 컴퓨터는 그게 데이터인지 아닌지 모릅니다.<br>
  그래서 통에 넣어서 이게 데이터다.. 라고 알려주는데, 이 통을 <span style="color: #C00">변수, 상수</span> 등의 이름으로 부릅니다.<br>
  이 중에서 앞으로 계속 바꿔가면서 사용할 통을 변수라고 하죠.<br><br>

  - 그럼 컴퓨터가 이 변수를 구분해서 사용할 수 있도록 변수에 이름을 줘야 하겠죠.<br>
  우리는 두 개의 숫자를 데이터로 줄 것이니까 변수도 2개를 만들 것입니다.<br>
  이름은 a와 b라고 하죠.<br>
  a에는 1을, b에는 2를 던져 줄 것입니다.<br><br>

  - 다음으로 a와 b를 더해주는 연산을 수행합니다.<br>
  연산의 결과는 c라고 하는 변수를 만들어서 그 안에 넣어줍시다.<br><br>

  - 이제 연산이 끝난 c의 값을 출력해 봅니다.<br>

<div class="insert-image" style="text-align: left;">
    <img style="width: 600px;" src="/materials/python/images/S01-01-03-01_01-003.png">
</div>

- 이렇게 하나의 프로그램 코드가 완성되었습니다.<br>
위의 방법이 프로그래밍의 가장 기본적인 방법입니다.<br><br>


## 2. 자료형 (Data Type)

### 2.1 기본 자료형

- 데이터에는 여러 종류가 있습니다.<br>
1, 2, 3... 과 같은 자연수 뿐만 아니라 -1, -2.. 와 같은 음수를 포함한 정수의 개념도 있고.. 0.123, -2.532 외 같은 소수점을 가진 실수도 있습니다.<br>
점점 들어가면 제곱했을때 음수의 값을 가지는 허수를 포함한 복소수라는 것도 있죠.<br>
수치 데이터를 벗어나 a, b, c, d... 가, 나, 다, 라.. 와 같은 문자로 구성된 데이터도 있습니다.<br><br>

- 파이썬에서는 데이터의 형태를 직접 선언하지 않아도 됩니다.

    ```python
    a = 1               # 자연수(=양의 정수)
    b = -2              # 음의 정수
    c = 0.2             # 실수
    d = 3+5j            # 복소수 → 파이썬에서는 복소수 표현 시 i가 아니라 j를 사용함
    e = "Hello"         # 문자열(영어)
    f = "안녕하세요"     # 문자열(한글)
    ```

    ```python
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    ```

    ```python
    print(a, ':', type(a))
    print(b, ':', type(b))
    print(c, ':', type(c))
    print(d, ':', type(d))
    print(e, ':', type(e))
    print(f, ':', type(f))
    ```

- 그런데 a, b, c, d, e.... 변수 명이 단조롭죠.<br>
프로그램이 점점 커지고 변수의 이름을 정의한 곳과 변수를 사용한 곳의 거리가 멀어질수록, 그리고 변수의 개수가 많아질수록 헷갈리기 시작합니다.<br>
그래서 변수의 명은 가능하면 쉽게 알아볼 수 있는 단어를 사용하시는 것이 좋습니다.<br><br>

- 변수의 이름은 기억하기 쉽고 구분하기 쉽게 작성하시면 됩니다.<br>
변수의 이름을 정할 때 반드시 따라야 하는 규칙같은 것은 없지만 각 언어에 따라, 또는 팀에 따라 권장하는 형태는 있습니다.<br>
혼자만 작성하고 보고 사용할 코드가 아니라면 언어별, 팀별로 권장하는 규칙을 따라주는 것이 협업에 도움이 되겠죠.

    ```python
    int_data_1 = 1
    int_data_2 = -2
    float_data = 0.2
    complex_data = 3+5j
    string_data_eng = "Hello"
    string_data_kor = "안녕하세요"
    ```
    ```python
    print(int_data_1)
    print(int_data_2)
    print(float_data)
    print(complex_data)
    print(string_data_eng)
    print(string_data_kor)
    ```

### 2.2 집합 자료형

- 다음으로 지금까지 사용해 본 변수들이 하나가 아니라 여러 개가 모여서 구성된 데이터의 형태가 있습니다.<br>
리스트, 튜플, 딕셔너리가 그것입니다.

- **리스트(List)**

    - 리스트(List)는 대부분의 프로그래밍 언어에서 <span style="color: #C00">배열, Array라고 표현하는 것과 비슷</span>합니다.<br>
    - 배열은 동일한 형태(실수끼리, 문자열끼리... 등)의 데이터 여러 개가 순서대로 모여 있는 것을 말하지만 파이썬의 List는 꼭 <span style="color: #C00">동일한 형태가 아니어도 관계없습니다.</span><br>
    예를 들면 Java의 Collection 이란 것과 비슷하다고 볼 수 있습니다.(Collection이 무엇인지 몰라도 상관없습니다. 그냥 참고만 하세요.)<br><br>

    - 리스트의 데이터는 대괄호 [ ]로 둘러싸서 표시하며 내부의 데이터는 콤마( , )로 구분합니다.

        ```python
        list_1 = [1, 3, 5, 7, 9]
        list_2 = ['hello', 'good morning', 'a', 'b', 'c']
        list_3 = [2, 4, 'bye', [6, 8, 10], ['see', 'you', 'later']]
        ```

    - list_1에는 숫자만, list_2에는 문자열과 문자, list_3에는 숫자, 문자열, 또다른 리스트들이 들어가 있죠.<br>
    각각을 출력해보면 다 잘 인식되고, 처리되고 있습니다.

        ```python
        print(list_1)
        print(list_2)
        print(list_3)
        ```

    - 리스트 안의 데이터를 하나씩 지정하고 싶을때는 [x] 와 같은 형태로 사용하시면 됩니다.<br>
    [x]에서 x에는 리스트 안의 몇 번째 데이터를 사용할 것인지 지정하는 숫자가 들어가는데 이 때 숫자, 즉 인덱스는 0부터 시작합니다.<br>
    리스트 안의 리스트를 지정하여 그 안의 데이터를 사용하려면 한 단계 더 지정해 주시면 됩니다.

        ```python
        print(list_1[0])
        print(list_2[1])
        print(list_3[4][1])
        ```

    - print(list_3[4][1]) 의 의미는 list_3 변수에서 0, 1, 2, 3, 4에서 4, 즉 다섯 번째의 값인 ['see', 'you', 'later']를 지정하고, 그 안에서 0, 1 즉 두 번째의 데이터인 'you'를 지정한 것입니다.

- **튜플(Tuple)**

    - 튜플(Tuple)은 리스트와 비슷한데 리스트는 안에 들어있는 값을 바꿀 수 있는 반면에 튜플은 <span style="color: #C00">바꿀 수 없다</span>는 차이가 있습니다.<br>
    튜플의 데이터는 괄호 ( ) 로 둘러싸서 표시하며 안에 있는 데이터는 콤마( , )로 구분합니다.

        ```python
        list_4 = [1, 2, 3, 4, 5]
        tuple_1 = (1, 2, 3, 4, 5)

        print(list_4)
        list_4[2] = 10
        print(list_4)
        ```

        ```python
        print(tuple_1)
        tuple_1[2] = 10
        print(tuple_1)
        ```

    - 리스트와 달리 튜플에서는 안에 있는 값을 바꾸려고 하니 오류가 발생하는 것을 확인할 수 있습니다.


- **딕셔너리(Dictionary)**

    - 마지막으로 사전, 딕셔너리(Dictionary)라고 부르는 데이터의 형태가 있습니다.<br><br>

    - 딕셔너리의 데이터는 중괄호 { } 로 둘러싸서 표시하며 내부의 데이터는 콤마( , )로 구분합니다.
    - 딕셔너리의 데이터는 "키:값"의 형태를 가진 한 쌍의 요소로 구성되어 있습니다.
    - 리스트, 튜플과 달리 <span style="color: #C00">딕셔너리는 데이터에 인덱스가 지정되어있지 않습니다.</span>
    - "키:값"으로 구성된 데이터 쌍에서 "키"를 [ ] 안에 입력하여 원하는 데이터를 지정하고 사용합니다.<br><br>

    - <span style="color: #C00">딕셔너리의 키와 값에는 어떤 데이터라도 사용할 수 있습니다.</span><br>
    딕셔너리의 값에는 리스트, 튜플, 딕셔너리가 들어갈 수도 있습니다.<br>
    그런데 "키"라는 것은 단 하나를 지칭하기 위한 것이니까 여기에는 리스트와 같은 것을 지정하시면 안되겠죠.<br>
    그리고 리스트와 마찬가지로 딕셔너리도 내부의 값을 바꿀 수 있습니다.

        ```python
        dictionary_1 = {0:'False', 'a':"small A", "b":"small B", 'three':3, 'four':4, 1:['a', 'b']}
        print(dictionary_1)
        ```

        ```python
        print(dictionary_1[1])
        print(dictionary_1["a"])
        print(dictionary_1['four'])
        ```

        ```python
        dictionary_1["b"] = "alphabet small B"
        print(dictionary_1["b"])
        ```


## 3. 조건문 (if 구문)

- 데이터의 처리는 단순한 사칙연산만 있는게 아니죠.<br>
이럴땐 이렇게 처리하고, 저럴땐 저렇게 처리하고.. 우리가 원하는 여러가지 처리를 위한 공식이나 조건들이 있을것입니다.<br>
컴퓨터는 우리가 머리로 생각하는 것처럼 이것저것 뚝딱뚝딱 처리하지 못합니다.<br>
그냥 시키는대로.. 순서대로.. 쭈~욱 이어서 처리를 해 나가는거죠.<br><br>

- <span style="color: #C00">그러면 우리는 무엇을 어떻게 시켜야할까요?</span>
    - 그냥 컴퓨터가 할 수 있는 방식으로 순서대로 쭉 이어 가다가<br>
    <span style="color: #00C">이런 경우에는 이렇게 해라~라고 조건을 주고 그 조건에 따른 동작을 지정</span>해 주면 되겠죠.<br><br>

- 이렇게 <span style="color: #00C">컴퓨터에게 처리를 위한 조건을 알려주는 구문</span>을 <span style="color: #C00">**조건문**</span>이라고 합니다.
    - 가장 기본적인 조건문은 <span style="color: #C00">if ☆☆☆ else ○○○</span> 구문이 있습니다.<br>
        <span style="color: #C00">만약 ☆☆☆이면 ○○○를 해라.</span> 라는 의미입니다.

    ```python
    a = 10
    b = 20

    if a > b:
        print("a가 b보다 큽니다.")

    else:
        print("a가 b보다 크지 않습니다.")
    ```

- 그런데 여기서 특이한 것이 보입니다.<br>
    <span style="color: #C00">if a > b:</span> 와 같이 마지막에 <span style="color: #C00">콜론( : )</span>이 붙어있습니다.<br>
    예전 코드에는 이런 것은 없었죠.
- 만약 a > b 라면.. 이라는 구문을 만족할때 우리는 "a가 b보다 큽니다."라는 문자열을 출력하려고 하는데..<br>
    그 구문을 만족할 때 과언 어디까지 실행을 해야 할까요?<br>
    그것을 지정해 주지 않으면 끝까지 흘러가 버리겠죠.<br>
    그래서 우리는 명령들을 블록이란 것으로 나누어서 표현합니다.<br>
    <span style="color: #C00">콜론( : )</span>은 블록을 지정하는 구문의 끝에 붙여서 <span style="color: #00C">다음 줄부터 블록이 시작된다..</span> 라고 표시해 주는 것입니다.<br><br>

- 그럼 <span style="color: #C00">블록의 끝</span>은 어떻게 표시할까요?<br>
    파이썬에서는 블록의 구분을 <span style="color: #C00">들여쓰기</span>를 이용해서 지정합니다.
- 소스코드를 보시면 print 구문이 몇 칸 안으로 들어와있죠?<br>
    저렇게 같은 간격으로 들여쓰기 된 구문들은 모두 같은 블록이다.. 라고 인식하는 것입니다.<br>
    그렇다면 블록의 끝을 지나면 들여쓰기의 간격을 원래대로 돌려두면 되겠죠.<br>
    위의 코드를 보시면 print 구문이 끝나고 들여쓰기가 해제되었습니다.<br>
    이제 블록은 끝났다는 뜻입니다.<br>
- 그럼 if 구문에 따른 블록은 끝났으니 다음 명령어로 내려가겠죠.<br>
    <span style="color: #C00">else</span> 라는 구문을 만났습니다.<br>
    역시 콜론( : )을 통해서 블록의 시작을 알려주고 print 구문을 실행했습니다.<br><br>

- 이것으로 if ☆☆☆ else OOO 조건문이 끝났습니다.<br><br>

- 그런데 조건이 딱 한 가지만 있진 않겠죠.<br>
    이럴땐 이렇게, 저럴땐 저렇게, 그 외에도 또 다른 조건을 동시에 지정할 수 있습니다.<br>
    이럴 경우에는 <span style="color: #C00">if ☆☆☆ else ○○○ 문을 중첩</span>해서 사용하는 방법과 if ☆☆☆ elif ○○○ else ◇◇◇ 와 같이 <span style="color: #C00">if ☆☆☆ else ○○○ 구문이 확장된 표현</span>을 사용하는 방법중 하나를 선택할 수 있습니다.

    ```python
    a = 20
    b = 20

    if a > b:
        print("a가 b보다 큽니다.")
    elif a == b:
        print("a와 b가 같습니다.")
    else:
        print("a가 b보다 작습니다.")
    ```

    ```python
    a = 30
    b = 20

    if a > b:
        print("a가 b보다 큽니다.")
    else:
        if a == b:
            print("a와 b가 같습니다.")
        else:
            print("a가 b보다 작습니다.")
    ```

- 위의 코드에서 a == b 라는 구문은 a와 b가 같다는 것을 의미합니다.<br>
    일반적으로 프로그래밍에서는 왼쪽 데이터, 변수와 오른쪽 데이터, 변수가 같다고 표현할 때에는 <span style="color: #C00">==</span>를 사용합니다.<br>
    왼쪽 항이 크다 (>), 왼쪽 항이 작다 (<)의 경우는 기존의 수학(산수?)과 같은 부호를 사용하는데 같다의 경우에는 기존의 등호(=)를 사용하지 않죠.
- 이유는 기존의 등호 (=)는 오른쪽 항의 값을 왼쪽 항에 넣는다는 의미로 사용되기 때문에 두 가지를 구분하기 위해서 입니다.<br>
    그래서 기존의 등호 (=)는 할당연산자(또는 대입연산자, 이항연산자)라고 불립니다.<br>
    파이썬에도 다양한 연산자가 있는데 앞으로 필요할 때마다 설명을 드리도록 하겠습니다.<br><br>

- 위의 두 종류의 코드는 같은 결과를 출력해 줍니다.<br>
    여기서 우리가 알 수 있는 것은 if ☆☆☆ elif ○○○ else ◇◇◇ 구문만이 아니라.. 동일한 결과를 위한 <span style="color: #C00">코드의 작성에 정답은 없다..</span> 라는 것입니다.<br>
    if ☆☆☆ elif ○○○ else ◇◇◇ 구문을 사용해도 되고 if ☆☆☆ else ○○○ 구문을 중첩해서 사용해도 됩니다.<br>
    나중에 배울 또 다른 조건문을 사용해도 되죠. 편하신 대로 코드를 작성하시면 됩니다.<br><br>

- 물론 좀 더 실력이 좋아지고 컴퓨터가 처리하는 내부과정까지 이해하게 되면 각 구문들의 차이점이나 효율성 등을 따져서 작성하실 수 있습니다.<br>
    그런 프로그래밍을 위해서 더욱 빠르고 효율적인 처리가 가능하도록 정리된 것이 흔히 말하는 좋은 알고리즘이라고 하는 것들이죠.<br>
    아직까지는 거기까지 신경쓰지 않으셔도 됩니다.<br><br>
- 지금까지 가장 기본적인 조건문에 대하여 살펴보았습니다.

## 4. 반복문

- 사람은 하기 힘들어하지만 <span style="color: #C00">컴퓨터가 가장 잘 하는 일</span>은 무엇일까요?<br>
    <span style="color: #C00">바로 단순한 작업을 계속해서 반복하는 것</span>이겠죠.<br>
    사람에게 단순 작업을 반복해서 시키면 금방 지겨워져서 집중해서 일을 하지 못할 것입니다.<br>
    실수도 늘어나겠죠. 그렇지만 컴퓨터는 그럴 일이 없으니 반복 작업을 시키기에는 가장 좋은 일꾼입니다.<br>
    따라서 반복작업을 지시하는 명령은 컴퓨터에게 일을 잘 시키기 위한 매우 중요한 수단입니다.<br><br>

- 파이썬에서 사용하는 반복문은 크게 <span style="color: #C00">For 문</span>과 <span style="color: #C00">While 문</span>의 두 가지로 구분됩니다.<br>
    그 중에서 가장 많이 사용되는 것이 For 문인데.. For 문은 다양한 활용 형태를 가집니다.

### 4.1 for 문

- **기본형**

    - 먼저 가장 기본적인 형태를 살펴보겠습니다.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
        ```

    - 먼저 1, 3, 4, 6, 7을 값으로 가지는 리스트를 area라는 변수에 할당했습니다.<br>
        그 다음의 for 문은 area 리스트 안의 값을 순서대로 하나씩 꺼내서 x에 할당하고, x에 하나의 값이 할당될 때 마다 print 문을 실행하라~ 라는 의미입니다.<br>
        그렇다면 area는 총 5개의 값을 가지고 있으니 print 구문을 5번 실행하겠죠.<br>
        그리고 출력되는 내용인 x는 area 리스트에서 순서대로 꺼낸 1, 3, 4, 6, 7이 순서대로 출력될 것입니다.<br><br>

    - for 구문에는 <span style="color: #C00">리스트 외에도 문자열</span>(문자열도 엄밀히 말하면 문자들의 리스트입니다), <span style="color: #C00">딕셔너리, 범위 등이 사용</span>될 수 있습니다.<br><br>
    - 각각의 예제를 살펴봅시다.

        ```python
        # 문자열을 사용한 예
        area_string = "Hello"

        for x in area_string:
            print(x)
        ```

        ```python
        # 딕셔너리를 사용한 예
        area_dictionary = {'a':'Nice', 'b':'to', 'c':'meet', 'd':'you'}

        for x in area_dictionary:
            print(x)
        ```

    - 딕셔너리를 사용한 예에서도 4개의 값을 잘 출력했습니다.<br>
        그런데 "Nice", "to", "meet", "you"를 출력하고 싶은데 이것은 어떻게 할까요?<br>
        for 구문이 사용하는 것은 딕셔너리에 포함된 값을 순서대로 꺼낸 것이기 때문에 "키:값"의 쌍에서 "키"만을 사용하고 있습니다.<br>
        그러면 우리는 area_dictionary 변수를 사용해서 출력해 줄 수 있겠죠.<br>
        키를 사용할 수 있으니까요.

        ```python
        area_dictionary = {'a':'Nice', 'b':'to', 'c':'meet', 'd':'you'}

        for x in area_dictionary:
            print(x, ':', area_dictionary[x])
        ```

    - 키와 값이 모두 잘 출력되었습니다.<br>
        그런데 파이썬에서는 <span style="color: #C00">함수의 결과로 하나의 값만 받는 것이 아니라 두 개 이상의 값을 받을 수 있습니다.</span><br>
        그래서 area_dictionary 라는 딕셔너리 변수에서 우리가 사용할 수 있는 items라는 함수를 한 번 사용해 보겠습니다.

        ```python
        area_dictionary = {'a':'Nice', 'b':'to', 'c':'meet', 'd':'you'}

        for key, value in area_dictionary.items():
            print(key, ':', value)
        ```

    - <span style="color: #C00">파이썬이 사용하는 모든 데이터 타입(형태)은 클래스라는 구조로 만들어져</span> 있습니다.<br>
        클래스에 대한 것은 다음에 설명드리도록 하겠습니다.<br>
        지금은 클래스로 되어 있다는 것만 아셔도 됩니다.<br>
        아무튼.. 모든 데이터 타입은 클래스로 되어 있기때문에 클래스가 가지고 있는 다양한 함수를 끌어내어 사용할 수 있습니다.<br><br>

    - 바로 위의 예제에서 사용한 items( ) 라는 것도 딕셔너리 클래스가 가지고 있는 함수 중의 하나입니다.<br>
        딕셔너리 변수가 가지고 있는 각 항목들을 키와 값의 쌍으로 돌려주는 기능을 하죠.<br>
        따라서 for 구문을 실행하는데 area_dictionary가 가지고 있는 키와 값의 쌍을 순서대로 꺼내어서 그 개수만큼 for문의 블록 안에 있는 구문을 실행하라~ 라는 의미를 가지게 됩니다.<br>
        그리고 area_dictionary 변수의 키:값 쌍을 함께 받아와서 키는 key 변수에, 값은 value 변수에 넣어두었기 때문에 key 변수와 value 변수를 출력하는 print문을 블록에 사용했습니다.<br><br>
    - 앞에서 본 예제와 같은 결과를 확인할 수 있습니다.

- **range( ) 함수 사용**

    - 다음은 range 함수를 사용한 예제를 살펴보겠습니다.

        ```python
        for x in range(5):
            print(x)
        ```

        ```python
        for x in range(3, 5):
            print(x)
        ```

    - <span style="color: #C00">range( ) 함수는 내부에 지정된 범위 안의 숫자를 돌려주는 함수</span>입니다.<br>
        위의 예제 중 첫 번째는 먼저 0부터 5의 앞까지의 범위 안에 있는 숫자(0, 1, 2, 3, 4)를 순서대로 꺼내어 print 문으로 출력을 반복하라.. 라는 의미입니다.<br>
        이때, range( ) 함수 안의 범위는 위의 두 번째 예제와 같이 지정할 수 있습니다.<br>
        3부터 5의 앞까지의 범위 안의 숫자 (3, 4)를 순서대로 꺼내어 print 문으로 출력을 반복하라.. 라는 의미죠.<br>
        범위를 지정할 때 뒤에 있는 5는 5까지.. 의 의미가 아니라 5의 앞까지.. 의 범위입니다.<br>
        또한 range( ) 함수에서는 범위 안의 숫자 사이의 간격을 지정할 수도 있습니다.<br><br>

    - 다음 예제는 3부터 10의 앞까지의 숫자를 꺼내되, 각 숫자는 2씩 증가하도록 꺼내어서 print 문으로 출력하라.. 라는 의미입니다.<br>
        2만큼의 간격대로 3, 5, 7, 9가 출력된 것을 볼 수 있습니다.

        ```python
        for x in range(3, 10, 2):
            print(x)
        ```

    - 그럼 for ~ range 문을 이용하여 구구단을 작성해 봅시다.

        ```python
        for dan in range(2, 10):
            print(dan, "단")

            for hang in range(2, 10):
                print(dan , "*", hang, "=", dan*hang)
            print()
        ```
    
- **for 문의 확장**

    - For문은 반복적인 처리를 위해서 사용하는 구문이라고 말씀드렸습니다.<br>
        그런데 그런 <span style="color: #C00">반복된 처리 속에서 반복 작업을 중단하고 나가야 한다거나 흐름에 변형을 주어야 할 때</span>에는 어떻게 할까요?<br>
        앞의 예제어서 다루었던 if ~~ 구문을 이용해서 For 문을 빠져나간다거나.. 반복 작업 속에서 상황에 따라 다른 작업을 선택하도록 할 수도 있습니다.<br>
        실제로 많은 경우에 그런 방식으로 For문을 사용하고 있습니다.<br><br>

    - 파이썬에서는 이런 경우에 대하여 <span style="color: #C00">for ~ continue ~ break</span>구문과 <span style="color: #C00">for ~ else</span> 구문을 제공하고 있습니다.<br>
        사실 continue 구문은 직관적으로 눈에 들어오지 않는 편이어서 좀 헷갈리거나.. 이런 것을 왜 쓰지? 라고 하실 수도 있습니다.<br>
        그러나 continue 구문도 필요한 곳이 있으며, 또한 continue 구문은 프로세스가 어떻게 흘러가는지 익히기 위한 연습으로 좋습니다.<br><br>

    - 먼저 for ~ continue ~ break 구문을 살펴보겠습니다.<br>
        <span style="color: #C00">for ~ continue ~ break 구문은 반복문을 수행하던 도중에 어떤 조건을 만나면 반복문을 계속 수행하고, 그 조건을 만족하지 않으면 반복문을 빠져나가도록 하는 구문</span>입니다.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x < 4:
                continue
            else:
                break
        ```

    - 예제 코드에 continue, break를 적용한 예제입니다.
    - 의미는 1, 3, 4, 6, 7 이라는 값을 가진 리스트 area에서 값들을 순서대로 꺼내어 반복문을 실행하라는 의미인데 1, 3, 4, 6, 7 값이 순서대로 x에 할당되고 해당하는 x 값을 출력하는 코드입니다.<br>
        이 때 x를 출력하고 난 다음, 만약 x가 4보다 작은 값이라면 for 구문을 계속 반복하고 x가 4 이상의 값이라면 for 구문을 빠져나가라는 의미입니다.<br>
        리스트 area의 세 번째 값인 4가 x에 할당되었을 때, if 문을 만나지만 if 문 이전에 print(x)를 만나기때문에 4까지 출력이 되고 종료하게 됩니다.<br><br>

    - 조금 변형시켜 볼까요?

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            if x < 4:
                print(x)
                continue
            else:
                break
        ```

    - print(x) 구문을 if 문 안으로 넣어보았습니다.<br>
        이번에는 x = 4 인 경우 바로 for 문을 빠져나갔습니다.<br>
        그냥 continue 같은 것을 사용하지 않고 빠져나가는 조건만 사용해도 되지 않을까? 라고 생각하실 수 있습니다.<br>
        실제로도 continue 구문은 헷갈린다고 빠져나가는 조건만 사용하는 경우도 많이 있습니다.<br><br>

    - 다음 예제를 보시죠.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x >= 4:
                break
        ```

    - if 문에서의 조건만 살짝 바꾸어서 빠져나가는 구문만 적용한 예제입니다. 결과는 처음 본 예제와 같습니다.<br><br>

    - 이렇게 보면 continue 구문은 왜 있는 것일까? 라고 생각하실 수 있지만...뭐.. 그래도 어딘가에 필요하니까 만들어진 구문이겠죠.<br>
        어디에 필요한지 한 번 보도록 하죠.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x < 4:
                continue
            else:
                break
            print("continue")
        ```

    - 위의 예제를 보면... 분명히 print("continue") 구문이 for문 안에 있습니다.<br>
        그렇지만 실제로 출력이 된 적은 없습니다.<br><br>

    - continue 구문은 해당 조건을 만족할 경우 반복문을 계속 실행하라는 의미인데.. 중요한 것은 continue 구문 이후의 명령들은 생략하라~ 라는 의미라는 것입니다.<br>
        그렇기 때문에 print("continue")는 언제나 생략되어 출력되지 않은 것입니다.<br><br>

    - 하나 더 볼까요?

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x < 4:
                continue
            elif x > 6:
                break
            print("continue")
        ```

    - 위의 예제에서는 if 문에서 다루는 조건은 x가 4보다 작은 경우는 continue, x가 6보다 큰 경우는 break를 만납니다.<br>
        그 외의 x에는 if문이 걸려있지 않죠.<br>
        그래서 x가 4보다 작은 경우는 print("continue")가 생략되었고 x가 6보다 커지면서 for문을 벗어나 버렸습니다.<br>
        그 조건 외에는 print("continue")가 실행된 것을 확인할 수 있습니다.<br><br>

    - 이런 차이가 있는거죠.<br>
        그런데 해당 조건을 벗어났다는 것은 위의 조건외의 모든 것이라고 할 수 있기때문에 다음의 예제처럼 구현할 수도 있습니다.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x < 4:
                continue
            elif x > 6:
                break
            else:
                print("continue")
        ```

    - 이처럼 각각의 목적에 따라 다르게 구현할 수 있고, 또 동일한 목적이 있는 구문이라도 프로그래머의 생각에 따라서 다르게 구현될 수 있습니다.<br>
        지난 글에서 "정답은 없다"라고 말씀드렸던 것처럼... 바로 이런 이야기입니다.<br><br>

    - 이번에는 for ~ else 구문을 살펴볼까요?<br>
        for ~ else 구문은 반복문이 break 명령에 의해서 중단되지 않은 경우에만 else 안의 명령을 수행하라는 구문입니다.

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
        else:
            print("finish")
        ```

    - 위의 예제에서는 break 명령이 없습니다.<br>
        따라서 print('Finish')는 잘 실행되었습니다.<br><br>

    - 그럼 중간에 break를 걸어볼까요?

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x > 6:
                break
        else:
            print("finish")
        ```

    - x가 6보다 큰 경우 break를 걸어보았습니다.<br>
        print(x) 명령은 if문 이전에 있기 때문에 break는 걸려있지만 이미 출력은 모두 완료되었죠.<br>
        그러나 어쨋든 break로 for문이 중단된 것이기때문에 else 구문 안의 print('Finish')는 실행되지 않았습니다.<br><br>

    - 그럼 이렇게 바꿔볼까요?

        ```python
        area = [1, 3, 4, 6, 7]

        for x in area:
            print(x)
            if x > 7:
                break
        else:
            print("finish")
        ```

    - x가 7보다 큰 경우에 break를 걸었습니다.<br>
        그러나 리스트 area의 마지막 값인 7이 x > 7을 만족하지 않았기때문에 if문을 만족하기 전에 for문이 끝나버렸습니다.<br>
        break 명령에 걸리지 않은 것입니다. 따라서 "Finish"는 잘 출력되었습니다.<br><br>

    - 여기까지 해서 for ~ continue ~ break 구문과 for ~ else 구문을 살펴보았습니다.

### 4.2 while 문

- while 문은 for 문과 다르게 <span style="color: #C00">반복하기 위한 조건</span>이 주어진다는 점이 특징입니다.<br>
    while 문에 따라오는 <span style="color: #C00">조건을 만족하는 동안 계속해서 반복 작업을 수행하라..</span> 라는 의미입니다.<br><br>

- 예제 코드를 먼저 보겠습니다.

    ```python
    x = 1

    while x < 10:
        x = x + 1
        if x < 5:
            continue
        print(x)

        if x > 7:
            break
    ```

- 예제 코드의 내용은 먼저 x의 초기 값을 1로 잡아 두고 x가 10보다 작은 동안에는 계속 반복 작업을 시키는 것입니다.<br>
    "x가 10보다 작은 동안"이라는 말은 "x < 10" 이라는 조건이 "참(True)"이 되는 경우를 말하죠.<br>
    즉 주어진 x에 대하여 x < 10을 만족한다면 계속 반복 작업을 시키고, x < 10을 만족하지 않는다면, 다시 말해서 x가 10 이상의 값을 가지게 된다면 반복 작업을 중단시켜라.. 라는의미입니다.<br><br>

- 그렇다면 x의 값이 변하지 않으면 while 문은 영원히 반복하게 되겠죠. 그래서 반복 작업을 한 번 수행할 때마다 x의 값을 1씩 증가시킵니다.<br>
    x = x + 1 구문은 x의 값에 1을 더한 값을 다시 x로 할당하여라.. 라는 명령입니다.<br>
    즉 x의 값을 x + 1로 바꿔주라는... 1을 증가시키라는 뜻이죠.<br>
    그러면 반복 작업을 한 번 수행할 때마다 x의 값은 2, 3, 4, ... 와 같이 계속 증가하게 될 것입니다.<br><br>

- 이렇게 x의 값을 증가시키도록 명령하고 나서 조건문이 들어왔습니다.<br>
    만약 x가 5보다 작으면 continue 명령에 따라 뒤에 따라오는 명령은 무시하고 다시 처음부터 작업을 반복합니다.<br>
    x의 처음 값은 1이니까 1씩 증가하다가 x = 5가 되는 순간부터 continue 명령을 벗어나서 print(x)를 실행하게 됩니다.<br>
    이때 print(x) 명령 다음에 다시 x > 7인 경우에 대한 조건문이 들어왔죠.<br>
    x가 7보다 커지면 while 반복문을 빠져나가라는 명령입니다.<br>
    x = 5, 6, 7까지 print문을 수행하고 나서, 여전히 x > 7을 만족하지 않으니까 한 번 더 반복 작업을 실행합니다.<br>
    x = 8이 되었고 첫 번째의 조건문을 만족하지 않으니까, 즉 첫 번째 조건의 결과가 False가 되니까 print(x)를 수행합니다.<br>
    8까지 출력이 되었죠.<br><br>

- 다음으로 두 번째의 조건문을 만났습니다.<br>
    지금 x의 값은 8이니까 x > 7 이라는 조건문을 만족합니다.<br>
    그럼 break 명령에 따라서 반복문을 빠져나가게 됩니다.<br>
    그럼 결과는 5, 6, 7, 8의 네 개의 숫자가 출력이 되겠죠.<br><br>

- 이렇게 while 문을 이용한 반복문을 살펴보았습니다.
- 그런데 <span style="color: #C00">우리가 사용하는 많은 프로그램들은 직접 종료시키지 않으면 무한정 반복하는 프로그램</span>들이 대부분이죠.<br>
    공부할 때 사용하는 예제코드에서는 처음부터 특정 조건을 주고 반복을 시키지만 일반적인 프로그램들은 그냥 계속 동작합니다.<br>
    대표적인 예로 윈도우, 리눅스와 같은 OS, 즉 <span style="color: #C00">운영체제 프로그램</span>을 들 수 있습니다.<br>
    우리가 컴퓨터를 종료시키지 않으면 계속... 끝없이 동작합니다.<br><br>

- 이처럼 강제로 무한루프(끝없이 반복, 즉 루프를 도는 구조)를 지정하려고 하면 어떻게 해야 할까요?<br>
    앞에서 while문은 따라오는 조건을 만족하는 한 계속 반복한다고 말씀드렸습니다.<br>
    그렇다면 그 <span style="color: #C00">조건이 언제나 참(True)이 되면</span> 되겠네요.

    ```python
    x = 1

    while True:
        x = x + 1
        if x < 5:
            continue
        print(x)

        if x > 7:
            break
    ```

- 아래의 예제는 무한루프를 돌게 됩니다. 
    - Colab을 사용 중이시라면 적당한 시점에 셀의 좌측상단에 있는 (▶) 또는 (◼) 버튼을 눌러서 중지시켜주세요.
    - 로컬 시스템의 자체 가상환경을 사용 중이시라면 Ctrl-C를 눌러서 중지시켜주세요.

        ```python
        x = 1

        while 1:
            print(x)
            x = x + 1
        ```

- 예제 코드에서 사용한 것처럼 그냥 <span style="color: #C00">조건문 자리에 True를 넣어버리면 해당 반복문은 break를 만날때까지 무한루프</span>를 돌게 됩니다.<br>
    여기서 True (참) / False (거짓) 라고 하는 값은 boolean 값(논리 값이라고도 합니다)이라고 부르며 어떤 조건에 대하여 참과 거짓을 나타내는 두 개의 값을 가지는 자료형(데이터 타입)입니다.<br>
    참(True)은 1, 거짓(False)은 0으로 할당이 되어 있기 때문에 while문에 따라오는 조건에 True 대신 1을 넣어주어도 동일하게 동작합니다.<br>
    그렇지만 보기 쉽게 True / False를 이용하는 것을 권장하기도 합니다.

    ```python
    x = 1

    while 1:
        x = x + 1
        if x < 5:
            continue
        print(x)

        if x > 7:
            break
    ```

- 여기까지 해서 While문을 이용한 반복문에 대하여 알아보았습니다.<br><br>

- 지금까지 살펴본 데이터 형, 기본적인 연산 및 명령, 조건문, 반복문 정도만 잘 활용해도 어지간한 프로그램은 구현할 수 있습니다.<br>
    Windows 10과 같은 OS를 포함해서... 아무리 큰 프로그램이라도 각 기능별로 하나하나 쪼개나가면 그 밑바닥에는 결국 위의 4가지 정도가 기본이 되어 구현됩니다.<br>
    그 이후에는 더 고차원적인 연산, 기능을 구현하기 위해서 점점 코드를 확장하고 키워나가는것입니다.<br><br>

### 4.3 문제 풀어보기

- 구구단 출력해보기

    ```python
    for dan in range(2, 10):
        print(dan, "단")
        for hang in range(2, 10):
            print(dan , "*", hang, "=", dan*hang)
        print()
    ```

## 5. 문자열 처리

- 문자열을 조작하는 작업은 이후 데이터 분석, AI 학습을 위한 데이터 조작 등에서 많이 활용되므로 다양하게 익혀두는 것이 좋습니다.

- **특정 위치의 문자 얻기**

```python
txt1 = 'A tale that was not right'
txt2 = '이 또한 지나가리라'
```

```python
print(txt1[5])
print(txt2[-2])
```

- **지정한 구간의 문자열 얻기**

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

- **홀수 번째 문자만 추출하기**

```python
txt = 'aAbBcCdDeEfFgG'

result = txt[::2]
print(result)
```

- **문자열 거꾸로 만들기**

```python
txt = 'abcdefg'

result = txt[::-1]
print(result)
```

- **특정 문자가 있는지 확인하기**

```python
msg = '안녕하세요'

if 'a' in msg:
    print('문자열에 a가 포험되어 있음')
else:
    print('문자열에 a가 포함되어 있지 않음')
```

- **숫자인지 알파벳인지 검사하기**

    - 문자열이 숫자만으로 구성되었는지 확인하기

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

    - 문자열이 알파벳만으로 구성되었는지 확인하기

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

    - 문자열이 알파벳과 숫자로 구성되었는지 확인하기

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


## 6. 함수

- **함수(Function) 개요**

    - **함수의 개념**
        - 함수(Function)는 특정 작업을 수행하는 코드 블록을 말합니다. 
        - 프로그래밍에서의 함수는 마치 수학의 함수처럼, 입력을 받아서 정해진 연산을 수행한 후 결과를 반환하는 역할을 합니다.
        - 함수는 프로그래밍에서 코드의 효율적인 관리, 재사용, 구조화 및 유지보수를 위한 핵심적인 도구입니다.

    - **핵심 아이디어**
        - 코드의 묶음: 여러 줄의 코드를 하나의 의미 있는 단위로 묶어 놓은 것
        - 재사용성: 한번 정의된 함수는 필요할 때마다 여러 번 호출하여 동일한 코드를 반복해서 작성할 필요가 없음
        - 모듈화 및 가독성 향상: 복잡한 프로그램을 여러 개의 작은 함수로 나누어 작성하면 코드의 구조가 명확해지고 이해하기 쉬워짐
        - 유지보수 용이성: 특정 기능에 대한 코드가 함수 안에 격리되어 있어, 해당 기능을 수정해야 할 때 함수 내부만 변경하면 됨

    - **함수의 주요 구성 요소**
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

    > - 비유하자면...
    >   - 함수: 요리 레시피
    >   - 매개변수: 레시피에 필요한 재료
    >   - 인수: 실제로 사용하는 재료의 양
    >   - 반환 값: 완성된 요리
    {: .common-quote}

- **함수의 예시 (Python 기준)**

    ```python
    def add(a, b):  # 함수 정의: 이름은 'add', 매개변수는 a, b
    result = a + b
    return result  # 결과 값 반환

    sum_result = add(5, 3)  # 함수 호출: 인수 5와 3을 전달
    print(sum_result)  # 출력: 8
    ```

- **함수 실습 예제**

    - **함수의 선언 방법**

        ```python
        def 함수이름(인자1, 인자2, ...):
            코드들
            return 결과값
        ```

    - **함수 선언과 사용**

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

    - **함수 호출 시 인자의 전달 순서 및 전달 인자 지정방법**

        ```python
        add_text(text2, text1)
        add_text(t2=text2, t1=text1)
        ```

    - **함수 선언 시 인자의 기본값 설정방법**

        - 함수 선언 시 인자의 기본값 설정할 때, 기본값이 설정된 인자는 뒤쪽에 위치해야 합니다.

            ```python
            def add_number(n1=100, n2=200):
                result = n1 + n2
                return result

            result = add_number(30)
            print(result)

            result = add_number(20, 30)
            print(result)

            result = add_number()
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

    - **함수의 결과값의 다중 반환**

        ```python
        def reverse(x, y, z):
            return z, y, x

        a, b, c = reverse(10, 20, 30)
        print(a, b, c)
        ```

        ```python
        result = reverse(10, 20, 30)
        print(result)
        print(type(result))
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

## 7. 클래스

- **클래스(Class) 개요**

    - **클래스의 개념**
        - 클래스(Class)는 객체(Object)를 만들기 위한 설계도 또는 틀입니다. 객체를 선언하기 위한 자료구조로서의 틀이라고 생각할 수 있습니다.
        - 클래스는 객체를 선언하기 위한 틀이므로 그 자체로는 사용할 수 없으며, 인스턴스 객체를 생성하여 사용합니다.
        - 현실 세계의 사물이나 개념을 프로그램 내에서 표현하기 위해 사용되며, 데이터(속성)와 기능(메서드)을 하나의 단위로 묶습니다.<br><br>
        - 클래스는 객체 지향 프로그래밍의 핵심 개념으로, 데이터와 기능을 묶어 현실 세계를 프로그램 내에 효과적으로 모델링하고, 코드의 재사용성과 유지보수성을 높이는 데 중요한 역할을 합니다.

    - **핵심 아이디어**
        - 객체 지향 프로그래밍(OOP)의 핵심: 클래스는 객체 지향 프로그래밍의 중요한 구성 요소입니다.
        - 데이터와 기능의 캡슐화: 관련된 데이터와 해당 데이터를 조작하는 기능을 하나의 클래스 안에 묶어 관리합니다.
        - 코드의 재사용성 및 확장성: 한번 정의된 클래스를 기반으로 여러 개의 객체를 생성할 수 있으며, 상속 등의 기능을 통해 코드를 확장하고 재사용하기 용이합니다.
        - 추상화: 복잡한 시스템을 단순화하여 모델링하고 이해하기 쉽게 만들어줍니다.

    - **클래스의 주요 구성 요소**
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

    > - 비유하자면...
    >   - 클래스: 자동차 설계도
    >   - 속성: 설계도에 명시된 차량의 색상, 모델, 엔진 종류 등
    >   - 메서드: 설계도에 명시된 가속 기능, 제동 기능 등
    >   - 객체(인스턴스): 설계도에 따라 실제로 만들어진 특정 자동차 (예: 빨간색 소나타)
    {: .common-quote}

- **클래스의 예시 (Python 기준)**

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

- **클래스 실습 예제**

    - **클래스의 선언과 사용**

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

    - **생성자와 소멸자**

        - **생성자**

            ```python
            class MyClass2:
                def __init__(self):
                    self.var = "안녕하세요"
                    print("MyClass2 인스턴스 객체가 생성되었습니다.")

            obj = MyClass2()
            print(obj.var)
            ```

        - **소멸자**

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

    - **클래스의 상속**

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


## 8. 모듈, 패키지, 라이브러리

- **모듈, 패키지, 라이브러리의 구분**
    - 파이썬에서는 모듈, 패키지, 라이브러리 등 다양한 표현을 비슷한 의미로 혼용하여 사용하고 있습니다.<br>
    그러나 정확하게는 서로 다른 의미를 가지고 있습니다.<br><br>

    - **모듈(Module)**
        - 정의: <span style="color: #C00">기능을 구현한 파이썬 코드를 담고 있는 하나의 파일(.py 확장자)</span>
        - 내용: 변수, 함수, 클래스 등을 정의하고 포함할 수 있음
        - 목적: 특정 기능을 수행하는 코드들을 논리적으로 그룹화하여 관리하고 재사용하기 위함
        - 예시: math.py, random.py 등과 사용자가 직접 작성한 user_modyle.py 등

    - **패키지(Package)**
        - 정의: <span style="color: #C00">모듈들을 담고 있는 디렉토리 (폴더)</span>
        - 특징: 패키지 디렉토리 안에 반드시 '__ init__.py'라는 (비어 있을 수도 있는) 파일이 있어야 파이썬이 해당 디렉토리를 패키지로 인식함
        - 목적: 관련된 모듈들을 계층적으로 관리하고 이름 충돌을 방지하기 위함(파일 시스템의 폴더 구조와 유사)
        - 예시: 'numpy', 'pandas', 'django.contrib' 등

    - **라이브러리(Library)**
        - 정의: <span style="color: #C00">특정 목적을 위해 함께 제공되는 모듈과 패키지의 집합</span>
        - 특징: 때로는 C나 C++ 등으로 작성된 확장 모듈을 포함하기도 함
        - 목적: 특정 분야의 다양한 기능을 편리하게 사용할 수 있도록 제공함
            - 예를 들면, 데이터 분석, 웹 개발, 머신러닝 등 특정 작업을 수행하는 데 필요한 도구들의 모음
        - 예시: NumPy (수치 계산), Pand~as (데이터 분석), Scikit-learn (머신러닝), Django (웹 프레임워크) 등

    - **요약표**

        <div class="info-table">
            <table>
                <thead>
                    <th>구분</th>
                    <th>정의</th>
                    <th>구성 요소</th>
                    <th>목적</th>
                    <th>예시</th>
                </thead>
                <tbody>
                    <tr>
                        <td class="td-rowheader">모듈</td>
                        <td>하나의 파이썬 파일 (.py)</td>
                        <td>변수, 함수, 클래스 등</td>
                        <td>코드 재사용 및 논리적 그룹화</td>
                        <td>math.py, my_module.py</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader">패키지</td>
                        <td>모듈들을 담고 있는 디렉토리 (__ init__.py 포함)</td>
                        <td>모듈, 하위 패키지</td>
                        <td>관련된 모듈 관리 및 이름 충돌 방지</td>
                        <td>numpy, 'my_package/'</td>
                    </tr>
                    <tr>
                        <td class="td-rowheader">라이브러리</td>
                        <td>특정 목적을 위한 모듈과 패키지의 집합</td>
                        <td>모듈, 패키지, 기타 파일</td>
                        <td>특정 분야의 다양한 기능 제공 및 편리한 사용</td>
                        <td>NumPy, Pandas, Django, TensorFlow</td>
                    </tr>
                </tbody>
            </table>
        </div>

        > - 비유하자면
        >   - 모듈: 레시피의 개별적인 요리법 (예: 김치찌개 레시피)
        >   - 패키지: 여러 레시피를 담은 요리책 (예: 한식 요리책)
        >   - 라이브러리: 다양한 요리책과 조리 도구를 모아놓은 주방 (예: 한식 전문 주방)
        {: .common-quote}

        - 라이브러리는 종종 여러 개의 패키지로 구성될 수 있으며, 각 패키지는 여러 개의 모듈을 포함할 수 있습니다.<br>
        따라서 라이브러리는 가장 큰 범위의 개념이라고 볼 수 있습니다. (<span style="color: #C00">**모듈 ⊂ 패키지 ⊂ 라이브러리**</span>)

- **모듈의 사용 예시 코드**

    ```python
    import time

    print('2초간 프로그램을 정지합니다.')
    print(time.localtime())

    time.sleep(2)

    print(time.localtime())
    print('2초가 지났습니다.')
    ```

    ```python
    import time

    print('2초간 프로그램을 정지합니다.')
    now = time.localtime()
    print('%04d/%02d/%02d %02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, 
                                            now.tm_hour, now.tm_min, now.tm_sec))

    time.sleep(5)

    now = time.localtime()
    print('{}/{}/{} {}:{}:{}'.format(now.tm_year, now.tm_mon, now.tm_mday, 
                                    now.tm_hour, now.tm_min, now.tm_sec))
    print('2초가 지났습니다.')
    ```

    ```python
    print('{5}/{2}/{0} {1}:{3}:{4}'.format(now.tm_year, now.tm_mon, now.tm_mday, 
                                            now.tm_hour, now.tm_min, now.tm_sec))
    ```

- **모듈 임포트(import) 방법**

    - **일반적인 방법**

        ```python
        import time

        print(time.localtime())
        ```

    - **모듈 안의 특정 클래스, 패키지 안의 특정 모듈을 임포트하는 방법**

        ```python
        from time import sleep

        now = time.localtime()
        print('%04d/%02d/%02d %02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday,
                                                now.tm_hour, now.tm_min, now.tm_sec))

        sleep(2)

        now = time.localtime()
        print('{}/{}/{} {}:{}:{}'.format(now.tm_year, now.tm_mon, now.tm_mday,
                                        now.tm_hour, now.tm_min, now.tm_sec))
        ```

    - **코드에서 다른 이름으로 사용하고자 할 때**(이름이 너무 길거나 복잡한 경우 등에 활용)

        ```python
        import time as t

        now = t.localtime()
        print('%04d/%02d/%02d %02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday,
                                                now.tm_hour, now.tm_min, now.tm_sec))

        t.sleep(2)

        now = t.localtime()
        print('{}/{}/{} {}:{}:{}'.format(now.tm_year, now.tm_mon, now.tm_mday,
                                        now.tm_hour, now.tm_min, now.tm_sec))
        ```

## 9. 예외 처리

- **예외 처리(Exception Handling)의 개념**
    - 프로그램 실행 중에 발생할 수 있는 <span style="color: #00C">예상치 못한 오류(예외, Exception)</span>에 대비하고
    - 이러한 오류가 발생했을 때 프로그램이 <span style="color: #00C">비정상적으로 종료되는 것을 방지</span>하며 
    - <span style="color: #00C">우아하게(gracefully) 처리</span>하는 메커니즘입니다.<br><br>
    - 예외 처리는 <span style="color: #C00">프로그래밍에 있어서 필수적인 부분이며, 견고하고 안정적인 프로그램을 만들기 위해 반드시 고려해야 하는 중요한 개념</span>입니다.

- **핵심 아이디어**
    - 예상 가능한 문제 상황 대비 
        - 개발자는 코드를 작성하면서 발생할 가능성이 있는 오류 상황 (예: 파일 없음, 0으로 나누기, 잘못된 입력 등)을 미리 예측하고 대비
    - 오류 발생 시 대처 
        - 프로그램 실행 중 예상된 오류가 실제로 발생하면, 
        - 미리 정의해둔 예외 처리 코드가 실행되어 오류를 적절히 처리
    - 프로그램의 안정성 확보 
        - 예외 처리를 통해 오류 발생에도 프로그램이 멈추지 않고 계속 실행되거나, 
        - 오류 메시지를 출력하고 안전하게 종료할 수 있도록 함

- **예외 처리의 중요성**
    - 프로그램의 안정성 향상
        - 예외 처리 없이 오류가 발생하면 프로그램이 갑자기 종료되어 사용자 경험을 저하시키고 데이터 손실을 초래할 수 있음
    - 유연한 오류 대응
        - 각기 다른 유형의 오류에 대해 맞춤형으로 대처할 수 있도록 함
    - 디버깅 용이성
        - 예외 발생 시 오류 정보 (오류 유형, 발생 위치 등)를 제공하여 디버깅을 도와줌

- **일반적인 예외 처리 구조** (Python 기준)

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


- **예외 처리 실습 예제**

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