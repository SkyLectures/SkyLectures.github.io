---
layout: page
title:  "AI SW 테스트 기법"
date:   2025-07-29 10:00:00 +0900
permalink: /materials/S03-10-01-05_01-AiSwTestingTechniques
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. AI 소프트웨어 테스팅 개요

- **AI 소프트웨어 테스팅의 방향성**
    - AI 소프트웨어 테스팅은 전통적인 소프트웨어 테스팅과는 다른 접근이 필요함
    - AI 시스템은 명시적으로 프로그래밍된 규칙이 아닌 데이터로부터 학습하기 때문에, 테스트 방법론도 이에 맞게 진화해야 함
    - AI 소프트웨어 테스팅의 주요 목표는 모델의 정확성, 견고성, 공정성, 안전성 등을 확보하는 것

- **AI 시스템 테스팅에서 고려해야 할 요소**
    - 테스트 오라클 문제(정확한 기대 출력값을 알기 어려운 상황)
    - 데이터 품질과 다양성
    - 모델의 내부 구조 이해와 검증
    - 실제 환경에서의 성능과 안정성

## 2. 주요 AI 테스트 기법

### 2.1 메타모픽 테스트 기법(Metamorphic Testing)

- 테스트 오라클 문제를 해결하기 위한 기법
- 입력과 출력 간의 관계인 '메타모픽 관계(Metamorphic Relations, MRs)'를 활용하여 소프트웨어의 신뢰성을 검증

- **핵심 원리**:
    1. 원본 입력과 출력이 있을 때, 입력에 특정 변환을 적용하여 새로운 입력을 만듦
    2. 새로운 입력에 대한 출력이 원본 출력과 어떤 관계를 가져야 하는지 정의
    3. 이 관계가 성립하는지 확인하여 소프트웨어의 올바른 동작 여부 판단

- **예시**
    - 이미지 분류 AI에서 이미지를 회전시켜도 같은 객체로 인식해야 한다는 규칙을 검증하는 방식

### 2.2 뉴런 커버리지 테스트 기법(Neuron Coverage Testing)

- 신경망 모델의 내부 뉴런들이 테스트 실행 중에 얼마나 활성화되었는지 측정하는 화이트박스 테스팅 기법

- **핵심 원리**:
    - 뉴런 커버리지: 활성화된 뉴런의 수를 총 뉴런 수로 나눈 값으로 측정
        - 소프트웨어 테스팅의 코드 커버리지 개념을 신경망에 적용한 것
    - 높은 뉴런 커버리지는 신경망의 더 많은 부분이 테스트되었음을 의미함

**계산 방법**:
```
뉴런 커버리지(%) = (활성화된 뉴런의 수 / 총 뉴런 수) × 100
```

### 2.3 최대 안전 반경 테스트 기법(Maximum Safe Radius Test)

- 딥 뉴럴 네트워크(DNN)의 견고성을 검증하기 위한 테스트 기법
- 입력 데이터에 작은 변화를 주었을 때 모델의 예측이 변경되지 않는 최대 반경(거리)을 계산

- **핵심 원리**:
    1. 원본 입력 데이터 주변에 존재하는 안전한 영역 정의
        - 이 영역 내에서는 어떤 입력 변화가 있더라도 AI 모델의 출력이 동일하게 유지됨
    2. 최소 안전 반경(MinUR)과 최대 안전 반경(MaxRR)을 함께 계산하여 모델의 안정성을 평가

- **활용**
    - 적대적 예제(adversarial examples)에 대한 모델의 취약성 평가
    - 안전이 중요한 미션 크리티컬(Mission Critical) 시스템에서 AI의 신뢰성을 보장하는 데 중요함

### 2.4 커버리지 검증 기법(Coverage Verification)

- 테스트가 AI 모델의 입력 공간, 내부 상태, 의사 결정 경로 등을 얼마나 광범위하게 다루고 있는지를 측정하고 평가하는 방법

- **주요 커버리지 유형**:
    - **결정 경계 커버리지(Decision Boundary Coverage)**
        - AI 모델의 의사 결정 경계가 얼마나 테스트되었는지 측정
    - **입력 공간 커버리지(Input Space Coverage)**
        - 모델이 처리해야 할 전체 입력 데이터 공간 중 얼마나 많은 부분이 테스트되었는지 측정
    - **테스트 케이스 다양성(Test Case Diversity)**
        - 생성된 테스트 케이스들이 얼마나 다양하고 독립적인 입력 변화를 포함하는지 측정

- **활용**
    - 모델의 일반화 능력을 평가 후, 테스트가 충분히 이루어지지 않은 영역을 식별하여 추가 테스트를 진행할 수 있음

## 3. AI 테스트 기법 비교

<div class="info-table">
    <table>
        <thead>
            <th>특징/기법</th>
            <th>메타모픽 테스트<br>(Metamorphic Testing)</th>
            <th>뉴런 커버리지 테스트<br>(Neuron Coverage Testing)</th>
            <th>최대 안전 반경 테스트<br>(Maximum Safe Radius Test)</th>
            <th>커버리지 검증 기법<br>(Coverage Verification)</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">목표</td>
                <td class="td-left">테스트 오라클 문제 해결 및<br>신뢰성 검증</td>
                <td class="td-left">신경망 내부 동작(활성화) 검증 및<br>테스트 철저도 측정</td>
                <td class="td-left">모델의 견고성(robustness) 및<br>안정성 검증</td>
                <td class="td-left">테스트 케이스의 충분성 및<br>AI 모델의 포괄적 검증</td>
            </tr>
            <tr>
                <td class="td-rowheader">접근 방식</td>
                <td class="td-left">블랙박스<br>(입력-출력 관계 기반)</td>
                <td class="td-left">화이트박스<br>(모델 내부 뉴런 활성화 기반)</td>
                <td class="td-left">블랙박스/그레이박스<br>(입력 변화에 따른 출력 안정성 기반)</td>
                <td class="td-left">화이트박스/블랙박스/그레이박스<br>(다양한 기준의 포괄 범위 측정)</td>
            </tr>
            <tr>
                <td class="td-rowheader">검증 대상</td>
                <td class="td-left">입력과 출력 간의 관계<br>(예: 이미지가 회전해도 분류 결과는<br>동일)</td>
                <td class="td-left">신경망 내부 뉴런의 활성화 여부</td>
                <td class="td-left">입력 데이터의 작은 변화에 대한<br>모델 출력의 일관성</td>
                <td class="td-left">모델의 입력 공간, 내부 상태,<br>의사 결정 경로 등</td>
            </tr>
            <tr>
                <td class="td-rowheader">주요 활용</td>
                <td class="td-left">- 테스트 오라클이 없는 경우<br>- AI 모델의 일반화 능력 검증</td>
                <td class="td-left">- 학습된 모델의 훈련 범위 평가<br>- 잠재적 취약 영역 식별</td>
                <td class="td-left">- 적대적 예제(adversarial examples)<br>&nbsp;&nbsp;방어력 평가<br>- 안전이 중요한 시스템의 신뢰성 검증</td>
                <td class="td-left">- 테스트 케이스 생성 최적화<br>- 테스트 미흡 영역 발견 및 보강</td>
            </tr>
            <tr>
                <td class="td-rowheader">장점</td>
                <td class="td-left">- 테스트 오라클 문제 해결에 효과적<br>- 복잡한 AI 시스템에 적용 용이</td>
                <td class="td-left">- 모델의 '내부' 동작 가시화<br>- 테스트 케이스 부족 영역 식별</td>
                <td class="td-left">- 모델의 견고성을 정량적으로 측정<br>- 실제 위협에 대한 모델 저항력 평가</td>
                <td class="td-left">- 테스트의 체계적 관리<br>- 잠재적 결함 감소 및 품질 향상에 기여</td>
            </tr>
            <tr>
                <td class="td-rowheader">단점/한계</td>
                <td class="td-left">- 메타모픽 관계 정의의 어려움<br>- 일부 모델에는 적용이 어려움</td>
                <td class="td-left">- 뉴런 커버리지가 곧 성능/품질을<br>&nbsp;&nbsp;의미하지는 않음<br>- 복잡한 모델에 적용 시 계산 비용</td>
                <td class="td-left">- 계산 복잡성이 높을 수 있음<br>- '최대 안전 반경' 정의가 어려울 수 있음</td>
                <td class="td-left">- 어떤 커버리지를 목표로 할지 선정의<br>&nbsp;&nbsp;어려움<br>- 과도한 커버리지 목표는 비효율적일<br>&nbsp;&nbsp;수 있음</td>
            </tr>
        </tbody>
    </table>
</div>

> - 각 기법은 AI 모델의 특정 측면을 평가하고 검증하는 데 강점을 가지므로
> - 개발하는 AI 시스템의 특성과 목표에 따라 적절한 기법들을 조합하여 활용하는 것이 중요함
{: .summary-quote}
