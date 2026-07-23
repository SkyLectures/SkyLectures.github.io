---
layout: page
title:  "비전 AI 기반 검사 시스템 구조"
date:   2026-07-22 15:00:00 +0900
permalink: /materials/S06-04-04-01_01-VisionAiBasedInspectionSystem
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - 비전 AI 검사는 카메라만 좋은 것을 달아놓는다고 끝나지 않음
> - 조명, 렌즈, 전처리(Edge PC), AI 모델, 그리고 PLC 제어 신호까지 0.1초 안에 톱니바퀴처럼 돌아가는 **하드웨어-소프트웨어 융합 시스템**
{: .common-quote}

- **비전 AI 기반 검사시스템(Vision AI Inspection System)**
    - 사람의 눈 역할을 하는 광학 장비(카메라, 렌즈, 조명)와 인공지능의 뇌 역할을 하는 딥러닝/머신러닝 알고리즘을 결합하여
    - 제조 현장에서 제품의 외관 불량, 치수 오류, 부품 누락 등을 자동으로 판정하고 분류하는 지능형 품질 검사 시스템<br><br>
    - 과거에는 베테랑 작업자가 눈으로 하나하나 들여다보며 불량을 찾아내던 '숙련자의 시각적 직관'을
    - 컴퓨터와 AI 소프트웨어로 그대로 구현한 시스템

## 1. 기존 Rule 기반 비전 vs AI 비전 검사

- **Rule 기반 비전 (Rule-based Machine Vision)**
    - **작동 원리:**
        - 엔지니어가 수학적/기하학적 규칙(Rule)을 직접 코딩함
            - 예: 픽셀 밝기 값이 180 이상이고, 가로 5px × 세로 10px 이상의 검은색 덩어리가 있으면 NG(불량)로 판정하라

    - **장점:**
        - 치수 측정(Width, Height, Gap), 명확한 정형 불량, 단순 존재 유무(Presence/Absence) 판단 시 
        - 처리 속도가 어마어마하게 빠르고 절대적인 정확도를 가짐

    - **치명적 한계:**
        - **환경 변화에 취약:**
            - 공장 창문으로 들어오는 햇빛 위치가 바뀌거나, 조명 밝기가 5%만 변해도 픽셀 수치가 달라짐<br>
                🡪 **과검(False Positive, 정상인데 불량 처리)** 폭발

        - **비정형 불량 대응 불가:**
            - 스크래치, 눌림, 얼룩, 오염처럼 "형태와 크기가 매번 다르게 발생하는 불량"은 규칙으로 정의할 수 없음


- **AI 비전 기반 검사 (AI-based Vision Inspection)**
    - **작동 원리:**
        - 인간의 시각 뇌 영역(시각 피질)을 모방한 딥러닝(CNN, Vision Transformer) 구조
        - 규칙을 코딩하는 것이 아니라 **수천 장의 양품/불량 이미지를 보여주고 패턴을 스스로 학습**하게 함

    - **장점:**
        - **정성적/문맥적 판단:**
            - "이 표면은 금속 결 특성상 원래 무늬가 불규칙하지만, 이 부분의 미세한 곡선은 긁힘(스크래치)이다"라는
            - **인간 베테랑 검사원의 직관적 판단력**을 재현
        - **노이즈 내성:**
            - 약간의 조명 변동이나 광택 표면의 미세한 반사 차이가 있어도
            - 불량의 핵심 특징(Feature)을 유지하여 안정적으로 판단

- **비교 요약 표:**

    <div class="info-table">
    <table>
        <thead>
            <th style="width: 150px;">구분</th>
            <th style="width: 400px;">기존 Rule 기반 비전</th>
            <th style="width: 400px;">AI 비전 기반 검사</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">판단 주체</td>
                <td>엔지니어가 작성한 <b>수학적 조건문(IF-THEN)</b></td>
                <td>데이터를 통해 스스로 학습한 <b>AI 신경망 모델</b></td>
            </tr>
            <tr>
                <td class="td-rowheader">주요 대상</td>
                <td><b>정형 불량</b> 및 <b>치수/각도 측정</b></td>
                <td><b>비정형 불량</b> (스크래치, 찍힘, 얼룩, 이물)</td>
            </tr>
            <tr>
                <td class="td-rowheader">환경 변화</td>
                <td>조명/외광 변화 시 과검률 급증</td>
                <td>광학적 노이즈에 대한 강인함(Robustness)</td>
            </tr>
            <tr>
                <td class="td-rowheader">유지 보수</td>
                <td>제품 변경 시 매번 알고리즘 재설정 필요</td>
                <td>신규 불량 이미지 추가 후 <b>재학습(Re-training)</b></td>
            </tr>
        </tbody>
    </table>
    </div>


## 2. 비전 AI 시스템 4대 구성 요소

- 실제 공장에 설치되는 비전 AI 검사기 장비 내부의 데이터 흐름 순서에 맞추어 4개 계층으로 나눔

```text
[ 1. 광학계 (Hardware) ] ➔ [ 2. 수집/전처리 (Edge) ] ➔ [ 3. AI 추론 엔진 ] ➔ [ 4. OT/PLC 인터페이스 ]
   카메라, 렌즈, 조명           Frame Grabber, Crop          ONNX / TensorRT           OK/NG 배출 로봇/신호
```


### 2.1 광학계

> - Garbage In, Garbage Out
> - 카메라 렌즈와 조명이 엉망이라 불량이 눈에 보이지 않게 촬영되면, 세계 최고의 AI 모델을 가져와도 불량을 못 잡음
{: .common-quote}

- **카메라 종류의 선택:**
    - **Area Scan Camera (면적 스캔):**
        - 일반 카메라처럼 한 번에 2D 사각형 이미지를 촬상
        - 컨베이어 벨트상에서 잠시 멈추거나 연속적으로 이동하는 일반적인 제품 검사에 사용

    - **Line Scan Camera (선 스캔):**
        - 1픽셀 세로선 단위로 초당 수만 번 연속으로 찍어 긴 이미지를 합성
        - **디스플레이 글래스, 필름, 금속 코일 연속 공정**처럼 고속으로 지나가는 거대한 웹(Web) 제품 검사에 필수적

- **조명(Lighting)의 결정적 역할 (이미지 품질의 80% 결정):**
    - 제품 재질과 불량 특성에 따라 빛의 반사 방식을 컨트롤해야 함
        - **링 조명 (Ring Light):** 가장 일반적, 중앙 부위 강조
        - **동축 조명 (Coaxial Light):** 반사가 심한 금속, 유리 표면의 미세 스크래치를 선명하게 띄움
        - **돔 조명 (Dome Light):** 곡면이 있거나 광택이 나는 제품에 그림자 없는 균일한 빛을 비춤


### 2.2 수집 및 전처리 계층

> - 고해상도 카메라로 찍은 4K, 8K 원본 이미지를 그대로 AI 모델에 넣으면 🡪 모델이 너무 무거워져 라인 속도를 못 맞춤
{: .common-quote}

- **Frame Grabber (프레임 그래버):**
    - 초당 수십~수백 프레임의 고용량 비전 데이터를
    - 유실 없이 초고속으로 메모리에 올려주는 산업용 인터페이스 카드 (GigE, CoaXPress 등)

- **이미지 전처리 (Preprocessing Pipeline):**
    - **ROI(Region of Interest) Crop:**
        - 4K 원본 이미지 전체를 분석하는 대신,
         실제 검사 대상 부품이 위치한 좌표 영역만 잘라냄

    - **Resize & Normalization:**
        - AI 모델 입력을 위해 512x512 또는 224x224 픽셀 크기로 줄이고,
        - 픽셀 데이터 값의 범위를 $$0 \sim 1$$ 사이로 정규화


### 2.3 AI 추론 엔진

> - 학습(Training)은 클라우드나 거대한 GPU 서버에서 하지만,
> - **추론(Inference)은 공장 현장의 Edge PC에서 0.05초 만에 완료**되어야 함
{: .common-quote}

- **Edge Computing 인프라:**
    - 현장 설비 옆에 붙는 산업용 PC(IPC) 또는 임베디드 AI 모듈(NVIDIA Jetson, Industrial PC with RTX GPU)

- **모델 경량화 및 추론 최적화 (ONNX / TensorRT):**
    - Python/PyTorch 환경에서 학습시킨 무게감 있는 AI 모델을 C++ 기반의 고속 추론 포맷(ONNX, NVIDIA TensorRT)으로 변환
    - **양자화(Quantization):**
        - 32비트 부동소수점(FP32) 연산을 8비트 정수(INT8) 연산으로 다이어트시켜,
        - 정확도 손실은 0.5% 미만으로 유지하면서 추론 속도를 $$3 \sim 5\text{배}$$ 향상시킴


### 2.4 OT 인터페이스 (Operational Technology Interface Layer)

> - AI가 'NG입니다'라고 모니터에 표시만 하고 끝나면 자동화가 아님
> - 물리적 액추에이터를 움직여 불량품을 라인 밖으로 튕겨 내야 검사가 완결됨
{: .common-quote}

- **PLC 통신 및 신호 전달:**
    - AI 추론 결과(OK: 0, NG: 1)를 **Digital I/O 카드, Ethernet/IP, PROFINET, Modbus** 프로토콜을 통해 설비의 메인 제어기(PLC)로 전송

- **실시간 제어 및 인터락 (Interlock):**
    - **Tact Time(공정 주기) 준수:**
        - 카메라 찍고 🡪 AI 판단 🡪 PLC 신호 전달까지 **전체 프로세스가 보통 $$0.1 \sim 0.3\text{초}$$ 이내**에 끝나야 라인이 밀리지 않음

    - **Physical Actuator 연동:**
        - NG 신호를 받은 PLC가 즉시 **에어 블로워(Air Blower)를 분사하여 불량품을 밖으로 쳐내거나, 로봇 암(Robot Arm)에 피킹 명령**을 내림

<br>

> - **'비전 AI 시스템 구조'**의 핵심은 **밸런스**
>   - 조명이 부실해서 불량이 묻히면 아무리 뛰어난 AI 모델도 답을 낼 수 없고,
>   - AI 모델이 아무리 정확해도 추론 속도가 느려 PLC에 제때 판정 신호를 못 보내면 컨베이어 벨트에서 불량품을 놓침
>   - 성공적인 비전 AI 프로젝트를 위해서는
>   - **광학(조명/카메라) 🡪 전처리 🡪 AI 경량화 🡪 PLC 제어까지 연결되는 전체 파이프라인의 하드웨어-소프트웨어 융합 구조**를 반드시 이해하고 설계해야 함
{: .expert-quote}