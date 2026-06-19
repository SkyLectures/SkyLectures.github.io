---
layout: page
title:  "Spark 개요와 분산 데이터 처리"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-05-01-01_01-SparkOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. 분산 데이터 처리의 이해

> - **분산 데이터 처리**
>   - 하나의 거대한 데이터를 단일 컴퓨터(서버)가 아닌,
>   - **네트워크로 연결된 여러 대의 컴퓨터(클러스터, Cluster)에 분산하여 저장하고 병렬로 연산하는 방식**
{: .common-quote}

### 1.1 등장 배경: Scale-up vs Scale-out

- 과거의 경우
    - 데이터량이 늘어나면 서버의 CPU, RAM, 디스크를 더 좋은 것으로 교체하는 **Scale-up(수직 확장)** 방식 사용
    - 한계점
        - **하드웨어적 한계:**
            - 고성능 부품일수록 가격이 기하급수적으로 상승함
            - 단일 보드에 장착할 수 있는 부품의 한계가 존재함
        - **단일 장애점(SPOF):**
            - 고가의 서버 한 대가 고장 나면 전체 시스템이 마비됨

- **해결 방안**
    - 값싼 범용 컴퓨터(Commodity Hardware) 여러 대를 묶어 하나의 컴퓨터처럼 사용하는 **Scale-out(수평 확장)** 체계 도입
    - 이것이 분산 데이터 처리의 시초가 됨

### 1.2 분산 데이터 처리의 개념

<div class="insert-image">
    <img src="/materials/devtools/images/S13-05-01-01_01-001_DistributedDataProcessing.png" style="width: 90%;">
</div>

- **1단계: 데이터 분할 및 분산 저장 (Data Partitioning & Distributed Storage)**
    - 거대한 하나의 파일을 시스템이 처리할 수 있도록 작게 쪼개어 여러 컴퓨터에 나누어 저장하는 과정
    - 원본 데이터를 파티션 단위로 나누고, 분산 스토리지 클러스터 내의 여러 노드에 분산하여 저장함으로써 저장 공간의 한계를 극복하고 데이터 로딩 속도를 병렬화 함<br><br>

    - **대용량 소스 데이터 (Big Data Source):**
        - 시스템으로 유입되는 원본 데이터
        - 단일 서버의 하드웨어(디스크, RAM)로는 처리할 수 없을 만큼 크기가 큼
        - 그림에서는 거대한 데이터베이스 아이콘으로 표현되며, 여러 가닥의 데이터 스트림으로 분리되기 시작함

    - **데이터 파티션 (Data Partitions):**
        - 소스 데이터를 관리하고 연산할 수 있는 기본 단위(블록)로 나눈 것
        - 각 컴퓨터 노드(서버)에는 전체 데이터의 일부인 이 파티션들이 저장됨
        - 파티션의 크기를 어떻게 잡느냐(예: 128MB)가 전체 성능에 큰 영향을 미침

    - **분산 스토리지 클러스터 (Distributed Storage Cluster):**
        - 여러 대의 범용 컴퓨터(서버)가 네트워크로 연결되어 하나의 거대한 저장 공간처럼 동작하는 아키텍처
        - [예: HDFS, MinIO, S3] 같은 오픈 소스 또는 클라우드 스토리지 솔루션이 이 역할을 수행함
        - 그림에서는 6대의 서버 노드가 하나의 점선 뭉치로 묶여 하나의 클러스터를 구성하고 있음

<br>

- **2단계: 병렬 연산 및 분산 처리 (Parallel Computation & Distributed Processing)**
    - 저장된 데이터를 기반으로 실제로 연산을 수행하는 핵심 단계
    - 스파크(Spark)와 같은 분산 처리 엔진이 주도함
    - 클러스터 매니저의 지휘 아래, 각 처리 노드는 자신이 가진 데이터를 우선 처리하며,
    - 필요한 경우 셔플 과정을 통해 네트워크로 데이터를 교환하여 복잡한 연산을 수행함<br><br>

    - **클러스터 매니저 (Cluster Manager):**
        - 분산 환경의 '지휘자'
        - 각 노드의 자원(CPU, RAM) 상태를 파악하고, 작업을 어느 노드에 배분할지 결정
        - [예: Spark Driver, YARN, Kubernetes]가 이 역할을 수행함
        - 그림에서 오른쪽에 독립적으로 존재하며, 모든 처리 노드와 제어 신호를 주고받는 모습으로 표현됨

    - **분산 처리 노드 (Worker Nodes):**
        - 실제로 연산을 수행하는 '일꾼'
        - [예: Spark Executor]가 각 노드에서 실행되며,
        - 로컬 파티션에 대한 계산을 수행함

    - **로컬 파티션 처리 (Local Partition Processing):**
        - 데이터 지역성(Data Locality) 원리에 따라, 각 노드는 **자신이 저장하고 있는 데이터 파티션**을 우선적으로 처리함
        - 그림에서는 각 작업 노드 내의 'Process' 톱니바퀴 아이콘이 로컬 데이터 블록을 처리하는 모습을 표현

    - **네트워크 데이터 이동 (Data Shuffle / Exchange):**
        - **분산 처리에서 가장 중요한 개념이자 병목 구간**
        - 서로 다른 파티션에 흩어져 있는 데이터를 특정 기준(예: `GroupBy`의 키 값)으로 모아야 할 때, 네트워크를 통해 노드 간 데이터를 주고받음
        - 그림에서는 중앙의 3대 서버가 수많은 교차 화살표를 주고받는 모습 이 '셔플' 과정을 명확히 보여줌
        - 셔플을 최소화하는 것이 성능 튜닝의 핵심

<br>

- **3단계: 결과 집계 및 저장 (Result Aggregation & Storage)**
    - 분산되어 처리된 개별 연산 결과를 하나로 합치고, 최종 결과를 영구 저장하는 단계
    - 2단계의 병렬 연산이 끝나면,
        - 각 노드의 부분 결과를 최종적으로 집계하고,
        - 이 결과를 다시 파일 시스템에 저장하거나 사용자에게 반환하여 전체 처리를 마무리함

    - **결과 집계:**
        - 모든 처리 노드의 로컬 연산이 완료되면, 최종적으로 결과를 취합
            - 예: Count, Sum, 최상위 10개 결과 등
        - 그림에서 각 처리 노드에서 나온 결과 화살표가 하나의 문서 아이콘으로 모이는 모습으로 표현

    - **최종 집계 결과:**
        - 사용자가 요청한 최종 출력 데이터
        - 그림에서는 차트와 문서를 포함한 하나의 '완성된 보고서' 아이콘으로 표현

<br>

- **전체 흐름의 핵심**
    - 이 다이어그램은 분산 처리의 3가지 핵심 성공 요소를 보여줌
        - **확장성 (Scalability):**
            - 1단계의 분산 저장과 2단계의 병렬 처리는
            - 노드(컴퓨터) 수를 늘리면 성능도 그에 비례해 늘어나는 구조를 가짐
        - **데이터 지역성 (Data Locality):**
            - 2단계에서 각 노드가 자기 데이터를 처리하는 구조는 네트워크 사용을 최소화하여 성능을 극대화하는 핵심 원리
        - **결함 허용 (Fault Tolerance):**
            - 1단계에서 복제본을 저장하고 2단계에서 리니지(Lineage) 정보를 활용
            - 노드가 고장 나더라도 데이터를 복구할 수 있는 기반이 됨
            - (그림에는 명시되지 않음) 


### 1.2 분산 시스템의 핵심 배경 이론

- 분산 환경은 네트워크 단절, 노드 고장 등 수많은 변수가 존재함 🡲 이를 제어하기 위한 이론적 기반이 필수

- **CAP 정리 (Brewer's CAP Theorem)**
    - 분산 데이터 시스템은 다음 세 가지 속성 중 **최대 두 가지만** 만족할 수 있다는 이론
        - **일관성 (Consistency):**
            - 어떤 노드에 접근하든 모든 사용자는 동시에 같은 데이터를 보아야 함
        - **가용성 (Availability):**
            - 일부 노드가 다운되더라도 성공적으로 응답을 받아야 함
        - **분할 관용성 (Partition Tolerance):**
            - 노드 간 네트워크가 단절(Partition)되어도 시스템이 정상 동작해야 함

    - **실무적 선택:**
        - 분산 환경에서는 네트워크 장애가 필연적 🡲 **P(분할 관용성)를 기본으로 선택**
        - 비즈니스 성격에 따라 **CP**(금융권 등 정확성 중시)와 **AP**(소셜 미디어 등 지속 가능성 중시) 중 하나를 선택

- **PACELC 정리**
    - CAP 정리가 '네트워크 장애 상황'에만 초점을 맞춘 것을 보완
    - **정상 상황(Else)에서의 지연 시간(Latency)과 일관성(Consistency)의 상충 관계**까지 설명하는 이론

- **데이터 일관성 모델**
    - **강한 일관성 (Strong Consistency):**
        - 데이터 업데이트 후 모든 읽기 작업은 최신 값을 보장받음
    - **최종 일관성 (Eventual Consistency):**
        - 일시적으로 데이터가 다를 수 있으나,
        - 시간이 지나면 결국 모든 노드가 동일한 값으로 동기화됨 (예: NoSQL, Cassandra)


### 1.3 분산 데이터 처리의 기반 기술

- 분산 처리를 실현하기 위해서는 다음의 두 체계가 맞물려야 함
    - **어떻게 나누어 저장할 것인가(분산 저장)**
    - **어떻게 나누어 계산할 것인가(분산 컴퓨팅)** 

- **분산 저장 기술 (Storage Layer)**
    - **HDFS (Hadoop Distributed File System):**
        - 대용량 파일을 블록(Block) 단위로 쪼개어 여러 서버에 분산 저장
        - 복제본(Replica)을 생성하여 데이터 유실을 방지하는 전통적인 파일 시스템

    - **오브젝트 스토리지 (AWS S3, MinIO):**
        - 클라우드 네이티브 환경에서 각광받는 형태
        - 디렉터리 구조 없이 고유 키를 통해 비정형 데이터를 유연하게 분산 저장

- **분산 컴퓨팅 기술 (Compute Layer)**
    - **MapReduce (1세대):**
        - 데이터를 나누어 연산하는 `Map` 단계와 연산 결과를 합치는 `Reduce` 단계로 구분
        - 매 단계 결과를 디스크에 쓰고 읽기 때문에 디스크 I/O 병목이 발생함

    * **Apache Spark (2세대 - 인메모리):**
        - MapReduce의 디스크 병목을 해결하기 위해,
        - 연산 중간 결과를 메모리(RAM)에 유지하며 처리하는 방식을 도입
        - 연산 속도를 혁신적으로 끌어올림

- **클러스터 리소스 관리 (Cluster Manager)**
    - 여러 컴퓨터의 자원(CPU, Memory)을 효율적으로 배분하는 중재자 역할
        - **YARN:**
            - 하둡 생태계의 기본 리소스 관리자

        - **Kubernetes / Docker Compose:**
            - 최근 컨테이너 기반 환경에서 많이 활용되는 자원 격리 및 관리 기술



### 1.4 분산 처리 과정의 핵심 메커니즘과 난제

- 분산 처리가 실제로 수행될 때 성능과 직결되는 핵심 개념

- **파티셔닝 (Partitioning) & 샤딩 (Sharding)**
    - 대용량 데이터를 다룰 수 있는 작은 단위(파티션)로 나누어 저장하고 처리하는 기법
    - 데이터가 특정 노드에만 몰리지 않도록(Data Skew 현상 방지) 균등하게 분배하는 전략이 중요함

- **셔플링 (Shuffling)**
    - 분산되어 있는 데이터들을 특정 기준(예: Group By, Join Key)에 따라 재정렬하기 위해
    - **네트워크를 통해 노드 간 데이터를 주고받는 과정**
    - **문제점:**
        - 분산 처리에서 가장 많은 네트워크 비용과 지연(Latency)을 유발하는 병목 구간
        - 셔플링을 최소화하는 것이 성능 최적화의 핵심

- **결함 허용 (Fault Tolerance)**
    - **Lineage (Spark RDD):**
        - 데이터를 유실했을 때,
            - 처음부터 데이터를 다시 만드는 것이 아니라
            - 데이터가 생성된 히스토리(계보)를 기억하여
            - 유실된 파티션만 똑같이 복구해내는 기술

    - **Replication (하둡/NoSQL):**
        - 애초에 동일한 데이터를 3 군데 이상 복제하여 저장
        - 하나의 노드가 죽어도 즉시 다른 노드가 대체하도록 만듦

<br>

> - 분산 데이터 처리는 결국 **하드웨어의 한계를 소프트웨어 아키텍처로 극복**한 기술적 결실
> - 오늘날의 데이터 엔지니어는 단순히 데이터를 가공하는 코드를 짜는 것을 넘어,
>   - **"내가 작성한 코드가 네트워크 셔플을 얼마나 유발하는지"**
>   - **"데이터가 한쪽 노드로 치우치지 않고 파티셔닝이 잘 되었는지"**<br>
> 와 같은 분산 시스템의 생리를 깊이 이해하고 통제할 수 있어야 비로소 고성능의 파이프라인을 구축할 수 있음
{: .summary-quote}


<br><br>


## 2. Apache Spark 개요

> - **Apache Spark**
>   - 대규모 데이터 처리를 위한 오픈 소스 **분산 컴퓨팅 프레임워크**
>       - 대용량 데이터(Big Data)를 한 대의 컴퓨터가 아닌 수십, 수백 대의 클러스터 환경에서 빠르고 안전하게 병렬 처리할 수 있도록 지원하는 현대 데이터 엔지니어링의 표준 기술
>   - 기존의 빅데이터 처리 표준이었던 Hadoop MapReduce의 한계를 극복하기 위해 탄생
{: .common-quote}


### 2.1 개발 역사

- **초기 개발 및 오픈소스 전환 (2009년 ~ 2013년)**
    - **2009년: UC 버클리 AMPLab에서 탄생**
        - 마테이 자하리아(Matei Zaharia)를 비롯한 연구원들이 **Hadoop MapReduce의 느린 속도와 복잡성을 해결하기 위해** 연구 프로젝트로 시작
        - 기존 하둡이 매 단계마다 디스크 I/O를 유발하는 점을 극복하고자, 메모리 내에서 데이터를 처리하는 **RDD(Resilient Distributed Dataset)** 개념을 최초로 고안

    - **2010년: 오픈소스 공개**
        - BSD 라이선스로 소스코드가 처음 공개됨

    - **2013년: Apache 인큐베이터 진입 및 Databricks 설립**
        - 프로젝트의 규모가 커지면서 아파치 소프트웨어 재단(ASF)의 인큐베이터 프로젝트로 채택됨
        - 같은 해, Spark의 원천 개발자들이 모여 비즈니스화를 위한 기업 **Databricks**를 설립


- **Apache 탑레벨 프로젝트 승격 및 폭발적 성장 (2014년 ~ 2016년)**
    - **2014년: 아파치 탑레벨 프로젝트(Top-Level Project) 승격 및 1.0 버전 출시**
        - 인큐베이터 진입 후 단 8개월 만에 아파치의 최상위 프로젝트로 승격되며 빅데이터 진영의 전폭적인 지지를 받기 시작
        - 전형적인 대규모 데이터 가공 분석(SQL 등)을 지원하기 위한 **Spark SQL**과 **DataFrame API** 도입

    - **2015년: 하둡 MapReduce의 대체재로 급부상**
        - 전 세계 수많은 대기업(Yahoo, Tencent 등)이 기존 하둡 기반 인프라를 Spark로 전환하기 시작
        - 머신러닝 전용 라이브러리인 **MLlib**와 그래프 연산 엔진 **GraphX**가 발전하며 단순 처리를 넘어선 종합 분석 플랫폼으로 진화


- **Spark 2.0 시대와 구조적 API의 정착 (2016년 ~ 2019년)**
    - **2016년: Spark 2.0 출시 (성능 및 표준화)**
        - **Dataset API**가 정식 도입되면서 RDD 중심의 저수준 제어에서 DataFrame/Dataset 중심의 고수준 구조적(Structured) API로 패러다임 전환
        - Tungsten 엔진과 Catalyst 옵티마이저 등 내부 엔진 최적화를 통해 연산 속도가 비약적으로 상승
        - 실시간 스트리밍 처리를 SQL 엔진 위에서 직관적으로 다룰 수 있게 한 **Structured Streaming** 도입


- **Spark 3.0 시대와 클라우드 네이티브로의 진화 (2020년 ~ 현재)**
    - **2020년: Spark 3.0 출시 (적응형 쿼리 및 쿠버네티스 지원)**
        - 런타임 중에 쿼리 실행 계획을 스스로 최적화하는 **AQE(Adaptive Query Execution)** 기능 추가
            - 개발자가 세부 설정을 일일이 튜닝하지 않아도 지능적으로 최적의 파티셔닝과 조인(Join) 방식을 찾아가게 됨
        - 기존 Hadoop YARN 중심에서 벗어나 Kubernetes(K8s) 지원이 공식적으로 안정화(GA)
            - 클라우드 네이티브 인프라와의 결합이 가속화됨

    - **현재 (2020년대 중반): 레이크하우스(Lakehouse) 아키텍처의 중심**
        - 최근 데이터 엔지니어링 생태계
            - Apache Iceberg, Delta Lake 등 고성능 스토리지 테이블 포맷과 Spark를 연동<br>
            🡲 트랜잭션 처리가 가능한 레이크하우스(Lakehouse)를 구축하는 것이 대세
        - Python 생태계와의 완벽한 통합(Pandas API on Spark 등) 및 AI/ML 연산 성능 강화를 지속적으로 이어가고 있음


```text
[2009년] UC 버클리 탄생 (Hadoop의 디스크 병목 극복 목적)
   🡳
[2014년] Apache 탑레벨 승격 & 1.0 (인메모리 분산 처리의 대중화)
   🡳
[2016년] Spark 2.0 (DataFrame 및 Structured API 표준화)
   🡳
[2020년] Spark 3.0 (AQE 도입 및 Kubernetes 클라우드 네이티브 최적화)
   🡳
[현재]  Iceberg 등과 결합하여 '현대적 데이터 레이크하우스'의 표준 엔진으로 군림

```


### 2.2 핵심 개념: RDD

- Spark의 근간이 되는 데이터 모델은 **RDD (Resilient Distributed Dataset)**
    - **Resilient (탄력적):**
        - 메모리 내에서 데이터가 손실될 경우 리니지(Lineage)를 통해 자동으로 복구됨
    - **Distributed (분산):**
        - 클러스터 내의 여러 노드에 데이터가 나누어 저장됨
    - **Dataset (데이터셋):**
        - 객체들의 모음

- 현재는 RDD를 기반으로 최적화된 **DataFrame**과 **Dataset** API를 주로 사용하여 데이터 분석을 수행함


### 2.3 사용 목적 및 활용도

- **사용 목적**
    - **실시간 데이터 처리:**
        - 대량의 데이터를 스트리밍 방식으로 즉시 분석
    - **반복적 알고리즘 수행:**
        - 머신러닝 모델 학습과 같이 동일한 데이터를 여러 번 조회해야 하는 작업에 최적화됨
    - **통합 분석 환경:**
        - SQL 쿼리, 스트리밍, 그래프 처리, 머신러닝을 하나의 엔진에서 처리할 수 있음

- **주요 활용도**
    - **배치 처리 (Batch Processing):**
        - 대규모 로그 분석 및 ETL 작업
    - **머신러닝 (MLlib):**
        -분류, 회귀, 클러스터링 등 대규모 데이터 기반 학습
    - **실시간 스트리밍 (Spark Streaming):**
        -금융 사기 탐지, 실시간 센서 데이터 모니터링
    - **대화형 분석 (Spark SQL):**
        -SQL을 사용하여 대규모 데이터셋을 빠르게 탐색
    - **그래프 처리 (GraphX):**
        -소셜 네트워크 분석이나 추천 엔진


### 2.4 Apache Spark의 장단점

- **장점**
    - **속도 (In-memory Computing):**
        - 데이터를 디스크가 아닌 메모리에 유지하며 처리
        - MapReduce보다 최대 100배(디스크 기준 10배) 빠름
    - **사용 편의성:**
        - Java, Scala, Python, R 등 다양한 언어를 지원
        - 80개 이상의 고수준 연산자를 제공
    - **결함 허용 (Fault Tolerance):**
        - 작업 중 노드가 고장 나도 작업을 처음부터 다시 할 필요 없이 유실된 부분만 계산하여 복구
    - **통합성:**
        - Hadoop(HDFS), Cassandra, HBase, S3 등 다양한 데이터 소스와 쉽게 연동됨

- **단점**
    - **높은 메모리 비용:**
        - 모든 처리를 메모리 위에서 하려다 보니 RAM 사용량이 매우 많고 비용 부담이 발생할 수 있음
    - **설정의 복잡성:**
        - 최적의 성능을 내기 위해 메모리 관리, 파티션 크기 등 세부적인 튜닝(Tuning)이 까다로움
    - **실시간 응답 한계:**
        - '마이크로 배치' 방식을 사용하기 때문에
        - Apache Flink와 같은 순수 스트리밍 엔진에 비해 완전한 실시간(Millisecond 단위) 응답 속도는 미세하게 떨어질 수 있음


### 2.5 기술 스택 구성 요소

- Spark는 목적에 따라 다음과 같은 라이브러리를 포함함

<div class="info-table">
<table>
    <thead>
        <th style="width: 200px;">구성 요소</th>
        <th style="width: 400px;">설명</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">Spark SQL</td>
            <td>구조화된 데이터를 SQL로 처리</td>
        </tr>
        <tr>
            <td class="td-rowheader">Spark Streaming</td>
            <td>실시간 스트리밍 데이터 처리</td>
        </tr>
        <tr>
            <td class="td-rowheader">MLlib</td>
            <td>머신러닝 라이브러리</td>
        </tr>
        <tr>
            <td class="td-rowheader">GraphX</td>
            <td>그래프 및 병렬 그래프 계산</td>
        </tr>
    </tbody>    
</table>
</div>

- 과거에는 하둡이 빅데이터의 전부였다면,
- 현재는 "저장은 하둡(HDFS), 처리는 스파크(Spark)"라는 공식이 보편적으로 쓰일 만큼 데이터 엔지니어링 분야에서 필수적인 도구

### 3. Trino와의 비교

- **Trino**와 **Apache Spark**는 둘 다 빅데이터 생태계에서 가장 널리 쓰이는 오픈소스 분산 처리 엔진
- 현대적인 데이터 플랫폼(Lakehouse)을 운영하는 기업들은 이 둘을 경쟁 관계로 보지 않고 동시에 함께 사용(조합)

- **아키텍처 및 데이터 처리 방식의 근본적 차이**
    - 두 도구가 데이터를 다루는 방식은 'F1 레이싱카(Trino)'와 '거대 화물 열차(Spark)'의 차이로 비유할 수 있음

    <div class="info-table">
    <table>
        <thead>
            <th style="width: 150px;">비교 항목</th>
            <th style="width: 400px;">Trino (트리노)</th>
            <th style="width: 400px;">Apache Spark (스파크)</th>
        </thead>
        <tbody>
            <tr>
                <td class="td-rowheader">핵심 패러다임</td>
                <td>MPP (Massive Parallel Processing)</td>
                <td>MapReduce 기반 메모리 확장형 뼈대</td>
            </tr>
            <tr>
                <td class="td-rowheader">처리 방식</td>
                <td>인메모리 스트리밍 파이프라인<br>(중간 결과를 디스크에 쓰지 않고 다음 단계로 직송)</td>
                <td>스테이지(Stage) 기반 단계적 처리<br>(안정성을 위해 단계별로 디스크/메모리에 셔플링)</td>
            </tr>
            <tr>
                <td class="td-rowheader">인터페이스</td>
                <td>순수 표준 ANSI SQL만 지원</td>
                <td>다중 언어 프로그래밍 API 지원<br>(Python, Scala, Java, R + Spark SQL)</td>
            </tr>
            <tr>
                <td class="td-rowheader">내결함성<br>(Fault Tolerance)</td>
                <td>중(Medium)<br>기본적으로 중간 실패 시 전체 쿼리 재실행<br>(최근 아키텍처 개선으로 태스크 리트라이가 추가됨)</td>
                <td>최상 (Highest)**<br>데이터 유실 시 `RDD 계보(Lineage)`를 추적해 실패한 특정 파티션만 자동 복구 가능</td>
            </tr>
            <tr>
                <td class="td-rowheader">데이터 연합<br>(Data Federation)</td>
                <td>매우 강력<br>서로 다른 물리 DB를 SQL 한 줄로 실시간 조인</td>
                <td>제한적<br>코드로 구현은 가능하나 오버헤드가 크고 실시간 조인에 불리함</td>
            </tr>
            <tr>
                <td class="td-rowheader">동시성 (Concurrency)</td>
                <td>한 클러스터에서 수백 개의 동시 쿼리 처리 가능</td>
                <td>대규모 동시성(High Concurrency) 대응에 비효율적</td>
            </tr>
        </tbody>    
    </table>
    </div>


- **Trino와 Spark를 '같이' 사용하는 이유 (조합 패턴)**
    - **"목적이 비슷하다면 같이 사용할 이유가 없을 것이다???"**
    - 두 도구의 장점이 정반대에 있기 때문에, 현대적 데이터 레이크하우스(Data Lakehouse) 아키텍처에서는 두 엔진을 무조건 혼용하는 것이 표준 패턴(Best Practice)
    - 두 도구는 데이터의 저장 포맷(예: Apache Iceberg, Delta Lake)과 중앙 메타스토어(예: Hive Metastore, Nessie)를 공유하며 완벽하게 공존

- **실제 기업들의 협업 시나리오 (ETL은 Spark, 조회의 Trino)**
    - **[Spark 역할 - 거대 무쇠 가마솥]:**
        - 매일 새벽,
            - 전날 발생한 테라바이트급의 가공되지 않은 Raw 로그(JSON, CSV 등)를 긁어모아 복잡한 비즈니스 로직을 적용하고,
            - 압축률과 조회가 빠른 오픈 테이블 포맷(Parquet, Iceberg 등)으로 변환하여
            - S3(데이터 레이크)에 저장
        - 이 과정에서 노드가 몇 개 죽어도 Spark은 이어서 작업을 완수

    - **[Trino 역할 - 초고속 에스프레소 머신]:**
        - 출근한 데이터 분석가, 마케터, 혹은 사내 BI 대시보드(Tableau, Superset)가
            - S3에 잘 정제되어 저장된 Iceberg 테이블을 조회
            - Trino는 이 쿼리를 몇 초 만에 처리해 대시보드에 뿌려줌
            - 마케터가 "어제 로그 데이터랑 마케팅 MySQL DB에 있는 유저 정보를 합쳐서 보고 싶다"고 하면,
                - Trino가 그 자리에서 두 저장소를 엮어 즉시 결과를 도출


- **상황별 권장 가이드: 언제 무엇을 써야 할까?**

    - **Trino가 강력히 권장되는 경우 (Query & Analytics)**
        - **BI 대시보드 및 실시간 시각화 툴 연동:**
            - 사용자가 대시보드 필터를 바꿀 때마다 쿼리가 날아가므로,
            - 초 단위의 빠른 반응 속도가 필수적일 때
        - **Ad-hoc(대화형) 데이터 탐색:**
            - 데이터 사이언티스트나 분석가가 "이 데이터는 어떻게 생겼지?" 하고 이것저것 쿼리를 날려보며
            - 즉각적인 피드백을 원할 때
        * **데이터 이동(ETL) 없는 파편화된 데이터 조회:**
            - 데이터가 AWS S3, 로컬 PostgreSQL, MongoDB 등에 사방으로 흩어져 있고,
            - 이를 한 곳으로 복사해오는 파이프라인을 만들 시간적/비용적 여유가 없을 때
        * **엔지니어가 아닌 현업 분석가 중심 환경:**
            - 프로그래밍 언어(Python, Scala)를 모르는 분석가들이
            - 표준 SQL만으로 빅데이터를 다루어야 할 때

    - **Spark가 강력히 권장되는 경우 (Processing & Engineering)**
        * **무겁고 복잡한 배치 대용량 변환 (Heavy ETL):**
            - 몇 시간 동안 수십 TB의 데이터를 정제, 압축, 정렬하여 마트(Data Mart)를 구축해야 할 때
                - 도중에 작업이 터지면 안 되는 절대 안정성이 필요한 영역
        * **머신러닝(ML) 및 데이터 사이언스 파이프라인:**
            - 데이터 추출에 그치지 않고,
            - `Spark MLlib` 등을 활용해
            - 분산 환경에서 모델을 학습시키거나 대규모 예측(Prediction)을 수행해야 할 때
        * **비정형 데이터 가공 및 복잡한 프로그래밍 인터페이스 필요 시:**
            - 단순 SQL 함수만으로는 표현하기 힘든 복잡한 알고리즘, 파이썬 라이브러리(Pandas 등) 연계,
            - 혹은 문자열 파싱 로직을 UDF(사용자 정의 함수)나 코드로 촘촘하게 짜야 할 때
        * **실시간 스트림 처리(Streaming):**
            - Kafka 등에서 쏟아지는 데이터를 실시간으로 캐치하여 가공(Spark Structured Streaming)해야 할 때


> - **Spark**는 데이터 엔지니어가 대규모 데이터를 뚝딱거리고 가공하는 '생산 및 가공 전용 공장'
> - **Trino**는 데이터 분석가와 소비자가 가공된 데이터(혹은 원천 데이터)를 막힘없이 빠르게 꺼내 먹는 '고속 소비 창구'<br><br>
> - 따라서 인프라를 설계할 때는
>   - 무거운 정기 배치 작업은 **Spark**에 맡겨 안정적으로 데이터를 정제해 두고,
>   - 그 데이터를 포함한 전사 데이터 조회 인터페이스는 **Trino**로 단일화하여
>   - 분석 효율을 극대화하는 것이 가장 이상적
{: .summary-quote}



## 3. Apache Spark 설치 및 환경 설정

- **Docker Compose 방식** 추천

### 3.1 방법 A: 로컬 Standalone 설치 (Linux/macOS 기준)

- **사전 필수 요구사항 (Pre-requisites)**
    - Spark는 JVM(Java Virtual Machine) 위에서 동작하므로 **Java 설치 필수**
    - Python API(PySpark)를 사용하기 위해 **Python 환경**이 준비되어야 함<br><br>

    - **Java JDK:**
        - **Java 11 또는 Java 17**을 권장 (Spark 3.x 버전 이상 기준): 호환성이 가장 좋음
    - **Python:** 3.8 ~ 3.11 버전 권장
    - **운영체제:**
        - Linux 또는 macOS를 권장
        - Windows의 경우 `winutils.exe` 설정이 추가로 필요함

- **Spark 다운로드 및 압축 해제**
    1. Apache Spark 공식 다운로드 페이지에서 원하는 버전을 선택
        - 예: Spark 3.5.x, Pre-built for Apache Hadoop 3.3 이상
    2. 터미널에서 다운로드 및 압축 해제

        ```bash
        wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
        tar -xvzf spark-3.5.1-bin-hadoop3.tgz
        mv spark-3.5.1-bin-hadoop3 /opt/spark
        ```

- **환경 변수 설정 (`.bashrc` 또는 `.zshrc`)**
    - 어느 경로에서나 Spark 명령어를 실행할 수 있도록 환경 변수 등록

        ```bash
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 # 본인의 Java 경로에 맞게 수정
        export SPARK_HOME=/opt/spark
        export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
        export PYSPARK_PYTHON=python3
        ```

    - 설정 후 `source ~/.bashrc` (또는 `~/.zshrc`)를 실행하여 적용

- **실행 테스트**
    - **Spark Shell (Scala):** `spark-shell` 명령어 입력
    - **PySpark (Python):** `pyspark` 명령어 입력
    - 정상 실행 시 터미널에 대형 `Spark` 로고와 함께 대화형 콘솔이 나타남


### 3.2 방법 B: Docker Compose를 이용한 Master-Worker 구조 구축 (추천)

- 커리큘럼에 포함된 **Master-Worker(M-W) 구조**를 로컬 가상화 환경에 가장 깔끔하게 구축하는 방법
    - Bitnami에서 제공하는 검증된 이미지를 사용하면 편리함

- **`docker-compose.yml` 파일 작성**
    - 프로젝트 디렉터리를 만들고 아래와 같이 작성
    - 이 구조는 Master 노드 1대와 Worker 노드 2대를 띄우는 가상 클러스터

    ```yaml
    version: '3.8'

    services:
    spark-master:
        image: bitnami/spark:3.5
        environment:
        - SPARK_MODE=master
        - SPARK_RPC_AUTHENTICATION_ENABLED=no
        - SPARK_RPC_ENCRYPTION_ENABLED=no
        - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
        - SPARK_SSL_ENABLED=no
        ports:
        - '8080:8080' # Master Web UI
        - '7077:7077' # Spark Master 내부 통신 포트

    spark-worker-1:
        image: bitnami/spark:3.5
        environment:
        - SPARK_MODE=worker
        - SPARK_MASTER_URL=spark://spark-master:7077
        - SPARK_WORKER_MEMORY=2G
        - SPARK_WORKER_CORES=2
        depends_on:
        - spark-master

    spark-worker-2:
        image: bitnami/spark:3.5
        environment:
        - SPARK_MODE=worker
        - SPARK_MASTER_URL=spark://spark-master:7077
        - SPARK_WORKER_MEMORY=2G
        - SPARK_WORKER_CORES=2
        depends_on:
        - spark-master
    ```

- **클러스터 실행 및 웹 UI 확인**
    1. 터미널에서 해당 디렉터리로 이동 후 명령어 실행

        ```bash
        docker-compose up -d
        ```

    2. 웹 브라우저를 열고 `http://localhost:8080`에 접속
    3. **Spark Master Web UI**가 나타남
        - 하단 `Workers` 항목에 2대의 Worker 노드가 정상적으로 등록(Alive 상태)되어 있는지 확인


- **핵심 환경 설정 파일 튜닝 (`$SPARK_HOME/conf/`)**
    - 실무 및 대용량 데이터 처리를 위해 반드시 알아야 하는 주요 설정 파일들
        - Docker 환경의 경우 환경 변수나 SparkSession 생성 시 옵션으로 주입할 수 있음

    - **`spark-env.sh` (시스템 환경 설정)**
        - 메모리와 CPU 코어 자원을 제한하거나 지정할 때 사용

            ```bash
            SPARK_MASTER_HOST=localhost
            SPARK_WORKER_CORES=4          # 각 Worker가 사용할 CPU 코어 수
            SPARK_WORKER_MEMORY=4g        # 각 Worker가 사용할 총 메모리
            SPARK_DRIVER_MEMORY=2g        # 드라이버 프로그램이 사용할 메모리
            ```

    - **`spark-defaults.conf` (Spark 애플리케이션 기본값)**
        - 제출되는 모든 Spark 작업에 공통으로 적용될 옵션을 지정함
        - 향후 **Iceberg 및 MinIO 연동** 시 이 파일이나 작업 제출 시점에 커넥터 JAR 패키지 및 S3 엔드포인트를 지정해야 함

            ```properties
            # 셔플 파티션 수 기본값 조정 (대용량이 아닐 경우 200개는 너무 많으므로 줄여서 최적화)
            spark.sql.shuffle.partitions   50

            # 향후 전개될 MinIO(S3) 및 Iceberg 연동 예시 세팅
            spark.sql.extensions           org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
            spark.sql.catalog.demo         org.apache.iceberg.spark.SparkCatalog
            spark.sql.catalog.demo.type    hadoop
            spark.sql.catalog.demo.warehouse s3a://my-bucket/warehouse
            ```

> - **환경 설정 확인을 위한 체크리스트**
>   - 설정이 완료된 후 아래의 요소들이 정상 작동하는지 확인하면 준비가 끝남
>       1. **포트 충돌 여부:**
>           - Master UI(`8080`) 또는 Spark 애플리케이션 UI(`4040`)가 다른 서비스와 충돌하지 않는지 확인
>       2. **자원 할당 확인:**
>           - Web UI에서 할당된 Memory와 Cores가 시스템 스펙 내에서 적절히 잡혔는지 확인
>           - 데이터가 밀릴 경우 이 자원 설정(Data Skew 및 Memory 부족 이슈)을 가장 먼저 튜닝하게 됨
{: .common-quote}

- **Trino와 연동 시 8080 포트 충돌**
    - **Trino(과거 Presto) 역시 기본 포트로 `8080`을 사용**하기 때문에,
    - 동일한 호스트(서버)나 로컬 PC에서 Spark Master와 Trino를 동시에 띄우면 무조건 포트 충돌(Port Collision)이 발생함

    - 최근의 데이터 레이크하우스 아키텍처에서는 **"저장(MinIO/S3 + Iceberg) + 계산(Spark) + 인터랙티브 쿼리/대시보드(Trino)"** 조합이 글로벌 표준(대세)
    - 이 포트 충돌을 해결하는 표준적인 처리 패턴이 존재함

        - **Docker Compose 환경에서의 포트 포워딩 (가장 추천)**
            - Docker 환경을 사용 중이라면, 컨테이너 내부 포트는 `8080`으로 그대로 두더라도,
            - **호스트(사용자 PC)로 노출하는 외부 포트를 변경**하여 충돌을 간단히 피할 수 있음<br><br>

            - **Spark Master 설정 (외부 포트를 `8180`으로 변경)**

                ```yaml
                services:
                spark-master:
                    image: bitnami/spark:3.5
                    ports:
                    - '8180:8080' # [호스트 포트 8180] : [컨테이너 내부 포트 8080]
                    - '7077:7077'
                ```

            - **Trino 설정 (외부 포트를 `8080`으로 유지 또는 `8090`으로 변경)**

                ```yaml
                services:
                trino:
                    image: trinodb/trino
                    ports:
                    - '8080:8080' # Trino는 원래대로 8080 사용
                ```

            - 이렇게 설정하면 브라우저에서 Spark UI는 `http://localhost:8180`으로, Trino UI는 `http://localhost:8080`으로 충돌 없이 깔끔하게 접근할 수 있음


        - **Spark 자체 설정으로 기본 포트 변경하기 (Standalone 환경)**
            - Docker를 쓰지 않고 로컬 서버에 직접 설치한 환경이라면,
            - Spark의 환경 설정 파일이나 실행 스크립트에서 기본 웹 UI 포트를 다른 값(예: `8282`)으로 바꿀 수 있음<br><br>

            - **방법 A: `spark-env.sh` 파일 수정**
                - `$SPARK_HOME/conf/spark-env.sh` 파일에 아래 설정을 추가합니다.

                    ```bash
                    export SPARK_MASTER_WEBUI_PORT=8282
                    ```

            - **방법 B: 데몬 실행 시 옵션 주입**
                - 마스터 노드를 구동하는 스크립트를 실행할 때 포트를 직접 지정

                    ```bash
                    ./sbin/start-master.sh --webui-port 8282
                    ```

        - **Spark 애플리케이션 UI(`4040`)의 자동 포트 포워딩 기능 활용**
            - 마스터 UI(`8080`) 외에, 
            - 개별 Spark 애플리케이션(작업)이 실행될 때 열리는 Spark Driver UI(`4040`)도 충돌 가능성이 있음
                - 예를 들어 여러 명의 개발자가 한 서버에서 동시에 `pyspark`를 실행하거나 여러 개의 배치 작업이 동시에 돌면 `4040` 포트가 겹치게 됨

            - Spark는 `4040` 포트에 대해 **자체적인 충돌 회피 메커니즘**을 내장하고 있음
                - `4040` 포트가 이미 사용 중이면 -> `4041` 시도 -> 또 사용 중이면 -> `4042` 시도
                - 이 과정이 성공할 때까지 자동으로 포트를 1씩 올리며 바인딩함<br><br>
                - 만약 이 자동 변경 범위를 제어하거나 명시적으로 바꾸고 싶다면
                    - SparkSession을 생성할 때 아래 옵션을 부여

                        ```python
                        from pyspark.sql import SparkSession

                        spark = SparkSession.builder \
                            .appName("Iceberg-Trino-Spark-Test") \
                            .config("spark.ui.port", "4050") \ # 기본 4040 대신 4050부터 시작하도록 설정
                            .getOrCreate()
                        ```

<br>

> - **권장 아키텍처 구성**
>   - Trino와 Spark를 함께 엮어 데이터 레이크하우스를 구성할 때
>       - 아래와 같이 포트 맵핑을 깔끔하게 정리해 두고 진행하시는 것이 정신 건강(?)에 좋음<br><br>
>
>       <div class="info-table">
>       <table>
>           <thead>
>               <th style="width: 200px;">서비스 구성 요소</th>
>               <th style="width: 150px;">내부 기본 포트</th>
>               <th style="width: 200px;">추천 외부(호스트) 포트</th>
>               <th style="width: 350px;">용도</th>
>           </thead>
>           <tbody>
>               <tr>
>                   <td class="td-rowheader">Trino</td>
>                   <td>8080</td>
>                   <td>8080</td>
>                   <td>대시보드 연동 및 BI용 쿼리 UI</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">Spark Master</td>
>                   <td>8080</td>
>                   <td><b>8180 (변경)</b></td>
>                   <td>클러스터 및 Worker 상태 모니터링</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">Spark Driver App</td>
>                   <td>4040</td>
>                   <td><b>4040(자동 순차 증가)</b></td>
>                   <td>개별 Spark Job 실행 상세 트래킹</td>
>               </tr>
>               <tr>
>                   <td class="td-rowheader">MinIO Web UI</td>
>                   <td>9001</td>
>                   <td>9001</td>
>                   <td>Iceberg 데이터 저장소 오브젝트 브라우징</td>
>               </tr>
>           </tbody>    
>       </table>
>       </div>
{: .common-quote}

- **Docker Compose 사용 시, HOST 작업환경 구성**
    - Docker Compose 운영환경 그대로 사용할 경우
        - 컨테이너 내부에 직접 들어가서 명령어를 쳐야 하므로 코드 작업이 매우 불편함
        - **코드는 HOST(내 컴퓨터)에서 편하게 편집, 실행은 도커(컨테이너) 내부에서 돌아가도록 연결**할 것을 권장함<br><br>

    - **실무에서 가장 많이 쓰는 패턴**
        - **볼륨 마운트 (Volume Mount)**
            - 코드는 HOST의 VS Code 등으로 편집
            - 코드 파일이 들어있는 폴더를 도커 컨테이너 내부와 실시간으로 동기화(연결)<br><br>
            * **작업 방식:**
                1. 호스트 PC의 특정 폴더(예: `~/workspace/spark_project`)에 `app.py`라는 파이썬 코드 작성
                2. `docker-compose.yml` 설정에서 이 폴더를 Spark 컨테이너 내부의 특정 경로와 연결(`volumes` 옵션)
                3. 호스트에서 코드를 수정하고 저장하면, 컨테이너 내부에도 즉시 반영됨
                4. 실행할 때는 도커 터미널을 통해 컨테이너 안에서 `spark-submit app.py`를 입력해 실행<br><br>
                - **장점:**
                    - 호스트의 강력한 IDE(VS Code 등) 환경을 그대로 쓰면서,
                    - 실행은 깨끗한 도커 환경에서 할 수 있음

        - **주피터 노트북 (Jupyter Notebook) 패턴: 브라우저 활용**
            - 데이터 분석이나 Spark 실습 시 가장 직관적인 방법<br><br>
            * **작업 방식:**
                1. Spark 도커 이미지 안에 이미 주피터 노트북(또는 주피터 랩) 서버가 내장되어 있거나 추가되어 있음
                2. 도커를 띄우면 컨테이너 내부에서 주피터 서버가 돌아감
                3. 사용자는 호스트 PC의 크롬 브라우저를 열고 `http://localhost:8888`에 접속
                4. 웹 브라우저 화면에서 파이썬 코드를 작성하고 즉시 실행(`Shift + Enter`)<br><br>
                - **장점:**
                    - 코드 작성과 실행 결과 확인이 웹 브라우저 하나로 끝나므로 학습 및 프로토타이핑에 최적
                    - 생성된 `.ipynb` 파일은 앞서 말한 볼륨 마운트를 통해 호스트 PC에 안전하게 저장됨

        - **고급 패턴: VS Code의 'Dev Containers' 확장 기능 사용**
            - 최근 시니어 개발자나 엔지니어들이 가장 선호하는 깔끔한 방식<br><br>
            * **작업 방식:**
                1. 호스트 PC의 VS Code에 `Dev Containers`라는 공식 확장 프로그램을 설치
                2. VS Code 자체를 도커 컨테이너 내부로 접속
                3. 화면은 내 컴퓨터에 떠 있지만, VS Code가 인식하는 파이썬 환경, 확장 프로그램, 터미널은 전부 **도커 컨테이너 내부의 환경**이 됨<br><br>
                * **장점:**
                    - 호스트 PC에 파이썬을 깔지 않아도 코드 자동 완성(IntelliSense)이나 디버깅 기능이 컨테이너 내부 스택에 맞춰 완벽하게 작동함
