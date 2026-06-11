---
layout: page
title:  "MinIO 오브젝트 스토리지 구축"
date:   2026-06-11 10:00:00 +0900
permalink: /materials/S13-02-02-01_01-MinIoObjectStorage
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 오브젝트 스토리지 중심의 데이터 파이프라인

> - 왜 현대 데이터 엔지니어링 아키텍처는 과거의 파일 시스템(NAS, SAN)이나 관계형 데이터베이스(RDBMS)를 넘어 오브젝트 스토리지(Object Storage)를 중심축으로 삼게 되었는가?
{: .common-quote}


### 1.1 기존 아키텍처의 한계와 오브젝트 스토리지

- **과거의 데이터 관리 방식**
    - **디렉토리/파일 구조(POSIX 파일 시스템):**
        - 폴더 구조가 깊어질수록 파일 탐색 속도가 기하급수적으로 느려지는 계층적 한계가 있음

    - **데이터베이스 블록 구조(SAN/RDBMS):**
        - 고성능이지만 저장 비용이 매우 비쌈
        - 비정형 데이터(이미지, 영상, 대용량 로그)를 담기에 부적합

- **오브젝트 스토리지의 혁신**
    - 오브젝트 스토리지는 데이터 관리를 "평면적 구조(Flat Structure)"로 완전히 전환함

    - **고유 키(Key) 매핑:**
        - 폴더 계층 없이 모든 파일은 버킷(Bucket)이라는 거대한 단일 공간에 고유한 주소(Key, 예: `raw-data-lake/2026/06/11/log.txt`)를 가진 '오브젝트' 형태로 저장됨

    - **무한한 확장성(Scalability):**
        - 주소값만 알면 데이터가 페타바이트(PB), 엑사바이트(EB) 단위로 늘어나도 $$O(1)$$에 수렴하는 일정한 탐색 성능을 보장함


### 1.2 현대 데이터 스택의 핵심: 저장과 연산의 분리

- 과거의 데이터 웨어하우스(Hadoop, 전통적 RDBMS 등)
    - 데이터를 저장하는 하드웨어와 데이터를 계산(쿼리, 분석)하는 하드웨어가 물리적으로 묶여 있음
    - 이로 인해 분석을 많이 하려면 필요 없는 저장용 서버까지 세트로 증설해야 하는 비용 낭비가 심함

- 현대 데이터 아키텍처
    - 철저한 **컴퓨팅(Compute)과 스토리지(Storage)의 분리(Decoupling)** 구조로 설계

    - **스토리지 레이어 (MinIO):**
        - 오직 안전하고 저렴하게 데이터를 저장하는 데만 집중 (비용 극대화 효율)

    - **컴퓨팅 레이어 (Python, Spark, Trino 등):**
        - 분석이나 데이터 정제가 필요할 때만 CPU/GPU 자원을 켜서 스토리지에 요청하여 데이터를 가져옴 (자원 효율성)

        > - 이번 Python 실습이 바로 이 구조
        >   - MinIO(스토리지)는 가만히 있고, Python 스크립트(컴퓨팅)가 네트워크 API 통신을 통해 데이터를 넣고 빼는 구조적 분리를 이론적으로 경험하는 것
        {: .common-quote}


### 1.3 Schema-on-Read 개념

- 전통적인 데이터베이스(Oracle, MySQL 등)
    - 데이터를 집어넣기 전에 반드시 테이블 구조(Column 이름, 데이터 타입 등)를 엄격하게 정의해야 하는 **Schema-on-Write** (쓰기 시점의 스키마 정의) 방식
    - 규격에 맞지 않는 데이터는 에러 발생

- 오브젝트 스토리지 중심의 데이터 레이크(Data Lake)
    - **Schema-on-Read** (읽기 시점의 스키마 정의) 방식

    - **원시 데이터 그대로 저장(Landing):**
        - 데이터가 생성되는 시점에는 그것이 텍스트든 JSON이든 관계없이 오브젝트 스토리지(MinIO)에 날것(Raw) 그대로 빠르게 적재함

    - **해석은 읽는 자의 몫:**
        - 나중에 데이터를 가져가서 분석하는 프로그램(Python, Spark 등)이 데이터를 읽어 들이는 시점에 정의
            - 예: 데이터를 읽으면서 "이 구조는 콤마(,)로 분리되어 있으니 표 형태로 해석하겠다"라고 정의

    - **이점:**
        - 시스템 변화에 매우 유연함
        - 데이터 유실 없이 모든 원천 데이터를 영구 보관할 수 있음


### 1.4 RESTful API 기반 데이터 파이프라인의 표준화

- 과거에는 파일 전송을 위해 FTP, SAMBA, NFS 등 운영체제나 네트워크 환경에 종속적인 프로토콜을 사용해야 했음
- 이는 방화벽 설정이 복잡하고 웹 환경에서 다루기 어려움<br><br>

- 오브젝트 스토리지는 모든 데이터의 입출력을 웹의 기본 통신 규격인 **HTTP/HTTPS (RESTful API)** 기반으로 처리함

    ```
    - 데이터 업로드: HTTP PUT /raw-data-lake/sample.txt
    - 데이터 다운로드: HTTP GET /raw-data-lake/sample.txt
    - 데이터 삭제: HTTP DELETE /raw-data-lake/sample.txt
    ```

- Amazon S3가 정의한 이 HTTP 통신 규칙이 너무나 완벽했기에 전 세계 개발사들이 이를 표준으로 삼았음
- **MinIO는 이 HTTP API 명세를 로컬 시스템 상에 그대로 구현해 둔 소프트웨어**

<br>

> - **이론적 배경 요약**
>   - 이번 실습은 단순히 코드를 실행해 보는 것을 넘어,
>   - "비용이 저렴하고 무한 확장이 가능한 평면적 저장소(MinIO)에, 데이터 형식을 가리지 않고 원시 형태(Schema-on-Read)로 안전하게 던져놓고, 필요할 때마다 독립된 연산 장치(Python)가 웹 표준 프로토콜(S3 API)로 데이터를 제어하는 현대 데이터 엔지니어링의 기본 패러다임"을 몸소 체득하는 과정
{: .summary-quote}



## 2. MinIO기반 오브젝트 스토리지 파이프라인 구축 실습

- 파일이나 데이터를 프로그램(Python 등)을 통해 MinIO 스토리지에 직접 적재, 관리하는 ‘오브젝트 스토리지 중심의 데이터 파이프라인’ 구축
    - **저장소 계층 (Storage Layer):** 로컬 PC에 띄운 MinIO 엔진
    - **데이터 생성/소비 계층 (App Layer):** Python 스크립트 (AWS S3 표준 라이브러리인 `boto3` 활용)

### 2.1 MinIO 환경 구축

> - 가장 빠르고 간편한 **Docker 환경**을 기준으로 진행함

- **[1단계] 로컬에 MinIO 스토리지 엔진 구동하기**
    - 터미널(또는 커맨드 창)에서 아래 명령어를 실행하여 로컬 S3 환경을 만듦

        ```bash
        docker run -d \
        -p 9000:9000 \
        -p 9001:9001 \
        --name local-s3-storage \
        -v ~/minio/data:/data \
        -e "MINIO_ROOT_USER=myaccesskey" \
        -e "MINIO_ROOT_PASSWORD=mysecretkey" \
        minio/minio server /data --console-address ":9001"
        ```

- **[2단계] 웹 콘솔에서 실습용 버킷(Bucket) 만들기**
    1. 브라우저에서 `http://localhost:9001`에 접속
    2. ID: `myaccesskey` / PW: `mysecretkey`로 로그인
    3. **Buckets** 메뉴 $\rightarrow$ **Create Bucket**을 눌러 `raw-data-lake`라는 이름의 버킷 생성


### 2.2 Python 코드로 MinIO(S3 API) 연동

- Python 코드로 로컬 스토리지에 데이터를 넣고 빼는 실습 진행
- 아마존 S3와 100% 호환되기 때문에 AWS 공식 라이브러리인 `boto3`를 그대로 사용함

- **[1단계] 라이브러리 설치**

    ```bash
    pip install boto3
    ```

- **[2단계] 데이터 업로드 및 다운로드 스크립트**

```python
#//file: "app.py"
import boto3
from botocore.client import Config

# 1. 로컬 MinIO 서버 연결 설정 (핵심은 endpoint_url 지정!)
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',          # 로컬 MinIO API 포트
    aws_access_key_id='myaccesskey',               # MinIO ROOT_USER
    aws_secret_access_key='mysecretkey',           # MinIO ROOT_PASSWORD
    config=Config(signature_version='s3v4')
)

bucket_name = 'raw-data-lake'

# 2. 테스트용 텍스트 파일 생성 및 업로드
with open("local_sample.txt", "w", encoding="utf-8") as f:
    f.write("이 데이터는 로컬 MinIO 오브젝트 스토리지에 저장되는 테스트 데이터입니다.")

print("1. 파일 업로드 시작...")
s3_client.upload_file('local_sample.txt', bucket_name, 'cloud_storage_sample.txt')
print("업로드 완료! 웹 콘솔(http://localhost:9001)에서 파일을 확인해보세요.\n")


# 3. MinIO 스토리지에서 파일 다운로드 테스트
print("2. 스토리지에서 파일 다운로드 시작...")
s3_client.download_file(bucket_name, 'cloud_storage_sample.txt', 'downloaded_result.txt')

with open("downloaded_result.txt", "r", encoding="utf-8") as f:
    print(f"다운로드된 파일 내용: {f.read()}")
```

<br>

> - **요약 및 비유**
>   - 집(DataLake)을 지을 때,
>       - **MinIO는 '토지와 콘크리트 골조'**를 만드는 것이고,
>       - **Iceberg는 '방을 나누는 인테리어(구조화)'**,
>       - **Trino는 '거기에 입주해서 편리하게 가전을 이용하는 것(조회)'**과 같음<br><br>
>   - 지금은 인테리어(Iceberg)와 입주 가전(Trino)이 없더라도,
>       - **튼튼한 골조(MinIO 오브젝트 스토리지 구축)를 짜고
>       - 그 빈 공간에 택배 박스(데이터 파일)를 넣고 빼는 작업**은 100% 완벽하게 진행할 수 있음
