---
layout: page
title:  "MinIO 개요 및 설치, 환경설정"
date:   2026-06-01 10:00:00 +0900
permalink: /materials/S13-02-01-01_01-MinIoOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


## 1. MinIO 개요

> - **정의 및 개념**
>   - 클라우드 네이티브 환경에 최적화된 **고성능 오브젝트 스토리지(Object Storage)** 솔루션
>   - Amazon S3 API와 완벽하게 호환되며,
>   - 오픈 소스 기반으로 시작하여 현재는 기업용 데이터 레이크와 AI/ML 인프라의 핵심 구성 요소로 자리 잡고 있음
{: .common-quote}

<br>

- **주요 특징**
    - **'오브젝트' 단위 관리**
        - MinIO는 데이터를 '파일' 단위가 아닌 **'오브젝트'** 단위로 관리함
        - 계층적인 폴더 구조 대신 고유한 ID(Key)를 가진 데이터 객체로 저장하기 때문
        - 대규모 데이터 처리에 매우 유리함

    - **S3 호환성:**
        - Amazon S3 API를 표준처럼 사용
        - 기존 S3 기반 애플리케이션을 수정 없이 MinIO로 전환할 수 있음

    - **Kubernetes Native:**
        - 컨테이너 환경에서 동작하도록 설계됨
        - Docker나 Kubernetes 환경에서 배포 및 확장이 매우 간편

    * **고성능:**
        - 하드웨어 가속 및 어셈블리 최적화를 통해 초당 수백 GB의 읽기/쓰기 속도를 지원
        - 이는 AI 모델 트레이닝 등 고부하 작업에 적합함

<br>

- **사용 목적**
    - **프라이빗 클라우드 구축:**
        - 퍼블릭 클라우드(AWS, GCP 등)를 사용하지 않고 사내 인프라에 직접 S3와 같은 스토리지 환경을 구축할 때 사용

    - **데이터 레이크(Data Lake):**
        - 비정형 데이터(로그, 이미지, 비디오 등)를 대량으로 저장하고 분석하기 위한 저장소 역할

    - **백업 및 아카이빙:**
        - 랜섬웨어 방지를 위한 Object Locking 기능을 활용하여 안전한 백업 시스템을 구축

<br>

- **주요 활용도**
    - **AI 및 머신러닝:**
        - TensorFlow, PyTorch, Hugging Face 등과 연동하여 대규모 학습 데이터를 저장하고 모델 체크포인트를 관리
    - **로그 수집 및 분석:**
        - Splunk, ELK Stack 등과 결합하여 시스템 로그를 영구 보관하는 저장소로 활용
    - **하이브리드 클라우드:**
        - 로컬 서버와 퍼블릭 클라우드 간의 데이터 동기화 및 가용성 확보를 위한 가교 역할을 수행

<br>

- **장점과 단점**

<div class="info-table">
<table>
    <thead>
        <th style="width: 100px;">구분</th>
        <th style="width: 600px;">주요 내용</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">장점</td>
            <td class="td-left">
                - <b>초고속 성능:</b> 하드웨어 자원을 극한으로 활용하여 업계 최고 수준의 처리량을 제공<br>
                - <b>강력한 보안:</b> 서버 측 암호화, ID 관리(IAM), WORM(Write Once Read Many) 기능을 지원<br>
                - <b>단순함:</b> 설치와 운영이 매우 직관적이며 가벼움<br>
                - <b>확장성:</b> 테라바이트(TB)에서 엑사바이트(EB) 단위까지 유연하게 확장 가능
            </td>
        </tr>
        <tr>
            <td class="td-rowheader">단점</td>
            <td class="td-left">
                - <b>라이선스 변화:</b> 최근 AGPLv3 라이선스를 적용하여, 상업적 이용 시 라이선스 검토가 엄격해짐<br>
                - <b>메모리 의존성:</b> 고성능을 내기 위해 하드웨어 사양(특히 RAM) 요구치가 높을 수 있음<br>
                - <b>운영 부담:</b> 직접 인프라를 관리해야 하므로 하드웨어 장애 대응이나 업데이트 관리가 필요함
            </td>
        </tr>
    </tbody>    
</table>
</div>

<br>

> - **요약**
>   - MinIO는 "내 서버에 직접 구축하는 초고속 AWS S3"라고 이해하면 가장 정확함
>   - 특히 최근 AI 연구나 대규모 SW 개발 프로젝트에서 데이터 파이프라인의 핵심 저장소로 가장 많이 선택되는 솔루션 중 하나
>   - 하드웨어 성능을 최대한 끌어내면서도 클라우드 네이티브한 운영을 원한다면 최적의 선택지가 될 것
{: .summary-quote}



## 2. 기술적 메커니즘

> - MinIO는 단순히 파일을 저장하는 일반적인 파일 서버(NAS/SAN)와 달리,
> - 클라우드 네이티브 환경에서 초고속 성능과 높은 내구성을 달성하기 위해
> - 독창적인 로우레벨(Low-level) 아키텍처와 기술적 메커니즘을 채택하고 있음
{: .common-quote}


### 2.1 이레이저 코딩을 통한 데이터 보호

- 전통적인 RAID 방식이나 단순 복제(Replication) 대신
- 수학적 알고리즘 기반의 **이레이저 코딩(Erasure Coding)** 기술을 사용하여 데이터를 보호함

- **동작 원리:**
    - MinIO에 하나의 오브젝트(파일)가 업로드되면,
    - 시스템은 이를 여러 개의 데이터 블록(Data Blocks)과 패리티 블록(Parity Blocks)으로 쪼개어
    - 여러 디스크에 분산 저장함
    - 리드 솔로몬(Reed-Solomon) 코딩 알고리즘을 기반으로 함

- **높은 결함 허용(Fault Tolerance):**
    - 예를 들어 16개의 디스크로 구성된 세트에서 데이터 블록 8개, 패리티 블록 8개로 설정(`N/2`)하면,
    - **전체 디스크 중 임의의 8개가 동시에 고장 나더라도 원래 데이터를 100% 완벽하게 복구**할 수 있음

- **공간 효율성:**
    - 동일한 수준의 가용성을 확보하기 위해
        - 데이터를 3번 복제하는 방식(3-Way Replication)은 200%의 스토리지 오버헤드가 발생하지만,
        - **이레이저 코딩**은 훨씬 적은 오버헤드로도 동일하거나 더 높은 안정성을 제공함

* **비트 로트(Bit Rot) 방지:**
    - 저장장치에는 디스크가 물리적으로 고장 나지 않더라도 미세한 전류나 노후화로 인해 데이터가 조용히 손상되는 '비트 로트' 현상이 발생함
    - MinIO는
        - 각 블록마다 고유한 해시(HighwayHash)를 부여하여,
        - 데이터를 읽을 때 손상 여부를 실시간으로 감지하고
        - 이레이저 코딩을 통해 즉각 치료(Heal)함


### 2.2 하드웨어 가속화 및 로우레벨 최적화

- **CPU의 하드웨어 기능을 극한으로 활용하는 구조**
    - Go 언어로 작성되었지만,
    - 고성능 연산이 필요한 핵심 코어 부분은 **어셈블리(Assembly) 언어**로 최적화되어 있음
    - MinIO가 업계 최고 수준의 READ/WRITE 성능을 내는 비결

- **SIMD(Single Instruction Multiple Data) 활용:**
    - Intel/AMD CPU의 **AVX-512**나 ARM CPU의 **NEON** 등 하드웨어 가속 명령어를 직접 제어함
    - 대용량 데이터의 이레이저 코딩 연산이나 암호화 연산을 CPU 코어가 단 한 번의 명령어로 대량 처리(병렬화)하도록 만듦

- **제로 카피(Zero-Copy) 아키텍처:**
    - 네트워크 카드(NIC)에서 들어온 데이터 바이트를
        - 가상 머신이나 애플리케이션 계층 내에서 여러 번 복사하지 않고,
        - 메모리 버퍼를 통해 디스크 컨트롤러로 직접 전달하여
        - 커널 공간과 유저 공간 사이의 컨텍스트 스위칭 오버헤드를 최소화함


### 2.3 메타데이터와 데이터의 일체화

- Ceph 등 기존의 대규모 분산 스토리지들은
    - 파일의 이름, 크기, 권한 등의 정보(메타데이터)를 관리하기 위해
    - 별도의 외부 데이터베이스(MySQL, Cassandra, Redis 등)를 필수적으로 운영함
    - 이는 메타데이터 DB 자체가 병목 현상의 원인이 되거나 단일 장애점(SPOF)이 되는 한계를 가짐

- **DB 없는 구조:**
    - MinIO는 메타데이터를 저장하기 위한 **별도의 외부 DB를 사용하지 않음**

- **인라인(Inline) 메타데이터:**
    - 오브젝트의 실제 데이터(Data)와 메타데이터(Metadata)를
    - 디스크 상에 **하나의 아키텍처 파일 구조로 묶어서 동시**에 저장함

- **의의:**
    - 디스크에서 데이터를 찾을 때
        - "DB에서 위치 조회 🡲 실제 디스크 접근"이라는 2단계 과정을 거치지 않고,
        - 단 한 번의 디스크 I/O 연산만으로 메타데이터와 본문을 모두 읽어내기 때문에
        - 탐색 속도가 압도적으로 빠르며 아키텍처가 단순해짐


### 2.4 서버 분산 아키텍처

- 단일 노드 구동부터 엑사바이트(EB) 단위의 대형 클러스터까지 유연하게 확장할 수 있는 아키텍처 모델을 제공함

<div class="info-table">
<table>
    <thead>
        <th style="width: 280px;">아키텍처 형태</th>
        <th style="width: 690px;">설명</th>
    </thead>
    <tbody>
        <tr>
            <td class="td-rowheader">SNSD (Single Node Single Drive)</td>
            <td class="td-left">단일 서버, 단일 디스크 구조. 개발 및 실습(로컬 테스트) 환경에 주로 사용</td>
        </tr>
        <tr>
            <td class="td-rowheader">SNMD (Single Node Multiple Drives)</td>
            <td class="td-left">단일 서버 내에 여러 디스크를 장착하고 이레이저 코딩을 적용하여 가용성을 확보하는 단계</td>
        </tr>
        <tr>
            <td class="td-rowheader">MNMD (Multi-Node Multiple Drives)</td>
            <td class="td-left">여러 대의 독립된 서버들을 네트워크로 묶어 하나의 거대한 스토리지 풀(Pool)을 형성하는 <b>분산(Distributed) 모드</b></td>
        </tr>
    </tbody>    
</table>
</div>

- **Symmetric Architecture (대칭형 구조):**
    - 분산 모드에서 MinIO는 마스터(Master) 노드나 워커(Worker) 노드의 구분이 없음
    - 모든 노드가 완벽하게 대등한 권한을 가지며 API 요청을 처리함
    - 어떤 노드에 접속하더라도 클러스터 전체의 데이터를 조회할 수 있음 🡲 상단에 로드 밸런서(L4/L7)만 배치하면 손쉽게 수평 확장(Scale-out)이 가능함

- **서버 풀(Server Pools) 확장:**
    - 스토리지를 추가 확장해야 할 경우, 새로운 서버 그룹(Pool)을 기존 클러스터 뒤에 붙이기만 하면, 중단 없이 동적으로 저장 공간을 늘릴 수 있음

<br>

> - **요약**
>   - MinIO는 **"별도 DB 없이 파일과 메타데이터를 통째로 저장하고, 고성능 SIMD 어셈블리 코드로 초고속 연산을 수행하며, 리드 솔로몬 이레이저 코딩으로 디스크 장애를 완벽하게 방어하는 대칭형 분산 시스템"** 구조를 취함
>   - 이러한 로우레벨 최적화 덕분에 하드웨어의 최대 대역폭에 수렴하는 성능을 내어 현대 데이터 아키텍처에서 각광받고 있음
{: .summary-quote}


## 3. MinIO 설치 및 환경 설정

> - **MinIO는 로컬 환경(본인의 PC)에서 사용하기에 가장 최적화된 솔루션**
> - Amazon S3와 동일한 API를 사용하면서도, 별도의 비용이나 클라우드 연결 없이 본인의 컴퓨터 리소스만으로 완벽하게 동작함
> - MinIO는 복잡한 의존성 없이 단일 실행 파일 형태로 배포되기 때문에 설치와 환경 구성이 매우 직관적임
{: .common-quote}

### 3.1 전제 조건 및 포트(Port) 이해

- MinIO를 실행하면 기본적으로 **2개의 포트**가 활성화됨
    - **9000 Port (API Port):**
        - 애플리케이션 코드(Python `boto3`, Java SDK 등)나 CLI 툴이 MinIO 스토리지와 데이터를 주고받을 때 사용하는 백엔드 통신 포트
    - **9001 Port (Console Port):**
        - 웹 브라우저를 통해 접속하여 버킷(Bucket) 생성, 유저 관리, 파일 업로드 등을 시각적으로 처리하는 웹 GUI 관리 화면 포트

- 방화벽이나 컨테이너 설정 시 이 두 포트를 반드시 열어주어야 함


### 3.2 MinIO 설치

- **[방법 1] Docker를 이용한 초간단 설치 (가장 추천)**
    - 컨테이너 환경을 지원하는 Linux(Ubuntu/Mint)나 Windows 환경에서 가장 깔끔하게 설치하고 제거할 수 있는 방법

    - **[1 단계] 데이터 저장용 로컬 디렉토리 생성**
        - MinIO 컨테이너가 종료되어도 데이터가 사라지지 않도록 호스트 PC에 저장 공간을 바인딩할 디렉토리를 만듦

            ```bash
            mkdir -p ~/minio/data
            ```

    - **[2 단계] Docker 컨테이너 실행**
        - 아래 명령어를 실행하면 최신 버전의 MinIO 이미지를 내려받고 서버를 구동함

            ```bash
            docker run -d \
            -p 9000:9000 \
            -p 9001:9001 \
            --name minio_local \
            -v ~/minio/data:/data \
            -e "MINIO_ROOT_USER=minioadmin" \
            -e "MINIO_ROOT_PASSWORD=minioadmin" \
            minio/minio server /data --console-address ":9001"
            ```

            - `-v ~/minio/data:/data`: 로컬 디렉토리와 컨테이너 내부 저장소 연결(볼륨 마운트)
            - `-e "MINIO_ROOT_USER=..."`: 관리자 ID 설정 (기본값: `minioadmin`)
            - `-e "MINIO_ROOT_PASSWORD=..."`: 관리자 비밀번호 설정 (기본값: `minioadmin`, 최소 8자리 이상 추천)
            - `--console-address ":9001"`: 웹 GUI 콘솔 포트를 9001로 고정

- **[방법 2] Linux (Ubuntu/Mint) 환경 바이너리 직접 설치**
    - Docker 없이 시스템에 독립적인 데몬(Service)으로 가볍게 구동하고 싶을 때 사용하는 방법

    - **[1 단계] 바이너리 다운로드 및 권한 부여**

        ```bash
        # MinIO 서버 바이너리 다운로드
        wget https://dl.min.io/server/minio/release/linux-amd64/minio

        # 실행 권한 부여 및 시스템 바이너리 경로로 이동
        chmod +x minio
        sudo mv minio /usr/local/bin/
        ```

    - **[2 단계] 환경 변수 파일 및 저장소 생성**
        - 안전한 관리를 위해 설정 파일과 데이터 저장 경로를 분리합니다.

            ```bash
            # 데이터 저장 디렉토리 생성
            mkdir -p /mnt/data

            # 환경 변수 설정 파일 작성
            sudo mkdir -p /etc/minio
            sudo nano /etc/minio/minio.env
            ```

        - `minio.env` 파일 내부 내용:

            ```env
            MINIO_ROOT_USER=admin_user
            MINIO_ROOT_PASSWORD=strong_password_1234
            MINIO_VOLUMES="/mnt/data"
            MINIO_OPTS="--console-address :9001"
            ```

    - **[3 단계] 백그라운드 실행 (systemd 등록)**
        - 상시 구동을 위해 서비스 등록 파일을 생성함 (`sudo nano /etc/systemd/system/minio.service`)

            ```ini
            [Unit]
            Description=MinIO
            Documentation=https://min.io/docs/minio/linux/index.html
            Wants=network-online.target
            After=network-online.target

            [Service]
            Type=simple
            EnvironmentFile=/etc/minio/minio.env
            ExecStart=/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES
            Restart=always
            LimitNOFILE=65536

            [Install]
            WantedBy=multi-user.target
            ```

            ```bash
            # 서비스 활성화 및 시작
            sudo systemctl daemon-reload
            sudo systemctl enable minio
            sudo systemctl start minio
            ```


### 3.3 초기 웹 환경 설정

- 설치가 완료되면 웹 브라우저를 열고 데이터 레이크하우스 실습을 위한 기초 설정을 진행함

    1. **콘솔 접속:**
        - 브라우저 주소창에 `http://localhost:9001`을 입력

    2. **로그인:**
        - 설정한 계정(`minioadmin` / `minioadmin`)으로 로그인

    3. **버킷(Bucket) 생성:**
        - 좌측 메뉴에서 **Buckets** 🡲 **Create Bucket**을 클릭
        - 버킷 이름(예: `analytics-lakehouse`)을 입력하고 생성을 완료함
            - 이 공간이 향후 Apache Iceberg가 참조할 물리 데이터 저장소가 됨

    4. **액세스 키(Access Key) 발급:**
        * 외부 컴퓨팅 엔진(Trino 등)이나 소스 코드에서 접근할 수 있도록 권한을 넘겨주어야 함
        * 좌측 메뉴 **Access Keys** 🡲 **Create Access Key**를 클릭하여 생성된 **`Access Key`**와 **`Secret Key`**를 안전하게 복사해 둘 것


### 3.4 클라이언트 툴(mc) 설치 및 검증

- MinIO는 터미널 환경에서 S3 스토리지를 완벽하게 제어할 수 있는 MinIO Client(`mc`)라는 강력한 CLI 도구를 제공함
- 커리큘럼 내 데이터 파이프라인 연동 확인 시 매우 유용함

    ```bash
    # Linux 기준 mc 다운로드 및 설치
    wget https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x mc
    sudo mv mc /usr/local/bin/

    # 로컬에 뜬 MinIO 서버를 'local_s3'라는 별칭으로 등록
    mc alias set local_s3 http://localhost:9000 minioadmin minioadmin

    # 버킷 리스트 조회 테스트
    mc ls local_s3/
    ```

> - 이 단계까지 완료되면 로컬 PC에 완벽한 **프라이빗 AWS S3 대체 환경**이 마련된 것
> - 이제 이 인프라 위에 Apache Iceberg 테이블 포맷을 얹고, Trino 엔진을 연결하여 통합 레이크하우스 파이프라인을 구축해 나갈 수 있음
{: .common-quote}

### 3.5 로컬 사용 시의 이점

- **완벽한 비용 무료:**
    - 클라우드 크레딧이나 데이터 전송료(Egress) 걱정 없이 테라바이트급 테스트도 마음껏 할 수 있음

- **네트워크 독립성:**
    - 인터넷 연결이 불안정하거나 오프라인 상태에서도 개발 및 실습이 가능함

- **데이터 보안:**
    - 실습용 데이터가 외부 클라우드에 전송되지 않고 본인 디스크에만 저장됨

- **빠른 속도:**
    - 로컬 디스크 I/O를 사용하므로
    - 클라우드 S3보다 훨씬 빠른 응답 속도로 개발 피드백을 확인할 수 있음


### 3.6 S3 호환 실습 팁

- MinIO를 로컬에 띄워두고 코드(Python의 `boto3` 등)를 작성할 때, 딱 한 가지만 변경하면 됨
    - **Endpoint URL 설정:**
        - 기본적으로 S3 라이브러리는 AWS 주소를 바라보지만,
        - 이를 `http://localhost:9000`으로만 수정해 주면 됨

        ```python
        import boto3

        # AWS S3 대신 로컬 MinIO 연결
        s3 = boto3.client('s3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        ```

> - MinIO는
>   - Linux(Ubuntu/Mint) 환경이나 Windows 10 환경 어디서든 잘 돌아감
>   - 특히 **Linux 환경에서 Docker를 사용**하면, 실제 운영 환경과 거의 동일한 구성으로 실습할 수 있음<br>
    🡲 나중에 실제 프로젝트에 적용하시기에도 훨씬 수월할 것
> - 무료 크레딧 걱정 없이, 로컬에 MinIO를 통해 S3의 강력한 기능들을 마음껏 테스트할 수 있음
{: .summary-quote}