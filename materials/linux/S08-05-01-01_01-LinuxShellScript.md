---
layout: page
title:  "리눅스 쉘 스크립트"
date:   2025-02-27 09:00:00 +0900
permalink: /materials/S08-05-01-01_01-LinuxShellScript
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


> - **리눅스 쉘 스크립트(Shell Script)**
>   - 개발자와 데이터 엔지니어에게 **'반복 업무로부터의 자유'**를 선사하는 가장 강력한 무기
>   - 단순히 명령어를 나열하는 것을 넘어, 시스템을 자동화하고 복잡한 데이터 파이프라인을 구축함
{: .summary-quote}


## 1. 쉘 스크립트 개요

- 리눅스 쉘(Bash, Zsh 등)에서 실행되도록 작성된 **인터프리터 방식의 프로그램**
- 리눅스 커맨드라인에서 하나씩 입력하던 명령어들을 하나의 파일에 모아 순차적으로 실행하게 만든 일종의 '명령어 묶음'

- **의의와 목적**
    - **업무 자동화**
        - 수백 개의 로그 파일을 분석하거나, 매일 특정 시간에 DB를 백업하는 등 반복적인 작업을 자동화

    - **환경 구성의 일관성**
        - 서버를 새로 세팅할 때 필요한 패키지 설치 및 환경 설정을 스크립트 하나로 완벽하게 복제할 수 있음

    - **데이터 파이프라인의 접착제**
        - 서로 다른 언어(Python, Java 등)나 툴(Spark, Kafka) 사이에서 데이터를 전달하고 흐름을 제어하는 '접착제' 역할 수행


## 2. 쉘 스크립트 작성 방법과 학습 방향성

- **기본 작성 규칙**
    - **확장자:** 통상적으로 **`.sh`** 사용
    - **Shebang:** 파일의 최상단에 **`#!/bin/bash`**를 작성하여 이 파일을 실행할 인터프리터를 지정
    - **실행 권한:** 작성 후 **`chmod +x script.sh`** 명령으로 실행 권한을 부여해야 함

- **학습 방향성**
    - **변수와 매개변수:** 데이터를 동적으로 처리하는 법을 익힘
    - **제어문 (If, For, While):** 조건에 따라 동작을 분기하고 반복하는 법을 학습
    - **Exit Code:** 앞선 작업의 성공/실패 여부를 판단하여 안정적인 스크립트를 작성


## 3. 기초 예제

> - 쉘 스크립트는 **"내가 터미널에 치던 명령어를 순서대로 기록하고, 상황에 따라 판단(if)을 내리게 하는 것"**에서 시작함
> - 리눅스 쉘 스크립트 공부를 시작할 때 가장 좋은 방법은 **"변수, 입력, 조건문, 출력"**이라는 프로그래밍의 4대 요소를 한꺼번에 다루는 실용적인 예제를 만들어보는 것
{: .common-quote}


### 3.1 Welcome & System Checker 스크립트

- 실행 시 사용자의 이름을 묻고, 입력된 이름에 따라 인사를 건넨 뒤 현재 서버의 날짜와 디스크 사용량을 알려주는 **'나만의 시스템 안내 가이드'** 스크립트

1. **코드 작성**
    - **`nano hello.sh`** 또는 **`vi hello.sh`** 명령어로 파일을 열고 아래의 내용을 입력
        - 손에 익은 에디터를 사용하면 됨. 경험이 전혀 없다면 **`nano`** 사용을 추천함

    ```bash
    #//file: "hello.sh"
    #!/bin/bash

    # 1. 변수 설정 및 입력받기
    echo "안녕하세요! 성함이 어떻게 되시나요?"
    read user_name

    # 2. 조건문을 활용한 맞춤형 인사
    if [ "$user_name" = "admin" ]; then
        message="관리자님, 환영합니다. 시스템을 점검하겠습니다."
    else
        message="$user_name 님, 반갑습니다! 현재 시스템 정보입니다."
    fi

    # 3. 결과 출력 (변수 사용)
    echo "------------------------------------------"
    echo "$message"
    echo "오늘의 날짜: $(date)"
    echo "현재 접속 계정: $(whoami)"

    # 4. 시스템 명령어 실행 (디스크 사용량 확인)
    echo "------------------------------------------"
    echo "현재 디스크 사용량(핵심 요약):"
    df -h | grep '^/dev/'

    echo "------------------------------------------"
    echo "스크립트 실행이 완료되었습니다."
    ```

2. **💡 상세 설명 (핵심 포인트)**
    - **Shebang (#!/bin/bash)**
        - 스크립트의 맨 첫 줄에 위치함
        - 이 파일이 어떤 해석기(Shell)를 통해 실행될지 지정하는 약속
        - **`/bin/bash`**는 "이 파일은 bash 쉘로 실행해라"라는 의미

    - **read 명령어와 변수**
        - **`read user_name`**: 사용자가 키보드로 입력한 값을 **`user_name`**이라는 바구니(변수)에 넣기
        - **`$user_name`**: 변수에 담긴 내용을 꺼내 쓸 때는 이름 앞에 **`$`** 기호를 붙임

    - **조건문 (if [ condition ]; then ... fi)**
        - 리눅스 쉘에서 조건문은 **`[`** 와 **`]`** 사이에 공백이 반드시 있어야 함
        - **`=`**: 문자열이 같은지 비교
        - **`fi`**: **`if`**문의 끝을 알리는 표시 (if를 거꾸로 쓴 것)

    - **명령어 치환 ($(command))**
        - **`$(date)`**: 
            - 텍스트 중간에 리눅스 명령어의 실행 결과를 그대로 넣고 싶을 때 사용 
            - **`date`** 명령어가 실행된 결과값이 그 자리에 쏙 들어감

    - **파이프라인과 필터 (df -h | grep ...)**
        - **`df -h`**: 디스크 용량을 사람이 보기 편하게(Human-readable) 출력
        - **`| grep '^/dev/'`**: 전체 출력 결과 중 실제 물리 장치인 **`/dev/`**로 시작하는 줄만 골라서 보여줌

3. **스크립트 실행 방법**
    - 스크립트를 만든 후 바로 실행하면 "Permission denied"라는 에러가 뜸 🡲 실행 권한을 주어야 함

    -  **실행 권한 부여:**

        ```bash
        chmod +x hello.sh
        ```

    -  **스크립트 실행:**

        ```bash
        ./hello.sh
        ```


### 3.2 반복문 (For Loop): 여러 파일 일괄 처리

- 데이터 엔지니어링에서 가장 흔한 작업은 특정 폴더 내의 수많은 파일을 하나씩 순회하며 처리하는 것
- **`for`**문을 사용해서 특정 디렉토리에 있는 파일 목록을 하나씩 출력해보기
- 학습 포인트 🡲 대량의 데이터를 자동 처리하는 **'루프'**의 원리 이해

1. **코드 작성:**
    - 로그 파일 확장자 일괄 변경

    ```bash
    #//file: "loop.sh"
    #!/bin/bash

    # 현재 디렉토리의 모든 .txt 파일을 찾아서 처리
    echo "파일 변환 작업을 시작합니다..."

    for file in *.txt; do
        # 파일이 실제로 존재하는지 체크 (파일이 없을 경우 대비)
        if [ -e "$file" ]; then
            filename=$(basename "$file" .txt)
            mv "$file" "${filename}.log"
            echo "$file -> ${filename}.log 변환 완료"
        fi
    done

    echo "모든 작업이 끝났습니다."
    ```

2. **💡 상세 설명 (핵심 포인트)**
    - **`for file in *.txt; do`**:
        - 현재 디렉토리에서 `.txt`로 끝나는 모든 파일명을
        - 하나씩 꺼내어 `file`이라는 변수에 담고 반복
    - **`basename "$file" .txt`**
        - 파일명에서 경로와 확장자(`.txt`)를 떼어내고
        - 순수 이름만 추출
    - **`${filename}.log`**
        - 변수명 뒤에 바로 문자를 붙일 때는 `{}`를 사용하여 변수의 범위를 명확히 지정하는 것이 안전함


### 3.3 숫자 계산 (Arithmetic): 산술 연산 처리

- 리눅스 쉘은 기본적으로 모든 것을 '문자열'로 취급함 🡲 따라서 숫자를 계산할 때는 특수한 문법이 필요함
- 사용자가 입력한 두 숫자를 더해서 출력해보기 (쉘 스크립트의 산술 연산 **`(( ... ))`** 공부)
- 학습 포인트 🡲 텍스트와 숫자를 구분하여 다루는 **'데이터 타입'**의 개념 익히기

1. **코드 작성:**
    - 간단한 평균 점수 계산기

    ```bash
    #//file: "arithmetic.sh"
    #!/bin/bash

    echo "수학 점수를 입력하세요:"
    read math
    echo "영어 점수를 입력하세요:"
    read eng

    # (( )) 내부에서 산술 연산을 수행
    (( sum = math + eng ))
    # 소수점 계산을 위해 bc(Basic Calculator) 도구 사용
    avg=$(echo "scale=2; $sum / 2" | bc)

    echo "--------------------------"
    echo "총점: $sum"
    echo "평균: $avg"
    ```

2. **💡 상세 설명 (핵심 포인트)**
    - **`(( ... ))`**: 
        - 쉘에서 정수형 산술 연산을 수행할 때 사용
        - 이 안에서는 변수 앞에 **`$`**를 붙이지 않아도 인식이 가능함

    - **`bc` (Basic Calculator)**:
        - 쉘 자체는 소수점 계산에 약함
        - **`scale=2`**는 소수점 둘째 자리까지 표시하라는 설정
        - 파이프(**`|`**)를 통해 계산식을 **`bc`** 명령어로 전달하여 정밀한 값을 얻음


### 3.4 에러 처리 (Error Handling): 안정적인 스크립트

- 실무용 스크립트는 예상치 못한 상황(파일 없음, 권한 부족 등)에서 갑자기 멈추지 않고 적절한 안내를 제공해야 함
- 만약 사용자가 이름을 입력하지 않고 엔터를 치면 "이름을 입력해주세요"라고 다시 묻는 로직 추가하기
- 학습 포인트 🡲 어떤 상황에서도 시스템이 안전하게 동작하게 만드는 **'예외 처리'** 마인드 확립

1. **코드 작성:**
    - 디렉토리 접근 및 파일 확인

    ```bash
    #//file: "error_handling.sh"
    #!/bin/bash

    TARGET_DIR="./data_folder"

    # 1. 디렉토리 존재 여부 확인
    if [ ! -d "$TARGET_DIR" ]; then
        echo "에러: $TARGET_DIR 디렉토리가 존재하지 않습니다." >&2
        exit 1
    fi

    # 2. 이동 시도 및 성공 여부 확인
    cd "$TARGET_DIR" || { echo "에러: 디렉토리로 이동할 수 없습니다."; exit 1; }

    # 3. 파일 목록 확인
    files=$(ls)
    if [ -z "$files" ]; then
        echo "주의: 디렉토리가 비어 있습니다."
    else
        echo "현재 파일 목록:"
        echo "$files"
    fi
    ```

2. **💡 상세 설명 (핵심 포인트)**
    - **`>&2`**: 
        - 에러 메시지를 '표준 에러(stderr)' 스트림으로 보냄
        - 이는 로그를 남길 때 일반 출력과 에러를 분리하기 위함

    - **`exit 1`**:
        - 스크립트를 즉시 종료하며 '비정상 종료(1)' 상태 코드를 반환
        - 외부 시스템(Cron, Jenkins 등)에서 이 코드를 보고 작업 실패를 감지

    - **`||` (OR 연산자)**:
        - 앞의 명령어(`cd`)가 실패했을 때만 뒤의 중괄호(`{ }`) 내용을 실행하라는 의미
        - 매우 직관적인 에러 처리 방식

    - **`-z "$files"`**:
        - 변수가 비어있는지(Zero length) 확인하는 옵션



## 4. 핵심 실무 예제 및 상세 설명

- **예제 1: 데이터 백업 및 로그 관리 (데이터 엔지니어 기초)**
    - 특정 디렉토리의 로그 파일을 압축하여 날짜별로 저장하고, 오래된 파일은 삭제하는 스크립트

    ```bash
    #//file: "backup_log.sh"
    #!/bin/bash

    # 1. 변수 설정
    SOURCE_DIR="/var/log/myapp"
    BACKUP_DIR="/home/user/backups"
    DATE=$(date +%Y-%m-%d)
    BACKUP_FILE="log_backup_$DATE.tar.gz"

    # 2. 백업 디렉토리가 없으면 생성
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        echo "백업 디렉토리를 생성했습니다: $BACKUP_DIR"
    fi

    # 3. 로그 파일 압축 (표준 에러는 에러 로그 파일로 따로 저장)
    tar -czf "$BACKUP_DIR/$BACKUP_FILE" "$SOURCE_DIR"/*.log 2> "$BACKUP_DIR/error.log"

    # 4. 결과 확인 (Exit Code 사용)
    if [ $? -eq 0 ]; then
        echo "백업 성공: $BACKUP_FILE"
    else
        echo "백업 실패! 에러 로그를 확인하세요."
        exit 1
    fi

    # 5. 7일이 지난 백업 파일은 자동 삭제 (관리 효율성)
    find "$BACKUP_DIR" -type f -name "*.tar.gz" -mtime +7 -delete
    ```

    - **💡 상세 설명:**
        - **`DATE=$(date +%Y-%m-%d)`**: 시스템 날짜를 변수에 할당하여 매일 다른 파일명 생성
        - **`if [ ! -d ... ]`**: 디렉토리 존재 여부를 체크하는 조건문
        - **`$?`**: 마지막으로 실행된 명령어의 결과 코드. **0은 성공**, 그 외는 실패를 의미
        - **`find ... -delete`**: 시스템 디스크 용량이 가득 차는 것을 방지하는 실무 필수 로직


- **예제 2: 서비스 상태 모니터링 (개발 및 운영 기초)**
    - 특정 프로세스가 살아있는지 확인하고, 죽어있다면 자동으로 재시작하는 스크립트

    ```bash
    #//file: "monitoring.sh"
    #!/bin/bash

    PROCESS_NAME="nginx"

    # 프로세스 개수를 세어 확인
    COUNT=$(ps -ef | grep -v "grep" | grep "$PROCESS_NAME" | wc -l)

    if [ $COUNT -gt 0 ]; then
        echo "$(date): $PROCESS_NAME 서비스가 정상 작동 중입니다."
    else
        echo "$(date): $PROCESS_NAME 서비스가 중단되었습니다. 재시작을 시도합니다."
        sudo systemctl start "$PROCESS_NAME"
        
        # 재시작 후 결과 알림 (가상 시나리오: 슬랙 전송 등)
        if [ $? -eq 0 ]; then
            echo "서비스 재시작 성공"
        else
            echo "서비스 재시작 실패! 관리자의 확인이 필요합니다."
        fi
    fi
    ```

    - **💡 상세 설명:**
        - **`ps -ef | grep ...`**: 실행 중인 프로세스 목록에서 특정 이름 검색
        - **`grep -v "grep"`**: **`grep`** 명령어 자체도 프로세스 목록에 나타나기 때문에 이를 결과에서 제외함
        - **`wc -l`**: 출력된 결과의 줄 수를 세어 프로세스 존재 여부를 숫자로 반환


- **예제 3: 텍스트 처리 및 데이터 추출 (Data Engineer 필수)**
    - 데이터 엔지니어링의 기본은 '원시 데이터(Raw Data)'를 가공하는 것
    - 리눅스의 강력한 텍스트 처리 도구인 `awk`, `sed`, `grep`을 조합하여 대용량 CSV 파일이나 로그 파일에서 원하는 데이터만 추출하는 예제

    - **목적:** 대용량 파일에서 조건에 맞는 행만 필터링하고 특정 컬럼의 데이터를 재가공하기
    - **의의:** Python이나 Pandas를 띄우기엔 부담스러운 수십 GB의 텍스트 파일을 서버 단에서 전처리할 때 쉘 스크립트가 압도적으로 빠름

    ```bash
    #//file: "data_parser.sh"
    #!/bin/bash

    # 샘플 데이터 파일명
    INPUT_FILE="sales_data.csv"
    OUTPUT_FILE="filtered_sales.csv"

    # 1. 샘플 데이터 생성 (실습용)
    echo "Date,Product,Price,Status" > $INPUT_FILE
    echo "2023-10-01,Apple,1000,SUCCESS" >> $INPUT_FILE
    echo "2023-10-01,Banana,500,FAIL" >> $INPUT_FILE
    echo "2023-10-02,Orange,1200,SUCCESS" >> $INPUT_FILE
    echo "2023-10-03,Grape,2000,PENDING" >> $INPUT_FILE

    echo "📊 데이터 추출을 시작합니다..."

    # 2. awk를 활용한 데이터 필터링 (Status가 SUCCESS인 항목만 추출하여 가격을 10% 인상)
    # $0: 전체 행, $1~4: 각 컬럼 (구분자 ,)
    awk -F ',' '
    BEGIN { 
        print "Date,Product,NewPrice,Status" > "'$OUTPUT_FILE'" 
    }
    NR > 1 { 
        if ($4 == "SUCCESS") {
            new_price = $3 * 1.1;
            printf "%s,%s,%d,%s\n", $1, $2, new_price, $4 >> "'$OUTPUT_FILE'"
        }
    }' $INPUT_FILE

    echo "✅ 처리가 완료되었습니다. 결과 파일: $OUTPUT_FILE"
    cat $OUTPUT_FILE

    ```


- **예제 4: REST API 호출 및 JSON 파싱 (Developer & DE 필수)**
    - 현대 소프트웨어 개발은 API 통신의 연속
    - `curl`을 이용해 외부 API 데이터를 가져오고, `jq` (JSON 파서)를 이용해 쉘 스크립트 내에서 JSON 데이터를 파싱하는 방법은 매우 중요함
    - **목적:** `curl`로 HTTP 요청을 보내고, 응답받은 JSON 구조에서 원하는 값만 추출하기
    - **의의:** 슬랙(Slack) 알림 봇 생성, 날씨 데이터 수집 배치 작업, CI/CD 파이프라인에서 빌드 상태 체크 등 무궁무진하게 활용됨

    ```bash
    #//file: "api_fetcher.sh"
    #!/bin/bash

    # 사전에 jq 패키지 설치 필요 (sudo apt-get install jq)

    API_URL="https://jsonplaceholder.typicode.com/users"

    echo "🌐 외부 API에서 사용자 데이터를 가져옵니다..."

    # 1. curl로 데이터 가져오기 (-s 옵션으로 진행 상태 숨김)
    RESPONSE=$(curl -s $API_URL)

    # 2. 응답이 정상인지 확인 (JSON 배열 길이 체크)
    USER_COUNT=$(echo $RESPONSE | jq '. | length')

    if [ "$USER_COUNT" -gt 0 ]; then
        echo "총 ${USER_COUNT}명의 데이터를 성공적으로 가져왔습니다."
        echo "----------------------------------------"
        
        # 3. jq를 사용해 이름(name)과 이메일(email)만 추출하여 반복문 돌리기
        echo "$RESPONSE" | jq -c '.[] | {name: .name, email: .email}' | while read user; do
            NAME=$(echo $user | jq -r '.name')
            EMAIL=$(echo $user | jq -r '.email')
            echo "👤 이름: $NAME | 📧 이메일: $EMAIL"
        done
    else
        echo "❌ 데이터를 가져오는 데 실패했습니다."
        exit 1
    fi

    ```


- **예제 5: 프로젝트 환경 자동 셋업 (Developer 필수)**
    - 새로운 서버나 팀원의 로컬 PC에 프로젝트를 세팅할 때,
    - 필요한 프로그램이 설치되어 있는지 검사하고 디렉토리 구조 및 환경 설정 파일(**`.env`**)을 자동으로 생성해 주는 스크립트
    - **목적:** 의존성 프로그램(Python, Docker 등) 설치 여부 검증 및 초기 디렉토리 구조화
    - **의의:** DevOps 및 백엔드 개발자가 CI/CD 파이프라인을 구축하거나 배포 자동화를 할 때 가장 먼저 작성하게 되는 형태의 스크립트

    ```bash
    #//file: "project_bootstrap.sh"
    #!/bin/bash

    PROJECT_NAME="my_new_service"
    REQUIRED_TOOLS=("python3" "docker" "git")

    echo "🚀 [$PROJECT_NAME] 프로젝트 초기 환경 설정을 시작합니다."

    # 1. 필수 도구 설치 여부 검사 (command -v 활용)
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! command -v $tool &> /dev/null; then
            echo "❌ 에러: $tool 이(가) 설치되어 있지 않습니다. 설치 후 다시 실행해주세요."
            exit 1
        else
            echo "✅ $tool 확인 완료"
        fi
    done

    # 2. 프로젝트 디렉토리 생성 (이미 존재하면 건너뜀)
    mkdir -p "$PROJECT_NAME"/{src,logs,config}
    echo "📁 디렉토리 구조 생성 완료."

    # 3. 기본 .env 파일 생성
    ENV_FILE="$PROJECT_NAME/config/.env"
    if [ ! -f "$ENV_FILE" ]; then
        cat <<EOF > "$ENV_FILE"
    # [$PROJECT_NAME] Environment Variables
    DB_HOST=localhost
    DB_PORT=5432
    API_KEY=YOUR_API_KEY_HERE
    EOF
        echo "📄 기본 $ENV_FILE 파일이 생성되었습니다."
    else
        echo "⚠️ $ENV_FILE 파일이 이미 존재합니다. 덮어쓰지 않습니다."
    fi

    echo "🎉 모든 셋업이 완료되었습니다. 'cd $PROJECT_NAME' 명령어로 이동하세요."

    ```

<br><br>

> - **쉘 스크립트 실력 향상을 위한 조언**
>   - **한 줄씩 실행해보기:**
>       - 스크립트 파일로 만들기 전에 터미널에서 명령어를 직접 입력하며 결과 확인
>   - **공통 기능은 함수화:**
>       - 반복되는 코드는 **`function_name() { ... }`** 형태로 묶어 재사용성 향상
>   - **주석의 생활화:**
>       - 쉘 스크립트는 시간이 지나면 본인이 짠 코드도 이해하기 어려울 수 있음
>       - 각 로직의 이유를 반드시 주석(**`#`**)으로 남길것
{: .expert-quote}