---
layout: page
title:  "파이썬 기반 크롤러"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S02-02-01-02_01-PythonBasedCrawler
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}



## 1. 크롤링(Crawling)의 개념과 기술적 특징

### 1.1 개념 정의

- **웹 크롤링(Web Crawling)**
    - 조직적, 자동화된 방법으로 웹 페이지를 탐색하며 링크를 수집하고 색인(Indexing)하는 과정
    - 예: 구글 봇의 웹 페이지 순회

- **웹 스크래핑(Web Scraping)**
    - 특정 웹 페이지에서 우리가 **원하는 특정 데이터(가격, 제목, 이미지 등)를 추출**하여 구조화된 데이터로 저장하는 기법

### 1.2 주요 기술 스택 분류

- **정적 페이지 데이터 추출 (HTTP 통신)**
    - `requests`, `urllib`
        - 서버에 HTTP Request를 보내고 HTML 소스를 응답받는 라이브러리
    - `BeautifulSoup`
        - 받아온 HTML 문자열을 파싱(Parsing)하여 DOM 트리 구조로 변환, 데이터 추출을 용이하게 해주는 라이브러리

- **동적 페이지 및 SPA(Single Page Application) 대응**
    - `Selenium`, `Playwright`
        - 브라우저를 직접 원격 제어하여 JavaScript 렌더링, 클릭, 스크롤 등의 사용자 상호작용을 시뮬레이션하는 라이브러리


## 2. 크롤링 기술의 장단점 및 활용 사례

- **장점**
    - **데이터 확보의 무한성:** 공개된 웹의 방대한 데이터를 자산화할 수 있음
    - **자동화 및 효율성:** 수작업으로 수일이 걸릴 데이터 수집을 수분 내에 완료
    - **실시간성:** 주기적인 스크래핑을 통해 시장의 실시간 변화를 모니터링할 수 있음

- **단점 및 한계**
    - **대상 사이트 의존성 :** 타겟 웹사이트의 UI/UX나 HTML 구조가 조금만 바뀌어도 크롤러가 작동하지 않음 🡲 **유지보수 비용 발생**
    - **차단 및 제재 위험 :** 짧은 시간 내 많은 요청을 보내면 IP가 차단되거나 디도스(DDoS) 공격으로 오인받을 수 있음
    - **비정형 데이터 처리의 난해함 :** HTML 내 텍스트 뒤섞임, 숨겨진 데이터 등을 정제하는 데 많은 리소스가 소요됨

- **주요 활용 사례**
    - **이커머스 :** 경쟁사 상품 가격 비교 및 최저가 모니터링 시스템 구축
    - **금융/주식 :** 뉴스 키워드 감성 분석 및 기업 공시 정보 자동 수집을 통한 투자 분석
    - **제조/스마트팩토리 :** 공급망 부품 가격 변동성 추적 및 원자재 시장 동향 리포트 자동화
    - **AI/데이터 과학 :** 거대 언어 모델(LLM) 학습용 말뭉치 및 파인튜닝 데이터셋 확보


## 3. 라이선스 및 법적 주의점 (Compliance)

> ⚠️ **과거와 달리 현재는 무분별한 크롤링에 대한 법적 처벌 사례가 늘고 있음**
{: .expert-quote}

- **`robots.txt` 확인 (필수)**
    - 웹사이트 루트 경로(예: `https://example.com/robots.txt`)에 위치한 로봇 배제 표준을 반드시 확인하고 준수해야 함

- **서비스 이용약관(Terms of Service) 위반 여부**
    - 로그인이 필요한 서비스(Gated Data)의 경우,
        - 로그인 시 동의한 약관에 "자동화된 수단 이용 금지" 조항이 있다면 민사상 손해배상 청구 대상이 될 수 있음

- **저작권법 및 정보통신망법 유의**
    - 단순한 '사실(Facts, 예: 상품 가격, 수치)'은 저작권 보호 대상이 아니지만, 타인이 가공한 '창작성 있는 저작물(리뷰 글, 기사, 독창적 이미지 등)'을 무단으로 긁어가서 상업적으로 재배포하면 **저작권 침해**에 해당함
    - 서버에 무리를 주어 서비스를 마비시키면 정보통신망법 위반(컴퓨터장애업무방해)으로 형사 처벌을 받을 수 있음

- **개인정보보호법(개인식별정보 PII)**
    - 공개된 정보라 할지라도 이름, 전화번호, 이메일 주소 등을 대량으로 긁어 모으는 행위는 국내 개인정보보호법 및 글로벌 standard(GDPR, CCPA)에 전면 위반됨


## 4. 최근 업계의 반응 및 기술적 대응 트렌드

- ❌ **웹사이트 측의 방어 (Anti-Scraping) 고도화**

* **Cloudflare / Akamai 등 보안 솔루션:** 단순 IP 차단을 넘어 브라우저 핑거프린팅, 행동 패턴 분석을 통해 봇(Bot)을 실시간 차단합니다.
* **AI 기반 CAPTCHA:** 사람이 아닌 것으로 의심되면 한 차원 높은 캡차(CAPTCHA)를 요구합니다.
* **`ai.txt` 및 `llms.txt` 등장:** 최근 생성형 AI 붐으로 인해, 기존 `robots.txt`를 넘어 "AI 모델 학습용 데이터 수집(TDM)을 거부한다"는 목적 기반 제어(Purpose-Based Control) 규격이 업계 표준으로 자리 잡고 있습니다.

- ⭕ **크롤러 측의 진화 (Harness & Agentic Loop)**

* **헤드리스 브라우저의 고도화:** `Playwright` 등을 활용해 완벽하게 인간의 마우스 움직임과 스크롤 속도를 흉내 내는 방식이 주류를 이룹니다.
* **LLM 연동 스크래핑:** 과거에는 정규식이나 CSS Selector를 꼼꼼히 짜야 했지만, 최근에는 HTML 통째로 혹은 텍스트 스냅샷을 LLM 지시어(Prompt)에 넣어 원하는 정보만 JSON 구조로 뽑아내는 '에이전트형 스크래핑(Agentic Scraping)'으로 패러다임이 이동하고 있습니다.

---

## 5. 강의 실습용 예제 코드 및 설명

실습 코드는 강사들이 수업 환경에서 실행했을 때 오류가 없도록 안정성이 검증된 **네이버 뉴스 검색 결과 수집**을 예시로 합니다. (BeautifulSoup 기반 정적 크롤링)

### 🛠️ 사전 준비 (터미널 설치 명령어)

```bash
pip install requests beautifulsoup4

```

### 💻 실습 코드 (`naver_news_scraper.py`)

```python
import sys
import time
import requests
from bs4 import BeautifulSoup


def scrape_naver_news(search_keyword, max_pages=2):
    """네이버 뉴스에서 키워드를 검색하여 제목과 링크를 수집하는 함수

    :param search_keyword: 검색할 키워드 (str)
    :param max_pages: 수집할 페이지 수 (int)
    """
    print(f"[{search_keyword}] 관련 뉴스 수집을 시작합니다. (최대 {max_pages}페이지)")
    print("-" * 60)

    # 1. 크롤링 시 차단을 방지하기 위한 필수 헤더(User-Agent) 설정
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    results_count = 0

    for page in range(max_pages):
        # 네이버 뉴스 검색 페이징 공식: 1페이지(start=1), 2페이지(start=11), 3페이지(start=21) ...
        start_val = page * 10 + 1
        url = f"https://search.naver.com/search.naver?where=news&query={search_keyword}&start={start_val}"

        try:
            # 2. HTTP GET 요청 보냄
            response = requests.get(url, headers=headers, timeout=5)

            # 응답 상태 코드가 200(정상)이 아니면 예외 발생
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"📢 [에러] 페이지 요청 중 문제가 발생했습니다: {e}")
            break

        # 3. HTML 파싱
        soup = BeautifulSoup(response.text, "html.parser")

        # 4. 뉴스 기사 블록 요소 선택 (네이버 검색 개편 후 기준 최신 Selector 기재)
        # 각 뉴스 기사 카드는 'bx' 클래스를 가진 'li' 태그 내에 존재함
        news_items = soup.select("ul.list_news > li.bx")

        if not news_items:
            print(f"⚠️ {page+1}페이지에서 검색 결과 요소를 찾을 수 없습니다. 구조 변경을 확인하세요.")
            break

        print(f"📄 [진행] {page+1} 페이지 수집 중...")

        for item in news_items:
            # 뉴스 제목과 링크가 있는 <a> 태그 선택 (news_tit 클래스)
            title_element = item.select_one("a.news_tit")

            if title_element:
                title = title_element.get_text(strip=True)
                link = title_element["href"]

                results_count += 1
                print(f"{results_count}. {title}")
                print(f"   🔗 링크: {link}")

        # 5. 과도한 트래픽 유발 방지를 위한 정중한 매너 (Politeness Delay)
        # 서버에 부담을 주지 않기 위해 페이지 전환 시 1.5초간 대기
        time.sleep(1.5)

    print("-" * 60)
    print(f"✅ 수집 완료: 총 {results_count}개의 뉴스 데이터를 안전하게 수집했습니다.")


if __name__ == "__main__":
    # 인코딩 문제 방지 (윈도우 환경 대응)
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

    # 실습 키워드 설정 (예: AI 기술)
    keyword = "인공지능 트렌드"
    scrape_naver_news(keyword, max_pages=2)

```

### 📝 코드 구현 설명 (수업 진행 핵심 포인트)

1. **`User-Agent` 설정의 중요성:** 브라우저가 아닌 파이썬 코드로 직접 요청을 보내면 서버 측에서는 `Python-requests/x.x.x`라는 서명을 보고 즉시 봇으로 판단하여 403 Forbidden 에러를 던집니다. 이를 방지하기 위해 실제 크롬 브라우저인 것처럼 신분증(`User-Agent`)을 위장하는 법을 지도해야 합니다.
2. **`raise_for_status()` 예외 처리:** 네트워크 단절이나 대상 서버 다운 등 예외 상황 발생 시 스크립트가 비정상 종료되는 것을 막고 에러 로그를 남기는 방어적 코딩 기법입니다.
3. **`time.sleep()`의 의무화:** 크롤링 속도가 너무 빠르면 대상 서버 방어 시스템이 작동해 IP가 즉시 벤(Ban)당할 수 있습니다. 매너 있는 크롤러의 기본 소양으로 '딜레이 부여'를 꼭 가르치셔야 합니다.