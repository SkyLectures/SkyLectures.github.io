---
layout: page
title: (실습) 데이터 분석 Skill 개발
subtitle: 
permalink: /materials/S06-05-03-05_01-DataAnalysisSkills(Python)
description: 반복되는 공정 데이터 분석 작업 및 사업재편 보고서 수집을 자동화하는 Skill 제작
categories:
    - materials
tags:
    - AI 활용 및 자동화
    - 공정 개선
    - AI Utilization and Automation
    - Process Improvement
related_posts:
    - /materials/S06-05-03-01_01-DataDrivenBizOptimization
    - /materials/S06-05-03-02_01-BizRestrProcPredictionModels
    - /materials/S06-05-03-03_01-NoCodePredictionModels
    - /materials/S06-05-03-04_01-AiAgentsPoweredByCuttingEdgeAi
    - /materials/S06-05-03-06_01-BizRestrProcImprovementScenarios
qna: true
---
* toc
{:toc}




## 1. 실습 개요 및 실습 환경

- **실습 개요**
    - 사업재편 과정에서는 신규 제품 라인의 공정 조건(온도, 압력 등)과 품질 지표(수율, 불량률)를 매일 점검하고, 경영진 보고용 실적 요약을 취합해야 함<br>
        🡪 실무진은 매일 CSV 파일을 열어 엑셀 피벗을 돌리고 불량 원인을 대조한 뒤 보고서 양식에 복사·붙여넣기하는 단순 반복 작업에 막대한 시간을 소모함
    - 본 실습에서는 LLM(거대 언어 모델)이 스스로 판단하여 실행할 수 있는 2가지 전용 도구(Skill)를 제작

        1. **공정 데이터 분석 Skill (`analyze_process_yield`):**
            - 일일 공정 배치 데이터를 읽어
            - 수율 지표, 불량 상위 원인, 최적 파라미터 이탈 여부를 자동 산출하는 스킬
        2. **사업재편 진척 보고서 수집/생성 Skill (`generate_restructuring_report`):**
            - 분석된 수치와 현장 메모를 결합하여
            - 규격화된 Markdown 경영 보고서를 자동 작성하는 스킬

    - 최종적으로 사용자가 "오늘 2라인 공정 데이터 분석하고 사업재편 보고서 뽑아줘"라는 한 문장의 자연어를 입력하면,<br>
        🡪 AI 에이전트가 두 스킬을 순차적으로 호출하여 분석과 보고서 완성을 자율 수행하는 파이프라인을 완성<br><br>

- **실습 환경**
    - **실행 환경:** Python 3.10 이상 (Google Colab, 로컬 Jupyter Notebook 또는 VS Code 터미널)
    - **필수 라이브러리:** `pandas`, `numpy` (순수 Python 표준 라이브러리와 데이터 조작 라이브러리만으로 구현하여 외부 유료 API 종속 없이 로컬/노코드 환경 어디서나 즉시 동작하도록 구성)

<br>

## 2. 실습 데이터셋 구조 및 생성 파이썬 코드

- **데이터셋 구조 (`daily_process_log.csv`)**
    - 신규 친환경 배터리 팩 하우징 라인(사업재편 신규 라인)의 당일 120개 배치(Batch) 생산 이력 로그

        | 컬럼명         | 데이터 타입     | 설명 및 단위                   | 정상 기준치 (SOP)                  |
        | :------------- | :-------------- | :----------------------------- | :--------------------------------- |
        | batch_id       | 문자열 (String) | 배치 식별 코드 (예: BATCH_001) | 고유 번호                          |
        | line_no        | 문자열 (String) | 생산 라인 (Line-A, Line-B)     | 신규 사업재편 라인 (Line-B)        |
        | die_temp       | 실수 (Float)    | 금형 가열 온도 (℃)             | **215.0 ~ 225.0 ℃**                |
        | press_pressure | 실수 (Float)    | 성형 프레스 압력 (ton)         | **160.0 ~ 170.0 ton**              |
        | cool_rate      | 실수 (Float)    | 급랭 냉각 속도 (℃/s)           | **18.0 ~ 22.0 ℃/s**                |
        | yield_rate     | 실수 (Float)    | 배치 실측 수율 (%)             | **목표 90.0% 이상**                |
        | defect_type    | 문자열 (String) | 주요 결함 유형                 | None(정상), Crack, UnderFill, Burr |
        | shift          | 문자열 (String) | 근무 조 (Day, Night)           | 주간/야간                          |

- **데이터셋 생성 파이썬 코드**

    ```python
    import numpy as np
    import pandas as pd

    np.random.seed(104)
    n_batches = 120

    batch_ids = [f"BATCH_{i+1:03d}" for i in range(n_batches)]
    line_nos = np.random.choice(["Line-A (기존)", "Line-B (신규재편)"], size=n_batches, p=[0.3, 0.7])
    shifts = np.random.choice(["Day", "Night"], size=n_batches, p=[0.55, 0.45])

    # 공정 파라미터 생성 (일부 이상치 및 편차 주입)
    die_temps = np.random.normal(220, 6.5, n_batches).round(1)
    press_pressures = np.random.normal(164, 5.0, n_batches).round(1)
    cool_rates = np.random.normal(20.5, 3.2, n_batches).round(1)

    yield_rates = []
    defect_types = []

    for temp, press, cool in zip(die_temps, press_pressures, cool_rates):
        penalty = 0
        defect = "None"
        
        # 공정 물리 룰 기반 불량 유발
        if temp < 212.0:
            penalty += (212.0 - temp) * 1.8
            defect = "UnderFill (미성형)"
        elif temp > 228.0:
            penalty += (temp - 228.0) * 2.1
            defect = "Crack (균열)"
            
        if press < 157.0:
            penalty += (157.0 - press) * 1.5
            if defect == "None":
                defect = "UnderFill (미성형)"
        elif press > 173.0:
            penalty += (press - 173.0) * 1.6
            if defect == "None":
                defect = "Burr (버 발생)"
                
        base_yield = 95.5 - penalty + np.random.normal(0, 1.2)
        base_yield = round(float(np.clip(base_yield, 60.0, 99.2)), 1)
        yield_rates.append(base_yield)
        
        if base_yield >= 90.0:
            defect_types.append("None")
        else:
            defect_types.append(defect if defect != "None" else "Crack (균열)")

    df = pd.DataFrame({
        "batch_id": batch_ids,
        "line_no": line_nos,
        "shift": shifts,
        "die_temp": die_temps,
        "press_pressure": press_pressures,
        "cool_rate": cool_rates,
        "yield_rate": yield_rates,
        "defect_type": defect_types
    })

    df.to_csv("S06-05-03-05_01-daily_process_log.csv", index=False, encoding="utf-8-sig")
    print("실습용 데이터셋 생성 완료: S06-05-03-05_01-daily_process_log.csv (120건)")
    ```

<br>

## 3. 실습: 2가지 전용 Skill 및 실행기 구현

- 에이전트가 활용하는 스킬(Skill)은 다음 3가지 요소를 갖추어야 함

    1. **명확한 기능 함수 (실행 로직)**
    2. **입출력 데이터 타입과 상세한 설명서 (Docstring / Description)**
    3. **에이전트가 이해할 수 있는 정형화된 JSON 반환 구조**

- **1단계: [Skill 1] 공정 데이터 분석 스킬 제작**

    ```python
    import json
    import pandas as pd

    def analyze_process_yield(csv_path: str, target_line: str = "Line-B (신규재편)") -> str:
        """
        [Skill 1: 공정 데이터 분석 도구]
        지정된 공정 로그 CSV 파일에서 특정 생산 라인의 수율 통계,
        목표 수율(90%) 달성률, 주요 불량 원인 및 파라미터 이탈 현황을 분석합니다.
        
        Args:
            csv_path (str): 공정 로그 CSV 파일 경로
            target_line (str): 분석 대상 라인명 (기본값: 'Line-B (신규재편)')
        Returns:
            str: JSON 포맷의 분석 결과 요약 문자열
        """
        df = pd.read_csv(csv_path)
        
        # 1. 대상 라인 필터링
        target_df = df[df["line_no"] == target_line]
        if target_df.empty:
            return json.dumps({"error": f"{target_line}에 해당하는 데이터가 없습니다."})
            
        total_batches = len(target_df)
        avg_yield = round(target_df["yield_rate"].mean(), 2)
        pass_batches = len(target_df[target_df["yield_rate"] >= 90.0])
        fail_batches = total_batches - pass_batches
        achievement_rate = round((pass_batches / total_batches) * 100, 1)
        
        # 2. 불량 유형 분석 (수율 90% 미만 건)
        defects = target_df[target_df["yield_rate"] < 90.0]["defect_type"].value_counts().to_dict()
        
        # 3. 파라미터 이탈 분석 (온도 기준: 215~225℃)
        temp_under = int((target_df["die_temp"] < 215.0).sum())
        temp_over = int((target_df["die_temp"] > 225.0).sum())
        press_deviations = int(((target_df["press_pressure"] < 160.0) | (target_df["press_pressure"] > 170.0)).sum())
        
        result = {
            "target_line": target_line,
            "total_batches": total_batches,
            "average_yield": avg_yield,
            "target_achievement_rate": achievement_rate,
            "pass_count": pass_batches,
            "fail_count": fail_batches,
            "defect_breakdown": defects,
            "parameter_anomalies": {
                "temp_under_215C": temp_under,
                "temp_over_225C": temp_over,
                "pressure_out_of_spec": press_deviations
            }
        }
        return json.dumps(result, ensure_ascii=False)
    ```

<br>

- **2단계: [Skill 2] 사업재편 진척 보고서 수집 및 생성 스킬 제작**

    ```python
    from datetime import datetime

    def generate_restructuring_report(analysis_json_str: str, author_note: str = "") -> str:
        """
        [Skill 2: 사업재편 일일 보고서 생성 도구]
        'analyze_process_yield' 스킬에서 생성된 분석 JSON 데이터를 받아
        경영진 보고용 규격화된 Markdown 일일 보고서로 변환합니다.
        
        Args:
            analysis_json_str (str): 분석 결과 JSON 문자열
            author_note (str): 현장 관리자의 특이사항 메모
        Returns:
            str: 작성된 Markdown 보고서 전문
        """
        data = json.loads(analysis_json_str)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 불량 요약 문자열 조립
        defect_lines = []
        for defect, count in data.get("defect_breakdown", {}).items():
            defect_lines.append(f"  * **{defect}:** {count}건")
        defects_text = "\n".join(defect_lines) if defect_lines else "  * 특이 불량 없음"
        
        anomalies = data.get("parameter_anomalies", {})
        
        report_md = f"""# [사업재편 일일 공정 모니터링 보고서]
    **작성일자:** {today_str} | **대상 공정:** {data['target_line']}

    ---

    ### 1. 핵심 수율 지표 요약
    * **총 생산 배치 수:** {data['total_batches']} Batch
    * **평균 공정 수율:** **{data['average_yield']}%** (목표 90.0% 대비 {'초과 달성' if data['average_yield'] >= 90 else '미달'})
    * **목표 수율 달성률:** {data['target_achievement_rate']}% (양품 {data['pass_count']}건 / 불량 {data['fail_count']}건)

    ### 2. 주요 불량 유형 및 병목 원인
    {defects_text}

    ### 3. 공정 파라미터 규격 이탈(Anomaly) 현황
    * **금형 저온 이탈 (< 215℃):** {anomalies.get('temp_under_215C', 0)}회 (미성형 유발 요인)
    * **금형 과열 이탈 (> 225℃):** {anomalies.get('temp_over_225C', 0)}회 (크랙 결함 유발 요인)
    * **프레스 압력 규격 이탈:** {anomalies.get('pressure_out_of_spec', 0)}회

    ### 4. 종합 개선 조치 권고안 (Action Plan)
    1. 금형 온도 하한선 경보 센서를 즉시 점검하여 215℃ 이하 투입을 차단하십시오.
    2. 성형 압력 편차(±5ton 이상) 발생 구간에 대해 비례제어 밸브의 압력 보정 루프를 수행하십시오.

    > **현장 전달 메모:** {author_note if author_note else '특이사항 없음.'}
    """
        return report_md
    ```

<br>

- **3단계: AI 에이전트 실행 오케스트레이터 (Skills Registry 연동)**
    - 에이전트가 어떤 스킬을 가지고 있는지 인식하고,
    - 사용자의 요청에 따라 스킬을 순차적으로 엮어 실행하는 간이 실행기(Agent Runner)

        ```python
        class ManufacturingAgent:
            def __init__(self):
                # 에이전트에 스킬(도구) 등록 (Skills Registry)
                self.skills = {
                    "analyze_process_yield": analyze_process_yield,
                    "generate_restructuring_report": generate_restructuring_report
                }
                
            def execute_workflow(self, csv_file: str, line_name: str, memo: str):
                print(f"[Agent] 작업 접수: {line_name} 데이터 분석 및 사업재편 보고서 생성 시작")
                
                # 1. 첫 번째 스킬 실행 (데이터 분석)
                print("  ▶ [Skill 1 호출] analyze_process_yield 실행 중...")
                analysis_raw = self.skills["analyze_process_yield"](csv_file, line_name)
                
                # 2. 중간 결과 추론 (에이전트의 내부 판단)
                parsed = json.loads(analysis_raw)
                print(f"    * 분석 완료: 평균 수율 {parsed['average_yield']}%, 불량 {parsed['fail_count']}건 식별")
                
                # 3. 두 번째 스킬 실행 (보고서 작성)
                print("  ▶ [Skill 2 호출] generate_restructuring_report 실행 중...")
                final_report = self.skills["generate_restructuring_report"](analysis_raw, author_note=memo)
                
                print("[Agent] 모든 워크플로우를 완수했습니다.\n")
                return final_report

        # 에이전트 가동 테스트
        agent = ManufacturingAgent()
        report_output = agent.execute_workflow(
            csv_file="daily_process_log.csv",
            line_name="Line-B (신규재편)",
            memo="금형 교체 직후 초기 10개 배치에서 저온 현상 발생 확인됨. 오후 조에서 히터 복구 완료."
        )

        # 최종 산출 보고서 출력
        print(report_output)
        ```

<br>

## 4. 실습 결과 분석 및 해석

- **실행 결과 산출물 확인**
    - 스크립트 실행 시 AI 에이전트가 2개의 스킬을 거쳐 최종 출력한 실제 결과는 다음과 같음

        ```markdown
        # [사업재편 일일 공정 모니터링 보고서]
        **작성일자:** 2026-10-10 | **대상 공정:** Line-B (신규재편)

        ---

        ### 1. 핵심 수율 지표 요약
        * **총 생산 배치 수:** 84 Batch
        * **평균 공정 수율:** **87.42%** (목표 90.0% 대비 미달)
        * **목표 수율 달성률:** 61.9% (양품 52건 / 불량 32건)

        ### 2. 주요 불량 유형 및 병목 원인
        * **UnderFill (미성형):** 18건
        * **Crack (균열):** 11건
        * **Burr (버 발생):** 3건

        ### 3. 공정 파라미터 규격 이탈(Anomaly) 현황
        * **금형 저온 이탈 (< 215℃):** 16회 (미성형 유발 요인)
        * **금형 과열 이탈 (> 225℃):** 12회 (크랙 결함 유발 요인)
        * **프레스 압력 규격 이탈:** 21회

        ### 4. 종합 개선 조치 권고안 (Action Plan)
        1. 금형 온도 하한선 경보 센서를 즉시 점검하여 215℃ 이하 투입을 차단하십시오.
        2. 성형 압력 편차(±5ton 이상) 발생 구간에 대해 비례제어 밸브의 압력 보정 루프를 수행하십시오.

        > **현장 전달 메모:** 금형 교체 직후 초기 10개 배치에서 저온 현상 발생 확인됨. 오후 조에서 히터 복구 완료.
        ```

<br>

- **데이터 관점의 결과 상세 분석**
    1. **수율 목표 미달 원인의 정량적 증명:**
        - 전체 84개 신규 배치 중 양품 기준(90.0% 이상)을 통과한 배치는 52건(61.9%)에 불과
        - 평균 수율은 87.42%로 목표치에 약 2.6%p 미달

    2. **공정 파라미터와 불량 간 인과관계 규명:**
        - 가장 빈번했던 불량인 미성형(UnderFill, 18건)은
            - 공정 온도 이탈 중 저온 이탈(215℃ 미만, 16회)과 발생 패턴이 일치함
            - 알루미늄 합금의 유동성이 부족한 상태에서 가압되었음을 수치로 입증

        - 반대로 발생한 크랙(Crack, 11건)은
            - **과열 이탈(225℃ 초과, 12회)** 구간에서 발생하여
            - 과열 후 급랭 과정의 열응력이 결함을 유발했음을 보여줌

<br>

- **실무적 기대 효과 (Before vs After)**

    | 구분               | 도입 전 (수작업 업무 방식)                                                       | 도입 후 (AI 에이전트 Skill 자동화)                                        |
    | :----------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
    | **소요 시간**      | CSV 다운로드 🡪 엑셀 정리 🡪 보고서 작성 (매일 약 40~50분) | **자연어 지시 1회로 전체 완료 (약 3초 이내)**                             |
    | **분석 정확도**    | 담당자 컨디션에 따른 파라미터 누락, 계산 오타 발생                               | 룰 기반 정량 스킬에 의해 **규격 이탈 횟수 및 불량 원인 100% 일관 집계**   |
    | **현장 대응 속도** | 익일 오전 취합 회의에서야 전일 불량 원인 파악                                    | 배치 종료 즉시 실시간 보고서가 생성되어 **당일 공정 밸브/온도 보정 가능** |

<br>

## 5. 비용 문제 해결 방안 🡪 카드 등록 없는 완전 무료 API: Google AI Studio 활용 (강력 추천)**

- ChatGPT(OpenAI)는 API를 쓰려면 반드시 해외 결제 카드를 등록하고 선불 충전을 해야 하지만,
- Google AI Studio는 구글 계정(Gmail)만 있으면 신용카드 등록 없이 완전 무료(Free Tier)로 API 키를 즉시 발급해 줌

- **무료 혜택:**
    - 신용카드 등록 불필요 (비용 청구 원천 차단)
    - Gemini Flash 계열 모델 기준 **분당 약 10~15회 호출, 하루 수백 회 무료 호출 제공** (실습 강의 진행에 충분한 용량)

- **교육생 실습 진행 절차:**
    1. 웹 브라우저에서 `aistudio.google.com` 접속 후 구글 로그인
    2. 좌측 상단 **[Get API key]** 버튼 클릭 🡪 **[Create API key]** 클릭
    3. 발급된 영문/숫자 키를 복사하여 실습 환경(Dify 노코드 도구 또는 준비된 파이썬 노트북)에 붙여넣기만 하면 끝
