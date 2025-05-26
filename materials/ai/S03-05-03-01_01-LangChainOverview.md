---
layout: page
title:  "LangChain 개요"
date:   2025-04-01 09:00:00 +0900
permalink: /materials/S03-05-03-01_01-LangChainOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. LangChain 개요
### 1.1 LangChain이란?
- 대규모 언어 모델(LLM)을 활용하여 애플리케이션을 개발하기 위한 오픈 소스 프레임워크
    - 2022년 10월, 머신러닝 스타트업 Robust Intelligence의 해리슨 체이스에 의해 공개됨
- LLM의 잠재력을 극대화하고, 외부 데이터와 통합하여 보다 복잡하고 유용한 애플리케이션을 구축할 수 있도록 설계됨
- AI 애플리케이션 개발의 복잡성을 줄이고, 확장 가능성과 유연성을 제공하는 강력한 도구로 자리 잡음
    - 현재, 수많은 AI 서비스가 LLM + LangChain 기반으로 개발, 운영되고 있음

---

### 1.2 LangChain의 개념

- 이름 그대로 "언어의 체인"을 의미
- LLM에게 어떤 문제에 대한 일을 시키기 위하여 프롬프트를 작성할 경우, 사용자의 프롬프트를 곧바로 LLM에게 전달하는 것이 아니라 하나의 <span style="color: #AA3333">프롬프트 템플릿을 거쳐 전달하도록 추가적인 "연결고리"를 만들어 원하는 답변을 이끌어 내는 것</span>
- 따라서 LangChain에서는 
    - 프롬프트 변형을 위한 프롬프트 템플릿 제공
    - LLM 서비스 개발을 위한 다양한 모듈 제공
        - (RAG를 위한 모듈의 예)
            - Models: 여러 LLM을 애플리케이션에 통합
            - Prompts: 사용자의 프롬프트를 재가공
            - Document Loaders: 벡터 DB로 구축할 문서를 불러옴
            - Text Splitters: 불러온 문서를 여러 청크로 분할
            - Vector Stores: 분할된 텍스트 청크들을 저장
            - Outpt Parsers: 원하는 답변의 형태로 재가공

---

### 1.3 주요 특징
- 데이터와의 통합
    - 데이터베이스, 파일 시스템 등 다양한 외부 데이터 소스와 연결해 실시간 데이터를 활용하는 애플리케이션 개발 가능
- 유연한 구성 요소
    - 프롬프트 체인, 에이전트, 메모리 등을 통해 복잡한 작업을 간소화
- 오픈 소스 및 커뮤니티 지원
    - 무료로 사용 가능
    - 활발한 커뮤니티를 통해 지원받을 수 있음

---

### 1.4 주요 모듈과 기능

- Model I/O
    - 다양한 대규모 언어 모델(LLM), 채팅 모델, 임베딩 모델과의 인터페이스 제공.
    - 텍스트 데이터를 벡터로 변환하는 임베딩 모델 지원
    - 모델 출력에서 정보를 추출하는 기능 포함

- Prompt
    - 프롬프트 생성 및 관리, 최적화 도구 제공
        - Prompt Templates: 동적 프롬프트 생성
        - Example Selectors: 상황에 맞는 예제 선택
        - Output Parsers: LLM의 응답을 구조화된 형식으로 변환
    - 예제 선택 및 출력 파싱 기능 포함

- Retrieval (Data Connection)
    - 외부 데이터에서 정보를 검색하여 LLM과 연결(통합) 지원
    - 문서 로더, 데이터 분할, 벡터 저장소 등의 기능 제공
        - Document Loaders: 다양한 소스에서 문서 로드
        - Document Transformers: 문서 분할 및 변환
        - Vector Stores: 임베딩 데이터를 저장 및 검색
        - Retrievers: 데이터 쿼리 처리

- Memory
    - 대화형 애플리케이션에서 상태를 유지하기 위한 메모리 관리
    - 대화 데이터를 저장하고 상태를 유지하여 문맥 인식을 강화
        - 이전 대화나 작업 데이터를 저장하고 참조 가능

- Chain
    - 여러 작업을 순차적으로 연결하는 체인 구성
        - LLM Chains: 기본 프롬프트와 LLM 기반 체인
        - Sequential Chains: 순차적 작업 처리
        - Router Chains: 질문 주제에 따라 적절한 체인으로 라우팅

- Agents
    - 주어진 작업에 적합한 도구와 모델을 선택해 실행
    - LM이 작업 순서를 결정하고 외부 리소스와 상호작용 가능
    - Tools와 이를 묶은 Toolkits 제공

---

### 1.5 활용 분야

- 사용자 맞춤형 콘텐츠 생성
- 다국어 번역 및 질의응답 시스템
- 실시간 데이터 기반 애플리케이션 구축
