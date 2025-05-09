---
layout: page
title:  "GCP(Google Cloud Platform)"
date:   2025-04-11 03:20:00 +0900
permalink: /materials/S04-02-01-01_01-GcpOverview
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}

## 1. GCP(Google Cloud Platform) 개요

### 1.1 GCP란?
- Google이 제공하는확장성이 뛰어나고 안전하며 혁신적인 클라우드 컴퓨팅 서비스 플랫폼
- 강력한 인프라, 혁신적인 기술, 다양한 서비스 포트폴리오를 바탕으로 모든 규모의 기업과 개발자에게 최적의 클라우드 환경을 제공함
- Google의 글로벌 인프라를 기반으로 구축됨
- 컴퓨팅, 스토리지, 네트워킹, 빅데이터, 머신러닝, IoT 등 다양한 클라우드 서비스 제공
    - 특히 데이터 분석, 머신러닝, 컨테이너 기술 분야에서 강력한 경쟁력을 가지고 있음
- 전 세계의 기업과 개발자들이 혁신적인 솔루션을 구축하고 운영할 수 있도록 지원함
- 기업의 디지털 전환과 혁신을 위한 핵심적인 플랫폼 역할을 수행함
- 사용자는 인터넷을 통해 Google의 인프라, 머신러닝, 빅데이터, 네트워크, 저장소, 보안 등 서비스를 이용할 수 있음

### 1.2 기본 개념 및 주요 특징

- **글로벌 인프라**
    - GCP는 전 세계 여러 지역(Region)과 가용 영역(Zone)으로 구성된 강력한 글로벌 네트워크를 기반으로 함
    - 각 리전은 독립적인 지리적 위치
        - 전 세계에 위치한 데이터 센터를 기반으로 안정성(높은 가용성)과 빠른 응답성(낮은 지연 시간) 제공
        - 여러 개의 가용 영역으로 이루어져 높은 가용성과 재해 복구 능력을 제공함
    - Google의 강력한 인프라를 기반으로 애플리케이션을 쉽게 확장하고 안정적인 성능을 유지할 수 있음

- **강력한 보안** 
    - Google의 보안 인프라와 지속적인 업데이트 제공
    - Google이 사용하는 것과 동일한 보안 기술을 적용하여 데이터와 애플리케이션을 안전하게 보호

- **프로젝트**
    - GCP의 모든 리소스는 프로젝트라는 논리적인 컨테이너 내에서 관리됨
    - 프로젝트는 과금, 권한 관리, 리소스 격리 등의 기본 단위를 제공함

- **서비스**
    - 100개 이상의 다양한 클라우드 서비스 제공
    - 서비스는 필요에 따라 조합, 확장 및 축소하여 사용할 수 있음
    - 웹사이트, 애플리케이션, 데이터 분석 시스템 등 다양한 솔루션의 구축에 활용됨

- **오픈소스 및 사용자 친화적**
    - Kubernetes, TensorFlow 등 오픈소스 기술에 기반하며
    - 주요 오픈소스 프로젝트를 적극적으로 지원하고 통합하여 개발 편의성을 높임
    - 웹 기반 콘솔, CLI, API 등 다양한 관리 도구를 제공하여 리소스 관리 및 자동화를 용이하게 함

- **강력한 데이터 분석 및 AI/ML 기능**
    - 대규모 데이터 처리 및 분석을 위한 강력한 도구와 머신러닝 모델 개발 및 배포를 위한 포괄적인 플랫폼 제공
    - 머신러닝 (Vertex AI, TensorFlow), 빅데이터 분석 (BigQuery, Dataflow), 컨테이너 기술 (Google Kubernetes Engine - GKE) 등 Google의 첨단 기술을 활용할 수 있음

- **사용한 만큼 지불 (Pay-as-you-go)**
    - 사용한 컴퓨팅 파워, 스토리지 용량, 네트워크 트래픽 등에 따라 비용을 지불하는 종량제 방식 중심 채택
        - 잘 운영할 경우, 비용 효율성 향상 가능
    - 일부 서비스는 무료 제공(월간 제한 있음)
    - 할인제도 지원
        - 지속 사용 할인(Committed Use Discounts)
        - 자동 할인(Sustained Use Discounts)
    - 종량제, 지속 사용 할인, 약정 할인 등 다양한 가격 옵션을 제공하여 비용 효율적인 클라우드 환경을 구축할 수 있음

### 1.3 주요 서비스 모델

- 다양한 서비스 모델 제공 → 사용자의 요구 사항에 맞는 유연한 선택 가능

- 서비스 모델
    - **IaaS (Infrastructure as a Service)**
        - 컴퓨팅 인스턴스(Compute Engine), 스토리지(Cloud Storage, Persistent Disk), 네트워킹(Virtual Private Cloud) 등 기본적인 IT 인프라를 클라우드 환경에서 제공
        - 사용자는 운영체제, 미들웨어, 애플리케이션 등을 직접 관리

    - **PaaS (Platform as a Service)**
        - 애플리케이션 개발, 실행 및 관리를 위한 플랫폼 제공
            - App Engine, Cloud Functions, Cloud Run
        - 개발자는 인프라 관리에 대한 부담 없이 애플리케이션 개발에 집중할 수 있음

    - **SaaS (Software as a Service)**
        - 클라우드 기반으로 제공되는 소프트웨어 애플리케이션을 구독 형태로 이용
            - Google Workspace 등
        - 사용자는 소프트웨어 설치, 업데이트, 인프라 관리 없이 애플리케이션을 사용할 수 있음

### 1.4 다양한 활용 사례
- 웹/모바일 앱 호스팅
- 데이터 분석 및 머신러닝
- IoT 시스템 구축
- 비디오 스트리밍 서비스
- 게임 서버 운영


## 2. 주요 서비스 분야

### 2.1 컴퓨팅
- **가상머신**
    - Compute Engine: 가상 머신(VM) 인스턴스를 제공하는 IaaS 서비스

- **컨테이너 관리**
    - Kubernetes Engine(GKE): Kubernetes 클러스터 관리 서비스

- **서버리스 컴퓨팅**
    - Cloud Functions
    - Cloud Run: 컨테이너 기반의 서버리스 애플리케이션 실행

- **애플리케이션**
    - App Engine: 애플리케이션을 코드만으로 배포할 수 있는 PaaS 서비스

### 2.2 스토리지
- **객체 스토리지**
    - Cloud Storage: 객체 저장소 (대용량 파일, 이미지 등 저장용)

- **블록 스토리지**
    - Persistent Disk

- **파일 스토리지**
    - Filestore

### 2.3 데이터베이스 
- **관계형 데이터베이스**
    - Cloud SQL: 관리형 관계형 데이터베이스 (MySQL, PostgreSQL 등)
    - Cloud Spanner

- **NoSQL 데이터베이스**
    - Cloud Firestore / Datastore: 서버리스 NoSQL 데이터베이스
    - Cloud Bigtable: NoSQL 데이터베이스, 대규모 분석/IoT용

### 2.4 네트워킹
- **가상 사설망**
    - Virtual Private Cloud(VPC): 가상 네트워크 설정

- **로드 밸런싱**
    - Cloud Load Balancing: 글로벌 트래픽 부하 분산

- **콘텐츠 전송 네트워크**
    - Cloud CDN: 콘텐츠를 빠르게 전송하는 캐시 서비스

### 2.5 AI & 머신러닝
- **AI 플랫폼**
    - Vertex AI: ML 모델 구축, 학습, 배포를 위한 통합 플랫폼

- **자연어 처리**
    - Natural Language API: 사전 훈련된 자연어 처리 API

- **이미지/영상 분석**
    - Vision AI: 사전 훈련된 이미지/영상 분석 API

- **음성 인식**
    - Speech-to-Text: 사전 훈련된 음성인식 API

### 2.6 빅데이터 및 분석
- **데이터 웨어하우스**
    - BigQuery: 초고속 SQL 기반 데이터 웨어하우스

- **데이터 처리**
    - Dataflow: 스트리밍 및 배치 데이터 처리
    - Pub/Sub: 메시지 큐 / 이벤트 처리 시스템

- **데이터 통합**
    - Cloud Data Fusion

### 2.7 IoT (Internet of Things)
- **IoT 플랫폼**
    - Cloud IoT Core

### 2.8 개발자 도구
- **API 관리**

- **CI/CD**
    - Cloud Build
    
- **소스 코드 관리**
    - Cloud Source Repositories

### 2.9 관리 및 운영
- **모니터링**
    - Cloud Monitoring

- **로깅**
    - Cloud Logging

- **구성 관리**
    - Cloud Deployment Manager

### 2.10 보안 및 규정 준수
- Identity and Access Management (IAM)

- Cloud Security Command Center

- 데이터 암호화
