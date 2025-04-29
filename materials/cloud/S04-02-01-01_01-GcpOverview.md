---
layout: page
title:  "GCP(Google Cloud Platform)"
date:   2025-04-11 03:20:00 +0900
permalink: /materials/S04-02-01-01_01-GcpOverview
categories: materials
---

## 🌥️ GCP 개요

### 🔹 GCP란?
**Google Cloud Platform(GCP)**는 Google이 제공하는 **클라우드 컴퓨팅 서비스**입니다. 사용자는 인터넷을 통해 Google의 인프라, 머신러닝, 빅데이터, 네트워크, 저장소, 보안 등 다양한 서비스를 이용할 수 있음

---

### 🔹 주요 특징
- **확장성**: 필요에 따라 리소스를 쉽게 확장하거나 축소 가능
- **글로벌 인프라**: 전 세계에 위치한 데이터 센터 기반으로 안정성과 빠른 응답성 제공
- **보안**: Google의 보안 인프라와 지속적인 업데이트 제공
- **오픈소스 친화적**: Kubernetes, TensorFlow 등 오픈소스 기술에 기반

---

### 🔹 주요 서비스

#### 1. **컴퓨팅**
- **Compute Engine**: 가상 머신(VM) 인스턴스를 제공하는 IaaS 서비스
- **App Engine**: 애플리케이션을 코드만으로 배포할 수 있는 PaaS 서비스
- **Cloud Run**: 컨테이너 기반의 서버리스 애플리케이션 실행
- **Kubernetes Engine(GKE)**: Kubernetes 클러스터 관리 서비스

#### 2. **스토리지 & 데이터베이스**
- **Cloud Storage**: 객체 저장소 (대용량 파일, 이미지 등 저장용)
- **Cloud SQL**: 관리형 관계형 데이터베이스 (MySQL, PostgreSQL 등)
- **Bigtable**: NoSQL 데이터베이스, 대규모 분석/IoT용
- **Firestore / Datastore**: 서버리스 NoSQL 데이터베이스

#### 3. **네트워킹**
- **Cloud Load Balancing**: 글로벌 트래픽 부하 분산
- **VPC (Virtual Private Cloud)**: 가상 네트워크 설정
- **Cloud CDN**: 콘텐츠를 빠르게 전송하는 캐시 서비스

#### 4. **AI & 머신러닝**
- **Vertex AI**: ML 모델 구축, 학습, 배포를 위한 통합 플랫폼
- **Vision AI / Speech-to-Text / Natural Language API**: 사전 훈련된 AI API

#### 5. **빅데이터 및 분석**
- **BigQuery**: 초고속 SQL 기반 데이터 웨어하우스
- **Dataflow**: 스트리밍 및 배치 데이터 처리
- **Pub/Sub**: 메시지 큐 / 이벤트 처리 시스템

---

### 🔹 사용 사례
- **웹/모바일 앱 호스팅**
- **데이터 분석 및 머신러닝**
- **IoT 시스템 구축**
- **비디오 스트리밍 서비스**
- **게임 서버 운영**

---

### 🔹 요금 체계
- **종량제**: 사용한 만큼만 지불
- **무료 등급**: 일부 서비스는 무료로 제공 (월간 제한 있음)
- **할인제도**: 지속 사용 할인(Committed Use Discounts), 자동 할인(Sustained Use Discounts)

---

