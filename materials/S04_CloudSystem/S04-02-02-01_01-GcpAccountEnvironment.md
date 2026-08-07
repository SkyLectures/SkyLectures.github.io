---
layout: page
title:  "GCP 계정 생성 및 환경 설정"
date:   2025-03-01 10:00:00 +0900
permalink: /materials/S04-02-02-01_01-GcpAccountEnvironment
categories: materials
---
* toc
{:toc .large-only .toc-sticky:true}


GCP(Google Cloud Platform)를 사용하기 위한 환경 설정은 크게 다음의 단계로 이루어집니다.

**1단계: Google 계정 생성**

  * 이미 Gmail 계정이 있다면 해당 계정을 사용할 수 있습니다.
  * 새로운 Google 계정을 만들려면 [Google 계정 만들기](https://www.google.com/search?q=https://accounts.google.com/signup) 페이지로 이동하여 안내에 따라 계정을 생성합니다.

**2단계: GCP 프로젝트 생성**

  * Google Cloud 콘솔([https://console.cloud.google.com/](https://console.cloud.google.com/))에 로그인합니다.
  * 화면 상단의 프로젝트 선택 드롭다운 메뉴를 클릭하고 **새 프로젝트**를 선택합니다.
  * 프로젝트 이름과 조직 (선택 사항)을 입력하고 **만들기**를 클릭합니다.

**3단계: 결제 설정**

  * 새 프로젝트를 생성하면 결제 계정을 연결하라는 메시지가 표시될 수 있습니다. 또는 콘솔 왼쪽 메뉴에서 **결제**를 선택합니다.
  * 아직 결제 계정이 없다면 **결제 계정 만들기**를 클릭하고 안내에 따라 결제 정보를 입력합니다. Google Cloud는 신규 사용자에게 일정 금액의 무료 크레딧을 제공할 수 있습니다.
  * 기존 결제 계정이 있다면 해당 계정을 선택하고 생성한 프로젝트에 연결합니다.

**4단계: IAM(Identity and Access Management) 설정 (선택 사항)**

  * GCP 리소스에 대한 접근 권한을 관리하기 위해 IAM 설정을 할 수 있습니다.
  * 콘솔 왼쪽 메뉴에서 **IAM 및 관리자** \> **IAM**을 선택합니다.
  * **추가** 버튼을 클릭하여 사용자, 그룹 또는 서비스 계정에 역할을 부여할 수 있습니다. 역할을 통해 특정 리소스에 대한 접근 권한을 세밀하게 제어할 수 있습니다.

**5단계: Google Cloud CLI (gcloud CLI) 설치 및 초기화 (선택 사항)**

  * 명령줄 인터페이스를 통해 GCP를 관리하려면 gcloud CLI를 설치하는 것이 좋습니다.
  * [Google Cloud CLI 설치](https://cloud.google.com/sdk/docs/install) 페이지에서 운영체제에 맞는 설치 안내를 따릅니다.
  * 설치 후 터미널에서 `gcloud init` 명령어를 실행하여 CLI를 초기화하고 Google 계정과 프로젝트를 연결합니다.

**6단계: 필요한 API 활성화**

  * 사용하려는 GCP 서비스 (예: Compute Engine, Cloud Storage, BigQuery 등)의 API를 활성화해야 합니다.
  * 콘솔 왼쪽 메뉴에서 **API 및 서비스** \> **라이브러리**를 선택합니다.
  * 원하는 API를 검색하여 선택하고 **사용 설정** 버튼을 클릭합니다.

**7단계: 네트워킹 설정 (선택 사항)**

  * VM 인스턴스를 생성하거나 네트워킹 기능을 사용하려면 VPC(Virtual Private Cloud) 네트워크를 설정해야 할 수 있습니다.
  * 콘솔 왼쪽 메뉴에서 **VPC 네트워크** \> **VPC 네트워크**를 선택하고 필요에 따라 네트워크, 서브넷, 방화벽 규칙 등을 생성합니다.

**8단계: 스토리지 설정 (선택 사항)**

  * 데이터를 저장하고 관리하기 위해 Cloud Storage 버킷을 생성할 수 있습니다.
  * 콘솔 왼쪽 메뉴에서 **Cloud Storage** \> **버킷**을 선택하고 **만들기** 버튼을 클릭하여 버킷을 생성합니다.

이 단계를 완료하면 GCP를 사용할 수 있는 기본적인 환경 설정이 완료됩니다. 이후에는 필요한 서비스와 리소스를 생성하고 구성하여 원하는 작업을 수행할 수 있습니다.








