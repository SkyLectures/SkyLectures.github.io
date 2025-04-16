---
layout: page
title:  "GCP, AWS, Azure 비교"
date:   2025-04-11 03:30:00 +0900
permalink: /materials/S03-10-01-02_GCP_AWS_Azure_Comparison
categories: materials
---



## ✅ GCP의 주요 서비스별 세부 설명 & AWS / Azure 비교

| 서비스 영역 | GCP | AWS | Azure |
|-------------|-----|-----|-------|
| **컴퓨팅** | **Compute Engine**: VM 인스턴스 제공, 사용자 정의 가능. **App Engine**: PaaS 기반 자동 확장 앱 배포. **Cloud Run / GKE**: 컨테이너 기반 실행. | **EC2**: 가상 서버. **Elastic Beanstalk**: PaaS. **Lambda**: 서버리스 컴퓨팅. | **Azure VMs**, **App Service**, **Functions**, **AKS** |
| **스토리지** | **Cloud Storage**: 객체 스토리지. **Persistent Disk**, **Filestore**, **Cloud SQL**, **Bigtable**, **Firestore** | **S3**, **EBS**, **EFS**, **RDS**, **DynamoDB**, **Aurora** | **Blob Storage**, **Disk Storage**, **File Storage**, **SQL Database**, **Cosmos DB** |
| **네트워킹** | **VPC**, **Cloud Load Balancing**, **Cloud CDN**, **Cloud Interconnect** | **VPC**, **ELB**, **CloudFront**, **Direct Connect** | **VNet**, **Load Balancer**, **CDN**, **ExpressRoute** |
| **컨테이너/오케스트레이션** | **GKE (Google Kubernetes Engine)**: 매끄러운 Kubernetes 관리 및 운영. **Cloud Run**: 서버리스 컨테이너 실행. | **EKS**, **ECS**, **Fargate** | **AKS (Azure Kubernetes Service)**, **Container Instances** |
| **데이터베이스** | **Cloud SQL (MySQL, PostgreSQL 등)**, **Firestore**, **Bigtable**, **Spanner (글로벌 분산 DB)** | **RDS**, **Aurora**, **DynamoDB**, **Redshift** | **SQL Database**, **Cosmos DB**, **Table Storage**, **Synapse** |
| **AI/ML** | **Vertex AI**: ML 모델 구축·배포·운영 통합 플랫폼. **AutoML**, **TPU**, **Generative AI Studio** | **SageMaker**, **Bedrock**, **Comprehend**, **Rekognition** | **Azure Machine Learning**, **OpenAI on Azure**, **Cognitive Services** |
| **빅데이터** | **BigQuery**: 서버리스 데이터 웨어하우스. **Dataflow**, **Dataproc**, **Pub/Sub** | **Redshift**, **Glue**, **EMR**, **Kinesis** | **Synapse**, **HDInsight**, **Data Factory**, **Event Hubs** |
| **DevOps/CI-CD** | **Cloud Build**, **Cloud Deploy**, **Artifact Registry**, **Cloud Source Repositories** | **CodePipeline**, **CodeBuild**, **CodeCommit**, **CodeDeploy** | **Azure DevOps**, **GitHub Actions (MS 통합)**, **Azure Pipelines** |
| **보안 및 IAM** | **Cloud IAM**, **VPC Service Controls**, **Security Command Center** | **IAM**, **GuardDuty**, **Macie**, **Security Hub** | **Azure AD**, **Security Center**, **Sentinel**, **Key Vault** |
| **모니터링 & 로깅** | **Cloud Monitoring (구 Stackdriver)**, **Cloud Logging** | **CloudWatch**, **X-Ray**, **CloudTrail** | **Azure Monitor**, **Log Analytics**, **Application Insights** |

---

## 🔍 GCP vs AWS vs Azure 간 주요 특징 비교

| 항목 | GCP | AWS | Azure |
|------|-----|-----|-------|
| **강점** | AI/ML, 데이터 분석, Kubernetes(오픈소스 기반) | 서비스 범위 최다, 글로벌 인프라, 커뮤니티 강력 | 하이브리드 클라우드, Microsoft 제품과 통합 |
| **비용 측면** | 비교적 합리적이고 투명한 과금 | 다양하지만 복잡한 요금 구조 | Microsoft 사용 기업에 유리한 요금제 |
| **시장 점유율** | 3위 (성장 중) | 1위 | 2위 |
| **사용자 친화성** | 직관적인 UI, 빠른 구축 | 풍부한 기능 (다소 복잡) | 기업 친화적인 UI와 워크플로우 |
| **AI/ML 분야** | ✅ **Vertex AI**, BigQuery와의 통합 | SageMaker 중심 | Azure ML, OpenAI API 통합 |
| **오픈소스 친화성** | 매우 높음 (Kubernetes 창시자, TensorFlow 등) | 점점 확대 중 | 비교적 제한적 (MS 중심 생태계) |

---

## ✨ 정리

- GCP는 **AI/ML, 데이터 분석, Kubernetes 기반 워크로드에 매우 강력한 플랫폼**입니다.
- AWS는 **가장 다양한 서비스와 글로벌 커버리지를 제공**하며, 매우 유연하지만 복잡할 수 있습니다.
- Azure는 **Microsoft 환경과의 통합에 최적화된 기업형 플랫폼**으로 하이브리드 전략에 강점이 있습니다.

