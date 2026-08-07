**'AI 에이전트 특화 자율형 독립 OS'**
## 📋 [Project Reference] AI-Centric Autonomous OS (Codename: Self-Brain)

---

### 1. 연구의 핵심 목적 (Research Objective)

* **목표:** 범용 운영체제의 복잡성을 배제하고, 단일 하드웨어 개체의 **'인지(Sense)-추론(Think)-행동(Act)'** 루프에 최적화된 독립적·폐쇄적 64비트 OS 설계 및 구현.
* **철학:** 외부 통신 최소화, 자기 완결성(Self-contained), 실시간 추론 우선순위 보장.

---

### 2. 기술 아키텍처 및 특징 (Architecture & Characteristics)

| 구분 | 상세 명세 |
| --- | --- |
| **커널 유형** | **Agent-Centric Monolithic Kernel** (성능 극대화를 위해 AI 엔진과 커널이 밀착된 구조) |
| **대상 아키텍처** | **ARMv8-A (AArch64)** - Raspberry Pi 4/5 특화 |
| **핵심 스케줄러** | **Deterministic Priority Scheduler** (AI 추론 태스크에 최우선순위 및 고정 타임슬롯 할당) |
| **메모리 전략** | **Static Allocation First** (런타임 오류 방지를 위해 부팅 시 메모리 영역 확정, `no_std` 환경) |
| **인지 시스템** | **CSI Camera Driver(Bare-metal)** 기반 고속 이미지 스트림 획득 |
| **추론 엔진** | **Embedded Inference Engine** (TFLite Micro 또는 Rust 기반 `burn` 라이브러리 이식) |
| **행동 제어** | **Low-latency PWM/GPIO Control** (추론 결과와 하드웨어 제어 사이의 지연 시간 최소화) |

---

### 3. 개발 환경 및 도구 (Dev Environment)

* **언어:** Rust (`edition = "2021"`, `no_std`, `no_main`)
* **툴체인:** `aarch64-unknown-none-softfloat`, `cargo-make`, `nasm` (Bootloader)
* **에뮬레이션:** QEMU (`qemu-system-aarch64`) - 실제 보드 테스트 전 로직 검증용
* **디버깅:** UART(Universal Asynchronous Receiver/Transmitter) 시리얼 통신을 통한 커널 로그 확인

---

### 4. 단계별 개발 방향 (Development Roadmap)

1. **Phase 1: Bootstrapping**
* 라즈베리파이 4/5 전원 인가 시 Rust 엔트리 포인트(`_start`) 진입 및 UART를 통한 "Hello OS" 출력.


2. **Phase 2: Hardware Foundation**
* MMU 설정(가상 메모리 매핑), 인터럽트 벡터 테이블 구축, 기본적인 GPIO 및 PWM 드라이버 작성.


3. **Phase 3: Sensing & Memory**
* CSI 카메라 인터페이스 구현 및 이미지 데이터를 저장할 프레임 버퍼 관리 시스템 구축.


4. **Phase 4: AI Integration (The Brain)**
* 신경망 모델 가중치를 커널 바이너리에 포함시키고, `no_std` 환경에서 추론 연산 함수 구동.


5. **Phase 5: Self-Autonomous Loop**
* "인지-추론-행동" 루프를 완성하여 카메라 입력에 따라 물리적 액추에이터가 실시간 반응하도록 최적화.



---

### 🔍 딥리서치를 위한 핵심 질문 리스트 (Research Queries)

딥리서치 도구에 다음 질문들을 던져 상세 데이터를 수집하세요.

1. **[하드웨어 제어]** "Raspberry Pi 4/5의 Broadcom BCM2711/BCM2712 칩셋에서 OS 수준의 Bare-metal CSI 카메라 드라이버를 구현하기 위한 레지스터 맵과 초기화 시퀀스는?"
2. **[Rust OS 개발]** "Rust `no_std` 환경에서 정적 메모리 할당만을 사용하여 고용량 AI 모델 가중치(약 5~10MB)를 효율적으로 로드하고 관리하는 디자인 패턴은?"
3. **[실시간성]** "임베디드 AI 추론의 지연 시간을 줄이기 위해 ARM64 아키텍처의 L1/L2 캐시를 OS 커널 수준에서 어떻게 최적화할 수 있는가?"
4. **[추론 엔진 이식]** "Rust로 작성된 `burn` 또는 `tract` 라이브러리를 `aarch64-unknown-none` 타겟의 베어메탈 환경에 이식할 때 발생하는 주요 이슈와 해결 방안은?"
5. **[폐쇄형 OS]** "외부 라이브러리 의존성 없이 독립 개체로서 작동하는 OS에서, 환경의 변화(경험)를 기록하기 위한 최소한의 휘발성/비휘발성 데이터 구조 설계 사례는?"

---

### 💡 연구 가이드라인 (Tips for Suk-hwan)

석환님의 **AI 전문성**과 **하드웨어 수리 경험**을 결합한다면, 이 프로젝트는 단순한 OS를 넘어 **'디지털 생명체'**의 기초가 될 것입니다.

* **추천 도서:** *The Little Book of Semaphores* (동기화 이해용), *Operating Systems: Three Easy Pieces* (OS 구조 이해용)
* **참고 커뮤니티:** Reddit의 `r/osdev`, Rust Embedded Working Group.
