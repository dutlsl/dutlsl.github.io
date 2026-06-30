---
title: "[CVPR 2026] GazeShift: VR을 위한 비지도 시선 추정 프레임워크 논문 리뷰"
date: 2026-06-25T21:00:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "VR", "Unsupervised Learning", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Samsung SIRC와 Bar-Ilan University가 CVPR 2026에서 발표한 GazeShift 논문을 리뷰합니다. VR 환경에서의 비지도 시선 추정 프레임워크와 대규모 off-axis 데이터셋 VRGaze를 제안한 논문입니다."
cover:
  image: "/images/gazeshift/_page_1_Figure_0.jpeg"
  alt: "GazeShift Architecture"
  caption: "GazeShift 아키텍처 개요 (논문 Figure 1)"
---

> **논문 정보**
> - **제목:** GazeShift: Unsupervised Gaze Estimation and Dataset for VR
> - **저자:** Gil Shapira, Ishay Goldin, Evgeny Artyomov, Donghoon Kim, Yosi Keller, Niv Zehngut
> - **소속:** Samsung Semiconductor Israel R&D Center (SIRC), Samsung Electronics, Bar-Ilan University
> - **학회:** CVPR 2026
> - **코드 & 데이터셋:** [github.com/gazeshift3/gazeshift](https://github.com/gazeshift3/gazeshift)

---

## 1. 한 줄 요약

VR 헤드셋의 off-axis 근접 카메라 환경에 특화된 **대규모 시선 데이터셋(VRGaze, 210만 장)**과 **레이블 없이 시선 표현을 학습하는 어텐션 기반 프레임워크(GazeShift)**를 제안하여, 비지도 학습만으로 **평균 1.84° 오차**를 달성하고, VR 디바이스 GPU에서 **5 ms 추론**이 가능함을 보여준 논문.

---

## 2. 연구 배경 및 동기

### 2.1 시선 추정(Gaze Estimation)이 왜 중요한가?

시선 추정은 HCI, XR, 보조 기술 등 다양한 분야에서 핵심적인 역할을 합니다. 특히 VR/AR에서는 다음과 같은 응용이 가능합니다:
- **Foveated Rendering**: 시선이 향하는 영역만 고해상도로 렌더링하여 연산 효율 극대화
- **직관적 입력**: 시선 기반 UI 인터랙션
- **적응형 콘텐츠**: 사용자 주의(attention)에 따른 콘텐츠 조절

### 2.2 기존 연구의 한계

| 문제 | 설명 |
|------|------|
| **데이터 부족** | 기존 VR 시선 데이터셋은 대부분 on-axis 카메라 기반이며, 실제 상용 VR 헤드셋의 off-axis 카메라 구조를 반영하지 못함 |
| **레이블링 비용** | 시선 레이블은 대상 응시를 보장할 수 없어 부정확하고 비용이 높음 |
| **도메인 갭** | 원거리 RGB 카메라 기반 연구는 근접 적외선(IR) 이미지 환경과 큰 차이 |
| **모델 크기** | 기존 비지도 방법들이 전체 얼굴 이미지를 필요로 하거나, 무거운 모델을 사용 |

---

## 3. 제안 1: VRGaze 데이터셋

논문의 첫 번째 기여는 **VRGaze**, 최초의 대규모 off-axis VR 시선 추정 데이터셋입니다.

### 데이터셋 스펙

| 항목 | 내용 |
|------|------|
| 이미지 수 | **210만 장** (좌·우 눈 동기 촬영) |
| 참여자 수 | **68명** |
| 카메라 | off-axis 근접 적외선(IR) 카메라 |
| 해상도 | 400 × 400 |
| 프레임 레이트 | 30 fps |
| 레이블 | 2D Point of Regard (PoR) |
| 분할 | Train 61명 / Val 7명 |

### 기존 데이터셋과의 비교

논문에서 제시한 기존 VR 시선 데이터셋과의 비교를 정리하면:

- **OpenEDS2020**: 55만 장, 80명이지만 **전부 on-axis** — 상용 VR 헤드셋의 off-axis 구조와 괴리
- **NVGaze**: 250만 장이지만 off-axis는 약 26만 장(14명)에 불과하고, 시선 방향 다양성 부족
- **TEyeD**: 2,000만 장 이상이지만 VR 환경이 아니며, 계산적 방법으로 어노테이션된 데이터

![VRGaze 데이터셋 샘플 및 on-axis vs off-axis 비교](/images/gazeshift/_page_3_Figure_0.jpeg)
*Figure 2: off-axis VRGaze 샘플(좌), on-axis OpenEDS2020 샘플(중앙), VRGaze 시선 각도 분포(우). Off-axis 카메라는 시각적 방해를 줄이기 위해 비스듬한 각도로 장착되지만, 강한 원근 왜곡이 발생하여 on-axis 데이터와는 근본적으로 다른 분포를 형성합니다.*

### 데이터 수집 프로토콜

- VR 디스플레이 위의 **이동 타깃**을 참가자가 따라가는 방식(추적 + 고정 교대)
- 배경 밝기를 변화시켜 다양한 **동공 확대(pupil dilation)** 유도
- 참가자당 평균 **7세션** 기록, 결정적(deterministic) + 무작위 궤적 조합
- 성별, 민족, 연령의 다양성 확보 (여성 14 / 남성 54, 아시아 29 / 백인 39)

---

## 4. 제안 2: GazeShift 프레임워크

### 4.1 핵심 아이디어

GazeShift의 핵심 통찰은 다음과 같습니다:

> **VR 헤드셋 장착 카메라에서, 동일인의 같은 눈 프레임 간 외양(appearance) 변화의 대부분은 시선 변화에 기인한다.**

이 가정 하에, **소스 프레임을 타깃 프레임으로 변환하는 생성적 pretext task**를 설정합니다. 모델이 소스 이미지의 눈 모양을 타깃의 시선 방향으로 "리디렉션"하려면, 타깃에서 추출한 임베딩에 시선 방향 정보가 반드시 인코딩되어야 합니다. 즉, 명시적 레이블 없이도 시선 표현이 자연스럽게 학습되는 구조입니다.

학습 시에는 동일 피험자의 같은 눈에서 서로 다른 시점에 촬영된 프레임 쌍 (source, target)을 사용하여, 소스-타깃 간의 차이가 주로 시선 변화에 의한 것임을 보장합니다.

### 4.2 전체 아키텍처 개요

아래 Figure 1은 GazeShift의 전체 아키텍처를 보여줍니다. 학습 파이프라인은 크게 **4개의 모듈**로 구성됩니다: (1) Gaze Encoder, (2) Appearance Encoder, (3) Attention 기반 융합 모듈, (4) Decoder. 추론 시에는 Gaze Encoder와 경량 캘리브레이션 모듈만 사용되므로, 학습에 사용된 무거운 Appearance Encoder와 Decoder는 런타임에 전혀 부담을 주지 않습니다.

![GazeShift 아키텍처](/images/gazeshift/_page_1_Figure_0.jpeg)
*Figure 1: GazeShift 전체 아키텍처. 좌측의 Gaze Encoder는 타깃 이미지에서 시선 임베딩을 추출하고, Appearance Encoder는 소스 이미지의 공간적 외양 특징을 보존합니다. 중앙의 Self-Attention → Cross-Attention 블록에서 시선 정보로 외양 특징을 조건화(conditioning)하며, Decoder가 최종적으로 시선이 리디렉션된 이미지를 재구성합니다. 하단의 Attention Map은 Gaze-Focused Loss의 가중치로 재활용됩니다.*

아키텍처의 정보 흐름을 단계별로 상세히 살펴보겠습니다.

---

### 4.3 모듈 ①: 분리된 인코더 (Separate Gaze & Appearance Encoders)

GazeShift에서 가장 먼저 눈에 들어오는 설계 결정은 **시선(gaze)과 외양(appearance)을 위한 인코더를 완전히 분리**한 것입니다. 이전 SOTA인 Cross-Encoder(Sun et al., ICCV 2021)는 하나의 공유 인코더로 두 속성을 동시에 처리했는데, 이 구조는 디코더가 타깃의 외양 정보에 접근할 수 있어 gaze embedding에 외양 정보가 유출(leakage)되는 문제가 있었습니다.

소스 프레임 **$x_s$**와 타깃 프레임 **$x_t$**가 주어졌을 때, 두 인코더의 역할은 다음과 같습니다:

**Appearance Encoder** `f_app`:
- 입력: 소스 프레임 $x_s$
- 출력: 외양 특징 맵 **$A_s \in \mathbb{R}^{H \times W \times C_a}$**
- 특징: **얕은(shallow) 구조**로 설계하여 입력 이미지의 2D 공간 구조를 최대한 보존합니다. 외양은 픽셀 수준의 공간적·구체적 속성(눈꺼풀 형태, 홍채 색상, 피부 텍스처 등)이므로, 과도한 추상화 없이 공간 해상도를 유지하는 것이 중요합니다.

**Gaze Encoder** `f_gaze`:
- 입력: 타깃 프레임 $x_t$
- 출력: 시선 임베딩 벡터 **$g_t \in \mathbb{R}^{C_g}$**
- 특징: MobileNetV2의 **Inverted Bottleneck Block** 기반 **경량 설계** (342K 파라미터, 55 MFLOPs). 시선은 전체 이미지에서 2~3개의 실수값 (yaw, pitch 등)으로 표현되는 추상적·비공간적 속성이므로, 공간 정보를 압축하여 글로벌 벡터로 변환하는 깊은 인코더가 적합합니다.

수식으로 표현하면:

$$
A_s = f_{\text{app}}(x_s), \quad g_t = f_{\text{gaze}}(x_t) \tag{1}
$$

**이 비대칭 설계가 왜 중요한가?** 핵심은 **추론 시에는 gaze encoder만 사용**된다는 점입니다. 즉, appearance encoder와 decoder는 학습 과정에서만 시선 표현의 품질을 높이기 위한 "scaffolding(비계)"으로 기능하며, 배포 시에는 342K 파라미터짜리 경량 gaze encoder만 남습니다. 이 설계 덕분에 무거운 appearance encoder를 마음껏 사용하면서도 런타임 패널티가 전혀 없는, 학습/추론 비대칭의 이점을 극대화합니다.

---

### 4.4 모듈 ②: 시선 조건부 글로벌 모듈레이션 (Gaze-Conditioned Global Modulation)

이 모듈은 Figure 1 중앙에 위치한 Self-Attention → Cross-Attention 블록에 해당하며, "소스의 외양을 타깃의 시선 방향으로 어떻게 변환할 것인가?"라는 핵심 질문에 대한 답입니다. 크게 세 단계로 이루어집니다.

#### Step 1: Self-Attention을 통한 외양 특징 정제

먼저 appearance encoder에서 출력된 외양 특징 맵 **$A_s$**에 대해 **multi-head self-attention**을 적용합니다:

$$
A_s' = \text{SelfAttn}(A_s) \tag{2}
$$

여기서 self-attention은 외양 특징 맵 내부의 공간적 관계(spatial interactions)를 모델링합니다. 예를 들어, 홍채 영역과 눈꺼풀 경계 간의 상대적 위치 관계, 또는 동공 반사(glint)와 홍채의 공간적 연관성 등이 이 단계에서 캡처됩니다.

중요한 점은 이 self-attention의 **attention weight map $w \in \mathbb{R}^{H \times W}$**가 단순히 특징 정제에만 사용되는 것이 아니라, 후술할 Gaze-Focused Loss에서 **시선 관련 영역의 소프트 마스크**로 재활용된다는 것입니다. 이것이 이 논문의 가장 핵심적인 아이디어 중 하나입니다 (Figure 1 하단의 Attention Map → Loss 연결 화살표).

#### Step 2: Cross-Attention을 통한 시선 정보 주입

다음으로, 타깃의 시선 임베딩 **g_t**를 외양 특징에 주입합니다. 이때 직접적인 concatenation이나 element-wise multiplication 대신 **cross-attention 메커니즘**을 사용하는데, 여기서 핵심적인 설계 결정이 있습니다:

1. 시선 임베딩 **$g_t \in \mathbb{R}^{C_g}$** 를 선형 투영하여 외양 특징과 동일한 차원의 **단일 글로벌 쿼리 $q_g \in \mathbb{R}^{C_a}$** 로 변환합니다.
2. 이 단일 쿼리 q_g를 Query로, 정제된 외양 특징 $A_s'$를 Key와 Value로 사용하여 cross-attention을 수행합니다:


$$
c = \text{CrossAttn}(q_g, \; A_s', \; A_s') \tag{3}
$$


이 연산의 출력 **$c \in \mathbb{R}^{C_a}$** 는 "타깃 시선 방향의 관점에서 소스 외양 특징을 어떻게 전역적으로 조절해야 하는가"를 인코딩한 **gaze-conditioned global context vector**입니다.

**왜 단일 쿼리(single query)인가?** 시선은 프레임 전체에 대해 하나의 방향으로 정의되는 글로벌 속성이므로, 공간적으로 다양한 다수의 쿼리가 아닌 단일 글로벌 쿼리로 충분합니다. 이렇게 하면 시선 정보가 공간 해상도와 무관하게 전역적(global)으로 외양 특징을 조절할 수 있습니다.

#### Step 3: 잔차 연결을 통한 특징 융합

마지막으로, 글로벌 컨텍스트 벡터 **$c$**를 공간 차원 $H \times W$로 브로드캐스트하여 **$C \in \mathbb{R}^{H \times W \times C_a}$** 를 만들고, 이를 원래의 정제된 외양 특징 $A_s'$에 잔차(residual)로 더합니다:


$$
F = A_s' + C \tag{4}
$$


이 잔차 덧셈은 **feature-wise global modulation**으로 작용합니다: 시선 방향 정보(C)가 외양 특징의 전체 공간에 균일하게 더해지면서, 잠재 표현을 타깃 시선 방향으로 "밀어내는(steering)" 역할을 합니다. 동시에 잔차 연결 덕분에 소스의 공간적 구조(눈꺼풀 형태, 피부 텍스처 등)는 원래대로 보존됩니다.

**정보 유출 차단 메커니즘:** 이 cross-attention 구조에서 gaze encoder의 출력은 오직 쿼리 위치에만 관여하고, decoder로의 직접적인 경로(skip connection 등)가 없습니다. 이 구조적 격리(architectural isolation)가 **버퍼 레이어** 역할을 하여, Cross-Encoder에서 문제가 되었던 외양 정보의 시선 임베딩 유출을 원천 차단합니다. 결과적으로 gaze encoder는 순수한 시선 정보만을 인코딩하도록 강제됩니다.

융합된 특징 맵 **F**는 이후 Decoder에 입력되어, 소스의 외양을 유지하면서 타깃의 시선 방향으로 변환된 이미지 **$\hat{x}_t$**를 생성합니다.

---

### 4.5 모듈 ③: 시선 집중 재구성 손실 (Gaze-Focused Reconstruction Loss)

이 부분이 GazeShift의 **가장 독창적인 기여**이며, ablation study에서도 가장 큰 성능 향상(2.07° → 1.84°, **0.23° 감소**)을 가져온 핵심 컴포넌트입니다.

#### 문제: 균일한 픽셀 손실의 한계

일반적인 시선 리디렉션 모델은 재구성된 이미지 $\hat{x}_t$와 실제 타깃 $x_t$ 간의 **per-pixel MSE**를 손실 함수로 사용합니다:


$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i} (x_{t,i} - \hat{x}_{t,i})^2
$$


이 손실은 모든 픽셀을 균등하게 취급합니다. 그런데 눈 이미지에서 시선 변화에 따라 실질적으로 외양이 변하는 영역은 **홍채와 동공 주변의 제한된 영역**뿐이고, 눈꺼풀 경계, 피부, 배경 등은 시선과 무관하게 거의 동일합니다. 따라서 균일한 MSE 손실을 사용하면:

1. 모델이 시선과 무관한 배경/경계 영역까지 완벽히 재구성하려고 불필요한 capacity를 소모합니다.
2. 결과적으로 gaze embedding에 시선 외 외양 정보(텍스처, 조명 등)가 혼입되어 시선 특이성(gaze specificity)이 저하됩니다.

기존 연구들은 이 문제를 해결하기 위해 **외부 detector로 생성한 눈 마스크**나 **수작업 기하학적 프라이어**를 사용했습니다. 하지만 GazeShift는 외부 모듈 없이, 모델 자체의 내부 표현을 활용하는 훨씬 우아한 방법을 제안합니다.

#### 해법: Self-Attention 맵의 재활용

핵심 아이디어: 4.4절의 self-attention 단계(Step 1)에서 생성된 **attention weight map $w \in \mathbb{R}^{H \times W}$** 를 손실 함수의 공간적 가중치로 재활용합니다.

이것이 가능한 이유는, 소스-타깃 간 외양 차이의 대부분이 시선 변화에 기인한다는 가정 하에서, self-attention이 **자연스럽게 시선 관련 영역(홍채, 동공 주변)에 높은 가중치를 부여**하기 때문입니다. Figure 4의 attention map 시각화가 이를 명확히 보여줍니다.

![어텐션 맵 시각화](/images/gazeshift/_page_6_Picture_9.jpeg)
*Figure 4: 소스 외양 이미지(상단)와 대응하는 self-attention 맵(하단). 모델이 외부 supervision 없이도 홍채와 동공 주변의 시선 관련 영역에 자연스럽게 높은 attention 가중치를 부여하는 것을 확인할 수 있습니다. 이 맵이 Gaze-Focused Loss의 가중치로 직접 사용됩니다.*

attention weight map **w**를 타깃 이미지와 동일한 해상도로 업샘플링한 후, **sharpening parameter γ > 0**를 적용한 **Gaze-Focused Reconstruction Loss**는 다음과 같이 정의됩니다:


$$
\mathcal{L}_{\text{focus}} = \frac{1}{\sum_{i} w_i^{\gamma}} \sum_{i} w_i^{\gamma} \cdot (x_{t,i} - \hat{x}_{t,i})^2 \tag{5}
$$


여기서:
- **i**: 픽셀 위치 인덱스
- **$w_i$**: 업샘플된 attention 가중치 (홍채 주변에서 높은 값, 배경에서 낮은 값)
- **$\gamma$**: attention 집중도를 제어하는 sharpening 파라미터
- 정규화 항 $1/\sum w_i^{\gamma}$ 는 전체 가중치 합이 1이 되도록 보장

#### γ의 역할과 그래디언트 관점의 해석

$\gamma$ 값에 따른 동작을 직관적으로 이해하기 위해, 정규화된 가중치 $\tilde{w}_i = w_i^{\gamma} / \sum_j w_j^{\gamma}$ 에 대한 **픽셀별 그래디언트 크기**를 살펴보면:


$$
\left\| \frac{\partial \mathcal{L}_{\text{focus}}}{\partial \hat{x}_i} \right\| \propto \tilde{w}_i = \frac{w_i^{\gamma}}{\sum_j w_j^{\gamma}}
$$

즉, **$\gamma$를 증가시키면** 높은 attention을 받는 영역(시선 관련)의 그래디언트가 증폭되고, 낮은 attention 영역(배경)의 그래디언트는 억제됩니다. 이를 통해:

- **$\gamma = 1$**: 모델의 raw attention을 그대로 가중치로 사용. 시선 영역에 적절히 집중하면서도 주변 맥락(눈꺼풀 경계, 글린트 등)을 놓치지 않는 균형점 → **최적 성능(1.84°)**
- **$\gamma > 1$** (e.g., 2.0, 4.0): attention이 과도하게 좁은 영역(동공 중심)에 집중 → 주변 맥락 정보 상실 → 성능 저하 (2.19°, 2.41°)
- **$\gamma < 1$** (e.g., 0.5): attention이 지나치게 분산 → 시선/외양 신호가 혼재 → 성능 저하 (2.03°)

#### 양성 피드백 루프 (Positive Feedback Loop)

이 설계의 가장 우아한 점은 **self-reinforcing feedback loop**가 자연스럽게 형성된다는 것입니다:

1. Self-attention이 시선 관련 영역에 집중 → 해당 영역의 재구성 손실이 강화됨
2. 강화된 손실이 gaze encoder의 시선 표현 정밀도를 높임
3. 더 정밀한 시선 표현이 다시 attention의 시선 영역 로컬리제이션을 개선
4. (1로 돌아가 반복)

이 루프 덕분에 별도의 regularization이나 외부 supervision 없이도 attention map이 학습 과정에서 점진적으로 시선 관련 영역에 수렴합니다.

---

### 4.6 시선 캘리브레이션

비지도 사전학습 후, gaze encoder가 출력하는 임베딩은 시선 방향을 인코딩하지만, 실제 각도 값(yaw, pitch)으로의 매핑은 아직 학습되지 않은 상태입니다. 이를 위해 소량의 레이블 데이터를 사용한 캘리브레이션을 수행합니다.

| 설정 | 방법 | 상세 |
|------|------|------|
| **VR (Per-person)** | Ridge Regression | 소량의 고정 시선 포인트(17~60개)로 선형 리그레서 학습. **κ-angle**(광축과 시축의 차이)이 개인마다 다르므로 per-person calibration이 필수. 헤드셋 착용 위치 변화에 대응하기 위해 세션별로도 수행 |
| **원거리 카메라 (Person-agnostic)** | MLP Regressor | 전체 피험자에서 100~200개 레이블 샘플을 풀링하여 소형 MLP 학습. 개인별 보정 없이 범용적으로 적용 |

캘리브레이션은 학습과 완전히 분리된 별도 단계이며, gaze encoder의 가중치는 고정된 상태에서 진행됩니다. 이는 self-supervised learning에서의 linear probing과 유사한 접근입니다.

---

## 5. 실험 결과

### 5.1 VRGaze에서의 성능

| 학습 방식 | 방법 | 캘리브레이션 | 평균 오차 (°) |
|-----------|------|-------------|--------------|
| 지도 학습 | Appearance Based | - | **1.54** |
| 지도 학습 | Feature Based | - | 3.20 |
| 비지도 | VAE | Per-person | 5.30 |
| 비지도 | Cross-Encoder | Per-person | 2.15 |
| **비지도** | **GazeShift** | **Per-person** | **1.84** |
| 비지도 | Cross-Encoder | Person-agnostic (K=200) | 2.26 |
| **비지도** | **GazeShift** | **Person-agnostic (K=200)** | **2.13** |

**핵심 결과**: GazeShift는 비지도 방법임에도 불구하고, 지도 학습 기반 appearance-based 모델(1.54°)에 근접한 **1.84°**를 달성했습니다. 이는 시선 레이블 없이도 실용적 수준의 정확도에 도달할 수 있음을 보여줍니다.

### 5.2 OpenEDS2020에서의 성능

| 방법 | Per-person 오차 (°) | Person-agnostic 오차 (°) |
|------|---------------------|--------------------------|
| Cross-Encoder | 3.69 | 5.20 |
| **GazeShift** | **3.43** | **4.20** |

On-axis 환경에서도 GazeShift가 일관되게 Cross-Encoder를 능가합니다.

### 5.3 Cross-Dataset 일반화 (On-axis → Off-axis)

On-axis 데이터(OpenEDS2020)로 학습 후 off-axis 데이터(VRGaze)로 테스트하면 오차가 **5.2°**로 크게 증가합니다 (VRGaze로 직접 학습 시 1.84°). 이는 **off-axis 전용 데이터셋의 필요성**을 명확히 보여줍니다.

### 5.4 원거리 카메라 실험 (MPIIGaze)

| 학습 방식 | 방법 | 평균 오차 (°) | 파라미터 | FLOPs |
|-----------|------|--------------|---------|-------|
| 지도 학습 | ResNet-18 | 8.35 | 11M | 75M |
| 비지도 | Cross-Encoder | 8.32 | 11M | 75M |
| **비지도** | **GazeShift (MobileNetV2)** | **8.00** | **1M** | **2M** |
| 비지도 | GazeShift (ResNet-18) | **7.56** | 11M | 75M |

**비지도 fine-tuning 후 (MPIIGaze에서 학습 + 평가)**:

| 방법 | 평균 오차 (°) | 파라미터 |
|------|--------------|---------|
| Cross-Encoder | 7.20 | 11M |
| **GazeShift (MobileNetV2)** | **7.15** | **1M** |

GazeShift는 **10배 적은 파라미터, 35배 적은 FLOPs**로 Cross-Encoder를 능가합니다.

### 5.5 Ablation Study

논문의 ablation 결과가 각 컴포넌트의 기여도를 깔끔하게 보여줍니다:

| # | 분리 인코더 | 어텐션 기반 리디렉션 | 시선 집중 손실 | 평균 오차 (°) |
|---|-----------|-------------------|-------------|--------------|
| 1 | ✗ | ✗ | ✗ | 2.15 |
| 2 | ✓ | ✗ | ✗ | 2.10 |
| 3 | ✓ | ✓ | ✗ | 2.07 |
| 4 | ✓ | ✓ | ✓ | **1.84** |

**시선 집중 손실(Gaze-Focused Loss)**이 가장 큰 성능 향상(2.07° → 1.84°)을 가져왔으며, 이는 이 논문의 가장 독창적인 기여입니다.

### 5.6 γ 민감도 분석

| γ | 0.5 | 1.0 | 2.0 | 4.0 |
|---|-----|-----|-----|-----|
| 평균 오차 (°) | 2.03 | **1.84** | 2.19 | 2.41 |

- γ > 1: 어텐션이 너무 좁은 영역에 집중 → 주변 맥락 정보(눈꺼풀 경계, 글린트 등) 무시
- γ < 1: 어텐션이 지나치게 분산 → 시선/외양 신호 혼재
- **γ = 1이 최적**: raw attention을 그대로 사용할 때 가장 균형 잡힌 결과

### 5.7 모델 효율성 및 온디바이스 성능

| 항목 | 수치 |
|------|------|
| Gaze Encoder 파라미터 | **342K** |
| FLOPs | **55 MFLOPs** |
| VR 헤드셋 추론 시간 (양쪽 눈) | **5 ms** |
| 칩셋 | Exynos 2200 (Xclipse 920 GPU) |

### 5.8 시선-외양 분리(Disentanglement) 검증

![분리 분석](/images/gazeshift/_page_5_Figure_10.jpeg)
*Figure 3: 외양 변화 시 시선 임베딩 안정(좌), 시선 변화 시 외양 임베딩 안정(우)*

- **외양 변화** (조명·대비 100가지 변형, 시선 고정): 시선 임베딩 코사인 거리 = 0.08 (안정적), 외양 임베딩 = 0.12
- **시선 변화** (동일 눈 80개 다른 시선 방향): 시선 임베딩 = 0.17 (민감), 외양 임베딩 = 0.04 (안정적)

→ GazeShift가 **시선과 외양을 효과적으로 분리**하고 있음을 정량적으로 확인

### 5.9 잠재 공간 보간 (Latent Space Interpolation)

![잠재 공간 보간](/images/gazeshift/_page_7_Figure_11.jpeg)
*Figure 5: 두 타깃 시선 임베딩 간 보간 시 매끄러운 눈 움직임이 생성되며, 소스의 고유한 외양 특성이 보존됩니다.*

---

## 6. 한계점

저자들이 제시한 한계점:
- **가정의 제약**: 프레임 간 외양 변화의 대부분이 시선 변화라는 가정은 VR에서는 잘 성립하지만, **AR/MR 환경**에서는 외부 조명과 반사의 변화가 크기 때문에 추가 검증이 필요
- **비시선 외양 변화**: 깜빡임, 눈꺼풀 움직임, 동공 확대 등은 VR에서 저빈도이므로 큰 영향 없지만, 다른 환경에서는 미지수

---

## 7. 총평 및 개인적 의견

### 강점

1. **데이터셋 기여의 가치**: VRGaze는 최초의 대규모 off-axis VR 시선 데이터셋으로, 이 분야의 벤치마크 공백을 메우는 중요한 자원입니다. 68명, 210만 장이라는 규모도 충분히 의미 있습니다.

2. **단순하면서도 효과적인 아키텍처**: 복잡한 기하학적 프라이어나 warping field 없이, 표준 어텐션 모듈만으로 시선 리디렉션을 달성한 점이 인상적입니다. 아키텍처적 ad-hoc 설계를 배제했기에 일반화 가능성이 높습니다.

3. **시선 집중 손실의 독창성**: 모델 자체의 어텐션 맵을 손실 함수의 가중치로 재활용하는 셀프 가이드 메커니즘은 추가 모듈 없이도 시선 영역에 대한 학습을 강화합니다. Ablation에서 가장 큰 성능 향상(0.23°)을 가져온 점이 이를 뒷받침합니다.

4. **실용성**: 342K 파라미터의 경량 gaze encoder와 5ms 추론은 실제 VR 디바이스 배포에 적합합니다. 학계 논문에서 실제 디바이스에서의 추론 시간까지 측정한 점이 좋습니다.

### 아쉬운 점 / 향후 과제

1. **데이터셋 다양성**: 68명 중 여성 14명(20.6%), 아시아 29 / 백인 39라는 인구통계 구성은 아직 제한적입니다. 다양한 인종·연령 그룹에서의 성능 분석이 추가되면 좋겠습니다.

2. **비교 베이스라인 제한**: Off-axis VR에서의 비교 대상이 Cross-Encoder와 VAE로 한정되어 있습니다. 물론 저자들도 언급했듯이 재현 가능한 베이스라인이 제한적인 현실적 이유가 있긴 합니다.

3. **AR/MR 확장**: 논문이 한계점으로 인정한 AR/MR 환경에서의 비시선 외양 변화 문제는 향후 가장 중요한 연구 방향으로 보입니다. VR의 통제된 환경을 벗어나면 핵심 가정이 약해지기 때문입니다.

4. **Binocular 활용**: 지도 학습 베이스라인은 binocular Siamese 네트워크를 사용하는 반면, GazeShift는 단안(monocular) 기반입니다. 양안 정보를 활용하면 추가적인 성능 향상이 가능할 수 있습니다.

### 결론

GazeShift는 **"비지도 + 경량 + 실시간"** 이라는 세 가지 실용적 요소를 동시에 달성하면서도, 지도 학습에 근접하는 정확도를 보여주는 인상적인 연구입니다. 특히, 표준 어텐션 메커니즘만으로 시선-외양 분리를 달성한 설계 철학과, self-attention 맵을 손실 함수 가중치로 재활용하는 아이디어는 시선 추정을 넘어 다른 표현 학습 문제에도 적용 가능한 일반적인 프레임워크를 제시합니다.

---

> **참고 문헌**은 원 논문의 References 섹션을 참조하시기 바랍니다.
