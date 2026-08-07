---
title: "[TCSVT 2025] Vivim: 초음파 영상 분할을 위한 Video Vision Mamba"
date: 2026-08-07T20:02:00+09:00
draft: false
math: true
tags: ["Paper Review", "Medical Image Segmentation", "State Space Model", "Mamba", "Video Segmentation", "Ultrasound", "TCSVT 2025"]
categories: ["Paper Review"]
summary: "State Space Model(Mamba)을 초음파 비디오 분할에 최초로 도입하여, Transformer 대비 선형 복잡도로 시공간 장거리 의존성을 효과적으로 모델링하는 Vivim 프레임워크를 제안합니다."
cover:
  image: "/images/vivim/overview.jpeg"
  alt: "Vivim 프레임워크 개요도"
---

## 1. 한 줄 요약

Mamba 기반의 Spatiotemporal Selective Scan(ST-Mamba)을 계층적 Transformer 구조에 통합하여, 초음파 비디오의 시공간 장거리 의존성을 선형 복잡도로 모델링하고, 경계 인식 아핀 제약(Boundary-Aware Affine Constraint)으로 모호한 병변 경계의 분할 정확도를 향상시키는 Vivim 프레임워크를 제안합니다.

---

## 2. 연구 배경 및 동기

### 2.1 문제 정의

초음파 영상에서 병변과 조직의 자동 분할은 컴퓨터 보조 진단에 필수적이지만, 다음과 같은 고유한 어려움이 존재합니다.

![Figure 1: 초음파 비디오 분할의 주요 과제](/images/vivim/_page_1_Figure_2.jpeg)
*Figure 1: 초음파 비디오 분할의 주요 과제. (a) 저대비와 스페클 노이즈로 인한 모호한 병변 경계, (b) 환자 간 병변의 비균질적 분포, (c) 프로브 이동과 연조직 변형에 의한 프레임 간 동적 변화.*

- 저대비(low contrast)와 스페클 노이즈(speckle noise)로 인해 병변 경계가 모호합니다.
- 환자별 초음파 설정과 해부학적 차이로 병변의 외관이 크게 달라져 일반화가 어렵습니다.
- 프로브 이동과 연조직 변형으로 프레임 간 시간적 불일치가 발생하여 단순 프레임 단위 분할로는 해결이 어렵습니다.

이러한 과제를 해결하려면 시공간 의존성(spatiotemporal dependency)을 동시에 고려하는 강건한 접근법이 필요합니다.

### 2.2 기존 방법의 한계

CNN 기반 방법은 제한된 수용 영역(receptive field)으로 인해 전역 정보 포착에 한계가 있습니다. Transformer 기반 방법은 Multi-Head Self-Attention(MSA)을 통해 전역 정보를 추출할 수 있지만, 시간 차원의 self-attention 모듈을 추가하면 시간 축에 대해 복잡도가 이차적(quadratic)으로 증가합니다. 구체적으로, 영상 시각 시퀀스 $\mathbf{K} \in \mathbb{R}^{1 \times T \times M \times D}$에 대해 global self-attention의 계산 복잡도는 다음과 같습니다:

$$\Omega(\text{self-attention}) = 4(TM)D^2 + 2(TM)^2D$$

이는 전체 비디오 시퀀스 길이 $TM$에 대해 이차 복잡도를 가지며, 메모리가 제한된 의료 환경에서 긴 시퀀스를 처리할 때 심각한 병목이 됩니다. 예를 들어, DPSTT는 과적합 방지를 위해 상당한 데이터 증강이 필요하고 처리 속도가 느리며, FLA-Net은 많은 메모리를 점유합니다.

### 2.3 핵심 기여

이 논문의 핵심 기여는 다음과 같습니다:

1. SSM(State Space Model)을 초음파 비디오 분할에 최초로 도입한 Mamba 기반 인코더 + CNN 기반 디코더 프레임워크(Vivim)를 제안합니다.
2. 단순한 Mamba 적용이 아닌, 시공간 선택적 스캔(Spatiotemporal Selective Scan)을 설계하여 Temporal Mamba Block의 전역 인식 능력을 강화합니다.
3. 아핀 변환 최적화에 기반한 개선된 경계 인식 제약(Boundary-Aware Affine Constraint)을 도입하여 모호한 경계 예측을 개선합니다.
4. 픽셀 수준 어노테이션이 포함된 최초의 비디오 초음파 갑상선 분할 데이터셋 VTUS(100개 비디오, 9,342 프레임)를 구축하여 벤치마크 평가를 가능하게 합니다.

---

## 3. 제안 방법: Vivim 프레임워크

### 3.1 전체 구조 개요

![Figure 3: Vivim 프레임워크 개요](/images/vivim/_page_3_Figure_2.jpeg)
*Figure 3: (a) Vivim의 전체 구조. 비디오 시퀀스를 패치 임베딩과 다중 스케일 Temporal Mamba Block으로 인코딩한 뒤, CNN 기반 세그멘테이션 헤드로 분할 결과를 예측합니다. (b) Temporal Mamba Block의 구성. (c) ST-Mamba의 다방향 시공간 선택적 스캔 메커니즘.*

Vivim은 크게 두 가지 모듈로 구성됩니다:

- 계층적 인코더: 다중 Temporal Mamba Block을 쌓아 다양한 스케일에서 시공간 특징 시퀀스를 추출합니다.
- 경량 CNN 기반 세그멘테이션 헤드: 다수준 특징 시퀀스를 융합하여 분할 마스크를 예측합니다.

입력 비디오 클립 $\mathbf{V} = \{I^1, \dots, I^T\}$의 각 프레임을 $4 \times 4$ 크기의 오버랩 패치 임베딩으로 분할한 뒤, 계층적 Temporal Mamba 인코더에 입력하여 원본 프레임 대비 $\{1/4, 1/8, 1/16, 1/32\}$ 해상도의 다수준 시공간 특징을 얻습니다. 이후 CNN 기반 세그멘테이션 헤드에서 최종 분할 결과를 예측합니다.

### 3.2 State Space Model 기초 (수식 원리와 물리적 변환 개념)

State Space Model(SSM)은 1차원 함수 또는 시퀀스 $x(t) \in \mathbb{R}$를 은닉 상태 $h(t) \in \mathbb{R}^N$을 거쳐 출력 $y(t) \in \mathbb{R}$로 매핑하는 선형 시불변 시스템입니다. 이 시스템은 진화 파라미터 $\mathbf{A} \in \mathbb{R}^{N \times N}$과 프로젝션 파라미터 $\mathbf{B} \in \mathbb{R}^{N \times 1}$, $\mathbf{C} \in \mathbb{R}^{1 \times N}$으로 구성되며, 다음과 같은 선형 ODE로 정의됩니다:

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}h(t)$$

여기서 ODE는 상미분방정식(Ordinary Differential Equation)의 약자로, 시간에 따라 변화하는 물리량이나 신호의 변화 속도를 미분을 이용해 서술한 수학 공식입니다. SSM은 연속적으로 들어오는 비디오 신호와 은닉 상태 간의 동적인 연속 변화를 이 미분방정식을 빌려 설계합니다.

연속 파라미터를 컴퓨터가 연산할 수 있도록 이산 파라미터(디지털 신호)로 바꾸어 주는 물리적 변환 작업을 수행해야 하는데, 이때 타임스케일 파라미터 $\Delta$와 Zero-Order Hold(ZOH) 기법을 사용합니다:

$$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

- 타임스케일 파라미터 델타($\Delta$): 연속적인 시간 흐름 속에서 입력을 관찰하는 시간적인 샘플링 간격(보폭)을 의미합니다.
- Zero-Order Hold(ZOH, 영차홀드): 컴퓨터가 연속 신호를 디지털로 근사 변환하는 대표적인 방식입니다. 특정 시간 간격 $\Delta$ 동안 연속적인 신호 값이 변하지 않고 이전 입력값으로 일정하게 유지(Hold)된다고 단순화하여 물리적 신호를 채워 넣고 변환하는 기법입니다.

이산화 과정을 거친 SSM의 수식 표현은 다음과 같습니다:

$$h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$$

Mamba(S6)는 여기에 선택적 스캔(selective scan) 메커니즘을 도입하여 입력값에 맞추어 실시간으로 변화하는 파라미터화를 구현하고, 긴 시퀀스 간의 상호 의존 관계를 선형 복잡도로 매우 빠르게 연산합니다.

### 3.3 Temporal Mamba Block

Temporal Mamba Block은 Vivim 인코더의 핵심 구성 요소로, 시공간 정보를 동시에 활용합니다.

1. Efficient Spatial Self-Attention: 먼저 공간 정보의 초기 집계를 수행합니다. 이때 연산량을 줄이기 위해 SegFormer에서 제안된 시퀀스 축소(sequence reduction) 기법을 적용합니다.
- 시퀀스 축소 기법: Transformer의 셀프 어텐션 연산은 패치(토큰) 개수가 늘어날수록 계산비용이 제곱으로 폭증합니다. 이를 완화하기 위해, 셀프 어텐션을 계산하기 전에 연산의 대상이 되는 공간 특징 맵의 크기(가로, 세로)를 핵심 특징 위주로 다운샘플링하여 전체 연산 시퀀스 길이를 대폭 압축하는 효율화 기법입니다.
2. ST-Mamba 레이어: i번째 수준의 특징 임베딩 $\mathbf{F}_i \in \mathbb{R}^{T \times C_i \times H \times W}$에서 채널과 시간 차원을 전치(transpose)하고 시공간 특징을 1D 시퀀스 $\mathbf{h}_i \in \mathbb{R}^{C_i \times THW}$로 평탄화합니다. 이 시퀀스를 ST-Mamba 모듈에 입력하여 프레임 내/프레임 간 장거리 의존성을 학습합니다.
3. Detail-Specific Feedforward(DSF): $3 \times 3 \times 3$ depth-wise 컨볼루션을 피드포워드 네트워크에 통합하여 세밀한 디테일을 보존합니다.

스택된 Mamba 레이어의 연산은 다음과 같이 정의됩니다($l \in [1, N_m]$):

$$h^l = \text{ST-Mamba}(\text{LN}(h^{l-1})) + h^{l-1}$$
$$h^l = \text{DSF}(\text{LN}(h^l)) + h^l$$

### 3.4 시공간 선택적 스캔(Spatiotemporal Selective Scan)

![Figure 4: 시공간 선택적 스캔 메커니즘](/images/vivim/_page_4_Figure_2.jpeg)
*Figure 4: 시공간 선택적 스캔의 세 가지 방향 — 시간 순방향 스캔, 시간 역방향 스캔, 공간 스캔.*

S6의 인과적(causal) 특성은 시간 데이터에 적합하지만, 비디오는 시간적 정보뿐 아니라 비인과적(non-causal) 2D 공간 정보도 포함합니다. 이를 해결하기 위해 ST-Mamba는 세 가지 방향의 스캔을 병렬로 수행합니다:

- 시간 순방향 스캔(Temporal Forward): 각 프레임의 패치를 행/열 방향으로 언폴딩하여 시퀀스를 구성한 뒤 프레임 순서로 연결하여 $\mathbf{h}_i^t \in \mathbb{R}^{C_i \times T(HW)}$를 생성합니다.
- 시간 역방향 스캔(Temporal Backward): 동일한 시퀀스를 역방향으로 스캔하여 양방향 시간 의존성을 학습합니다.
- 공간 스캔(Spatial): 패치를 시간 축을 따라 쌓아 공간 우선 시퀀스 $\mathbf{h}_i^s \in \mathbb{R}^{C_i \times (HW)T}$를 구성하고, 모든 프레임의 동일 위치 픽셀 정보를 통합합니다.

이 메커니즘은 단일 프레임의 공간적 일관성과 프레임 간 일관성을 명시적으로 고려하며, SSM의 계산 복잡도는 다음과 같이 선형입니다:

$$\Omega(\text{SSM}) = 4(TM)(2D)N + (TM)(2D)N^2$$

여기서 기본 확장 비율은 2이고 $N$은 16으로 고정됩니다. Self-attention이 전체 비디오 시퀀스 길이 $TM$에 대해 이차인 반면, SSM은 선형이므로 장기 비디오 응용에 적합합니다.

### 3.5 경계 인식 아핀 제약(Boundary-Aware Affine Constraint)

![Figure 5: 학습 전략 개요](/images/vivim/_page_4_Figure_4.jpeg)
*Figure 5: 학습 전략 개요. 패치 수준 경계 인식 아핀 제약 L_affine이 분할 손실 L_seg 및 경계 이진 교차 엔트로피 손실 L_bce와 함께 Vivim을 최적화합니다.*

분할 지도 학습만으로는 모호하고 비구조적인 예측이 생기기 쉽습니다. 이를 완화하기 위해 InverseForm에서 영감을 받은 패치 수준 경계 인식 아핀 제약을 도입합니다.

구체적으로, Ground Truth 마스크에 Sobel 연산자를 적용하여 실제 물체의 경계선(GT 에지)을 추출합니다.
- Sobel(소벨) 연산자: 이미지 안의 각 픽셀 주변에서 밝기 값이 가로 및 세로 방향으로 얼마나 급격히 변하는지 계산하여, 사물이나 병변의 윤곽선(에지)을 추출해내는 고전적이고 직관적인 이미지 처리 필터링 기술입니다.

또한 보조 경계 헤드가 Mamba 인코더의 특징 패치에서 예측 에지를 생성합니다.
- 보조 경계 헤드: 네트워크가 영역뿐만 아니라 외각선(경계선) 자체를 정밀하게 그릴 수 있도록 유도하는 학습용 미니 신경망입니다. Mamba 인코더 특징 맵에 3개의 합성곱(convolutional) 레이어를 가볍게 추가하여 경계선을 예측하게 만듭니다. 이는 오직 모델 학습을 돕는 보조 감시 장치로 활용되며 학습이 끝난 추론 시에는 제거됩니다.

이후 사전 학습된 MLP를 통해 실제 경계선과 예측 경계선 간의 일치도를 아핀 변환 행렬을 이용해 추정 및 강제합니다.
- 아핀 변환(Affine Transformation): 이미지를 회전시키거나, 늘이고 줄이고, 대칭하거나, 평행 이동시키는 등 기하학적인 형태 변화를 유발하는 행렬 변환입니다.
- 아핀 제약 메커니즘: 모델이 예측한 테두리가 실제 정답 테두리와 비교했을 때 얼마나 찌그러졌거나 어긋났는지를 가상의 아핀 변환 행렬로 수학적으로 파악한 다음, 이 변환 관계가 변형이 아예 없는 항등 행렬(Identity Matrix)이 되도록 손실 함수 제약을 가합니다. 이를 통해 윤곽선 모양의 불일치와 미세한 어긋남을 제거합니다.

구체적으로 두 가지 아핀 변환 행렬을 계산합니다:
- $\hat{\theta}_i^t$: 타겟 프레임 $I^t$의 GT 에지 $B_{\text{gt}}^t$와 예측 에지 $B_{\text{pred}}^t$ 간의 아핀 변환 행렬
- $\hat{\theta}_i^1$: 첫 번째 프레임 $I^1$의 GT 에지 $B_{\text{gt}}^1$와 타겟 프레임 예측 에지 $B_{\text{pred}}^t$ 간의 아핀 변환 행렬

아핀 제약 손실은 다음과 같습니다:

$$\mathcal{L}_{\text{affine}} = \frac{1}{N_p} \sum_{i=1}^{N_p} \left( \Delta_1 \left| \hat{\theta}_i^t - \mathbb{I} \right|_F - \Delta_2 \left| \hat{\theta}_i^1 - \mathbb{I} \right|_F \right)$$

여기서 $N_p$는 패치 수, $|\cdot|_F$는 프로베니우스 노름, $\Delta_1=1.00$, $\Delta_2=0.01$입니다. 이 목적 함수는 예측 에지 $B_{\text{pred}}^t$를 GT 에지 $B_{\text{gt}}^t$ 방향으로 밀어내고 첫 프레임 GT 에지 $B_{\text{gt}}^1$로부터는 멀어지도록 하여, 프레임 간 병변 구조의 미세한 차이를 보존합니다.

최종 학습 손실은 다음과 같이 구성됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_1 \mathcal{L}_{\text{affine}} + \lambda_2 \mathcal{L}_{\text{bce}}$$

단, $\lambda_1 = \lambda_2 = 0.3$입니다.

### 3.6 디코더

다수준 특징 임베딩 $\{F_1, F_2, F_3, F_4\}$을 MLP 레이어로 채널 차원을 통일한 뒤, 동일 해상도로 업샘플링하여 연결(concatenate)합니다. 이후 MLP 레이어로 융합하고 $1 \times 1$ 컨볼루션으로 최종 분할 마스크 $M$을 예측합니다.

---

## 4. 실험 결과

### 4.1 데이터셋 및 구현 세부사항

실험은 세 가지 의료 비디오 분할 과제에서 수행됩니다:

- VTUS (갑상선 초음파): 자체 수집 데이터셋. 100개 비디오(환자 1명당 1개), 총 9,342 프레임. 훈련/테스트 7:3 분할.
- BUV2022 (유방 병변 초음파): 63개 비디오, 4,619 프레임.
- CVC-300, CVC-612, ASU-Mayo (대장 내시경 용종): 기존 공개 데이터셋 활용.

구현은 NVIDIA RTX 4090 1장, PyTorch 기반, 100 에폭, Adam 옵티마이저(학습률 $10^{-4} \to 10^{-6}$), 프레임 해상도 $256 \times 256$, 배치 크기 4(클립당 5프레임)로 수행됩니다. Temporal Mamba Block의 Efficient Spatial Self-Attention과 Mix-FFN은 SegFormer의 사전 학습 가중치를 활용합니다.

### 4.2 갑상선 및 유방 병변 초음파 분할 결과

| Methods | Venue | Type | VTUS Dice | VTUS Jaccard | BUV2022 Dice | BUV2022 Jaccard | FPS |
|---|---|---|---|---|---|---|---|
| UNet | MICCAI15 | image | 0.6662 | 0.5328 | 0.7303 | 0.6247 | 88.18 |
| UNet++ | DLMIA18 | image | 0.7656 | 0.6486 | 0.7179 | 0.6124 | 40.90 |
| TransUNet | arXiv21 | image | 0.7461 | 0.6250 | 0.6547 | 0.5358 | 65.10 |
| SETR | CVPR21 | image | 0.7288 | 0.6010 | 0.6649 | 0.5480 | 21.61 |
| DPSTT | MICCAI22 | video | 0.8063 | 0.7117 | 0.8255 | 0.7364 | 30.50 |
| FLA-Net | MICCAI23 | video | 0.8042 | 0.7075 | 0.8232 | 0.7315 | 31.22 |
| MemSAM | CVPR24 | video | 0.7922 | 0.7101 | 0.8149 | 0.7092 | 10.42 |
| Vivim (Ours) | — | video | 0.8324 | 0.7391 | 0.8356 | 0.7450 | 35.33 |

*Table I: VTUS 및 BUV2022 데이터셋에서의 정량적 비교. Vivim은 모든 지표에서 SOTA를 달성합니다.*

Vivim은 VTUS에서 2위 대비 Dice +2.61%, Jaccard +2.74%, BUV2022에서 Dice +1.01%, Jaccard +0.86%의 큰 폭의 성능 향상을 달성합니다. 동시에 비디오 기반 방법 중 가장 빠른 35.33 FPS의 추론 속도를 기록합니다.

![Figure 6: 갑상선 분할 시각적 비교](/images/vivim/_page_7_Figure_2.jpeg)
*Figure 6: 비디오 초음파 갑상선 분할의 시각적 비교. Vivim은 타겟 병변을 더 정확한 경계로 분할합니다.*

### 4.3 대장 내시경 용종 분할 결과

CVC-300-TV에서 maxDice 0.901(2위 대비 +2.7%), maxIoU 0.831(+2.2%)을 달성하며, CVC-612-V와 CVC-612-T에서도 일관되게 SOTA를 기록합니다.

![Figure 7: 용종 분할 시각적 비교](/images/vivim/_page_7_Figure_6.jpeg)
*Figure 7: CVC-612-T의 연속 프레임에서의 정성적 결과. Vivim은 더 정확한 경계로 용종을 분할합니다.*

### 4.4 어블레이션 연구

| 구성 | Tf | Tb | S | BAC | Dice | Jaccard |
|---|---|---|---|---|---|---|
| basic (SegFormer) | - | - | - | - | 0.8144 | 0.7188 |
| C1 (+순방향 시간 SSM) | check | - | - | - | 0.8159 | 0.7216 |
| C2 (+양방향 시간 SSM) | check | check | - | - | 0.8213 | 0.7264 |
| C3 (+공간 SSM) | check | check | check | - | 0.8259 | 0.7310 |
| Vivim (full) | check | check | check | check | 0.8324 | 0.7391 |

*Table III: VTUS 데이터셋에서의 어블레이션 결과. Tf: 시간 순방향 SSM, Tb: 시간 역방향 SSM, S: 공간 SSM, BAC: 경계 인식 아핀 제약.*

어블레이션 분석의 주요 관찰 결과:

- 순방향 시간 SSM(C1)만 추가해도 기본 SegFormer 대비 전반적인 성능이 향상되어, vanilla SSM이 시간적 의존성 탐색에 효과적임을 보여줍니다.
- 양방향 시간 SSM(C2)은 단방향(C1) 대비 프레임 간 일관성을 향상시킵니다.
- 공간 SSM(C3)을 추가하면 비인과적 공간 정보를 적응적으로 처리하여 Dice +0.46%, Recall +0.83%의 유의미한 향상을 달성합니다.
- 경계 인식 아핀 제약(BAC)은 최종적으로 Dice, Jaccard, Precision을 추가로 향상시킵니다.

### 4.5 ST-Mamba 효율성 분석

| Methods | Core Module | TM (M) | IM (M) | Run-time (s) | Is Global |
|---|---|---|---|---|---|
| M1 | 시공간 Self-attention | OOM | - | - | check |
| M2 | 시공간 Window Self-attention | 25,861 | 7,795 | 0.142 | ✗ |
| M3 | 시공간 Factorized Self-attention | 29,110 | 9,288 | 0.156 | ✗ |
| Vivim | 시공간 Mamba | 19,216 | 5,112 | 0.121 | check |

*Table IV: 32프레임×256^2 입력에 대한 어텐션 모듈 비교 (A6000 48GB 기준). TM: 학습 메모리, IM: 추론 메모리.*

![Figure 8: 프레임 수 증가에 따른 효율성 비교](/images/vivim/_page_8_Figure_11.jpeg)
*Figure 8: (a) 참조 프레임 수 증가에 따른 Dice 변화 — ST-Mamba는 프레임 수 증가 시에도 성능이 향상되는 반면 시공간 self-attention은 정체/하락합니다. (b) 시퀀스 길이 증가에 따른 메모리 비용 — Vivim은 RTX 4090 단일 GPU로 150프레임 이상 추론이 가능합니다.*

전역 시공간 self-attention(M1)은 32프레임×256^2 입력에서 OOM이 발생합니다. Window/Factorized self-attention(M2, M3)은 수용 영역을 타협하여 메모리 내에서 동작하지만, Vivim의 ST-Mamba는 전역 모델링을 유지하면서도 학습 메모리 19,216M, 추론 메모리 5,112M, 런타임 0.121초로 가장 효율적입니다.

---

## 5. 핵심 기여 정리

- SSM 기반 의료 비디오 분할 프레임워크의 개척: Vivim은 SSM(Mamba)을 초음파 비디오 분할 과제에 최초로 적용한 연구입니다. Mamba 기반 인코더가 시공간 전역 의존성을 선형 복잡도로 학습하고, CNN 기반 디코더가 로컬 디테일을 보존하는 하이브리드 아키텍처를 통해, Transformer 기반 방법의 이차 복잡도 문제를 근본적으로 해결합니다. 이를 통해 메모리가 제한된 의료 환경에서도 긴 비디오 시퀀스의 효과적인 분할이 가능해집니다.

- 시공간 선택적 스캔(ST-Mamba) 설계: Mamba의 인과적 특성은 시간 데이터에 적합하지만, 비디오의 비인과적 2D 공간 정보를 처리하기 어렵습니다. 이를 해결하기 위해 시간 순방향, 시간 역방향, 공간의 세 방향 병렬 스캔 메커니즘을 설계하여, 프레임 내 공간적 일관성과 프레임 간 시간적 일관성을 동시에 포착합니다. 어블레이션 결과에서 각 방향의 스캔이 점진적으로 성능을 향상시킴을 확인할 수 있습니다.

- 경계 인식 아핀 제약의 도입: 초음파 영상의 모호한 병변 경계 문제를 해결하기 위해, 예측 에지와 GT 에지 간의 아핀 변환을 항등 행렬 방향으로 최적화하는 제약을 도입합니다. 동시에 첫 프레임 GT 에지와는 적대적으로 최적화하여 프레임 간 병변 구조의 미세한 차이를 보존합니다. 이 이중 최적화 전략은 어블레이션에서 Dice, Jaccard, Precision의 추가 향상으로 그 효과가 검증됩니다.

- VTUS 데이터셋 구축: 픽셀 수준 어노테이션이 포함된 비디오 초음파 갑상선 분할 데이터셋이 공개적으로 존재하지 않았습니다. 본 연구는 3년 이상의 갑상선 진단 경험을 가진 3명의 전문가가 교차 어노테이션한 100개 비디오(9,342 프레임)의 VTUS 데이터셋을 구축하여, 초음파 비디오 분할 방법의 체계적 벤치마크 평가 기반을 마련합니다.
