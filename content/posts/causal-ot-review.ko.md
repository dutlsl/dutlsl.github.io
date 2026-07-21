---
title: "[CVPR 2026] Causal-OT: 비디오 및 시계열 데이터를 위한 불확실성 인식 비지도 도메인 적응 논문 리뷰"
date: 2026-06-30T17:30:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Optimal Transport", "CVPR 2026"]
categories: ["Paper Review"]
summary: "CVPR 2026에 발표된 Causal-OT 프레임워크 리뷰입니다."
cover:
  image: "/images/causal_ot/causal_ot_architecture.jpeg"
  alt: "Causal-OT Architecture"
---

> <b>논문 정보</b>
> - <b>제목:</b> Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport
> - <b>저자:</b> Khushboo Mishra, Varun Trivedi, Tanima Dutta
> - <b>소속:</b> Indian Institute of Technology (BHU) Varanasi, India
> - <b>학회:</b> CVPR 2026
> - <b>코드:</b> [github.com/mynameanonymous/CausalOT](https://github.com/mynameanonymous/CausalOT)

---

## 1. 한 줄 요약

시계열 및 비디오 데이터의 비지도 도메인 적응(UDA)에서, <b>Granger 인과 그래프 정규화</b>를 <b>최적 수송(Optimal Transport)</b> 정렬에 통합하고 <b>엔트로피 기반 불확실성 인식 유사-레이블링</b>을 결합한 통합 프레임워크 <b>Causal-OT</b>를 제안하여, 6개 시계열 벤치마크에서 <b>평균 4.5% 정확도 및 3.8% F1 향상</b>, 4개 비디오 벤치마크에서 <b>평균 2.5% 정확도 향상</b>을 달성한 논문.

---

## 2. 연구 배경 및 동기

### 2.1. 문제 정의

시계열 데이터(가속도계, 자이로스코프, EMG 등)는 사용자·기기·환경에 따라 도메인 간 상당한 차이를 보인다. 레이블이 있는 소스 도메인에서 학습된 모델을 레이블이 없는 타겟 도메인에 적용할 때, <b>통계적 분포 차이</b>와 <b>시간적 의존성 변화</b>라는 이중 과제가 발생한다.

### 2.2. 기존 방법의 한계

기존 UDA 접근법들은 아래 세 가지 핵심 차원을 <b>동시에</b> 다루지 못한다:

| 모델 | 분포 불일치 | 인과성 보존 | 불확실성 처리 |
|------|:---:|:---:|:---:|
| TransPL [15] | ✓ | ✗ | ✗ |
| CauDiTS [36] | ✗ | ✓ | ✗ |
| RAINCOAT [13] | ✓ | ✗ | ✗ |
| <b>Causal-OT (Ours)</b> | <b>✓</b> | <b>✓ (Granger)</b> | <b>✓ (entropy-aware)</b> |

> <b>Table 1 재구성.</b> 기존 방법들은 분포 정렬, 인과 구조 보존, 예측 불확실성 중 일부만 처리. Causal-OT는 세 가지를 모두 통합.

특히 TransPL은 높은 엔트로피 예측까지 포함한 넓은 범위의 유사-레이블에 의존하여 노이즈가 많은 감독 신호를 생성한다. 아래 그림은 타겟 도메인의 예측 불확실성이 소스보다 현저히 높음을 보여준다:

![소스 vs 타겟 도메인 불확실성 분포 비교 (SSC, MFD 데이터셋)](/images/causal_ot/uncertainty_distribution.jpeg)

---

## 3. 제안 방법: Causal-OT 프레임워크

### 3.1. 전체 아키텍처

![Causal-OT 프레임워크 전체 구조 (논문 Figure 3)](/images/causal_ot/causal_ot_architecture.jpeg)

Causal-OT의 학습 파이프라인은 다음 5단계로 구성된다:

1. <b>인과 그래프 구성 (Causal Graph Construction):</b> 소스와 타겟 도메인의 원시 시계열로부터 Granger 인과 그래프 $G_s$, $G_t$를 추출한다. 그림에서 검은 화살표는 변수 간 인과 관계, 빨간 화살표는 비인과 관계를 나타낸다.

2. <b>특징 추출 (Feature Extraction):</b> 공유 특징 추출기 $f_\theta$가 소스·타겟 데이터를 잠재 특징 공간 $Z_s$, $Z_t$로 매핑한다.

3. <b>인과 정렬 (Causal Alignment):</b> 특징 거리와 인과 그래프 거리 $D_G$를 결합한 비용 행렬 $C$를 기반으로 최적 수송(OT) 손실 $\mathcal{L}_{\text{OT}}$를 계산한다.

4. <b>분류 및 유사-레이블링:</b> 분류기 $h_\phi$가 소스 데이터로 학습되며 ($\mathcal{L}_{\text{src}}$), 타겟 도메인에서 높은 신뢰도의 유사-레이블을 생성한다 ($\mathcal{L}_{\text{PL}}$).

5. <b>최적화:</b> 세 손실의 가중합으로 모델 파라미터 $\theta$, $\phi$를 업데이트한다.

### 3.2. Granger 인과 그래프 추출

다변량 시계열 $X \in \mathbb{R}^{N \times T \times D}$에서 변수 간 방향성 인과 관계를 모델링한다. 구체적으로:

- <b>정상성 검정:</b> Augmented Dickey-Fuller 테스트로 신호의 정상성을 확인
- <b>VAR 차수 선택:</b> Bayesian Information Criterion (BIC)으로 최적 래그 차수 $p$ 결정
- <b>유의성 필터링:</b> p-value < 0.05인 엣지만 유지
- <b>행 정규화:</b> 인접 행렬 $W$를 행 합이 1이 되도록 정규화
- <b>스펙트럴 임베딩:</b> 라플라시안 $L = D - W$에서 $k$차원 임베딩 $\Phi(G) \in \mathbb{R}^{d \times k}$ 도출

> [!IMPORTANT]
> <b>하이브리드 그래프 업데이트 전략:</b> 초기 인과 그래프는 원시 데이터 $X$에서 추출하되, 학습 중 잠재 특징 $Z$에서 주기적으로 재추정하여 블렌딩한다.
>
> $$
> A^{(t)} = \alpha A_X + (1-\alpha) A_Z^{(t)}, \quad \alpha \in [0.6, 0.9]
> $$
>
> 이를 통해 안정적인 원시 신호 구조를 유지하면서 학습 진행에 따른 인과 구조 변화를 반영한다.

### 3.3. 인과-의존 최적 수송 비용

소스-타겟 샘플 쌍 간의 비용을 특징 유사도와 인과 기술자 일관성을 결합하여 정의한다:

$$
C_{ij} = \Vert f_s(x_i^s) - f_t(x_j^t) \Vert_2^2 + \lambda \Vert \phi_i^s - \phi_j^t \Vert_2^2
$$

여기서 $\phi_i^s$, $\phi_j^t$는 각각 소스·타겟 샘플의 인과 임베딩이다. 이 비용 행렬에 대해 엔트로피 정규화된 OT 문제를 Sinkhorn 반복법으로 풀어 최적 수송 계획 $\gamma^{\ast}$를 구한다:

$$
\gamma^{\ast} = \arg\min_{\gamma \in \Pi(\mu_s, \mu_t)} \langle \gamma, C \rangle + \varepsilon H(\gamma)
$$

이 설계의 핵심은 <b>인과 그래프 제약을 비용 행렬에 내장</b>하여, 정렬이 기하학적으로 의미있을 뿐 아니라 시간적 의존성과도 구조적으로 일관되게 하는 것이다.

### 3.4. 불확실성 인식 유사-레이블링

타겟 도메인의 인코딩된 특징 $Z_t^j$로부터 분류기 $h_\phi$의 소프트 클래스 예측을 구하고, 예측 분포의 <b>엔트로피</b>로 불확실성을 추정한다:

$$
U_t^j = -\sum_{k=1}^K \hat{y}_{t,j}^{(k)} \log \hat{y}_{t,j}^{(k)}
$$

엔트로피 임계값 $\rho$ 이하인 <b>저엔트로피(고신뢰도)</b> 샘플만 유사-레이블로 사용하여 노이즈 전파를 방지한다.

### 3.5. 전체 손실 함수

전체 손실 함수는 세 가지 항목의 가중합으로 구성됩니다:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{src}} + \alpha \mathcal{L}_{\text{OT}} + \beta \mathcal{L}_{\text{PL}}
$$

- <b>소스 도메인 분류 손실 ($\mathcal{L}_{\text{src}}$)</b>: Cross-Entropy

  $$
  \mathcal{L}_{\text{src}} = \frac{1}{N_s}\sum_i \text{CE}(h_\phi(Z_s^i), y_s^i)
  $$

- <b>인과 구조 보존 OT 정렬 손실 ($\mathcal{L}_{\text{OT}}$)</b>:

  $$
  \mathcal{L}_{\text{OT}} = \langle \gamma^{\ast}, C \rangle
  $$

- <b>불확실성 필터링 유사-레이블 손실 ($\mathcal{L}_{\text{PL}}$)</b>:

  $$
  \mathcal{L}_{\text{PL}} = \frac{1}{| I |}\sum_{j \in I} \text{CE}(h_\phi(Z_t^j), \hat{y}_t^j)
  $$

### 3.6. 이론적 분석

<b>Proposition 1 (Causal-OT 타겟 위험 상한):</b> Lipschitz 연속이고 유계인 대리 손실 $\ell$에 대해:

$$
\mathcal{R}_t(h) \leq \mathcal{R}_s(h) + L \, \mathbb{E}_{(x_s, x_t) \sim \gamma^{\ast}}[\Vert f_s(x_s) - f_t(x_t) \Vert] + \lambda \, \mathbb{E}_{(i,j) \sim \gamma^{\ast}}[\Vert \phi_i^s - \phi_j^t \Vert] + \mathcal{D}_{\mathcal{H}}(P_s, P_t)
$$

이 부등식은 Causal-OT 목적함수를 최소화하면 <b>특징 수준 거리, 인과 구조 불일치, 가설 불일치</b>를 동시에 줄여 도메인 시프트 하에서 전이 가능성을 향상시킴을 형식화한다.

### 3.7. 비디오 데이터 처리

비디오 데이터는 별도의 모델 수정 없이, 전처리 단계에서 시계열로 변환하여 동일 파이프라인에 통합한다:

1. 각 비디오 클립을 $T$개 시간 세그먼트로 균등 분할
2. 각 세그먼트에서 ResNet-101 또는 3D 백본으로 세그먼트-레벨 임베딩 추출
3. PCA로 차원 축소하여 $X \in \mathbb{R}^{d \times T}$ (기본 $d=2048$) 형태의 시퀀스 표현 획득
4. 이후 인과 그래프 추출, OT 정렬, 유사-레이블링 등 동일 절차 적용

---

## 4. 실험 결과

### 4.1. 시계열 벤치마크

6개 데이터셋(UCIHAR, WISDM, HHAR, SSC, MFD, Boiler)에서 평가. 아래는 WISDM과 UCIHAR 결과:

<b>Table 2. WISDM 정확도 (%) — 소스→타겟 도메인 쌍별</b>

| Algorithm | 7→18 | 20→30 | 35→31 | 17→23 | 6→19 | 2→11 | 33→12 | 5→26 | 28→4 | 23→32 | <b>Average</b> |
|-----------|------|-------|-------|-------|------|------|-------|------|------|-------|---------|
| No Adapt  | 80.2 | 64.1  | 66.3  | 48.3  | 62.1 | 31.6 | 60.9  | 73.2 | 83.3 | 27.5  | 59.8    |
| TransPL   | 84.9 | 66.0  | 63.9  | 66.7  | 48.5 | 61.8 | 88.5  | 74.4 | 54.5 | 30.4  | 64.0    |
| CoDATS    | 74.5 | 80.6  | 54.2  | 70.0  | 54.5 | 47.4 | 82.8  | 75.6 | 78.8 | 18.8  | 63.7    |
| RAINCOAT  | 53.1 | 58.5  | 40.2  | 30.6  | 58.9 | 77.8 | 37.0  | 37.8 | 67.6 | 11.8  | 47.3    |
| <b>Ours</b>  | <b>86.5</b> | <b>84.0</b> | 64.2 | 66.5 | 51.5 | 62.4 | 87.3 | <b>84.4</b> | 58.3 | 35.3 | <b>68.03</b> |

<b>Table 3. UCIHAR 정확도 (%) — 소스→타겟 도메인 쌍별</b>

| Algorithm | 2→11 | 6→23 | 7→13 | 9→18 | 12→16 | 18→27 | 20→5 | 24→8 | 28→27 | 30→20 | <b>Average</b> |
|-----------|------|------|------|------|-------|-------|------|------|-------|-------|---------|
| No Adapt  | 60.0 | 60.7 | 79.8 | 40.0 | 60.9  | 65.5  | 48.4 | 58.8 | 47.8  | 47.7  | 57.0    |
| TransPL   | 75.8 | 84.8 | 67.3 | 59.3 | 66.4  | 89.4  | 59.3 | 75.3 | 66.4  | 61.7  | 69.0    |
| SHOT      | 65.3 | 80.8 | 75.5 | 59.3 | 90.3  | 75.2  | 59.3 | 64.7 | 90.3  | 43.0  | 67.8    |
| <b>Ours</b>  | <b>84.4</b> | <b>86.2</b> | 71.4 | <b>62.0</b> | 68.2 | <b>89.7</b> | <b>64.5</b> | <b>78.2</b> | 68.6 | <b>66.5</b> | <b>73.97</b> |

> TransPL(69.0%) 대비 <b>+4.97%</b> 향상으로, 인과 구조 정렬과 불확실성 인식 유사-레이블링의 효과를 입증.

### 4.2. 비디오 벤치마크

<b>Table 4. UCF101 ↔ HMDB51 (full) 분류 정확도 (%)</b>

| Method | Backbone | U→H | H→U | <b>Average</b> |
|--------|----------|-----|-----|---------|
| Source Only | ResNet101 | 73.9 | 71.7 | 72.8 |
| TA3N | - | 78.3 | 81.8 | 80.1 |
| MA2LT-D | - | 85.0 | 86.6 | 85.8 |
| TransferAttn | - | 88.1 | 88.3 | 88.2 |
| <b>Ours</b> | ResNet101 | <b>90.2</b> | <b>89.5</b> | <b>89.85</b> |

### 4.3. Ablation Study

#### OT Solver 비교

![Sinkhorn, Greenkhorn, Unbalanced OT 간 정확도 및 계산 시간 비교](/images/causal_ot/ot_solver_comparison.jpeg)

Sinkhorn이 최고 정렬 점수(0.84)를 달성하고, 학습 시간도 합리적인 수준이다. Greenkhorn(0.81)과 Unbalanced OT(0.79)는 안정성과 계산 비용에서 미세한 트레이드오프를 보인다.

#### t-SNE 시각화

![Causal-OT 정렬 후 특징 공간 t-SNE 시각화 (HAR, HHAR, WISDM)](/images/causal_ot/tsne_visualization.jpeg)

정렬 이후 known/unknown 타겟 분포가 잘 분리되며, 잠재 공간 정렬의 효과를 정성적으로 확인할 수 있다.

#### Transferable Distance Loss 수렴

![에포크별 Transferable Distance Loss 변화](/images/causal_ot/distance_loss.jpeg)

모든 도메인 적응 시나리오에서 손실이 일관되게 하강하며, Causal-OT가 안정적이고 일반화 가능한 학습을 수행함을 보여준다.

### 4.4. 캘리브레이션 분석

![TransPL(baseline) vs Causal-OT의 캘리브레이션 성능 비교 (SSC, MFD)](/images/causal_ot/calibration_comparison.jpeg)

| 데이터셋 | TransPL ECE | Causal-OT ECE | 개선 |
|----------|:-----------:|:-------------:|:----:|
| SSC      | 13.55       | 11.23         | ↓ 2.32 |
| MFD      | 4.37        | 3.78          | ↓ 0.59 |

ECE(Expected Calibration Error) 감소는 Causal-OT가 <b>예측 확률과 실제 정확도 간의 정렬을 개선</b>하고, 과신을 줄임을 정량적으로 확인해준다.

---

## 5. 핵심 기여 정리

1. <b>통합 프레임워크:</b> 분포 정렬(OT) + 인과 구조 보존(Granger) + 불확실성 처리(entropy)를 하나의 프레임워크에서 동시에 해결
2. <b>인과 정규화 OT:</b> Granger 인과 그래프를 OT 비용 행렬에 직접 임베딩하여 구조적으로 일관된 지식 전이 실현
3. <b>불확실성 인식 유사-레이블링:</b> 엔트로피 + 인과 일관성 기반 이중 필터링으로 신뢰할 수 없는 타겟 샘플 제거
4. <b>모달리티 일반성:</b> 동일 파이프라인이 1D 시계열(6개 벤치마크)과 비디오(4개 벤치마크) 모두에서 SOTA 달성
5. <b>이론적 뒷받침:</b> Causal-OT 목적함수 최소화가 타겟 위험 상한을 줄임을 형식적으로 증명