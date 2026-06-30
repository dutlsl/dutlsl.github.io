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

> **논문 정보**
> - **제목:** Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport
> - **저자:** Khushboo Mishra, Varun Trivedi, Tanima Dutta
> - **소속:** Indian Institute of Technology (BHU) Varanasi, India
> - **학회:** CVPR 2026
> - **코드:** [github.com/mynameanonymous/CausalOT](https://github.com/mynameanonymous/CausalOT)

---

## 1. 한 줄 요약

시계열 및 비디오 데이터의 비지도 도메인 적응(UDA)에서, **Granger 인과 그래프 정규화**를 **최적 수송(Optimal Transport)** 정렬에 통합하고 **엔트로피 기반 불확실성 인식 유사-레이블링**을 결합한 통합 프레임워크 **Causal-OT**를 제안하여, 6개 시계열 벤치마크에서 **평균 4.5% 정확도 및 3.8% F1 향상**, 4개 비디오 벤치마크에서 **평균 2.5% 정확도 향상**을 달성한 논문.

---

## 2. 연구 배경 및 동기

### 2.1. 문제 정의

시계열 데이터(가속도계, 자이로스코프, EMG 등)는 사용자·기기·환경에 따라 도메인 간 상당한 차이를 보인다. 레이블이 있는 소스 도메인에서 학습된 모델을 레이블이 없는 타겟 도메인에 적용할 때, **통계적 분포 차이**와 **시간적 의존성 변화**라는 이중 과제가 발생한다.

### 2.2. 기존 방법의 한계

기존 UDA 접근법들은 아래 세 가지 핵심 차원을 **동시에** 다루지 못한다:

| 모델 | 분포 불일치 | 인과성 보존 | 불확실성 처리 |
|------|:---:|:---:|:---:|
| TransPL [15] | ✓ | ✗ | ✗ |
| CauDiTS [36] | ✗ | ✓ | ✗ |
| RAINCOAT [13] | ✓ | ✗ | ✗ |
| **Causal-OT (Ours)** | **✓** | **✓ (Granger)** | **✓ (entropy-aware)** |

> **Table 1 재구성.** 기존 방법들은 분포 정렬, 인과 구조 보존, 예측 불확실성 중 일부만 처리. Causal-OT는 세 가지를 모두 통합.

특히 TransPL은 높은 엔트로피 예측까지 포함한 넓은 범위의 유사-레이블에 의존하여 노이즈가 많은 감독 신호를 생성한다. 아래 그림은 타겟 도메인의 예측 불확실성이 소스보다 현저히 높음을 보여준다:

![소스 vs 타겟 도메인 불확실성 분포 비교 (SSC, MFD 데이터셋)](/images/causal_ot/uncertainty_distribution.jpeg)

---

## 3. 제안 방법: Causal-OT 프레임워크

### 3.1. 전체 아키텍처

![Causal-OT 프레임워크 전체 구조 (논문 Figure 3)](/images/causal_ot/causal_ot_architecture.jpeg)

Causal-OT의 학습 파이프라인은 다음 5단계로 구성된다:

1. **인과 그래프 구성 (Causal Graph Construction):** 소스와 타겟 도메인의 원시 시계열로부터 Granger 인과 그래프 G<sub>s</sub>, G<sub>t</sub>를 추출한다. 그림에서 검은 화살표는 변수 간 인과 관계, 빨간 화살표는 비인과 관계를 나타낸다.

2. **특징 추출 (Feature Extraction):** 공유 특징 추출기 f<sub>θ</sub>가 소스·타겟 데이터를 잠재 특징 공간 Z<sub>s</sub>, Z<sub>t</sub>로 매핑한다.

3. **인과 정렬 (Causal Alignment):** 특징 거리와 인과 그래프 거리 D<sub>G</sub>를 결합한 비용 행렬 C를 기반으로 최적 수송(OT) 손실 ℒ<sub>OT</sub>를 계산한다.

4. **분류 및 유사-레이블링:** 분류기 h<sub>φ</sub>가 소스 데이터로 학습되며 (ℒ<sub>src</sub>), 타겟 도메인에서 높은 신뢰도의 유사-레이블을 생성한다 (ℒ<sub>PL</sub>).

5. **최적화:** 세 손실의 가중합으로 모델 파라미터 θ, φ를 업데이트한다.

### 3.2. Granger 인과 그래프 추출

다변량 시계열 X ∈ ℝ<sup>N × T × D</sup>에서 변수 간 방향성 인과 관계를 모델링한다. 구체적으로:

- **정상성 검정:** Augmented Dickey-Fuller 테스트로 신호의 정상성을 확인
- **VAR 차수 선택:** Bayesian Information Criterion (BIC)으로 최적 래그 차수 p 결정
- **유의성 필터링:** p-value < 0.05인 엣지만 유지
- **행 정규화:** 인접 행렬 W를 행 합이 1이 되도록 정규화
- **스펙트럴 임베딩:** 라플라시안 L = D - W에서 k차원 임베딩 Φ(G) ∈ ℝ<sup>d × k</sup> 도출

> [!IMPORTANT]
> **하이브리드 그래프 업데이트 전략:** 초기 인과 그래프는 원시 데이터 X에서 추출하되, 학습 중 잠재 특징 Z에서 주기적으로 재추정하여 블렌딩한다.
>
> <b>A<sup>(t)</sup> = α A<sub>X</sub> + (1 - α) A<sub>Z</sub><sup>(t)</sup></b>  (단, α ∈ [0.6, 0.9])
>
> 이를 통해 안정적인 원시 신호 구조를 유지하면서 학습 진행에 따른 인과 구조 변화를 반영한다.

### 3.3. 인과-의존 최적 수송 비용

소스-타겟 샘플 쌍 간의 비용을 특징 유사도와 인과 기술자 일관성을 결합하여 정의한다:

<b>C<sub>ij</sub> = ‖ f<sub>s</sub>(x<sub>i</sub><sup>s</sup>) - f<sub>t</sub>(x<sub>j</sub><sup>t</sup>) ‖<sub>2</sub><sup>2</sup> + λ ‖ φ<sub>i</sub><sup>s</sup> - φ<sub>j</sub><sup>t</sup> ‖<sub>2</sub><sup>2</sup></b>

여기서 φ<sub>i</sub><sup>s</sup>, φ<sub>j</sub><sup>t</sup>는 각각 소스·타겟 샘플의 인과 임베딩이다. 이 비용 행렬에 대해 엔트로피 정규화된 OT 문제를 Sinkhorn 반복법으로 풀어 최적 수송 계획 γ*를 구한다:

<b>γ* = argmin<sub>γ ∈ Π(μ<sub>s</sub>, μ<sub>t</sub>)</sub> [ ⟨γ, C⟩ + ε H(γ) ]</b>

이 설계의 핵심은 **인과 그래프 제약을 비용 행렬에 내장**하여, 정렬이 기하학적으로 의미있을 뿐 아니라 시간적 의존성과도 구조적으로 일관되게 하는 것이다.

### 3.4. 불확실성 인식 유사-레이블링

타겟 도메인의 인코딩된 특징 Z<sub>t</sub><sup>j</sup>로부터 분류기 h<sub>φ</sub>의 소프트 클래스 예측을 구하고, 예측 분포의 **엔트로피**로 불확실성을 추정한다:

<b>U<sub>t</sub><sup>j</sup> = - Σ<sub>k=1</sub><sup>K</sup> ŷ<sub>t,j</sub><sup>(k)</sup> log(ŷ<sub>t,j</sub><sup>(k)</sup>)</b>

엔트로피 임계값 ρ 이하인 **저엔트로피(고신뢰도)** 샘플만 유사-레이블로 사용하여 노이즈 전파를 방지한다.

### 3.5. 전체 손실 함수

전체 손실 함수는 세 가지 항목의 가중합으로 구성됩니다:

<b>ℒ<sub>total</sub> = ℒ<sub>src</sub> + α ℒ<sub>OT</sub> + β ℒ<sub>PL</sub></b>

- **소스 도메인 분류 손실 (ℒ<sub>src</sub>)**: Cross-Entropy
  <b>ℒ<sub>src</sub> = (1 / N<sub>s</sub>) Σ<sub>i</sub> CE(h<sub>φ</sub>(Z<sub>s</sub><sup>i</sup>), y<sub>s</sub><sup>i</sup>)</b>

- **인과 구조 보존 OT 정렬 손실 (ℒ<sub>OT</sub>)**:
  <b>ℒ<sub>OT</sub> = ⟨γ*, C⟩</b>

- **불확실성 필터링 유사-레이블 손실 (ℒ<sub>PL</sub>)**:
  <b>ℒ<sub>PL</sub> = (1 / |I|) Σ<sub>j ∈ I</sub> CE(h<sub>φ</sub>(Z<sub>t</sub><sup>j</sup>), ŷ<sub>t</sub><sup>j</sup>)</b>

### 3.6. 이론적 분석

**Proposition 1 (Causal-OT 타겟 위험 상한):** Lipschitz 연속이고 유계인 대리 손실 ℓ에 대해:

<b>ℛ<sub>t</sub>(h) ≤ ℛ<sub>s</sub>(h) + L · 𝔼<sub>(x<sub>s</sub>, x<sub>t</sub>) ∼ γ*</sub>[ ‖ f<sub>s</sub>(x<sub>s</sub>) - f<sub>t</sub>(x<sub>t</sub>) ‖ ] + λ · 𝔼<sub>(i,j) ∼ γ*</sub>[ ‖ φ<sub>i</sub><sup>s</sup> - φ<sub>j</sub><sup>t</sup> ‖ ] + 𝒟<sub>ℋ</sub>(P<sub>s</sub>, P<sub>t</sub>)</b>

이 부등식은 Causal-OT 목적함수를 최소화하면 **특징 수준 거리, 인과 구조 불일치, 가설 불일치**를 동시에 줄여 도메인 시프트 하에서 전이 가능성을 향상시킴을 형식화한다.

### 3.7. 비디오 데이터 처리

비디오 데이터는 별도의 모델 수정 없이, 전처리 단계에서 시계열로 변환하여 동일 파이프라인에 통합한다:

1. 각 비디오 클립을 T개 시간 세그먼트로 균등 분할
2. 각 세그먼트에서 ResNet-101 또는 3D 백본으로 세그먼트-레벨 임베딩 추출
3. PCA로 차원 축소하여 X ∈ ℝ<sup>d × T</sup> (기본 d=2048) 형태의 시퀀스 표현 획득
4. 이후 인과 그래프 추출, OT 정렬, 유사-레이블링 등 동일 절차 적용

---

## 4. 실험 결과

### 4.1. 시계열 벤치마크

6개 데이터셋(UCIHAR, WISDM, HHAR, SSC, MFD, Boiler)에서 평가. 아래는 WISDM과 UCIHAR 결과:

**Table 2. WISDM 정확도 (%) — 소스→타겟 도메인 쌍별**

| Algorithm | 7→18 | 20→30 | 35→31 | 17→23 | 6→19 | 2→11 | 33→12 | 5→26 | 28→4 | 23→32 | **Average** |
|-----------|------|-------|-------|-------|------|------|-------|------|------|-------|---------|
| No Adapt  | 80.2 | 64.1  | 66.3  | 48.3  | 62.1 | 31.6 | 60.9  | 73.2 | 83.3 | 27.5  | 59.8    |
| TransPL   | 84.9 | 66.0  | 63.9  | 66.7  | 48.5 | 61.8 | 88.5  | 74.4 | 54.5 | 30.4  | 64.0    |
| CoDATS    | 74.5 | 80.6  | 54.2  | 70.0  | 54.5 | 47.4 | 82.8  | 75.6 | 78.8 | 18.8  | 63.7    |
| RAINCOAT  | 53.1 | 58.5  | 40.2  | 30.6  | 58.9 | 77.8 | 37.0  | 37.8 | 67.6 | 11.8  | 47.3    |
| **Ours**  | **86.5** | **84.0** | 64.2 | 66.5 | 51.5 | 62.4 | 87.3 | **84.4** | 58.3 | 35.3 | **68.03** |

**Table 3. UCIHAR 정확도 (%) — 소스→타겟 도메인 쌍별**

| Algorithm | 2→11 | 6→23 | 7→13 | 9→18 | 12→16 | 18→27 | 20→5 | 24→8 | 28→27 | 30→20 | **Average** |
|-----------|------|------|------|------|-------|-------|------|------|-------|-------|---------|
| No Adapt  | 60.0 | 60.7 | 79.8 | 40.0 | 60.9  | 65.5  | 48.4 | 58.8 | 47.8  | 47.7  | 57.0    |
| TransPL   | 75.8 | 84.8 | 67.3 | 59.3 | 66.4  | 89.4  | 59.3 | 75.3 | 66.4  | 61.7  | 69.0    |
| SHOT      | 65.3 | 80.8 | 75.5 | 59.3 | 90.3  | 75.2  | 59.3 | 64.7 | 90.3  | 43.0  | 67.8    |
| **Ours**  | **84.4** | **86.2** | 71.4 | **62.0** | 68.2 | **89.7** | **64.5** | **78.2** | 68.6 | **66.5** | **73.97** |

> TransPL(69.0%) 대비 **+4.97%** 향상으로, 인과 구조 정렬과 불확실성 인식 유사-레이블링의 효과를 입증.

### 4.2. 비디오 벤치마크

**Table 4. UCF101 ↔ HMDB51 (full) 분류 정확도 (%)**

| Method | Backbone | U→H | H→U | **Average** |
|--------|----------|-----|-----|---------|
| Source Only | ResNet101 | 73.9 | 71.7 | 72.8 |
| TA3N | - | 78.3 | 81.8 | 80.1 |
| MA2LT-D | - | 85.0 | 86.6 | 85.8 |
| TransferAttn | - | 88.1 | 88.3 | 88.2 |
| **Ours** | ResNet101 | **90.2** | **89.5** | **89.85** |

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

ECE(Expected Calibration Error) 감소는 Causal-OT가 **예측 확률과 실제 정확도 간의 정렬을 개선**하고, 과신을 줄임을 정량적으로 확인해준다.

---

## 5. 핵심 기여 정리

1. **통합 프레임워크:** 분포 정렬(OT) + 인과 구조 보존(Granger) + 불확실성 처리(entropy)를 하나의 프레임워크에서 동시에 해결
2. **인과 정규화 OT:** Granger 인과 그래프를 OT 비용 행렬에 직접 임베딩하여 구조적으로 일관된 지식 전이 실현
3. **불확실성 인식 유사-레이블링:** 엔트로피 + 인과 일관성 기반 이중 필터링으로 신뢰할 수 없는 타겟 샘플 제거
4. **모달리티 일반성:** 동일 파이프라인이 1D 시계열(6개 벤치마크)과 비디오(4개 벤치마크) 모두에서 SOTA 달성
5. **이론적 뒷받침:** Causal-OT 목적함수 최소화가 타겟 위험 상한을 줄임을 형식적으로 증명

---