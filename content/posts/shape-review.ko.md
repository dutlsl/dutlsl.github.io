---
title: "[CVPR 2026] SHAPE: 의료 영상 분할을 위한 구조 인식 계층적 비지도 도메인 적응 논문 리뷰"
date: 2026-06-30T18:00:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Medical Image Segmentation", "CVPR 2026"]
categories: ["Paper Review"]
summary: "CVPR 2026에 발표된 SHAPE 프레임워크 리뷰입니다. 하이퍼그래프 기반 해부학적 타당성 평가로 의료 영상 UDA의 새로운 패러다임을 제시합니다."
cover:
  image: "/images/shape/pipeline.jpeg"
  alt: "SHAPE Framework Pipeline"
---

> **논문 정보**
> - **제목:** SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation
> - **저자:** Linkuan Zhou, Yinghao Xia, Yufei Shen, Xiangyu Li, Wenjie Du, Cong Cong, Leyi Wei, Ran Su, Qiangguo Jin
> - **소속:** Northwestern Polytechnical University, HIT, USTC, Macquarie University, Macao Polytechnic University, Tianjin University
> - **학회:** CVPR 2026
> - **코드:** [github.com/BioMedIA-repo/SHAPE](https://github.com/BioMedIA-repo/SHAPE)

---

## 1. 한 줄 요약

의료 영상의 크로스-모달리티 비지도 도메인 적응(UDA)에서, DINOv3 기반의 **클래스 인식 계층적 특징 변조(HFM)**, **하이퍼그래프 기반 해부학적 타당성 평가(HPE)**, 그리고 **구조적 이상치 가지치기(SAP)**를 통합한 프레임워크 **SHAPE**를 제안하여, 심장 데이터셋에서 Dice **90.08%** (MRI→CT) / **78.51%** (CT→MRI), 복부 데이터셋에서 **87.48%** / **86.89%**의 SOTA를 달성한 논문.

---

## 2. 연구 배경 및 동기

### 2.1. 문제 정의

의료 영상 분할 모델은 학습 데이터와 테스트 데이터가 동일한 분포를 따른다고 가정하지만, 실제 임상 환경에서는 **촬영 장비, 모달리티(CT/MRI), 파라미터** 차이로 인해 심각한 성능 저하가 발생한다. UDA는 레이블이 있는 소스 도메인의 지식을 레이블 없는 타겟 도메인으로 전이하여, 비용이 많이 드는 재주석(re-annotation)을 피하는 것을 목표로 한다.

### 2.2. 기존 방법의 한계

기존 UDA 접근법은 크게 두 가지 범주로 나뉘며, 각각 근본적인 한계를 가진다:

| 범주 | 대표 방법 | 핵심 한계 |
|------|----------|----------|
| **정렬 기반** | CycleGAN, SIFA, SASAN, DDFP | 의미론적으로 무관한(content-agnostic) 전역 변환 → 클래스 간 세밀한 매핑 관계 파괴 |
| **유사-레이블 기반** | MAPSeg, GenericSSL, IPLC, UPL-SFDA | 픽셀 수준 신뢰도(엔트로피, 일관성)에만 의존 → 해부학적으로 불가능한 구조의 유사-레이블 허용 |

> 핵심 문제는 두 가지로 압축된다:
> 1. **의미론적 비인식 특징 정렬:** 모놀리식 스타일 변환이 서로 다른 해부학적 구조의 스타일 특성을 평균화하여 클래스별 정보를 소실시킴
> 2. **전역 해부학적 제약 무시:** 픽셀 수준 검증만으로는 형태적으로 불가능한 구조(예: 비정상적 형태나 공간 배치)를 가진 유사-레이블의 형성을 방지할 수 없음

---

## 3. 제안 방법: SHAPE 프레임워크

### 3.1. 전체 아키텍처

![SHAPE 프레임워크 전체 파이프라인 (논문 Figure 1)](/images/shape/pipeline.jpeg)

SHAPE는 세 가지 핵심 모듈의 시너지적 파이프라인으로 구성된다:

1. **Hierarchical Feature Modulation (HFM):** 클래스 인식, 공간 차별화된 특징 정렬
2. **Hypergraph Plausibility Estimation (HPE):** 하이퍼그래프를 통한 전역 해부학적 타당성 평가
3. **Structural Anomaly Pruning (SAP):** 교차 뷰 안정성 기반 구조적 이상치 제거

### 3.2. Hierarchical Feature Modulation (HFM)

사전학습된 **DINOv3 ViT** 인코더(가중치 고정)에서 추출한 특징 맵 $\mathbf{F} = \Phi(\mathbf{I}) \in \mathbb{R}^{C \times H \times W}$에 대해 **이중 세분화** 접근을 적용한다.

**전역 수준: AdaIN 스타일 정렬**

소스 $\mathbf{F}_s$와 타겟 $\mathbf{F}_t$ 간의 텍스처 속성을 정렬하는 스타일 변환 맵 $\mathbf{F}_{s \to t}$를 계산한다:

$$
\mathbf{F}_{s \to t} = \sigma(\mathbf{F}_t) \left( \frac{\mathbf{F}_s - \mu(\mathbf{F}_s)}{\sigma(\mathbf{F}_s) + \epsilon} \right) + \mu(\mathbf{F}_t) \tag{1}
$$

여기서 $\mu(\cdot)$와 $\sigma(\cdot)$는 채널별 평균과 표준편차이다.

**지역 수준: 구조 인식 토큰 혼합**

특징 맵을 더 세밀한 해상도로 업샘플링한 후, 각 토큰의 **순도 점수(purity score)**를 계산하여 의미적 코어(pure)와 구조적 경계(impure)로 분류한다:

$$
\mathcal{P}(\mathbf{m}^i) = \frac{\max_{k \in \{0..K-1\}} \sum_{v \in \mathbf{m}^i} \mathbb{I}(v=k)}{|\mathbf{m}^i|} \tag{2}
$$

이 분류에 따라 **차별화된 변조 전략**을 적용한다:

- **순수 토큰 (semantic cores):** 동일 클래스의 타겟 토큰과 직접 선형 보간 $(1-\lambda)\mathbf{f}_s^i + \lambda\mathbf{f}_t^j$
- **불순 토큰 (structural boundaries):** 소스·타겟 경계 토큰의 통계량을 보간한 정규화 적용

> [!IMPORTANT]
> **HFM의 핵심 통찰:** 기존 AdaIN은 모든 특징에 균일한 변환을 적용하여 클래스 간 분리성을 파괴한다. HFM은 해부학적 코어와 경계를 구분하여 각각에 최적화된 변조를 수행함으로써, 도메인 갭을 해소하면서도 세밀한 구별 능력을 보존한다.

### 3.3. Hypergraph Plausibility Estimation (HPE)

각 예측 분할 맵 $\mathbf{M} \in \{0,...,K-1\}^{H \times W}$를 **다층 구조 하이퍼그래프** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$로 모델링한다.

**정점 신뢰도 점수:**

$$
S_{\text{vertex}}(\mathcal{G}) = \frac{1}{|\mathcal{V}|} \sum_{p \in \mathcal{V}} w_p \tag{4}
$$

여기서 가중치 $w_p = (1 - \frac{H(\bar{\mathbf{M}}_p)}{\log K}) \cdot (1 - \frac{\text{JSD}(\{\mathbf{M}_p^n\})}{\log K})$는 확실성(평균 엔트로피)과 일관성(JSD)을 결합한다.

**클래스 내 형태 점수 (Intra-class Shape):**

클래스 하이퍼엣지 $\{e_k \in \mathcal{E}_{\mathcal{C}}\}$의 형태 기술자로 **등주비(isoperimetric ratio)** $\phi(e_k) = 4\pi \cdot \text{Area}(\mathbf{M}_k) / (\text{Perimeter}(\mathbf{M}_k))^2$를 사용하고, Z-score 기반의 softmax 가중 합산으로 점수화한다:

$$
S_{\text{intra}}(\mathcal{G}) = \sum_{k=1}^{K-1} S_{\phi,k} \cdot \frac{\exp(-S_{\phi,k}/\tau)}{\sum_{j=1}^{K-1} \exp(-S_{\phi,j}/\tau)} \tag{5}
$$

**클래스 간 배치 점수 (Inter-class Layout):**

레이아웃 하이퍼엣지 $e_l$은 클래스 중심점 간의 상대적 방향 코사인 $\psi_{ij}$를 통해 클래스 간 공간 배치의 타당성을 평가한다:

$$
S_{\text{inter}}(\mathcal{G}) = \sum_{i,j=1}^{K-1} S_{\psi,ij} \cdot \frac{\exp(-S_{\psi,ij}/\tau)}{\sum_{u,v=1}^{K-1} \exp(-S_{\psi,uv}/\tau)} \tag{6}
$$

**최종 타당성 점수:**

$$
S_{\text{final}} = S_{\text{vertex}}(\mathcal{G}) \cdot (\alpha \, S_{\text{intra}}(\mathcal{G}) + (1-\alpha) \, S_{\text{inter}}(\mathcal{G})) \tag{7}
$$

$S_{\text{final}}$이 에포크 내 상위 $\rho$ 백분위수로 결정되는 동적 임계값을 초과하는 샘플만 자기 학습에 사용된다.

### 3.4. Structural Anomaly Pruning (SAP)

HPE를 통과한 예측에도 특정 클래스 수준의 아티팩트(환각)가 남아 있을 수 있다. SAP는 $N_{\text{ens}}$개의 교사 예측에서 각 클래스의 **구조적 불안정성 점수**를 변동 계수로 정량화한다:

$$
\Upsilon(k) = \frac{\text{std}(\mathbf{c}_k)}{\bar{c}_k + \epsilon} \tag{8}
$$

여기서 $\mathbf{c}_k = \langle C(\mathbf{M}^1, k), \dots, C(\mathbf{M}^{N_{\text{ens}}}, k) \rangle$는 각 앙상블 예측에서의 클래스 $k$ 픽셀 수이다. 안정적인 해부학적 구조는 낮은 분산을, 모델 환각은 높은 변동성을 보인다.

불안정성 점수가 배치 내 유의미한 전경 클래스의 $q$번째 백분위수 임계값 $\theta_A$를 초과하는 클래스를 이상 클래스로 판정하여 제거한다:

$$
\mathcal{K}_{\text{anom}} = \{ k \in \{1, \dots, K-1\} \mid \Upsilon(k) > \theta_A \} \tag{9}
$$

### 3.5. 전체 손실 함수

전체 학습 목적함수는 지도 손실과 비지도 손실의 가중합이다:

- **소스 도메인 지도 손실:** 원본 및 HFM 변조 특징 집합 $\mathcal{F}_s = \{\mathbf{F}_s, \mathbf{F}_{s \to t}, \mathbf{F}_{s,\text{cross}}\}$에 대해:

$$
\mathcal{L}_{\text{sup}} = \frac{1}{|\mathcal{F}_s|} \sum_{\mathbf{F}' \in \mathcal{F}_s} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}'), \mathbf{L}_s) \tag{11}
$$

- **타겟 도메인 비지도 손실:** HPE + SAP 검증을 통과한 고충실도 유사-레이블 $\mathbf{M}'$로 학습:

$$
\mathcal{L}_{\text{unsup}} = \frac{1}{|\mathcal{B}_{\text{sel}}|} \sum_{i \in \mathcal{B}_{\text{sel}}} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}_t^i), (\mathbf{M}')^i, w_p^i) \tag{12}
$$

- **전체 손실:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \gamma_{\text{unsup}} \mathcal{L}_{\text{unsup}}$

교사 모델은 학생 디코더의 EMA(Exponential Moving Average)이며, $\mathcal{L}_{\text{seg}}$는 Dice + Focal loss 조합이다.

---

## 4. 실험 결과

### 4.1. 실험 설정

- **데이터셋:** MMWHS (심장, CT/MRI 각 20개), MICCAI 2015 + ISBI 2019 CHAOS (복부, CT 30개 / MRI 20개)
- **구현:** DINOv3 ViT-S/16 인코더 (고정) + UNet 디코더, 200 에포크, RTX 4090, 배치 크기 64
- **평가 지표:** Dice Score (DSC), Average Surface Distance (ASD)

### 4.2. 심장 데이터셋 결과

**Table 1. Cardiac MRI → CT — DSC(%) / ASD(mm)**

| Method | AA | LAC | LVC | MYO | **Avg DSC** | **Avg ASD** |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 68.29 | 61.41 | 18.24 | 35.71 | 45.91 | 38.02 |
| CycleGAN | 63.29 | 72.50 | 45.98 | 50.73 | 58.13 | 21.28 |
| SIFA | 82.72 | 75.21 | 75.41 | 65.17 | 74.63 | 10.22 |
| GenericSSL | 82.02 | 77.18 | 84.28 | 67.65 | 77.78 | 6.64 |
| IPLC | 87.63 | 78.21 | 86.11 | 71.68 | 80.91 | 6.37 |
| IPLC+ | 63.69 | 86.15 | 86.71 | 89.17 | 81.43 | 3.91 |
| DDFP | 72.03 | 85.30 | 89.64 | 90.86 | 84.46 | 3.79 |
| **SHAPE** | **79.58** | **92.18** | **94.53** | **94.03** | **90.08** | **2.62** |
| Supervised (상한) | 91.28 | 92.49 | 95.56 | 94.16 | 93.37 | 2.11 |

> SHAPE는 2위 대비 **+5.62% DSC** 향상을 달성하며, 지도 학습 상한(93.37%)과의 격차를 **단 3.29%**로 좁혔다.

**Table 2. Cardiac CT → MRI — DSC(%) / ASD(mm)**

| Method | AA | LAC | LVC | MYO | **Avg DSC** | **Avg ASD** |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 36.56 | 46.49 | 49.23 | 15.35 | 36.91 | 23.06 |
| SIFA | 55.47 | 66.43 | 72.52 | 60.69 | 63.78 | 13.84 |
| IPLC+ | 65.09 | 77.56 | 88.95 | 74.33 | 76.48 | 5.37 |
| DDFP | 66.26 | 76.04 | 88.55 | 70.64 | 75.37 | 8.58 |
| **SHAPE** | **70.25** | **79.11** | **86.08** | **78.59** | **78.51** | **4.70** |

### 4.3. 복부 데이터셋 결과

**Table 3. Abdominal MRI → CT — DSC(%)**

| Method | LIV | RK | LK | SPL | **Avg DSC** | **Avg ASD** |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 84.98 | 84.08 | 84.37 | 84.92 | 84.59 | 7.56 |
| IPLC+ | 85.29 | 88.63 | 85.12 | 83.79 | 85.71 | 5.15 |
| DDFP | 86.40 | 87.66 | 88.05 | 78.55 | 85.17 | 3.14 |
| **SHAPE** | **88.26** | **86.99** | **83.37** | **91.28** | **87.48** | **2.94** |

**Table 4. Abdominal CT → MRI — DSC(%)**

| Method | LIV | RK | LK | SPL | **Avg DSC** | **Avg ASD** |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 82.43 | 86.88 | 80.52 | 89.79 | 84.91 | 6.71 |
| DDFP | 82.49 | 86.43 | 87.15 | 89.01 | 86.27 | 5.17 |
| **SHAPE** | **86.83** | **88.86** | **88.30** | **83.58** | **86.89** | **2.81** |

### 4.4. 정성적 결과 및 특징 분석

![SHAPE와 대표 방법들의 정성적 비교. 노란 화살표는 SHAPE가 우수한 영역을 표시](/images/shape/qualitative_results.jpeg)

![소스-타겟 특징 분포의 t-SNE 시각화: (a) 적응 전, (b) AdaIN 후, (c) HFM 후](/images/shape/tsne_features.jpeg)

t-SNE 분석에서 핵심적인 관찰:
- **(a) 적응 전:** 소스(원)와 타겟(사각형) 특징이 완전히 분리됨
- **(b) AdaIN 후:** 타겟 특징을 소스 쪽으로 끌어당기지만, **분포 수축(distributional contraction)**이 발생하여 클래스 간 구별력이 소실됨
- **(c) HFM 후:** 타겟 클래스 중심이 소스와 정밀하게 정렬되면서도, 각 클래스의 **내부 분포 구조가 보존**됨

### 4.5. Ablation Study

| 구성 | HFM | HPE | SAP | MRI→CT DSC | CT→MRI DSC |
|------|:---:|:---:|:---:|:----------:|:----------:|
| (a) Baseline (DINOv3) | | | | 82.02 | 71.58 |
| (b) + HFM | ✓ | | | 85.67 | 75.46 |
| (c) + HPE | | ✓ | | 82.71 | 72.09 |
| (d) + HFM + HPE | ✓ | ✓ | | 85.80 | 75.81 |
| (e) + HFM + SAP | ✓ | | ✓ | 86.03 | 76.23 |
| **(f) SHAPE (Full)** | **✓** | **✓** | **✓** | **90.08** | **78.51** |

> **HFM**이 단독으로 가장 큰 성능 향상(+3.65%)을 가져오며, 세 모듈의 완전 통합 시 시너지 효과로 **+8.06%**의 극적 향상을 달성한다.

### 4.6. 하이퍼파라미터 민감도 분석

![하이퍼파라미터 민감도 분석](/images/shape/hyperparameter_sensitivity.jpeg)

- **타당성 융합 가중치** $\alpha$와 **이상 임계값** $\theta_A$는 명확한 최적 영역을 보임
- **순도 임계값** $\tau_p$가 엄격해질수록, 배치 크기가 커질수록 성능이 일관되게 향상
- HPE는 더 큰 배치에서 안정적 통계를 확보하지만, SAP의 배치-내 동적 임계값 덕분에 작은 배치에서도 효과적

---

## 5. 핵심 기여 정리

1. **클래스 인식 특징 변조 (HFM):** 의미론적 코어와 구조적 경계를 구분하여 각각에 최적화된 변조를 수행, 기존 모놀리식 정렬의 분포 충실도 문제를 해결
2. **하이퍼그래프 기반 해부학적 타당성 평가 (HPE):** 표준 그래프의 이진 관계를 넘어, 하이퍼그래프로 다수 해부학적 구조의 고차 상호작용을 모델링하여 전역 구조 타당성을 검증
3. **구조적 이상치 가지치기 (SAP):** 교차 뷰 안정성 분석을 통해 클래스 수준의 환각(hallucination)을 식별 및 제거
4. **통합 패러다임 전환:** 픽셀 수준 정확도에서 **전역 해부학적 타당성**으로의 UDA 패러다임 전환을 실현
5. **SOTA 성능:** 심장 및 복부 크로스-모달리티 벤치마크 4개 시나리오 모두에서 기존 방법 대비 의미있는 성능 향상 달성

---

## 6. 총평 및 개인적 의견

### 강점

1. **패러다임 전환의 설득력:** "픽셀 정확도 → 해부학적 타당성"이라는 관점 전환이 매우 설득력 있다. 의료 영상에서 구조적으로 불가능한 분할 결과를 아무리 높은 픽셀 신뢰도로 생성해봐야 임상적 가치가 없다는 점을 정확히 짚었다.

2. **하이퍼그래프 활용의 참신성:** 하이퍼그래프를 유사-레이블의 품질 게이트로 활용한 것은 이 분야에서 새로운 시도이다. 등주비를 통한 클래스 내 형태 검증과, 중심점 방향 코사인을 통한 클래스 간 배치 검증이라는 이중 구조가 직관적이면서도 효과적이다.

3. **HFM의 구조 보존 능력:** t-SNE 시각화(Fig. 3)에서 AdaIN이 분포를 균질화시키는 반면, HFM이 클래스 간 분리성을 보존하면서 정렬하는 것이 매우 인상적이다. 이는 단순히 수치적 개선이 아니라, **왜** 개선되는지를 명확히 보여준다.

4. **철저한 실험 설계:** 4개 적응 시나리오(심장 MRI↔CT, 복부 MRI↔CT) × 10개 이상의 비교 방법 × 체계적 ablation으로 신뢰성 있는 검증을 수행했다.

### 아쉬운 점 / 향후 과제

1. **AA 클래스의 성능 저하:** MRI→CT에서 SHAPE의 AA(대동맥) DSC가 79.58%로, IPLC(87.63%)보다 낮다. 논문에서 이에 대한 분석이 부족하다. HFM의 클래스별 토큰 매칭에서 AA처럼 작은 구조는 대표 토큰 풀이 부족할 수 있다.

2. **DINOv3 의존성:** 프레임워크 전체가 DINOv3 ViT-S/16에 고정적으로 의존한다. 다른 파운데이션 모델(SAM, CLIP 등)이나 더 가벼운 백본에서의 일반화 가능성이 검증되지 않았다.

3. **3D 활용 제한:** MMWHS는 3D 볼륨 데이터이지만, 2D 슬라이스 단위로 처리한다. 하이퍼그래프의 해부학적 타당성 평가가 3D 공간 일관성까지 확장되면 더 강력할 것이다.

4. **계산 비용 미기재:** HFM의 토큰 분류와 HPE의 하이퍼그래프 구성에 소요되는 추가 계산 비용이 명시되지 않았다. 실제 배포 시 이 오버헤드가 얼마나 되는지 궁금하다.

### 결론

SHAPE는 의료 영상 UDA에서 **"구조적으로 말이 되는 예측만 학습에 사용하자"**라는 심플하지만 강력한 원칙을 하이퍼그래프라는 수학적 도구로 우아하게 구현한 연구이다. HFM이 정밀한 특징 정렬로 좋은 출발점을 제공하고, HPE + SAP가 해부학적 검증으로 유사-레이블의 품질을 보증하는 시너지적 파이프라인은 잘 설계되어 있다. 특히, 심장 MRI→CT에서 **90.08%** Dice를 달성하여 지도 학습 상한(93.37%)에 근접한 것은 UDA 분야에서 매우 인상적인 성과이다.

---

> **참고 문헌**은 원 논문의 References 섹션을 참조하시기 바랍니다.
