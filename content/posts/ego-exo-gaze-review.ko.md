---
title: "[CVPR 2026 Workshop Best Poster] Ego-Exo Gaze: 대화 장면 시선 추정을 위한 Ego-Exo 시각 표현 학습 / Learning Ego-Exo Visual Representations for Conversational Gaze Estimation"
date: 2026-08-11T19:12:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "Egocentric Vision", "Self-Supervised Learning", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Meta Reality Labs와 Idiap Research Institute가 CVPR 2026 GAZE Workshop에서 발표하고 Best Poster Award를 수상한 논문을 리뷰합니다. 대화 상황에서 타인의 exocentric gaze 정보를 self-supervised learning으로 함께 학습하여, 단일 프레임만으로 egocentric gaze estimation 성능을 향상시키는 방법을 제안합니다."
cover:
  image: "/images/ego-exo-gaze/_page_0_Picture_10.jpeg"
  alt: "Ego-Exo Gaze Alignment Overview"
---

> 논문 정보
> - 제목: Learning Ego-Exo Visual Representations for Conversational Gaze Estimation
> - 저자: Anshul Gupta, Yijun Qian, Ruohan Gao, Ishwarya Ananthabhotla, Jean-Marc Odobez, Vamsi Krishna Ithapu, Calvin Murdock
> - 소속: Meta Reality Labs Research, Idiap Research Institute, EPFL, University of Maryland
> - 학회: CVPR 2026 GAZE Workshop. 본 논문은 GAZE 2026에서 Best Poster Award를 수상했습니다.

---

## 1. 한 줄 요약

대화 장면에서 한 쌍의 착용자가 동시에 촬영한 egocentric video를 활용하여 ego 및 exo gaze representation을 self-supervised alignment로 함께 학습하고, 추론 시에는 단일 프레임 및 단일 브랜치만으로 향상된 egocentric gaze estimation 성능을 달성한 논문입니다.

---

## 2. 연구 배경 및 동기

### 2.1 Problem Definition

egocentric gaze estimation은 1인칭 카메라 착용자가 장면 내 어디를 바라보고 있는지를 예측하는 과제입니다. AR/VR 웨어러블 기기에서 직관적인 상호작용, 소음 환경에서의 화자 추적, 대화 맥락 이해 등 다양한 응용에 핵심적인 역할을 합니다.

경량 웨어러블 플랫폼에서는 정밀한 eye-tracker 추가가 하드웨어 비용과 전력 소모를 늘려 부담이 됩니다. 이 때문에 RGB 이미지 프레임만으로 시선을 추정하는 접근법이 주로 연구되고 있습니다.

### 2.2 Limitations of Existing Methods

기존 방법은 주로 비디오 시퀀스의 temporal cues에 의존합니다. 하지만 temporal model은 static single-frame model에 비해 연산량과 메모리 소모가 최대 10배 이상 큽니다.

반면 single-frame inference를 수행하는 모델은 시야 내에 여러 인물이 존재할 때 target ambiguity가 심해지는 한계가 있습니다. 이때 다른 인물의 시점에서 관찰되는 exocentric gaze cues를 통합하면 타깃의 모호성을 유의미하게 해소할 수 있습니다.

![Figure 1. egocentric gaze target 추정의 모호성과 exocentric gaze cues를 활용한 개선. 훈련 시에는 두 사람의 동시 시점을 Siamese 구조로 정렬하고, 추론 시에는 단일 브랜치만으로 향상된 성능을 달성합니다.](/images/ego-exo-gaze/_page_0_Picture_10.jpeg)

*Figure 1: egocentric gaze target 추정의 모호성과 Ego-Exo Alignment를 통한 개선*

### 2.3 Main Contributions

본 연구의 주된 기여는 다음과 같습니다:

- single-frame egocentric gaze estimation 탐색: 최신 ViT 아키텍처를 기반으로 단일 프레임만으로도 우수한 시선 추정 성능을 달성할 수 있음을 입증합니다.
- ego-exo gaze representation 학습: Time Synchronization, Implicit Matching, Explicit Matching 등 세 가지 self-supervised alignment 기법을 제안합니다.
- exocentric gaze probing 검증: 학습된 Encoder가 실제로 의미 있는 exocentric gaze 정보를 잘 포착하고 있음을 probing 실험으로 증명합니다.
- 새로운 평가 메트릭 도입: gaze following 연구 분야의 평가 방식을 준용하여 Distance 및 Looking at Heads를 뜻하는 LAH 메트릭을 새롭게 도입합니다.

---

## 3. 제안 방법 (Proposed Framework)

본 논문의 아키텍처는 두 사람 A와 B의 egocentric image frame인 $I^A$와 $I^B$를 동시에 입력받는 Siamese 구조를 따릅니다.

![Figure 3. 제안된 Ego-Exo 시선 표현 학습 아키텍처. Encoder가 각 사람의 시점에서 특징을 추출하고, Ego-Exo Alignment 과정에서 Time Synchronization 또는 Head Matching 방식을 적용한 후, Ego Decoder가 시선 히트맵을 예측합니다.](/images/ego-exo-gaze/_page_3_Figure_0.jpeg)

*Figure 3: Ego-Exo 시선 표현 학습 아키텍처 개요*

### 3.1 Feature Extraction

ViT 기반 Encoder $V$를 이용하여 각 입력 프레임으로부터 특징 $F$를 추출합니다.

$$F^A = V(I^A)$$
$$F^B = V(I^B)$$

이 과정은 두 사람의 시점에 독립적으로 적용됩니다.

### 3.2 Ego-Exo Alignment

Ego-Exo Alignment 모듈은 ego representation과 exo representation을 정렬하여 self-supervised learning을 가능하게 합니다. 한 착용자의 ego gaze feature는 자신이 응시하는 위치 정보를 이미 포함하므로, 다른 사람 시점에서 촬영된 본인의 exo representation을 학습시키는 지도 신호가 됩니다.

#### 3.2.1 Time Synchronization

동일 세션 내에서 같은 타임스탬프에 촬영된 피험자 A와 B의 egocentric feature인 CLS 토큰은 같은 대화 상황을 공유하므로 positive pair로 설정하여 가깝게 정렬합니다. 반대로 타임스탬프가 다르거나 다른 세션에서 가져온 feature는 negative pair로 설정하여 멀어지게 유도합니다.

$$G_{ego}^A = \text{CLS}(F^A)$$

이때 A의 exocentric feature $G_{exo}^A$는 B의 egocentric feature $G_{ego}^B$와 직접 매칭됩니다. 유사도 $S$는 $L_2$ distance로 계산됩니다.

$$S = \|G_{ego}^A - G_{exo}^A\|_2$$

Triplet loss 기반의 Time Synchronization Loss를 적용하여 동일 시간대의 ego-exo feature 거리를 최소화하고, 다른 시간대나 다른 세션의 negative sample과의 거리는 극대화합니다.

#### 3.2.2 Head Matching

참가자 B의 FoV 내에 보이는 타인의 head bounding box $B^B$ 영역에서 ROI-Align을 통해 exocentric feature를 추출합니다.

ROI-Align은 전체 이미지 feature map에서 특정 관심 영역의 특징을 잘라내어 일정 크기의 feature 벡터로 변환하는 딥러닝 기법입니다. B의 1인칭 카메라 이미지에서 관찰되는 A의 머리는 외부 3인칭 시점인 exocentric 객체이므로, B의 feature map 중 A의 head box 위치를 ROI-Align으로 추출하면 B의 관점에서 관찰된 A의 머리 방향 및 외형 정보를 담은 A의 exocentric feature가 됩니다.

$$G_{exo}^B = \text{ROI-Align}(F^B, B^B)$$
$$G_{ego}^A = \text{CLS}(F^A)$$

두 feature 간의 유사도 $S^A$는 내적(dot product)으로 구합니다.

$$S^A = G_{exo}^B \cdot G_{ego}^A$$

Head Matching 방식은 레이블 제공 여부에 따라 두 가지 정렬 손실 함수로 구별됩니다:

- Explicit Matching: GT head box identity 정보가 존재하는 경우 사용합니다. 이는 누가 피험자 A인가에 대한 ground truth 레이블에 해당하며, B 시야 내 여러 head box 영역 중 실제 A의 head box에 해당하는 유사도가 가장 높아지도록 Cross-Entropy Loss를 적용합니다.
- Implicit Matching: GT identity 레이블이 제공되지 않는 자율 정렬 상황에서 사용합니다. 유사도 분포 $S$의 불확실성을 줄이고 한 개의 특정 head box에 시선이 쏠리도록 자율적으로 유도하기 위해 Entropy Loss를 적용합니다.

### 3.3 Prediction & Loss

Prediction 모듈은 Feature Extraction 모듈에서 얻은 visual feature를 전달받아, 각 참가자의 최종 시선 예측 지도를 복원하는 역할을 담당합니다.

이 모듈은 4개의 Transformer layer와 하나의 linear projection layer로 구성된 Ego Decoder $D_{ego}$를 탑재하고 있습니다. Ego Decoder는 추출된 token representation을 처리하여 입력 이미지와 매핑되는 2차원 해상도의 최종 egocentric gaze heatmap $H^A$와 $H^B$를 산출합니다.

$$H^A = D_{ego}(F^A)$$
$$H^B = D_{ego}(F^B)$$

전체 네트워크의 종단간 최적화를 위한 손실 함수 $L$은 각 착용자의 시선 추정 정확도를 직접 지도하는 gaze prediction loss와, 두 착용자 사이의 시각 표현 일관성을 강제하는 alignment loss의 합으로 설계되었습니다.

$$L = L_{gaze}^A + L_{gaze}^B + L_{ego-exo}$$

각 손실항의 상세 역할은 다음과 같습니다:

- $L_{gaze}$: 예측된 히트맵과 ground truth 시선 히트맵 사이의 차이를 픽셀 수준에서 계산하는 픽셀 단위 교차 엔트로피 손실입니다. 모델이 시선이 머무는 정확한 2차원 좌표 근방을 잘 짚어내도록 강제합니다.
- $L_{ego-exo}$: 두 시점 간의 시선 표현 일관성을 보장하기 위한 정렬 손실입니다. 훈련 단계에 적용하는 정렬 기법의 설계에 따라 Triplet loss 기반의 Time Synchronization Loss, Cross-Entropy loss 기반의 Explicit Matching Loss, 또는 Entropy loss 기반의 Implicit Matching Loss가 손실항으로 유동적으로 적용됩니다.

---

## 4. 실험 결과 (Experimental Results)

### 4.1 Datasets & Evaluation Metrics

실험은 Aria 안경으로 수집된 대규모 멀티모달 대화 데이터셋인 RLR-CHAT 및 Ego4D 데이터셋에서 진행되었습니다.

![Figure 2. RLR-CHAT 세션 분포](/images/ego-exo-gaze/_page_2_Figure_0.jpeg)

*Figure 2: RLR-CHAT 세션 분포*

평가 메트릭으로는 $L_2$ 거리를 측정하는 Distance의 평균 및 중앙값, 그리고 시선이 타인의 머리에 도달했는지를 측정하는 LAH의 Precision, Recall, F1 점수를 사용합니다.

### 4.2 Egocentric Gaze Estimation Performance

RLR-CHAT golden subset에서의 베이스라인 성능 비교 결과는 다음과 같습니다.

| Model | Distance (Mean) ↓ | Distance (Median) ↓ | LAH Prec ↑ | LAH Recall ↑ | LAH F1 ↑ |
|-------|-------------------|---------------------|------------|--------------|----------|
| Predict center | 0.107 | 0.093 | 0.633 | 0.146 | 0.237 |
| Predict avg of train data | 0.105 | 0.092 | 0.638 | 0.130 | 0.216 |
| Predict closest head to center | 0.131 | 0.073 | 0.396 | 0.863 | 0.543 |
| U-Net | 0.105 | 0.072 | 0.520 | 0.610 | 0.561 |
| MAV-Gaze | 0.098 | 0.065 | 0.617 | 0.724 | 0.667 |
| EgoGazeViT (Standard Training) | 0.096 | 0.057 | 0.507 | 0.798 | 0.620 |

*Table 2: RLR-CHAT golden subset에서의 베이스라인 비교*

Standard Training으로 훈련된 EgoGazeViT는 단일 프레임 이미지 입력만으로 가장 우수한 Distance 점수를 기록했습니다.

| Initialization | Distance (Mean) ↓ | Distance (Median) ↓ | LAH Prec ↑ | LAH Recall ↑ | LAH F1 ↑ |
|----------------|-------------------|---------------------|------------|--------------|----------|
| Standard Training | 0.102 | 0.057 | 0.538 | 0.819 | 0.650 |
| Synchronization | 0.100 | 0.055 | 0.536 | 0.843 | 0.656 |
| Implicit Matching | 0.101 | 0.056 | 0.533 | 0.833 | 0.650 |
| Explicit Matching | 0.101 | 0.055 | 0.545 | 0.836 | 0.660 |

*Table 3: EgoGazeViT의 초기화 방법에 따른 시선 추정 성능 비교*

Explicit Matching 초기화를 적용했을 때 가장 높은 LAH F1 점수(0.660)를 얻었습니다.

### 4.3 Probing for Exocentric Gaze

Encoder가 exocentric gaze representation을 실제로 포착하는지 확인하기 위해 frozen Encoder 뒤에 2-layer MLP 구조의 Exo Decoder $D_{exo}$를 붙여 LAH를 예측하는 probing 실험을 수행했습니다.

![Figure 4. 외부 시선 표현 프로빙 아키텍처](/images/ego-exo-gaze/_page_7_Figure_0.jpeg)

*Figure 4: 외부 시선 표현 프로빙 아키텍처*

| Initialization | LAH AP ↑ |
|----------------|----------|
| Random init | 0.178 |
| Standard Training | 0.262 |
| Synchronization | 0.498 |
| Implicit Matching | 0.371 |
| Explicit Matching | 0.304 |

*Table 5: RLR-CHAT에서의 exocentric gaze probing 결과*

Synchronization 기법이 0.498의 LAH AP를 기록하며 가장 우수한 exocentric representation 포착 능력을 보였습니다.

### 4.4 Qualitative Results

![Figure 5. RLR-CHAT에서의 정성적 결과. 상단은 egocentric gaze를 보여주며 초록색 점은 ground truth 시선 위치입니다. 하단은 exocentric gaze 예측 대상인 LAH 결과를 나타냅니다. 모델은 대부분의 경우 시선 대상을 정확히 식별하며, 외부 시선 단서를 활용하여 모호성을 해소합니다.](/images/ego-exo-gaze/_page_7_Figure_4.jpeg)

*Figure 5: RLR-CHAT에서의 정성적 결과*

---

## 5. 결론 및 시사점 (Conclusion and Key Takeaways)

본 논문은 대화 장면에서 착용자 간 동시 관찰 영상을 활용해 ego 및 exo gaze representation을 결합하여 학습하는 self-supervised alignment 아키텍처를 제안했습니다.

추론 단계에서는 한쪽 브랜치(EgoGazeViT)만을 활용함으로써 추가적인 연산이나 동시 영상 입력 없이도 단일 프레임 egocentric gaze estimation 성능을 향상시켰습니다.

실험 결과, Explicit Matching은 egocentric 시선 추정에서 우수한 성능을 나타냈고, Synchronization은 exocentric representation 포착 및 교차 데이터셋 일반화에서 뛰어난 강점을 보였습니다.

이러한 우수한 우수성을 인정받아 CVPR 2026의 GAZE 2026 워크숍에서 Best Poster Award를 수상했습니다. 경량 웨어러블 기기에서 타인의 gaze cues를 활용한 고성능 시선 추정의 실용적 가능성을 제시하며, 향후 spatial audio 통합 및 temporal 확장 연구의 중요한 기초를 마련했습니다.
