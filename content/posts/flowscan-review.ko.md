---
title: "[CVPRW 2026] FlowScan: Self-Supervised 특징과 Flow Matching Regularization을 활용한 Gaze Scanpath 예측"
date: 2026-08-14T18:53:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Prediction", "Scanpath", "Flow Matching", "Self-Supervised Learning", "CVPRW 2026"]
categories: ["Paper Review"]
summary: "CVPR 2026 GAZE Workshop에서 발표된 FlowScan 논문을 리뷰합니다. Frozen DINOv3 backbone, deformable pixel decoder, 그리고 inference 시 제거되는 auxiliary flow matching head를 결합하여 COCO-Search18 벤치마크의 모든 지표에서 기존 SOTA 모델인 HAT를 뛰어넘는 성능을 달성한 연구입니다."
cover:
  image: "/images/flowscan/flowscan_architecture.jpeg"
  alt: "FlowScan Architecture Overview"
---

> 논문 정보
> - 제목: FlowScan: Self-Supervised Features and Flow Matching Regularization for Gaze Scanpath Prediction
> - 저자: Brahan Aklilu*, Ofir Itzhak Shahar*, Ohad Ben-Shahar (*Equal contribution)
> - 소속: Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Israel
> - 학회: CVPR 2026 GAZE Workshop (7th International Workshop on Eye and Gaze in Computer Vision)

---

## 1. 한 줄 요약

기존 HAT 프레임워크에 frozen DINOv3 backbone, deformable pixel decoder, 그리고 훈련 시에만 사용되는 flow matching auxiliary head를 도입하여, 추가적인 추론 연산 비용 없이 COCO-Search18 벤치마크의 모든 scanpath 평가 지표를 개선한 논문입니다.

---

## 2. 연구 배경 및 동기

### 2.1 Problem Definition

Visual search는 일상에서 가장 보편적으로 일어나는 시각적 행동입니다. 사람이 장면 속에서 특정 물체를 찾고자 할 때, 시각 시스템은 상향식 시각적 두드러짐(bottom-up saliency), 주어진 탐색 목표에 따른 하향식 유도(top-down task guidance), 그리고 이전 응시 위치에 대한 기억을 종합하여 일련의 시선 궤적, 즉 scanpath를 생성합니다.

Scanpath 예측은 단순히 사람들이 평균적으로 어디를 바라보는지 정적인 확률 맵으로 나타내는 saliency map 예측과 구별됩니다. Scanpath 예측은 매 시선 고정(fixation)이 탐색 과제와 장면 맥락, 과거의 시선 이력에 순차적으로 영향을 받는 autoregressive generation 과제로 정의됩니다. 이러한 모델링은 인간-컴퓨터 상호작용(HCI), 시각 보조 기술, 이미지 검색, 그리고 인간 시각 주의 메커니즘 규명 등 다양한 응용 분야에서 핵심적인 역할을 수행합니다.

### 2.2 Limitations of Existing Methods

현재 COCO-Search18 벤치마크에서 최고의 성능을 보여준 모델은 HAT(Human Attention Transformer)입니다. HAT는 task-conditioned transformer encoder-decoder, foveated working memory, multi-scale deformable attention pixel decoder를 결합하여 강력한 기준점을 제시하였습니다. 하지만 저자들은 기존 HAT의 구조에 두 가지 중요한 한계가 존재함을 지적합니다.

첫째, HAT가 채택한 지도학습 기반 ResNet-50 backbone은 ImageNet 분류 레이블에 맞춰진 특징을 추출합니다. 이로 인해 목표 지향적 visual search에 필수적인 세밀한 공간 구조와 풍부한 의미론적 맥락을 포착하는 데 한계가 있습니다. 반면 self-supervised ViT 계열(DINO 시리즈)은 레이블 없이도 객체의 경계와 조밀한 의미론적 대응 관계를 잘 보존하므로, 시각 탐색 과제에 훨씬 적합한 시각 표현을 제공합니다.

둘째, HAT는 오직 2D fixation heatmap을 정답과 비교하는 focal loss만으로 학습됩니다. 이 손실 함수는 예측 확률 분포의 정점(peak)만을 간접적으로 감독할 뿐, decoder 내부의 특징 표현이 정밀한 2차원 시선 좌표를 직접 인코딩하도록 제약하지 못합니다.

### 2.3 Main Contributions

본 연구의 주된 기여는 다음과 같이 정리할 수 있습니다:

- Frozen self-supervised ViT backbone 도입: 지도학습 기반 ResNet-50을 완전히 고정된 DINOv3 ViT-B/16으로 대체하여 Sequence Score(SS)를 12.4% 향상시켰습니다.
- Flow matching regularization 제안: auxiliary flow matching head를 통해 decoder 내부 특징에 직접적인 좌표 회귀 신호를 주입하였습니다. 이 모듈은 추론 시 완전히 제거되어 추가 연산 비용을 전혀 발생시키지 않으면서도 SS를 12.2% 개선하였습니다.
- 공정한 비교를 통한 전 지표 성능 개선: 동일한 데이터 분할(seed 42)에서 HAT를 직접 재학습하여 엄밀한 1:1 비교를 수행하였으며, 5가지 평가 지표 전체에서 기존 SOTA를 일관되게 상회함을 입증하였습니다(SS 0.603 vs. 0.575, SemSS 0.538 vs. 0.499).

---

## 3. 제안 프레임워크

FlowScan은 기존 HAT 아키텍처를 기반으로 하되, 시각 표현 추출, 다중 해상도 디코딩, 내부 표현 정규화 방식에 세 가지 핵심 개선을 더한 구조를 갖습니다. 전체 프레임워크의 구조와 데이터 흐름은 아래 그림과 같습니다.

![FlowScan Architecture](/images/flowscan/flowscan_architecture.jpeg)

*Figure 1: FlowScan 아키텍처 개요. Frozen DINOv3 backbone에서 추출된 patch token이 deformable pixel decoder를 거쳐 coarse(P1) 및 fine-grained(P4) 표현으로 변환됩니다. Foveated Working Memory는 주변 맥락(pooled P1)과 중심와 크롭(P4 기반)을 결합하여 token sequence를 구성하고, transformer encoder-decoder가 task-conditioned query를 생성하여 dense heatmap head와 soft-argmax decoding을 수행합니다. Flow matching head(점선)는 훈련 시에만 보조 정규화기로 작동하며 추론 시에는 제거됩니다.*

### 3.1 Frozen DINOv3 Backbone

Visual search에서는 단순히 사물이 무엇인지 분류하는 것을 넘어, 탐색 목표와 장면 속 객체들 사이의 공간적 배치 및 세밀한 객체 경계를 이해하는 능력이 필수적입니다. 지도학습 기반의 CNN은 분류 레이블에 편향된 특징을 학습하기 쉬운 반면, self-supervised ViT는 이미지 전체의 dense context와 객체 단위의 구조를 매우 풍부하게 담아냅니다. FlowScan은 이러한 고품질 시각 특징을 온전히 활용하기 위해 대규모 큐레이션 데이터셋(LVD-142M) 기반으로 학습된 DINOv3 ViT-B/16을 시각 특징 추출기로 채택합니다.

특히 backbone을 파인튜닝하지 않고 완전히 고정(frozen)함으로써, representation collapse를 방지하고 학습 효율성을 극대화합니다. DINOv3는 register token을 도입하여 attention map의 artifact를 억제하고 패치 단위의 특징 표현력을 높인 모델입니다.

입력 이미지($320 \times 512$)는 패치 분할을 거쳐 $20 \times 32$ 그리드 형태의 768차원 patch token 시퀀스로 변환됩니다. Backbone의 85.7M개 파라미터는 완전히 고정되므로, 모델 전체에서 학습 대상 파라미터는 약 11M개에 불과합니다. 덕분에 단일 NVIDIA RTX 4090 GPU 환경에서도 약 4시간 만에 전체 학습이 완료됩니다.

### 3.2 Deformable Pixel Decoder

인간이 시각 탐색을 수행할 때는 전체적인 장면 맥락(주변 시각)과 집중해서 바라보는 영역의 세밀한 디테일(중심와 시각)을 동시에 활용합니다. 이를 모델에서 구현하려면 넓은 시야를 포괄하는 저해상도 특징과 정밀한 위치를 짚어내는 고해상도 특징을 모두 만들어낼 수 있어야 합니다. Deformable Pixel Decoder는 ViT가 추출한 단일 해상도의 patch token을 다중 해상도 피라미드로 확장하고, 중요한 위치에만 선택적으로 attention을 수행하여 효율적이면서도 풍부한 공간 표현을 완성하는 역할을 담당합니다.

먼저 backbone의 patch token을 선형 투영하여 $d = 256$차원으로 맞춘 후 2D feature map으로 재구성합니다. 이후 strided convolution을 적용하여 3단계 Feature Pyramid Network(FPN)를 구축합니다. 이 다중 해상도 피라미드 위에서 6개 층으로 구성된 Multi-Scale Deformable Attention(MSDeformAttn, 8 head, level당 4개 sampling point)을 거쳐 서로 다른 스케일 간 정보 교환이 이루어집니다.

최종 디코더 출력은 두 가지 해상도 레벨로 분리되어 후속 모듈에 전달됩니다:

- P1 (coarse): 원본 ViT 그리드 해상도($20 \times 32$)를 선형 투영한 특징 맵으로, 전체적인 주변 시각 맥락(peripheral context)을 제공합니다.
- P4 (fine): 학습 가능한 transposed convolution을 통해 4배 업샘플링된 $79 \times 127$ 해상도의 세밀한 특징 맵으로, 중심와 영역의 크롭(foveal crop)과 고해상도 fixation heatmap 예측에 활용됩니다.

두 레벨 모두 $d = 256$차원을 유지합니다. 아울러 기존 HAT의 custom CUDA kernel 의존성을 제거하고 pure PyTorch로 deformable attention을 재구현하였습니다.

### 3.3 Foveated Working Memory 및 Transformer Encoder-Decoder

인간의 눈은 중심와(fovea)에서는 선명한 고해상도 정보를, 주변부(periphery)에서는 거친 저해상도 정보를 받아들이며, 지금까지 탐색했던 위치들의 기억(working memory)을 바탕으로 다음 시선 위치를 결정합니다. FlowScan은 이러한 인간 시각 인지 메커니즘을 모사하여, 전체 장면의 거친 맥락과 이전 fixation 지점들의 선명한 국소 정보를 결합한 Working Memory를 구성합니다. 그리고 Transformer Encoder-Decoder를 통해 사용자가 찾고자 하는 탐색 목표(task category)에 맞추어 다음 fixation을 유도하는 쿼리 벡터를 생성합니다.

각 시선 생성 스텝에서 Foveated Working Memory는 두 가지 토큰 집합으로 구성됩니다:

- Peripheral token ($7 \times 7 = 49$개): P1 특징 맵을 adaptive average pooling하여 고정된 그리드로 압축한 토큰입니다. 탐색 전 과정에 걸쳐 공유되는 전체 장면 맥락을 나타냅니다.
- Foveal token (스텝당 $7 \times 7 = 49$개): 현재까지 누적된 각 fixation 위치를 중심으로 P4 특징 맵에서 국소 영역을 크롭하여 얻은 토큰입니다. 과거에 응시했던 영역들의 세밀한 시각 정보를 담고 있습니다.

각 토큰에는 스케일 임베딩(peripheral vs. foveal), 시간 임베딩(fixation 순서), 2차원 공간 위치 임베딩이 더해집니다. 따라서 $k$번째 fixation 이력까지 고려할 때 총 $49 + 49k$개의 메모리 토큰이 생성됩니다.

이 토큰 시퀀스는 3-layer Transformer Encoder(4 head, FFN 차원 1024, Pre-LN)를 거쳐 심층적으로 인코딩됩니다. 이후 6-layer Transformer Decoder는 탐색 목표 카테고리(COCO-Search18의 18개 클래스 중 하나)의 학습 가능한 task embedding을 초기 쿼리로 삼아 인코딩된 메모리에 cross-attention을 수행합니다. 디코더의 최종 출력은 다음 fixation에 대한 종합적인 정보를 압축한 단일 $d$차원 쿼리 벡터가 됩니다.

### 3.4 Output Heads

Transformer Decoder가 출력하는 결과물은 단지 '어떤 시각적 특징과 맥락에 주목해야 하는지'를 요약한 1차원 벡터(query) 형태입니다. 하지만 scanpath 생성을 위해 우리가 실제로 얻어야 하는 최종 정보는 두 가지입니다:

1. 2D 이미지 화면 상에서 다음에 응시할 정확한 위치 좌표 $(x, y)$
2. 현재 응시를 끝으로 탐색을 마칠 것인가에 대한 종료 여부

Output Heads는 이 1차원 쿼리 벡터를 해석하여 실제 시선 위치와 종료 확률로 변환해주는 역할을 담당합니다. 세부적으로 다음과 같은 세 단계 모듈로 동작합니다.

첫째, Dense Heatmap Head는 다음에 바라볼 위치의 확률 지도(heatmap)를 생성합니다. 디코딩된 1차원 쿼리 벡터를 2-layer MLP($d \to d$, ReLU)로 변환한 뒤, 앞서 Pixel Decoder가 생성한 고해상도 특징 맵 P4와 픽셀별 내적(pixel-wise dot product)을 수행합니다. 즉, 쿼리가 가리키는 탐색 정보와 이미지 각 위치의 시각 특징이 얼마나 잘 부합하는지 2차원 유사도 점수를 계산하는 것입니다. 이 점수 맵에 공간적 softmax를 적용하여 2D 확률 분포를 얻고, 경량 CNN 정제 모듈(Conv $1 \to 16$, GroupNorm, ReLU, Conv $16 \to 1$)을 거쳐 $79 \times 127$ 해상도의 최종 fixation 확률 맵을 완성합니다. 훈련 단계에서는 실제 정답 위치 주변에 Gaussian blur($\sigma = 1.5$ pixel)를 적용한 target에 대해 Focal Loss를 적용하여 학습합니다.

둘째, Soft-argmax Decoding은 확률 지도에서 정밀한 연속 좌표를 추출합니다. 완성된 히트맵에서 단순히 가장 확률이 높은 단일 격자 픽셀(argmax)을 선택하면, 이산적인 정수 격자 단위로만 좌표가 결정되어 미세한 시선 위치를 놓치게 됩니다. 이를 해결하기 위해 FlowScan은 히트맵의 정점(peak) 주변 $9 \times 9$ 윈도우 내부의 확률 분포를 가중치로 삼아 무게중심(centroid)을 계산하는 Soft-argmax를 적용합니다. 이를 통해 픽셀 사이의 연속적인 실수 좌표(sub-pixel precision)를 매끄럽게 얻어냅니다.

셋째, Termination Head는 탐색 종료 여부를 판정합니다. 2-layer MLP($256 \to 128 \to 1$, Dropout 적용)로 구성되며, 현재 fixation에서 목표 객체를 찾았다고 판단하여 탐색을 마칠 확률을 예측합니다. 이 헤드는 정답 종료 시점과의 오차를 줄이도록 Binary Cross-Entropy Loss로 학습됩니다.

### 3.5 Flow Matching Auxiliary Head

기존 프레임워크가 사용하는 Heatmap 기반 Focal Loss는 2차원 격자판 위에서 정답 픽셀 주변에만 불을 켜주는 방식입니다. 이 방식은 최고점 픽셀 주변에만 그래디언트를 집중시키기 때문에, 디코더가 좌표 공간에서의 거리감(정답에서 얼마나 멀리 떨어져 있는지)을 직접 학습하기 어렵습니다. 즉, 디코더 내부의 특징 벡터 자체가 정밀한 2차원 좌표를 온전히 인코딩하고 있어야 고품질 히트맵이 유도되는데, 2D 히트맵을 거쳐 간접적으로만 감독을 받다 보니 좌표 표현력이 다소 뭉개지는 문제가 있었습니다.

FlowScan은 생성 모델 분야에서 발전한 Flow Matching 기법을 내부 특징을 단련하는 훈련 도우미(auxiliary regularizer)로 도입하여 이 문제를 해결합니다.

Flow Matching의 기본 아이디어는 무작위 노이즈에서 실제 데이터로 이동하는 매끄러운 '속도장(velocity field)'을 학습하는 것입니다. 이를 시선 예측 과제에 쉽게 비유하자면, 화면 위 아무 곳에나 떨어진 무작위 점(노이즈)이 디코더가 건네주는 힌트(디코더 출력 벡터)를 나침반 삼아 실제 정답 fixation 좌표까지 미끄러지듯 직진하여 도착하도록, 각 위치마다 이동 방향과 속도를 알려주는 '화살표 지도'를 학습시키는 것입니다. 이 화살표 지도를 올바르게 그리려면 디코더의 내부 특징이 정답 좌표의 정확한 위치와 방향을 완벽하게 꿰뚫고 있어야만 합니다.

결정적으로 이 Flow Matching Head는 오직 학습 시에 디코더의 내부 특징 표현을 정규화하는 용도로만 사용되며, 실제 추론(inference) 시에는 완전히 제거됩니다. 따라서 실제 scanpath 생성 시 연산 지연이나 메모리 소모가 전혀 발생하지 않습니다.

구체적인 포뮬레이션은 다음과 같습니다. Flow Matching Head는 디코더의 출력 벡터 $c$를 조건(conditioning context)으로 받아들이는 3-layer MLP 형태의 속도장 $v_\theta(x_t, t \mid c)$로 정의됩니다.

실제 다음 fixation 좌표를 $x_1 \in [0, 1]^2$(정규화된 2D 이미지 좌표), 표준 정규분포에서 샘플링한 초기 노이즈를 $x_0 \sim \mathcal{N}(0, I)$라 할 때, 임의의 시간 $t \sim \mathcal{U}(0, 1)$에 대한 선형 보간 경로 $x_t$는 다음과 같이 정의됩니다:

$$x_t = (1 - t)x_0 + tx_1$$

이때 직선 경로를 따라 이동하는 이상적인 목표 속도 벡터는 $v^* = x_1 - x_0$이 되며, Flow Matching 보조 손실 함수는 예측 속도장과 목표 속도 벡터 간의 평균제곱오차(MSE)로 정의됩니다:

$$\mathcal{L}_{\text{FM}} = \| v_\theta(x_t, t \mid c) - (x_1 - x_0) \|^2$$

이 보조 손실 함수는 디코더 학습에 다음과 같은 세 가지 결정적 이점을 제공합니다:

- 정답 좌표 주변에 완만하고 연속적인 loss landscape를 형성하여 학습을 안정화합니다.
- 2D 히트맵을 거치지 않고 좌표 오차 그래디언트를 디코더 특징 벡터로 즉각 역전파합니다.
- 속도장이 특정 좌표 지점으로 수렴하도록 유도함으로써 시선의 공간적 국소성(locality)을 암묵적으로 제약합니다.

이 헤드는 약 150K개의 파라미터(전체 학습 가능 파라미터의 약 1.4%)로 구성되어 학습 오버헤드가 매우 미미하며, 추론 시에는 완전히 버려지므로 추가 비용 없이 디코더의 표현력만 극대화하는 효과를 냅니다.

### 3.6 전체 학습 목적 함수

모델 전체는 2D fixation heatmap 정확도, flow matching 기반의 내부 특징 정규화, 그리고 탐색 종료 시점 예측이라는 세 가지 목표를 균형 있게 달성하도록 통합 학습됩니다.

전체 손실 함수 $\mathcal{L}$은 세 손실 함수의 가중합으로 구성됩니다:

$$\mathcal{L} = \lambda_{\text{heat}} \mathcal{L}_{\text{focal}} + \lambda_{\text{FM}} \mathcal{L}_{\text{FM}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}}$$

여기서 $\mathcal{L}_{\text{focal}}$은 heatmap 예측에 대한 focal loss, $\mathcal{L}_{\text{FM}}$은 속도장 학습을 위한 flow matching auxiliary loss, $\mathcal{L}_{\text{term}}$은 종료 예측을 위한 binary cross-entropy loss입니다. 하이퍼파라미터 민감도 분석을 통해 최적화된 기본 가중치는 $\lambda_{\text{heat}} = 1.0, \lambda_{\text{FM}} = 0.5, \lambda_{\text{term}} = 1.0$입니다.

### 3.7 Scanpath 생성

실제 추론 과정에서는 Flow Matching Head를 떼어내고, 인간의 시각 탐색 행동처럼 이전 응시 위치를 바탕으로 한 단계씩 다음 응시 위치를 순차적으로 예측하는 autoregressive 방식으로 scanpath를 완성합니다.

COCO-Search18 표준 프로토콜에 따라 화면 정중앙을 첫 번째 fixation으로 설정하고 시작합니다. 이후 각 스텝마다 다음 4단계를 반복합니다:

1. 현재까지의 fixation 이력으로부터 peripheral 토큰과 foveal 크롭 토큰을 결합하여 Working Memory를 구성합니다.
2. Transformer Encoder로 메모리를 인코딩하고, Transformer Decoder에서 주어진 탐색 과제 임베딩과 cross-attention을 수행하여 쿼리 벡터를 추출합니다.
3. Dense Heatmap Head와 Soft-argmax Decoding을 거쳐 sub-pixel 수준의 다음 fixation 좌표를 결정합니다.
4. Termination Head를 통해 탐색 종료 확률을 계산합니다.

이 과정은 최대 $K = 6$ 스텝(초기 중앙 응시 제외)에 도달하거나, 종료 확률이 0.5를 초과할 때까지 지속됩니다.

---

## 4. 실험 결과

### 4.1 데이터셋 및 평가 프로토콜

실험은 목표 지향적 scanpath 예측의 표준 벤치마크인 COCO-Search18에서 수행되었습니다. 이 데이터셋은 10명의 피험자가 18개 COCO 객체 카테고리에 대해 6,202장의 이미지에서 수행한 시선 추적 데이터를 포함합니다. 본 논문은 탐색 대상이 이미지 내에 실제로 존재하는 target-present(TP) 조건에 초점을 맞춥니다.

COCO-Search18의 원본 test set은 비공개 상태이므로, 저자들은 validation set을 70:30 비율(seed 42)로 무작위 분할하여 977개의 test scanpath(319개 image-task 쌍)를 구축하였습니다. 특히 기존 HAT 모델이 보고한 수치는 비공개 데이터 분할에 기반한 것이므로, 엄밀하고 공정한 1:1 비교를 위해 저자들의 공식 코드를 사용하여 동일한 split에서 HAT를 직접 재학습(HAT retrained)하였습니다.

평가 지표로는 scanpath의 공간적 유사도를 평가하는 Sequence Score(SS), 의미론적 유사도를 평가하는 Semantic Sequence Score(SemSS), 그리고 매 스텝의 현저성 맵 품질을 평가하는 조건부 지표들인 conditional Information Gain(cIG), conditional NSS(cNSS), conditional AUC(cAUC)를 사용하였습니다.

### 4.2 메인 결과

| Method | SS↑ | SemSS↑ | cIG↑ | cNSS↑ | cAUC↑ |
|---|---|---|---|---|---|
| HAT retrained | 0.575 | 0.499 | 2.491 | 4.649 | 0.915 |
| FlowScan | 0.603 | 0.538 | 2.759 | 4.810 | 0.928 |

*Table 1: COCO-Search18 target-present 조건에서의 메인 비교. 동일 test split(seed 42) 기준.*

동일한 test split 상에서 FlowScan은 Sequence Score(SS) 0.603을 달성하여 HAT retrained(0.575) 대비 4.9% 향상된 성능을 기록하였습니다. 또한 의미론적 유사도를 측정하는 SemSS에서도 0.538을 기록하여 HAT retrained(0.499) 대비 7.8% 높은 수치를 달성하였으며, cIG, cNSS, cAUC 등 모든 조건부 현저성 지표에서도 일관되게 우위를 보였습니다.

흥미로운 점은 HAT의 논문 발표 수치(SS 0.470)와 동일 데이터 split에서 직접 재학습한 수치(SS 0.575) 사이에 상당한 차이가 존재한다는 사실입니다. 이는 데이터 분할 및 평가 코드 구현에 따라 측정값의 편차가 클 수 있음을 시사하며, 본 연구처럼 동일한 데이터셋 분할과 평가 프로토콜을 적용한 1:1 재학습 비교가 필수적임을 보여줍니다.

Scanpath 길이 분포 관점에서도 유의미한 차이가 관찰됩니다. HAT retrained는 실제 사람의 정답 평균 길이(3.68)에 비해 지나치게 짧은 scanpath(평균 2.58)를 생성하며 전체 예측의 60.5%가 2회의 fixation만으로 조기 종료된 반면, FlowScan은 평균 3.77회의 fixation을 생성하여 실제 인간의 탐색 길이와 매우 유사하게 종료 시점을 보정해냅니다.

### 4.3 Backbone 비교

| Backbone | Pre-training | SS↑ | SemSS↑ | cIG↑ | cNSS↑ | cAUC↑ |
|---|---|---|---|---|---|---|
| ResNet-50 | Supervised | 0.517 | 0.439 | −0.31 | 3.25 | 0.816 |
| ViT-B/16 | Supervised | 0.568 | 0.490 | 1.30 | 3.83 | 0.871 |
| DINO ViT-B/16 | SSL | 0.559 | 0.473 | 1.51 | 3.90 | 0.887 |
| DINOv2 ViT-B/14 | SSL | 0.586 | 0.514 | 1.83 | 4.19 | 0.896 |
| DINOv3 ViT-B/16 | SSL | 0.581 | 0.501 | 2.11 | 4.17 | 0.901 |

*Table 2: Backbone 비교 (FPN pixel decoder, FM 활성화, 모든 backbone frozen).*

다양한 backbone을 고정한 채 FPN pixel decoder 및 Flow Matching을 적용하여 비교한 결과, self-supervised ViT 모델들이 지도학습 기반 모델들을 큰 폭으로 앞섰습니다. DINOv2는 SS(0.586)와 SemSS(0.514)에서 가장 높은 수치를 보였고, DINOv3는 조건부 지표인 cIG(2.11)와 cAUC(0.901)에서 가장 뛰어난 성능을 나타냈습니다. 저자들은 매 스텝마다의 예측 정보량을 반영하는 conditional saliency 성능을 고려하여 최종 모델의 backbone으로 DINOv3를 채택하였습니다.

모든 ViT 계열 backbone은 지도학습 ResNet-50 대비 SS 기준 +0.042~0.069의 큰 격차를 보이며, 파라미터를 고정한 상태에서도 vision transformer 특징이 scanpath 예측에 훨씬 적합함을 입증하였습니다.

### 4.4 Flow Matching Ablation

| Flow Matching | SS↑ | ΔSS | SemSS↑ | cIG↑ | cNSS↑ |
|---|---|---|---|---|---|
| 활성화 | 0.581 | — | 0.501 | 2.11 | 4.17 |
| 비활성화 | 0.518 | −0.063 | 0.434 | 0.78 | 3.81 |

*Table 3: Flow matching 효과 (DINOv3 backbone, FPN pixel decoder). FM은 inference 시 제거됩니다.*

Flow Matching auxiliary head의 효과를 분석한 결과, 추론 시에는 이 모듈이 완전히 제거됨에도 불구하고 SS가 0.518에서 0.581로 12.2%(+0.063) 상승하였습니다. 특히 cIG 지표가 0.78에서 2.11로 대폭 상승하였는데, 이는 flow matching 손실 함수가 디코더 내부의 특징 표현으로 하여금 정밀한 공간 좌표를 인코딩하도록 효과적으로 유도하였음을 보여줍니다.

### 4.5 Deformable Pixel Decoder 효과

단순 FPN 구조(SS 0.581)에서 Deformable Pixel Decoder(SS 0.603)로 교체하였을 때 SS가 3.8%(+0.022) 추가 향상되었으며, cIG는 2.11에서 2.76으로 31% 상승하였습니다. Multi-scale deformable attention을 통한 스케일 간 특징 융합이 foveal crop과 고해상도 heatmap 생성 모두의 품질을 높이는 데 기여함을 알 수 있습니다.

### 4.6 Hyperparameter Sensitivity

![Sensitivity Analysis](/images/flowscan/flowscan_sensitivity.jpeg)

*Figure 2: 손실 가중치에 대한 민감도 분석. (a) $\lambda_{\text{heat}}$: SS는 안정적이나 cIG는 가중치에 따라 증가하며 1.0이 최적의 균형점입니다. (b) $\lambda_{\text{FM}} = 0.5$에서 최적의 정규화 효과를 달성합니다. (c) $\lambda_{\text{term}} = 1.0$이 SS와 cIG 모두에 결정적인 영향을 미칩니다.*

손실 가중치 분석 결과, $\lambda_{\text{heat}}$는 1.0에서 scanpath 구조 유사도와 fixation 정보량 간 최적의 균형을 이루었습니다. $\lambda_{\text{FM}}$의 경우 0.5에서 cIG가 2.76으로 정점을 기록하였으며, 가중치가 너무 커지면(2.0일 때 0.586) 디코더를 과도하게 제약하여 성능이 저하되었습니다. $\lambda_{\text{term}}$은 모델 성능에 가장 민감하게 작용하여, 1.0에서 벗어날 경우 SS와 cIG가 모두 급격히 하락하였습니다.

| σ | SS↑ | SemSS↑ | cIG↑ | cNSS↑ |
|---|---|---|---|---|
| 1.0 | 0.595 | 0.528 | 2.518 | 4.758 |
| 1.5 | 0.603 | 0.538 | 2.759 | 4.810 |
| 2.0 | 0.595 | 0.523 | 2.535 | 4.710 |

*Table 4: Heatmap Gaussian blur 반경 $\sigma$에 따른 성능 변화 (DINOv3, deformable, FM).*

Ground-truth heatmap의 Gaussian blur 반경 $\sigma$는 1.5 pixel에서 전 지표 최적의 결과를 보였습니다. $\sigma = 1.0$으로 너무 작으면 그래디언트가 희소해지고, $\sigma = 2.0$으로 너무 커지면 위치 정밀도가 떨어지는 현상이 확인되었습니다.

### 4.7 정성적 결과

![Qualitative Results](/images/flowscan/flowscan_qualitative.jpeg)

*Figure 3: 정성적 scanpath 예측 결과. 각 행은 서로 다른 테스트 이미지이며 좌측에 탐색 과제가 표시되어 있습니다. 초록색 점선 박스는 target 객체, 번호가 부여된 원은 fixation 순서, 화살표는 saccade 이동 방향을 나타냅니다.*

정성적 결과에서 FlowScan은 인간 피험자의 실제 탐색 경로와 유사하게 주변 문맥을 살피며 체계적으로 목표 객체로 접근하는 양상을 보입니다. 반면 baseline 모델은 목표 객체와 무관한 위치를 맴돌거나 지나치게 이른 시점에 탐색을 중단하는 경향을 나타냅니다.

---

## 5. 결론 및 핵심 시사점

FlowScan은 기존 HAT 아키텍처에 세 가지 정교한 설계를 결합하여 COCO-Search18 target-present 조건에서 전 지표 성능 향상을 이끌어낸 연구입니다.

첫째, 시각 탐색 과제에서 self-supervised ViT의 우수성을 체계적으로 입증하였습니다. 고정된 DINOv3 backbone은 지도학습 ResNet-50 대비 Sequence Score를 크게 끌어올렸으며, 레이블 없이 학습된 dense semantic feature가 인간의 복잡한 시각 주의 메커니즘을 모델링하는 데 훨씬 적합함을 증명하였습니다. 이는 visual search에서 task-target 매칭과 foveal crop 추출의 품질을 동시에 개선하는 기반이 되었습니다.

둘째, 생성 모델의 Flow Matching 기법을 내부 표현 정규화 도구로 창의적으로 활용하였습니다. Heatmap focal loss가 2D 공간의 확률 분포 정점에만 그래디언트를 집중시키는 한계를 극복하기 위해, 가우시안 노이즈로부터 fixation 좌표로의 속도장을 학습하는 보조 헤드를 도입하였습니다. 이 방식은 연속 좌표 오차를 디코더 특징에 직접 주입하여 내부 표현력을 비약적으로 끌어올렸으며, 추론 시에는 해당 헤드를 완전히 제거함으로써 연산 비용 증가 없는 성능 개선을 달성하였습니다. 이러한 정규화 기법은 scanpath 예측을 넘어 다양한 순차적 좌표 예측 문제에도 확장 적용될 수 있는 잠재력을 지닙니다.

셋째, Deformable Pixel Decoder를 통해 다중 해상도 특징 융합의 중요성을 재확인하였습니다. Pure PyTorch로 재구현된 deformable attention은 주변 시각을 담당하는 거친 특징(P1)과 중심와 시각을 담당하는 미세 특징(P4) 사이의 유기적인 정보 교환을 가능하게 하여 모델의 예측 정밀도를 한 단계 더 높였습니다.

저자들은 향후 연구 방향으로 목표물이 존재하지 않는 target-absent search로의 확장, flow matching head를 직접 활용한 다양한 scanpath 샘플 생성, COCO-Search18 이외의 다양한 시선 추적 데이터셋 평가, 그리고 공간 의존적 saccadic prior의 도입을 제시하였습니다.
