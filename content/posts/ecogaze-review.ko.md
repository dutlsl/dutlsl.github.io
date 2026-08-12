---
title: "[CVPRW 2026 (GAZE Best Paper)] ECOGaze: 미래 정보가 인과적 시선 추정에 얼마나 도움이 되는가?"
date: 2026-08-12T16:19:00+09:00
draft: false
math: true
tags: ["Paper Review", "Egocentric Gaze Estimation", "Future-Privileged Supervision", "Knowledge Distillation", "Causal Inference", "CVPRW 2026"]
categories: ["Paper Review"]
summary: "미래 프레임 접근을 훈련 시에만 허용하는 제어된 프레임워크를 통해, 엄격히 인과적인 시선 추정 모델의 성능을 향상시키는 ECOGaze를 제안합니다. 최적의 미래 참조 범위가 약 1.7~3.3초의 유한한 시간 창 내에 존재함을 실증합니다."
cover:
  image: "/images/ecogaze/_page_0_Figure_9.jpeg"
  alt: "ECOGaze 프레임워크 개요"
---

## 1. 한 줄 요약

ECOGaze는 훈련 단계에서만 미래 프레임을 참조하는 future-privileged supervision 프레임워크입니다. 추론 시에는 미래 프레임 없이 strictly causal 방식으로 동작합니다. 훈련 중 미래 맥락의 예측적 단서를 효과적으로 모델에 전수하여, EGTEA Gaze+와 Ego4D 벤치마크에서 미래 참조의 효용이 약 1.7~3.3초의 유한한 시간 창 내에서 극대화됨을 입증합니다.

---

## 2. 연구 배경 및 동기

### 2.1 문제 정의

egocentric gaze estimation은 카메라 착용자의 gaze point를 비디오 영상으로부터 추론하는 과제입니다. AR 어시스턴트, 웨어러블 시스템, 대규모 attention 분석 등 다양한 실시간 인터랙션 어플리케이션의 핵심 기술입니다.

![Figure 1: 오프라인 vs. 온라인 시선 추정 설정의 차이](/images/ecogaze/_page_0_Figure_9.jpeg)
*Figure 1: 기존 오프라인 모델은 미래 프레임을 자유롭게 참조합니다. 반면 실시간 시스템은 과거와 현재 프레임만으로 gaze를 예측해야 합니다. ECOGaze는 훈련 시에만 미래 맥락을 사용하고, 추론 시에는 strictly causal하게 동작합니다.*

기존 고성능 모델은 대개 오프라인 환경을 가정합니다. 즉, 훈련과 추론 모두에서 전체 비디오 시퀀스를 양방향으로 참조합니다. 물론 미래 프레임을 일부 들여다보는 look-ahead 설정 역시, 일정 수준의 latency를 감수하면 실시간 온라인 추론이 아예 불가능하지는 않습니다.

그러나 사용자 움직임에 실시간으로 즉각 반응해야 하는 AR 헤드셋 환경에서는 단 몇 프레임의 latency도 치명적입니다. 더군다나 평가 단계에서 look-ahead를 허용하면, 실제 작동 환경보다 벤치마크 성능 수치가 부풀려지는(inflate) 왜곡이 발생합니다.

따라서 본 연구는 지연이 전혀 없는 strictly causal constraint를 적용합니다. 즉, 시점 t에서 예측한 gaze map $\hat{G}_t$는 미래 프레임 $x_s$ ($s > t$)에 전혀 의존하지 않아야 합니다 ($\frac{\partial \hat{G}_t}{\partial x_s} = 0$).

추론 시 미래 프레임을 단 1프레임도 참조하지 않는 엄격한 조건 하에서, 본 연구는 다음 질문들에 답하고자 합니다.

- 미래 맥락이 완전히 차단되면, gaze estimation에 유용한 anticipatory cue를 근본적으로 활용하지 못하는가?
- 만약 미래 정보가 유용하다면, 이를 causal model에 전수할 때 얼마만큼의 look-ahead horizon을 적용하는 것이 훈련 효율 면에서 최적인가?

### 2.2 기존 방법의 한계

초기 egocentric gaze estimation 연구는 hand-crafted feature, visual saliency, 단기 temporal modeling 등에 주로 의존했습니다. 최근에는 GLC 등 transformer 기반 모델이 도입되면서 spatiotemporal dependency를 포착하는 성능이 비약적으로 발전했습니다. 그러나 이러한 최신 모델조차 오프라인에서 양방향 temporal context를 전제하는 경우가 많습니다.

일부 RNN 계열 방법이 causal inference를 구현하긴 했으나, 훈련 시 미래 정보를 활용해 causal inference 성능 자체를 높이려는 체계적인 시도는 부족했습니다.

참고로 미래 gaze point를 예견하는 gaze anticipation 연구가 활발히 진행 중이나, 이는 현재 시점의 gaze를 복원하는 gaze estimation 연구와는 다른 목적을 가집니다. 또한 action recognition 영역에서 privileged supervision이나 knowledge distillation으로 temporal context를 다룬 사례는 있었으나, gaze estimation 분야에서 look-ahead horizon에 따른 효용 변화를 체계적으로 프로토콜화하여 다룬 연구는 없었습니다.

### 2.3 핵심 기여

본 연구의 핵심 기여는 다음과 같이 정리할 수 있습니다.

1. 미래 맥락 분석을 위한 제어 프레임워크 제안: egocentric gaze estimation을 causal online 설정으로 포멀하게 정의하고, 추론 아키텍처를 고정한 상태에서 look-ahead horizon의 영향만을 투명하게 분리 분석하는 프레임워크를 수립합니다.
2. 최적 미래 참조 범위의 실증적 규명: EGTEA Gaze+ 및 Ego4D 벤치마크 실험을 통해, future-privileged supervision이 causal baseline을 일관되게 개선하며 그 효과가 약 1.7~3.3초 범위에서 수렴함을 보입니다.
3. 실시간 gaze modeling에 대한 실무적 가이드라인 제시: 가벼운 구조의 causal decoder가 훈련 중 future-aware signal을 충분히 흡수하면서도 추론 시 causality를 엄격히 지킬 수 있음을 입증하여 실시간 시스템 구축의 가이드라인을 제공합니다.

---

## 3. 제안 방법: ECOGaze 프레임워크

### 3.1 전체 구조 개요

![Figure 2: ECOGaze 프레임워크 개요](/images/ecogaze/_page_3_Figure_0.jpeg)
*Figure 2: ECOGaze의 훈련 프로세스입니다. 동결된 DINOv3 인코더와 가중치가 공유되는 Shared Spatio-Temporal Decoder를 활용합니다. 두 브랜치는 temporal attention mask 설정만 다르게 씌워, future-aware teacher(상단)와 strictly causal student(하단)의 출력을 동시에 계산합니다. 훈련 종료 시 student만 실시간 추론에 활용합니다.*

ECOGaze 프레임워크는 visual representation과 decoder capacity를 완벽히 통제하고, 오직 훈련 중 look-ahead horizon $H$의 영향만을 정밀하게 평가하도록 구성되었습니다. 주요 물리적 컴포넌트는 다음과 같습니다.

- Frozen DINOv3: Meta의 사전 학습 모델 DINOv3 Vision Transformer를 장면 인코더로 사용하며, 파라미터를 완전히 동결(frozen)합니다. 연산 효율을 챙기는 한편, 훈련 도중 피처 표현력이 흔들리는 현상(feature drift 등)을 원천 통제하여 성능 변화 요인이 온전히 미래 정보 제공 덕분임을 증명하기 위한 장치입니다.
- Shared Spatio-Temporal Decoder: 동결된 DINOv3 피처로부터 spatiotemporal 흐름을 해석하는 Divided Space-Time Attention 기반의 경량 transformer decoder입니다. 훈련 단계에서 두 가지 temporal mask가 각각 덧씌워져 동시에 작동합니다.
  1. Future-aware teacher: temporal attention mask를 확장하여 과거, 현재 프레임 외에 $H$개의 미래 프레임까지 함께 보도록 열어줍니다.
  2. Strictly causal student: 하삼각(lower-triangular) temporal mask를 강제하여 과거와 현재 프레임만 참조할 수 있습니다 ($H=0$).
- GLF & Conv Head: 최종 spatial gaze probability map $\hat{G}_t$를 예측하기 위한 경량 모듈입니다. 전역 정보 유도를 위한 GLF(Global-Local Focusing) 연산을 수행한 뒤, $1 \times 1$ convolution과 temperature-scaled softmax를 거칩니다.

### 3.2 디코더 동형 설계의 의도

교사와 학생 브랜치가 동일한 디코더 파라미터를 공유하는 isomorphic 설계는 실험의 투명성을 위한 핵심 장치입니다. 만약 모델 성능이 훨씬 강력한 제3의 별도 teacher 모델을 도입한다면, 성능 발달 요인이 look-ahead 정보 유입 덕분인지 단순히 teacher 모델의 가중치 capacity 차이 때문인지 구분할 수 없게 됩니다. 디코더 가중치를 단일 모델로 결속해 둠으로써 오직 temporal mask(미래 참조 여부) 변경에 따른 정보 획득 효과만을 순수하게 발라낼 수 있습니다.

### 3.3 Global-Local Focusing 및 예측 파이프라인의 연산 원리

디코더의 마지막 레이어를 통과한 특징들은 GLF, $1 \times 1$ convolution, 그리고 temperature-scaled softmax 연산을 순차적으로 거치며 정제된 gaze map으로 변환됩니다. 각 단계의 구체적인 작동 원리와 도입 목적은 다음과 같습니다.

- GLF(Global-Local Focusing)의 residual gate 연산: transformer의 최종 패치 특징들에서 gaze와 밀접한 로컬 영역(손, 상호작용 중인 도구 등)을 동적으로 강조하는 장치입니다. 학습 가능한 prior(시선의 중앙 편향 등)와 프레임 전역 특징(global token)을 묶어 query vector를 구성합니다. 이 query와 로컬 패치 특징 간의 코사인 유사도를 계산하여 gaze와 무관한 배경 영역을 억제하고 관련 영역을 강조하는 residual gate $\alpha_t$를 생성합니다. 이를 아래 공식처럼 기존 특징에 잔차로 융합합니다.
  
  $X_{t,n}^{\text{focus}} = X_{t,n}^{(L)} + \alpha_{t,n} X_{t,n}^{(L)}$

- $1 \times 1$ convolution을 통한 채널 축소 및 업샘플링: GLF를 통과한 시퀀스 토큰들을 원래의 2D 공간 격자로 재배열하고 spatial 해상도로 복원합니다. 이후 $1 \times 1$ convolution을 적용하여 여러 채널의 차원을 1차원의 단일 맵(gaze map logit)으로 축소하며 목표 공간 크기(예: 64×64)로 매핑합니다.

- Temperature-scaled softmax의 완화 효과: 훈련 시 교사와 학생의 출력 분포를 완만하게 누그러뜨려(smoothing) 지식 증류의 안정성을 높입니다. 단순한 소프트맥스는 특정 위치에 확률이 과도하게 쏠리는 원인이 되어 부차적인 시선 후보군 영역들의 상대적 관계 정보를 유실시킵니다. 따라서 온도 파라미터 $\tau$(본 연구에서는 $\tau=2$로 설정)로 나누어 연산함으로써, 주변 분포의 경향(dark knowledge)을 보존한 완화된 타겟 맵을 생성하여 교사에서 학생으로의 부드러운 전수를 돕습니다.

### 3.4 훈련 목적 함수

훈련 단계에서 student 브랜치는 정답 데이터 손실과 teacher 브랜치로부터 증류된 신호 두 가지를 조화롭게 학습합니다. 

위에서 정의한 temperature-scaled softmax를 거쳐 student의 완화 분포 $\tilde{G}_t^{stu}$와 teacher의 완화 분포 $\tilde{G}_t^{fut}$를 계산한 뒤, 다음의 최종 손실 $\mathcal{L}$을 최소화하도록 최적화합니다.

$\mathcal{L} = \alpha \mathcal{L}_{GT} + \beta \mathcal{L}_{FPS}$

두 가지 세부 손실은 KL divergence로 산출합니다.

- $\mathcal{L}_{GT} = \sum_{t} D_{KL}(G_t \parallel \tilde{G}_t^{stu})$: 실제 ground truth와 student 출력 분포 간의 거리입니다.
- $\mathcal{L}_{FPS} = \sum_{t} D_{KL}(\text{sg}(\tilde{G}_t^{fut}) \parallel \tilde{G}_t^{stu})$: teacher의 미래 인식 출력 분포와 student의 인과적 출력 분포 간의 거리입니다. (sg는 stop-gradient 연산입니다.)

KL divergence 손실은 타겟 분포(교사의 $\text{sg}(\tilde{G}_t^{fut})$)를 고정한 채 예측 분포(학생의 $\tilde{G}_t^{stu}$)가 이를 모사하도록 유도합니다. 이때 교사 출력에 stop-gradient를 적용하여 교사 쪽 경로의 그래디언트 전파를 차단하는 것은 두 가지 핵심적인 역할을 합니다.

첫째, 디코더 파라미터의 일방향 갱신을 보장합니다. 그래디언트가 오직 student 측의 연산 그래프를 통해서만 역전파되므로, student가 teacher의 분포를 일방적으로 따라 하도록(mimic) 파라미터가 갱신됩니다. 이 과정에서 teacher가 지닌 미래 맥락 정보가 student로 강제 증류됩니다.

둘째, 공동 퇴화(co-degradation)를 원천 방지합니다. 만약 stop-gradient를 걸지 않으면, 손실을 낮추기 위해 교사가 인과 정보밖에 모르는 학생 수준으로 예측 성능을 떨어뜨려 분포의 간격을 좁히려는 잘못된 타협이 일어납니다. 즉, 교사의 우수한 예측 능력을 지켜내기 위해 교사를 고정 좌표(anchor)로 묶어두는 필수 조치입니다.

실험에서는 두 손실의 가중치를 $\alpha = 1, \beta = 1$로 유지했습니다.

---

## 4. 실험 결과

### 4.1 실험 설정

- 데이터셋: EGTEA Gaze+(24 FPS, 요리 비디오 28시간 분량, 훈련 8,299개 / 테스트 2,022개 클립) 및 Ego4D gaze subset(30 FPS, normalized 2D 좌표)을 활용합니다.
- Evaluation metric: 예측 맵과 ground truth 영역 간 중첩 정확도를 재는 Adaptive F1 Score, Precision, Recall을 적용합니다.
- 구현 세부사항: 단일 A5000 GPU 환경에서 ViT-S/16 DINOv3 백본을 동결하고 AdamW optimizer(lr $1\times10^{-4}$, weight decay 0.05, 코사인 스케줄링)로 훈련시킵니다. EGTEA Gaze+는 25 에폭, Ego4D는 15 에폭 학습을 적용합니다.

### 4.2 미래 맥락의 효용: 얼마나 도움이 되는가?

| | EGTEA Gaze+ | | | Ego4D | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| H | F1 | Rec. | Prec. | F1 | Rec. | Prec. |
| H = 0 (Baseline) | 44.7 | 60.3 | 35.5 | 40.7 | 56.3 | 31.9 |
| H = 1 | 45.1 | 59.2 | 36.4 | 41.8 | 56.1 | 33.4 |
| H = 3 | 45.5 | 61.5 | 36.1 | 41.5 | 56.8 | 32.7 |
| H = 5 | 45.9 | 61.1 | 36.7 | 41.9 | 55.7 | 33.6 |
| H = 7 | 45.6 | 60.3 | 36.6 | 42.6 | 56.9 | 34.0 |
| H = 10 | 45.9 | 63.6 | 35.9 | 42.7 | 57.6 | 34.0 |
| H = 15 | 45.4 | 61.5 | 36.1 | 41.9 | 56.6 | 33.2 |

*Table 1: look-ahead horizon $H$ 설정에 따른 causal gaze estimation 성능입니다. future-privileged supervision을 추가하면 두 데이터셋 모두 성능이 우상향하며, $H \in [5, 10]$ 수준에서 정점을 맞이한 후 극단적인 $H=15$ 수준에서는 하락세로 전환됩니다.*

실험 결과를 들여다보면 흥미로운 경향성을 확인할 수 있습니다.

- 미래 정보는 학습에 확실히 도움을 줍니다. 미래 프레임을 단서로 주었을 때 EGTEA Gaze+의 F1 점수는 44.7($H=0$)에서 45.9($H=5/10$)로 올랐고, Ego4D 역시 40.7($H=0$)에서 42.7($H=10$)로 크게 개선되었습니다.
- 미래를 길게 보여준다고 해서 마냥 더 좋아지지는 않습니다. $H=15$처럼 미래를 지나치게 길게 참조해 버리면 성능 향상 폭이 도리어 꺾였습니다.
- 최적의 미래 정보 범위는 시간으로 환산했을 때 약 1.7~3.3초였습니다. 샘플링 단위를 실시간으로 변환하면 EGTEA Gaze+의 경우 약 1.67~3.33초 범위($H \in [5, 10]$)가 가장 알맞았고, Ego4D는 약 2.67초($H=10$) 구간에서 효과가 극대화되었습니다.

이 발견은 인간의 실제 인지 패턴과도 연결됩니다. 사람은 보통 행동하기 0.5~1초(500~1000ms) 전에 시선을 먼저 움직입니다. 하지만 ECOGaze 연구 결과에 따르면 미래 지도의 혜택은 이보다 훨씬 긴 2~3초 범위까지 닿아 있었습니다. 모델이 단순히 눈앞의 반응 속도를 넘어서, 손을 뻗고 물건을 쥐는 것과 같은 2~3초 길이의 일련의 행동과 환경 변화를 전반적으로 담아냈기 때문입니다. 반대로 4~5초($H=15$)를 뛰어넘는 미래 정보는 현재 행동과 연관이 적은 잡음들이 섞이면서 오히려 학습을 망치게 됩니다.

### 4.3 기존 방법과의 비교

| | EGTEA Gaze+ | | | Ego4D | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Method | F1 | Rec. | Prec. | F1 | Rec. | Prec. |
| Center Prior | 10.7 | 32.0 | 6.4 | 14.9 | 21.9 | 11.3 |
| GBVS | 15.7 | 45.1 | 9.5 | 18.0 | 47.2 | 11.1 |
| EgoGaze | 16.3 | 16.3 | 16.3 | – | – | – |
| Gaze MLE† | 26.6 | 35.7 | 21.3 | – | – | – |
| Joint Learning† | 34.0 | 42.7 | 28.3 | – | – | – |
| I3D-R50† | 40.9 | 57.2 | 31.8 | – | – | – |
| Attention Transition | 37.2 | 51.9 | 29.0 | 36.4 | 47.5 | 29.5 |
| GLC (Causal) | 41.6 | 57.9 | 32.4 | 41.2 | 56.1 | 32.5 |
| ECOGaze (Ours) | 45.9 | 61.1 | 36.7 | 42.7 | 57.6 | 34.0 |

*Table 2: 기존 egocentric gaze estimation 방법들과의 성능 대조표입니다. ECOGaze는 양대 벤치마크 테스트에서 기존 주요 방법들을 고르게 능가했습니다.*

ECOGaze는 최적의 설정(EGTEA Gaze+는 $H=5$, Ego4D는 $H=10$)을 적용해 최고의 F1 점수를 기록했습니다. 인과적 버전으로 재구성한 GLC 모델과 비교해도 EGTEA Gaze+에서 +4.3, Ego4D에서 +1.5의 F1 향상을 이루었습니다. 본 프레임워크가 분석 도구 역할을 넘어 실제 뛰어난 성능의 인과적 예측기를 만들어냄을 보여줍니다.

### 4.4 효율성 분석

![Figure 4: 정확도-효율성 트레이드오프](/images/ecogaze/_page_6_Figure_0.jpeg)
*Figure 4: 정확도와 연산 효율성 그래프입니다. ECOGaze는 GLC 대비 연산량(GFLOPs)을 적게 쓰면서도 높은 F1 점수를 달성합니다.*

| Model | EGTEA Gaze+ | | Ego4D | |
|:---|:---:|:---:|:---:|:---:|
| | Para. (M) | FPS ↑ | Para. (M) | FPS ↑ |
| GLC (Causal) | 70.18 | 30.28 | 70.18 | 29.27 |
| ECOGaze (Ours) | 14.19 | 59.01 | 14.19 | 59.73 |

*Table 3: 모델 크기와 추론 속도 비교입니다. ECOGaze는 GLC보다 파라미터는 5배 적고 처리 속도는 2배 빠릅니다.*

ECOGaze의 파라미터 수는 14.2M입니다. 기존 GLC(70.2M)보다 약 5배가 가볍습니다. 단일 NVIDIA A5000 GPU 기준 추론 속도 역시 60 FPS에 달해 GLC(30 FPS)보다 2배 빠릅니다. 가볍고 빠른 구조 덕분에 실시간 웨어러블 장치에 탑재하기 적합합니다.

### 4.5 컴포넌트별 Ablation

| SAttn | TAttn | GLF | FPS | F1 | Rec. | Prec. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✘ | ✘ | ✘ | ✘ | 39.1 | 63.9 | 28.2 |
| ✔ | ✘ | ✘ | ✘ | 42.3 | 58.8 | 33.0 |
| ✔ | ✔ | ✘ | ✘ | 44.4 | 58.7 | 35.8 |
| ✔ | ✔ | ✔ | ✘ | 44.7 | 60.3 | 35.5 |
| ✔ | ✔ | ✔ | ✔ | 45.9 | 61.1 | 36.7 |

*Table 4: EGTEA Gaze+에서의 모듈별 Ablation 결과입니다. 각 모듈을 하나씩 추가할 때마다 성능이 점진적으로 올라갑니다.*

제안한 요소들이 결합하며 시너지를 만들어가는 흐름은 다음과 같습니다.

- 정적 특징만 사용하는 기본형(F1 39.1): Recall(63.9)은 높지만 Precision(28.2)이 낮습니다. gaze 타겟을 명확히 맞추기보다 화면에 어수선하게 흩뿌려 예측합니다.
- spatial attention(SAttn)의 결합(F1 42.3): 패치 간의 위치 연동성을 계산하여 정적 프레임 내부의 공간 정보 조율이 유효해집니다.
- temporal attention(TAttn)의 통합(F1 44.4): 시간적 움직임과 temporal dynamics 흐름이 들어서며 F1 성능이 크게 개선됩니다.
- global-local focusing(GLF) 추가(F1 44.7): global context를 토대로 gaze 예상 지점을 좀 더 섬세하게 조율합니다.
- future-privileged supervision(FPS)의 완성(F1 45.9): 마지막으로 미래 교사 신호를 이식하여 causal student가 미래 맥락의 단서까지 안정적으로 흡수하도록 보완합니다.

### 4.6 실패 사례 분석

![Figure 6: 추론 시 미래 프레임 정보가 완전히 차단된 causal 환경이기에 조우할 수밖에 없는 주요 실패 유형입니다.](/images/ecogaze/_page_7_Figure_10.jpeg)
*Figure 6: 추론 시 미래 프레임 정보가 완전히 차단된 causal 환경이기에 조우할 수밖에 없는 주요 실패 유형입니다. (1) 시선 응시가 또렷해지기 전 visual search 단계에서의 gaze diffusion 현상, (2) 돌발적인 머리 회전으로 생긴 motion blur에 따른 visual feature 손상, (3) 대안 객체가 가득한 cluttered environment에서의 target ambiguity 현상.*

추론할 때 오직 과거와 현재만 봐야 하기에 겪게 되는 근본적인 한계 시나리오는 크게 세 가지로 추려볼 수 있습니다.

- 응시 전의 산만한 탐색: 사용자가 눈을 맞추기 전 단순히 목표를 훑는 탐색 상태에서는 예측 영역이 너무 둥글고 넓게 gaze diffusion이 발생합니다.
- 급격한 머리 움직임에 따른 motion blur: 화면이 순간적으로 흐려져 DINOv3 인코더의 spatial feature 퀄리티가 깨집니다.
- 어수선한 배경 속 대상 판단의 한계: 잡다한 물건이 깔린 cluttered environment에서는 짧은 과거 history 정보만으로 사용자의 세밀한 손-눈 매칭 의도(target ambiguity)를 온전히 파악하기 역부족이었습니다.

향후 이 한계들은 모델에 더 긴 수용폭을 지닌 long-term memory 장치나 시선 외의 타 멀티모달 센서 신호를 더해 채워야 할 지점입니다.

---

## 5. 핵심 기여 정리

1. 제어된 실험 프레임워크 구축: ECOGaze는 파라미터를 공유하는 isomorphic 디코더 구조를 사용합니다. 동결된 DINOv3 scene encoder와 함께 오직 temporal attention mask만 다르게 가져갑니다. 이로써 모델 용량이나 visual representation 변화 같은 변수를 완벽히 제어한 상태에서 미래 프레임의 순수한 효과만 정밀하게 측정해냈습니다.

2. 최적 미래 참조 범위의 발견: future-privileged supervision은 causal model 성능을 항상 향상시킵니다. 그러나 그 효과는 무한하지 않으며 약 1.7~3.3초($H \in [5, 10]$) 범위 내에서 가장 높습니다. 이는 단순한 반응 시간을 넘어 2~3초 단위의 과제 수행 과정을 모델이 흡수하기 때문입니다. 4~5초 이상의 너무 먼 미래는 오히려 무관한 행동 노이즈를 섞이게 만들어 학습을 방해합니다.

3. 경량·실시간 인과적 시선 예측기: ECOGaze는 기존 GLC 모델보다 파라미터는 5배 적고(14.2M vs. 70.2M), 추론 속도는 2배 빠릅니다(약 60 FPS vs. 약 30 FPS). 그럼에도 EGTEA Gaze+(F1 45.9)와 Ego4D(F1 42.7) 벤치마크 모두에서 최고 성능을 달성하여, 실시간 AR 및 웨어러블 보조 장치에 즉시 활용할 수 있는 뛰어난 실용성을 증명했습니다.
