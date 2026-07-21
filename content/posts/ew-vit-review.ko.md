---
title: "[WACV 2025] EW-ViT: 주파수 도메인 기반 Vision Transformer 개선을 통한 강건한 의료 영상 분할"
date: 2026-07-16T20:41:00+09:00
draft: false
math: true
tags: ["Paper Review", "Medical Image Segmentation", "Vision Transformer", "Wavelet", "WACV 2025"]
categories: ["Paper Review"]
summary: "WACV 2025에 발표된 EW-ViT 논문 리뷰입니다. 웨이블릿 분해를 셀프 어텐션에 통합하고, 프롬프트 기반 고주파 정제 모듈(PGHFR)을 도입하여 열화된 의료 영상에서도 강건한 분할 성능을 달성합니다."
cover:
  image: "/images/ew_vit/ew_vit_architecture.jpeg"
  alt: "EW-ViT Architecture"
---

> <b>논문 정보</b>
> - <b>제목:</b> Frequency-Domain Refinement of Vision Transformers for Robust Medical Image Segmentation under Degradation
> - <b>저자:</b> Sanaz Karimijafarbigloo, Sina Ghorbani Kolahi, Reza Azad, Ulas Bagci, Dorit Merhof
> - <b>소속:</b> University of Regensburg, Tarbiat Modares University, Northwestern University, Fraunhofer MEVIS
> - <b>학회:</b> WACV 2025
> - <b>코드:</b> [GitHub](https://github.com/)

---

## 1. 한 줄 요약

Vision Transformer의 셀프 어텐션 블록에 <b>웨이블릿 분해</b>를 통합하여 저주파·고주파 성분을 적응적으로 정제하고, <b>프롬프트 기반 고주파 정제기(PGHFR)</b>와 <b>대조 학습</b>을 결합하여 노이즈·블러 등 다양한 열화 조건 하에서도 Synapse, ISIC2018, ACDC 데이터셋에서 SOTA 성능을 달성한 의료 영상 분할 프레임워크.

---

## 2. 연구 배경 및 동기

### 2.1. 문제 정의

의료 영상 분할(Medical Image Segmentation)은 정밀 진단, 치료 계획, 질병 모니터링의 핵심 과정이다. CNN 기반 방법(특히 U-Net)은 괄목할 성과를 거뒀지만 <b>장거리 의존성 모델링에 한계</b>가 있다. Vision Transformer(ViT)는 셀프 어텐션을 통해 전역 문맥 정보를 포착하지만, <b>지역 특징(local feature) 표현 능력이 부족</b>하다는 구조적 단점이 존재한다.

### 2.2. 기존 방법의 한계

ViT의 지역 특징 부족을 보완하기 위해 CNN-Transformer 하이브리드 아키텍처(TransUNet, HiFormer, CoTr 등)가 제안되었으나, 다음과 같은 한계를 지닌다:

- <b>높은 파라미터 수와 연산 비용</b>: TransUNet은 96M 파라미터와 88.91G FLOPs로 비효율적이다.
- <b>CNN 백본 의존성</b>: HiFormer, CoTr 등은 여전히 CNN 백본에 크게 의존하며, 다중 스케일 정보의 통합이 불충분하다.
- <b>고주파 성분 포착 실패</b>: 연구에 따르면 셀프 어텐션은 <b>저역 통과 필터(low-pass filter)</b> 역할을 하여 고주파 성분(경계, 세밀한 텍스처 등)을 충분히 포착하지 못한다.
- <b>열화 조건 미고려</b>: 기존 주파수 기반 방법(Laplacian-Former, WaveFormer)도 고주파 정보를 유지하지만, 의료 영상에 흔한 <b>노이즈·블러·헤이즈</b> 등의 열화가 고주파 성분을 오염시키는 상황을 처리하지 못한다.

### 2.3. 핵심 기여

1. <b>웨이블릿 기반 주파수 도메인 모델링</b>: 셀프 어텐션 내에서 웨이블릿 분해를 수행하여 공간 차원을 절반으로 줄이면서 다중 스케일 표현을 동시에 포착하고, 연산 복잡도를 감소시킨다.
2. <b>PGHFR 모듈 도입</b>: 학습 가능한 프롬프트를 활용해 열화 관련 정보를 암묵적으로 인코딩하고, 이를 기반으로 고주파 성분을 동적으로 정제하여 노이즈 효과를 제거한다.
3. <b>대조 학습 기반 특징 일관성</b>: 원본과 열화-증강 특징 간의 대조 학습을 통해 표현 공간의 일관성을 유지하고, PGHFR 모듈에 간접적 가이던스를 제공한다.
4. <b>열화 강건성 검증</b>: 가우시안 노이즈, 블러, 색상 지터링 등 다양한 열화 유형에 대한 포괄적 어블레이션 스터디를 통해 모델의 강건성을 입증한다.

---

## 3. 제안 방법: EW-ViT 프레임워크

### 3.1. 전체 아키텍처

![EW-ViT 전체 아키텍처](/images/ew_vit/ew_vit_architecture.jpeg)

*Figure 1: EW-ViT의 전체 구조. U자형 인코더-디코더 구성이며, 인코더에서 EW-ViT 블록과 패치 임베딩을 반복하고, 디코더에서는 PGHFR이 통합된 EW-ViT 블록과 패치 확장을 사용한다. Contrastive Head가 특징 일관성 학습을 담당한다.*

EW-ViT는 U자형 아키텍처로 구성되며, <b>인코더</b>, <b>디코더</b>, 그리고 두 개의 헤드(<b>Contrastive Head</b>, <b>Segmentation Head</b>)로 이루어진다.

- <b>인코더</b>: 3개 레이어로 구성되며, 각 레이어는 EW-ViT 블록 뒤에 패치 임베딩(다운샘플링 + 차원 확장)이 따른다.
- <b>디코더</b>: 3개 레이어로 구성되며, EW-ViT 블록(PGHFR 포함) 뒤에 패치 확장(업샘플링)이 따른다.
- <b>Contrastive Head</b>: 인코더 병목(bottleneck) 레이어의 출력으로 대조 학습을 수행한다.

### 3.2. Enhanced Wave ViT (EW-ViT) 블록

![Enhanced Wave Attention Block 상세 구조](/images/ew_vit/ew_vit_block.jpeg)

*Figure 2: EW-ViT 블록의 상세 구조. DWT로 4개 서브밴드를 분리하고, PGHFR이 고주파 성분을 정제한다. CPG와 IPI 모듈의 내부 동작을 함께 나타낸다.*

EW-ViT 블록의 동작 과정은 다음과 같다:

<b>① 채널 축소 및 웨이블릿 분해</b>

2D 특징 맵 $X \in \mathbb{R}^{H \times W \times D}$에 선형 변환을 적용하여 채널을 1/4로 축소한 뒤 ($\widetilde{X} = XW_d$), <b>이산 웨이블릿 변환(DWT)</b>을 적용한다. 고전적인 Haar 웨이블릿을 사용하여 저역/고역 필터를 행·열 방향으로 순차 적용하면, 4개의 서브밴드가 생성된다:

- $X_{LL}$: 저주파 성분 — 객체의 기본 구조를 조대(coarse) 수준에서 포착
- $X_{LH}, X_{HL}, X_{HH}$: 고주파 성분 — 세밀한 텍스처, 경계 정보를 보존

각 서브밴드는 공간 해상도가 원본의 절반($H/2 \times W/2$)이므로, 이후 셀프 어텐션의 <b>연산량이 자연스럽게 감소</b>한다.

<b>② PGHFR을 통한 고주파 정제</b>

고주파 서브밴드 3개를 채널 방향으로 결합한 $X_H = [X_{LH}, X_{HL}, X_{HH}]$에 PGHFR 모듈을 적용하여 열화 영향을 제거한 $X'_H$를 얻는다 (자세한 구조는 3.3절 참조).

<b>③ 웨이블릿-어텐션 수행</b>

정제된 고주파 성분 $X'_H$와 저주파 성분 $X_L$을 결합한 뒤 $3 \times 3$ 컨볼루션으로 지역성을 강화하여 다운샘플링된 특징 맵 $X^d$를 생성한다. 이 $X^d$를 키($K$)와 값($V$)으로 변환하고, 원본 특징 맵에서 추출한 쿼리($Q$)와 멀티헤드 셀프 어텐션을 수행한다:

$$head_j = \text{Softmax}\left(\frac{Q_j K_j^T}{\sqrt{D_h}}\right) V_j$$

<b>④ IDWT 경로와 최종 출력 합성</b>

별도로 $[X_L, X'_H]$에 <b>역이산 웨이블릿 변환(IDWT)</b>을 적용하여 원래 해상도의 정제된 특징을 복원한다. 이 IDWT 경로의 출력과 멀티헤드 어텐션 출력을 결합하여 최종 EW-ViT 블록의 출력을 구성한다:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, \ldots, \text{head}_{N_h}) \widetilde{W}^O$$

### 3.3. Prompt-Guided High-Frequency Refiner (PGHFR)

PGHFR은 고주파 성분에서 열화(노이즈, 블러)의 영향을 제거하는 핵심 모듈이며, <b>Component Prompt Generation (CPG)</b>과 <b>Implicit Prompt Interaction (IPI)</b> 두 서브 모듈로 구성된다.

$$X'_H = \text{IPI}(\text{CPG}(P_c, X_H), X_H)$$

<b>Component Prompt Generation (CPG)</b>

학습 가능한 프롬프트 컴포넌트 $P_c \in \mathbb{R}^{N \times \frac{H}{2} \times \frac{W}{2} \times D'}$를 입력 고주파 특징 $X_H$의 맥락에 맞게 동적으로 조합한다.

1. $X_H$에 <b>Global Average Pooling(GAP)</b>과 <b>Global Max Pooling(GMP)</b>을 적용하여 공간 정보를 압축한다.
2. 두 풀링 결과를 합산하고 <b>Point-Wise Convolution(PWC)</b> + Softmax를 거쳐 프롬프트 가중치 $w \in \mathbb{R}^{D'}$를 생성한다.
3. 가중치로 프롬프트 컴포넌트를 가중합한 뒤 $3 \times 3$ 컨볼루션으로 공간 관계를 정제하여 최종 프롬프트 $P$를 생성한다.

$$w_i = \text{Softmax}(\text{PWC}(\text{GAP}(X_H) + \text{GMP}(X_H)))$$
$$P = \text{Conv}_{3 \times 3}\left(\sum_{c=1}^{D'} w_i P_c\right)$$

<b>Implicit Prompt Interaction (IPI)</b>

생성된 프롬프트 $P$와 고주파 입력 $X_H$를 채널 방향으로 결합한 뒤, 어텐션 블록을 통해 상호작용시킨다. 이 과정에서 프롬프트가 열화 관련 정보를 전달하여 고주파 성분의 노이즈를 선택적으로 억제한다:

$$X'_H = \text{Conv}_{3 \times 3}(\text{Attention}(X_H, P))$$

### 3.4. 대조 학습 기반 특징 일관성

인코더 병목 레이어에서 고주파 성분에 <b>랜덤 열화 노이즈</b> $G(X_H)$를 주입한 뒤 IDWT로 복원하여 증강 특징 $X^{aug}$를 생성한다. 원본 특징과 증강 특징에서 동일 공간 위치의 픽셀을 샘플링하여 대조 학습 후보로 사용한다.

클래스 $k$의 프로토타입은 해당 클래스에 속하는 픽셀 특징의 평균으로 계산된다:

$$c_k = \frac{1}{|S_k|} \sum_{k \in S_k} X_k$$

증강 후보 픽셀을 원본의 클래스 프로토타입에 정렬시키는 대조 손실을 다음과 같이 정의한다:

$$\mathcal{L}_{c_k} = -\frac{1}{|S_k|} \sum_{x_k^{aug} \in S_k} \log \frac{\exp(\text{sim}(x_k^{aug}, c_k) / \tau)}{\sum_{j \neq k} \exp(\text{sim}(x_k^{aug}, c_j) / \tau)}$$

여기서 $\text{sim}(\cdot, \cdot)$은 코사인 유사도, $\tau$는 온도 파라미터다. 전체 대조 손실은 모든 클래스에 대해 합산하며, 클래스 프로토타입 간 분리를 강화하는 정규화 항 $L_{sep} = \sum_{i \neq j} \frac{1}{\|c_i - c_j\|_2}$도 추가된다.

최종 학습 목적 함수는 다음과 같다:

$$Loss = \lambda_s \mathcal{L}_{Seg} + \lambda_c \mathcal{L}_c$$

---

## 4. 실험 결과

### 4.1. 데이터셋 및 구현 세부사항

| 데이터셋 | 유형 | 규모 | 분할 대상 |
|---------|------|------|----------|
| <b>Synapse</b> | 복부 CT | 30 케이스, 3,779 슬라이스 | 8개 장기 (비장, 신장 등) |
| <b>ISIC2018</b> | 피부경 영상 | — | 피부 병변 |
| <b>ACDC</b> | 심장 MRI | 100 스캔 (train 70/val 10/test 20) | RV, Myo, LV |

- <b>옵티마이저</b>: Synapse — SGD (lr=0.05, momentum=0.9), ISIC/ACDC — Adam (lr=0.0001)
- <b>손실 함수</b>: Synapse/ISIC — BDOU + 대조 손실, ACDC — CE + DICE + 대조 손실
- <b>GPU</b>: RTX 3090 1장

### 4.2. Synapse 데이터셋 결과

| 방법 | Params (M) | FLOPs (G) | Avg DSC↑ | HD95↓ |
|------|-----------|-----------|----------|-------|
| TransUNet | 96.07 | 88.91 | 77.49 | 31.69 |
| Swin-UNet | 27.17 | 6.16 | 79.13 | 21.55 |
| MISSFormer | 42.46 | 9.89 | 81.96 | 18.20 |
| HiFormer-B | 25.51 | 8.05 | 80.39 | 14.70 |
| Laplacian-Former | 27.54 | 6.68 | 81.90 | 18.86 |
| WaveFormer | 47.01 | 7.75 | 81.92 | 18.41 |
| VM-UNet | 44.27 | 6.52 | 81.08 | 19.21 |
| <b>EW-ViT</b> | <b>33.35</b> | <b>6.03</b> | <b>83.51</b> | <b>16.68</b> |

EW-ViT는 33.35M 파라미터, 6.03G FLOPs로 TransUNet 대비 <b>파라미터 약 1/3, FLOPs 약 1/15</b> 수준이면서도 DSC를 <b>6.02%p</b> 향상시켰다. 기존 주파수 기반 방법인 Laplacian-Former(81.90%)와 WaveFormer(81.92%) 대비 각각 <b>1.61%p, 1.59%p</b> 우위를 보인다.

![Synapse 데이터셋 시각적 비교](/images/ew_vit/synapse_visual.jpeg)

*Figure 3: Synapse 데이터셋에서 EW-ViT, WaveFormer, MissFormer의 분할 결과 비교. EW-ViT는 좌측 신장(1행)과 위(2행)의 경계를 가장 정밀하게 분할한다.*

### 4.3. ISIC2018 데이터셋 결과

| 방법 | DSC | SE | SP | ACC |
|------|-----|----|----|-----|
| Swin-UNet | 0.8946 | 0.9056 | 0.9798 | 0.9645 |
| VM-UNet | 0.8971 | 0.9112 | 0.9613 | 0.9491 |
| Laplacian-Former | 0.9128 | 0.9290 | 0.9626 | 0.9640 |
| <b>EW-ViT</b> | <b>0.9164</b> | <b>0.9297</b> | 0.9571 | <b>0.9643</b> |

EW-ViT는 DSC 0.9164, 민감도(SE) 0.9297로 전체 최고 성능을 기록했다. 특이도(SP)는 0.9571로 다소 낮은데, 이는 고주파 기반 특징 강화가 병변 경계를 적극적으로 검출하면서 약간의 과분할(over-segmentation)을 유발할 수 있음을 시사한다.

### 4.4. ACDC 데이터셋 결과

| 방법 | Avg DICE | RV | Myo | LV |
|------|----------|-----|------|-----|
| TransUNet | 89.71 | 88.86 | 84.53 | 95.73 |
| Swin-UNet | 90.00 | 88.55 | 85.62 | 95.83 |
| MT-UNet | 90.43 | 86.64 | 89.04 | 95.62 |
| <b>EW-ViT</b> | <b>90.29</b> | 88.05 | <b>87.56</b> | 95.25 |

ACDC에서 EW-ViT는 Avg DICE 90.29%로 TransUNet, Swin-UNet을 각각 0.58%p, 0.29%p 능가했다. 특히 심근(Myo) 분할에서 87.56%를 기록하며 Swin-UNet(85.62%) 대비 1.94%p 우수한 성능을 보였다. MT-UNet이 전체 평균에서 0.14%p 높지만, 열화 조건에서의 강건성은 EW-ViT가 크게 앞선다.

### 4.5. 열화 강건성 분석

| 방법 | 가우시안 노이즈 ($\sigma$=0.05) | 가우시안 블러 ($\sigma$=1.5) | 색상 지터링 ($\gamma$=1.75) |
|------|:---:|:---:|:---:|
| | DSC / $\Delta$↓ | DSC / $\Delta$↓ | DSC / $\Delta$↓ |
| Swin-UNet | 88.20 / 1.80 | 85.49 / 4.51 | 82.54 / 7.46 |
| MT-UNet | 88.12 / 2.31 | 81.53 / 8.90 | 75.53 / 14.90 |
| <b>EW-ViT</b> | <b>89.52 / 0.77</b> | <b>86.68 / 3.61</b> | <b>83.52 / 6.77</b> |

ACDC 데이터셋 기준, EW-ViT는 모든 열화 유형과 강도에서 <b>가장 적은 성능 하락($\Delta$)</b>을 보였다. 특히 색상 지터링 $\gamma$=1.75 조건에서 MT-UNet은 14.90%p 하락한 반면, EW-ViT는 6.77%p 하락에 그쳤다.

Synapse 데이터셋에서도 높은 노이즈($\sigma$=0.25) 하에서 EW-ViT는 DSC 79.70% ($\Delta$=3.81)을 유지한 반면, TransUNet은 67.87% ($\Delta$=9.62), WaveFormer는 74.41% ($\Delta$=7.51)으로 큰 폭의 성능 저하를 보였다.

![노이즈 조건에서의 어텐션 맵 시각화](/images/ew_vit/attention_map_noise.jpeg)

*Figure 4: 노이즈가 없는 조건($\sigma$=0.0)과 높은 노이즈($\sigma$=0.3) 조건에서 EW-ViT의 GradCAM 어텐션 맵 비교. 높은 노이즈에서도 관심 객체에 정확히 집중함을 확인할 수 있다.*

---

## 5. 핵심 기여 정리

<b>① 웨이블릿 분해 기반 주파수 도메인 셀프 어텐션 재정의</b>

기존 ViT의 셀프 어텐션이 저역 통과 필터 역할을 하여 고주파 성분을 손실하는 문제를 해결하기 위해, DWT를 셀프 어텐션 내부에 통합했다. 이를 통해 (1) 저주파(구조)와 고주파(경계/텍스처) 성분을 명시적으로 분리하여 각각에 적합한 처리 경로를 제공하고, (2) 서브밴드의 공간 해상도가 원본의 절반이므로 셀프 어텐션의 Key·Value 차원이 자연스럽게 축소되어 연산 복잡도가 감소한다.

<b>② PGHFR: 프롬프트 기반 적응적 고주파 정제</b>

의료 영상에서 빈번히 발생하는 노이즈, 블러, 헤이즈 등이 고주파 성분을 오염시키는 문제를 학습 가능한 프롬프트로 해결했다. CPG 모듈이 입력 특징의 맥락을 반영하여 프롬프트를 동적으로 생성하고, IPI 모듈이 이 프롬프트와 고주파 특징을 어텐션 기반으로 상호작용시켜 열화를 선택적으로 제거한다. 이 설계 덕분에 열화 유형과 강도에 대한 사전 지식 없이도 범용적으로 동작한다.

<b>③ 대조 학습을 통한 열화 불변 표현 학습</b>

인코더 병목에서 고주파 성분에 랜덤 열화를 주입한 증강 특징과 원본 특징 간의 대조 학습을 수행하여, 모델이 열화 여부와 무관하게 일관된 표현을 학습하도록 유도한다. 이 전략은 PGHFR 모듈에 간접적 학습 신호를 제공하는 동시에, 클래스 프로토타입 분리항을 통해 클래스 간 판별력도 강화한다.

<b>④ 포괄적 열화 강건성 검증</b>

가우시안 노이즈(3단계), 가우시안 블러(3단계), 색상 지터링(3단계) 총 9가지 열화 조건에 대한 광범위한 어블레이션을 수행했다. 모든 조건에서 EW-ViT가 기존 방법들 대비 가장 적은 성능 하락을 보이며, 특히 극심한 열화($\sigma$=0.25 노이즈)에서도 WaveFormer 대비 5.29%p 높은 DSC를 유지하여 실제 임상 환경에서의 실용성을 입증했다.
