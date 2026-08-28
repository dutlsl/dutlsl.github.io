---
title: "[GAZE 2026] Learning to Look: CLIP 기반 Dual-Crop Fusion으로 Head Position에 강건한 Gaze Estimation"
date: 2026-08-28T15:10:14+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "CLIP", "Fusion", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Mercedes-Benz R&D가 CVPR 2026 GAZE Workshop에서 발표한 논문을 리뷰합니다. 눈 영역과 얼굴 전체에서 각각 추출한 gaze 예측을 CLIP이 안내하는 fusion 모듈로 적응적으로 결합하고, head position embedding과 learnable transformation을 도입하여 머리 위치 변화에 강건한 gaze estimation을 달성합니다."
cover:
  image: "/images/clip-dual-crop-gaze/fig2_architecture.jpeg"
  alt: "Learning to Look Architecture Overview"
---

> 논문 정보
> - 제목: Learning to Look: CLIP-Guided Dual-Crop Fusion for Head Position-Invariant Gaze Estimation
> - 저자: Sourav Lakhotia, Chaviti Vasantha Lakshmi, Aratrik Chattopadhyay
> - 소속: Mercedes-Benz Research and Development, Karnataka, India
> - 학회: The 7th International Workshop on Eye and Gaze in Computer Vision (GAZE 2026) at CVPR 2026

![Learning to Look 아키텍처 개요도](/images/clip-dual-crop-gaze/fig2_architecture.jpeg)

---

## 1. 한 줄 요약

눈 crop과 얼굴 crop을 독립적으로 처리하는 dual-stream 구조에, Fourier 기반 head position embedding과 learnable coordinate transformation을 결합하고, CLIP의 의미론적 시각-언어 표현으로 두 예측을 적응적으로 융합하여, 세 가지 대표 벤치마크(IVGaze, GazeGene, MPIIFaceGaze)에서 기존 최고 baseline 대비 6.3~39.3%의 error 감소를 달성한 논문입니다.

---

## 2. 연구 배경 및 동기

### 2.1 문제 정의

Appearance-based gaze estimation은 안면 또는 안구 이미지로부터 3차원 시선 벡터를 직접 회귀하는 컴퓨터 비전 분야의 핵심 과제입니다. 차량 내 운전자 주의 상태 모니터링, 시선 기반 인간-차량 인터페이스(HCI), 운전자 행동 분석 등 다양한 실시간 시스템에서 중요한 역할을 담당합니다.

그러나 실제 차량 내부 환경은 시선 추정 모델에게 매우 혹독한 조건을 부여합니다:

첫째, 시간대와 차량 주행 방향에 따라 변화하는 극단적인 조명 변화와 직사광선, 깊은 그림자가 발생합니다.
둘째, 선글라스, 도수 안경의 강한 반사광, 마스크 착용 등으로 인해 눈이나 하안부가 가려지는 심각한 occlusion이 빈번하게 일어납니다.
셋째, 운전자의 자유로운 고개 돌림과 전후좌우 위치 이동으로 인해 카메라와 머리 사이의 3차원 기하학적 배치가 끊임없이 변화합니다.

![Figure 1](/images/clip-dual-crop-gaze/fig1_qualitative_teaser.jpeg)

*Figure 1: 안경 착용, 마스크 가림 및 극단적 시선(Lizard Gaze) 등 다양한 극한 시나리오에서의 정성적 시선 추정 결과입니다. 초록 화살표는 ground truth, 파란 화살표는 제안 방법(Ours), 빨간 화살표는 기존 SOTA baseline입니다. 각 이미지 하단에는 yaw 및 pitch 각도 오차가 도(°) 단위로 표시되어 있습니다.*

과거의 제어된 환경 벤치마크(MPIIFaceGaze, GazeCapture)는 상대적으로 고정된 머리 자세와 정돈된 조명 아래 수집되어 이러한 차량 내 극한 환경의 복잡성을 반영하지 못했습니다. 최근 제안된 IVGaze 데이터셋이 적외선(IR) 카메라를 활용하여 실제 주행 조건의 44,705장 데이터를 구축하면서 실환경 시선 추정 연구가 본격적으로 다루어지기 시작했습니다.

### 2.2 기존 방법의 한계 및 관련 연구

기존 appearance-based gaze estimation 접근법들은 입력 영역의 트레이드오프와 좌표계 정규화 방식 측면에서 명확한 한계를 지니고 있었습니다.

#### 1. 입력 영역의 트레이드오프와 Multi-mapping 딜레마

기존 방법론은 모델 입력 구성에 따라 크게 두 가지 갈래로 나뉩니다:

- 얼굴 전체 입력(Full-face) 방식: 얼굴 전체의 거시적 윤곽과 머리 회전 자세를 포착하는 데 유리하지만, 고해상도 안면 영역 안에서 미세한 동공 및 홍채의 회전 단서가 희석되어 정밀한 시선 각도 추정에 한계를 보입니다.
- 눈 영역 크롭(Eye-crop) 방식: 안구 주변의 미세한 국소적 ocular cue를 집중적으로 분석할 수 있지만, 머리와 카메라 사이의 3차원 상대 위치 및 전역적 얼굴 맥락이 완전히 제거됩니다.

눈 크롭 방식에서 발생하는 가장 치명적인 문제는 Multi-mapping(다대일 및 일대다 매핑) 현상입니다. 시선 방향은 물리적으로 안구의 회전과 머리의 3차원 자세가 결합되어 결정됩니다. 동일한 절대 시선 방향을 바라보더라도 운전자의 머리 자세(Head Pose)와 위치가 바뀌면, 2차원 카메라 평면에 투영되는 눈 이미지의 외형은 완전히 다른 pixel 패턴으로 나타납니다.

![Figure 4](/images/clip-dual-crop-gaze/fig4_multi_mapping.jpeg)

*Figure 4: IVGaze 데이터셋에서 머리 자세 변화로 인해 동일한 시선 방향임에도 눈의 이미지 외형이 달라지는 multi-mapping 문제 예시입니다. 머리의 3차원 회전에 따라 안구 중심과 홍채의 상대적 위치가 크게 변형되는 현상을 보여줍니다.*

반대로 동일한 눈 외형 이미지라도 머리가 향한 방향에 따라 전혀 다른 절대 시선 방향을 가리킬 수 있습니다. 이러한 다대일 매핑 관계는 단순 회귀 모델의 학습을 극도로 불안정하게 만듭니다.

#### 2. 전통적 Gaze Normalization의 취약점

이러한 Multi-mapping을 제거하기 위해 전통적으로 널리 쓰여온 기법이 3차원 기하학 기반의 Gaze Normalization입니다.

대표적으로 Zhang 등이 제안한 MPIIGaze 정규화 방식과 IVGaze의 축 정규화(Axis Normalization) 방식이 있습니다. 이 방식들은 안면 랜드마크나 별도의 헤드 포즈 추정기를 통해 머리의 3차원 위치(Head Position, HP)와 회전(Head Rotation)을 추정한 뒤, 눈의 중심을 바라보는 가상의 정규화 카메라 좌표계(Normalized Camera Space)를 수학적으로 정의하여 입력 이미지를 투영 변환(Perspective Warping)합니다.

그러나 이러한 전통적 정규화 파이프라인은 다음과 같은 구조적 결함을 가집니다:

- 머리 위치(HP) 추정 오차에 대한 극단적 민감도: 외부 헤드 포즈 추정기나 랜드마크 검출기가 머리 위치를 수 센티미터만 잘못 추정해도, 가상 카메라 행렬 계산에 심각한 오차가 증폭되어 입력 이미지가 비정상적으로 왜곡됩니다.
- 오차 전파(Error Propagation): 머리 위치가 카메라 중심에서 10cm만 이동하거나 추정 오차가 누적되어도 시선 각도 오차(Angular Error)가 7° 이상으로 급증하는 치명적인 성능 저하가 발생합니다.

### 2.3 핵심 기여

본 논문은 이러한 Multi-mapping 문제와 기존 정규화의 오차 민감성을 해결하기 위해 다음과 같은 4가지 핵심 기여를 제시합니다:

- 눈 크롭과 얼굴 크롭을 독립적으로 처리하는 Dual-stream 아키텍처($\Phi_{\text{eye}}, \Phi_{\text{face}}$)를 구축하여, 미세한 국소 안구 단서와 전역적 안면 맥락을 동시에 보존합니다.
- 복잡하고 불안정한 3차원 가상 카메라 캘리브레이션 대신, 핀홀 카메라 사영 기하학을 경량 MLP로 파라미터화한 Learnable Coordinate Transformation $h(\cdot)$를 제안하여 머리 위치 변화에 강건한 좌표계 변환을 달성합니다.
- 타이트한 크롭 과정에서 손실되는 3차원 공간 맥락을 복원하기 위해, 3차원 머리 위치(HP) 벡터를 고차원 주파수 공간으로 매핑하는 Fourier 기반 Head Position Embedding을 도입합니다.
- 상황에 따라 변하는 눈과 얼굴의 가시성을 판단하기 위해, 사전학습된 CLIP의 시각-언어 의미 공간을 활용하여 두 브랜치의 예측을 동적으로 가중 결합하는 CLIP-Guided Fusion 모듈을 제안합니다.

---

## 3. 제안 프레임워크

![Figure 2](/images/clip-dual-crop-gaze/fig2_architecture.jpeg)

*Figure 2: 제안 방법의 전체 아키텍처입니다. 입력 안면 이미지 I로부터 HRNet을 통해 좌우 눈 크롭($I_c^L, I_c^R$)과 얼굴 크롭($I_f$)을 분리합니다. $\Phi_{\text{eye}}$(4-stack Hourglass)와 $\Phi_{\text{face}}$(6-stack Hourglass)는 각각 Fourier HP 임베딩과 결합하여 Crop 좌표계($\mathcal{C}_{\text{crop}}$)에서 시선을 회귀한 뒤, 바운딩 박스 기반 변환 모듈 $h(\cdot)$를 통해 Image 좌표계($\mathcal{C}_{\text{img}}$)로 정렬합니다. 마지막으로 CLIP의 시각-언어 의미 특징 $F_{\text{CLIP}}$을 입력받는 $\Phi_{\text{fuse}}$가 두 예측을 적응적으로 융합합니다.*

제안 프레임워크는 입력 안면 이미지 $I \in \mathbb{R}^{H \times W}$가 주어졌을 때, 사전학습된 HRNet 백본을 통해 2차원 안면 랜드마크를 추출하는 것으로 시작합니다. 추출된 랜드마크를 기준으로 좌우 눈 크롭 영역 $I_c^L, I_c^R \in \mathbb{R}^{H_c \times W_c}$과 얼굴 전체 크롭 영역 $I_f \in \mathbb{R}^{H_f \times W_f}$ (여기서 $H_f=96, W_f=224$)를 분리합니다.

전체 시스템은 (1) 좌표계 변환 모듈 $h(\cdot)$, (2) 눈 브랜치 $\Phi_{\text{eye}}$, (3) 얼굴 브랜치 $\Phi_{\text{face}}$, (4) CLIP-Guided Fusion 모듈 $\Phi_{\text{fuse}}$의 네 가지 핵심 구성 요소로 이루어집니다.

### 3.1 Learnable Coordinate Transformation $h(\cdot)$의 수식 기호 완벽 해체 및 3D 기하학 해설

시선 추정에서 머리가 움직일 때 발생하는 시각적 혼란(Multi-mapping)을 해결하는 핵심 아이디어는 시선을 두 개의 서로 다른 공간에서 나누어 다루는 것입니다:

1. 안구만을 클로즈업한 작은 Crop 공간($\mathcal{C}_{\text{crop}}$)에서 시선 회귀를 먼저 수행합니다. 이 작은 공간에서는 머리가 어디로 돌아가든 눈동자의 중심 위치와 회전 방향이 항상 1대1로 일정하게 대응되므로 모델이 안구 회전 특징을 매우 안정적으로 학습할 수 있습니다.
2. 이후 Crop 공간에서 도출된 시선 예측값을 실제 차량 전체 화면 기준인 Image 공간($\mathcal{C}_{\text{img}}$)으로 되돌려놓아야만 실제 전방 도로 상의 어디를 보는지 손실 함수를 계산하고 최종 결과를 평가할 수 있습니다.

이 두 공간 사이의 징검다리 역할을 해주는 것이 바로 좌표 변환 모듈 $h(\cdot)$입니다.

![Figure 3](/images/clip-dual-crop-gaze/fig3_transformation.jpeg)

*Figure 3: Pinhole 카메라 모델에 기반한 Crop 좌표계($\mathcal{C}_{\text{crop}}$)와 이미지 좌표계($\mathcal{C}_{\text{img}}$) 사이의 3차원-2차원 기하학적 투영 및 평행이동 행렬 $M_T$ 변환 관계입니다.*

#### 1. 3D 점을 2D 사진 픽셀로 변환하는 기본 핀홀 수식 해체

논문의 첫 번째 핵심 수식은 다음과 같습니다:

$$[p_{\text{2D}}, 1]^T = K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] [P_{\text{3D}}, 1]^T$$

이 수식에 쓰인 기호들을 3차원 공간의 물리적 의미와 1:1로 매칭하여 해체하면 다음과 같습니다:

- $P_{\text{3D}} = (X, Y, Z)$: 실제 3차원 공간 속에서 안구가 위치한 실제 물리적 3D 위치 좌표입니다. 가로($X$), 세로($Y$), 카메라로부터의 거리($Z$)를 나타내는 3개의 숫자입니다.
- $[P_{\text{3D}}, 1]^T$에서 1과 $T$의 의미:
  수학에서 3차원 회전(곱셈)과 이동(덧셈)을 따로따로 쓰면 수식이 복잡해집니다. 따라서 $(X, Y, Z)$ 맨 아래에 숫자 1을 덧붙여 4칸짜리 세로 기둥 벡터 $[X, Y, Z, 1]^T$로 만듭니다. 여기서 윗첨자 $T$(Transpose, 전치)는 가로로 적힌 숫자를 행렬 곱셈을 위해 세로 기둥 형태로 세워놓았다는 단순한 표기 기호입니다. 이렇게 숫자 1을 덧붙인 4칸 벡터를 동차 좌표(Homogeneous Coordinates)라고 부르며, 회전과 이동을 행렬 곱셈 단 한 번으로 계산할 수 있게 해줍니다.
- $[R_{\text{img}} \mid T_{\text{img}}]$에서 세로선($\mid$)의 의미:
  가운데 세로선($\mid$)은 두 행렬을 나란히 옆으로 이어 붙였다는 뜻(블록 행렬)입니다.
  왼쪽의 $R_{\text{img}}$는 $3 \times 3$ 크기의 3차원 회전 행렬(Rotation)로, 운전자의 머리나 시선이 위아래/양옆으로 몇 도 회전했는지를 나타내는 9개의 숫자입니다.
  오른쪽의 $T_{\text{img}}$는 $3 \times 1$ 크기의 3차원 평행이동 벡터(Translation)로, 머리가 카메라 렌즈 기준으로 앞뒤/좌우/상하로 몇 cm 떨어져 있는지(3D 머리 위치)를 나타내는 3개의 숫자입니다.
  이 둘을 좌우로 붙인 $3 \times 4$ 행렬에 4칸짜리 3D 점 $[P_{\text{3D}}, 1]^T$를 곱하면, 3차원 세계의 점이 카메라 렌즈 기준의 3D 좌표로 이동하고 회전합니다.
- $K_{\text{img}}$의 물리적 의미:
  $3 \times 3$ 크기의 $K_{\text{img}}$는 카메라 렌즈와 센서가 3D 공간의 빛을 2D 평면 픽셀로 담아내는 카메라 렌즈 스펙(내부 파라미터)입니다:
  $$K_{\text{img}} = \begin{bmatrix} f & 0 & W/2 \\ 0 & f & H/2 \\ 0 & 0 & 1 \end{bmatrix}$$
  여기서 $f$는 카메라 렌즈의 초점거리(Focal Length)로, 3D 사물이 사진에 얼마나 크게 찍히는지를 결정하는 배율입니다.
  $(W/2, H/2)$는 렌즈의 중심축이 사진 가로 폭 $W$의 절반, 세로 높이 $H$의 절반인 정중앙 픽셀에 정확히 꽂힌다는 뜻입니다.
- $p_{\text{2D}} = (u, v)$:
  최종적으로 2D 디지털 사진 화면에서 맺힌 픽셀의 가로 위치 $u$와 세로 위치 $v$입니다. 마찬가지로 계산 편의를 위해 끝에 1을 붙여 세로로 세운 형태가 $[p_{\text{2D}}, 1]^T = [u, v, 1]^T$입니다.

즉, 이 수식 전체의 직관적 의미는: 3차원 공간 속에서 특정 각도로 회전($R_{\text{img}}$)하고 특정 위치에 놓인($T_{\text{img}}$) 안구의 3D 점 $P_{\text{3D}}$가 카메라 렌즈($K_{\text{img}}$)를 통과하여 전체 사진의 $(u, v)$ 픽셀로 찰칵 찍히는 과정을 하나의 행렬곱으로 압축한 것입니다.

#### 2. 크롭 공간의 가상 핀홀 투영 수식 해체

$$[p'_{\text{2D}}, 1]^T = K_c [R_c \mid T_c] [P_{\text{3D}}, 1]^T$$

얼굴 전체 사진에서 눈 영역만을 네모나게 잘라낸(Crop) 작은 이미지를 상상해 보면, 이 작은 이미지는 마치 눈동자 바로 코앞에 전용 미니 카메라를 새로 두고 찍은 것과 기하학적으로 완벽히 동일합니다.

- $p'_{\text{2D}} = (u', v')$: 크롭된 작은 이미지 안에서의 2D 픽셀 위치입니다.
- $R_c$: 크롭된 작은 이미지 안에서 관측되는 순수한 안구 회전 각도(Crop space gaze)입니다.
- $T_c$: 크롭 가상 카메라 기준의 머리 위치입니다.
- $K_c$: 크롭 이미지 전용 가상 렌즈 스펙입니다:
  $$K_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  여기서 $(f_x, f_y)$는 눈만 클로즈업하면서 확대된 가상 렌즈의 초점거리 배율이고, $(c_x, c_y)$는 크롭 이미지 내부에서의 중심점 좌표입니다.

#### 3. 바운딩 박스 이동 행렬 $M_T$와 회전 변환식의 도출

전체 사진($\mathcal{C}_{\text{img}}$)의 픽셀 좌표 $(u, v)$와 잘라낸 작은 사진($\mathcal{C}_{\text{crop}}$)의 픽셀 좌표 $(u', v')$ 사이의 관계는 아주 단순합니다. 전체 사진의 왼쪽 위 끝점 $(0, 0)$에서 눈 네모 상자(바운딩 박스)가 시작하는 왼쪽 위 좌표 $(x_s, y_s)$만큼 좌표를 빼준 것에 불과합니다:

$$u' = u - x_s, \quad v' = v - y_s$$

이 단순한 픽셀 빼기 연산($-x_s, -y_s$)을 행렬 곱셈 형태로 포장한 것이 바로 평행이동 행렬 $M_T$입니다:

$$[p'_{\text{2D}}, 1]^T = \begin{bmatrix} 1 & 0 & -x_s \\ 0 & 1 & -y_s \\ 0 & 0 & 1 \end{bmatrix} [p_{\text{2D}}, 1]^T = M_T [p_{\text{2D}}, 1]^T$$

이제 앞선 전체 사진 투영 수식의 좌변에 $M_T$를 곱하면, 크롭 사진 투영 수식과 완전히 같은 픽셀을 가리키게 되므로 다음과 같은 통합 등식이 만들어집니다:

$$M_T K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] = K_c [R_c \mid T_c]$$

우리가 인공지능으로 찾아내고 싶은 진짜 정답은 바로 전체 차량 기준의 시선 회전 행렬 $R_{\text{img}}$입니다. 위 등식에서 회전 성분($R$)만을 떼어내고, 양변에 역연산(역행렬 $^{-1}$)을 곱하여 좌변에 $R_{\text{img}}$만 홀로 남기면 다음과 같은 최종 닫힌 형태(closed-form) 변환 수식이 완성됩니다:

$$R_{\text{img}} = K_c \cdot R_c \cdot M_T^{-1} \cdot K_{\text{img}}^{-1}$$

여기서 쓰인 기호들의 역변환 원리는 다음과 같습니다:
- $K_{\text{img}}^{-1}$: 원본 2D 사진 픽셀을 카메라 렌즈 이전의 3D 빛 방향으로 거꾸로 되돌립니다 (역행렬 $^{-1}$은 반대 방향으로 되돌리는 연산입니다).
- $M_T^{-1}$: 잘라냈던 바운딩 박스 위치만큼 다시 더해서($+x_s, +y_s$) 원래 전체 사진 위치로 되돌립니다.
- $K_c$: 크롭 공간의 가상 렌즈 스케일을 적용합니다.
- $R_c$: 크롭 이미지 안에서 인공지능 신경망이 예측한 순수 안구 회전 각도입니다.
- 결과: 크롭 이미지에서 예측한 국소 시선 $R_c$가 전체 차량 화면 기준의 3차원 절대 시선 $R_{\text{img}}$로 완벽하게 복원됩니다.

#### 4. MLP를 이용한 가상 렌즈 파라미터 $K_c$의 동적 학습

위 변환식에서 원본 카메라 $K_{\text{img}}$는 정해져 있지만, 가상 크롭 렌즈 $K_c$의 미지수 $[f_x, f_y, c_x, c_y]$는 운전자가 고개를 움직여 눈 바운딩 박스 $b = [x_s, y_s, w, h]$(시작점 $x_s, y_s$와 가로 폭 $w$, 세로 높이 $h$)의 크기와 위치가 바뀔 때마다 매 순간 달라집니다.

논문은 이 4가지 파라미터를 복잡한 3D 수작업 캘리브레이션 대신, 2D 바운딩 박스 좌표 $b$(4개의 숫자)를 입력받는 경량 신경망 MLP $\Phi^M$이 실시간으로 알아맞히도록 설계했습니다:

$$[f_x, f_y, c_x, c_y]^T = \Phi^M(b)$$

$$K_c = g_2(\Phi^M(b))$$

최종 좌표 변환 함수 $h(\cdot)$는 다음과 같이 완성됩니다:

$$R_{\text{img}} = g_2(\Phi^M(b)) \cdot R_c \cdot g_1(x_s, y_s)^{-1} \cdot K_{\text{img}}^{-1} = h(\Phi^M(b), b, K_{\text{img}}, R_c)$$

이 방식의 핵심적인 장점은 불안정하고 오차가 큰 외부 3D 헤드 포즈 추정기에 전혀 의존하지 않는다는 점입니다. 2D 바운딩 박스의 크기와 위치 좌표 $b$만을 이용해 기하학적 투영 행렬을 스스로 학습하므로, 머리 위치가 크게 흔들리거나 이동하더라도 각도 오차가 증폭되지 않고 극도로 강건한 시선 변환을 유지할 수 있습니다.

### 3.2 Eye Branch $\Phi_{\text{eye}}$의 상세 구조

눈 브랜치는 좌우 눈 크롭 이미지 $I_c^L, I_c^R$로부터 국소적 안구 회전 단서를 정밀하게 추출합니다:

1. 특징 추출: 좌우 눈 크롭을 가중치를 공유하는 4-stack Hourglass CNN 인코더 $\Phi_e$에 통과시켜 국소 특징 맵 $F^L, F^R$을 추출합니다:
   $$F^L = \Phi_e(I_c^L), \quad F^R = \Phi_e(I_c^R)$$

2. 3D 공간 맥락 복원 (Fourier HP Embedding): 안구 크롭 시 손실되는 3차원 위치 맥락을 제공하기 위해, 전체 안면 이미지 $I$를 사전학습된 CNN 모델 $\Phi_{\text{HP}}$에 통과시켜 3차원 머리 위치 벡터 $\text{HP} = \Phi_{\text{HP}}(I)$를 획득합니다. 이를 Fourier embedding $\Psi^{\text{HP}}$와 MLP $\Phi_e^{\text{HP}}$에 통과시켜 고차원 공간 임베딩을 생성합니다.

3. Crop 공간 시선 회귀: 국소 특징과 공간 임베딩을 concatenation한 뒤, 눈 전용 회귀기 $R^e$에 입력하여 Crop 좌표계 시선 예측 $\tilde{R}_c^L, \tilde{R}_c^R$을 얻습니다:
   $$\tilde{R}_c^L = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^L)$$
   $$\tilde{R}_c^R = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^R)$$

4. Image 공간 변환 및 가중 평균: 앞서 정의한 변환 함수 $h(\cdot)$와 전용 MLP $\Phi_c^M$을 통해 Image 공간으로 사영한 뒤, 좌우 눈의 예측을 가중 평균하여 최종 눈 시선 예측 $\tilde{R}_{\text{img}}^e$를 도출합니다:
   $$\tilde{R}_{\text{img}}^L = h(\Phi_c^M(b_c^L), b_c^L, K_{\text{img}}, \tilde{R}_c^L)$$
   $$\tilde{R}_{\text{img}}^R = h(\Phi_c^M(b_c^R), b_c^R, K_{\text{img}}, \tilde{R}_c^R)$$
   $$\tilde{R}_{\text{img}}^e = w(\tilde{R}_{\text{img}}^L, \tilde{R}_{\text{img}}^R)$$

눈 브랜치는 정답 시선 벡터 $R_{\text{img}}^{\text{GT}}$와의 각도 오차 코사인 손실 $\mathcal{L}_{\text{eye}}$로 학습됩니다:

$$\mathcal{L}_{\text{eye}} = \cos^{-1}((\tilde{R}_{\text{img}}^e)^T R_{\text{img}}^{\text{GT}})$$

### 3.3 Face Branch $\Phi_{\text{face}}$의 상세 구조

얼굴 브랜치는 안면 전체의 거시적 맥락과 머리 자세를 바탕으로 시선을 추정합니다:

1. 특징 추출: $96 \times 224$ 해상도의 안면 크롭 이미지 $I_f$를 6-stack Hourglass CNN 인코더 $\Phi_f$에 통과시켜 전역 안면 특징 $F^f$를 추출합니다:
   $$F^f = \Phi_f(I_f)$$

2. HP 임베딩 결합: 동일하게 추출된 HP 값을 얼굴 전용 MLP $\Phi_f^{\text{HP}}$로 인코딩한 뒤 결합하고, 얼굴 회귀기 $R^f$를 통해 Crop 공간 시선 예측 $\tilde{R}_c^f$를 생성합니다:
   $$\tilde{R}_c^f = R^f(\Phi_f^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^f)$$

3. Image 공간 변환: 얼굴 바운딩 박스 $b_f$와 MLP $\Phi_f^M$을 통해 Image 공간 시선 예측 $\tilde{R}_{\text{img}}^f$로 변환합니다:
   $$\tilde{R}_{\text{img}}^f = h(\Phi_f^M(b_f), b_f, K_{\text{img}}, \tilde{R}_c^f)$$

얼굴 브랜치 역시 독립적인 각도 손실 함수 $\mathcal{L}_{\text{face}}$로 지도 학습됩니다:

$$\mathcal{L}_{\text{face}} = \cos^{-1}((\tilde{R}_{\text{img}}^f)^T R_{\text{img}}^{\text{GT}})$$

### 3.4 CLIP-Guided Fusion 모듈의 작동 원리

마스크 착용 시에는 하안부가 가려져 $\Phi_{\text{face}}$가 흔들리지만 $\Phi_{\text{eye}}$가 안정적이며, 반대로 안경 반사광이나 짙은 그림자가 눈을 가릴 때는 $\Phi_{\text{eye}}$가 부정확해지고 $\Phi_{\text{face}}$가 머리 자세를 기반으로 정확한 가이드를 제공합니다.

본 논문은 두 예측의 중요도를 고정하지 않고, 사전학습된 Vision-Language 모델인 CLIP($\Phi_{\text{CLIP}}$)을 활용하여 시각적 가림 상태를 의미론적으로 판단합니다.

1. 시각-언어 의미 특징 추출: 원본 이미지 $I$와 눈의 시각적 상태를 묘사하는 텍스트 맥락 벡터 $c$(fixed semantic prompt)를 CLIP 모델 $\Phi_{\text{CLIP}}$에 입력하여 멀티모달 특징 $F_{\text{CLIP}}$을 추출합니다:
   $$F_{\text{CLIP}} = \Phi_{\text{CLIP}}(I, c)$$
   이 context vector $c$는 모델이 임의의 task-agnostic 구조를 유지하면서도 "눈이 명확히 보이는지", "안경 반사나 마스크로 가려졌는지"에 주목하도록 유도하는 의미론적 정규화(semantic regularization) 역할을 수행합니다.

2. 동적 가중치 산출 및 적응적 결합: $F_{\text{CLIP}}$을 융합 전용 MLP $\Phi_{\text{fuse}}$에 전달하여 softmax 정규화된 동적 중요도 가중치를 산출하고, 두 브랜치의 시선 예측을 가중 결합하여 최종 예측 $\tilde{R}_{\text{img}}^{\text{fused}}$를 완성합니다:
   $$\tilde{R}_{\text{img}}^{\text{fused}} = \sigma(\Phi_{\text{fuse}}(F_{\text{CLIP}})^T) [\tilde{R}_{\text{img}}^e, \tilde{R}_{\text{img}}^f]^T$$

최종 융합 손실 $\mathcal{L}_{\text{fuse}}$는 다음과 같이 계산됩니다:

$$\mathcal{L}_{\text{fuse}} = \cos^{-1}((\tilde{R}_{\text{img}}^{\text{fused}})^T R_{\text{img}}^{\text{GT}})$$

전체 프레임워크는 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{eye}} + \mathcal{L}_{\text{face}} + \mathcal{L}_{\text{fuse}}$의 다중 손실 합을 바탕으로 End-to-End로 동시에 최적화됩니다.

---

## 4. 실험 결과

### 4.1 데이터셋 및 평가 프로토콜

실험은 실제 주행 IR 환경, 합성 RGB 환경, 일상 웹캠 환경을 아우르는 세 가지 벤치마크에서 엄격하게 수행되었습니다:

| 데이터셋 | 도메인 및 모달리티 | 규모 및 피험자 | 평가 프로토콜 |
|---|---|---|---|
| IVGaze | 차량 내부 적외선(IR) | 44,705 이미지, 125명 | 3-fold cross-validation |
| GazeGene | 대규모 다중시점 합성 RGB | 9개 카메라 뷰, 56개 캐릭터 | 마지막 10명 캐릭터 test |
| MPIIFaceGaze | 실환경 웹캠 RGB | 15명 피험자 | Leave-one-subject-out (LOSO) |

주요 평가 지표는 정답 시선 벡터와 예측 벡터 사이의 각도 오차(Angular Error, AE, 단위: 도 °)이며, IVGaze에서는 다양한 오차 허용 임계값($<2^\circ, <4^\circ, <6^\circ, <8^\circ$)에서의 Average Precision(AP, %)을 함께 평가했습니다.

학습 설정: $\Phi_{\text{face}}$와 $\Phi_{\text{eye}}$는 AdamW 옵티마이저를 사용하고, $\Phi_{\text{fuse}}$는 Adam 옵티마이저를 사용하며 초기 학습률은 0.001로 설정되었습니다. StepLR(step size 60, gamma 0.5) 스케줄러로 총 100 에포크 동안 학습되었습니다. A100 GPU 기준 추론 지연시간은 평균 42ms로 측정되었습니다.

### 4.2 주요 정량적 비교 (Benchmark Comparison)

| Method | IVGaze (IR) | GazeGene (Synth RGB) | MPIIFaceGaze (Wild RGB) |
|---|---|---|---|
| FullFace | 13.67° | 3.54° | 4.93° |
| DWG | 8.82° | - | - |
| Gaze360 | 8.15° | - | - |
| $\text{FullFace}^+$ | 7.48° | - | - |
| GazeTR | 7.33° | 3.37° | 4.00° |
| XGaze | 7.06° | - | - |
| GazePTR | 7.04° | - | - |
| GazeDPTR | 6.71° | - | - |
| Dilated-Net | - | 3.08° | 4.42° |
| ResNet-50 | - | 3.36° | 4.20° |
| RT-Gene | - | - | 4.30° |
| CA-Net | - | - | 4.10° |
| L2CS | - | - | 3.92° |
| Ours (제안 방법) | 6.29° | 1.87° | 3.66° |

제안 방법은 세 개 벤치마크 모두에서 기존 SOTA 모델들을 일관되게 압도했습니다.

특히 주목할 점은 대규모 다중시점 합성 데이터셋인 GazeGene에서의 39.3% 오차 감소(3.08° → 1.87°)입니다. 9개의 다양한 카메라 시점과 극단적인 머리 위치 변화가 존재하는 환경에서, Learnable Transformation $h(\cdot)$와 Fourier HP 임베딩이 머리 위치 불변성을 완벽히 확보했음을 실증합니다. 실제 차량 주행 IR 데이터인 IVGaze에서도 기존 최고 모델 GazeDPTR 대비 6.3%의 오차를 추가로 줄였습니다.

### 4.3 IVGaze 임계값별 Average Precision (AP) 분석

| Method | AP ($< 2^\circ$) | AP ($< 4^\circ$) | AP ($< 6^\circ$) | AP ($< 8^\circ$) |
|---|---|---|---|---|
| FullFace | 2.3% | 8.8% | 17.8% | 28.0% |
| DWG | 6.6% | 21.7% | 33.8% | 53.2% |
| Gaze360 | 9.2% | 27.3% | 44.6% | 58.9% |
| $\text{FullFace}^+$ | 14.2% | 31.1% | 46.7% | 61.3% |
| GazeTR | 17.0% | 32.8% | 45.7% | 64.7% |
| XGaze | 11.7% | 32.7% | 51.6% | 65.2% |
| GazePTR | 17.6% | 34.6% | 49.3% | 66.7% |
| GazeDPTR | 22.1% | 36.0% | 50.3% | 68.4% |
| Ours | 16.0% | 38.5% | 58.7% | 73.5% |
| Ours (w/o Sunglass) | 16.2% | 39.1% | 59.6% | 74.6% |

오차 허용 범위에 따른 정밀도 분석에서 제안 방법은 $<4^\circ, <6^\circ, <8^\circ$의 모든 주요 구간에서 최고 정밀도를 달성했습니다. 특히 $<6^\circ$ 구간에서 기존 최고 모델 대비 16.69% 향상된 58.7%를 기록하여 실용적인 주행 안전 허용 범위에서 압도적인 신뢰성을 보여주었습니다.

### 4.4 악세서리 및 가림 조건별 강건성 분석

| Method | 안경 착용 | 안경 미착용 | 마스크 착용 | 마스크 미착용 | 선글라스 |
|---|---|---|---|---|---|
| FullFace | 14.43° | 12.40° | 15.20° | 13.35° | 21.39° |
| DWG | 9.20° | 8.19° | 9.43° | 8.69° | 17.43° |
| Gaze360 | 8.30° | 7.91° | 8.95° | 7.99° | 17.99° |
| $\text{FullFace}^+$ | 7.59° | 7.30° | 8.37° | 7.30° | 16.50° |
| XGaze | 7.07° | 7.03° | 7.80° | 6.90° | 15.15° |
| GazeTR | 7.40° | 7.22° | 8.12° | 7.17° | 17.49° |
| GazePTR | 7.13° | 6.90° | 7.78° | 6.89° | 16.54° |
| GazeDPTR | 6.77° | 6.63° | 7.44° | 6.57° | 16.41° |
| Ours | 6.21° | 5.71° | 6.72° | 5.39° | 18.70° |

안경 착용(6.21°) 및 마스크 착용(6.72°) 조건 모두에서 기존 모델들을 큰 차이로 앞섰습니다. 불투명 선글라스의 경우 눈 정보가 원천적으로 차단되어 18.70°로 상승하였으나, 이를 제외한 모든 실제 가림 상황에서 견고한 성능을 입증했습니다.

### 4.5 융합 모듈 및 개별 브랜치 어블레이션

| 조건 | $\Phi_{\text{face}}$ (단독) | $\Phi_{\text{eye}}$ (단독) | $\Phi_{\text{fuse}}$ (제안 융합) |
|---|---|---|---|
| 안경 착용 | 6.65° | 6.98° | 6.21° |
| 안경 미착용 | 6.06° | 6.38° | 5.71° |
| 마스크 착용 | 7.22° | 6.79° | 6.72° |
| 마스크 미착용 | 5.71° | 6.31° | 5.39° |
| 전체 평균 | 6.69° | 6.77° | 6.29° |

개별 브랜치 어블레이션 결과, 마스크 착용 시에는 하안부 가림으로 인해 $\Phi_{\text{face}}$(7.22°)가 크게 무너지지만 $\Phi_{\text{eye}}$(6.79°)가 이를 지탱합니다. 반대로 안경 착용 시에는 반사광으로 인해 $\Phi_{\text{eye}}$(6.98°)가 저하되지만 $\Phi_{\text{face}}$(6.65°)가 보완합니다. CLIP 기반 $\Phi_{\text{fuse}}$는 이러한 상보적 신호를 동적으로 융합하여 전체 평균 6.29°로 두 단일 브랜치보다 뛰어난 결과를 달성했습니다.

또한 ResNet-50 기반 단순 융합(6.41°)과 비교했을 때 CLIP 특징 융합(6.29°)이 전 조건에서 우수함을 보였으며, 마스크 미착용 조건에서는 6.00°에서 5.39°로 10.16%의 뚜렷한 오차 감소를 기록했습니다.

### 4.6 정성적 결과 및 시각화 비교

![Figure 5](/images/clip-dual-crop-gaze/fig5_qualitative_ablation.jpeg)

*Figure 5: IVGaze(IR) 및 GazeGene(RGB) 벤치마크에서의 정성적 비교 및 어블레이션 결과입니다. 안경 반사광, 마스크 가림, 극단적 머리 회전 및 머리카락 가림 상황에서도 제안 방법의 예측 벡터(파란 화살표)가 ground truth(초록 화살표)에 완벽히 일치함을 보여줍니다.*

정성적 시각화 결과, 안경 반사광이나 극단적인 머리 꺾임, 마스크 착용 등 기존 모델들이 크게 빗나가는 상황에서도 제안 방법의 예측 벡터가 Ground Truth와 거의 일치하는 궤적을 유지했습니다.

### 4.7 Head Position(HP) 오차 민감도 및 Normalization 비교

| 조건 | Zhang's Normalization | Ours (Learnable $h(\cdot)$) |
|---|---|---|
| 안경 착용 | 7.10° | 6.65° |
| 안경 미착용 | 6.32° | 6.06° |
| 마스크 착용 | 7.39° | 7.22° |
| 마스크 미착용 | 6.72° | 5.71° |
| 전체 평균 | 7.02° | 6.69° |

동일한 안면 브랜치 구조에서 전통적 Zhang 정규화 방식을 본 논문의 Learnable Transformation $h(\cdot)$로 교체했을 때, 전체 평균 오차가 7.02°에서 6.69°로 크게 개선되었습니다.

![Figure 6](/images/clip-dual-crop-gaze/fig6_sensitivity_plot.jpeg)

*Figure 6: Head Position(HP) 오차 증가에 따른 각도 오차 민감도 곡선입니다. 전통적인 IVGaze normalization 방식은 머리 위치 오차가 증가함에 따라 각도 오차가 7° 이상으로 급격히 폭발하지만, 제안 방법(Ours)은 2° 미만의 매우 안정적인 오차를 유지하여 뛰어난 위치 불변성을 입증합니다.*

인위적으로 HP 오차를 주입한 민감도 곡선 실험(Figure 6)에서, 기존 IVGaze 축 정규화 방식은 10cm 변위 시 각도 오차가 7° 이상으로 치솟았으나, 제안 방법은 2° 미만을 유지하며 압도적인 강건성을 실증했습니다.

---

## 5. 결론 및 핵심 시사점

본 논문은 실환경 시선 추정에서 머리 위치 변화로 인해 발생하는 Multi-mapping 모호성과 전통적 Gaze Normalization 기법의 오차 전파 문제를 체계적인 기하학적 학습 프레임워크로 해결한 연구입니다.

논문이 제시하는 핵심 학술적 시사점은 다음과 같습니다:

첫째, Crop 공간과 Image 공간의 수학적 분리 및 MLP 기반 가상 내부 파라미터($K_c$) 학습 설계입니다. Crop 공간에서 시선 방향의 1대1 고유성을 확보하고, 핀홀 사영 기하학을 바운딩 박스 좌표의 함수로 학습함으로써, 불안정한 3D 헤드 포즈 추정기 없이도 10cm 머리 이동 오차에서 2° 미만의 안정성을 유지하는 강력한 위치 불변성을 확립했습니다.

둘째, 4-stack Hourglass(눈)와 6-stack Hourglass(얼굴)의 Dual-stream 구조에 Fourier Head Position 임베딩과 CLIP 멀티모달 의미 정규화를 결합한 하이브리드 설계입니다. 마스크나 안경 등 다양한 차폐 조건에서 시각-언어 모델의 의미 이해를 바탕으로 상보적 신호를 동적으로 융합하는 실용적 패러다임을 제시했습니다.

셋째, 실제 차량 적외선(IVGaze), 대규모 다중시점 합성(GazeGene), 일상 웹캠(MPIIFaceGaze) 등 모달리티와 환경을 넘나드는 포괄적 평가를 통해 특정 도메인에 편향되지 않는 일반화 성능을 성공적으로 입증했습니다.

다만 안구 영역이 완전히 차폐되는 불투명 선글라스 상황(18.70°)에 대한 추가적인 컨텍스트 추론과, 차량 탑재를 위한 실시간 경량화 최적화(현재 A100 기준 42ms)는 향후 지속적인 연구가 필요한 과제입니다.
