---
title: "[ECCV 2026] SegFS: 느린 경로와 빠른 경로의 분업으로 실시간 Open-Vocabulary Video Instance Segmentation을 달성하다"
date: 2026-09-22T20:16:59+09:00
draft: false
math: true
tags: ["Paper Review", "Video Instance Segmentation", "Open-Vocabulary", "Real-Time", "Mobile", "Dual-Path", "ECCV 2026"]
categories: ["Paper Review"]
summary: "SegFS는 무거운 object-centric 모델을 keyframe에서만 실행하는 slow path와, backbone feature만으로 마스크를 생성하는 lightweight fast path를 교대 운용하여, 모바일 기기에서 30 FPS 이상의 실시간 Open-Vocabulary Video Instance Segmentation을 달성합니다."
cover:
  image: "/images/segfs/_page_5_Figure_2.jpeg"
  alt: "SegFS Dual-Path Architecture Overview"
---

> 참조 논문
> - Barsellotti, L. et al. "Segmenting, Fast and Slow: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing." ECCV 2026.

![Figure 1: SegFS Overview](/images/segfs/_page_5_Figure_2.jpeg)
*Figure 1: SegFS 아키텍처 전체 개요. Slow path는 sparse keyframe에서 object embedding을 추출하고, fast path는 inter-frame에서 backbone feature만으로 경량 마스크 예측을 수행합니다.*

---

## 1. 한 줄 요약

Open-Vocabulary Video Instance Segmentation의 기존 모델들은 모든 프레임마다 무거운 Feature Enhancer를 반복 실행하여 모바일 기기에서 실시간 추론이 불가능합니다. SegFS는 이 무거운 모듈을 드문드문 샘플링한 keyframe에서만 실행하는 slow path와, 나머지 프레임에서는 backbone feature만으로 마스크를 생성하는 경량 fast path를 교대 운용합니다. 그 결과 기존 모바일 모델 MOBIUS 대비 최대 14배 낮은 latency로 Samsung Galaxy S25 Ultra에서 30 FPS 실시간 추론을 달성합니다.

---

## 2. 연구 배경과 동기

### 2.1 문제 정의

붐비는 교차로에서 길을 건너는 보행자들을 바라보는 장면을 떠올려 봅니다. 시야에 들어오는 수많은 사람과 차량을 마주할 때, 인간의 뇌는 1초에 수십 번씩 눈을 깜빡이며 매 찰나마다 "저 물체는 어떤 옷을 입은 사람인가?", "저 바퀴 달린 것은 승용차인가 트럭인가?"라며 언어적이고 개념적인 분석을 매 순간 처음부터 반복하지 않습니다. 처음 시선이 머무는 찰나에 "저기 파란 옷을 입은 사람과 검은 승용차가 지나간다"는 대상의 의미적 정체(semantic identity)를 한 번 깊이 파악하고 나면, 이후 몇 초 동안은 복잡한 의미 분석을 멈추고 망막에 맺히는 시각적 윤곽과 움직임만을 가볍게 따라가며 대상을 추적합니다. 깊고 무거운 개념 이해는 드물게 수행하고, 연속적인 시각 추적은 빠르고 직관적인 감각으로 분업하여 뇌의 인지적 과부하를 막는 것입니다.

Open-Vocabulary Video Instance Segmentation, 줄여서 OV-VIS는 AI가 이와 동일한 시각 지각 능력을 갖추도록 하는 과제입니다. 스마트폰 카메라로 거리를 비추면서 사람, 자전거, 강아지 같은 자유 형식의 텍스트를 입력하면, 사전에 학습하지 않은 미지의 카테고리까지 실시간으로 검출하고 추적하며 픽셀 단위 세그멘테이션 마스크를 생성해야 합니다.

이 과제의 표준 아키텍처는 DETR 계열의 object-centric 프레임워크를 따릅니다. 전체 파이프라인은 세 개의 시각 컴포넌트가 직렬로 연결된 구조입니다.

첫 번째는 Visual Backbone입니다. 입력 프레임으로부터 multi-scale feature pyramid를 추출합니다. 높은 해상도의 $P_2$부터 낮은 해상도의 $P_5$까지, 서로 다른 축척의 시각 특징 맵 네 장을 생성합니다. 큰 물체는 축소된 저해상도 맵에서, 작은 물체는 원본에 가까운 고해상도 맵에서 포착됩니다.

두 번째는 Feature Enhancer입니다. 이 모듈이 가장 무거운 핵심 연산을 담당합니다. Pixel decoder가 multi-scale deformable attention으로 서로 다른 축척의 feature map 사이에 정보를 교환하고, early fusion 모듈이 텍스트 임베딩과 시각 feature 사이에 양방향 cross-attention을 수행합니다. 사용자가 입력한 텍스트 카테고리의 의미를 시각 특징에 녹여내는 과정으로, 이 단계를 거쳐야 비로소 feature map이 어디에 사람이 있고 어디에 자전거가 있는지를 구분할 수 있게 됩니다.

세 번째는 Object Decoder입니다. Feature Enhancer가 출력한 텍스트 정렬 feature map 위에서, 학습 가능한 object query들이 cross-attention을 통해 인스턴스별 embedding을 추출합니다. 이 embedding은 텍스트 임베딩과의 유사도로 카테고리를 판별하고, 동시에 고해상도 feature map과의 $1 \times 1$ convolution으로 인스턴스별 세그멘테이션 마스크를 생성합니다.

프레임 간 추적은 MinVIS 패러다임을 따릅니다. 별도의 시공간 추적 모듈 없이, 인접 프레임의 object embedding 사이에 코사인 유사도 기반 bipartite matching을 수행하여 동일 인스턴스를 연결합니다.

### 2.2 기존 방법의 한계: Feature Enhancer라는 병목

![Figure 2: Efficiency Analysis](/images/segfs/_page_1_Figure_2.jpeg)
*Figure 2: OV-VIS 아키텍처의 효율성 분석. MOBIUS-Mini-M, GLEE-Lite, TROY-VIS의 각 컴포넌트별 FLOPs와 Samsung Galaxy S25 Ultra에서의 on-device latency를 비교합니다.*

앞서 설명한 지각 비유로 돌아가면, 기존의 OV-VIS 연구들은 마치 1초에 서른 번씩 눈을 깜빡일 때마다 매 프레임 "저것은 무엇인가?"라는 무거운 언어-시각 교차 주의집중을 강박적으로 반복하는 것과 같습니다.

MOBIUS나 TROY-VIS 같은 최근 연구들은 이 Feature Enhancer를 더 가볍게 재설계하여 모바일 친화적인 구조를 제안했습니다. MOBIUS는 pixel decoder 내 single intermediate scale만 선택하여 텍스트 융합과 multi-scale attention을 수행하고, TROY-VIS는 가장 낮은 해상도의 feature map에서만 vision-language cross-attention을 적용한 뒤 bottom-up으로 전파하는 방식을 택했습니다.

그러나 Figure 2가 보여주듯, 이러한 경량화에도 불구하고 Feature Enhancer는 여전히 전체 추론 비용의 압도적인 비중을 차지합니다. 이론적 FLOPs보다 실제 on-device latency에서 이 불균형은 더욱 두드러집니다. 그 이유는 Feature Enhancer 내부의 dense interaction, 즉 multi-scale deformable attention과 vision-language cross-attention이 GPU 연산 유닛의 병렬 활용에 비효율적인 메모리 접근 패턴을 보이기 때문입니다. 입력 해상도가 커지고 텍스트 어휘 크기가 늘어날수록 이 격차는 가파르게 벌어집니다.

기존의 경량화 연구들이 마치 무거운 분석 작업의 부피만 조금씩 줄여보려 했다면, 모든 프레임에서 고차원 의미 융합을 혼자 감당해야 하는 근본적인 구조 자체는 바꾸지 못한 셈입니다. MobileInst나 TROY-VIS처럼 keyframe 기반으로 Object Decoder만 건너뛰는 접근 역시 진짜 병목인 Feature Enhancer를 매 프레임 실행해야 하므로, 총 연산 시간의 절감 효과가 극히 제한적이었습니다.

### 2.3 SegFS의 핵심 기여

- 기존 keyframe 방식이 Object Decoder만 생략한 것과 달리, Feature Enhancer와 Object Decoder를 모두 sparse keyframe으로 이전하는 dual-path 접근을 제안합니다. 가장 무거운 모듈까지 slow path로 밀어내어 근본적인 병목을 제거합니다.
- 이 접근을 실현하는 새로운 프레임워크 SegFS를 설계합니다. Slow path에서 추출한 object embedding을 backbone feature space로 역투영하여 fast path에 주입하는 Fast Feature Aggregator를 통해, backbone feature만으로 높은 품질의 인스턴스 마스크를 생성합니다.
- 다양한 OV-VIS 벤치마크에서 SegFS가 무거운 object-centric 모델의 zero-shot 능력과 마스크 품질을 유지하면서도, Samsung Galaxy S25 Ultra에서 30 FPS 이상의 실시간 추론 속도를 달성함을 실험으로 검증합니다.

---

## 3. 제안 프레임워크: SegFS

SegFS의 핵심 직관을 한 문장으로 요약하면 이렇습니다. Backbone feature map에는 이미 fine-grained localization에 필요한 공간적 의미 정보가 충분히 인코딩되어 있으므로, 무거운 Feature Enhancer를 매 프레임 실행할 필요 없이, keyframe에서 한 번 추출한 object embedding만 주입하면 된다는 것입니다.

직관적으로 이해해 봅니다. Backbone이 추출하는 feature pyramid는 이미 "여기에 윤곽이 있고, 저기에 덩어리가 있다"는 공간 정보를 풍부하게 담고 있습니다. 다만 이 feature만으로는 "저 덩어리가 사람인지 자전거인지"를 판별할 수 없습니다. 그 의미적 판별 능력은 Feature Enhancer와 Object Decoder가 텍스트와의 cross-attention을 통해 부여하는 것입니다. SegFS의 발상은 이 의미적 판별 결과를 한 번만 생성한 뒤 여러 프레임에 걸쳐 재사용하자는 것입니다. 5~6 프레임 사이에 장면이 급격히 바뀌지 않는 한, "저 덩어리가 사람이다"라는 판단은 유효하게 유지되기 때문입니다.

### 3.1 전체 아키텍처: Slow Path와 Fast Path의 교대 운용

이 지각 분업 구조를 AI 시스템으로 체계화한 것이 바로 SegFS의 dual-path 처리 아키텍처입니다. 무거운 의미 이해는 드물게, 가벼운 마스크 예측은 빈번하게라는 원칙에 따라 두 개의 네트워크를 전략적으로 교대 실행합니다.

Slow path는 기존의 object-centric OV-VIS 모델 전체를 말합니다. GLEE, MOBIUS, TROY-VIS 등 어떤 모델이든 가중치를 얼려놓은 채 그대로 사용할 수 있습니다. 이 slow path는 sparse하게 샘플링된 keyframe에서만 실행되어, Visual Backbone → Feature Enhancer → Object Decoder의 전체 파이프라인을 거쳐 텍스트 정렬된 object embedding을 추출하는 느리고 정밀한 시스템입니다. 5~6 프레임에 한 번만 이 무거운 이해 과정을 수행합니다.

Fast path는 나머지 $T$개의 inter-frame에서 실행됩니다. Feature Enhancer와 Object Decoder를 완전히 건너뛰고 backbone의 multi-scale feature만 사용합니다. 대신 slow path가 keyframe에서 확립해 둔 object embedding을 fast feature space로 주입받아, 최소한의 연산으로 연속 프레임의 마스크를 생성하는 빠른 시스템입니다.

프레임 간 추적은 별도의 추적 모듈 없이, keyframe에서 추출한 object embedding의 코사인 유사도 기반 bipartite matching으로 MinVIS 패러다임을 그대로 따릅니다.

### 3.2 Object Embedding의 투영과 두 갈래 분기

지각 분업 비유로 풀어보면, slow path가 파악한 "저 대상이 보행자이고 자전거이다"라는 고차원의 개념적 정체(object embedding)를 연속 프레임을 바라보는 빠른 시각 피질의 눈높이(backbone feature space)에 맞추어 번역해 전달하는 단계입니다.

Slow path에서 추출한 object embedding은 Feature Enhancer와 Object Decoder를 거쳐 텍스트 카테고리와 정밀하게 정렬된 고차원 표현입니다. 반면 fast path가 다루는 대상은 visual backbone이 직접 출력한 순수한 시각 feature map입니다. 두 표현은 차원도 다르고 분포 공간도 상이하므로, object embedding을 fast path에 곧바로 주입하면 공간적 불일치가 발생합니다.

이 간극을 해소하기 위해 SegFS는 3-layer FFN과 LayerNorm을 사용하여 object embedding을 backbone feature space로 투영합니다. 이렇게 투영된 임베딩은 두 갈래로 나뉘어 fast path의 서로 다른 역할을 분담합니다.

- 첫 번째 갈래: 빠른 시각 feature를 유도할 의미 안내표 (Conditioning Tokens)
  - Slow network가 출력한 300개의 object query 중 텍스트 카테고리와의 최대 유사도 점수를 기준으로 상위 $K$개만 선별합니다. DETR 계열 모델이 출력하는 수백 개의 쿼리 중에는 배경 영역이나 검출 실패에 해당하는 저품질 쿼리가 다수 포함되어 있습니다. 길거리의 허상이나 시야 밖의 찌꺼기 신호가 주입되면 마스크 품질이 급격히 저하되므로, 텍스트 정합 점수를 기준으로 확실한 인스턴스만 걸러냅니다.
  - 선별된 $K$개 토큰에 학습 가능한 background token $t_{\text{bg}}$를 하나 추가하여 총 $K+1$개의 토큰을 구성합니다. 대상이 없는 빈 바탕(배경)을 담아둘 명시적인 슬롯이 없다면, 후속 공간 매칭 과정에서 모든 공간 셀이 억지로 특정 인스턴스에 배정되어 배경까지 물체로 오인하는 거짓 양성 마스크를 쏟아내기 때문입니다.
  - 준비된 $K+1$개의 토큰은 self-attention과 cross-attention이 교차하는 두 블록을 통과합니다. Cross-attention 레이어에서는 $K+1$개 토큰이 Query 역할을 담당하고, keyframe backbone의 최저 해상도 feature map $P_5$가 Key와 Value 역할을 담당합니다. 이 과정을 통해 고차원 의미 토큰들이 실제 영상 속 공간 배치에 맞추어 위치 정보를 재정렬합니다.
- 두 번째 갈래: 최종 마스크 생성을 위한 커널 (Mask Kernels)
  - 별도의 3-layer FFN을 통해 object embedding을 마스크 생성을 위한 커널 공간으로 투영합니다. 첫 번째 갈래가 화면 어디에 무엇이 있는지를 알려주는 안내도라면, 이 커널은 나중에 고해상도 시각 feature map 위에서 해당 객체의 윤곽을 단번에 떠낼 필터 역할을 하며, $1 \times 1$ convolution을 통해 인스턴스별 마스크 activation을 도출합니다.

### 3.3 Fast Feature Aggregator: 경량 고해상도 Feature Map 생성

![Figure 3: Fast Feature Aggregator](/images/segfs/_page_6_Figure_2.jpeg)
*Figure 3: Fast Feature Aggregator 상세 구조. Keyframe object embedding이 $P_5$ backbone feature에 주입된 뒤, progressive upsampling과 fusion을 거쳐 최종 고해상도 feature map을 생성합니다.*

시각 지각 비유로 돌아가 봅니다. 인간이 대상을 인식하고 추적할 때, 방금 전 "저기 파란 옷을 입은 사람과 검은 승용차가 지나간다"는 개념적 판단을 한 번 내리고 나면, 이후 1초 동안 눈앞에서 빠르게 움직이는 대상을 쫓을 때는 더 이상 복잡한 언어적 추론을 매 순간 반복하지 않습니다. 오직 망막에 연속으로 맺히는 빠른 시각 신호 위에서, 이미 기억해 둔 대상의 정체를 힌트 삼아 눈 깜짝할 사이에 형태와 윤곽을 가볍게 포착해 나갑니다.

Fast Feature Aggregator는 바로 이 빠른 형태 포착 과정을 담당하는 핵심 연산 장치입니다. 초당 수십 번씩 밀려드는 연속 프레임에서 무거운 Feature Enhancer를 전혀 거치지 않고, 오직 visual backbone이 출력한 순수 시각 feature map에 keyframe의 object embedding을 유기적으로 결합하여 고품질 세그멘테이션 마스크를 실시간으로 완성합니다. 전체 연산 흐름은 세 단계로 체계화됩니다.

#### 1. 채널 차원 규격화 (Preprocessing)

가장 먼저 해결해야 할 문제는 백본 간의 규격 호환성입니다. MobileNetV4, ResNet50, EfficientViT 등 visual backbone의 종류에 따라 출력되는 multi-scale feature maps $F = \{P_2, P_3, P_4, P_5\}$의 채널 수는 제각각입니다. 이를 일관된 표준 차원 $D = 256$으로 선형 투영하여 이후의 모든 연산 규격을 통일합니다.

- 저해상도 맵 ($P_4, P_5$): 해상도가 낮고 전역적 문맥이 풍부하므로, 표준 $1 \times 1$ convolution을 적용해 채널 차원만 가볍게 축소합니다.
- 고해상도 맵 ($P_2, P_3$): 공간 해상도가 크고 디테일이 중요한 영역이므로, 모바일 환경에 최적화된 경량 DSConvGN 블록을 적용합니다. DSConvGN은 $3 \times 3$ depthwise convolution, $1 \times 1$ pointwise convolution, GroupNorm, SiLU 활성화 함수로 구성되어 공간 문맥을 효율적으로 보존합니다.

지각 비유에서 어떤 사람이 안경을 쓰든 맨눈으로 보든 망막 신호가 시각 피질의 일정한 신경 통로로 규격화되어 들어오듯, 이 전처리는 백본의 체급이 바뀌더라도 Fast Feature Aggregator 자체의 연산량을 항상 일정하게 고정합니다. 실제로 객체 수 $K$를 10에서 100으로 10배 늘려도 이 모듈의 연산량은 6.323에서 6.333 GFLOPs로 0.16% 미만의 변화에 그칩니다.

#### 2. 최저 해상도에서의 집중적 의미 주입 (Object Guidance)

![Figure 4: Object Guidance](/images/segfs/_page_6_Picture_4.jpeg)
*Figure 4: Object Guidance 모듈 상세. Slow path에서 투영된 object embedding이 $P_5$의 각 spatial cell에 주입되어 object-aware feature map을 생성합니다.*

Object Guidance는 SegFS의 가장 독창적인 심장부입니다. 앞서 3.2절에서 준비한 $K+1$개의 object embedding을 최저 해상도 맵인 $P_5$에 집중적으로 주입합니다.

여기서 근본적인 설계 질문이 생깁니다. 왜 모든 스케일이 아니라 오직 가장 낮은 해상도의 $P_5$에서만 주입을 수행하는 것일까요?

지각 비유로 풀어보면 그 해답이 분명해집니다. 인간이 복잡한 거리에서 누군가를 찾을 때, 처음부터 시야의 미세한 모래알이나 옷감의 실밥 같은 극단적인 세부 디테일에 눈을 바짝 대고 대조하지 않습니다. 시야 전체를 널찍하고 성기게 조망하며 "저쯤에 사람이 있고 저쯤에 차가 있다"는 대략적인 위치와 정체를 한눈에 짚어내는 것과 같습니다.

$P_5$는 원본 해상도의 32분의 1 크기로 spatial cell 수가 가장 적어 연산량이 극도로 낮습니다. 동시에 네트워크의 가장 깊은 곳에서 추출되어 수용 영역이 가장 넓고, 영상 전체의 포괄적인 문맥 정보가 가장 농축되어 있습니다. 만약 고해상도인 $P_2$나 $P_3$의 수만 개 spatial cell을 상대로 토큰 매칭을 시도한다면 연산량이 폭증하여 실시간 처리가 불가능해집니다. 따라서 가장 작고 의미가 밀집된 $P_5$ 한 곳에만 의미를 주입한 뒤 상위 해상도로 전파하는 것이 연산 비용과 의미 밀도의 이상적인 교차점입니다.

Object Guidance는 Object Injection과 Gated Fusion의 두 단계로 정밀하게 맞물립니다.

첫 번째 하위 단계인 Object Injection은 공간 셀에 인스턴스의 정체를 각인하는 과정입니다. 지각 비유로는 "방금 본 파란 옷 보행자가 저 구역에 있고, 검은 차가 저 구역에 있다"며 시야의 대략적인 구역마다 대상의 이름표를 배정하는 작업입니다. $P_5$의 각 spatial cell 벡터와 $K+1$개 토큰 사이의 multi-head cosine similarity를 계산합니다. 여기에 head별로 초기화된 학습 가능한 온도 파라미터 $\tau$를 적용하고 softmax를 취합니다:

$$w_{i, k} = \frac{\exp(\tau \cdot \text{sim}(P_{5, i}, t_k))}{\sum_{j=1}^{K+1} \exp(\tau \cdot \text{sim}(P_{5, i}, t_j))}$$

이 확률 가중치 $w_{i, k}$를 바탕으로 $K+1$개 토큰의 가중합을 구하면 object-aware feature map $I_5$가 생성됩니다. 여기서 핵심은 학습 가능한 temperature $\tau$의 동적 변화입니다. 학습 초기에는 $\tau$가 커서 여러 이름표가 부드럽게 섞이지만, 학습이 진행되며 $\tau$가 낮아지면 softmax 분포가 극도로 첨예해집니다. 결과적으로 각 공간 셀은 자신과 의미적으로 가장 닮은 단 하나의 object embedding으로 100% 치환되며 뚜렷한 인스턴스 정체를 획득합니다.

두 번째 하위 단계인 Gated Fusion은 주입된 의미를 현재 프레임의 공간 디테일과 융합하는 과정입니다. 지각 비유로는 머릿속에 기억해 둔 "저것은 사람이다"라는 과거의 지식과, 지금 이 찰나 망막에 비치는 "지금 팔다리가 이렇게 움직이고 있다"는 생생한 현재의 시각 신호를 하나로 결합하는 과정입니다. $I_5$는 의미는 명확하지만 keyframe에서 건너온 정보이므로, 현재 프레임에서 객체가 움직이며 생긴 미세한 외곽선이나 텍스처 같은 시각적 생생함은 결여되어 있습니다.

이를 해결하기 위해 먼저 DSConvGN 블록으로 $I_5$의 거친 경계를 매끄럽게 평활화한 뒤, 원본 백본 맵 $P_5$와 delta-gated 메커니즘으로 결합합니다:

$$\tilde{P}_5 = P_5 + \sigma(\text{GateConv}(P_5 \parallel I_5)) \odot \text{DSConvGN}(I_5)$$

여기서 $\parallel$는 채널 방향 concatenation, $\sigma$는 sigmoid 활성화 함수, GateConv는 공간적 혼합 마스크를 예측하는 $1 \times 1$ convolution이며, $\odot$는 원소별 곱입니다. 게이트는 마치 감각의 밸브처럼 작동하여, 객체의 내부 영역에는 기억해 둔 의미를 강력하게 주입하고, 배경이나 정밀한 외곽 경계 영역에서는 지금 눈앞에 맺힌 원본 $P_5$의 고유 시각 단서를 그대로 살려냅니다.

#### 3. 점진적 해상도 복원 및 마스크 생성 (Progressive Upsampling)

객체 의미를 머금은 $\tilde{P}_5$가 완성되면, 이를 상위 해상도로 계단식 복원하는 Progressive Upsampling이 이어집니다.

지각 비유로 풀어보면 대상의 대략적인 정체와 구역을 파악한 직후, 초점을 순식간에 또렷하게 맞추며 시야 속 대상의 날카로운 외곽선과 형태를 살려내는 과정입니다.

- $P_5$에서 $P_4$로의 복원: $\tilde{P}_5$를 bilinear interpolation으로 2배 확대한 뒤, 상위 백본 맵 $P_4$와 채널 방향으로 결합하고 DSConvGN 블록으로 융합합니다.
- $P_4$에서 $P_3, P_2$로의 복원: 동일한 2배 확대 및 결합 과정을 $P_3$를 거쳐 최종 1/4 해상도인 $P_2$까지 순차적으로 반복합니다.

이 계단식 복원 과정이 중요한 이유는, 단계가 올라갈 때마다 상위 백본 맵이 보관하고 있던 미세한 공간 디테일과 윤곽선 정보가 차곡차곡 덧입혀지기 때문입니다. 그 결과 저해상도에서 출발했음에도 불구하고 마스크의 외곽선이 뭉개지지 않고 날카롭게 살아납니다.

마침내 최종 $P_2$ 해상도의 고해상도 feature map이 도출되면, 앞서 3.2절에서 준비해 둔 keyframe의 object embedding 커널을 입력하여 $1 \times 1$ convolution을 수행합니다. 각 커널이 feature map 위를 스캔하면서 자신과 일치하는 객체 영역에서 높은 activation을 출력하며, 최종적으로 $N$개의 인스턴스별 세그멘테이션 마스크가 완성됩니다.

### 3.4 학습 전략: 비디오 없이 이미지만으로 학습

SegFS의 학습 방법에는 한 가지 놀라운 반전이 숨어 있습니다. 분명 비디오에서 움직이는 객체를 추적하는 모델인데, 학습할 때는 비디오 정답 데이터를 전혀 사용하지 않는다는 점입니다. 심지어 비디오 데이터셋을 쓸 때조차 영상을 연속 재생하지 않고, 한 장씩 따로 떼어낸 정적 이미지처럼 취급합니다.

비디오 데이터는 영상 속 모든 프레임마다 움직이는 사물을 일일이 따라가며 외곽선을 그려야 하므로 사람이 라벨링하기에 비용이 너무 비싸고 데이터 양도 적습니다. 반면 멈춰 있는 사진 데이터셋은 인터넷에 사물의 종류별로 수백만 장이 넘쳐납니다. 저자들은 이 지점에서 기발한 착안을 합니다. 굳이 비싸게 비디오를 보여줄 필요 없이, 멈춰 있는 사진 한 장 안에서 느린 경로(slow path)와 빠른 경로(fast path)가 바통을 주고받는 법만 가르치면 된다는 것입니다.

학습이 진행되는 원리는 마치 노련한 선배와 신입 조수가 한 장의 사진을 함께 검토하는 협업 과정과 같습니다.

사진 한 장이 들어오면, 먼저 이미 똑똑하게 사전학습된 slow network가 나섭니다. 이 선배 모델은 사진을 깊이 있게 분석하여 "여기에 사람이 있고 저기에 자전거가 있다"는 대상의 정체와 대략적인 사각형 위치를 정확히 짚어냅니다. slow network의 가중치는 꽁꽁 얼려둔 채(frozen) 정답 힌트인 object query만 쏙 뽑아냅니다.

그다음 이 힌트를 fast network에게 건네줍니다. 신입인 fast network는 복잡한 언어 분석이나 무거운 연산은 일절 하지 않고, 선배가 건네준 힌트만 슬쩍 참고하여 사진의 원본 시각 특징 맵 위에서 "그럼 사람 모양의 정밀한 마스크를 이렇게 칠하면 되겠구나"라며 픽셀 단위 마스크를 그려냅니다.

그런데 여기서 DETR 계열 모델 학습의 가장 고질적인 문제가 발생합니다. 모델이 여러 개의 마스크를 그렸을 때, 어느 마스크가 실제 정답 사람이고 어느 마스크가 자전거인지 일대일로 연결해 주는 이분 매칭(bipartite Hungarian matching)을 거쳐야 합니다.

만약 이제 막 학습을 시작한 fast network에게 정답 매칭을 전적으로 맡겨버리면 어떻게 될까요? 아직 실력이 부족한 신입 모델은 마스크를 서투르게 그립니다. 그 불완전한 마스크를 기준으로 정답을 매칭하려 들면, 사람 마스크 자리에 자전거 정답이 연결되는 등 매 순간 매칭이 엉뚱하게 뒤바뀌어 버립니다. 정답 기준 자체가 흔들리니 모델은 무엇을 배워야 할지 갈피를 못 잡고 학습이 완전히 망가지게 됩니다.

저자들은 이 문제를 해결하기 위해 논문 4.2절에서 다음과 같은 핵심 해법을 제시합니다.

> "For the bipartite Hungarian matching with the ground-truth instance masks and categories, we construct a hybrid matching cost: we utilize the highly accurate category logits and bounding box predictions from the pre-trained slow network, while incorporating the mask predictions from the fast network. By anchoring the matching process with the slow network, we ensure stable and consistent ground-truth assignment."

해법의 핵심은 slow network를 든든한 닻(anchor)으로 삼는 하이브리드 매칭 비용입니다. "이 물체의 정체가 무엇이고 어디에 있는가"를 판단하는 일은 이미 완벽하게 학습된 slow network의 정확한 예측값을 그대로 가져다 씁니다. 정답과의 매칭을 slow network가 단단하게 고정해 주는 것입니다.

선배 모델이 정답 매칭을 흔들림 없이 꽉 잡아주니, 신입인 fast network는 물체의 이름이나 위치를 맞히는 복잡한 고민을 할 필요가 없습니다. "나는 할당된 정답 객체의 테두리만 깔끔하게 따내면 된다"며 오직 마스크 손실(Mask Loss)과 영역 일치도 손실(DICE Loss)에만 모든 신경을 집중하여 마스크 그리는 실력을 빠르게 키워나갈 수 있습니다.

---

## 4. 실험 결과

### 4.1 실험 설정

학습 데이터로는 COCO, LVIS, BDD의 이미지 instance segmentation 데이터셋과, YouTubeVIS19, YouTubeVIS21, OVIS의 비디오 데이터셋을 이미지로 취급하여 사용했습니다. RefCOCO 계열 referring segmentation 데이터셋과 UVO, SA-1B의 open-world segmentation 데이터셋도 포함했습니다. 4개 A100 GPU에서 배치 크기 128, 학습률 1e-4로 500,000 iteration을 학습했습니다. Slow model은 frozen 상태로 300개 object query를 출력하며, 텍스트 카테고리와의 유사도 기준 상위 50개를 선별하여 Object Guidance에 사용합니다.

평가는 학습 시 포함된 YouTubeVIS19과 OVIS, 그리고 zero-shot 성능 검증을 위한 BURST와 LV-VIS에서 수행했습니다. 입력은 짧은 변 480 pixel로 리사이즈하고, $T=5$ 프레임 간격으로 마스크를 전파합니다. On-device latency는 Qualcomm AI Hub를 통해 Samsung Galaxy S25 Ultra에서 측정했습니다.

### 4.2 정량적 결과: 정확도와 효율의 균형

SegFS는 MOBIUS, GLEE, TROY-VIS 등 다섯 가지 slow network와 조합하여 평가되었습니다. 결과를 이해하기 위해 먼저 비교 기준선들을 소개합니다.

Copy는 가장 단순한 하한선입니다. Keyframe에서 생성한 마스크를 후속 $T$ 프레임에 그대로 복사합니다. 연산 비용은 0이지만, 움직이는 객체를 전혀 추적하지 못합니다.

Reuse Objects는 상한선에 가까운 기준입니다. MobileInst와 TROY-VIS에서 영감을 받아, inter-frame에서 Feature Enhancer까지는 실행하되 Object Decoder만 건너뛰고 keyframe의 인스턴스 embedding을 재사용합니다. 가장 무거운 Feature Enhancer를 매 프레임 실행하므로 정확도는 높지만 속도 이점은 제한적입니다.

아래 표는 MOBIUS-Mini-M 조합의 결과를 요약합니다.

| 방법 | YTVIS19 AP | OVIS AP | BURST HOTA | LV-VIS AP | Amortized FPS |
|------|-----------|---------|------------|-----------|---------------|
| Slow Network 매 프레임 | 48.7 | 23.6 | 19.7 | 16.7 | 8.7 |
| Copy 하한선 | 20.1 | 5.1 | 11.4 | 10.4 | 51.8 |
| Reuse Objects 상한선 | 42.1 | 15.0 | 15.2 | 15.8 | 12.7 |
| MPVSS | 39.8 | 13.9 | 13.8 | 14.8 | 16.5 |
| SegFS | 41.6 | 14.1 | 14.7 | 15.2 | 38.2 |

이 표에서 주목할 점은 두 가지입니다.

첫째, SegFS와 Reuse Objects의 정확도 차이가 매우 작습니다. YTVIS19에서 -0.5 AP, OVIS에서 -0.9 AP에 불과합니다. Reuse Objects는 매 프레임 Feature Enhancer를 실행하여 backbone feature를 텍스트로 정렬한 뒤 keyframe embedding과 매칭하는 반면, SegFS는 Feature Enhancer를 완전히 건너뛰고 backbone feature에 embedding을 직접 주입합니다. 이 정확도 차이가 1 AP 이내라는 것은, backbone feature에 이미 충분한 공간 의미 정보가 인코딩되어 있어서 Feature Enhancer의 추가 정제가 거의 불필요하다는 SegFS의 핵심 가설을 실험적으로 뒷받침합니다.

둘째, FPS의 격차가 극적입니다. Reuse Objects의 12.7 FPS에서 SegFS의 38.2 FPS로 약 3배 향상되었으며, MOBIUS 기반 조합 중 유일하게 30 FPS 실시간 기준선을 돌파합니다.

TROY-VIS를 slow network로 사용할 때 SegFS는 모든 조합 중 가장 높은 절대 정확도를 기록합니다. 이 시너지의 원인은 TROY-VIS의 설계 철학에 있습니다. TROY-VIS는 backbone에 더 많은 표현력을 할당하고 Feature Enhancer를 공격적으로 경량화한 모델입니다. 즉 backbone feature 자체가 이미 풍부한 의미 정보를 담고 있으므로, backbone feature에 의존하는 SegFS의 fast path와 완벽하게 상보적입니다. 다만 이 조합은 backbone 자체가 무거운 EfficientViT-L2여서 SegFS 조합 중 가장 느린 구성이 됩니다.

Latency 측면의 비교는 더욱 인상적입니다. SegFS의 MobileNetV4-CM 변형은 10.1 GFLOPs, Samsung Galaxy S25 Ultra에서 8.3 ms의 fast path latency를 기록합니다. MOBIUS-Mini-M의 115.7 ms, GLEE-Lite의 587.9 ms와 비교하면 14~70배의 latency 절감입니다.

### 4.3 Optical Flow 기반 방법과의 비교

keyframe 마스크를 후속 프레임으로 전파하는 또 다른 접근은 optical flow를 활용한 mask warping입니다. RAFT나 LiteFlowNet2 같은 optical flow 모델이 프레임 간 움직임 벡터를 추정하고, 이 벡터에 따라 keyframe의 마스크를 기하학적으로 변형하여 다음 프레임에 맞추는 방식입니다.

이 방법들은 정확도 면에서 SegFS와 큰 격차를 보입니다. MOBIUS-Mini-M 기반에서 LiteFlowNet2는 YTVIS19 AP 28.6, RAFT는 28.9에 머무르며, SegFS의 41.6과 13 AP 이상의 차이를 보입니다.

MPVSS는 인스턴스별 optical flow를 학습하여 object-conditioned warping을 수행하는 더 정교한 방법으로, 39.8 AP로 flow 기반 중 가장 높은 정확도를 달성합니다. 그러나 mask warping의 근본적인 한계가 남아 있습니다. Warping은 본질적으로 이전 프레임에 이미 존재하는 마스크를 기하학적으로 이동시키는 연산입니다. 따라서 새롭게 장면에 진입하는 객체처럼, 이전 프레임에 마스크가 존재하지 않았던 spatial content에 대해서는 아무리 정교한 flow를 추정해도 warping할 대상 자체가 없어 추적이 실패합니다.

SegFS는 이와 근본적으로 다릅니다. SegFS는 마스크를 warping하는 것이 아니라, 현재 프레임의 backbone feature를 object embedding으로 조건화하여 마스크를 처음부터 새로 생성합니다. 따라서 새로운 객체가 등장하든, 기존 객체가 급격히 움직이든, 현재 프레임의 시각 정보를 바탕으로 인스턴스를 재탐지할 수 있습니다.

### 4.4 Ablation Study

![Figure 5: K sensitivity and T-FPS tradeoff](/images/segfs/_page_13_Figure_2.jpeg)
*Figure 5: 왼쪽 두 패널은 선별 object embedding 수 $K$에 따른 AP 변화, 오른쪽 두 패널은 propagation interval $T$에 따른 AP와 amortized FPS의 tradeoff를 보여줍니다.*

Propagation interval $T$에 따른 성능 분석에서, 두 가지 대조적인 경향이 드러납니다. Reuse Objects, MPVSS, SegFS처럼 의미적 인스턴스 embedding을 전파하는 방법들은 $T$가 증가해도 성능이 완만하게 하락합니다. 반면 Copy, RAFT, LiteFlowNet2처럼 픽셀 수준 마스크를 직접 전파하거나 warping하는 방법들은 급격한 성능 저하를 보입니다.

이 차이의 원인은 직관적으로 이해됩니다. 마스크를 직접 warping하면 프레임이 진행될수록 기하학적 오차가 누적됩니다. 한 프레임에서 1 pixel 밀린 마스크가 다음 프레임에서 또 1 pixel 밀리면, 5 프레임 후에는 경계가 크게 어긋납니다. 반면 의미적 embedding을 전파하는 방식은 매 프레임마다 현재 프레임의 visual feature를 활용하여 마스크를 새로 생성하므로, 공간 오차의 누적이 구조적으로 발생하지 않습니다.

$K$에 대한 민감도 분석에서는 $K=50$에서 AP가 최고점을 기록한 뒤 약간 낮은 값으로 수렴합니다. $K=50$을 넘어서면 추가 embedding이 저품질 query에 해당하여 노이즈로 작용합니다. 앞서 언급했듯 $K=10$에서 $K=100$으로 증가해도 FLOPs 변화는 0.16% 미만이어서, $K$ 선택이 효율성에 미치는 영향은 무시할 수 있습니다.

컴포넌트별 ablation에서는 Object Injection과 attention 기반 prototype refinement가 가장 큰 성능 향상을 가져왔습니다. Object Injection 없이는 fast path가 "이 영역이 무엇인지"에 대한 의미적 조건화를 전혀 받지 못하고, attention 기반 refinement 없이는 object query가 fast path의 feature space에 적절히 적응하지 못한 채 주입되기 때문입니다.

### 4.5 정성적 결과

![Figure 6: Qualitative Results](/images/segfs/_page_13_Figure_4.jpeg)
*Figure 6: YouTubeVIS19 시퀀스에 대한 세그멘테이션 결과. MOBIUS 매 프레임 실행, MPVSS, SegFS의 예측을 비교합니다.*

Figure 6은 YouTubeVIS19 시퀀스에서 세 방법의 세그멘테이션 결과를 시각적으로 비교합니다. MPVSS는 화면 상단 경계에서 새롭게 진입하는 객체를 정확히 추적하지 못합니다. 4.3절에서 설명한 optical flow의 근본적 한계, 즉 이전 프레임에 존재하지 않던 spatial content에 대해 마스크를 warping할 수 없다는 문제가 실제로 발생하고 있는 것입니다.

반면 SegFS는 현재 프레임의 backbone feature를 object embedding으로 동적으로 조건화하여 인스턴스를 재탐지하므로, 객체가 새롭게 등장하든 급격히 움직이든 안정적인 세그멘테이션을 유지합니다. 그 결과 무거운 slow network를 매 프레임 실행한 것과 비견되는 마스크 품질을 달성합니다.

---

## 5. 결론 및 핵심 시사점

SegFS가 제시한 핵심 통찰은 명확합니다. OV-VIS 파이프라인에서 가장 무거운 Feature Enhancer와 Object Decoder는 매 프레임 실행할 필요가 없습니다. Backbone feature map에는 이미 fine-grained localization에 충분한 공간 의미 정보가 인코딩되어 있으며, 한 번 추출한 인스턴스 embedding을 backbone feature space로 역투영하여 주입하는 것만으로도 inter-frame에서의 마스크 품질을 유지할 수 있습니다.

이 관찰에서 비롯된 dual-path 설계가 실용적으로 매력적인 이유는 두 가지입니다.

첫째, slow path의 모델 선택에 제약이 없습니다. GLEE, MOBIUS, TROY-VIS 등 어떤 object-centric 모델이든 가중치를 얼려놓은 채 plug-in으로 사용할 수 있습니다. 향후 더 정확한 모델이 등장하면 slow path만 교체하여 즉시 SegFS에 통합할 수 있다는 뜻입니다. 기존 모델들의 발전이 곧바로 SegFS의 발전으로 이어지는 구조입니다.

둘째, fast path의 연산 비용이 backbone 용량과 거의 무관합니다. Backbone feature를 고정된 256 채널로 투영하는 전처리 덕분에, Fast Feature Aggregator 자체의 비용은 backbone이 무엇이든 거의 일정하게 유지됩니다. Latency 변동은 backbone 자체의 무게에서만 발생합니다.

실험 결과는 이 접근의 실용성을 강력하게 뒷받침합니다. MOBIUS-Mini-M 기반 SegFS는 Samsung Galaxy S25 Ultra에서 8.3 ms의 fast path latency로 30 FPS 이상의 amortized 실시간 추론을 달성하며, Feature Enhancer까지 매 프레임 실행하는 Reuse Objects 대비 1 AP 이내의 정확도 차이만을 보입니다. Optical flow 기반 대안들과 비교했을 때, 픽셀 수준 mask warping이 아닌 의미적 embedding propagation이라는 설계 선택이 특히 긴 propagation interval과 새로운 객체 등장 상황에서 월등한 강건성을 제공합니다.

SegFS는 결국 multimodal semantic understanding과 dense mask prediction을 시간 축에서 분리하여, 모든 프레임에서 모든 것을 처리해야 한다는 기존의 암묵적 가정을 깨뜨린 연구입니다. 인간이 무거운 개념적 식별과 빠른 시각적 추적을 분업하여 뇌의 인지적 과부하 없이 세상을 기민하게 바라보듯, 무거운 이해는 드물게, 가벼운 형태 예측은 빈번하게 수행하는 지각 분업의 원칙이 모바일 기기에서의 실시간 OV-VIS를 현실로 가져왔습니다.
