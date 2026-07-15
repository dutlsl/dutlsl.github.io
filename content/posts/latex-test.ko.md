---
title: "LaTeX 렌더링 무결성 테스트 (ko)"
date: 2026-07-15T20:30:00+09:00
draft: false
math: true
tags: ["Test", "LaTeX"]
categories: ["Test"]
summary: "Hugo 환경에서 LaTeX 수식 렌더링이 정상적으로 이루어지는지 테스트합니다."
cover:
  image: ""
  alt: ""
---

## 1. 인라인 수식 (Inline Math)

다음은 본문 속에 포함되는 인라인 수식입니다. (아티팩트용 유니코드/HTML 기호에서 순수 LaTeX로 치환됨)

특징 표현은 $z_i^m$로 나타내고, 예측 확률 분포는 $q_i^m$로 나타냅니다.
손실 함수 식에서 정렬 손실은 $\mathcal{L}_{\text{Align}}$입니다.
타깃 샘플은 $x_i^t$이고 프로토타입은 $w_j$입니다.

## 2. 블록 수식 (Block Math)

결합 분포 최적 수송 비용은 다음과 같이 정의됩니다:

$$ C_{i,j}^m = \alpha \cdot d(z_i^m, w_j) + L(q_i^m, y_j) $$

최적 수송 계획 $\Gamma^*$는 다음과 같이 구합니다:

$$ \Gamma^* = \arg\min_{\Gamma} \int C_v d\Gamma $$

## 3. 결론

위의 수식들은 오직 순수 LaTeX 문법만을 사용하여 작성되었습니다. 이 방식을 사용하면 Hugo 빌드 및 KaTeX 렌더링 파이프라인에서 완벽하게 표시됨을 확인하기 위한 테스트입니다.
