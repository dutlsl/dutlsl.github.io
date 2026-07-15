---
title: "LaTeX 렌더링 무결성 테스트 포스트"
date: 2026-07-15T20:00:00+09:00
draft: false
math: true
tags: ["Test", "LaTeX"]
categories: ["Test"]
summary: "새로운 AGENTS.md 규정에 따라 LaTeX 수식 렌더링 무결성을 검증하기 위한 자동 생성 테스트 포스트입니다."
---

이 포스트는 블로그(Hugo/KaTeX) 환경에서 수식이 깨지지 않고 완벽하게 렌더링되는지 확인하기 위해 작성되었습니다.

## 1. 인라인 수식 (Inline Math)

본문 속에 자연스럽게 섞여 들어가는 인라인 수식입니다. Hugo의 `passthrough` 확장이 작동하여 언더스코어(`_`)가 이탤릭체로 오작동하지 않고 올바른 밑첨자로 렌더링되어야 합니다.

- 특징 표현: $z_i^m$
- 예측 확률 분포: $q_i^m$
- 정렬 손실: $\mathcal{L}_{\text{Align}}$
- 타깃 샘플: $x_i^t$
- 프로토타입: $w_j$

## 2. 블록 수식 (Block Math)

독립된 단락으로 중앙에 표시되는 블록 수식입니다.

$$ C_{i,j}^m = \alpha \cdot d(z_i^m, w_j) + L(q_i^m, y_j) $$

$$ \Gamma^* = \arg\min_{\Gamma} \int C^v d\Gamma $$

## 결론

만약 위 수식들이 백슬래시(`\`)나 불필요한 별표(`*`), HTML 태그 없이 깔끔한 수학 기호로 보인다면, 현재 작성 원칙(순수 LaTeX 문법 사용 및 `\_` 이스케이프 금지)이 완벽하게 작동하고 있는 것입니다.
