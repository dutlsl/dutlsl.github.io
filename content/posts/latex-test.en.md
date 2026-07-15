---
title: "LaTeX Rendering Integrity Test (en)"
date: 2026-07-15T20:40:00+09:00
draft: false
math: true
tags: ["Test", "LaTeX"]
categories: ["Test"]
summary: "Testing if LaTeX math rendering works perfectly in the Hugo environment."
cover:
  image: ""
  alt: ""
---

## 1. Inline Math

Here are inline math equations. (Converted from Artifact Unicode/HTML to standard LaTeX)

Feature representation is denoted as $z_i^m$, and predictive probability distribution is denoted as $q_i^m$.
The alignment loss in the loss function is $\mathcal{L}_{\text{Align}}$.
The target sample is $x_i^t$ and the prototype is $w_j$.

## 2. Block Math

The joint distribution optimal transport cost is defined as:

$$ C_{i,j}^m = \alpha \cdot d(z_i^m, w_j) + L(q_i^m, y_j) $$

The optimal transport plan $\Gamma^*$ is obtained as:

$$ \Gamma^* = \arg\min_{\Gamma} \int C_v d\Gamma $$

## 3. Conclusion

The formulas above are written strictly using pure LaTeX syntax. This test confirms that this method ensures flawless display within the Hugo build and KaTeX rendering pipeline.
