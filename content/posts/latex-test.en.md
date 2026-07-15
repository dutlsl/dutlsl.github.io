---
title: "LaTeX Rendering Integrity Test Post"
date: 2026-07-15T20:00:00+09:00
draft: false
math: true
tags: ["Test", "LaTeX"]
categories: ["Test"]
summary: "An auto-generated test post to verify LaTeX formula rendering integrity in accordance with the new AGENTS.md guidelines."
---

This post was created to verify that mathematical formulas render perfectly without breaking in the blog (Hugo/KaTeX) environment.

## 1. Inline Math

These are inline formulas naturally blended into the text. Thanks to Hugo's `passthrough` extension, underscores (`_`) should not be mistakenly parsed as italics but should correctly render as subscripts.

- Feature representation: $z_i^m$
- Predictive probability distribution: $q_i^m$
- Alignment loss: $\mathcal{L}_{\text{Align}}$
- Target sample: $x_i^t$
- Prototype: $w_j$

## 2. Block Math

These are block formulas displayed centrally in an independent paragraph.

$$ C_{i,j}^m = \alpha \cdot d(z_i^m, w_j) + L(q_i^m, y_j) $$

$$ \Gamma^* = \arg\min_{\Gamma} \int C^v d\Gamma $$

## Conclusion

If the formulas above appear as clean mathematical symbols without backslashes (`\`), unnecessary asterisks (`*`), or HTML tags, it means the current authoring principle (using pure LaTeX syntax and prohibiting `\_` escaping) is working perfectly.
