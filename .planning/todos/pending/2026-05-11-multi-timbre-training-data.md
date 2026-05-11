---
created: 2026-05-11T02:51:04.911Z
title: 扩展训练数据音色多样性（A+B 混合路线）
area: general
files:
  - scripts/generate_data.py
  - synth/nn/decoder.py
  - DESIGN.md:9.6
---

## Problem

当前 decoder 仅在 135 个单一长笛合成片段上训练，权重空间被困在"长笛流形"上。Phase 2 的能量偏置只能在 decoder 输出上做后处理，无法产生根性不同的音色。Phase 3 hypernetwork 需要在足够宽的权重空间中导航才能有效——若 decoder 只见过一种音色，hypernetwork 走到哪都是长笛变形。

详见 DESIGN.md 9.6 诊断章节。

## Solution

A+B 混合路线：
- **A（合成模型）**：扩展 `scripts/generate_data.py`，增加弦乐（弓噪 + 密高次谐波）、铜管（强奇次谐波 + 力度敏感的亮度）、人声共振峰（固定频谱峰包络）、纯电子波形（锯齿/方波/三角）等大类
- **B（公开数据集）**：引入 NSynth 等公开数据集，补充真实乐器的单音孤立音符作为训练补充

目标：decoder 权重空间覆盖足够宽的音色类别，为 Phase 3 hypernetwork 导航提供足够疆域。
