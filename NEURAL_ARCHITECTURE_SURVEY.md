# 神经网络音频合成架构选型调研

## 1. 评估框架

五个评估维度，按项目优先级排列：

| # | 维度 | 权重 | 判定标准 |
|---|------|------|---------|
| 1 | **实时推理延迟** | 硬约束 | < 10ms 端到端延迟，可部署于嵌入式 NPU |
| 2 | **拓扑可调制性** | 核心差异化 | 推理过程中能否动态改变权重/连接；改变时声音连续性；是否支持短/中/长期累积 |
| 3 | **声音质量** | 硬约束 | 原始音频生成，覆盖谐波→非谐波→噪音全频谱 |
| 4 | **训练可行性** | 现实约束 | 数据需求、训练稳定性、收敛速度 |
| 5 | **多 Voice 并发** | 扩展性 | 5 个独立 instance 同时运行的计算开销 |

---

## 2. 候选架构深度分析

### 2.1 RAVE / BRAVE（实时音频变分自编码器）

**来源**：Caillon & Esling (IRCAM, 2021) / Caspe, McPherson, Sandler (QMUL, JAES 2025)

#### 架构概要

```
Audio → PQMF Filter Bank → Encoder (CNN) → Latent z (128-dim)
                                              ↓
Audio ← PQMF Inverse      ← Decoder (CNN) ←  z + conditioning
                           + Noise Generator
```

- 原始 RAVE: 17.6M 参数，2048× 压缩比，~1047ms 感受野
- **BRAVE (2025)**: 4.9M 参数，128× 压缩比，517ms 感受野，causal-only 训练

#### 维度 1: 实时推理延迟 — 7/10

| 指标 | 原始 RAVE | BRAVE |
|------|----------|-------|
| 端到端延迟 | ~653ms（causal 模式）| **< 10ms** |
| Jitter | 高 | **~3ms** |
| CPU 实时因子 | ~20× | 实时 |
| 嵌入式部署 | Raspberry Pi 4 可跑 `raspberry` 配置 | 插件形式已验证 (VST/AU) |

- RAVE 有官方 ONNX export 支持和 `raspberry` 轻量配置
- BRAVE 延迟满足硬约束，但依赖 C++ 定制推理引擎 (RTNeural)，NPU 移植需自行量化
- **扣分**：BRAVE 尚未验证 INT8 量化 + NPU 部署

#### 维度 2: 拓扑可调制性 — 3/10（原始 RAVE）/ 5/10（+ Hypernetwork）

**原生 RAVE 的调制方式**：
- 控制入口是**隐空间 z**，不是权重
- z-vector 插值、FiLM 条件调制可实现音色变换
- Carvalho et al. (2024) 验证了分层隐空间解耦（低层编码动态，高层编码音色）

**隐空间方案的问题**：
- 能量注入 → z 偏移 → 声音改变：短期反馈可行
- 长期发育：z 偏移累积 → "发育"=隐空间的位移。这不是拓扑改变，是同一空间内的位置移动
- 类比：不是在"改建房子"（拓扑注入），而是在"同一个房子里走到不同的房间"（隐空间漫游）
- z 的语义结构可能不支持四个能量方向的明确分离

**Hypernetwork 方案的可行性**：
- HyperGANStrument (Zhang & Akama, ICASSP 2024) 已验证：hypernetwork → 预测 ΔW → 调节生成器权重
- 思路：Decoder weights = W_base + ΔW(energy_history)
- 四方向各对应一个 hypernetwork 或共享 hypernetwork 的多头输出
- 短期：hypernetwork 前向 → 权重更新 → 即时声音变化
- 长期：ΔW 累积存储到 FRAM → 权重渐进偏移 → 相变
- **核心挑战**：CNN decoder 的权重空间高维且高度耦合；ΔW 必须平滑施加，否则音频输出会产生 click/pop 伪影

#### 维度 3: 声音质量 — 9/10

- 原始波形生成，覆盖全频谱（谐波 + 噪音 + 瞬态）
- 音色迁移质量优秀，BRAVE 保持了原始 RAVE 的音色复现能力
- 噪音生成器分支专门处理随机/纹理成分
- BRAVE 移除了噪音生成器以降低延迟，对噪音纹理可能有轻微影响

#### 维度 4: 训练可行性 — 7/10

- VAE 目标函数，训练稳定
- 需要目标音色数据集
- 预训练模型可用
- 两阶段训练（先训 base generator，再训 hypernetwork）增加复杂度

#### 维度 5: 多 Voice 并发 — 5/10

- BRAVE: 4.9M × 5 = 24.5M 参数，全部活跃
- 5 路独立 CNN decoder 前向推理，计算量大
- 潜在优化：共享 encoder + 独立 decoder；或共享 decoder + 独立 hypernetwork 调制
- 在嵌入式 NPU 上同时跑 5 个 BRAVE instance 可能有压力

#### 综合评估

RAVE/BRAVE 是"纯神经波形生成"路线的最强候选。隐空间控制成熟但不符合"拓扑注入"语义；加 hypernetwork 后可实现真正的权重调制，但工程复杂度高。

**关键风险**：hypernetwork 输出的 ΔW 施加到 CNN decoder 时，如何保证音频连续性——这是未被验证的研究问题。

---

### 2.2 DDSP（可微分数字信号处理）

**来源**：Engel et al. (Google Magenta, ICLR 2020)，后续大量扩展

#### 架构概要

```
Audio Features (f0, loudness) → Encoder (MLP/GRU) → Control Params
                                                       ↓
                                    ┌→ Harmonic Synth (additive/wavetable)
                                    │   - harmonic_distribution
                                    │   - amplitudes
                                    │   - f0_hz
Audio ←── + ───────────────────────┤
                                    │→ Filtered Noise Synth
                                    │   - noise_magnitudes (65-band mel)
                                        → Trainable Reverb (optional)
```

- 核心思想：神经网络不直接生成波形，而是**预测 DSP 合成器的控制参数**
- 合成器可微分 → 端到端训练
- 参数少：DDSP-tiny ~280K，ultra-lightweight vocoder 低至 15 MFLOPS

#### 维度 1: 实时推理延迟 — 9/10

| 指标 | 数值 |
|------|------|
| 算法延迟 | **8ms**（ASRU 2025, causal 模式）|
| 计算量 | 15 MFLOPS（ultra-lightweight vocoder）|
| CPU 实时因子 | 0.003（vocoder-only），0.044（overall）|
| 嵌入式部署 | 已验证可部署于可穿戴设备 |

- DDSP 的推理极其轻量——MLP encoder + DSP 合成器
- DSP 部分（谐波 + 噪音合成）是经典信号处理，计算量可忽略
- 神经网络部分（MLP encoder）极适合 NPU INT8 量化

#### 维度 2: 拓扑可调制性 — 9/10

DDSP 的混合架构为拓扑注入提供了**双层调制接口**，这是其他候选不具备的：

**第一层：参数偏置（短期反馈）**

能量注入 → 对 DDSP 控制参数施加偏置：

| 能量方向 | 调制目标 | 效果 |
|---------|---------|------|
| 张 (Tension) | harmonic_distribution 偏置 | 谐波幅度向整数倍集中，谱峰锐化 |
| 扰 (Turbulence) | harmonic_distribution 边带偏移 + noise_magnitudes 调制 | 每个谐波旁分裂出非谐波小峰 |
| 密 (Density) | harmonic_distribution 低频段增益 + noise_magnitudes 低频增强 | 低中频路径增生，声音变"实心" |
| 忆 (Memory) | 历史 harmonic_distribution 快照混合 | 过去的声音质感叠加到当前 |

参数层的优势：
- 直接对应声音的物理维度，可解释、可调试
- 改变始终平滑（DSP 参数天然适合插值）
- 短期反馈清晰明确

**第二层：Encoder 权重调制（长期发育）**

Hypernetwork 根据能量积累历史 → 预测 encoder MLP 的 ΔW → 改变"音高/响度→控制参数"的映射：

```
当前 f0, loudness ──→ [Encoder( W_base + ΔW(energy_history) )] ──→ control params
```

- 同一音高、同一力度，不同发育阶段的 Voice 产生不同的谐波结构和噪音纹理
- ΔW 累积到相变阈值 → encoder 行为发生不可逆定性改变 → 新基线
- Encoder 是轻量 MLP（~100K 参数），权重调制计算量远小于 CNN decoder

**为什么 DDSP 的拓扑注入比 RAVE 更可行**：
1. Encoder 权重空间小且低维（100K vs 4.9M），hypernetwork 输出规模可控
2. DSP decoder 是天然的"连续性保证"——无论 encoder 权重怎么变，合成器输出始终是平滑音频
3. 参数层（短期）和权重层（长期）共享同一个下游（控制参数→DSP 合成），因果链统一

#### 维度 3: 声音质量 — 7/10

- 谐波 + 过滤噪音分解覆盖了从纯音到噪音的全频谱
- 对 pitched sound（管弦乐、人声）质量优秀，MOS 4.36
- **弱项**：纯噪音纹理、极端非谐波声音、瞬态丰富的打击乐——DDSP 的 SMS 模型不是为这些设计的
- NoiseBandNet (2024) 用 filterbank 扩展了噪音建模能力
- 相比 RAVE 的直接波形生成，DDSP 在"有机感"和"意想不到的声音"上可能受限

#### 维度 4: 训练可行性 — 9/10

- DSP backbone 提供强归纳偏置 → 训练极稳定
- 数据需求远小于纯神经网络（DSP 已经编码了大量音频先验知识）
- 小数据集可训练（几百个音符即可）
- 不需要对抗训练即可获得可用质量（加对抗损失可进一步提升）
- 两阶段扩展：先训 base model，再加 hypernetwork（hypernetwork 训练极轻量）

#### 维度 5: 多 Voice 并发 — 9/10

- DDSP-tiny: 280K params × 5 = 1.4M 参数
- DSP synthesis × 5 在嵌入式 CPU 上毫无压力
- 5 个独立 encoder + 5 个独立 hypernetwork + 共享 DSP 合成器
- 5 路并发的计算量小于 1 个 BRAVE instance

#### 综合评估

DDSP 在拓扑可调制性上具有压倒性优势。双层调制接口（参数短期 + 权重长期）完美匹配项目的短/中/长期发育模型。主要风险是声音质量的"有机感"上限——可能无法产生 RAVE 那种"意想不到的"音色。

---

### 2.3 SNN（脉冲神经网络）

**来源**：多源；关键突破在 2024-2025

#### 架构概要

神经元通过离散 spike 通信，天然时间动力学：
- 膜电位累积 → 阈值 → 发放 spike → 不应期 → 重置
- STDP（脉冲时间依赖可塑性）：spike 时序决定突触权重变化
- 全局神经调节：调节阈值、时间常数、学习率

#### 维度 1: 实时推理延迟 — 10/10

| 平台 | 延迟 | 功耗 |
|------|------|------|
| Polyn NASP (2025) | **50μs** | 34μW |
| Loihi 2 (Intel) | < 1ms | mW 级 |
| SpiNNaker | 实时 | ~1W/芯片 |
| Spiking-FullSubNet (Loihi) | 实时 | 极低 |

- SNN 的延迟和功耗是压倒性优势
- 但是：需要专用神经形态硬件，**不是标准 NPU**

#### 维度 2: 拓扑可调制性 — 10/10

SNN 是**唯一原生支持"拓扑注入"的架构**。不需要 hypernetwork——脉冲动力学本身就提供了权重改变机制：

| 能量方向 | 神经调节机制 | 效果 |
|---------|------------|------|
| 张 (Tension) | 提高发放阈值 + 增强同步连接 | 更精确的 spike 时间，谐波锁定 |
| 扰 (Turbulence) | 降低发放阈值 + 引入 spike 时序抖动 | 边带分叉，纹理出现 |
| 密 (Density) | 降低 LTP 诱导阈值 + 激活静默突触 | 新路径增生 |
| 忆 (Memory) | STDP 时间窗口延长 + 膜时间常数增大 | 过去模式更易重新激活 |

- STDP 提供天然的长时程结构变化：频繁共发放的神经元对 → 突触增强 → 新默认路径
- 相变是 STDP 的自然涌现：权重累积超阈值 → 网络动力学定性改变
- 能量注入 = 神经调节（neuromodulation），是 SNN 领域已有理论框架的概念

**哲学对齐**：项目的"培育声音生命体"隐喻与 SNN 的生物启发计算几乎完美对应。Voice = 一个 SNN 微环路，发育 = STDP 驱动的突触重塑，能量 = 神经调节信号。

#### 维度 3: 声音质量 — 2/10

**这是 SNN 的致命短板**：

- SpikeVoice (2024): 首个 SNN TTS，质量接近 ANN 但仅用 10.5% 能量——但这是**语音合成**，不是乐器声音
- Spiking Vocos (2025): SNN vocoder，质量接近 ANN（UTMOS 3.74 vs 3.82）——但这是声码器（重合成），不是生成式合成
- **目前没有任何公开发表的 SNN 系统能直接生成高质量乐器音频**
- 音乐生成方面（MuSpike, 2025）仅限符号音乐（MIDI/乐谱），非音频
- NIME 2025 的 SNN + IHC 工作用于控制映射，非生成

#### 维度 4: 训练可行性 — 2/10

- 替代梯度（surrogate gradient）方法是当前 SOTA，但训练大尺度 SNN 仍极困难
- 主要范式：ANN 训练 → 知识蒸馏 → SNN 转换，非端到端 SNN 训练
- 框架和工具链不成熟（snnTorch, Lava, Nengo 等仍小众）
- 音频生成 SNN 的研究社区极小

#### 维度 5: 多 Voice 并发 — 10/10

- 5 个独立 SNN 微环路在神经形态芯片上天然并行
- 功耗极低，5 voice × 34μW ≈ 170μW 持续运行
- 但是：5 路独立 STDP 发育的硬件验证从未被演示过

#### 综合评估

SNN 是哲学上最正确、工程上最不成熟的选项。**建议作为长期 R&D 储备方向，不作为当前实现的首选。**

---

### 2.4 Hypernetwork 架构（元架构）

**来源**：HyperGANStrument (Zhang & Akama, ICASSP 2024), HyperStyle (CVPR 2021)

#### 核心思想

不是独立的音频合成架构，而是一种**权重调制元架构**，可叠加到任何基础生成器上：

```
Energy Inputs ──→ Hypernetwork (lightweight MLP) ──→ ΔW per layer
                                                       ↓
Audio Features ──→ Base Generator (weights = W_base + ΔW) ──→ Audio
```

#### 拓扑调制适用性

- 最直接实现"拓扑注入"的工程路径
- Hypernetwork 输出 = 能量注入的直接物理载体
- 四方向可共享 hypernetwork backbone + 独立输出头，或四个独立 micro-hypernetwork
- FRAM 存储累积 ΔW，实现跨 session 发育
- 短期：hypernetwork forward pass → 即时 ΔW → 即时声音反馈
- 长期：ΔW 累积 → EMA/积分 → 超过阈值 → 写入 W_base → 相变

#### 关键设计选择

| 选择 | 优势 | 劣势 |
|------|------|------|
| Hypernetwork + DDSP encoder | encoder 权重空间小，ΔW 规模可控；DSP decoder 保证连续性 | 声音受限于 DDSP 的表达力 |
| Hypernetwork + BRAVE decoder | 纯神经波形生成，声音潜力大 | decoder 权重空间高维，ΔW 平滑性难保证 |
| Hypernetwork + NEWT | 轻量、causal、waveshaper 物理可解释 | 声音质量上限不确定 |

#### 综合评估

Hypernetwork 不是选项——它是**任何候选要实现拓扑注入都需要采纳的工程手段**。与 DDSP 或 BRAVE 结合使用。

---

### 2.5 Neural Waveshaping Synthesis (NEWT)

**来源**：Hayes, Saitis, Fazekas (QMUL, ISMIR 2021)

#### 架构概要

```
Pitch/Loudness → Control Embedding → α, β affine params
                                         ↓
Oscillator Bank → [Learned Waveshapers] × 64 → + → Noise Synth → Reverb → Audio
                  (time-distributed MLP
                   with sin activation)
```

- 266K 参数，fully causal，CPU 实时
- FastNEWT: MLP → LUT 查表，O(1) 推理
- 波形域直接操作，非频谱域

#### 各维度评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 延迟 | 9/10 | Causal by design，LUT 模式下极快 |
| 拓扑调制 | 7/10 | Waveshaper 函数可被 hypernetwork 调制；affine 参数 α,β 是自然调制目标；不如 DDSP 的参数那样物理可解释 |
| 声音质量 | 7/10 | 谐波丰富（waveshaping 天然产生），噪音纹理能力待验证；演示限于小提琴/小号/长笛 |
| 训练 | 8/10 | 266K 参数极轻量；频谱损失 + 对抗损失；小数据集 |
| 多 Voice | 9/10 | 266K × 5 = 1.33M，极轻量 |

#### 综合评估

NEWT 是一个值得关注的轻量替代方案。如果 DDSP 的声音"有机感"不够，NEWT 的 waveshaping 非线性可能产生更有机的谐波结构。但研究社区更小，工具链更不成熟。

---

### 2.6 其他值得注意的方向

#### 2.6.1 GAN 基合成器（GANSynth, GANStrument）

- GANSynth (Magenta, 2020): 非实时，训练后固定
- GANStrument: mel 谱图反演，被 HyperGANStrument 使用
- 延迟和实时性不如 RAVE/BRAVE
- 作为基础生成器不如 DDSP/RAVE 成熟

#### 2.6.2 扩散模型 (Diffusion-based)

- 音质 SOTA（MusicGen, Stable Audio 等），但推理慢
- MDSGen (2025): 5M 参数扩散，36× 加速，但远未到实时
- 不适合实时演奏场景

#### 2.6.3 自适应/进化网络

- "Streamlines" (AIMC 2024): 实时拓扑生长，但是艺术项目非工程系统
- "Neural Topology Synthesizer" (Fitch et al., 2025): 推理时架构自适应，但专利级别，无实现细节
- 方向对但成熟度低

#### 2.6.4 MagentaRT (Google, 2024)

- 800M 参数的实时音乐生成，风格 morphing 通过动态 embedding
- 延迟 ~2s（风格切换），不适合 <10ms 交互
- 参考价值：embedding 级别的实时风格控制是可行的

---

## 3. 拓扑注入可行性对比

这是项目的核心差异化——四种能量输入如何在推理中动态改变网络连接：

| 架构 | 原生调制接口 | 能量→拓扑的映射路径 | 短期反馈 | 中期累积 | 长期相变 | 连续性保证 |
|------|------------|-------------------|---------|---------|---------|-----------|
| **RAVE** | 隐空间 z | z → 隐空间位移 (不是拓扑改变) | ✅ 清晰 | ⚠️ z 偏移积分 | ❌ z 偏移≠结构改变 | ✅ 隐空间平滑 |
| **RAVE+Hypernet** | Decoder 权重 | hypernetwork → ΔW_decoder | ✅ 可行 | ✅ ΔW 累积 | ✅ ΔW→新基线 | ❓ 未验证 |
| **DDSP** | 控制参数 | 参数偏置 (不是拓扑改变) | ✅ 清晰 | ⚠️ 偏置积分 | ❌ 偏置≠结构改变 | ✅ DSP 天然平滑 |
| **DDSP+Hypernet** | Encoder 权重 | hypernetwork → ΔW_encoder | ✅ 参数层反馈 | ✅ ΔW 累积 | ✅ encoder 重映射→相变 | ✅ DSP decoder 保证 |
| **SNN** | 突触权重 + 神经元参数 | STDP + 神经调节 | ✅ spike 模式变化 | ✅ STDP 累积 | ✅ 突触重塑→新动力学 | ✅ 膜电位连续 |
| **NEWT+Hypernet** | Waveshaper 函数 | hypernetwork → shaper 参数 | ✅ 类似 DDSP | ✅ ΔW 累积 | ✅ shaper 重映射 | ⚠️ 需平滑机制 |

**结论**：DDSP + Hypernetwork 和 SNN 是唯二能完整支持短/中/长期拓扑注入的方案。DDSP+Hypernetwork 是工程上更成熟的选择；SNN 是哲学上更优雅的选择。

---

## 4. 综合评分卡

| 维度 (权重) | RAVE/BRAVE | RAVE+Hypernet | DDSP | DDSP+Hypernet | SNN | NEWT+Hypernet |
|------------|------------|---------------|------|---------------|-----|---------------|
| 实时延迟 (硬约束) | 7 | 6 | 9 | 8 | **10** | 8 |
| 拓扑可调制性 (核心) | 3 | 5 | 5 | **9** | **10** | 7 |
| 声音质量 (硬约束) | **9** | **9** | 7 | 7 | 2 | 7 |
| 训练可行性 | 7 | 5 | **9** | 7 | 2 | 6 |
| 多 Voice 并发 | 5 | 5 | **9** | **9** | **10** | 8 |
| **综合** | **6.2** | **6.0** | **7.8** | **8.0** | **6.8** | **7.2** |

> 权重：延迟 20%，拓扑 30%，音质 25%，训练 10%，多 Voice 15%

---

## 5. 推荐方案

### 首选：DDSP + Hypernetwork（评分 8.0/10）

**理由**：

1. **拓扑注入的最佳工程载体**：Encoder 权重空间小（~100K params）、DSP decoder 提供天然连续性保证、参数层 + 权重层双层调制完美匹配短/中/长期发育模型

2. **实时性已超额满足**：8ms 延迟、15 MFLOPS、CPU 即可实时。5 voice × 280K = 1.4M 参数，嵌入式 NPU 无压力

3. **训练策略清晰**：
   - Phase 1: 训练 base DDSP model（稳定、数据需求低）
   - Phase 2: 训练 hypernetwork 实现能量→ΔW 映射
   - Phase 3: 设计相变检测逻辑和 FRAM 存储格式

4. **风险可控**：如果 DDSP 纯音色上限不够，DSP decoder 可替换为更复杂的合成结构（FM + 波表 + 物理建模），encoder + hypernetwork 架构不变

**主要风险**：声音"有机感"和"意外惊喜"的上限可能不如纯神经方案。需通过实验验证（而非推理毙掉）。

### 次选：BRAVE + Hypernetwork（评分 6.0/10）

**适用场景**：如果 DDSP 路线的声音质量实验不达标，BRAVE + hypernetwork 是退路。纯神经波形生成的声音潜力更大，但 hypernetwork→decoder CNN 的权重调制是未被验证的研究挑战。

### 长期 R&D 储备：SNN 混合架构

**方向**：SNN 作为"发育大脑"层 + DSP/神经 decoder 作为"发声器官"：
- SNN 层接收能量输入，通过 STDP 积累突触权重变化
- SNN 的输出（spike pattern 或膜电位分布）映射到 decoder 的控制参数
- 发育 = SNN 的 STDP 累积；声音质量 = decoder 的保证
- 等待神经形态硬件（Polyn NASP、Loihi 3）更成熟、SNN 音频生成研究更丰富后切换

---

## 6. 实施路线图建议

### Phase 1: DDSP 基础验证（2-4 周）

- 在桌面端用 PyTorch 训练一个基础 DDSP model
- 验证声音质量（录制/合成训练数据）
- 测试实时推理性能（Python → C++/ONNX 导出）
- **通过标准**：单一音色下，谐波+噪音合成质量满足乐器演奏需求

### Phase 2: 能量→参数映射（2-4 周）

- 手动设计四种能量方向到 DDSP 控制参数的映射规则
- 验证短期反馈的感知签名（是否能听到"张=谐波锁紧"、"扰=边带分叉"）
- 这一步不需要 hypernetwork——用显式规则映射即可验证交互手感
- **通过标准**：四方向独立可感知，演奏者能区分，声音连续无损

### Phase 3: Hypernetwork + 中长期发育（4-8 周）

- 实现 hypernetwork: energy_history → ΔW_encoder
- 设计 ΔW 累积和相变检测机制
- 实现 FRAM 存储格式和读写逻辑
- **通过标准**：反复注入同方向能量 → 系统默认状态可测量偏移 → 相变触发

### Phase 4: 多 Voice + 频谱竞争（4-8 周）

- 5 路独立 encoder + hypernetwork 并行
- 频谱调度器：感知 5 voice 频谱占用 → 基于竞争权重分配资源
- 串扰和耦合机制
- **通过标准**：5 voice 同时发声不混乱，长期培育后竞争行为可感知

### Phase 5: 嵌入式部署（时间待定）

- 模型 INT8 量化 + 目标 NPU 适配
- FRAM 驱动开发和集成
- 硬件原型
- **通过标准**：< 10ms 延迟，5 voice 并发，FRAM 持久化正常

### Phase 6 (可选): 纯神经路径探索

- 若 DDSP 声音上限验证不够：BRAVE + hypernetwork 实验
- 若 SNN 生态成熟：SNN 发育层 + decoder 混合架构

---

## 7. 开放研究问题

1. **Hypernetwork 的权重平滑**：ΔW 如何施加才能保证音频输出无 click/pop？低通滤波 ΔW？权重插值？逐步施加（多帧内完成过渡）？

2. **相变检测**：如何定义"累积够了"？能量历史的时间积分？ΔW 的范数？还是让相变自然涌现（ΔW 超过阈值自动写入 W_base）？

3. **四个 hypernetwork 的架构**：共享 backbone + 独立 head？四个独立 micro-hypernetwork？输出 ΔW 如何组合（加性？门控？）

4. **训练数据策略**：需要什么样的训练数据覆盖四个能量方向的声音效果？合成数据生成策略？人类演奏者录制数据？

5. **频谱调度器训练**：如何训练一个有记忆、有个性的频谱竞争调度器？需要什么样的训练信号？

---

## 参考资料

- RAVE: Caillon & Esling, "RAVE: A variational autoencoder for fast and high-quality neural audio synthesis" (2021)
- BRAVE: Caspe, Shier, Sandler, Saitis, McPherson, "Designing Neural Synthesizers for Low Latency Interaction" (JAES 2025)
- DDSP: Engel et al., "DDSP: Differentiable Digital Signal Processing" (ICLR 2020)
- Ultra-lightweight DDSP Vocoder: Agrawal et al., ICASSP 2024
- NoiseBandNet: Barahona-Ríos & Collins, IEEE/ACM TASLP 2024
- HyperGANStrument: Zhang & Akama, ICASSP 2024
- NEWT: Hayes, Saitis, Fazekas, "Neural Waveshaping Synthesis" (ISMIR 2021)
- SpikeVoice: Yao et al., "First TTS with SNNs" (2024)
- Spiking Vocos: Chen et al. (2025)
- Spiking-FullSubNet: Hao et al., Intel N-DNS Challenge Winner (2024)
- MuSpike: Liang, Tang, Zeng, "Benchmark for SNN Music Generation" (2025)
- Polyn NASP: Polyn Technology, world's first neuromorphic analog signal processing chip (2025)
