# 全息数据与长度泛化 — 综合报告 (Holographic Data for Length Generalization)

**一句话结论**：对**标准 RoPE decoder-only Transformer**(纯 next-token 预训练),用 (β,γ)
数据塑形(包括 holo.pdf 主张的"全息/混沌边缘")**不能诱导长度泛化**;表面上的"全息优势"是
**retention 指标在低训练精度下的低分母伪信号**。更大的模型与 SSM 架构带来的提升,都大于数据塑形。

## 1. 设置
- **任务** `nested_monoid`(holo.pdf 全息构造的可学习实例化):交错的仿射/置换**寄存器机 + 命名算子召回**
  (`DEF name idx` / `USE name → result`)。β = 召回距离(Zipf recency),γ = filler 比例。
  `op_kind=perm`(固定置换池查表;`affine` 模运算被验证 grokking 困难、不可学,故弃用)。
- **模型** RoPE decoder-only Transformer(holo_small 19M / holo_100m 101M);Mamba 对照。
- **指标** answer-masked 精度;**retention = acc(L_long)/acc(L_short)**,train L=256 → eval L=2048。
  ⚠️ **cheat-guard**:retention 仅在 train_acc≥0.9 时可信(否则低分母抬高比值)。

## 2. H1 — (β,γ) 相图:**REFUTED** ❌
- **Phase A 锚点(N=3)**:retention 排序 Natural(0.250) ≈ Abyss(0.247) > Edge(0.200) > CoT(0.187)。
  **Edge-of-chaos 非最优**,与 holo.pdf 预测相反。
- **Phase B 完整 6×6 网格(cheat-guarded)**:可学习区 retention ≈ **0.18–0.24,平坦,无 ridge**。
  小β×γ=0.8 处的"高 retention(0.30–0.36)"是伪信号(那里 train_acc 0.27–0.31、绝对 L2048 acc 最低 0.08–0.13)。
- **Phase D 100M(N=3)**:同一排序(Natural>Abyss>Edge>CoT),整体 retention 更高(0.26–0.33)
  → **更大模型一致提升长度泛化,但不产生混沌边缘优势**。
- 机制:vanilla Transformer 在**局部依赖**(大β)下长度外推更稳;小β 长程召回在测试长度超出训练距离时更难。

## 3. H2 — holographic-short vs truncated-long(matched budget):**部分支持** ⚠️
- truncated(长程序尾部切片、含悬挂召回)train_acc 仅 0.18–0.70(ill-posed,学不会);其高 retention 同为伪信号。
- 按**绝对 L2048 acc**:holographic 3/4 锚点更高,且训练精度普遍 ~1.0。
- → holographic-short 更可学、长程绝对更好,但部分因 truncated 切片本身 ill-posed,结论偏弱。

## 4. H3 — Transformer vs Mamba:**初步支持 Mamba ≥ Tx**(完整版计算中)
- Natural 锚点:Mamba retention 0.307 / L2048 0.267 vs Transformer 0.250 / 0.236 —— Mamba 略优,
  与本项目此前 N=3"Mamba 长度泛化优于 Transformer"一致。完整 4 锚点(快版,eval≤1024)计算中。

## 5. 方法学贡献(cheat-guard)
retention 比值在模型未学会(低 acc(L_short))时严重误导——本研究的"全息 ridge"恰是此类伪信号。
**报告长度泛化必须同时给出绝对长程精度 + 训练精度门控**。(与本组 ILF/KS 的 frozen-field、bounded-baseline cheat-guard 同源。)

## 6. 结论与局限
- **核心结论**:holo.pdf 的全息/混沌边缘长度泛化主张,对标准 decoder-only Transformer **被证伪**;
  数据塑形 < 模型规模 < 架构(Mamba)。这是一个干净、cheat-guarded 的负面结果。
- **局限**:Phase B 网格 N=1(锚点 N=3);H3 仅 Natural 完成(快版进行中);op_kind=perm(非 affine);单任务。
- 图:`../results/holo_phaseB_heatmap.png`。数据:`../results/holo_phase{A,B,C,D}_*.md`、`../runs/holo_*`。
